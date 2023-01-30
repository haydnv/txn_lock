//! A futures-aware read-write lock on a [`BTreeSet`] which supports transactional versioning.
//!
//! Example usage:
//! ```
//! use std::sync::Arc;
//! use futures::executor::block_on;
//!
//! use txn_lock::set::*;
//! use txn_lock::Error;
//!
//! let set = TxnSetLock::<u64, String>::new(0);
//!
//! let one = Arc::new("one".to_string());
//! assert!(!block_on(set.contains(1, &one)).expect("contains"));
//! block_on(set.insert(1, one.clone())).expect("insert");
//! assert!(set.try_contains(1, &one).expect("contains"));
//! assert_eq!(set.try_insert(2, one.clone()).unwrap_err(), Error::WouldBlock);
//! // set.commit(1);
//! // assert!(set.try_contains_key(2, &one).expect("contains"));
//! // set.finalize(2);
//! // assert!(set.try_contains_key(3, &one).expect("contains"));
//! ```

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::Poll;
use std::{fmt, iter};

use super::semaphore::Semaphore;
use super::{Error, Result};

pub use super::range::Range;

struct State<I, T> {
    canon: BTreeSet<Arc<T>>,
    committed: BTreeMap<I, Option<BTreeSet<Arc<T>>>>,
    pending: BTreeMap<I, BTreeSet<Arc<T>>>,
    finalized: Option<I>,
}

impl<I: Ord, T: Ord> State<I, T> {
    fn new(txn_id: I, version: BTreeSet<Arc<T>>) -> Self {
        State {
            canon: BTreeSet::new(),
            committed: BTreeMap::new(),
            pending: BTreeMap::from_iter(iter::once((txn_id, version))),
            finalized: None,
        }
    }

    #[inline]
    fn check_pending(&self, txn_id: &I) -> Result<()> {
        if self.finalized.as_ref() >= Some(txn_id) {
            Err(Error::Outdated)
        } else if self.committed.contains_key(txn_id) {
            Err(Error::Committed)
        } else {
            Ok(())
        }
    }

    #[inline]
    fn contains_canon(&self, txn_id: &I, key: &T) -> bool {
        let committed = self
            .committed
            .iter()
            .rev()
            .skip_while(|(id, _)| *id > txn_id)
            .map(|(_, version)| version);

        for version in committed {
            if let Some(version) = version {
                return version.contains(key);
            }
        }

        self.canon.contains(key)
    }

    #[inline]
    fn contains_committed(&self, txn_id: &I, key: &T) -> Poll<Result<bool>> {
        if self.finalized.as_ref() > Some(&txn_id) {
            Poll::Ready(Err(Error::Outdated))
        } else if self.committed.contains_key(&txn_id) {
            assert!(!self.pending.contains_key(&txn_id));
            Poll::Ready(Ok(self.contains_canon(&txn_id, &key)))
        } else {
            Poll::Pending
        }
    }

    #[inline]
    fn contains_pending(&self, txn_id: &I, key: &T) -> bool {
        if let Some(version) = self.pending.get(&txn_id) {
            return version.contains(key);
        }

        self.contains_canon(txn_id, key)
    }

    #[inline]
    fn insert(&mut self, txn_id: I, key: Arc<T>) {
        match self.pending.entry(txn_id) {
            Entry::Occupied(mut entry) => {
                entry.get_mut().insert(key);
            }
            Entry::Vacant(entry) => {
                entry.insert(BTreeSet::from_iter(iter::once(key)));
            }
        }
    }
}

/// A futures-aware read-write lock on a [`BTreeSet`] which supports transactional versioning.
pub struct TxnSetLock<I, T> {
    state: Arc<Mutex<State<I, T>>>,
    semaphore: Semaphore<I, Range<T>>,
}

impl<I, T> Clone for TxnSetLock<I, T> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            semaphore: self.semaphore.clone(),
        }
    }
}

impl<I, T> TxnSetLock<I, T> {
    #[inline]
    fn state(&self) -> MutexGuard<State<I, T>> {
        self.state.lock().expect("lock state")
    }
}

impl<I: Copy + Ord, T: Ord + fmt::Debug> TxnSetLock<I, T> {
    /// Construct a new, empty [`TxnSetLock`].
    pub fn new(txn_id: I) -> Self {
        Self {
            state: Arc::new(Mutex::new(State::new(txn_id, BTreeSet::new()))),
            semaphore: Semaphore::new(),
        }
    }

    /// Construct a new [`TxnSetLock`] with the given `contents`.
    pub fn with_contents<C: IntoIterator<Item = T>>(txn_id: I, contents: C) -> Self {
        let version = contents.into_iter().map(Arc::new).collect();

        Self {
            state: Arc::new(Mutex::new(State::new(txn_id, version))),
            semaphore: Semaphore::with_reservation(txn_id, Range::All),
        }
    }

    /// Check whether the given `key` is present in this [`TxnSetLock`] at `txn_id`.
    pub async fn contains(&self, txn_id: I, key: &Arc<T>) -> Result<bool> {
        // before acquiring a permit, check if this version has already been committed
        if let Poll::Ready(result) = self.state().contains_committed(&txn_id, key) {
            return result;
        }

        let _permit = self.semaphore.read(txn_id, Range::One(key.clone())).await?;
        Ok(self.state().contains_pending(&txn_id, key))
    }

    /// Synchronously check whether the given `key` is present in this [`TxnSetLock`], if possible.
    pub fn try_contains(&self, txn_id: I, key: &Arc<T>) -> Result<bool> {
        // before acquiring a permit, check if this version has already been committed
        if let Poll::Ready(result) = self.state().contains_committed(&txn_id, key) {
            return result;
        }

        let _permit = self.semaphore.try_read(txn_id, Range::One(key.clone()))?;
        Ok(self.state().contains_pending(&txn_id, key))
    }

    /// Insert a new entry into this [`TxnMapLock`] at `txn_id`.
    pub async fn insert(&self, txn_id: I, key: Arc<T>) -> Result<()> {
        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let range = Range::One(key.clone());
        let _permit = self.semaphore.write(txn_id, range).await?;
        Ok(self.state().insert(txn_id, key))
    }

    /// Insert a new entry into this [`TxnMapLock`] at `txn_id` synchronously, if possible.
    pub fn try_insert(&self, txn_id: I, key: Arc<T>) -> Result<()> {
        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let _permit = self.semaphore.try_write(txn_id, Range::One(key.clone()))?;
        Ok(self.state().insert(txn_id, key))
    }
}
