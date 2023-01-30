//! A futures-aware read-write lock on a [`BTreeMap`] which supports transactional versioning.
//!
//! Example usage:
//! ```
//! use std::sync::Arc;
//! use futures::executor::block_on;
//!
//! use txn_lock::map::*;
//! use txn_lock::Error;
//!
//! let map = TxnMapLock::<u64, String, f32>::new(1);
//!
//! let one = Arc::new("one".to_string());
//! block_on(map.insert(1, one.clone(), 1.0)).expect("insert");
//!
//! let value = block_on(map.get(1, &one)).expect("read").expect("value");
//! assert_eq!(value, 1.0);
//!
//! assert_eq!(map.try_insert(1, one.clone(), 2.0).unwrap_err(), Error::WouldBlock);
//!
//! std::mem::drop(value);  // commit will panic if any updated value is still locked
//! map.commit(1);
//!
//! let value = map.try_get(2, &one).expect("read");
//! assert_eq!(*(value.expect("guard")), 1.0);
//!
//! map.finalize(2);
//!
//! let value = map.try_get(3, &one).expect("read");
//! assert_eq!(*(value.expect("guard")), 1.0);
//! ```

use std::cmp::Ordering;
use std::collections::btree_map::{BTreeMap, Entry};
use std::ops::Deref;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::Poll;
use std::{fmt, iter};

use super::semaphore::*;
use super::{Error, Result};

pub use super::range::Range;

/// A read guard on a value in a [`TxnMapLock`]
#[derive(Debug)]
pub enum ValueReadGuard<K, V> {
    Committed(Arc<V>),
    Pending(Permit<Range<K>>, Arc<V>),
}

impl<K, V> Deref for ValueReadGuard<K, V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Committed(value) => value.deref(),
            Self::Pending(_permit, value) => value.deref(),
        }
    }
}

impl<K, V: PartialOrd> PartialOrd<V> for ValueReadGuard<K, V> {
    fn partial_cmp(&self, other: &V) -> Option<Ordering> {
        self.deref().partial_cmp(other)
    }
}

impl<K, V: PartialEq> PartialEq<V> for ValueReadGuard<K, V> {
    fn eq(&self, other: &V) -> bool {
        self.deref().eq(other)
    }
}

type Version<K, V> = BTreeMap<Arc<K>, Arc<V>>;

struct State<I, K, V> {
    canon: BTreeMap<Arc<K>, Arc<V>>,
    committed: BTreeMap<I, Option<BTreeMap<Arc<K>, Arc<V>>>>,
    pending: BTreeMap<I, Version<K, V>>,
    finalized: Option<I>,
}

impl<I: Ord, K: Ord, V> State<I, K, V> {
    #[inline]
    fn new(txn_id: I, version: Version<K, V>) -> Self {
        Self {
            canon: BTreeMap::new(),
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
    fn get_canon(&self, txn_id: &I, key: &K) -> Option<Arc<V>> {
        let committed = self
            .committed
            .iter()
            .rev()
            .skip_while(|(id, _)| *id > txn_id)
            .map(|(_, version)| version);

        for version in committed {
            if let Some(version) = version {
                if let Some(value) = version.get(key) {
                    return Some(value.clone());
                }
            }
        }

        self.canon.get(key).cloned()
    }

    #[inline]
    fn get_committed(&self, txn_id: &I, key: &K) -> Poll<Result<Option<ValueReadGuard<K, V>>>> {
        if self.finalized.as_ref() > Some(&txn_id) {
            Poll::Ready(Err(Error::Outdated))
        } else if self.committed.contains_key(&txn_id) {
            assert!(!self.pending.contains_key(&txn_id));
            let canon = self.get_canon(&txn_id, &key);
            Poll::Ready(Ok(canon.map(ValueReadGuard::Committed)))
        } else {
            Poll::Pending
        }
    }

    #[inline]
    fn get_pending(
        &self,
        txn_id: &I,
        key: &K,
        permit: Permit<Range<K>>,
    ) -> Option<ValueReadGuard<K, V>> {
        if let Some(version) = self.pending.get(&txn_id) {
            if let Some(value) = version.get(key) {
                return Some(ValueReadGuard::Pending(permit, value.clone()));
            }
        }

        let canon = self.get_canon(&txn_id, key);
        canon.map(ValueReadGuard::Committed)
    }

    #[inline]
    fn insert(&mut self, txn_id: I, key: Arc<K>, value: V) {
        #[inline]
        fn version_entry<K: Ord, V>(version: &mut Version<K, V>, key: Arc<K>, value: Arc<V>) {
            match version.entry(key) {
                Entry::Occupied(mut entry) => {
                    let existing_delta = entry.get_mut();
                    *existing_delta = value;
                }
                Entry::Vacant(entry) => {
                    entry.insert(value);
                }
            }
        }

        let value = Arc::new(value);
        match self.pending.entry(txn_id) {
            Entry::Occupied(mut entry) => version_entry(entry.get_mut(), key, value),
            Entry::Vacant(entry) => {
                let version = entry.insert(BTreeMap::new());
                version_entry(version, key, value)
            }
        }
    }
}

/// A futures-aware read-write lock on a [`BTreeMap`] which supports transactional versioning
// TODO: handle the case where a write permit is acquired and then dropped without committing
pub struct TxnMapLock<I, K, V> {
    state: Arc<Mutex<State<I, K, V>>>,
    semaphore: Semaphore<I, Range<K>>,
}

impl<I, K, V> Clone for TxnMapLock<I, K, V> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            semaphore: self.semaphore.clone(),
        }
    }
}

impl<I, K, V> TxnMapLock<I, K, V> {
    #[inline]
    fn state(&self) -> MutexGuard<State<I, K, V>> {
        self.state.lock().expect("lock state")
    }
}

impl<I: Ord + Copy + fmt::Display, K: Ord + fmt::Debug, V: fmt::Debug> TxnMapLock<I, K, V> {
    /// Construct a new [`TxnMapLock`].
    pub fn new(txn_id: I) -> Self {
        Self {
            state: Arc::new(Mutex::new(State::new(txn_id, Version::new()))),
            semaphore: Semaphore::new(),
        }
    }

    /// Construct a new [`TxnMapLock`] with the given `contents`.
    pub fn with_contents<C: IntoIterator<Item = (K, V)>>(txn_id: I, contents: C) -> Self {
        let version = contents
            .into_iter()
            .map(|(key, value)| (Arc::new(key), Arc::new(value)))
            .collect();

        Self {
            state: Arc::new(Mutex::new(State::new(txn_id, version))),
            semaphore: Semaphore::with_reservation(txn_id, Range::All),
        }
    }

    /// Commit the state of this [`TxnMapLock`] at `txn_id`.
    pub fn commit(&self, txn_id: I) {
        let mut state = self.state();

        self.semaphore.finalize(&txn_id, false);

        let version = state.pending.remove(&txn_id);

        match state.committed.entry(txn_id) {
            Entry::Occupied(_) => {
                assert!(version.is_none());
                #[cfg(feature = "logging")]
                log::warn!("duplicate commit at {}", txn_id);
            }
            Entry::Vacant(entry) => {
                entry.insert(version);
            }
        }
    }

    /// Finalize the state of this [`TxnMapLock`] at `txn_id`.
    /// This will merge in deltas and prevent further reads of versions earlier than `txn_id`.
    pub fn finalize(&self, txn_id: I) {
        let mut state = self.state();

        while let Some(version_id) = state.committed.keys().next().copied() {
            if version_id <= txn_id {
                if let Some(version) = state.committed.remove(&version_id).expect("version") {
                    for (key, value) in version {
                        state.canon.insert(key, value);
                    }
                }
            } else {
                break;
            }
        }

        self.semaphore.finalize(&txn_id, true);
        state.finalized = Some(txn_id);
    }

    /// Read a value from this [`TxnMapLock`] at `txn_id`.
    pub async fn get(&self, txn_id: I, key: &Arc<K>) -> Result<Option<ValueReadGuard<K, V>>> {
        // before acquiring a permit, check if this version has already been committed
        if let Poll::Ready(result) = self.state().get_committed(&txn_id, key) {
            return result;
        }

        let permit = self.semaphore.read(txn_id, Range::One(key.clone())).await?;
        let state = self.state();
        Ok(state.get_pending(&txn_id, key, permit))
    }

    /// Read a value from this [`TxnMapLock`] at `txn_id` synchronously, if possible.
    pub fn try_get(&self, txn_id: I, key: &Arc<K>) -> Result<Option<ValueReadGuard<K, V>>> {
        // before acquiring a permit, check if this version has already been committed
        if let Poll::Ready(result) = self.state().get_committed(&txn_id, key) {
            return result;
        }

        let permit = self.semaphore.try_read(txn_id, Range::One(key.clone()))?;
        let state = self.state();
        Ok(state.get_pending(&txn_id, key, permit))
    }

    /// Insert a new entry into this [`TxnMapLock`] at `txn_id`.
    pub async fn insert(&self, txn_id: I, key: Arc<K>, value: V) -> Result<()> {
        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let range = Range::One(key.clone());
        let _permit = self.semaphore.write(txn_id, range).await?;

        let mut state = self.state();
        Ok(state.insert(txn_id, key, value))
    }

    /// Insert a new entry into this [`TxnMapLock`] at `txn_id` synchronously, if possible.
    pub fn try_insert(&self, txn_id: I, key: Arc<K>, value: V) -> Result<()> {
        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let _permit = self.semaphore.try_write(txn_id, Range::One(key.clone()))?;

        let mut state = self.state();
        Ok(state.insert(txn_id, key, value))
    }
}
