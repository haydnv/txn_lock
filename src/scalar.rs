//! A transactional lock on a scalar value.
//!
//! Example:
//! ```
//! use futures::executor::block_on;
//! use txn_lock::scalar::*;
//! use txn_lock::Error;
//!
//! let lock = TxnLock::new(0, "zero");
//!
//! assert_eq!(block_on(lock.read(0)).expect("read"), "zero");
//! assert_eq!(*lock.try_read(0).expect("read"), "zero");
//! // assert_eq!(lock.try_write(1).unwrap_err(), Error::WouldBlock);
//!
//! {
//!     // let commit = block_on(lock.commit(0)).expect("commit guard");
//!     // assert_eq!(*commit, "zero");
//!     // this commit guard will block future commits until dropped
//! }
//!
//! {
//!     // let mut guard = lock.try_write(1).expect("write lock");
//!     // *guard = "one";
//! }
//!
//! // assert_eq!(*lock.try_read(0).expect("read past version"), "zero");
//! // assert_eq!(*lock.try_read(1).expect("read current version"), "one");
//!
//! // block_on(lock.commit(1));
//!
//! // assert_eq!(*lock.try_read_exclusive(2).expect("new value"), "one");
//!
//! // lock.rollback(&2);
//!
//! {
//!     // let mut guard = lock.try_write(3).expect("write lock");
//!     // *guard = "three";
//! }
//!
//! // assert_eq!(*block_on(lock.finalize(&1)).expect("finalized version"), "one");
//!
//! // assert_eq!(lock.try_read(0).unwrap_err(), Error::Outdated);
//! // assert_eq!(*lock.try_read(3).expect("current value"), "three");
//! ```

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::iter;
use std::ops::Deref;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::Poll;

use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use super::semaphore::{Overlap, Overlaps, Permit, Semaphore};
use super::{Error, Result};

/// A range used to reserve a [`Permit`] to guard access to a [`TxnLock`]
#[derive(Debug)]
pub struct Range;

impl Overlaps<Range> for Range {
    fn overlaps(&self, _other: &Range) -> Overlap {
        Overlap::Equal
    }
}

/// A read guard on a transactional value
pub enum TxnLockReadGuard<T> {
    Committed(Arc<T>),
    Pending(Permit<Range>, OwnedRwLockReadGuard<T>),
}

impl<T> Deref for TxnLockReadGuard<T> {
    type Target = T;

    fn deref(&self) -> &T {
        match self {
            Self::Committed(value) => value.deref(),
            Self::Pending(_permit, value) => value.deref(),
        }
    }
}

impl<T: PartialEq> PartialEq<T> for TxnLockReadGuard<T> {
    fn eq(&self, other: &T) -> bool {
        self.deref().eq(other)
    }
}

impl<T: PartialOrd> PartialOrd<T> for TxnLockReadGuard<T> {
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        self.deref().partial_cmp(other)
    }
}

impl<T: fmt::Debug> fmt::Debug for TxnLockReadGuard<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.deref().fmt(f)
    }
}

impl<T: fmt::Display> fmt::Display for TxnLockReadGuard<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.deref().fmt(f)
    }
}

struct State<I, T> {
    canon: Option<Arc<T>>,
    committed: BTreeMap<I, Option<Arc<T>>>,
    pending: BTreeMap<I, Arc<RwLock<T>>>,
    finalized: Option<I>,
}

impl<I: Ord, T> State<I, T> {
    fn new(txn_id: I, value: T) -> Self {
        let version = Arc::new(RwLock::new(value));

        State {
            canon: None,
            committed: BTreeMap::new(),
            pending: BTreeMap::from_iter(iter::once((txn_id, version))),
            finalized: None,
        }
    }

    #[inline]
    fn read_canon(&self, txn_id: &I) -> Result<Arc<T>> {
        let committed = self
            .committed
            .iter()
            .rev()
            .skip_while(|(id, _)| *id > txn_id)
            .map(|(_, version)| version);

        for version in committed {
            if let Some(version) = version {
                return Ok(version.clone());
            }
        }

        self.canon.clone().ok_or_else(|| Error::Outdated)
    }

    #[inline]
    fn read_committed(&self, txn_id: &I) -> Poll<Result<Arc<T>>> {
        if self.finalized.as_ref() > Some(&txn_id) {
            Poll::Ready(Err(Error::Outdated))
        } else if self.committed.contains_key(&txn_id) {
            assert!(!self.pending.contains_key(&txn_id));
            Poll::Ready(self.read_canon(&txn_id))
        } else {
            Poll::Pending
        }
    }

    #[inline]
    fn read_pending(&self, txn_id: &I, permit: Permit<Range>) -> Result<TxnLockReadGuard<T>> {
        if let Some(version) = self.pending.get(&txn_id) {
            // the permit means it's safe to call try_read_owned().expect()
            let guard = version.clone().try_read_owned().expect("version");
            return Ok(TxnLockReadGuard::Pending(permit, guard));
        }

        self.read_canon(txn_id).map(TxnLockReadGuard::Committed)
    }
}

/// A futures-aware read-write lock on a scalar value which supports transactional versioning.
pub struct TxnLock<I, T> {
    state: Arc<Mutex<State<I, T>>>,
    semaphore: Semaphore<I, Range>,
}

impl<I, T> Clone for TxnLock<I, T> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            semaphore: self.semaphore.clone(),
        }
    }
}

impl<I, T> TxnLock<I, T> {
    #[inline]
    fn state(&self) -> MutexGuard<State<I, T>> {
        self.state.lock().expect("lock state")
    }
}

impl<I: Copy + Ord, T: fmt::Debug> TxnLock<I, T> {
    /// Construct a new [`TxnLock`].
    pub fn new(txn_id: I, initial_value: T) -> Self {
        Self {
            state: Arc::new(Mutex::new(State::new(txn_id, initial_value))),
            semaphore: Semaphore::new(),
        }
    }

    /// Acquire a read lock for this value at `txn_id`.
    pub async fn read(&self, txn_id: I) -> Result<TxnLockReadGuard<T>> {
        // before acquiring a permit, check if this version has already been committed
        if let Poll::Ready(result) = self.state().read_committed(&txn_id) {
            return result.map(TxnLockReadGuard::Committed);
        }

        let permit = self.semaphore.read(txn_id, Range).await?;
        self.state().read_pending(&txn_id, permit)
    }

    /// Acquire a read lock for this value at `txn_id` synchronously, if possible.
    pub fn try_read(&self, txn_id: I) -> Result<TxnLockReadGuard<T>> {
        // before acquiring a permit, check if this version has already been committed
        if let Poll::Ready(result) = self.state().read_committed(&txn_id) {
            return result.map(TxnLockReadGuard::Committed);
        }

        let permit = self.semaphore.try_read(txn_id, Range)?;
        self.state().read_pending(&txn_id, permit)
    }
}
