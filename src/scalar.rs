//! A transactional lock on a scalar value.
//!
//! Example:
//! ```
//! use futures::executor::block_on;
//! use txn_lock::scalar::*;
//! use txn_lock::Error;
//!
//! let lock = TxnLock::new("zero");
//!
//! let value = block_on(lock.read(1)).expect("read");
//! assert_eq!(*value, "zero");
//! assert_eq!(lock.try_write(1).unwrap_err(), Error::WouldBlock);
//! assert_eq!(*lock.try_read(1).expect("read"), "zero");
//!
//! lock.commit(1);
//!
//! {
//!     let mut guard = lock.try_write(2).expect("write lock");
//!     *guard = "two";
//! }
//!
//! assert_eq!(*lock.try_read(0).expect("read past version"), "zero");
//! assert_eq!(*lock.try_read(2).expect("read current version"), "two");
//!
//! lock.commit(2);
//!
//! assert_eq!(*lock.try_read(3).expect("new value"), "two");
//!
//! lock.rollback(&3);
//!
//! {
//!     let mut guard = lock.try_write(3).expect("write lock");
//!     *guard = "three";
//! }
//!
//! lock.finalize(1);
//!
//! assert_eq!(lock.try_read(0).unwrap_err(), Error::Outdated);
//! assert_eq!(*lock.try_read(3).expect("current value"), "three");
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, RwLock as RwLockInner};
use std::task::Poll;

use tokio::sync::RwLock;

use super::guard::{TxnReadGuard, TxnWriteGuard};
use super::semaphore::{Overlap, Overlaps, PermitRead, PermitWrite, Semaphore};
use super::{Error, Result};

/// A read guard on a [`TxnLock`]
pub type TxnLockReadGuard<T> = TxnReadGuard<Range, T>;

/// A write guard on a [`TxnLock`]
pub type TxnLockWriteGuard<T> = TxnWriteGuard<Range, T>;

/// A range used to reserve a permit to guard access to a [`TxnLock`]
#[derive(Debug)]
pub struct Range;

impl Overlaps<Range> for Range {
    fn overlaps(&self, _other: &Range) -> Overlap {
        Overlap::Equal
    }
}

struct State<I, T> {
    canon: Arc<T>,
    committed: BTreeMap<I, Option<Arc<T>>>,
    pending: BTreeMap<I, Arc<RwLock<T>>>,
    finalized: Option<I>,
}

impl<I: Ord, T> State<I, T> {
    fn new(canon: T) -> Self {
        State {
            canon: Arc::new(canon),
            committed: BTreeMap::new(),
            pending: BTreeMap::new(),
            finalized: None,
        }
    }

    #[inline]
    fn check_pending(&self, txn_id: &I) -> Result<()> {
        if self.finalized.as_ref() > Some(txn_id) {
            Err(Error::Outdated)
        } else if self.committed.contains_key(txn_id) {
            Err(Error::Committed)
        } else {
            Ok(())
        }
    }

    #[inline]
    fn read_canon(&self, txn_id: &I) -> &Arc<T> {
        let committed = self
            .committed
            .iter()
            .rev()
            .skip_while(|(id, _)| *id > txn_id)
            .map(|(_, version)| version);

        for version in committed {
            if let Some(version) = version {
                return version;
            }
        }

        &self.canon
    }

    #[inline]
    fn read_committed(&self, txn_id: &I) -> Poll<Result<Arc<T>>> {
        if self.finalized.as_ref() > Some(&txn_id) {
            Poll::Ready(Err(Error::Outdated))
        } else if self.committed.contains_key(&txn_id) {
            assert!(!self.pending.contains_key(&txn_id));
            Poll::Ready(Ok(self.read_canon(&txn_id).clone()))
        } else {
            Poll::Pending
        }
    }

    #[inline]
    fn read_pending(&mut self, txn_id: &I, permit: PermitRead<Range>) -> TxnLockReadGuard<T> {
        if let Some(version) = self.pending.get(txn_id) {
            // the permit means it's safe to call try_read_owned().expect()
            let value = version.clone().try_read_owned().expect("version");
            TxnLockReadGuard::pending_write(permit, value)
        } else {
            let value = self.read_canon(txn_id).clone();
            TxnLockReadGuard::pending_read(permit, value)
        }
    }
}

impl<I: Ord, T: Clone> State<I, T> {
    #[inline]
    fn get_or_create_version(&mut self, txn_id: I) -> Arc<RwLock<T>> {
        if let Some(version) = self.pending.get(&txn_id) {
            version.clone()
        } else {
            let canon = self.read_canon(&txn_id);
            let value = T::clone(&*canon);
            let version = Arc::new(RwLock::new(value));
            self.pending.insert(txn_id, version.clone());
            version
        }
    }

    #[inline]
    fn write(&mut self, txn_id: I, permit: PermitWrite<Range>) -> Result<TxnLockWriteGuard<T>> {
        // the permit means it's safe to call try_write_owned().expect()
        let value = self
            .get_or_create_version(txn_id)
            .try_write_owned()
            .expect("version");

        Ok(TxnLockWriteGuard::new(permit, value))
    }
}

/// A futures-aware read-write lock on a scalar value which supports transactional versioning.
///
/// For non-scalar data types (e.g. a list, set, map, etc) consider using an alternate lock type,
/// or implementing a custom lock type using [`Semaphore`].
///
/// The type `T` to lock must implement [`Clone`] in order to support versioning.
/// [`Clone::clone`] is called once when [`TxnLock::write`] is called with a valid new `txn_id`.
pub struct TxnLock<I, T> {
    state: Arc<RwLockInner<State<I, T>>>,
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
    fn state(&self) -> impl Deref<Target = State<I, T>> + '_ {
        self.state.read().expect("read lock state")
    }

    #[inline]
    fn state_mut(&self) -> impl DerefMut<Target = State<I, T>> + '_ {
        self.state.write().expect("write lock state")
    }
}

impl<I: Copy + Ord + fmt::Display, T: fmt::Debug> TxnLock<I, T> {
    /// Construct a new [`TxnLock`].
    pub fn new(canon: T) -> Self {
        Self {
            state: Arc::new(RwLockInner::new(State::new(canon))),
            semaphore: Semaphore::new(),
        }
    }

    /// Commit the state of this [`TxnLock`] at `txn_id`.
    pub fn commit(&self, txn_id: I) {
        let mut state = self.state_mut();

        if state.finalized.as_ref() >= Some(&txn_id) {
            #[cfg(feature = "logging")]
            log::warn!("committed already-finalized version {}", txn_id);
            return;
        }

        self.semaphore.finalize(&txn_id, false);

        let new_value = state.pending.remove(&txn_id).map(|version| {
            if let Ok(lock) = Arc::try_unwrap(version) {
                Arc::new(lock.into_inner())
            } else {
                panic!("value to commit at {} is still locked!", txn_id);
            }
        });

        if state.pending.keys().next() == Some(&txn_id) {
            assert!(!state.committed.contains_key(&txn_id));
            state.canon = new_value.expect("committed version");
            state.finalized = Some(txn_id);
        } else if let Some(value) = new_value {
            assert!(state.committed.insert(txn_id, Some(value)).is_none());
        } else if let Some(prior_commit) = state.committed.insert(txn_id, None) {
            assert!(prior_commit.is_none());
            #[cfg(feature = "logging")]
            log::warn!("duplicate commit at {}", txn_id);
        }
    }

    /// Finalize the state of this [`TxnLock`] at `txn_id`.
    /// This will merge in deltas and prevent further reads of versions earlier than `txn_id`.
    pub fn finalize(&self, txn_id: I) {
        let mut state = self.state_mut();

        while let Some(version_id) = state.committed.keys().next().copied() {
            if version_id <= txn_id {
                if let Some(version) = state.committed.remove(&version_id).expect("version") {
                    state.canon = version;
                }
            } else {
                break;
            }
        }

        self.semaphore.finalize(&txn_id, true);
        state.finalized = Some(txn_id);
    }

    /// Acquire a read lock for this value at `txn_id`.
    pub async fn read(&self, txn_id: I) -> Result<TxnLockReadGuard<T>> {
        // before acquiring a permit, check if this version has already been committed
        if let Poll::Ready(result) = self.state().read_committed(&txn_id) {
            return result.map(TxnLockReadGuard::Committed);
        }

        let permit = self.semaphore.read(txn_id, Range).await?;
        Ok(self.state_mut().read_pending(&txn_id, permit))
    }

    /// Acquire a read lock for this value at `txn_id` synchronously, if possible.
    pub fn try_read(&self, txn_id: I) -> Result<TxnLockReadGuard<T>> {
        // before acquiring a permit, check if this version has already been committed
        if let Poll::Ready(result) = self.state().read_committed(&txn_id) {
            return result.map(TxnLockReadGuard::Committed);
        }

        let permit = self.semaphore.try_read(txn_id, Range)?;
        Ok(self.state_mut().read_pending(&txn_id, permit))
    }

    /// Roll back the state of this [`TxnLock`] at `txn_id`.
    pub fn rollback(&self, txn_id: &I) {
        let mut state = self.state_mut();

        assert!(
            !state.committed.contains_key(&txn_id),
            "cannot roll back committed transaction {}",
            txn_id
        );

        self.semaphore.finalize(txn_id, false);
        state.pending.remove(txn_id);
    }
}

impl<I: Copy + Ord, T: Clone + fmt::Debug> TxnLock<I, T> {
    /// Acquire a write lock for this value at `txn_id`.
    pub async fn write(&self, txn_id: I) -> Result<TxnLockWriteGuard<T>> {
        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let permit = self.semaphore.write(txn_id, Range).await?;
        self.state_mut().write(txn_id, permit)
    }

    /// Acquire a write lock for this value at `txn_id` synchronously, if possible.
    pub fn try_write(&self, txn_id: I) -> Result<TxnLockWriteGuard<T>> {
        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let permit = self.semaphore.try_write(txn_id, Range)?;
        self.state_mut().write(txn_id, permit)
    }
}
