//! A futures-aware read-write lock which supports transaction-specific versioning
//!
//! The value to lock must implement [`Clone`] since the lock may keep track of multiple past
//! values after committing.
//!
//! Example:
//! ```
//! use futures::executor::block_on;
//! use txn_lock::*;
//!
//! let lock = TxnLock::new(0, "zero");
//! {
//!     let mut guard = lock.try_write(1).expect("write lock");
//!     *guard = "one";
//! }
//!
//! block_on(lock.commit(&1));
//!
//! assert_eq!(*lock.try_read(&0).expect("old value"), "zero");
//! assert_eq!(*lock.try_read(&1).expect("current value"), "one");
//! assert_eq!(*lock.try_read_exclusive(2).expect("new value"), "one");
//!
//! lock.rollback(&2);
//!
//! {
//!     let mut guard = lock.try_write(3).expect("write lock");
//!     *guard = "three";
//! }
//!
//! lock.finalize(&2);
//!
//! assert_eq!(lock.try_read(&0).unwrap_err(), Error::Outdated);
//! assert_eq!(*lock.try_read(&2).expect("old value"), "one");
//! assert_eq!(*lock.try_read(&3).expect("current value"), "three");
//! assert_eq!(lock.try_read(&4).unwrap_err(), Error::WouldBlock);
//!
//! ```

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

use tokio::sync::{Notify, OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum Error {
    Committed,
    Conflict,
    Outdated,
    WouldBlock,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Self::Committed => "cannot acquire an exclusive lock after committing",
            Self::Conflict => "there is already a transactional write lock in the future",
            Self::Outdated => "the value has already been finalized",
            Self::WouldBlock => "unable to acquire a lock",
        })
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for Error {}

type Result<T> = std::result::Result<T, Error>;

pub enum TxnLockReadGuard<T> {
    Committed(Arc<T>),
    Pending(Arc<Notify>, OwnedRwLockReadGuard<T>),
}

impl<T> Deref for TxnLockReadGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Committed(value) => value,
            Self::Pending(_notify, guard) => guard.deref(),
        }
    }
}

impl<T> Drop for TxnLockReadGuard<T> {
    fn drop(&mut self) {
        match self {
            Self::Committed(_) => {}
            Self::Pending(notify, _) => notify.notify_waiters(),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for TxnLockReadGuard<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self.deref(), f)
    }
}

impl<T: fmt::Display> fmt::Display for TxnLockReadGuard<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self.deref(), f)
    }
}

enum ExclusiveReadGuardState<TxnId, T> {
    Canon(TxnLock<TxnId, T>, TxnId, Arc<T>),
    Pending(Arc<Notify>, OwnedRwLockWriteGuard<T>),
    Upgraded,
}

pub struct TxnLockReadGuardExclusive<TxnId, T> {
    state: ExclusiveReadGuardState<TxnId, T>,
}

impl<TxnId: Ord, T: Clone> TxnLockReadGuardExclusive<TxnId, T> {
    /// Upgrade this exclusive read lock to a write lock.
    pub fn upgrade(mut self) -> TxnLockWriteGuard<T> {
        let mut state = ExclusiveReadGuardState::Upgraded;
        std::mem::swap(&mut self.state, &mut state);

        let (notify, guard) = match state {
            ExclusiveReadGuardState::Pending(notify, guard) => (notify, guard),
            ExclusiveReadGuardState::Canon(txn_lock, txn_id, value) => {
                let lock = txn_lock.create_value(txn_id, T::clone(&*value));
                (
                    txn_lock.notify.clone(),
                    lock.try_write_owned().expect("write guard"),
                )
            }
            ExclusiveReadGuardState::Upgraded => unreachable!(),
        };

        TxnLockWriteGuard { notify, guard }
    }
}

impl<TxnId, T> Deref for TxnLockReadGuardExclusive<TxnId, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match &self.state {
            ExclusiveReadGuardState::Canon(_lock, _txn_id, value) => value,
            ExclusiveReadGuardState::Pending(_notify, guard) => guard.deref(),
            ExclusiveReadGuardState::Upgraded => unreachable!(),
        }
    }
}

impl<TxnId, T> Drop for TxnLockReadGuardExclusive<TxnId, T> {
    fn drop(&mut self) {
        match &self.state {
            ExclusiveReadGuardState::Canon(txn_lock, _, _) => txn_lock.notify.notify_waiters(),
            ExclusiveReadGuardState::Pending(notify, _) => notify.notify_waiters(),
            ExclusiveReadGuardState::Upgraded => {}
        };
    }
}

impl<TxnId, T: fmt::Debug> fmt::Debug for TxnLockReadGuardExclusive<TxnId, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self.deref(), f)
    }
}

impl<TxnId, T: fmt::Display> fmt::Display for TxnLockReadGuardExclusive<TxnId, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self.deref(), f)
    }
}

pub struct TxnLockWriteGuard<T> {
    notify: Arc<Notify>,
    guard: OwnedRwLockWriteGuard<T>,
}

impl<T> Deref for TxnLockWriteGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

impl<T> DerefMut for TxnLockWriteGuard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard.deref_mut()
    }
}

impl<T> Drop for TxnLockWriteGuard<T> {
    fn drop(&mut self) {
        self.notify.notify_waiters();
    }
}

impl<T: fmt::Debug> fmt::Debug for TxnLockWriteGuard<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self.deref(), f)
    }
}

impl<T: fmt::Display> fmt::Display for TxnLockWriteGuard<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self.deref(), f)
    }
}

enum Version<T> {
    Committed(Arc<T>),
    Pending(Arc<RwLock<T>>),
}

impl<T> Version<T> {
    fn is_pending(&self) -> bool {
        match self {
            Self::Committed(_) => false,
            Self::Pending(_) => true,
        }
    }
}

impl<T> Clone for Version<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Committed(value) => Self::Committed(value.clone()),
            Self::Pending(lock) => Self::Pending(lock.clone()),
        }
    }
}

struct LockState<TxnId, T> {
    versions: VecDeque<(TxnId, Version<T>)>,
}

pub struct TxnLock<TxnId, T> {
    notify: Arc<Notify>,
    state: Arc<Mutex<LockState<TxnId, T>>>,
}

impl<TxnId, T> Clone for TxnLock<TxnId, T> {
    fn clone(&self) -> Self {
        Self {
            notify: self.notify.clone(),
            state: self.state.clone(),
        }
    }
}

impl<TxnId: Ord, T> TxnLock<TxnId, T> {
    /// Create a new transactional lock.
    pub fn new(last_commit: TxnId, value: T) -> Self {
        let mut versions = VecDeque::new();
        versions.push_back((last_commit, Version::Committed(Arc::new(value))));

        Self {
            notify: Arc::new(Notify::new()),
            state: Arc::new(Mutex::new(LockState { versions })),
        }
    }
}

impl<TxnId: Ord, T> TxnLock<TxnId, T> {
    fn create_value(&self, txn_id: TxnId, value: T) -> Arc<RwLock<T>> {
        let mut state = self.state.lock().expect("lock state");

        if let Some((prev_id, _)) = state.versions.iter().last() {
            assert!(prev_id < &txn_id);
        } else {
            panic!("transaction lock has no canonical version");
        }

        let lock = Arc::new(RwLock::new(value));
        let version = Version::Pending(lock.clone());
        state.versions.push_back((txn_id, version));
        lock
    }
}

enum Write<TxnId, T> {
    Pending(TxnId),
    Version(Arc<RwLock<T>>),
}

impl<TxnId: fmt::Display + fmt::Debug + Ord, T: Clone> TxnLock<TxnId, T> {
    fn read_inner(&self, txn_id: &TxnId) -> Result<Option<(Ordering, Version<T>)>> {
        let state = self.state.lock().expect("lock state");
        assert!(!state.versions.is_empty());

        let mut version = None;
        for (candidate_id, candidate) in &state.versions {
            if candidate_id < txn_id {
                if candidate.is_pending() {
                    return Ok(None);
                } else {
                    version = Some((Ordering::Less, candidate));
                }
            } else if candidate_id == txn_id {
                return Ok(Some((Ordering::Equal, candidate.clone())));
            } else {
                break;
            }
        }

        if let Some((ordering, version)) = version {
            return Ok(Some((ordering, version.clone())));
        } else {
            Err(Error::Outdated)
        }
    }

    /// Lock this value for reading at the given `txn_id`.
    pub async fn read(&self, txn_id: &TxnId) -> Result<TxnLockReadGuard<T>> {
        loop {
            if let Some((_ordering, version)) = self.read_inner(txn_id)? {
                return match version {
                    Version::Pending(lock) => {
                        let guard = lock.read_owned().await;
                        Ok(TxnLockReadGuard::Pending(self.notify.clone(), guard))
                    }
                    Version::Committed(value) => Ok(TxnLockReadGuard::Committed(value)),
                };
            }

            self.notify.notified().await;
        }
    }

    /// Synchronously Lock this value for reading at the given `txn_id`, if possible.
    pub fn try_read(&self, txn_id: &TxnId) -> Result<TxnLockReadGuard<T>> {
        if let Some((_ordering, version)) = self.read_inner(txn_id)? {
            match version {
                Version::Pending(lock) => lock
                    .try_read_owned()
                    .map(|guard| TxnLockReadGuard::Pending(self.notify.clone(), guard))
                    .map_err(|_err| Error::WouldBlock),

                Version::Committed(value) => Ok(TxnLockReadGuard::Committed(value)),
            }
        } else {
            Err(Error::WouldBlock)
        }
    }

    /// Lock this value for exclusive reading at the given `txn_id`.
    pub async fn read_exclusive(
        &self,
        txn_id: TxnId,
    ) -> Result<TxnLockReadGuardExclusive<TxnId, T>> {
        loop {
            if let Some((ordering, version)) = self.read_inner(&txn_id)? {
                return match version {
                    Version::Committed(_) if ordering == Ordering::Equal => Err(Error::Committed),
                    Version::Committed(value) => {
                        assert_eq!(ordering, Ordering::Less);
                        let state = ExclusiveReadGuardState::Canon(self.clone(), txn_id, value);
                        Ok(TxnLockReadGuardExclusive { state })
                    }
                    Version::Pending(lock) => {
                        assert_eq!(ordering, Ordering::Equal);
                        let guard = lock.write_owned().await;
                        let state = ExclusiveReadGuardState::Pending(self.notify.clone(), guard);
                        Ok(TxnLockReadGuardExclusive { state })
                    }
                };
            }

            self.notify.notified().await;
        }
    }

    /// Synchronously lock this value for exclusive reading at the given `txn_id`, if possible.
    pub fn try_read_exclusive(&self, txn_id: TxnId) -> Result<TxnLockReadGuardExclusive<TxnId, T>> {
        if let Some((ordering, version)) = self.read_inner(&txn_id)? {
            match version {
                Version::Committed(_) if ordering == Ordering::Equal => Err(Error::Committed),
                Version::Committed(value) => {
                    assert_eq!(ordering, Ordering::Less);
                    let state = ExclusiveReadGuardState::Canon(self.clone(), txn_id, value);
                    Ok(TxnLockReadGuardExclusive { state })
                }
                Version::Pending(lock) => {
                    assert_eq!(ordering, Ordering::Equal);
                    let guard = lock.try_write_owned().map_err(|_err| Error::WouldBlock)?;
                    let state = ExclusiveReadGuardState::Pending(self.notify.clone(), guard);
                    Ok(TxnLockReadGuardExclusive { state })
                }
            }
        } else {
            Err(Error::WouldBlock)
        }
    }

    fn write_inner(&self, txn_id: TxnId) -> Result<Write<TxnId, T>> {
        let mut state = self.state.lock().expect("lock state");
        assert!(!state.versions.is_empty());

        if let Some((version_id, version)) = state.versions.back() {
            if version_id == &txn_id {
                return match version {
                    Version::Pending(lock) => Ok(Write::Version(lock.clone())),
                    Version::Committed(_value) => Err(Error::Committed),
                };
            } else if version_id > &txn_id {
                return Err(Error::Conflict);
            } else if version_id < &txn_id {
                if version.is_pending() {
                    return Ok(Write::Pending(txn_id));
                }
            }
        }

        if &state.versions.front().unwrap().0 > &txn_id {
            return Err(Error::Outdated);
        }

        let (left, _right) = binary_search(&state.versions, &txn_id);
        let canon = match &state.versions[left].1 {
            Version::Committed(canon) => canon,
            Version::Pending(_lock) => unreachable!(),
        };

        let lock = Arc::new(RwLock::new(T::clone(&*canon)));
        state
            .versions
            .push_back((txn_id, Version::Pending(lock.clone())));

        Ok(Write::Version(lock.clone()))
    }

    /// Lock this value for writing at the given `txn_id`.
    pub async fn write(&self, mut txn_id: TxnId) -> Result<TxnLockWriteGuard<T>> {
        loop {
            txn_id = match self.write_inner(txn_id)? {
                Write::Pending(txn_id) => txn_id,
                Write::Version(lock) => {
                    let guard = lock.write_owned().await;
                    return Ok(TxnLockWriteGuard {
                        notify: self.notify.clone(),
                        guard,
                    });
                }
            };

            self.notify.notified().await;
        }
    }

    /// Synchronously lock this value for writing at the given `txn_id`, if possible.
    pub fn try_write(&self, txn_id: TxnId) -> Result<TxnLockWriteGuard<T>> {
        match self.write_inner(txn_id)? {
            Write::Pending(_) => Err(Error::WouldBlock),
            Write::Version(lock) => lock
                .try_write_owned()
                .map(|guard| TxnLockWriteGuard {
                    notify: self.notify.clone(),
                    guard,
                })
                .map_err(|_err| Error::WouldBlock),
        }
    }
}

impl<TxnId: fmt::Display + Ord, T: Clone> TxnLock<TxnId, T> {
    /// Commit the value of this [`TxnLock`] at the given `txn_id`.
    /// This will wait until any earlier write locks have been committed or rolled back.
    ///
    /// Panics:
    ///  - when called twice with the same `txn_id`
    ///  - when called with a `txn_id` which has already been finalized
    pub async fn commit(&self, txn_id: &TxnId) -> Arc<T> {
        let mut state = loop {
            let state = self.state.lock().expect("lock state");
            assert!(!state.versions.is_empty());

            if state
                .versions
                .iter()
                .filter(|(version_id, _)| version_id < &txn_id)
                .any(|(_, version)| version.is_pending())
            {
                std::mem::drop(state);
                self.notify.notified().await;
            } else {
                break state;
            }
        };

        let pending = state
            .versions
            .back()
            .map(|(version_id, _version)| version_id == txn_id)
            .expect("latest version");

        let canon = if pending {
            let (txn_id, version) = state.versions.pop_back().expect("version to commit");

            let value = match version {
                Version::Committed(_value) => panic!("duplicate commit {}", txn_id),
                Version::Pending(lock) => {
                    let guard = lock.try_read().expect("canon");
                    Arc::new(T::clone(&*guard))
                }
            };

            state
                .versions
                .push_back((txn_id, Version::Committed(value.clone())));

            value
        } else {
            if &state.versions[0].0 > txn_id {
                panic!("value has already been finalized at {}", txn_id);
            }

            let (left, _right) = binary_search(&state.versions, txn_id);
            match &state.versions[left] {
                (version_id, Version::Committed(value)) => {
                    assert!(version_id <= txn_id);
                    value.clone()
                }
                _ => unreachable!(),
            }
        };

        self.notify.notify_waiters();
        canon
    }

    /// Roll back the value of this [`TxnLock`] at the given `txn_id`.
    pub fn rollback(&self, txn_id: &TxnId) {
        let mut state = self.state.lock().expect("lock state");
        assert!(!state.versions.is_empty());

        if &state.versions[0].0 > txn_id {
            return;
        }

        let (left, right) = binary_search(&state.versions, txn_id);
        if left == right {
            state.versions.remove(left);
            self.notify.notify_waiters();
        }
    }

    /// Drop all values of this [`TxnLock`] older than the given `txn_id`.
    pub fn finalize(&self, txn_id: &TxnId) {
        let mut state = self.state.lock().expect("lock state");
        assert!(!state.versions.is_empty());

        while state.versions.len() > 1 && &state.versions[1].0 <= txn_id {
            state.versions.pop_front();
        }
    }
}

fn binary_search<TxnId: PartialOrd, T>(
    versions: &VecDeque<(TxnId, T)>,
    txn_id: &TxnId,
) -> (usize, usize) {
    let mut left = 0;
    let mut right = versions.len();

    while (right - left) > 1 {
        let mid = right / 2;

        match &versions[mid].0 {
            version_id if version_id > txn_id => right = mid,
            version_id if version_id < txn_id => left = mid,
            _ => {
                right = mid;
                left = mid;
            }
        }
    }

    (left, right)
}
