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
//! let lock = TxnLock::new("example", 0, "zero");
//!
//! assert_eq!(*lock.try_read(0).expect("read"), "zero");
//! assert_eq!(lock.try_write(1).unwrap_err(), Error::WouldBlock);
//!
//! block_on(lock.commit(&0));
//!
//! {
//!     let mut guard = lock.try_write(1).expect("write lock");
//!     *guard = "one";
//! }
//!
//! assert_eq!(*lock.try_read(0).expect("read past version"), "zero");
//! assert_eq!(*lock.try_read(1).expect("read current version"), "one");
//!
//! block_on(lock.commit(&1));
//!
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
//! assert_eq!(lock.try_read(0).unwrap_err(), Error::Outdated);
//! assert_eq!(*lock.try_read(3).expect("current value"), "three");
//!
//! ```

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

#[cfg(feature = "logging")]
use log::{debug, trace, warn};

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

pub enum TxnLockReadGuard<TxnId, T> {
    Committed(Arc<T>),
    Pending(Arc<Notify>, Arc<RwLock<T>>, TxnId, OwnedRwLockReadGuard<T>),
}

impl<TxnId: Clone, T> Clone for TxnLockReadGuard<TxnId, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Committed(value) => Self::Committed(value.clone()),
            Self::Pending(notify, lock, txn_id, _guard) => {
                let guard = lock.clone().try_read_owned().expect("read lock");
                Self::Pending(notify.clone(), lock.clone(), txn_id.clone(), guard)
            }
        }
    }
}

impl<TxnId, T> Deref for TxnLockReadGuard<TxnId, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Committed(value) => value,
            Self::Pending(_notify, _lock, _txn_id, guard) => guard.deref(),
        }
    }
}

impl<TxnId, T> Drop for TxnLockReadGuard<TxnId, T> {
    fn drop(&mut self) {
        match self {
            Self::Committed(_) => {}
            Self::Pending(notify, _, _, _) => notify.notify_waiters(),
        }
    }
}

impl<TxnId, T: fmt::Debug> fmt::Debug for TxnLockReadGuard<TxnId, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "transactional read lock on {:?}", self.deref())
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

        let state = WriteGuardState::Pending(notify, guard);
        TxnLockWriteGuard { state }
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

enum WriteGuardState<T> {
    Pending(Arc<Notify>, OwnedRwLockWriteGuard<T>),
    Downgraded,
}

pub struct TxnLockWriteGuard<T> {
    state: WriteGuardState<T>,
}

impl<T> TxnLockWriteGuard<T> {
    pub fn downgrade<TxnId>(mut self) -> TxnLockReadGuardExclusive<TxnId, T> {
        let mut state = WriteGuardState::Downgraded;
        std::mem::swap(&mut self.state, &mut state);

        match state {
            WriteGuardState::Pending(notify, guard) => TxnLockReadGuardExclusive {
                state: ExclusiveReadGuardState::Pending(notify, guard),
            },
            WriteGuardState::Downgraded => unreachable!(),
        }
    }
}

impl<T> Deref for TxnLockWriteGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match &self.state {
            WriteGuardState::Pending(_notify, guard) => guard.deref(),
            WriteGuardState::Downgraded => unreachable!(),
        }
    }
}

impl<T> DerefMut for TxnLockWriteGuard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match &mut self.state {
            WriteGuardState::Pending(_notify, guard) => guard.deref_mut(),
            WriteGuardState::Downgraded => unreachable!(),
        }
    }
}

impl<T> Drop for TxnLockWriteGuard<T> {
    fn drop(&mut self) {
        match &self.state {
            WriteGuardState::Pending(notify, _guard) => notify.notify_waiters(),
            WriteGuardState::Downgraded => {}
        }
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
    name: String,
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
    pub fn new<Name: fmt::Display>(name: Name, txn_id: TxnId, value: T) -> Self {
        let mut versions = VecDeque::new();
        versions.push_back((txn_id, Version::Pending(Arc::new(RwLock::new(value)))));

        Self {
            notify: Arc::new(Notify::new()),
            state: Arc::new(Mutex::new(LockState {
                name: name.to_string(),
                versions,
            })),
        }
    }
}

impl<TxnId: Ord, T> TxnLock<TxnId, T> {
    fn create_value(&self, txn_id: TxnId, value: T) -> Arc<RwLock<T>> {
        let mut state = self.state.lock().expect("lock state");
        debug_assert!(!state.versions.is_empty());

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
        debug_assert!(!state.versions.is_empty());

        #[cfg(feature = "logging")]
        trace!("read {} at {}", state.name, txn_id);

        let mut version = None;
        for (candidate_id, candidate) in &state.versions {
            if candidate_id < txn_id {
                if candidate.is_pending() {
                    #[cfg(feature = "logging")]
                    debug!(
                        "read of {} at {} is pending a write at {}",
                        state.name, txn_id, candidate_id
                    );

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
    pub async fn read(&self, txn_id: TxnId) -> Result<TxnLockReadGuard<TxnId, T>> {
        loop {
            if let Some((_ordering, version)) = self.read_inner(&txn_id)? {
                return match version {
                    Version::Pending(lock) => {
                        let guard = lock.clone().read_owned().await;
                        Ok(TxnLockReadGuard::Pending(
                            self.notify.clone(),
                            lock,
                            txn_id,
                            guard,
                        ))
                    }
                    Version::Committed(value) => Ok(TxnLockReadGuard::Committed(value)),
                };
            }

            self.notify.notified().await;
        }
    }

    /// Synchronously Lock this value for reading at the given `txn_id`, if possible.
    pub fn try_read(&self, txn_id: TxnId) -> Result<TxnLockReadGuard<TxnId, T>> {
        if let Some((_ordering, version)) = self.read_inner(&txn_id)? {
            match version {
                Version::Pending(lock) => lock
                    .clone()
                    .try_read_owned()
                    .map(|guard| {
                        TxnLockReadGuard::Pending(self.notify.clone(), lock, txn_id, guard)
                    })
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
        debug_assert!(!state.versions.is_empty());

        #[cfg(feature = "logging")]
        trace!("write {} at {}", state.name, txn_id);

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
                    #[cfg(feature = "logging")]
                    debug!(
                        "write to {} at {} is pending a write at {}",
                        state.name, txn_id, version_id
                    );

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
                        state: WriteGuardState::Pending(self.notify.clone(), guard),
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
                    state: WriteGuardState::Pending(self.notify.clone(), guard),
                })
                .map_err(|_err| Error::WouldBlock),
        }
    }
}

impl<TxnId: fmt::Display + Ord, T: Clone> TxnLock<TxnId, T> {
    fn commit_inner(&self, txn_id: &TxnId) -> Option<Arc<T>> {
        let mut state = self.state.lock().expect("lock state");

        #[cfg(feature = "logging")]
        trace!("commit {} at {}", state.name, txn_id);

        for (version_id, version) in state.versions.iter() {
            if version_id >= txn_id {
                break;
            }

            if version.is_pending() {
                #[cfg(feature = "logging")]
                debug!(
                    "commit at {} is waiting on a commit at {}",
                    txn_id, version_id
                );

                return None;
            }
        }

        let pending = state
            .versions
            .back()
            .map(|(version_id, _version)| version_id == txn_id)
            .expect("latest version");

        if pending {
            let (txn_id, version) = state.versions.pop_back().expect("version to commit");

            let value = match version {
                Version::Committed(value) => {
                    #[cfg(feature = "logging")]
                    warn!("duplicate commit {}", txn_id);
                    value
                }
                Version::Pending(lock) => {
                    let guard = lock.try_read().expect("canon");
                    Arc::new(T::clone(&*guard))
                }
            };

            state
                .versions
                .push_back((txn_id, Version::Committed(value.clone())));

            Some(value)
        } else {
            if &state.versions[0].0 > txn_id {
                panic!("value has already been finalized at {}", txn_id);
            }

            let (left, _right) = binary_search(&state.versions, txn_id);
            match &state.versions[left] {
                (version_id, Version::Committed(value)) => {
                    assert!(version_id <= txn_id);
                    Some(value.clone())
                }
                _ => unreachable!(),
            }
        }
    }

    /// Commit the value of this [`TxnLock`] at the given `txn_id`.
    /// This will wait until any earlier write locks have been committed or rolled back.
    ///
    /// Panics:
    /// - when called with a `txn_id` which has already been finalized.
    pub async fn commit(&self, txn_id: &TxnId) -> Arc<T> {
        let canon = loop {
            if let Some(canon) = self.commit_inner(txn_id) {
                break canon;
            }
        };

        self.notify.notify_waiters();
        canon
    }

    /// Roll back the value of this [`TxnLock`] at the given `txn_id`.
    pub fn rollback(&self, txn_id: &TxnId) {
        #[cfg(feature = "logging")]
        trace!("rollback at {}", txn_id);

        let mut state = self.state.lock().expect("lock state");
        debug_assert!(!state.versions.is_empty());

        if &state.versions[0].0 > txn_id {
            return;
        }

        let (left, right) = binary_search(&state.versions, txn_id);
        if left == right {
            state.versions.remove(left);
            self.notify.notify_waiters();
        }

        assert!(!state.versions.is_empty());
    }

    /// Drop all values of this [`TxnLock`] older than the given `txn_id`.
    pub fn finalize(&self, txn_id: &TxnId) {
        #[cfg(feature = "logging")]
        trace!("finalize {}", txn_id);

        let mut state = self.state.lock().expect("lock state");
        debug_assert!(!state.versions.is_empty());

        while &state.versions[0].0 < txn_id {
            state.versions.pop_front();
        }

        if &state.versions[0].0 == txn_id {
            if state.versions.len() == 1 {
                assert!(!state.versions[0].1.is_pending());
            } else {
                state.versions.pop_front();
            }
        }

        assert!(!state.versions.is_empty());
        self.notify.notify_waiters();
    }
}

impl<TxnId, T> fmt::Debug for TxnLock<TxnId, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let state = self.state.lock().expect("lock state");
        write!(f, "transaction lock {}", state.name)
    }
}

fn binary_search<TxnId: PartialOrd, T>(
    versions: &VecDeque<(TxnId, T)>,
    txn_id: &TxnId,
) -> (usize, usize) {
    assert!(!versions.is_empty());

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
