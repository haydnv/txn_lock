//! A futures-aware read-write lock which supports transaction-specific versioning
//!
//! The value to lock must implement [`Clone`] since the lock keeps track of past
//! versions after committing. Call `finalize(txn_id)` to clear past versions which are older than
//! both `txn_id` and the most recent commit.
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
//! lock.finalize(&1);
//!
//! assert_eq!(lock.try_read(0).unwrap_err(), Error::Outdated);
//! assert_eq!(*lock.try_read(3).expect("current value"), "three");
//!
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

#[cfg(feature = "logging")]
use log::{debug, trace, warn};

use tokio::sync::{Notify, OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

/// An error which may occur when attempting to acquire a transactional lock
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

/// A read-only view of a [`TxnLock`] at a specific `TxnId`
pub enum TxnLockReadGuard<TxnId, T> {
    Read(Arc<Notify>, Arc<RwLock<T>>, TxnId, OwnedRwLockReadGuard<T>),
    Committed(Arc<T>),
}

impl<TxnId: Copy, T> Clone for TxnLockReadGuard<TxnId, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Read(notify, lock, txn_id, _guard) => {
                let guard = lock.clone().try_read_owned().expect("read lock");
                Self::Read(notify.clone(), lock.clone(), *txn_id, guard)
            }
            Self::Committed(value) => Self::Committed(value.clone()),
        }
    }
}

impl<TxnId, T> Deref for TxnLockReadGuard<TxnId, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Read(_notify, _lock, _txn_id, guard) => guard.deref(),
            Self::Committed(value) => value,
        }
    }
}

impl<TxnId, T> Drop for TxnLockReadGuard<TxnId, T> {
    fn drop(&mut self) {
        match self {
            Self::Read(notify, _, _, _) => notify.notify_waiters(),
            Self::Committed(_) => {}
        }
    }
}

impl<TxnId, T: fmt::Debug> fmt::Debug for TxnLockReadGuard<TxnId, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "transactional read lock on {:?}", self.deref())
    }
}

enum ExclusiveReadGuardState<TxnId, T> {
    Read(TxnLock<TxnId, T>, TxnId, OwnedRwLockWriteGuard<T>),
    Pending(Arc<Notify>, OwnedRwLockWriteGuard<T>),
    Upgraded,
}

/// An exclusive, upgradable read-only view of a [`TxnLock`] at a specific `TxnId`
pub struct TxnLockReadGuardExclusive<TxnId, T> {
    state: ExclusiveReadGuardState<TxnId, T>,
}

impl<TxnId: Ord, T: Clone> TxnLockReadGuardExclusive<TxnId, T> {
    /// Upgrade this exclusive read lock to a write lock.
    pub fn upgrade(mut self) -> TxnLockWriteGuard<T> {
        let mut state = ExclusiveReadGuardState::Upgraded;
        std::mem::swap(&mut self.state, &mut state);

        let (notify, guard) = match state {
            ExclusiveReadGuardState::Read(txn_lock, txn_id, guard) => {
                let mut state = txn_lock.state.lock().expect("lock state");
                let version = state.versions.get_mut(&txn_id).expect("version");

                let lock = match version {
                    Version::Read(lock) => lock.clone(),
                    Version::Pending(_lock) => unreachable!(),
                    Version::Committed(_) => unreachable!(),
                };

                *version = Version::Pending(lock);
                (txn_lock.notify.clone(), guard)
            }
            ExclusiveReadGuardState::Pending(notify, guard) => (notify, guard),
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
            ExclusiveReadGuardState::Read(_lock, _txn_id, value) => value,
            ExclusiveReadGuardState::Pending(_notify, guard) => guard.deref(),
            ExclusiveReadGuardState::Upgraded => unreachable!(),
        }
    }
}

impl<TxnId, T> Drop for TxnLockReadGuardExclusive<TxnId, T> {
    fn drop(&mut self) {
        match &self.state {
            ExclusiveReadGuardState::Read(txn_lock, _, _) => txn_lock.notify.notify_waiters(),
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

/// An exclusive mutable view of a [`TxnLock`] at a specific `TxnId`
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
    Read(Arc<RwLock<T>>),
    Pending(Arc<RwLock<T>>),
}

impl<T> Version<T> {
    fn is_final(&self) -> bool {
        match self {
            Self::Committed(_) => true,
            _ => false,
        }
    }
}

impl<T> Clone for Version<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Committed(value) => Self::Committed(value.clone()),
            Self::Read(lock) => Self::Read(lock.clone()),
            Self::Pending(lock) => Self::Pending(lock.clone()),
        }
    }
}

struct LockState<TxnId, T> {
    name: String,
    versions: BTreeMap<TxnId, Version<T>>,
}

/// A futures-aware read-write lock which supports transaction-specific versioning
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
        let mut versions = BTreeMap::new();
        versions.insert(txn_id, Version::Pending(Arc::new(RwLock::new(value))));

        Self {
            notify: Arc::new(Notify::new()),
            state: Arc::new(Mutex::new(LockState {
                name: name.to_string(),
                versions,
            })),
        }
    }
}

impl<TxnId: fmt::Display + fmt::Debug + Ord + Copy, T: Clone> TxnLock<TxnId, T> {
    fn read_inner(&self, txn_id: TxnId) -> Result<Option<Version<T>>> {
        let mut state = self.state.lock().expect("lock state");
        debug_assert!(!state.versions.is_empty());

        #[cfg(feature = "logging")]
        trace!("read {} at {}", state.name, txn_id);

        if let Some(version) = state.versions.get(&txn_id) {
            return Ok(Some(version.clone()));
        }

        let mut candidates = state
            .versions
            .iter()
            .rev()
            .filter(|(candidate_id, _)| *candidate_id < &txn_id);

        let canon = loop {
            if let Some((candidate_id, candidate)) = candidates.next() {
                assert_ne!(candidate_id, &txn_id);

                match candidate {
                    Version::Pending(_) => {
                        #[cfg(feature = "logging")]
                        debug!(
                            "read of {} at {} is pending a lock at {}",
                            state.name, txn_id, candidate_id
                        );

                        return Ok(None);
                    }
                    Version::Read(lock) => {
                        let value = lock.try_read().expect("read lock");
                        break T::clone(&*value);
                    }
                    Version::Committed(value) => break T::clone(&*value),
                }
            } else {
                return Err(Error::Outdated);
            }
        };

        let version = Version::Read(Arc::new(RwLock::new(canon)));
        state.versions.insert(txn_id, version.clone());
        Ok(Some(version))
    }

    /// Lock this value for reading at the given `txn_id`.
    pub async fn read(&self, txn_id: TxnId) -> Result<TxnLockReadGuard<TxnId, T>> {
        loop {
            if let Some(version) = self.read_inner(txn_id)? {
                let lock = match version {
                    Version::Read(lock) => lock,
                    Version::Pending(lock) => lock,
                    Version::Committed(value) => {
                        return Ok(TxnLockReadGuard::Committed(value));
                    }
                };

                let guard = lock.clone().read_owned().await;
                return Ok(TxnLockReadGuard::Read(
                    self.notify.clone(),
                    lock,
                    txn_id,
                    guard,
                ));
            };

            self.notify.notified().await;
        }
    }

    /// Synchronously Lock this value for reading at the given `txn_id`, if possible.
    pub fn try_read(&self, txn_id: TxnId) -> Result<TxnLockReadGuard<TxnId, T>> {
        if let Some(version) = self.read_inner(txn_id)? {
            let lock = match version {
                Version::Read(lock) => lock,
                Version::Pending(lock) => lock,
                Version::Committed(value) => return Ok(TxnLockReadGuard::Committed(value)),
            };

            lock.clone()
                .try_read_owned()
                .map(|guard| TxnLockReadGuard::Read(self.notify.clone(), lock, txn_id, guard))
                .map_err(|_err| Error::WouldBlock)
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
            if let Some(version) = self.read_inner(txn_id)? {
                return match version {
                    Version::Read(lock) => {
                        let guard = lock.write_owned().await;
                        let state = ExclusiveReadGuardState::Read(self.clone(), txn_id, guard);
                        Ok(TxnLockReadGuardExclusive { state })
                    }
                    Version::Pending(lock) => {
                        let guard = lock.write_owned().await;
                        let state = ExclusiveReadGuardState::Pending(self.notify.clone(), guard);
                        Ok(TxnLockReadGuardExclusive { state })
                    }
                    Version::Committed(_value) => Err(Error::Committed),
                };
            }

            self.notify.notified().await;
        }
    }

    /// Synchronously lock this value for exclusive reading at the given `txn_id`, if possible.
    pub fn try_read_exclusive(&self, txn_id: TxnId) -> Result<TxnLockReadGuardExclusive<TxnId, T>> {
        if let Some(version) = self.read_inner(txn_id)? {
            match version {
                Version::Read(lock) => {
                    let guard = lock.try_write_owned().map_err(|_err| Error::WouldBlock)?;
                    let state = ExclusiveReadGuardState::Read(self.clone(), txn_id, guard);
                    Ok(TxnLockReadGuardExclusive { state })
                }
                Version::Pending(lock) => {
                    let guard = lock.try_write_owned().map_err(|_err| Error::WouldBlock)?;
                    let state = ExclusiveReadGuardState::Pending(self.notify.clone(), guard);
                    Ok(TxnLockReadGuardExclusive { state })
                }
                Version::Committed(_value) => Err(Error::Committed),
            }
        } else {
            Err(Error::WouldBlock)
        }
    }

    fn write_inner(&self, txn_id: TxnId) -> Result<Option<Arc<RwLock<T>>>> {
        let mut state = self.state.lock().expect("lock state");
        debug_assert!(!state.versions.is_empty());

        #[cfg(feature = "logging")]
        trace!("write {} at {}", state.name, txn_id);

        let latest_read = *state.versions.keys().last().expect("latest version ID");

        if latest_read > txn_id {
            #[cfg(feature = "logging")]
            debug!(
                "cannot lock {} at {} since it already has a lock in the future at {}",
                state.name, txn_id, latest_read
            );

            return Err(Error::Conflict);
        }

        if latest_read == txn_id {
            let version = state
                .versions
                .get_mut(&latest_read)
                .expect("latest version");
            let lock = match version {
                Version::Read(lock) => lock.clone(),
                Version::Pending(lock) => return Ok(Some(lock.clone())),
                Version::Committed(_) => return Err(Error::Committed),
            };

            *version = Version::Pending(lock.clone());
            return Ok(Some(lock));
        }

        assert!(latest_read < txn_id);
        let mut versions = state.versions.values().rev();
        let canon = loop {
            if let Some(version) = versions.next() {
                match version {
                    Version::Read(_) => {}
                    Version::Pending(_) => return Ok(None),
                    Version::Committed(canon) => break canon,
                }
            } else {
                panic!("{} has no canonical version", state.name);
            }
        };

        let lock = Arc::new(RwLock::new(T::clone(&*canon)));
        let version = Version::Pending(lock.clone());
        state.versions.insert(txn_id, version);
        Ok(Some(lock))
    }

    /// Lock this value for writing at the given `txn_id`.
    pub async fn write(&self, txn_id: TxnId) -> Result<TxnLockWriteGuard<T>> {
        loop {
            if let Some(lock) = self.write_inner(txn_id)? {
                let guard = lock.write_owned().await;
                return Ok(TxnLockWriteGuard {
                    state: WriteGuardState::Pending(self.notify.clone(), guard),
                });
            }

            self.notify.notified().await;
        }
    }

    /// Synchronously lock this value for writing at the given `txn_id`, if possible.
    pub fn try_write(&self, txn_id: TxnId) -> Result<TxnLockWriteGuard<T>> {
        if let Some(lock) = self.write_inner(txn_id)? {
            lock.try_write_owned()
                .map(|guard| WriteGuardState::Pending(self.notify.clone(), guard))
                .map(|state| TxnLockWriteGuard { state })
                .map_err(|_err| Error::WouldBlock)
        } else {
            Err(Error::WouldBlock)
        }
    }
}

enum Commit<T> {
    Canon(Arc<T>),
    Pending,
    Noop,
}

impl<TxnId: fmt::Display + Ord + Copy, T: Clone> TxnLock<TxnId, T> {
    fn commit_inner(&self, txn_id: &TxnId) -> Commit<T> {
        let mut state = self.state.lock().expect("lock state");

        #[cfg(feature = "logging")]
        trace!("commit {} at {}...", state.name, txn_id);

        if let Some((txn_id, version)) = state.versions.remove_entry(txn_id) {
            let canon = match version {
                Version::Read(lock) => lock.try_read().expect("new canon").clone(),
                Version::Pending(lock) => {
                    if let Ok(value) = lock.try_read() {
                        value.clone()
                    } else {
                        return Commit::Pending;
                    }
                }
                Version::Committed(value) => {
                    #[cfg(feature = "logging")]
                    warn!("duplicate commit at {}...", txn_id);

                    T::clone(&*value)
                }
            };

            let canon = Arc::new(canon);

            state
                .versions
                .insert(txn_id, Version::Committed(canon.clone()));

            Commit::Canon(canon)
        } else {
            Commit::Noop
        }
    }

    /// Commit the value of this [`TxnLock`] at the given `txn_id`.
    /// This will wait until any earlier write locks have been committed or rolled back.
    ///
    /// Panics:
    /// - when called with a `txn_id` which has already been finalized.
    pub async fn commit(&self, txn_id: &TxnId) -> Option<Arc<T>> {
        let canon = loop {
            match self.commit_inner(txn_id) {
                Commit::Pending => {}
                Commit::Noop => break None,
                Commit::Canon(canon) => break Some(canon),
            }
        };

        self.notify.notify_waiters();
        canon
    }

    /// Roll back the value of this [`TxnLock`] at the given `txn_id`.
    ///
    /// Returns the version that was rolled back, if any lock was acquired at `txn_id`.
    pub async fn rollback(&self, txn_id: &TxnId) -> Option<impl Deref<Target = T>> {
        #[cfg(feature = "logging")]
        trace!("rollback at {}", txn_id);

        let lock = {
            let mut state = self.state.lock().expect("lock state");
            debug_assert!(!state.versions.is_empty());

            if let Some(version) = state.versions.remove(txn_id) {
                assert!(
                    !state.versions.is_empty(),
                    "canonical version of {} was rolled back at {}",
                    state.name,
                    txn_id
                );

                self.notify.notify_waiters();

                match version {
                    Version::Read(lock) => Some(lock),
                    Version::Pending(lock) => Some(lock),
                    Version::Committed(_value) => {
                        panic!("tried to roll back a committed version at {}", txn_id)
                    }
                }
            } else {
                None
            }
        };

        if let Some(lock) = lock {
            Some(lock.read_owned().await)
        } else {
            None
        }
    }

    /// Drop all values of this [`TxnLock`] older than the given `txn_id`,
    /// except for the last commit.
    ///
    /// Returns `Some(Arc<T>)` with the finalized version, if any lock was acquired at `txn_id`.
    pub fn finalize(&self, txn_id: &TxnId) -> Option<Arc<T>> {
        #[cfg(feature = "logging")]
        trace!("finalize {}", txn_id);

        let mut state = self.state.lock().expect("lock state");
        assert!(!state.versions.is_empty());

        let last_commit = if let Some(version_id) = state
            .versions
            .iter()
            .rev()
            .filter(|(_id, version)| version.is_final())
            .map(|(version_id, _)| version_id)
            .next()
        {
            *version_id
        } else {
            panic!("{} has no canonical version", state.name);
        };

        let cutoff = Ord::min(last_commit, *txn_id);

        // this should normally be empty, i.e. it won't allocate
        let mut older: Vec<TxnId> = state
            .versions
            .keys()
            .rev()
            .filter(|version_id| *version_id < &cutoff)
            .copied()
            .collect();

        while let Some(version_id) = older.pop() {
            state.versions.remove(&version_id);
        }

        if txn_id == &last_commit {
            if let Some(version) = state.versions.get(txn_id) {
                if let Version::Committed(canon) = version {
                    Some(canon.clone())
                } else {
                    panic!("{} has no canonical version", state.name);
                }
            } else {
                None
            }
        } else if let Some(version) = state.versions.remove(txn_id) {
            if let Version::Committed(canon) = version {
                Some(canon)
            } else {
                panic!("{} has no canonical version", state.name);
            }
        } else {
            None
        }
    }
}

impl<TxnId, T> fmt::Debug for TxnLock<TxnId, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let state = self.state.lock().expect("lock state");
        write!(f, "transaction lock {}", state.name)
    }
}
