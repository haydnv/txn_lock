//! A futures-aware read-write lock which supports transaction-specific versioning
//!
//! The value to lock must implement [`Clone`] since the lock keeps track of past
//! versions after committing. Call `finalize(txn_id)` to clear past versions which are older than
//! both `txn_id` and the most recent commit.

use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

#[cfg(feature = "logging")]
use log::{debug, trace};
use tokio::sync::{Notify, OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use super::{Error, Result};

#[derive(Copy, Debug, Clone, Eq, PartialEq)]
enum VersionState {
    Read,
    Exclude,
    Write,
    Commit,
}

struct Version<T> {
    state: Arc<Mutex<VersionState>>,
    lock: Arc<RwLock<T>>,
}

impl<T> Version<T> {
    fn new(state: VersionState, value: T) -> Self {
        Self {
            state: Arc::new(Mutex::new(state)),
            lock: Arc::new(RwLock::new(value)),
        }
    }

    fn is_exclusive(&self) -> bool {
        let state = self.state.lock().expect("version state");
        match &*state {
            VersionState::Read => false,
            VersionState::Exclude => true,
            VersionState::Write => true,
            VersionState::Commit => false,
        }
    }

    fn is_final(&self) -> bool {
        let state = self.state.lock().expect("version state");
        &*state == &VersionState::Commit
    }

    fn commit(&self) -> Option<OwnedRwLockReadGuard<T>> {
        let guard = self.lock.clone().try_read_owned().ok()?;

        let mut state = self.state.lock().expect("version state");
        *state = VersionState::Commit;

        Some(guard)
    }

    async fn rollback(self) -> OwnedRwLockReadGuard<T> {
        if self.is_final() {
            panic!("cannot roll back a committed version");
        }

        self.lock.read_owned().await
    }

    async fn read(self, notify: Arc<Notify>) -> TxnLockReadGuard<T> {
        let guard = self.lock.clone().read_owned().await;

        TxnLockReadGuard {
            lock: self.lock,
            guard,
            notify,
        }
    }

    fn try_read(self, notify: Arc<Notify>) -> Result<TxnLockReadGuard<T>> {
        self.lock
            .clone()
            .try_read_owned()
            .map(|guard| TxnLockReadGuard {
                lock: self.lock,
                guard,
                notify,
            })
            .map_err(|_| Error::WouldBlock)
    }

    async fn read_exclusive(self, notify: Arc<Notify>) -> Result<TxnLockReadGuardExclusive<T>> {
        {
            let mut state = self.state.lock().expect("version state");
            if &*state == &VersionState::Commit {
                return Err(Error::Committed);
            } else if &*state == &VersionState::Read {
                *state = VersionState::Exclude;
            };
        }

        let guard = self.lock.clone().write_owned().await;

        Ok(TxnLockReadGuardExclusive {
            state: ReadGuardState::Active(guard, self),
            notify,
        })
    }

    fn try_read_exclusive(self, notify: Arc<Notify>) -> Result<TxnLockReadGuardExclusive<T>> {
        {
            let mut state = self.state.lock().expect("version state");
            if &*state == &VersionState::Commit {
                return Err(Error::Committed);
            } else if &*state == &VersionState::Read {
                *state = VersionState::Exclude;
            };
        }

        self.lock
            .clone()
            .try_write_owned()
            .map(|guard| TxnLockReadGuardExclusive {
                state: ReadGuardState::Active(guard, self),
                notify,
            })
            .map_err(|_| Error::WouldBlock)
    }

    async fn write(self, notify: Arc<Notify>) -> Result<TxnLockWriteGuard<T>> {
        {
            let mut state = self.state.lock().expect("version state");
            if &*state == &VersionState::Commit {
                return Err(Error::Committed);
            } else {
                *state = VersionState::Write;
            };
        }

        let guard = self.lock.clone().write_owned().await;

        Ok(TxnLockWriteGuard {
            state: WriteGuardState::Active(guard, self, notify),
        })
    }

    fn try_write(self, notify: Arc<Notify>) -> Result<TxnLockWriteGuard<T>> {
        {
            let mut state = self.state.lock().expect("version state");
            if &*state == &VersionState::Commit {
                return Err(Error::Committed);
            } else {
                *state = VersionState::Write;
            };
        }

        self.lock
            .clone()
            .try_write_owned()
            .map(|guard| TxnLockWriteGuard {
                state: WriteGuardState::Active(guard, self, notify),
            })
            .map_err(|_| Error::WouldBlock)
    }
}

impl<T> Clone for Version<T> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            lock: self.lock.clone(),
        }
    }
}

/// A read-only view of a [`TxnLock`] at a specific `TxnId`
pub struct TxnLockReadGuard<T> {
    lock: Arc<RwLock<T>>,
    guard: OwnedRwLockReadGuard<T>,
    notify: Arc<Notify>,
}

impl<T> Clone for TxnLockReadGuard<T> {
    fn clone(&self) -> Self {
        Self {
            lock: self.lock.clone(),

            guard: self
                .lock
                .clone()
                .try_read_owned()
                .expect("read guard clone"),

            notify: self.notify.clone(),
        }
    }
}

impl<T> Deref for TxnLockReadGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

impl<T> Drop for TxnLockReadGuard<T> {
    fn drop(&mut self) {
        self.notify.notify_waiters()
    }
}

impl<T: fmt::Debug> fmt::Debug for TxnLockReadGuard<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "transactional read lock on {:?}", self.deref())
    }
}

enum ReadGuardState<T> {
    Active(OwnedRwLockWriteGuard<T>, Version<T>),
    Downgraded,
    Upgraded,
}

/// An exclusive, upgradable read-only view of a [`TxnLock`] at a specific `TxnId`
pub struct TxnLockReadGuardExclusive<T> {
    state: ReadGuardState<T>,
    notify: Arc<Notify>,
}

impl<T: Clone> TxnLockReadGuardExclusive<T> {
    /// Downgrade this exclusive read lock to a non-exclusive read lock.
    pub fn downgrade(mut self) -> TxnLockReadGuard<T> {
        let mut state = ReadGuardState::Downgraded;
        std::mem::swap(&mut self.state, &mut state);

        match state {
            ReadGuardState::Active(guard, version) => TxnLockReadGuard {
                lock: version.lock,
                guard: guard.downgrade(),
                notify: self.notify.clone(),
            },
            ReadGuardState::Downgraded => unreachable!("exclusive read downgrade"),
            ReadGuardState::Upgraded => unreachable!("upgrade downgraded exclusive read"),
        }
    }

    /// Upgrade this exclusive read lock to a write lock.
    pub fn upgrade(mut self) -> TxnLockWriteGuard<T> {
        let mut state = ReadGuardState::Upgraded;
        std::mem::swap(&mut self.state, &mut state);

        match state {
            ReadGuardState::Active(guard, version) => TxnLockWriteGuard {
                state: WriteGuardState::Active(guard, version, self.notify.clone()),
            },
            ReadGuardState::Downgraded => unreachable!("exclusive read downgrade"),
            ReadGuardState::Upgraded => unreachable!("upgrade downgraded exclusive read"),
        }
    }
}

impl<T> Deref for TxnLockReadGuardExclusive<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match &self.state {
            ReadGuardState::Active(guard, _) => guard.deref(),
            ReadGuardState::Downgraded => unreachable!("downgraded read lock"),
            ReadGuardState::Upgraded => unreachable!("upgraded read lock"),
        }
    }
}

impl<T> Drop for TxnLockReadGuardExclusive<T> {
    fn drop(&mut self) {
        match &self.state {
            ReadGuardState::Active(_guard, version) => {
                {
                    let mut state = version.state.lock().expect("version state");
                    if &*state == &VersionState::Exclude {
                        *state = VersionState::Read;
                    }
                }

                self.notify.notify_waiters()
            }
            ReadGuardState::Downgraded => self.notify.notify_waiters(),
            ReadGuardState::Upgraded => {}
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for TxnLockReadGuardExclusive<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self.deref(), f)
    }
}

impl<T: fmt::Display> fmt::Display for TxnLockReadGuardExclusive<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self.deref(), f)
    }
}

enum WriteGuardState<T> {
    Active(OwnedRwLockWriteGuard<T>, Version<T>, Arc<Notify>),
    Downgraded,
}

/// An exclusive mutable view of a [`TxnLock`] at a specific `TxnId`
pub struct TxnLockWriteGuard<T> {
    state: WriteGuardState<T>,
}

impl<T> TxnLockWriteGuard<T> {
    /// Downgrade to an exclusive read guard
    pub fn downgrade(mut self) -> TxnLockReadGuardExclusive<T> {
        let mut state = WriteGuardState::Downgraded;
        std::mem::swap(&mut self.state, &mut state);

        match state {
            WriteGuardState::Active(guard, version, notify) => TxnLockReadGuardExclusive {
                state: ReadGuardState::Active(guard, version),
                notify,
            },
            WriteGuardState::Downgraded => unreachable!("write guard downgrade"),
        }
    }
}

impl<T> Deref for TxnLockWriteGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match &self.state {
            WriteGuardState::Active(guard, _, _) => guard.deref(),
            WriteGuardState::Downgraded => unreachable!("downgraded write guard"),
        }
    }
}

impl<T> DerefMut for TxnLockWriteGuard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match &mut self.state {
            WriteGuardState::Active(guard, _, _) => guard.deref_mut(),
            WriteGuardState::Downgraded => unreachable!("downgraded write guard"),
        }
    }
}

impl<T> Drop for TxnLockWriteGuard<T> {
    fn drop(&mut self) {
        match &self.state {
            WriteGuardState::Active(_, _, notify) => notify.notify_waiters(),
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

struct LockState<TxnId, T> {
    name: String,
    versions: BTreeMap<TxnId, Version<T>>,
}

impl<TxnId: fmt::Display + Copy + Ord, T: Clone> LockState<TxnId, T> {
    fn create_version(
        &mut self,
        txn_id: TxnId,
        state: VersionState,
        last_commit: Option<&TxnId>,
    ) -> Result<Option<Version<T>>> {
        let mut versions = self
            .versions
            .iter()
            .rev()
            .skip_while(|(id, _)| *id > &txn_id);

        let canon = loop {
            let (version_id, version) = if let Some(entry) = versions.next() {
                entry
            } else {
                return Err(Error::Outdated);
            };

            if version.is_exclusive() {
                if let Some(last_commit) = last_commit {
                    if last_commit > version_id {
                        #[cfg(feature = "logging")]
                        debug!("the version of {} at {} is dead", self.name, version_id);
                        continue;
                    }
                }

                #[cfg(feature = "logging")]
                debug!(
                    "cannot yet lock {} at {} due to an exclusive lock at {}",
                    self.name, txn_id, version_id
                );

                return Ok(None);
            } else if version.is_final() {
                let value = version.lock.try_read().expect("canonical version");
                break T::clone(&*value);
            }
        };

        #[cfg(feature = "logging")]
        trace!("creating new version of {} at {}...", self.name, txn_id);

        let version = Version::new(state, canon);
        self.versions.insert(txn_id, version.clone());
        Ok(Some(version))
    }
}

/// A futures-aware read-write lock which supports transaction-specific versioning
pub struct TxnLock<TxnId, T> {
    state: Arc<Mutex<LockState<TxnId, T>>>,
    last_commit: Arc<RwLock<Option<TxnId>>>,
    notify: Arc<Notify>,
}

impl<TxnId, T> Clone for TxnLock<TxnId, T> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            last_commit: self.last_commit.clone(),
            notify: self.notify.clone(),
        }
    }
}

impl<TxnId: Ord, T> TxnLock<TxnId, T> {
    /// Create a new transactional lock.
    pub fn new<Name: fmt::Display>(name: Name, txn_id: TxnId, value: T) -> Self {
        let mut versions = BTreeMap::new();
        versions.insert(txn_id, Version::new(VersionState::Write, value));

        Self {
            state: Arc::new(Mutex::new(LockState {
                name: name.to_string(),
                versions,
            })),
            last_commit: Arc::new(RwLock::new(None)),
            notify: Arc::new(Notify::new()),
        }
    }
}

impl<TxnId: fmt::Display + fmt::Debug + Ord + Copy, T: Clone> TxnLock<TxnId, T> {
    fn read_inner(&self, txn_id: TxnId, last_commit: Option<&TxnId>) -> Result<Option<Version<T>>> {
        let mut state = self.state.lock().expect("lock state");

        if let Some(version) = state.versions.get(&txn_id) {
            return Ok(Some(version.clone()));
        }

        state.create_version(txn_id, VersionState::Read, last_commit)
    }

    /// Lock this value for reading at the given `txn_id`.
    pub async fn read(&self, txn_id: TxnId) -> Result<TxnLockReadGuard<T>> {
        loop {
            let version = {
                let last_commit = self.last_commit.read().await;
                self.read_inner(txn_id, last_commit.as_ref())?
            };

            if let Some(version) = version {
                let guard = version.read(self.notify.clone()).await;
                return Ok(guard);
            }

            self.notify.notified().await;
        }
    }

    /// Synchronously Lock this value for reading at the given `txn_id`, if possible.
    pub fn try_read(&self, txn_id: TxnId) -> Result<TxnLockReadGuard<T>> {
        let last_commit = self.last_commit.try_read().map_err(|_| Error::WouldBlock)?;
        if let Some(version) = self.read_inner(txn_id, last_commit.as_ref())? {
            version.try_read(self.notify.clone())
        } else {
            Err(Error::WouldBlock)
        }
    }

    fn lock_exclusive(
        &self,
        txn_id: TxnId,
        version_state: VersionState,
        last_commit: Option<&TxnId>,
    ) -> Result<Option<Version<T>>> {
        debug_assert_ne!(version_state, VersionState::Read);
        debug_assert_ne!(version_state, VersionState::Commit);

        let mut state = self.state.lock().expect("lock state");
        assert!(!state.versions.is_empty());

        let latest = state.versions.keys().last().expect("latest_read");
        if latest > &txn_id {
            #[cfg(feature = "logging")]
            debug!(
                "cannot lock {} exclusively at {} due to a future lock at {}",
                state.name, txn_id, latest
            );

            return Err(Error::Conflict);
        }

        // as long as there are no versions ahead of this one, it's no problem to lock

        if let Some(version) = state.versions.get(&txn_id) {
            return Ok(Some(version.clone()));
        }

        state.create_version(txn_id, version_state, last_commit)
    }

    /// Lock this value for exclusive reading at the given `txn_id`.
    pub async fn read_exclusive(&self, txn_id: TxnId) -> Result<TxnLockReadGuardExclusive<T>> {
        loop {
            let version = {
                let last_commit = self.last_commit.read().await;
                self.lock_exclusive(txn_id, VersionState::Exclude, last_commit.as_ref())?
            };

            if let Some(version) = version {
                return version.read_exclusive(self.notify.clone()).await;
            }

            self.notify.notified().await;
        }
    }

    /// Synchronously lock this value for exclusive reading at the given `txn_id`, if possible.
    pub fn try_read_exclusive(&self, txn_id: TxnId) -> Result<TxnLockReadGuardExclusive<T>> {
        let last_commit = self.last_commit.try_read().map_err(|_| Error::WouldBlock)?;

        let version = self.lock_exclusive(txn_id, VersionState::Exclude, last_commit.as_ref())?;
        if let Some(version) = version {
            version.try_read_exclusive(self.notify.clone())
        } else {
            Err(Error::WouldBlock)
        }
    }

    /// Lock this value for writing at the given `txn_id`.
    pub async fn write(&self, txn_id: TxnId) -> Result<TxnLockWriteGuard<T>> {
        loop {
            let version = {
                let last_commit = self.last_commit.read().await;
                self.lock_exclusive(txn_id, VersionState::Write, last_commit.as_ref())?
            };

            if let Some(version) = version {
                return version.write(self.notify.clone()).await;
            }

            self.notify.notified().await;
        }
    }

    /// Synchronously lock this value for writing at the given `txn_id`, if possible.
    pub fn try_write(&self, txn_id: TxnId) -> Result<TxnLockWriteGuard<T>> {
        let last_commit = self.last_commit.try_read().map_err(|_| Error::WouldBlock)?;

        let version = self.lock_exclusive(txn_id, VersionState::Write, last_commit.as_ref())?;
        if let Some(version) = version {
            version.try_write(self.notify.clone())
        } else {
            Err(Error::WouldBlock)
        }
    }
}

/// A RAII guard which blocks commits until dropped, allowing access to the just-committed value
pub struct TxnLockCommit<TxnId, T> {
    guard: OwnedRwLockReadGuard<T>,
    _last_commit: OwnedRwLockWriteGuard<Option<TxnId>>,
}

impl<TxnId, T> Deref for TxnLockCommit<TxnId, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

/// Guard allowing access to a rolled-back value
pub struct TxnLockRollback<T> {
    guard: OwnedRwLockReadGuard<T>,
}

impl<T> Deref for TxnLockRollback<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

/// Guard allowing access to a finalized value
pub struct TxnLockFinalize<T> {
    guard: OwnedRwLockReadGuard<T>,
}

impl<T> Deref for TxnLockFinalize<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

enum Commit<T> {
    NoOp,
    Pending,
    Version(OwnedRwLockReadGuard<T>),
}

impl<TxnId: fmt::Debug + fmt::Display + Copy + Ord, T: Clone> TxnLock<TxnId, T> {
    fn commit_inner(&self, txn_id: &TxnId, last_commit: Option<&TxnId>) -> Commit<T> {
        let state = self.state.lock().expect("lock state");

        if !state.versions.contains_key(txn_id) {
            return Commit::NoOp;
        }

        for (version_id, version) in state.versions.iter().rev() {
            match &*version.state.lock().expect("version state") {
                VersionState::Exclude | VersionState::Write if version_id < txn_id => {
                    if let Some(last_commit) = last_commit {
                        assert_ne!(version_id, last_commit);

                        if version_id > last_commit {
                            #[cfg(feature = "logging")]
                            debug!(
                                "commit of {} at {} is pending a commit at {}",
                                state.name, txn_id, version_id
                            );

                            return Commit::Pending;
                        } else {
                            // this version is dead, don't let it halt progress
                        }
                    } else {
                        #[cfg(feature = "logging")]
                        debug!(
                            "commit of {} at {} is pending a commit at {}",
                            state.name, txn_id, version_id
                        );

                        return Commit::Pending;
                    }
                }
                VersionState::Commit if version_id < txn_id => break,
                VersionState::Commit if version_id == txn_id => {
                    #[cfg(feature = "logging")]
                    log::warn!("duplicate commit of {} at {}", state.name, txn_id);
                }
                _ => {}
            }
        }

        let version = state.versions.get(txn_id).expect("version to commit");

        if let Some(guard) = version.commit() {
            Commit::Version(guard)
        } else {
            Commit::Pending
        }
    }

    /// Commit the value of this [`TxnLock`] at the given `txn_id`.
    /// This will wait until any earlier write locks have been committed or rolled back.
    ///
    /// **Panics**:
    ///  - when called with a `txn_id` which has already been finalized.
    ///  - when attempting to commit a version at a `txn_id` less than the last committed version
    pub async fn commit(&self, txn_id: TxnId) -> Option<TxnLockCommit<TxnId, T>> {
        let guard = loop {
            {
                let mut last_commit = self.last_commit.clone().write_owned().await;

                match self.commit_inner(&txn_id, last_commit.as_ref()) {
                    Commit::Pending => {}
                    Commit::NoOp => {
                        *last_commit = last_commit
                            .map(|last_commit| Ord::max(txn_id, last_commit))
                            .or_else(|| Some(txn_id));

                        break None;
                    }
                    Commit::Version(guard) => {
                        *last_commit = last_commit
                            .map(|last_commit| Ord::max(txn_id, last_commit))
                            .or_else(|| Some(txn_id));

                        break Some(TxnLockCommit {
                            guard,
                            _last_commit: last_commit,
                        });
                    }
                }
            }

            self.notify.notified().await;
        };

        self.notify.notify_waiters();
        guard
    }

    /// Roll back the value of this [`TxnLock`] at the given `txn_id`.
    ///
    /// Returns the version that was rolled back, if any lock was acquired at `txn_id`.
    ///
    /// **Panics**:
    ///  - if the initial version is rolled back
    ///  - if a committed version is rolled back
    pub async fn rollback(&self, txn_id: &TxnId) -> Option<TxnLockRollback<T>> {
        #[cfg(feature = "logging")]
        trace!("rollback at {}", txn_id);

        let version = {
            let mut state = self.state.lock().expect("lock state");
            let version = state.versions.remove(txn_id);

            assert!(
                !state.versions.is_empty(),
                "the only version of {} was rolled back at {}",
                state.name,
                txn_id
            );

            version
        }?;

        let guard = version.rollback().await;
        self.notify.notify_waiters();
        Some(TxnLockRollback { guard })
    }

    /// Drop all values of this [`TxnLock`] as old as than the given `txn_id` up to the last commit.
    ///
    /// Returns `Some(Arc<T>)` with the finalized version, if any lock was acquired at `txn_id`.
    ///
    /// **Panics**:
    ///  - if the initial version is finalized before being committed
    ///  - if the last commit is finalized
    pub async fn finalize(&self, txn_id: &TxnId) -> Option<TxnLockFinalize<T>> {
        #[cfg(feature = "logging")]
        trace!("finalize {}", txn_id);

        let last_commit = self.last_commit.read().await;

        let mut state = self.state.lock().expect("lock state");
        assert!(!state.versions.is_empty());

        if let Some(last_commit) = last_commit.as_ref() {
            if last_commit < txn_id {
                panic!(
                    "cannot finalize {} at {} since the last commit was at {}",
                    state.name, txn_id, last_commit
                );
            }
        } else if state.versions.len() == 1 {
            let version_id = state.versions.keys().next().expect("latest version ID");
            if version_id > txn_id {
                panic!(
                    "cannot finalize the only version {} of {} at {}",
                    version_id, state.name, txn_id
                );
            }
        }

        let version = state.versions.remove(txn_id);

        while let Some(next) = state.versions.keys().next().copied() {
            if &next < txn_id {
                state.versions.remove(&next);
            } else {
                break;
            }
        }

        assert!(!state.versions.is_empty());

        if let Some(version) = version {
            let guard = version
                .lock
                .try_read_owned()
                .expect("read finalized version");

            Some(TxnLockFinalize { guard })
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
