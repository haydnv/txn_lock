//! A [`TxnLock`] to support transaction-specific versioning

use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

use tokio::sync::{Notify, OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum ErrorKind {
    Conflict,
    Outdated,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Conflict => f.write_str("Conflict"),
            Self::Outdated => f.write_str("Outdated"),
        }
    }
}

pub struct Error {
    kind: ErrorKind,
    message: String,
}

impl Error {
    fn new<M: fmt::Display>(kind: ErrorKind, message: M) -> Self {
        Self {
            kind,
            message: message.to_string(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
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

enum ExclusiveReadGuardState<TxnId, T> {
    Canon(TxnLock<TxnId, T>, TxnId, Arc<T>),
    Pending(Arc<Notify>, OwnedRwLockWriteGuard<T>),
    Upgraded,
}

pub struct TxnLockReadGuardExclusive<TxnId, T> {
    state: ExclusiveReadGuardState<TxnId, T>,
}

impl<TxnId: Copy + Ord, T: Clone> TxnLockReadGuardExclusive<TxnId, T> {
    /// Upgrade this exclusive read lock to a write lock.
    pub fn upgrade(mut self) -> TxnLockWriteGuard<T> {
        let mut state = ExclusiveReadGuardState::Upgraded;
        std::mem::swap(&mut self.state, &mut state);

        let guard = match state {
            ExclusiveReadGuardState::Pending(_notify, guard) => guard,
            ExclusiveReadGuardState::Canon(txn_lock, txn_id, value) => {
                let lock = txn_lock.create_value(txn_id, T::clone(&*value));
                lock.try_write_owned().expect("write guard")
            }
            ExclusiveReadGuardState::Upgraded => unreachable!(),
        };

        TxnLockWriteGuard { guard }
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

pub struct TxnLockWriteGuard<T> {
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
    versions: BTreeMap<TxnId, Version<T>>,
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

impl<TxnId: Copy + Ord, T> TxnLock<TxnId, T> {
    /// Create a new transactional lock.
    pub fn new(last_commit: TxnId, value: T) -> Self {
        let mut versions = BTreeMap::new();
        versions.insert(last_commit, Version::Committed(Arc::new(value)));

        Self {
            notify: Arc::new(Notify::new()),
            state: Arc::new(Mutex::new(LockState { versions })),
        }
    }
}

impl<TxnId: Ord, T> TxnLock<TxnId, T> {
    fn create_value(&self, txn_id: TxnId, value: T) -> Arc<RwLock<T>> {
        let mut state = self.state.lock().expect("lock state");
        assert!(!state.versions.contains_key(&txn_id));

        let lock = Arc::new(RwLock::new(value));
        let version = Version::Pending(lock.clone());
        state.versions.insert(txn_id, version);
        lock
    }
}

impl<TxnId: fmt::Display + fmt::Debug + Copy + Ord, T> TxnLock<TxnId, T> {
    fn read_inner(&self, txn_id: &TxnId) -> Result<Option<(TxnId, Version<T>)>> {
        let state = self.state.lock().expect("lock state");
        if let Some(version) = state.versions.get(txn_id) {
            return Ok(Some((*txn_id, version.clone())));
        }

        let mut version = None;
        for (candidate_id, candidate) in &state.versions {
            if candidate_id < txn_id {
                if candidate.is_pending() {
                    return Ok(None);
                } else {
                    version = Some((candidate_id, candidate))
                }
            } else if let Some((version_id, version)) = version {
                return Ok(Some((*version_id, version.clone())));
            }
        }

        Err(Error::new(
            ErrorKind::Outdated,
            format!("value has already been finalized at {}", txn_id),
        ))
    }

    /// Lock this value for reading at the given `txn_id`.
    pub async fn read(&self, txn_id: &TxnId) -> Result<TxnLockReadGuard<T>> {
        loop {
            if let Some((_version_id, version)) = self.read_inner(txn_id)? {
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

    /// Lock this value for exclusive reading at the given `txn_id`.
    pub async fn read_exclusive(
        &self,
        txn_id: &TxnId,
    ) -> Result<TxnLockReadGuardExclusive<TxnId, T>> {
        loop {
            if let Some((version_id, version)) = self.read_inner(txn_id)? {
                return match version {
                    Version::Committed(_) if &version_id == txn_id => Err(Error::new(ErrorKind::Conflict, "cannot acquire an exclusive lock on a transactional version after committing")),
                    Version::Committed(value) => {
                        assert!(&version_id < txn_id);
                        let state = ExclusiveReadGuardState::Canon(self.clone(), *txn_id, value);
                        Ok(TxnLockReadGuardExclusive { state })
                    }
                    Version::Pending(lock) => {
                        assert_eq!(&version_id, txn_id);
                        let guard = lock.write_owned().await;
                        let state = ExclusiveReadGuardState::Pending(self.notify.clone(), guard);
                        Ok(TxnLockReadGuardExclusive { state })
                    }
                };
            }

            self.notify.notified().await;
        }
    }

    /// Lock this value for exclusive reading at the given `txn_id`.
    pub async fn write(&self, _txn_id: &TxnId) -> Result<TxnLockWriteGuard<T>> {
        todo!()
    }
}

impl<TxnId, T> TxnLock<TxnId, T> {
    /// Commit the value of this [`TxnLock`] at the given `txn_id`.
    pub fn commit(_txn_id: &TxnId) {
        todo!()
    }

    /// Roll back the value of this [`TxnLock`] at the given `txn_id`.
    pub fn rollback(_txn_id: &TxnId) {
        todo!()
    }

    /// Drop all values of this [`TxnLock`] older than the given `txn_id`.
    pub fn finalize(_txn_id: &TxnId) {
        todo!()
    }
}
