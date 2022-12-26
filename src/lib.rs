//! A [`TxnLock`] to support transaction-specific versioning

use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, RwLockWriteGuard};

use tokio::sync::{RwLock, RwLockReadGuard};

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum ErrorKind {
    Conflict,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Conflict => f.write_str("Conflict"),
        }
    }
}

pub struct Error {
    kind: ErrorKind,
    message: String,
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

pub struct TxnLockReadGuard<'a, T> {
    guard: RwLockReadGuard<'a, T>,
}

impl<'a, T> Deref for TxnLockReadGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

pub struct TxnLockReadGuardExclusive<'a, TxnId, T> {
    lock: &'a TxnLock<TxnId, T>,
    guard: RwLockWriteGuard<'a, T>,
}

impl<'a, TxnId, T> TxnLockReadGuardExclusive<'a, TxnId, T> {
    /// Upgrade this exclusive read lock to a write lock.
    pub fn upgrade(self) -> TxnLockWriteGuard<'a, T> {
        todo!()
    }
}

impl<'a, TxnId, T> Deref for TxnLockReadGuardExclusive<'a, TxnId, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

pub struct TxnLockWriteGuard<'a, T> {
    guard: RwLockWriteGuard<'a, T>,
}

impl<'a, T> Deref for TxnLockWriteGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

impl<'a, T> DerefMut for TxnLockWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard.deref_mut()
    }
}

enum Version<T> {
    Committed(T),
    Pending(RwLock<T>),
}

struct LockState<TxnId, T> {
    last_commit: TxnId,
    versions: BTreeMap<TxnId, Version<T>>,
}

#[derive(Clone)]
pub struct TxnLock<TxnId, T> {
    state: Arc<Mutex<LockState<TxnId, T>>>,
}

impl<TxnId: Copy + Ord, T> TxnLock<TxnId, T> {
    /// Create a new transactional lock.
    pub fn new(last_commit: TxnId, value: T) -> Self {
        let mut versions = BTreeMap::new();
        versions.insert(last_commit, Version::Committed(value));

        Self {
            state: Arc::new(Mutex::new(LockState {
                last_commit,
                versions,
            })),
        }
    }
}

impl<TxnId, T> TxnLock<TxnId, T> {
    /// Lock this value for reading at the given `txn_id`.
    pub async fn read(&self, _txn_id: &TxnId) -> Result<TxnLockReadGuard<T>> {
        todo!()
    }

    /// Lock this value for exclusive reading at the given `txn_id`.
    pub async fn read_exclusive(
        &self,
        _txn_id: &TxnId,
    ) -> Result<TxnLockReadGuardExclusive<TxnId, T>> {
        todo!()
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
