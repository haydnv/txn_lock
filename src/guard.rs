use std::cmp::Ordering;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard};

use super::semaphore::Permit;

/// A permit to read a value in a pending read transaction
#[derive(Debug)]
pub struct PendingRead<R, T> {
    #[allow(unused)]
    permit: Permit<R>,
    value: Arc<T>,
}

impl<R, T> Deref for PendingRead<R, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value.deref()
    }
}

/// A permit to read a value in a pending write transaction
#[derive(Debug)]
pub struct PendingWrite<R, T> {
    #[allow(unused)]
    permit: Permit<R>,
    value: OwnedRwLockReadGuard<T>,
}

impl<R, T> Deref for PendingWrite<R, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value.deref()
    }
}

/// A read guard on a transactional value
#[derive(Debug)]
pub enum TxnReadGuard<R, T> {
    Committed(Arc<T>),
    PendingRead(PendingRead<R, T>),
    PendingWrite(PendingWrite<R, T>),
}

impl<R, T> TxnReadGuard<R, T> {
    /// Construct a new [`TxnReadGuard`] for a committed transaction.
    pub fn committed(value: Arc<T>) -> Self {
        Self::Committed(value)
    }

    /// Construct a new [`TxnReadGuard`] for a value read as part of a pending transaction.
    pub fn pending_read(permit: Permit<R>, value: Arc<T>) -> Self {
        Self::PendingRead(PendingRead { permit, value })
    }

    /// Construct a new [`TxnReadGuard`] for a value mutated as part of a pending transaction.
    pub fn pending_write(permit: Permit<R>, value: OwnedRwLockReadGuard<T>) -> Self {
        Self::PendingWrite(PendingWrite { permit, value })
    }
}

impl<R, T> Deref for TxnReadGuard<R, T> {
    type Target = T;

    fn deref(&self) -> &T {
        match self {
            Self::Committed(value) => value.deref(),
            Self::PendingRead(permit) => permit.deref(),
            Self::PendingWrite(permit) => permit.deref(),
        }
    }
}

impl<R, T: PartialEq> PartialEq<T> for TxnReadGuard<R, T> {
    fn eq(&self, other: &T) -> bool {
        self.deref().eq(other)
    }
}

impl<R, T: PartialOrd> PartialOrd<T> for TxnReadGuard<R, T> {
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        self.deref().partial_cmp(other)
    }
}

/// A write guard on a transactional value
#[derive(Debug)]
pub struct TxnWriteGuard<R, T> {
    #[allow(unused)]
    permit: Permit<R>,
    value: OwnedRwLockWriteGuard<T>,
}

impl<R, T> TxnWriteGuard<R, T> {
    /// Construct a guard for a mutable value as part of a pending transaction.
    pub fn new(permit: Permit<R>, value: OwnedRwLockWriteGuard<T>) -> Self {
        Self { permit, value }
    }
}

impl<R, T> Deref for TxnWriteGuard<R, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value.deref()
    }
}

impl<R, T> DerefMut for TxnWriteGuard<R, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<R, T: PartialEq> PartialEq<T> for TxnWriteGuard<R, T> {
    fn eq(&self, other: &T) -> bool {
        self.deref().eq(other)
    }
}

impl<R, T: PartialOrd> PartialOrd<T> for TxnWriteGuard<R, T> {
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        self.deref().partial_cmp(other)
    }
}