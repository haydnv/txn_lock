//! Utilities to support transactional versioning.
//!
//! Example:
//! ```
//! use futures::executor::block_on;
//! use txn_lock::lock::*;
//! use txn_lock::Error;
//!
//! let lock = TxnLock::new("example", 0, "zero");
//!
//! assert_eq!(*lock.try_read(0).expect("read"), "zero");
//! assert_eq!(lock.try_write(1).unwrap_err(), Error::WouldBlock);
//!
//! {
//!     let commit = block_on(lock.commit(0)).expect("commit guard");
//!     assert_eq!(*commit, "zero");
//!     // this commit guard will block future commits until dropped
//! }
//!
//! {
//!     let mut guard = lock.try_write(1).expect("write lock");
//!     *guard = "one";
//! }
//!
//! assert_eq!(*lock.try_read(0).expect("read past version"), "zero");
//! assert_eq!(*lock.try_read(1).expect("read current version"), "one");
//!
//! block_on(lock.commit(1));
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
//! assert_eq!(*block_on(lock.finalize(&1)).expect("finalized version"), "one");
//!
//! assert_eq!(lock.try_read(0).unwrap_err(), Error::Outdated);
//! assert_eq!(*lock.try_read(3).expect("current value"), "three");
//! ```

pub mod lock;
pub mod semaphore;

use std::fmt;
use tokio::sync::AcquireError;

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
            Self::WouldBlock => "synchronous lock acquisition failed",
        })
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for Error {}

impl From<tokio::sync::AcquireError> for Error {
    fn from(_: AcquireError) -> Self {
        Self::Outdated
    }
}

type Result<T> = std::result::Result<T, Error>;
