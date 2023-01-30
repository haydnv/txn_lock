//! Utilities to support transactional versioning.
//!
//! General-purpose locks and usage examples are provided
//! in the [`map`], [`scalar`], and [`set`] modules.
//!
//! More complex transaction locks (e.g. for a relational database) can be constructed using
//! the [`semaphore`] module.

pub mod lock;
pub mod map;
pub mod scalar;
pub mod semaphore;
pub mod set;

mod guard;
mod range;

use std::fmt;

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
            Self::Conflict => "there is a conflicting transactional lock on this resource",
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
    fn from(_: tokio::sync::AcquireError) -> Self {
        Self::Outdated
    }
}

impl From<tokio::sync::TryAcquireError> for Error {
    fn from(cause: tokio::sync::TryAcquireError) -> Self {
        match cause {
            tokio::sync::TryAcquireError::Closed => Error::Outdated,
            tokio::sync::TryAcquireError::NoPermits => Error::WouldBlock,
        }
    }
}

type Result<T> = std::result::Result<T, Error>;
