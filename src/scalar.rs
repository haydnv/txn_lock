//! A transactional lock on a scalar value.
//!
//! Example:
//! ```
//! use futures::executor::block_on;
//! use txn_lock::scalar::*;
//! use txn_lock::Error;
//!
//! let lock = TxnLock::new(0, "zero");
//!
//! // assert_eq!(*lock.try_read(0).expect("read"), "zero");
//! // assert_eq!(lock.try_write(1).unwrap_err(), Error::WouldBlock);
//!
//! {
//!     // let commit = block_on(lock.commit(0)).expect("commit guard");
//!     // assert_eq!(*commit, "zero");
//!     // this commit guard will block future commits until dropped
//! }
//!
//! {
//!     // let mut guard = lock.try_write(1).expect("write lock");
//!     // *guard = "one";
//! }
//!
//! // assert_eq!(*lock.try_read(0).expect("read past version"), "zero");
//! // assert_eq!(*lock.try_read(1).expect("read current version"), "one");
//!
//! // block_on(lock.commit(1));
//!
//! // assert_eq!(*lock.try_read_exclusive(2).expect("new value"), "one");
//!
//! // lock.rollback(&2);
//!
//! {
//!     // let mut guard = lock.try_write(3).expect("write lock");
//!     // *guard = "three";
//! }
//!
//! // assert_eq!(*block_on(lock.finalize(&1)).expect("finalized version"), "one");
//!
//! // assert_eq!(lock.try_read(0).unwrap_err(), Error::Outdated);
//! // assert_eq!(*lock.try_read(3).expect("current value"), "three");
//! ```

use core::fmt;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::iter;
use std::sync::{Arc, Mutex};

use super::semaphore::{Overlap, Overlaps, Semaphore};

#[derive(Debug)]
struct Range<T>(T);

impl<T> From<T> for Range<T> {
    fn from(scalar: T) -> Self {
        Self(scalar)
    }
}

impl<T: Ord> Overlaps<Range<T>> for Range<T> {
    fn overlaps(&self, other: &Range<T>) -> Overlap {
        match self.0.cmp(&other.0) {
            Ordering::Less => Overlap::Less,
            Ordering::Equal => Overlap::Equal,
            Ordering::Greater => Overlap::Greater,
        }
    }
}

struct State<I, T> {
    canon: Option<Arc<T>>,
    committed: BTreeMap<I, Option<Arc<T>>>,
    pending: BTreeMap<I, Arc<T>>,
    finalized: Option<I>,
}

impl<I: Ord, T: Ord> State<I, T> {
    fn new(txn_id: I, version: Arc<T>) -> Self {
        State {
            canon: None,
            committed: BTreeMap::new(),
            pending: BTreeMap::from_iter(iter::once((txn_id, version))),
            finalized: None,
        }
    }
}

/// A futures-aware read-write lock on a scalar value which supports transactional versioning.
pub struct TxnLock<I, T> {
    state: Arc<Mutex<State<I, T>>>,
    semaphore: Semaphore<I, Range<T>>,
}

impl<I, T> Clone for TxnLock<I, T> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            semaphore: self.semaphore.clone(),
        }
    }
}

impl<I: Ord, T: Ord + fmt::Debug> TxnLock<I, T> {
    /// Construct a new [`TxnLock`].
    pub fn new(txn_id: I, initial_value: T) -> Self {
        Self {
            state: Arc::new(Mutex::new(State::new(txn_id, Arc::new(initial_value)))),
            semaphore: Semaphore::new(),
        }
    }
}
