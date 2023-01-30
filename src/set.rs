//! A futures-aware read-write lock on a [`BTreeSet`] which supports transactional versioning.
//!
//! Example usage:
//! ```
//! use std::sync::Arc;
//! use futures::executor::block_on;
//!
//! use txn_lock::set::*;
//! use txn_lock::Error;
//!
//! let set = TxnSetLock::<u64, String>::new(0);
//!
//! let one = Arc::new("one".to_string());
//! // block_on(set.insert(1, one.clone())).expect("insert");
//! // assert!(block_on(set.contains_key(&one)).expect("contains"));
//! // assert_eq!(set.try_insert(2, one.clone()).unwrap_err(), Error::WouldBlock);
//! // set.commit(1);
//! // assert!(set.try_contains_key(2, &one).expect("contains"));
//! // set.finalize(2);
//! // assert!(set.try_contains_key(3, &one).expect("contains"));
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::{fmt, iter};

use super::semaphore::Semaphore;

pub use super::range::Range;

struct State<I, T> {
    canon: BTreeSet<Arc<T>>,
    committed: BTreeMap<I, Option<BTreeSet<Arc<T>>>>,
    pending: BTreeMap<I, BTreeSet<Arc<T>>>,
    finalized: Option<I>,
}

impl<I: Ord, T: Ord> State<I, T> {
    fn new(txn_id: I, version: BTreeSet<Arc<T>>) -> Self {
        State {
            canon: BTreeSet::new(),
            committed: BTreeMap::new(),
            pending: BTreeMap::from_iter(iter::once((txn_id, version))),
            finalized: None,
        }
    }
}

/// A futures-aware read-write lock on a [`BTreeSet`] which supports transactional versioning.
pub struct TxnSetLock<I, T> {
    state: Arc<Mutex<State<I, T>>>,
    semaphore: Semaphore<I, Range<T>>,
}

impl<I: Copy + Ord, T: Ord + fmt::Debug> TxnSetLock<I, T> {
    /// Construct a new, empty [`TxnSetLock`].
    pub fn new(txn_id: I) -> Self {
        Self {
            state: Arc::new(Mutex::new(State::new(txn_id, BTreeSet::new()))),
            semaphore: Semaphore::new(),
        }
    }

    /// Construct a new [`TxnSetLock`] with the given `contents`.
    pub fn with_contents<C: IntoIterator<Item = T>>(txn_id: I, contents: C) -> Self {
        let version = contents.into_iter().map(Arc::new).collect();

        Self {
            state: Arc::new(Mutex::new(State::new(txn_id, version))),
            semaphore: Semaphore::with_reservation(txn_id, Range::All),
        }
    }
}
