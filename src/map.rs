//! A futures-aware read-write lock on a [`BTreeMap`] which supports transactional versioning.
//!
//! Example usage:
//! ```
//! use txn_lock::map::*;
//! let map = TxnMapLock::<u64, String, f32>::new();
//! ```

use std::collections::BTreeMap;
use std::iter;
use std::sync::{Arc, Mutex};
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use super::semaphore::*;

enum Range<K> {
    One(Arc<K>),
    All,
}

impl<K> Clone for Range<K> {
    fn clone(&self) -> Self {
        match self {
            Self::One(k) => Self::One(k.clone()),
            Self::All => Self::All,
        }
    }
}

impl<K: Eq> Overlap for Range<K> {
    fn overlaps(&self, other: &Self) -> bool {
        match (self, other) {
            (Range::One(l), Range::One(r)) => l == r,
            (Range::All, _) => true,
            (_, Range::All) => true,
        }
    }
}

enum Value<V> {
    One(RwLock<V>),
    All,
}

enum Delta<K, V> {
    Read(Range<K>),
    Write(Range<K>, Value<V>),
}

enum ValueRead<V> {
    Canon(Arc<V>),
    Version(OwnedRwLockReadGuard<V>),
}

/// A read guard on a value in a [`TxnMapLock`]
pub struct ValueReadGuard<K, V> {
    _permit: Permit<Range<K>>,
    value: ValueRead<V>,
}

/// A write lock on a value in a [`TxnMapLock`]
pub struct ValueWrite<K, V> {
    _permit: Permit<Range<K>>,
    value: OwnedRwLockWriteGuard<V>,
}

struct State<I, K, V> {
    deltas: BTreeMap<I, Vec<Delta<K, V>>>,
    history: Mutex<BTreeMap<I, BTreeMap<Arc<K>, Arc<V>>>>,
}

/// A futures-aware read-write lock on a [`BTreeMap`] which supports transactional versioning
pub struct TxnMapLock<I, K, V> {
    state: Arc<State<I, K, V>>,
    semaphore: Semaphore<I, Range<K>>,
}

impl<I, K, V> Clone for TxnMapLock<I, K, V> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            semaphore: self.semaphore.clone(),
        }
    }
}

impl<I: Ord + Copy, K: Ord, V> TxnMapLock<I, K, V> {
    /// Construct a new [`TxnMapLock`].
    pub fn new() -> Self {
        Self {
            state: Arc::new(State {
                deltas: BTreeMap::new(),
                history: Mutex::new(BTreeMap::new()),
            }),
            semaphore: Semaphore::new(),
        }
    }

    /// Construct a new [`TxnMapLock`] with the given `contents`.
    pub fn with_contents<C: IntoIterator<Item = (K, V)>>(txn_id: I, contents: C) -> Self {
        let mut reserved = Vec::new();
        let mut deltas = Vec::new();

        for (k, v) in contents {
            let range = Range::One(Arc::new(k));
            reserved.push(range.clone());
            deltas.push(Delta::Write(range, Value::One(RwLock::new(v))));
        }

        let state = State {
            deltas: BTreeMap::from_iter(iter::once((txn_id, deltas))),
            history: Mutex::new(BTreeMap::new()),
        };

        Self {
            state: Arc::new(state),
            semaphore: Semaphore::with_reservations(txn_id, reserved),
        }
    }
}
