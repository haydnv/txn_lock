//! A futures-aware read-write lock on a [`BTreeMap`] which supports transactional versioning.
//!
//! Example usage:
//! ```
//! use txn_lock::map::*;
//! let map = TxnMapLock::<u64, String, f32>::new();
//! map.insert(1, "one".to_string(), 1.0);
//! ```

use std::collections::btree_map::{BTreeMap, Entry};
use std::iter;
use std::sync::{Arc, Mutex};
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use super::semaphore::*;
use super::{Error, Result};

#[derive(Eq, PartialEq, Ord, PartialOrd)]
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

impl<K: Eq> Overlap<Self> for Range<K> {
    fn overlaps(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::One(l), Self::One(r)) => l == r,
            (Self::All, _) => true,
            (_, Self::All) => true,
        }
    }
}

impl<K: Eq> Overlap<K> for Range<K> {
    fn overlaps(&self, other: &K) -> bool {
        match self {
            Self::One(this) => &**this == other,
            Self::All => true,
        }
    }
}

enum Value<V> {
    One(RwLock<V>),
    All,
}

enum Delta<V> {
    Read,
    Write(Value<V>),
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
    deltas: BTreeMap<I, BTreeMap<Range<K>, Delta<V>>>,
    canon: BTreeMap<Arc<K>, Arc<V>>,
    finalized: Option<I>,
}

/// A futures-aware read-write lock on a [`BTreeMap`] which supports transactional versioning
pub struct TxnMapLock<I, K, V> {
    state: Arc<Mutex<State<I, K, V>>>,
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
            state: Arc::new(Mutex::new(State {
                deltas: BTreeMap::new(),
                canon: BTreeMap::new(),
                finalized: None,
            })),
            semaphore: Semaphore::new(),
        }
    }

    /// Construct a new [`TxnMapLock`] with the given `contents`.
    pub fn with_contents<C: IntoIterator<Item = (K, V)>>(txn_id: I, contents: C) -> Self {
        let deltas = contents
            .into_iter()
            .map(|(k, v)| {
                let range = Range::One(Arc::new(k));
                let delta = Delta::Write(Value::One(RwLock::new(v)));
                (range, delta)
            })
            .collect::<BTreeMap<Range<K>, Delta<V>>>();

        let reserved = deltas
            .keys()
            .cloned()
            .chain(iter::once(Range::All))
            .collect::<Vec<Range<K>>>();

        let state = State {
            deltas: BTreeMap::from_iter(iter::once((txn_id, deltas))),
            canon: BTreeMap::new(),
            finalized: None,
        };

        Self {
            state: Arc::new(Mutex::new(state)),
            semaphore: Semaphore::with_reservations(txn_id, reserved),
        }
    }

    fn insert_inner(&self, txn_id: I, range: Range<K>, delta: Delta<V>) -> Result<()> {
        let mut state = self.state.lock().expect("lock state");

        if state.finalized >= Some(txn_id) {
            return Err(Error::Outdated);
        }

        fn update_deltas<K: Ord, V>(
            deltas: &mut BTreeMap<Range<K>, Delta<V>>,
            range: Range<K>,
            delta: Delta<V>,
        ) {
            match deltas.entry(range) {
                Entry::Occupied(mut entry) => {
                    let existing_delta = entry.get_mut();
                    *existing_delta = delta;
                }
                Entry::Vacant(entry) => {
                    entry.insert(delta);
                }
            }
        }

        match state.deltas.entry(txn_id) {
            Entry::Occupied(mut entry) => update_deltas(entry.get_mut(), range, delta),
            Entry::Vacant(entry) => {
                let deltas = entry.insert(BTreeMap::new());
                update_deltas(deltas, range, delta)
            }
        };

        Ok(())
    }

    /// Insert a new entry into this [`TxnMapLock`].
    pub async fn insert(&self, txn_id: I, key: K, value: V) -> Result<()> {
        let key = Arc::new(key);
        let range = Range::One(key);
        let _permit = self.semaphore.write(txn_id, range.clone()).await?;

        let delta = Delta::Write(Value::One(RwLock::new(value)));
        self.insert_inner(txn_id, range, delta)
    }
}
