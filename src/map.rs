//! A futures-aware read-write lock on a [`BTreeMap`] which supports transactional versioning.
//!
//! Example usage:
//! ```
//! use txn_lock::map::*;
//! let map = TxnMapLock::<u64, String, f32>::new();
//! map.insert(1, "one".to_string(), 1.0);
//! ```

use std::cmp::Ordering;
use std::collections::btree_map::{BTreeMap, Entry};
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};
use std::{fmt, iter};
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use super::semaphore::*;
use super::{Error, Result};

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd)]
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

impl<K: PartialEq> PartialEq<K> for Range<K> {
    fn eq(&self, other: &K) -> bool {
        match self {
            Self::One(key) => &**key == other,
            Self::All => false,
        }
    }
}

impl<K: Eq + Ord> Overlaps<Self> for Range<K> {
    fn overlaps(&self, other: &Self) -> Overlap {
        match self {
            Self::All => match other {
                Self::All => Overlap::Equal,
                _ => Overlap::Wide,
            },
            this => match other {
                Self::All => Overlap::Narrow,
                Self::One(that) => this.overlaps(&**that),
            },
        }
    }
}

impl<K: Eq + Ord> Overlaps<K> for Range<K> {
    fn overlaps(&self, other: &K) -> Overlap {
        match self {
            Self::All => Overlap::Wide,
            Self::One(this) => match (&**this).cmp(other) {
                Ordering::Less => Overlap::Less,
                Ordering::Equal => Overlap::Equal,
                Ordering::Greater => Overlap::Greater,
            },
        }
    }
}

enum Value<V> {
    One(Arc<RwLock<V>>),
    All,
}

pub enum ValueRead<V> {
    Committed(Arc<V>),
    Pending(OwnedRwLockReadGuard<V>),
}

impl<V> Deref for ValueRead<V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Committed(value) => value.deref(),
            Self::Pending(version) => version.deref(),
        }
    }
}

/// A read guard on a value in a [`TxnMapLock`]
pub struct ValueReadGuard<K, V> {
    _permit: Permit<Range<K>>,
    value: ValueRead<V>,
}

impl<K, V> Deref for ValueReadGuard<K, V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        self.value.deref()
    }
}

/// A write lock on a value in a [`TxnMapLock`]
pub struct ValueWrite<K, V> {
    _permit: Permit<Range<K>>,
    value: OwnedRwLockWriteGuard<V>,
}

impl<K, V> Deref for ValueWrite<K, V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        self.value.deref()
    }
}

impl<K, V> DerefMut for ValueWrite<K, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value.deref_mut()
    }
}

struct State<I, K, V> {
    canon: BTreeMap<Arc<K>, Arc<V>>,
    committed: BTreeMap<I, BTreeMap<Arc<K>, Arc<V>>>,
    pending: BTreeMap<I, BTreeMap<Range<K>, Value<V>>>,
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

impl<I: Ord + Copy, K: Ord + fmt::Debug, V> TxnMapLock<I, K, V> {
    /// Construct a new [`TxnMapLock`].
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                canon: BTreeMap::new(),
                committed: BTreeMap::new(),
                pending: BTreeMap::new(),
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
                let delta = Value::One(Arc::new(RwLock::new(v)));
                (range, delta)
            })
            .collect::<BTreeMap<Range<K>, Value<V>>>();

        let reserved = deltas
            .keys()
            .cloned()
            .chain(iter::once(Range::All))
            .collect::<Vec<Range<K>>>();

        let state = State {
            canon: BTreeMap::new(),
            committed: BTreeMap::new(),
            pending: BTreeMap::from_iter(iter::once((txn_id, deltas))),
            finalized: None,
        };

        Self {
            state: Arc::new(Mutex::new(state)),
            semaphore: Semaphore::with_reservations(txn_id, reserved),
        }
    }

    /// Read a value from this [`TxnMapLock`] at the given `txn_id`.
    pub async fn get(&self, txn_id: I, key: &K) -> Result<Option<ValueReadGuard<K, V>>> {
        let range = {
            if let Some(range) = self.semaphore.maybe_range::<K>(txn_id, key).await {
                Range::<K>::clone(&*range)
            } else {
                let state = self.state.lock().expect("lock state");
                if let Some((key, _)) = state.canon.get_key_value(key) {
                    Range::One(key.clone())
                } else {
                    return Ok(None);
                }
            }
        };

        let _permit = self.semaphore.read(txn_id, range.clone()).await?;

        let state = self.state.lock().expect("lock state");
        if state.finalized >= Some(txn_id) {
            // in this case a new permit has been created at a finalized txn_id
            // it will have to wait until the next call to finalize to be cleaned up
            return Err(Error::Outdated);
        }

        if let Some(version) = state.pending.get(&txn_id) {
            if let Some(delta) = version.get(&range) {
                match delta {
                    Value::All => {}
                    Value::One(lock) => {
                        // calling try_read is safe because the semaphore takes care of blocking
                        let guard = lock.clone().try_read_owned().expect("value read guard");

                        return Ok(Some(ValueReadGuard {
                            _permit,
                            value: ValueRead::Pending(guard),
                        }));
                    }
                }
            }
        }

        for version in state.committed.values().rev() {
            if let Some(value) = version.get(key) {
                return Ok(Some(ValueReadGuard {
                    _permit,
                    value: ValueRead::Committed(value.clone()),
                }));
            }
        }

        if let Some(value) = state.canon.get(key) {
            return Ok(Some(ValueReadGuard {
                _permit,
                value: ValueRead::Committed(value.clone()),
            }));
        }

        unreachable!("found a range for a nonexistent key")
    }

    /// Insert a new entry into this [`TxnMapLock`] at the given `txn_id`.
    pub async fn insert(&self, txn_id: I, key: K, value: V) -> Result<()> {
        let key = Arc::new(key);
        let range = Range::One(key);
        let _permit = self.semaphore.write(txn_id, range.clone()).await?;

        let value = Value::One(Arc::new(RwLock::new(value)));

        let mut state = self.state.lock().expect("lock state");

        if state.finalized >= Some(txn_id) {
            return Err(Error::Outdated);
        }

        fn update_deltas<K: Ord, V>(
            deltas: &mut BTreeMap<Range<K>, Value<V>>,
            range: Range<K>,
            value: Value<V>,
        ) {
            match deltas.entry(range) {
                Entry::Occupied(mut entry) => {
                    let existing_delta = entry.get_mut();
                    *existing_delta = value;
                }
                Entry::Vacant(entry) => {
                    entry.insert(value);
                }
            }
        }

        match state.pending.entry(txn_id) {
            Entry::Occupied(mut entry) => update_deltas(entry.get_mut(), range, value),
            Entry::Vacant(entry) => {
                let deltas = entry.insert(BTreeMap::new());
                update_deltas(deltas, range, value)
            }
        };

        Ok(())
    }
}
