//! A futures-aware read-write lock on a [`BTreeMap`] which supports transactional versioning.
//!
//! Example usage:
//! ```
//! use std::sync::Arc;
//! use futures::executor::block_on;
//!
//! use txn_lock::map::*;
//! use txn_lock::Error;
//!
//! let map = TxnMapLock::<u64, String, f32>::new();
//!
//! let one = Arc::new("one".to_string());
//! block_on(map.insert(1, one.clone(), 1.0)).expect("insert");
//!
//! let value = block_on(map.get(1, one.clone())).expect("read");
//! assert_eq!(*(value.expect("guard")), 1.0);
//!
//! map.commit(1);
//!
//! let value = block_on(map.get(2, one.clone())).expect("read");
//! assert_eq!(*(value.expect("guard")), 1.0);
//!
//! map.finalize(2);
//!
//! let value = block_on(map.get(3, one.clone())).expect("read");
//! assert_eq!(*(value.expect("guard")), 1.0);
//! ```

use std::cmp::Ordering;
use std::collections::btree_map::{BTreeMap, Entry};
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};
use std::{fmt, iter};
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use super::semaphore::*;
use super::{Error, Result};

/// A range used to reserve [`Semaphore`] permits in a [`TxnMapLock`]
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Range<K> {
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

/// A read guard on a value in a [`TxnMapLock`]
#[derive(Debug)]
pub enum ValueReadGuard<K, V> {
    Committed(Arc<V>),
    Pending(Permit<Range<K>>, OwnedRwLockReadGuard<V>),
}

impl<K, V> Deref for ValueReadGuard<K, V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Committed(value) => value.deref(),
            Self::Pending(_permit, value) => value.deref(),
        }
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

type Version<K, V> = BTreeMap<Arc<K>, Arc<RwLock<V>>>;

struct State<I, K, V> {
    canon: BTreeMap<Arc<K>, Arc<V>>,
    committed: BTreeMap<I, BTreeMap<Arc<K>, Arc<V>>>,
    pending: BTreeMap<I, Version<K, V>>,
    finalized: Option<I>,
}

impl<I: Copy, K, V> State<I, K, V> {
    fn last_commit(&self) -> Option<I> {
        self.committed.keys().last().copied()
    }
}

impl<I: Ord, K: Ord, V> State<I, K, V> {
    fn get_canon(&self, txn_id: &I, key: &K) -> Option<Arc<V>> {
        let committed = self
            .committed
            .iter()
            .rev()
            .skip_while(|(id, _)| *id > txn_id)
            .map(|(_, version)| version);

        for version in committed {
            if let Some(value) = version.get(key) {
                return Some(value.clone());
            }
        }

        self.canon.get(key).cloned()
    }
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

impl<I: Ord + Copy + fmt::Display, K: Ord + fmt::Debug, V: fmt::Debug> TxnMapLock<I, K, V> {
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
        let version = contents
            .into_iter()
            .map(|(key, value)| (Arc::new(key), Arc::new(RwLock::new(value))))
            .collect();

        let state = State {
            canon: BTreeMap::new(),
            committed: BTreeMap::new(),
            pending: BTreeMap::from_iter(iter::once((txn_id, version))),
            finalized: None,
        };

        Self {
            state: Arc::new(Mutex::new(state)),
            semaphore: Semaphore::with_reservation(txn_id, Range::All),
        }
    }

    /// Commit the state of this [`TxnMapLock`] at the given `txn_id`.
    pub fn commit(&self, txn_id: I) {
        let mut state = self.state.lock().expect("lock state");

        self.semaphore.finalize(&txn_id, false);

        let version = if let Some(pending) = state.pending.remove(&txn_id) {
            pending
                .into_iter()
                .map(|(key, value)| {
                    let value = Arc::try_unwrap(value).expect("value");
                    let value = value.into_inner();
                    (key, Arc::new(value))
                })
                .collect()
        } else {
            BTreeMap::new()
        };

        match state.committed.entry(txn_id) {
            Entry::Occupied(_) => {
                assert!(version.is_empty());
                #[cfg(feature = "logging")]
                log::warn!("duplicate commit at {}", txn_id);
            }
            Entry::Vacant(entry) => {
                entry.insert(version);
            }
        }
    }

    /// Finalize the state of this [`TxnMapLock`] at the given `txn_id`.
    /// This will merge in deltas and prevent further reads of versions earlier than `txn_id`.
    pub fn finalize(&self, txn_id: I) {
        let mut state = self.state.lock().expect("lock state");

        while let Some(version_id) = state.committed.keys().next().copied() {
            if version_id <= txn_id {
                let version = state.committed.remove(&version_id).expect("version");
                for (key, value) in version {
                    state.canon.insert(key, value);
                }
            } else {
                break;
            }
        }

        self.semaphore.finalize(&txn_id, true);
        state.finalized = Some(txn_id);
    }

    /// Read a value from this [`TxnMapLock`] at the given `txn_id`.
    pub async fn get(&self, txn_id: I, key: Arc<K>) -> Result<Option<ValueReadGuard<K, V>>> {
        {
            // before acquiring a permit, check if this version has already been committed
            let state = self.state.lock().expect("lock state");
            if state.finalized.as_ref() > Some(&txn_id) {
                return Err(Error::Outdated);
            } else if state.committed.contains_key(&txn_id) {
                assert!(!state.pending.contains_key(&txn_id));
                let canon = state.get_canon(&txn_id, &key);
                return Ok(canon.map(ValueReadGuard::Committed));
            }
        }

        let permit = self.semaphore.read(txn_id, Range::One(key.clone())).await?;

        let state = self.state.lock().expect("lock state");

        if let Some(version) = state.pending.get(&txn_id) {
            if let Some(value) = version.get(&key) {
                // it's safe to call try_read after acquiring the semaphore permit
                let guard = value.clone().try_read_owned().expect("read guard");
                return Ok(Some(ValueReadGuard::Pending(permit, guard)));
            }
        }

        let canon = state.get_canon(&txn_id, &key);
        return Ok(canon.map(ValueReadGuard::Committed));
    }

    /// Insert a new entry into this [`TxnMapLock`] at the given `txn_id`.
    pub async fn insert(&self, txn_id: I, key: Arc<K>, value: V) -> Result<()> {
        {
            let state = self.state.lock().expect("lock state");
            if state.finalized >= Some(txn_id) {
                return Err(Error::Outdated);
            } else if state.committed.contains_key(&txn_id) {
                return Err(Error::Committed);
            }
        }

        let range = Range::One(key.clone());
        let _permit = self.semaphore.write(txn_id, range.clone()).await?;

        let mut state = self.state.lock().expect("lock state");

        fn version_entry<K: Ord, V>(
            version: &mut Version<K, V>,
            key: Arc<K>,
            value: Arc<RwLock<V>>,
        ) {
            match version.entry(key) {
                Entry::Occupied(mut entry) => {
                    let existing_delta = entry.get_mut();
                    *existing_delta = value;
                }
                Entry::Vacant(entry) => {
                    entry.insert(value);
                }
            }
        }

        let value = Arc::new(RwLock::new(value));
        match state.pending.entry(txn_id) {
            Entry::Occupied(mut entry) => version_entry(entry.get_mut(), key, value),
            Entry::Vacant(entry) => {
                let version = entry.insert(BTreeMap::new());
                version_entry(version, key, value)
            }
        };

        Ok(())
    }
}
