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
//! let one = Arc::new("one".to_string());
//! let two = Arc::new("two".to_string());
//!
//! let map = TxnMapLock::<u64, String, f32>::new(1);
//!
//! block_on(map.insert(1, one.clone(), 1.0)).expect("insert");
//!
//! let value = block_on(map.get(1, &one)).expect("read").expect("value");
//! assert_eq!(value, 1.0);
//!
//! assert_eq!(map.try_insert(1, one.clone(), 2.0).unwrap_err(), Error::WouldBlock);
//!
//! std::mem::drop(value);
//!
//! map.commit(1);
//!
//! let value = map.try_get(2, &one).expect("read");
//! assert_eq!(*(value.expect("guard")), 1.0);
//!
//! let value = block_on(map.remove(2, one.clone())).expect("remove").expect("value");
//! assert_eq!(*value, 1.0);
//!
//! assert!(map.try_remove(2, one.clone()).expect("remove").is_none());
//!
//! map.try_insert(2, two.clone(), 2.0).expect("insert");
//!
//! map.rollback(&2);
//!
//! let value = map.try_get(1, &one).expect("read");
//! assert_eq!(*(value.expect("guard")), 1.0);
//!
//! assert!(map.try_remove(3, two).expect("remove").is_none());
//!
//! map.finalize(2);
//!
//! assert_eq!(map.try_get(1, &one).unwrap_err(), Error::Outdated);
//!
//! let value = map.try_get(3, &one).expect("read");
//! assert_eq!(*(value.expect("guard")), 1.0);
//! ```

use std::collections::btree_map::{BTreeMap, Entry};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::Poll;
use std::{fmt, iter};

use tokio::sync::RwLock;

use super::guard::TxnReadGuard;
use super::semaphore::*;
use super::{Error, Result};

pub use super::range::Range;

/// A read guard on a value in a [`TxnMapLock`]
pub type TxnMapValueReadGuard<K, V> = TxnReadGuard<Range<K>, V>;

type Canon<K, V> = BTreeMap<Arc<K>, Arc<V>>;
type Committed<I, K, V> = BTreeMap<I, Option<BTreeMap<Arc<K>, Option<Arc<V>>>>>;
type Version<K, V> = BTreeMap<Arc<K>, Option<Arc<RwLock<V>>>>;

struct State<I, K, V> {
    canon: Canon<K, V>,
    committed: Committed<I, K, V>,
    pending: BTreeMap<I, Version<K, V>>,
    finalized: Option<I>,
}

impl<I: Copy + Ord, K: Ord, V: fmt::Debug> State<I, K, V> {
    #[inline]
    fn new(txn_id: I, version: Version<K, V>) -> Self {
        Self {
            canon: BTreeMap::new(),
            committed: BTreeMap::new(),
            pending: BTreeMap::from_iter(iter::once((txn_id, version))),
            finalized: None,
        }
    }

    #[inline]
    fn check_pending(&self, txn_id: &I) -> Result<()> {
        if self.finalized.as_ref() >= Some(txn_id) {
            Err(Error::Outdated)
        } else if self.committed.contains_key(txn_id) {
            Err(Error::Committed)
        } else {
            Ok(())
        }
    }

    #[inline]
    fn get_canon(&self, txn_id: &I, key: &K) -> Option<Arc<V>> {
        get_canon(&self.canon, &self.committed, txn_id, key)
    }

    #[inline]
    fn get_committed(
        &self,
        txn_id: &I,
        key: &K,
    ) -> Poll<Result<Option<TxnMapValueReadGuard<K, V>>>> {
        if self.finalized.as_ref() > Some(&txn_id) {
            Poll::Ready(Err(Error::Outdated))
        } else if self.committed.contains_key(&txn_id) {
            assert!(!self.pending.contains_key(&txn_id));
            let canon = self.get_canon(&txn_id, &key);
            Poll::Ready(Ok(canon.map(TxnMapValueReadGuard::Committed)))
        } else {
            Poll::Pending
        }
    }

    #[inline]
    fn get_pending(
        &self,
        txn_id: &I,
        key: &K,
        permit: Permit<Range<K>>,
    ) -> Option<TxnMapValueReadGuard<K, V>> {
        if let Some(version) = self.pending.get(&txn_id) {
            if let Some(delta) = version.get(key) {
                return if let Some(value) = delta {
                    // the permit means it's safe to call try_read_owned().expect()
                    let guard = value.clone().try_read_owned().expect("read version");
                    Some(TxnMapValueReadGuard::pending_write(permit, guard))
                } else {
                    None
                };
            }
        }

        self.get_canon(&txn_id, key)
            .map(|value| TxnMapValueReadGuard::pending_read(permit, value))
    }

    #[inline]
    fn insert(&mut self, txn_id: I, key: Arc<K>, value: V) {
        #[inline]
        fn version_entry<K: Ord, V>(
            version: &mut Version<K, V>,
            key: Arc<K>,
            value: Arc<RwLock<V>>,
        ) {
            match version.entry(key) {
                Entry::Occupied(mut entry) => {
                    let existing_delta = entry.get_mut();
                    *existing_delta = Some(value);
                }
                Entry::Vacant(entry) => {
                    entry.insert(Some(value));
                }
            }
        }

        let value = Arc::new(RwLock::new(value));
        match self.pending.entry(txn_id) {
            Entry::Occupied(mut entry) => version_entry(entry.get_mut(), key, value),
            Entry::Vacant(entry) => {
                let version = entry.insert(BTreeMap::new());
                version_entry(version, key, value)
            }
        }
    }

    #[inline]
    fn remove(&mut self, txn_id: I, key: Arc<K>) -> Option<Arc<V>> {
        match self.pending.entry(txn_id) {
            Entry::Occupied(mut pending) => match pending.get_mut().entry(key) {
                Entry::Occupied(mut entry) => {
                    if let Some(lock) = entry.insert(None) {
                        let lock = Arc::try_unwrap(lock).expect("removed value");
                        Some(Arc::new(lock.into_inner()))
                    } else {
                        None
                    }
                }
                Entry::Vacant(entry) => {
                    let prior = get_canon(&self.canon, &self.committed, &txn_id, entry.key());
                    entry.insert(None);
                    prior
                }
            },
            Entry::Vacant(pending) => {
                let prior_value = get_canon(&self.canon, &self.committed, &txn_id, &key);
                let version = iter::once((key, None)).collect();
                pending.insert(version);
                prior_value
            }
        }
    }
}

#[inline]
fn get_canon<I: Ord, K: Ord, V>(
    canon: &Canon<K, V>,
    committed: &Committed<I, K, V>,
    txn_id: &I,
    key: &K,
) -> Option<Arc<V>> {
    let committed = committed
        .iter()
        .rev()
        .skip_while(|(id, _)| *id > txn_id)
        .map(|(_, version)| version);

    for version in committed {
        if let Some(version) = version {
            if let Some(delta) = version.get(key) {
                return delta.clone();
            }
        }
    }

    canon.get(key).cloned()
}

/// A futures-aware read-write lock on a [`BTreeMap`] which supports transactional versioning
// TODO: handle the case where a write permit is acquired and then dropped without committing
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

impl<I, K, V> TxnMapLock<I, K, V> {
    #[inline]
    fn state(&self) -> MutexGuard<State<I, K, V>> {
        self.state.lock().expect("lock state")
    }
}

impl<I: Ord + Copy + fmt::Display, K: Ord + fmt::Debug, V: fmt::Debug> TxnMapLock<I, K, V> {
    /// Construct a new [`TxnMapLock`].
    pub fn new(txn_id: I) -> Self {
        Self {
            state: Arc::new(Mutex::new(State::new(txn_id, Version::new()))),
            semaphore: Semaphore::new(),
        }
    }

    /// Construct a new [`TxnMapLock`] with the given `contents`.
    pub fn with_contents<C: IntoIterator<Item = (K, V)>>(txn_id: I, contents: C) -> Self {
        let version = contents
            .into_iter()
            .map(|(key, value)| (Arc::new(key), Some(Arc::new(RwLock::new(value)))))
            .collect();

        Self {
            state: Arc::new(Mutex::new(State::new(txn_id, version))),
            semaphore: Semaphore::with_reservation(txn_id, Range::All),
        }
    }

    /// Commit the state of this [`TxnMapLock`] at `txn_id`.
    /// Panics:
    ///  - if any new value to commit is still locked (for reading or writing)
    pub fn commit(&self, txn_id: I) {
        let mut state = self.state();

        self.semaphore.finalize(&txn_id, false);

        let version = if let Some(version) = state.pending.remove(&txn_id) {
            let version = version
                .into_iter()
                .map(|(key, delta)| {
                    let value = if let Some(present) = delta {
                        if let Ok(value) = Arc::try_unwrap(present) {
                            Some(Arc::new(value.into_inner()))
                        } else {
                            panic!("a value to commit at {} is still locked", txn_id);
                        }
                    } else {
                        None
                    };

                    (key, value)
                })
                .collect();

            Some(version)
        } else {
            None
        };

        match state.committed.entry(txn_id) {
            Entry::Occupied(_) => {
                assert!(version.is_none());
                #[cfg(feature = "logging")]
                log::warn!("duplicate commit at {}", txn_id);
            }
            Entry::Vacant(entry) => {
                entry.insert(version);
            }
        }
    }

    /// Finalize the state of this [`TxnMapLock`] at `txn_id`.
    /// This will merge in deltas and prevent further reads of versions earlier than `txn_id`.
    pub fn finalize(&self, txn_id: I) {
        let mut state = self.state();

        while let Some(version_id) = state.committed.keys().next().copied() {
            if version_id <= txn_id {
                if let Some(version) = state.committed.remove(&version_id).expect("version") {
                    for (key, delta) in version {
                        match delta {
                            Some(value) => state.canon.insert(key, value),
                            None => state.canon.remove(&key),
                        };
                    }
                }
            } else {
                break;
            }
        }

        self.semaphore.finalize(&txn_id, true);
        state.finalized = Some(txn_id);
    }

    /// Read a value from this [`TxnMapLock`] at `txn_id`.
    pub async fn get(&self, txn_id: I, key: &Arc<K>) -> Result<Option<TxnMapValueReadGuard<K, V>>> {
        // before acquiring a permit, check if this version has already been committed
        if let Poll::Ready(result) = self.state().get_committed(&txn_id, key) {
            return result;
        }

        let permit = self.semaphore.read(txn_id, Range::One(key.clone())).await?;
        let state = self.state();
        Ok(state.get_pending(&txn_id, key, permit))
    }

    /// Read a value from this [`TxnMapLock`] at `txn_id` synchronously, if possible.
    pub fn try_get(&self, txn_id: I, key: &Arc<K>) -> Result<Option<TxnMapValueReadGuard<K, V>>> {
        // before acquiring a permit, check if this version has already been committed
        if let Poll::Ready(result) = self.state().get_committed(&txn_id, key) {
            return result;
        }

        let permit = self.semaphore.try_read(txn_id, Range::One(key.clone()))?;
        let state = self.state();
        Ok(state.get_pending(&txn_id, key, permit))
    }

    /// Insert a new entry into this [`TxnMapLock`] at `txn_id`.
    pub async fn insert(&self, txn_id: I, key: Arc<K>, value: V) -> Result<()> {
        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let range = Range::One(key.clone());
        let _permit = self.semaphore.write(txn_id, range).await?;

        let mut state = self.state();
        Ok(state.insert(txn_id, key, value))
    }

    /// Insert a new entry into this [`TxnMapLock`] at `txn_id` synchronously, if possible.
    pub fn try_insert(&self, txn_id: I, key: Arc<K>, value: V) -> Result<()> {
        let mut state = self.state();

        // before acquiring a permit, check if this version has already been committed
        state.check_pending(&txn_id)?;

        let _permit = self.semaphore.try_write(txn_id, Range::One(key.clone()))?;

        Ok(state.insert(txn_id, key, value))
    }

    /// Remove and return the value at `key` from this [`TxnMapLock`] at `txn_id`, if present.
    pub async fn remove(&self, txn_id: I, key: Arc<K>) -> Result<Option<Arc<V>>> {
        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let range = Range::One(key.clone());
        let _permit = self.semaphore.write(txn_id, range).await?;

        let mut state = self.state();
        Ok(state.remove(txn_id, key))
    }

    /// Remove and return the value at `key` from this [`TxnMapLock`] at `txn_id`, if present.
    pub fn try_remove(&self, txn_id: I, key: Arc<K>) -> Result<Option<Arc<V>>> {
        let mut state = self.state();

        // before acquiring a permit, check if this version has already been committed
        state.check_pending(&txn_id)?;

        let range = Range::One(key.clone());
        let _permit = self.semaphore.try_write(txn_id, range)?;

        Ok(state.remove(txn_id, key))
    }

    /// Roll back the state of this [`TxnMapLock`] at `txn_id`.
    pub fn rollback(&self, txn_id: &I) {
        let mut state = self.state();

        assert!(
            !state.committed.contains_key(&txn_id),
            "cannot roll back committed transaction {}",
            txn_id
        );

        self.semaphore.finalize(txn_id, false);
        state.pending.remove(txn_id);
    }
}
