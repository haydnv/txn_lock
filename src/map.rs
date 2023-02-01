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
//! let mut value = map.try_get_mut(2, one.clone()).expect("read").expect("value");
//! assert_eq!(value, 1.0);
//! *value = 2.0;
//!
//! assert_eq!(map.try_remove(2, one.clone()).unwrap_err(), Error::WouldBlock);
//! std::mem::drop(value);
//!
//! let value = block_on(map.remove(2, one.clone())).expect("remove").expect("value");
//! assert_eq!(*value, 2.0);
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
//!
//! map.commit(3);
//!
//! let extension = [
//!     ("one".to_string(), 1.0),
//!     ("two".to_string(), 2.0),
//!     ("three".to_string(), 3.0),
//!     ("four".to_string(), 4.0),
//!     ("five".to_string(), 5.0),
//! ];
//!
//! map.try_extend(4, extension).expect("extend");
//!
//! for (key, mut value) in map.try_iter_mut(4).expect("iter") {
//!     *value *= 2.;
//! }
//!
//! // note: alphabetical order
//! let expected = [
//!     ("two".to_string(), 4.0),
//!     ("three".to_string(), 6.0),
//!     ("one".to_string(), 2.0),
//!     ("four".to_string(), 8.0),
//!     ("five".to_string(), 10.0),
//! ];
//!
//! let actual = map.try_iter(4).expect("iter").rev().collect::<Vec<_>>();
//! assert_eq!(actual.len(), expected.len());
//!
//! for ((lk, lv), (rk, rv)) in actual.into_iter().zip(&expected) {
//!     assert_eq!(&*lk, rk);
//!     assert_eq!(&*lv, rv);
//! }
//!
//! let actual = map.try_clear(4).expect("clear");
//! assert_eq!(actual, expected.into_iter().map(|(k, v)| (k.into(), v.into())).collect());
//!
//! ```

use std::cmp::Ordering;
use std::collections::btree_map::{BTreeMap, Entry};
use std::collections::BTreeSet;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, RwLock as RwLockInner};
use std::task::Poll;
use std::{fmt, iter};

use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use super::guard::{TxnReadGuard, TxnWriteGuard};
use super::semaphore::*;
use super::{Error, Result};

pub use super::range::Range;

/// A read guard on a value in a [`TxnMapLock`]
pub type TxnMapValueReadGuard<K, V> = TxnReadGuard<Range<K>, V>;

/// A write guard on a value in a [`TxnMapLock`]
pub type TxnMapValueWriteGuard<K, V> = TxnWriteGuard<Range<K>, V>;

type Canon<K, V> = BTreeMap<Arc<K>, Arc<V>>;
type Delta<K, V> = BTreeMap<Arc<K>, Option<Arc<V>>>;
type Committed<I, K, V> = BTreeMap<I, Option<Delta<K, V>>>;
type Pending<K, V> = BTreeMap<Arc<K>, Option<Arc<RwLock<V>>>>;

#[derive(Debug)]
enum PendingValue<V> {
    Committed(Arc<V>),
    Pending(OwnedRwLockReadGuard<V>),
}

struct State<I, K, V> {
    canon: Canon<K, V>,
    committed: Committed<I, K, V>,
    pending: BTreeMap<I, Pending<K, V>>,
    finalized: Option<I>,
}

impl<I: Copy + Ord, K: Ord, V: fmt::Debug> State<I, K, V> {
    #[inline]
    fn new(txn_id: I, version: Pending<K, V>) -> Self {
        Self {
            canon: BTreeMap::new(),
            committed: BTreeMap::new(),
            pending: BTreeMap::from_iter(iter::once((txn_id, version))),
            finalized: None,
        }
    }

    #[inline]
    fn check_committed(&self, txn_id: &I) -> Result<bool> {
        match self.finalized.as_ref().cmp(&Some(txn_id)) {
            Ordering::Greater => Err(Error::Outdated),
            Ordering::Equal => Ok(true),
            Ordering::Less => Ok(self.committed.contains_key(txn_id)),
        }
    }

    #[inline]
    fn check_pending(&self, txn_id: &I) -> Result<()> {
        if self.finalized.as_ref() > Some(txn_id) {
            Err(Error::Outdated)
        } else if self.committed.contains_key(txn_id) {
            Err(Error::Committed)
        } else {
            Ok(())
        }
    }

    #[inline]
    fn clear(&mut self, txn_id: I) -> Canon<K, V> {
        let mut map = self.canon.clone();

        let committed =
            self.committed
                .iter()
                .filter_map(|(id, version)| if id < &txn_id { version.as_ref() } else { None });

        for deltas in committed {
            for (key, delta) in deltas {
                if let Some(value) = delta {
                    map.insert(key.clone(), value.clone());
                } else {
                    map.remove(key);
                }
            }
        }

        if let Some(version) = self.pending.remove(&txn_id) {
            for (key, delta) in version {
                if let Some(value) = delta {
                    let value = Arc::try_unwrap(value).expect("value");
                    map.insert(key, Arc::new(value.into_inner()));
                } else {
                    map.remove(&key);
                }
            }
        }

        let version = map.keys().cloned().map(|key| (key, None)).collect();
        self.pending.insert(txn_id, version);

        map
    }

    #[inline]
    fn extend<Q, E>(&mut self, txn_id: I, other: E)
    where
        Q: Into<Arc<K>>,
        E: IntoIterator<Item = (Q, V)>,
    {
        let entries = other
            .into_iter()
            .map(|(key, value)| (key.into(), Some(Arc::new(RwLock::new(value)))));

        match self.pending.entry(txn_id) {
            Entry::Occupied(mut entry) => {
                let version = entry.get_mut();
                version.extend(entries);
            }
            Entry::Vacant(entry) => {
                let version = entry.insert(BTreeMap::new());
                version.extend(entries);
            }
        }
    }

    #[inline]
    fn get_canon(&self, txn_id: &I, key: &K) -> Option<Arc<V>> {
        get_canon(&self.canon, &self.committed, txn_id, key).cloned()
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
            let value = self.get_canon(&txn_id, key);
            Poll::Ready(Ok(value.map(TxnMapValueReadGuard::committed)))
        } else {
            Poll::Pending
        }
    }

    #[inline]
    fn get_pending(&self, txn_id: &I, key: &K) -> Option<PendingValue<V>> {
        if let Some(version) = self.pending.get(&txn_id) {
            if let Some(delta) = version.get(key) {
                return if let Some(value) = delta {
                    // the permit means it's safe to call try_read_owned().expect()
                    let guard = value.clone().try_read_owned().expect("read version");
                    Some(PendingValue::Pending(guard))
                } else {
                    None
                };
            }
        }

        self.get_canon(&txn_id, key)
            .map(|value| PendingValue::Committed(value))
    }

    #[inline]
    fn insert(&mut self, txn_id: I, key: Arc<K>, value: V) {
        #[inline]
        fn insert_entry<K: Ord, V>(
            version: &mut Pending<K, V>,
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
            Entry::Occupied(mut entry) => insert_entry(entry.get_mut(), key, value),
            Entry::Vacant(entry) => {
                let version = entry.insert(BTreeMap::new());
                insert_entry(version, key, value)
            }
        }
    }

    #[inline]
    fn keys_committed(&self, txn_id: &I) -> BTreeSet<Arc<K>> {
        let mut keys = self.canon.keys().cloned().collect();

        let committed = self.committed.iter().filter_map(|(id, version)| {
            if id <= &txn_id {
                version.as_ref()
            } else {
                None
            }
        });

        for deltas in committed {
            merge_keys(&mut keys, deltas);
        }

        keys
    }

    #[inline]
    fn keys_pending(&self, txn_id: I) -> BTreeSet<Arc<K>> {
        let mut keys = self.keys_committed(&txn_id);

        if let Some(pending) = self.pending.get(&txn_id) {
            merge_keys(&mut keys, pending);
        }

        keys
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
                    if let Some(prior) =
                        get_canon(&self.canon, &self.committed, &txn_id, entry.key()).cloned()
                    {
                        entry.insert(None);
                        Some(prior)
                    } else {
                        None
                    }
                }
            },
            Entry::Vacant(pending) => {
                if let Some(prior) = get_canon(&self.canon, &self.committed, &txn_id, &key).cloned()
                {
                    pending.insert(iter::once((key, None)).collect());
                    Some(prior)
                } else {
                    None
                }
            }
        }
    }
}

impl<I: Copy + Ord, K: Ord, V: Clone + fmt::Debug> State<I, K, V> {
    #[inline]
    fn get_mut(&mut self, txn_id: I, key: Arc<K>) -> Option<OwnedRwLockWriteGuard<V>> {
        #[inline]
        fn new_value<V: Clone>(canon: &V) -> (Arc<RwLock<V>>, OwnedRwLockWriteGuard<V>) {
            let value = V::clone(canon);
            let value = Arc::new(RwLock::new(value));
            let guard = value.clone().try_write_owned().expect("write version");
            (value, guard)
        }

        match self.pending.entry(txn_id) {
            Entry::Occupied(mut pending) => match pending.get_mut().entry(key) {
                Entry::Occupied(delta) => {
                    let value = delta.get().as_ref()?;

                    value
                        .clone()
                        .try_write_owned()
                        .map(Some)
                        .expect("write version")
                }
                Entry::Vacant(delta) => {
                    let canon = get_canon(&self.canon, &self.committed, &txn_id, delta.key())?;
                    let (value, guard) = new_value(&**canon);

                    delta.insert(Some(value));

                    Some(guard)
                }
            },
            Entry::Vacant(pending) => {
                let canon = get_canon(&self.canon, &self.committed, &txn_id, &key)?;
                let (value, guard) = new_value(&**canon);

                let version = iter::once((key, Some(value))).collect();
                pending.insert(version);

                Some(guard)
            }
        }
    }
}

#[inline]
fn get_canon<'a, I: Ord, K: Ord, V>(
    canon: &'a Canon<K, V>,
    committed: &'a Committed<I, K, V>,
    txn_id: &'a I,
    key: &'a K,
) -> Option<&'a Arc<V>> {
    let committed = committed
        .iter()
        .rev()
        .skip_while(|(id, _)| *id > txn_id)
        .map(|(_, version)| version);

    for version in committed {
        if let Some(deltas) = version {
            if let Some(delta) = deltas.get(key) {
                return delta.as_ref();
            }
        }
    }

    canon.get(key)
}

#[inline]
fn merge_keys<K: Ord, V>(keys: &mut BTreeSet<Arc<K>>, deltas: &Delta<K, V>) {
    for (key, delta) in deltas {
        if delta.is_some() {
            keys.insert(key.clone());
        } else {
            keys.remove(key);
        }
    }
}

/// A futures-aware read-write lock on a [`BTreeMap`] which supports transactional versioning
///
/// The `get_mut` and `try_get_mut` methods require the value type `V` to implement [`Clone`]
/// in order to support multiple transactions with different versions.
// TODO: handle the case where a write permit is acquired and then dropped without committing
pub struct TxnMapLock<I, K, V> {
    state: Arc<RwLockInner<State<I, K, V>>>,
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
    fn state(&self) -> impl Deref<Target = State<I, K, V>> + '_ {
        self.state.read().expect("lock state")
    }

    #[inline]
    fn state_mut(&self) -> impl DerefMut<Target = State<I, K, V>> + '_ {
        self.state.write().expect("lock state")
    }
}

impl<I: Ord + Copy + fmt::Display, K: Ord + fmt::Debug, V: fmt::Debug> TxnMapLock<I, K, V> {
    /// Construct a new [`TxnMapLock`].
    pub fn new(txn_id: I) -> Self {
        Self {
            state: Arc::new(RwLockInner::new(State::new(txn_id, Pending::new()))),
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
            state: Arc::new(RwLockInner::new(State::new(txn_id, version))),
            semaphore: Semaphore::with_reservation(txn_id, Range::All),
        }
    }

    /// Commit the state of this [`TxnMapLock`] at `txn_id`.
    /// Panics:
    ///  - if any new value to commit is still locked (for reading or writing)
    pub fn commit(&self, txn_id: I) {
        let mut state = self.state_mut();

        if state.finalized.as_ref() >= Some(&txn_id) {
            #[cfg(feature = "logging")]
            log::warn!("committed already-finalized version {}", txn_id);
            return;
        }

        self.semaphore.finalize(&txn_id, false);

        let finalize = state.pending.keys().next() == Some(&txn_id);
        let version = state.pending.remove(&txn_id).map(|version| {
            version
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
                .collect()
        });

        if finalize {
            assert!(!state.committed.contains_key(&txn_id));
            let deltas = version.expect("committed version");
            merge(&mut state.canon, deltas);
            state.finalized = Some(txn_id);
        } else if let Some(deltas) = version {
            assert!(state.committed.insert(txn_id, Some(deltas)).is_none());
        } else if let Some(prior_commit) = state.committed.insert(txn_id, None) {
            assert!(prior_commit.is_none());
            #[cfg(feature = "logging")]
            log::warn!("duplicate commit at {}", txn_id);
        }
    }

    /// Finalize the state of this [`TxnMapLock`] at `txn_id`.
    /// This will finalize commits and prevent further reads of versions earlier than `txn_id`.
    pub fn finalize(&self, txn_id: I) {
        let mut state = self.state_mut();

        while let Some(version_id) = state.committed.keys().next().copied() {
            if version_id <= txn_id {
                if let Some(deltas) = state.committed.remove(&version_id).expect("version") {
                    merge(&mut state.canon, deltas);
                }
            } else {
                break;
            }
        }

        self.semaphore.finalize(&txn_id, true);
        state.finalized = Some(txn_id);
    }

    /// Remove and return all entries from this [`TxnMapLock`] at `txn_id`.
    pub async fn clear(&self, txn_id: I) -> Result<Canon<K, V>> {
        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let _permit = self.semaphore.write(txn_id, Range::All).await?;

        Ok(self.state_mut().clear(txn_id))
    }

    /// Remove and return all entries from this [`TxnMapLock`] at `txn_id`.
    pub fn try_clear(&self, txn_id: I) -> Result<Canon<K, V>> {
        let mut state = self.state_mut();

        // before acquiring a permit, check if this version has already been committed
        state.check_pending(&txn_id)?;

        let _permit = self.semaphore.try_write(txn_id, Range::All)?;

        Ok(state.clear(txn_id))
    }

    /// Insert the entries from `other` [`TxnMapLock`] at `txn_id`.
    pub async fn extend<Q, E>(&self, txn_id: I, other: E) -> Result<()>
    where
        Q: Into<Arc<K>>,
        E: IntoIterator<Item = (Q, V)>,
    {
        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let _permit = self.semaphore.write(txn_id, Range::All).await?;

        Ok(self.state_mut().extend(txn_id, other))
    }

    /// Insert the entries from `other` [`TxnMapLock`] at `txn_id` synchronously, if possible.
    pub fn try_extend<Q, E>(&self, txn_id: I, other: E) -> Result<()>
    where
        Q: Into<Arc<K>>,
        E: IntoIterator<Item = (Q, V)>,
    {
        let mut state = self.state_mut();

        // before acquiring a permit, check if this version has already been committed
        state.check_pending(&txn_id)?;

        let _permit = self.semaphore.try_write(txn_id, Range::All)?;

        Ok(state.extend(txn_id, other))
    }

    /// Read a value from this [`TxnMapLock`] at `txn_id`.
    pub async fn get(&self, txn_id: I, key: &Arc<K>) -> Result<Option<TxnMapValueReadGuard<K, V>>> {
        // before acquiring a permit, check if this version has already been committed
        if let Poll::Ready(result) = self.state().get_committed(&txn_id, key) {
            return result;
        }

        let permit = self.semaphore.read(txn_id, Range::One(key.clone())).await?;
        let value = self
            .state()
            .get_pending(&txn_id, key)
            .map(|value| match value {
                PendingValue::Committed(value) => TxnMapValueReadGuard::committed(value),
                PendingValue::Pending(value) => TxnMapValueReadGuard::pending_write(permit, value),
            });

        Ok(value)
    }

    /// Read a value from this [`TxnMapLock`] at `txn_id` synchronously, if possible.
    pub fn try_get(&self, txn_id: I, key: &Arc<K>) -> Result<Option<TxnMapValueReadGuard<K, V>>> {
        let state = self.state();

        // before acquiring a permit, check if this version has already been committed
        if let Poll::Ready(result) = state.get_committed(&txn_id, key) {
            return result;
        }

        let permit = self.semaphore.try_read(txn_id, Range::One(key.clone()))?;
        let value = state.get_pending(&txn_id, key).map(|value| match value {
            PendingValue::Committed(value) => TxnMapValueReadGuard::committed(value),
            PendingValue::Pending(value) => TxnMapValueReadGuard::pending_write(permit, value),
        });

        Ok(value)
    }

    /// Construct an iterator over the entries in this [`TxnMapLock`] at `txn_id`.
    pub async fn iter(&self, txn_id: I) -> Result<Iter<I, K, V>> {
        {
            let state = self.state();

            // before acquiring a permit, check if this version has already been committed
            if state.check_committed(&txn_id)? {
                let keys = state.keys_committed(&txn_id);
                return Ok(Iter::new(self.state.clone(), txn_id, None, keys));
            }
        }

        let permit = self.semaphore.read(txn_id, Range::All).await?;
        let keys = self.state().keys_pending(txn_id);
        return Ok(Iter::new(self.state.clone(), txn_id, Some(permit), keys));
    }

    /// Construct an iterator over the entries in this [`TxnMapLock`] at `txn_id` synchronously.
    pub fn try_iter(&self, txn_id: I) -> Result<Iter<I, K, V>> {
        let state = self.state();

        // before acquiring a permit, check if this version has already been committed
        if state.check_committed(&txn_id)? {
            let keys = state.keys_committed(&txn_id);
            return Ok(Iter::new(self.state.clone(), txn_id, None, keys));
        }

        let permit = self.semaphore.try_read(txn_id, Range::All)?;
        let keys = state.keys_pending(txn_id);
        return Ok(Iter::new(self.state.clone(), txn_id, Some(permit), keys));
    }

    /// Insert a new entry into this [`TxnMapLock`] at `txn_id`.
    pub async fn insert<Q: Into<Arc<K>>>(&self, txn_id: I, key: Q, value: V) -> Result<()> {
        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let key = key.into();
        let range = Range::One(key.clone());
        let _permit = self.semaphore.write(txn_id, range).await?;

        Ok(self.state_mut().insert(txn_id, key, value))
    }

    /// Insert a new entry into this [`TxnMapLock`] at `txn_id` synchronously, if possible.
    pub fn try_insert<Q: Into<Arc<K>>>(&self, txn_id: I, key: Q, value: V) -> Result<()> {
        let mut state = self.state_mut();

        // before acquiring a permit, check if this version has already been committed
        state.check_pending(&txn_id)?;

        let key = key.into();
        let _permit = self.semaphore.try_write(txn_id, Range::One(key.clone()))?;

        Ok(state.insert(txn_id, key, value))
    }

    /// Remove and return the value at `key` from this [`TxnMapLock`] at `txn_id`, if present.
    pub async fn remove<Q: Into<Arc<K>>>(&self, txn_id: I, key: Q) -> Result<Option<Arc<V>>> {
        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let key = key.into();
        let range = Range::One(key.clone());
        let _permit = self.semaphore.write(txn_id, range).await?;

        Ok(self.state_mut().remove(txn_id, key))
    }

    /// Remove and return the value at `key` from this [`TxnMapLock`] at `txn_id`, if present.
    pub fn try_remove<Q: Into<Arc<K>>>(&self, txn_id: I, key: Q) -> Result<Option<Arc<V>>> {
        let mut state = self.state_mut();

        // before acquiring a permit, check if this version has already been committed
        state.check_pending(&txn_id)?;

        let key = key.into();
        let range = Range::One(key.clone());
        let _permit = self.semaphore.try_write(txn_id, range)?;

        Ok(state.remove(txn_id, key))
    }

    /// Roll back the state of this [`TxnMapLock`] at `txn_id`.
    pub fn rollback(&self, txn_id: &I) {
        let mut state = self.state_mut();

        assert!(
            !state.committed.contains_key(&txn_id),
            "cannot roll back committed transaction {}",
            txn_id
        );

        self.semaphore.finalize(txn_id, false);
        state.pending.remove(txn_id);
    }
}

impl<I: Ord + Copy + fmt::Display, K: Ord + fmt::Debug, V: Clone + fmt::Debug> TxnMapLock<I, K, V> {
    /// Read a mutable value from this [`TxnMapLock`] at `txn_id`.
    pub async fn get_mut<Q: Into<Arc<K>>>(
        &self,
        txn_id: I,
        key: Q,
    ) -> Result<Option<TxnMapValueWriteGuard<K, V>>> {
        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let key = key.into();
        let range = Range::One(key.clone());
        let permit = self.semaphore.write(txn_id, range).await?;

        if let Some(value) = self.state_mut().get_mut(txn_id, key) {
            Ok(Some(TxnMapValueWriteGuard::new(permit, value)))
        } else {
            Ok(None)
        }
    }

    /// Read a mutable value from this [`TxnMapLock`] at `txn_id`.
    pub fn try_get_mut<Q: Into<Arc<K>>>(
        &self,
        txn_id: I,
        key: Q,
    ) -> Result<Option<TxnMapValueWriteGuard<K, V>>> {
        let mut state = self.state_mut();

        // before acquiring a permit, check if this version has already been committed
        state.check_pending(&txn_id)?;

        let key = key.into();
        let permit = self.semaphore.try_write(txn_id, Range::One(key.clone()))?;

        if let Some(value) = state.get_mut(txn_id, key) {
            Ok(Some(TxnMapValueWriteGuard::new(permit, value)))
        } else {
            Ok(None)
        }
    }

    /// Construct a mutable iterator over the entries in this [`TxnMapLock`] at `txn_id`.
    pub async fn iter_mut(&self, txn_id: I) -> Result<IterMut<I, K, V>> {
        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let permit = self.semaphore.write(txn_id, Range::All).await?;
        let keys = self.state().keys_pending(txn_id);
        Ok(IterMut::new(self.state.clone(), txn_id, permit, keys))
    }

    /// Construct a mutable iterator over the entries in this [`TxnMapLock`] at `txn_id`,
    /// synchronously if possible.
    pub fn try_iter_mut(&self, txn_id: I) -> Result<IterMut<I, K, V>> {
        let state = self.state();

        // before acquiring a permit, check if this version has already been committed
        state.check_pending(&txn_id)?;

        let permit = self.semaphore.try_write(txn_id, Range::All)?;
        let keys = state.keys_pending(txn_id);
        Ok(IterMut::new(self.state.clone(), txn_id, permit, keys))
    }
}

/// A guard on a value in an [`Iter`]
#[derive(Debug)]
pub struct TxnMapIterGuard<V> {
    value: PendingValue<V>,
}

impl<V> From<PendingValue<V>> for TxnMapIterGuard<V> {
    fn from(value: PendingValue<V>) -> Self {
        Self { value }
    }
}

impl<V> Deref for TxnMapIterGuard<V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        match &self.value {
            PendingValue::Committed(value) => value.deref(),
            PendingValue::Pending(value) => value.deref(),
        }
    }
}

impl<V: PartialEq> PartialEq<V> for TxnMapIterGuard<V> {
    fn eq(&self, other: &V) -> bool {
        self.deref().eq(other)
    }
}

impl<V: PartialOrd> PartialOrd<V> for TxnMapIterGuard<V> {
    fn partial_cmp(&self, other: &V) -> Option<Ordering> {
        self.deref().partial_cmp(other)
    }
}

/// An iterator over the entries in a [`TxnMapLock`] as of a specific transaction
pub struct Iter<I, K, V> {
    lock_state: Arc<RwLockInner<State<I, K, V>>>,
    txn_id: I,
    permit: Option<PermitRead<Range<K>>>,
    keys: <BTreeSet<Arc<K>> as IntoIterator>::IntoIter,
}

impl<I, K, V> Iter<I, K, V> {
    fn new(
        lock_state: Arc<RwLockInner<State<I, K, V>>>,
        txn_id: I,
        permit: Option<PermitRead<Range<K>>>,
        keys: BTreeSet<Arc<K>>,
    ) -> Self {
        Self {
            lock_state,
            txn_id,
            permit,
            keys: keys.into_iter(),
        }
    }
}

impl<I: Copy + Ord, K: Ord, V: fmt::Debug> Iterator for Iter<I, K, V> {
    type Item = (Arc<K>, TxnMapIterGuard<V>);

    fn next(&mut self) -> Option<Self::Item> {
        let state = self.lock_state.read().expect("lock state");

        loop {
            let key = self.keys.next()?;
            let value = get_key(&state, &self.txn_id, &key, self.permit.is_none());
            if let Some(value) = value {
                return Some((key, value.into()));
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.keys.size_hint()
    }
}

impl<I: Copy + Ord, K: Ord, V: fmt::Debug> DoubleEndedIterator for Iter<I, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let state = self.lock_state.read().expect("lock state");

        loop {
            let key = self.keys.next_back()?;
            let value = get_key(&state, &self.txn_id, &key, self.permit.is_none());
            if let Some(value) = value {
                return Some((key, value.into()));
            }
        }
    }
}

#[inline]
fn get_key<I: Copy + Ord, K: Ord, V: fmt::Debug>(
    state: &State<I, K, V>,
    txn_id: &I,
    key: &K,
    committed: bool,
) -> Option<PendingValue<V>> {
    if committed {
        state.get_canon(txn_id, &key).map(PendingValue::Committed)
    } else {
        state.get_pending(txn_id, &key)
    }
}

/// A guard on a mutable value in an [`IterMut`]
#[derive(Debug)]
pub struct TxnMapIterMutGuard<V> {
    guard: OwnedRwLockWriteGuard<V>,
}

impl<V> Deref for TxnMapIterMutGuard<V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

impl<V> DerefMut for TxnMapIterMutGuard<V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard.deref_mut()
    }
}

impl<V: PartialEq> PartialEq<V> for TxnMapIterMutGuard<V> {
    fn eq(&self, other: &V) -> bool {
        self.deref().eq(other)
    }
}

impl<V: PartialOrd> PartialOrd<V> for TxnMapIterMutGuard<V> {
    fn partial_cmp(&self, other: &V) -> Option<Ordering> {
        self.deref().partial_cmp(other)
    }
}

impl<V> From<OwnedRwLockWriteGuard<V>> for TxnMapIterMutGuard<V> {
    fn from(guard: OwnedRwLockWriteGuard<V>) -> Self {
        Self { guard }
    }
}

/// An iterator over the keys and mutable values in a [`TxnMapLock`]
pub struct IterMut<I, K, V> {
    lock_state: Arc<RwLockInner<State<I, K, V>>>,
    txn_id: I,

    #[allow(unused)]
    permit: PermitWrite<Range<K>>,
    keys: <BTreeSet<Arc<K>> as IntoIterator>::IntoIter,
}

impl<I, K, V> IterMut<I, K, V> {
    fn new(
        lock_state: Arc<RwLockInner<State<I, K, V>>>,
        txn_id: I,
        permit: PermitWrite<Range<K>>,
        keys: BTreeSet<Arc<K>>,
    ) -> Self {
        Self {
            lock_state,
            txn_id,
            permit,
            keys: keys.into_iter(),
        }
    }
}

impl<I: Copy + Ord, K: Ord, V: Clone + fmt::Debug> Iterator for IterMut<I, K, V> {
    type Item = (Arc<K>, TxnMapIterMutGuard<V>);

    fn next(&mut self) -> Option<Self::Item> {
        let mut state = self.lock_state.write().expect("lock state");

        loop {
            let key = self.keys.next()?;
            if let Some(guard) = state.get_mut(self.txn_id, key.clone()) {
                return Some((key, TxnMapIterMutGuard::from(guard)));
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.keys.size_hint()
    }
}

impl<I: Copy + Ord, K: Ord, V: Clone + fmt::Debug> DoubleEndedIterator for IterMut<I, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let mut state = self.lock_state.write().expect("lock state");

        loop {
            let key = self.keys.next_back()?;
            if let Some(guard) = state.get_mut(self.txn_id, key.clone()) {
                return Some((key, TxnMapIterMutGuard::from(guard)));
            }
        }
    }
}

#[inline]
fn merge<K: Ord, V>(canon: &mut Canon<K, V>, deltas: Delta<K, V>) {
    for (key, delta) in deltas {
        match delta {
            Some(value) => canon.insert(key, value),
            None => canon.remove(&key),
        };
    }
}
