//! A futures-aware read-write lock on a [`HashMap`] which supports transactional versioning.
//!
//! Example usage:
//! ```
//! use std::collections::HashMap;
//! use std::sync::Arc;
//!
//! use futures::executor::block_on;
//!
//! use txn_lock::map::*;
//! use txn_lock::Error;
//!
//! let one = "one";
//! let two = "two";
//!
//! let map = TxnMapLock::<u64, String, f32>::new(1);
//!
//! assert_eq!(block_on(map.insert(1, one.to_string(), 1.0)).expect("insert"), None);
//!
//! let value = block_on(map.get(1, one)).expect("read").expect("value");
//! assert_eq!(value, 1.0);
//!
//! assert_eq!(map.try_insert(1, one.to_string(), 2.0).unwrap_err(), Error::WouldBlock);
//!
//! std::mem::drop(value);
//!
//! let (state_at_txn_1, deltas) = block_on(map.read_and_commit(1));
//! assert_eq!(deltas.expect("deltas").len(), 1);
//! assert_eq!(state_at_txn_1.len(), 1);
//!
//! let mut value = map.try_get_mut(2, one).expect("read").expect("value");
//! assert_eq!(value, 1.0);
//! *value = 2.0;
//!
//! assert_eq!(map.try_remove(2, one).unwrap_err(), Error::WouldBlock);
//! std::mem::drop(value);
//!
//! let value = block_on(map.remove(2, one)).expect("remove").expect("value");
//! assert_eq!(*value, 2.0);
//!
//! assert_eq!(map.try_insert(2, two.to_string(), 1.0).expect("insert"), None);
//!
//! assert!(map.try_remove(2, one).expect("remove").is_none());
//!
//! assert_eq!(map.try_insert(2, two.to_string(), 2.0).expect("insert"), Some(1.0.into()));
//!
//! map.rollback(&2);
//!
//! let value = map.try_get(1, one).expect("read");
//! assert_eq!(*(value.expect("guard")), 1.0);
//!
//! assert!(map.try_remove(3, two).expect("remove").is_none());
//!
//! map.finalize(2);
//!
//! assert_eq!(map.try_get(1, one).unwrap_err(), Error::Outdated);
//!
//! let value = map.try_get(3, one).expect("read");
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
//! let expected = HashMap::<String, f32>::from_iter([
//!     ("two".to_string(), 4.0),
//!     ("three".to_string(), 6.0),
//!     ("one".to_string(), 2.0),
//!     ("four".to_string(), 8.0),
//!     ("five".to_string(), 10.0),
//! ]);
//!
//! let actual = map.try_iter(4).expect("iter").collect::<Vec<_>>();
//! assert_eq!(actual.len(), expected.len());
//!
//! for (k, v) in actual {
//!     assert_eq!(expected.get(&*k), Some(&*v));
//! }
//!
//! let actual = map.try_clear(4).expect("clear");
//! assert_eq!(actual, expected.into_iter().map(|(k, v)| (k.into(), v.into())).collect());
//!
//! ```

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::hash_map::{self, HashMap};
use std::collections::HashSet;
use std::hash::Hash;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, RwLock as RwLockInner};
use std::task::Poll;
use std::{fmt, iter};

use collate::Collator;
use ds_ext::{OrdHashMap, OrdHashSet};
use futures::TryFutureExt;
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use super::guard::{TxnReadGuard, TxnReadGuardMap, TxnWriteGuard};
use super::semaphore::*;
use super::{Error, Result};

pub use super::range::{Key, Range};

type Canon<K, V> = HashMap<Key<K>, Arc<V>>;
type Delta<K, V> = HashMap<Key<K>, Option<Arc<V>>>;
type Deltas<I, K, V> = OrdHashMap<I, Delta<K, V>>;
type Pending<K, V> = HashMap<Key<K>, Option<Arc<RwLock<V>>>>;

/// A read guard on a value in a [`TxnMapLock`]
pub type TxnMapValueReadGuard<K, V> = TxnReadGuard<Range<K>, V>;

/// A mapped read guard on a value in a [`TxnMapLock`]
pub type TxnMapValueReadGuardMap<K, V> = TxnReadGuardMap<Range<K>, V>;

/// A write guard on a value in a [`TxnMapLock`]
pub type TxnMapValueWriteGuard<K, V> = TxnWriteGuard<Range<K>, V>;

#[derive(Debug)]
enum PendingValue<V> {
    Committed(Arc<V>),
    Pending(OwnedRwLockReadGuard<V>),
}

struct State<I, K, V> {
    canon: Canon<K, V>,
    commits: OrdHashSet<I>,
    deltas: Deltas<I, K, V>,
    pending: OrdHashMap<I, Pending<K, V>>,
    finalized: Option<I>,
}

impl<I: Ord + Hash + fmt::Debug, K, V> State<I, K, V> {
    #[inline]
    fn new(txn_id: I, version: Pending<K, V>) -> Self {
        Self {
            canon: Canon::new(),
            commits: OrdHashSet::new(),
            deltas: OrdHashMap::new(),
            pending: OrdHashMap::from_iter(iter::once((txn_id, version))),
            finalized: None,
        }
    }
}

impl<I, K, V> State<I, K, V>
where
    I: Copy + Hash + Ord + fmt::Debug,
    K: Hash + Ord,
    V: fmt::Debug,
{
    #[inline]
    fn check_committed(&self, txn_id: &I) -> Result<bool> {
        match self.finalized.as_ref().cmp(&Some(txn_id)) {
            Ordering::Greater => Err(Error::Outdated),
            Ordering::Equal => Ok(true),
            Ordering::Less => Ok(self.commits.contains(txn_id)),
        }
    }

    #[inline]
    fn check_pending(&self, txn_id: &I) -> Result<()> {
        if self.finalized.as_ref() > Some(txn_id) {
            Err(Error::Outdated)
        } else if self.commits.contains(txn_id) {
            Err(Error::Committed)
        } else {
            Ok(())
        }
    }

    #[inline]
    fn canon(&self, txn_id: &I) -> Canon<K, V> {
        let mut canon = self.canon.clone();

        let deltas = self
            .deltas
            .iter()
            .take_while(|(id, _)| *id <= txn_id)
            .map(|(_id, version)| version);

        for version in deltas {
            for (key, delta) in version {
                if let Some(value) = delta {
                    canon.insert(key.clone(), value.clone());
                } else {
                    canon.remove(key);
                }
            }
        }

        canon
    }

    #[inline]
    fn clear(&mut self, txn_id: I) -> Canon<K, V> {
        let mut map = self.canon(&txn_id);

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
    fn commit_version(&mut self, txn_id: &I) -> Option<Delta<K, V>> {
        self.pending.remove(txn_id).map(|version| {
            version
                .into_iter()
                .map(|(key, delta)| {
                    let value = if let Some(present) = delta {
                        if let Ok(value) = Arc::try_unwrap(present) {
                            Some(Arc::new(value.into_inner()))
                        } else {
                            panic!("a value at {:?} is still locked", txn_id);
                        }
                    } else {
                        None
                    };

                    (key, value)
                })
                .collect()
        })
    }

    #[inline]
    fn commit(&mut self, txn_id: I) {
        if self.commits.contains(&txn_id) {
            #[cfg(feature = "logging")]
            log::warn!("duplicate commit at {:?}", txn_id);
        } else if let Some(version) = self.commit_version(&txn_id) {
            self.deltas.insert(txn_id, version);
        }

        self.commits.insert(txn_id);
    }

    #[inline]
    fn contains_canon<Q>(&self, txn_id: &I, key: &Q) -> bool
    where
        Q: Eq + Hash + ?Sized,
        Key<K>: Borrow<Q>,
    {
        contains_canon(&self.canon, &self.deltas, txn_id, key)
    }

    #[inline]
    fn contains_committed<Q>(&self, txn_id: &I, key: &Q) -> Poll<Result<bool>>
    where
        Q: Eq + Hash + ?Sized,
        Key<K>: Borrow<Q>,
    {
        match self.finalized.as_ref().cmp(&Some(txn_id)) {
            Ordering::Greater => Poll::Ready(Err(Error::Outdated)),
            Ordering::Equal => Poll::Ready(Ok(self.contains_canon(txn_id, key))),
            Ordering::Less => {
                if self.commits.contains(txn_id) {
                    Poll::Ready(Ok(self.contains_canon(txn_id, key)))
                } else {
                    Poll::Pending
                }
            }
        }
    }

    #[inline]
    fn contains_pending<Q>(&self, txn_id: &I, key: &Q) -> bool
    where
        Q: Eq + Hash + ?Sized,
        Key<K>: Borrow<Q>,
    {
        if let Some(delta) = self.pending.get(txn_id) {
            if let Some(key_state) = delta.get(key) {
                return key_state.is_some();
            }
        }

        self.contains_canon(txn_id, key)
    }

    #[inline]
    fn extend<Q, E>(&mut self, txn_id: I, other: E)
    where
        Q: Into<Key<K>>,
        E: IntoIterator<Item = (Q, V)>,
    {
        let entries = other
            .into_iter()
            .map(|(key, value)| (key.into(), Some(Arc::new(RwLock::new(value)))));

        if let Some(version) = self.pending.get_mut(&txn_id) {
            version.extend(entries);
        } else {
            self.pending.insert(txn_id, entries.collect());
        }
    }

    #[inline]
    fn finalize(&mut self, txn_id: I) -> Option<&Canon<K, V>> {
        if self.finalized > Some(txn_id) {
            return None;
        }

        while let Some(version_id) = self.pending.keys().next().copied() {
            if version_id <= txn_id {
                self.pending.pop_first();
            } else {
                break;
            }
        }

        while let Some(version_id) = self.commits.first().copied() {
            if version_id <= txn_id {
                self.commits.pop_first();
            } else {
                break;
            }
        }

        while let Some(version_id) = self.deltas.keys().next().copied() {
            if version_id <= txn_id {
                let version = self.deltas.pop_first().expect("version");
                merge_owned(&mut self.canon, version);
            } else {
                break;
            }
        }

        self.finalized = Some(txn_id);

        Some(&self.canon)
    }

    #[inline]
    fn get_canon<Q>(&self, txn_id: &I, key: &Q) -> Option<Arc<V>>
    where
        Q: Hash + Eq + ?Sized,
        Key<K>: Borrow<Q>,
    {
        get_canon(&self.canon, &self.deltas, txn_id, key).cloned()
    }

    #[inline]
    fn get_committed<Q>(
        &self,
        txn_id: &I,
        key: &Q,
    ) -> Poll<Result<Option<TxnMapValueReadGuard<K, V>>>>
    where
        Q: Hash + Eq + ?Sized,
        Key<K>: Borrow<Q>,
    {
        if self.finalized.as_ref() > Some(txn_id) {
            Poll::Ready(Err(Error::Outdated))
        } else if self.commits.contains(txn_id) {
            debug_assert!(!self.pending.contains_key(txn_id));
            let value = self.get_canon(txn_id, key);
            Poll::Ready(Ok(value.map(TxnMapValueReadGuard::committed)))
        } else {
            Poll::Pending
        }
    }

    #[inline]
    fn get_pending<Q>(&self, txn_id: &I, key: &Q) -> Option<PendingValue<V>>
    where
        Q: Eq + Hash + ?Sized,
        Key<K>: Borrow<Q>,
    {
        if let Some(version) = self.pending.get(txn_id) {
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
    fn insert(&mut self, txn_id: I, key: Key<K>, value: V) -> Option<Arc<V>> {
        let value = Arc::new(RwLock::new(value));

        if let Some(deltas) = self.pending.get_mut(&txn_id) {
            match deltas.entry(key) {
                hash_map::Entry::Occupied(mut delta) => {
                    if let Some(prior) = delta.insert(Some(value)) {
                        let lock = Arc::try_unwrap(prior).expect("prior value");
                        let prior_value = Arc::new(lock.into_inner());
                        Some(prior_value)
                    } else {
                        get_canon(&self.canon, &self.deltas, &txn_id, delta.key()).cloned()
                    }
                }
                hash_map::Entry::Vacant(delta) => {
                    let prior = get_canon(&self.canon, &self.deltas, &txn_id, delta.key()).cloned();

                    delta.insert(Some(value));
                    prior
                }
            }
        } else {
            let prior = get_canon(&self.canon, &self.deltas, &txn_id, &key).cloned();

            let pending = iter::once((key, Some(value))).collect();
            self.pending.insert(txn_id, pending);

            prior
        }
    }

    #[inline]
    fn key<Q>(&self, txn_id: &I, key: &Q) -> Option<&Key<K>>
    where
        Q: Eq + Hash + ?Sized,
        Key<K>: Borrow<Q>,
    {
        if let Some((key, _)) = self.canon.get_key_value(key) {
            Some(key)
        } else if let Some(deltas) = self.pending.get(txn_id) {
            deltas.get_key_value(key).map(|(key, _delta)| key)
        } else {
            None
        }
    }

    #[inline]
    fn keys_committed(&self, txn_id: &I) -> HashSet<Key<K>> {
        let mut keys = self.canon.keys().cloned().collect();

        let deltas = self
            .deltas
            .iter()
            .take_while(|(id, _)| *id <= txn_id)
            .map(|(_, version)| version);

        for version in deltas {
            merge_keys(&mut keys, version);
        }

        keys
    }

    #[inline]
    fn keys_pending(&self, txn_id: I) -> HashSet<Key<K>> {
        let mut keys = self.keys_committed(&txn_id);

        if let Some(pending) = self.pending.get(&txn_id) {
            merge_keys(&mut keys, pending);
        }

        keys
    }

    #[inline]
    fn remove(&mut self, txn_id: I, key: Key<K>) -> Option<Arc<V>> {
        if let Some(pending) = self.pending.get_mut(&txn_id) {
            match pending.entry(key) {
                hash_map::Entry::Occupied(mut entry) => {
                    if let Some(lock) = entry.insert(None) {
                        let lock = Arc::try_unwrap(lock).expect("removed value");
                        Some(Arc::new(lock.into_inner()))
                    } else {
                        None
                    }
                }
                hash_map::Entry::Vacant(entry) => {
                    if let Some(prior) =
                        get_canon(&self.canon, &self.deltas, &txn_id, entry.key()).cloned()
                    {
                        entry.insert(None);
                        Some(prior)
                    } else {
                        None
                    }
                }
            }
        } else if let Some(prior) = get_canon(&self.canon, &self.deltas, &txn_id, &key).cloned() {
            let pending = iter::once((key, None)).collect();
            self.pending.insert(txn_id, pending);
            Some(prior)
        } else {
            None
        }
    }
}

#[inline]
fn contains_canon<I, K, V, Q>(
    canon: &Canon<K, V>,
    deltas: &OrdHashMap<I, Delta<K, V>>,
    txn_id: &I,
    key: &Q,
) -> bool
where
    I: Hash + Ord + fmt::Debug,
    K: Hash + Ord,
    Q: Eq + Hash + ?Sized,
    Key<K>: Borrow<Q>,
{
    let deltas = deltas
        .iter()
        .rev()
        .skip_while(|(id, _)| *id > txn_id)
        .map(|(_, version)| version);

    for delta in deltas {
        if let Some(key_state) = delta.get(key) {
            return key_state.is_some();
        }
    }

    canon.contains_key(key)
}

impl<I, K, V> State<I, K, V>
where
    I: Copy + Hash + Ord + fmt::Debug,
    K: Hash + Ord,
    V: Clone + fmt::Debug,
{
    #[inline]
    fn get_mut(&mut self, txn_id: I, key: Key<K>) -> Option<OwnedRwLockWriteGuard<V>> {
        #[inline]
        fn new_value<V: Clone>(canon: &V) -> (Arc<RwLock<V>>, OwnedRwLockWriteGuard<V>) {
            let value = V::clone(canon);
            let value = Arc::new(RwLock::new(value));
            let guard = value.clone().try_write_owned().expect("write version");
            (value, guard)
        }

        if let Some(pending) = self.pending.get_mut(&txn_id) {
            match pending.entry(key) {
                hash_map::Entry::Occupied(delta) => {
                    let value = delta.get().as_ref()?;

                    value
                        .clone()
                        .try_write_owned()
                        .map(Some)
                        .expect("write version")
                }
                hash_map::Entry::Vacant(delta) => {
                    let canon = get_canon(&self.canon, &self.deltas, &txn_id, delta.key())?;
                    let (value, guard) = new_value(&**canon);

                    delta.insert(Some(value));

                    Some(guard)
                }
            }
        } else {
            let canon = get_canon(&self.canon, &self.deltas, &txn_id, &key)?;
            let (value, guard) = new_value(&**canon);

            let version = iter::once((key, Some(value))).collect();
            self.pending.insert(txn_id, version);

            Some(guard)
        }
    }

    #[inline]
    fn insert_new(&mut self, txn_id: I, key: Key<K>, value: V) -> OwnedRwLockWriteGuard<V> {
        let value = Arc::new(RwLock::new(value));
        let guard = value.clone().try_write_owned().expect("value");

        if let Some(version) = self.pending.get_mut(&txn_id) {
            assert!(version.insert(key, Some(value)).is_none());
        } else {
            let version = iter::once((key, Some(value))).collect();
            self.pending.insert(txn_id, version);
        }

        guard
    }
}

#[inline]
fn get_canon<'a, I, K, V, Q>(
    canon: &'a Canon<K, V>,
    deltas: &'a Deltas<I, K, V>,
    txn_id: &'a I,
    key: &'a Q,
) -> Option<&'a Arc<V>>
where
    I: Hash + Ord + fmt::Debug,
    K: Hash + Ord,
    Q: Eq + Hash + ?Sized,
    Key<K>: Borrow<Q>,
{
    let committed = deltas
        .iter()
        .rev()
        .skip_while(|(id, _)| *id > txn_id)
        .map(|(_, version)| version);

    for version in committed {
        if let Some(delta) = version.get(key) {
            return delta.as_ref();
        }
    }

    canon.get(key)
}

#[inline]
fn merge_keys<K: Hash + Ord, V>(keys: &mut HashSet<Key<K>>, deltas: &Delta<K, V>) {
    for (key, delta) in deltas {
        if delta.is_some() {
            keys.insert(key.clone());
        } else {
            keys.remove(key);
        }
    }
}

/// An occupied entry in a [`TxnMapLock`]
pub struct EntryOccupied<K, V> {
    key: Key<K>,
    value: TxnMapValueWriteGuard<K, V>,
}

impl<K, V> EntryOccupied<K, V> {
    /// Borrow this entry's value.
    pub fn get(&self) -> &V {
        self.value.deref()
    }

    /// Borrow this entry's value mutably.
    pub fn get_mut(&mut self) -> &mut V {
        self.value.deref_mut()
    }

    /// Borrow this entry's key.
    pub fn key(&self) -> &K {
        &*self.key
    }
}

/// A vacant entry in a [`TxnMapLock`]
pub struct EntryVacant<I, K, V> {
    permit: PermitWrite<Range<K>>,
    txn_id: I,
    key: Key<K>,
    map_state: Arc<RwLockInner<State<I, K, V>>>,
}

impl<I, K, V> EntryVacant<I, K, V>
where
    I: Hash + Ord + Copy + fmt::Debug,
    K: Hash + Ord + Clone + fmt::Debug,
    V: Clone + fmt::Debug,
{
    /// Insert a new value at this [`Entry`].
    pub fn insert(self, value: V) -> TxnMapValueWriteGuard<K, V> {
        let mut map_state = self.map_state.write().expect("lock state");
        let value = map_state.insert_new(self.txn_id, self.key, value);
        TxnMapValueWriteGuard::new(self.permit, value)
    }

    /// Borrow this entry's key.
    pub fn key(&self) -> &K {
        &*self.key
    }
}

/// An entry in a [`TxnMapLock`]
pub enum Entry<I, K, V> {
    Occupied(EntryOccupied<K, V>),
    Vacant(EntryVacant<I, K, V>),
}

impl<I, K, V> Entry<I, K, V> {
    pub fn key(&self) -> &K {
        todo!()
    }
}

/// A futures-aware read-write lock on a [`HashMap`] which supports transactional versioning
///
/// The `get_mut` and `try_get_mut` methods require the value type `V` to implement [`Clone`]
/// in order to support multiple transactions with different versions.
pub struct TxnMapLock<I, K, V> {
    state: Arc<RwLockInner<State<I, K, V>>>,
    semaphore: Semaphore<I, Collator<K>, Range<K>>,
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

impl<I: Ord + Hash + fmt::Debug, K, V> TxnMapLock<I, K, V> {
    /// Construct a new [`TxnMapLock`].
    pub fn new(txn_id: I) -> Self {
        let collator = Collator::<K>::default();

        Self {
            state: Arc::new(RwLockInner::new(State::new(txn_id, Pending::new()))),
            semaphore: Semaphore::new(collator),
        }
    }
}

impl<I, K, V> TxnMapLock<I, K, V>
where
    I: Copy + Hash + Ord + fmt::Debug,
    K: Eq + Hash + Ord + fmt::Debug + Send + Sync,
    V: fmt::Debug,
{
    /// Construct a new [`TxnMapLock`] with the given `contents`.
    pub fn with_contents<KV: IntoIterator<Item = (K, V)>>(txn_id: I, contents: KV) -> Self {
        let version = contents
            .into_iter()
            .map(|(key, value)| (Key::new(key), Some(Arc::new(RwLock::new(value)))))
            .collect();

        let collator = Collator::<K>::default();

        Self {
            state: Arc::new(RwLockInner::new(State::new(txn_id, version))),
            semaphore: Semaphore::with_reservation(txn_id, collator, Range::All),
        }
    }

    /// Remove and return all entries from this [`TxnMapLock`] at `txn_id`.
    pub async fn clear(&self, txn_id: I) -> Result<Canon<K, V>> {
        #[cfg(feature = "logging")]
        log::trace!("clear {self:?} at {txn_id:?}");

        let _permit = self.semaphore.write(txn_id, Range::All).await?;

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        Ok(state.clear(txn_id))
    }

    /// Remove and return all entries from this [`TxnMapLock`] at `txn_id`.
    pub fn try_clear(&self, txn_id: I) -> Result<Canon<K, V>> {
        #[cfg(feature = "logging")]
        log::trace!("try_clear {self:?} at {txn_id:?}");

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        let _permit = self.semaphore.try_write(txn_id, Range::All)?;

        Ok(state.clear(txn_id))
    }

    /// Return `true` if this [`TxnMapLock`] has an entry at `key` at `txn_id`.
    pub async fn contains_key<Q>(&self, txn_id: I, key: &Q) -> Result<bool>
    where
        Q: Eq + Hash + ToOwned<Owned = K> + ?Sized,
        Key<K>: Borrow<Q>,
    {
        #[cfg(feature = "logging")]
        log::trace!("contains_key? {self:?} at {txn_id:?}");

        let range: Range<K> = {
            let state = self.state();
            if let Poll::Ready(result) = state.contains_committed(&txn_id, key) {
                return result;
            } else {
                Key::<K>::from((key, state.key(&txn_id, key))).into()
            }
        };

        let _permit = self.semaphore.read(txn_id, range).await?;
        Ok(self.state().contains_pending(&txn_id, key))
    }

    /// Synchronously check whether the given `key` is present in this [`TxnMapLock`]s, if possible.
    pub fn try_contains_key<Q>(&self, txn_id: I, key: &Q) -> Result<bool>
    where
        Q: Eq + Hash + ToOwned<Owned = K> + ?Sized,
        Key<K>: Borrow<Q>,
    {
        #[cfg(feature = "logging")]
        log::trace!("try_contains_key {self:?} at {txn_id:?}");

        let state = self.state();
        if let Poll::Ready(result) = state.contains_committed(&txn_id, key) {
            return result;
        }

        let range = Key::<K>::from((key, state.key(&txn_id, key))).into();
        let _permit = self.semaphore.try_read(txn_id, range)?;
        Ok(state.contains_pending(&txn_id, key))
    }

    /// Insert the entries from `other` [`TxnMapLock`] at `txn_id`.
    pub async fn extend<Q, E>(&self, txn_id: I, other: E) -> Result<()>
    where
        Q: Into<Key<K>>,
        E: IntoIterator<Item = (Q, V)>,
    {
        #[cfg(feature = "logging")]
        log::trace!("extend {self:?} at {txn_id:?}");

        let _permit = self.semaphore.write(txn_id, Range::All).await?;

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        Ok(state.extend(txn_id, other))
    }

    /// Insert the entries from `other` [`TxnMapLock`] at `txn_id` synchronously, if possible.
    pub fn try_extend<Q, E>(&self, txn_id: I, other: E) -> Result<()>
    where
        Q: Into<Key<K>>,
        E: IntoIterator<Item = (Q, V)>,
    {
        #[cfg(feature = "logging")]
        log::trace!("try_extend {self:?} at {txn_id:?}");

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        let _permit = self.semaphore.try_write(txn_id, Range::All)?;

        Ok(state.extend(txn_id, other))
    }

    /// Read a value from this [`TxnMapLock`] at `txn_id`.
    pub async fn get<Q>(&self, txn_id: I, key: &Q) -> Result<Option<TxnMapValueReadGuard<K, V>>>
    where
        Q: Eq + Hash + ToOwned<Owned = K> + ?Sized,
        Key<K>: Borrow<Q>,
    {
        #[cfg(feature = "logging")]
        log::trace!("get from {self:?} at {txn_id:?}");

        let range: Range<K> = {
            let state = self.state();

            if let Poll::Ready(result) = state.get_committed(&txn_id, key) {
                return result;
            }

            Key::<K>::from((key, state.key(&txn_id, key))).into()
        };

        let permit = self.semaphore.read(txn_id, range).await?;

        let state = self.state();

        if let Poll::Ready(result) = state.get_committed(&txn_id, key) {
            return result;
        }

        let value = state.get_pending(&txn_id, key).map(|value| match value {
            PendingValue::Committed(value) => TxnMapValueReadGuard::committed(value),
            PendingValue::Pending(value) => TxnMapValueReadGuard::pending_write(permit, value),
        });

        Ok(value)
    }

    /// Read a value from this [`TxnMapLock`] at `txn_id` synchronously, if possible.
    pub fn try_get<Q>(&self, txn_id: I, key: &Q) -> Result<Option<TxnMapValueReadGuard<K, V>>>
    where
        Q: Eq + Hash + ToOwned<Owned = K> + ?Sized,
        Key<K>: Borrow<Q>,
    {
        #[cfg(feature = "logging")]
        log::trace!("try_get from {self:?} at {txn_id:?}");

        let state = self.state();

        if let Poll::Ready(result) = state.get_committed(&txn_id, key) {
            #[cfg(feature = "log")]
            log::trace!(
                "{self:?} is already committed at {txn_id:?}, no need for a read permit..."
            );
            return result;
        } else {
            #[cfg(feature = "log")]
            log::trace!("{self:?} is not yet committed at {txn_id:?}, getting a read permit...");
        }

        let range = Key::<K>::from((key, state.key(&txn_id, key))).into();
        let permit = self.semaphore.try_read(txn_id, range)?;
        let value = state.get_pending(&txn_id, key).map(|value| match value {
            PendingValue::Committed(value) => TxnMapValueReadGuard::committed(value),
            PendingValue::Pending(value) => TxnMapValueReadGuard::pending_write(permit, value),
        });

        Ok(value)
    }

    /// Construct an iterator over the entries in this [`TxnMapLock`] at `txn_id`.
    pub async fn iter(&self, txn_id: I) -> Result<Iter<I, K, V>> {
        #[cfg(feature = "logging")]
        log::trace!("iter over {self:?} at {txn_id:?}");

        let permit = self.semaphore.read(txn_id, Range::All).await?;

        let state = self.state();

        if state.check_committed(&txn_id)? {
            let keys = state.keys_committed(&txn_id);
            Ok(Iter::new(self.state.clone(), txn_id, None, keys))
        } else {
            let keys = self.state().keys_pending(txn_id);
            Ok(Iter::new(self.state.clone(), txn_id, Some(permit), keys))
        }
    }

    /// Construct an iterator over the entries in this [`TxnMapLock`] at `txn_id` synchronously.
    pub fn try_iter(&self, txn_id: I) -> Result<Iter<I, K, V>> {
        #[cfg(feature = "logging")]
        log::trace!("try_iter over {self:?} at {txn_id:?}");

        let state = self.state();

        if state.check_committed(&txn_id)? {
            let keys = state.keys_committed(&txn_id);
            Ok(Iter::new(self.state.clone(), txn_id, None, keys))
        } else {
            let permit = self.semaphore.try_read(txn_id, Range::All)?;
            let keys = state.keys_pending(txn_id);
            Ok(Iter::new(self.state.clone(), txn_id, Some(permit), keys))
        }
    }

    /// Insert a new entry into this [`TxnMapLock`] at `txn_id` and return the prior value, if any.
    pub async fn insert<Q: Into<Key<K>>>(
        &self,
        txn_id: I,
        key: Q,
        value: V,
    ) -> Result<Option<Arc<V>>> {
        #[cfg(feature = "logging")]
        log::trace!("insert into {self:?} at {txn_id:?}");

        let key: Key<K> = key.into();
        let _permit = self.semaphore.write(txn_id, key.clone().into()).await?;

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        Ok(state.insert(txn_id, key, value))
    }

    /// Insert a new entry into this [`TxnMapLock`] at `txn_id` synchronously, if possible.
    pub fn try_insert<Q: Into<Key<K>>>(
        &self,
        txn_id: I,
        key: Q,
        value: V,
    ) -> Result<Option<Arc<V>>> {
        #[cfg(feature = "logging")]
        log::trace!("try_insert into {self:?} at {txn_id:?}");

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        let key = key.into();
        let _permit = self.semaphore.try_write(txn_id, key.clone().into())?;

        Ok(state.insert(txn_id, key, value))
    }

    /// Return `true` if this [`TxnMapLock`] is empty at the given `txn_id`.
    pub async fn is_empty(&self, txn_id: I) -> Result<bool> {
        #[cfg(feature = "logging")]
        log::trace!("is_empty? {self:?} at {txn_id:?}");

        self.len(txn_id).map_ok(|len| len == 0).await
    }

    /// Get the size of this [`TxnMapLock`] at the given `txn_id`.
    pub async fn len(&self, txn_id: I) -> Result<usize> {
        #[cfg(feature = "logging")]
        log::trace!("len of {self:?} at {txn_id:?}");

        let _permit = self.semaphore.read(txn_id, Range::All).await?;

        let state = self.state();

        if state.check_committed(&txn_id)? {
            let keys = state.keys_committed(&txn_id);
            Ok(keys.len())
        } else {
            let keys = state.keys_pending(txn_id);
            Ok(keys.len())
        }
    }

    /// Remove and return the value at `key` from this [`TxnMapLock`] at `txn_id`, if present.
    pub async fn remove<Q>(&self, txn_id: I, key: &Q) -> Result<Option<Arc<V>>>
    where
        Q: Eq + Hash + ToOwned<Owned = K> + ?Sized,
        Key<K>: Borrow<Q>,
    {
        #[cfg(feature = "logging")]
        log::trace!("remove from {self:?} at {txn_id:?}");

        let key: Key<K> = {
            let state = self.state();
            state.check_pending(&txn_id)?;
            (key, state.key(&txn_id, key)).into()
        };

        let _permit = self.semaphore.write(txn_id, key.clone().into()).await?;

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        Ok(state.remove(txn_id, key))
    }

    /// Remove and return the value at `key` from this [`TxnMapLock`] at `txn_id`, if present.
    pub fn try_remove<Q>(&self, txn_id: I, key: &Q) -> Result<Option<Arc<V>>>
    where
        Q: Eq + Hash + ToOwned<Owned = K> + ?Sized,
        Key<K>: Borrow<Q>,
    {
        #[cfg(feature = "logging")]
        log::trace!("try_remove from {self:?} at {txn_id:?}");

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        let key: Key<K> = (key, state.key(&txn_id, key)).into();
        let _permit = self.semaphore.try_write(txn_id, key.clone().into())?;

        Ok(state.remove(txn_id, key))
    }
}

impl<I, K, V> TxnMapLock<I, K, V>
where
    I: Copy + Hash + Ord + fmt::Debug,
    K: Hash + Ord + fmt::Debug + Send + Sync,
    V: Clone + fmt::Debug,
{
    /// Borrow an [`Entry`] mutably for writing at `txn_id`.
    pub async fn entry<Q: Into<Key<K>>>(&self, txn_id: I, key: Q) -> Result<Entry<I, K, V>> {
        #[cfg(feature = "logging")]
        log::trace!("get entry in {self:?} at {txn_id:?}");

        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let key: Key<K> = key.into();
        let range = key.clone().into();
        let permit = self.semaphore.write(txn_id, range).await?;

        if let Some(value) = self.state_mut().get_mut(txn_id, key.clone()) {
            Ok(Entry::Occupied(EntryOccupied {
                key,
                value: TxnMapValueWriteGuard::new(permit, value),
            }))
        } else {
            Ok(Entry::Vacant(EntryVacant {
                permit,
                key,
                txn_id,
                map_state: self.state.clone(),
            }))
        }
    }

    /// Read a mutable value from this [`TxnMapLock`] at `txn_id`.
    pub async fn get_mut<Q: Into<Key<K>>>(
        &self,
        txn_id: I,
        key: Q,
    ) -> Result<Option<TxnMapValueWriteGuard<K, V>>> {
        #[cfg(feature = "logging")]
        log::trace!("get_mut {self:?} at {txn_id:?}");

        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let key = key.into();
        let permit = self.semaphore.write(txn_id, key.clone().into()).await?;

        if let Some(value) = self.state_mut().get_mut(txn_id, key) {
            Ok(Some(TxnMapValueWriteGuard::new(permit, value)))
        } else {
            Ok(None)
        }
    }

    /// Read a mutable value from this [`TxnMapLock`] at `txn_id`.
    pub fn try_get_mut<Q>(&self, txn_id: I, key: &Q) -> Result<Option<TxnMapValueWriteGuard<K, V>>>
    where
        Q: Eq + Hash + ToOwned<Owned = K> + ?Sized,
        Key<K>: Borrow<Q>,
    {
        #[cfg(feature = "logging")]
        log::trace!("try_get_mut {self:?} at {txn_id:?}");

        let mut state = self.state_mut();

        // before acquiring a permit, check if this version has already been committed
        state.check_pending(&txn_id)?;

        let maybe_key = state.key(&txn_id, key);
        let key = Key::<K>::from((key, maybe_key));
        let permit = self.semaphore.try_write(txn_id, key.clone().into())?;

        if let Some(value) = state.get_mut(txn_id, key) {
            Ok(Some(TxnMapValueWriteGuard::new(permit, value)))
        } else {
            Ok(None)
        }
    }

    /// Construct a mutable iterator over the entries in this [`TxnMapLock`] at `txn_id`.
    pub async fn iter_mut(&self, txn_id: I) -> Result<IterMut<I, K, V>> {
        #[cfg(feature = "logging")]
        log::trace!("iter_mut {self:?} at {txn_id:?}");

        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let permit = self.semaphore.write(txn_id, Range::All).await?;
        let keys = self.state().keys_pending(txn_id);
        Ok(IterMut::new(self.state.clone(), txn_id, permit, keys))
    }

    /// Construct a mutable iterator over the entries in this [`TxnMapLock`] at `txn_id`,
    /// synchronously if possible.
    pub fn try_iter_mut(&self, txn_id: I) -> Result<IterMut<I, K, V>> {
        #[cfg(feature = "logging")]
        log::trace!("try_iter_mut {self:?} at {txn_id:?}");

        let state = self.state();

        // before acquiring a permit, check if this version has already been committed
        state.check_pending(&txn_id)?;

        let permit = self.semaphore.try_write(txn_id, Range::All)?;
        let keys = state.keys_pending(txn_id);
        Ok(IterMut::new(self.state.clone(), txn_id, permit, keys))
    }
}

impl<I, K, V> TxnMapLock<I, K, V>
where
    I: Copy + Hash + Ord + fmt::Debug,
    K: Eq + Hash + Ord + fmt::Debug + Send + Sync,
    V: fmt::Debug,
{
    /// Commit the state of this [`TxnMapLock`] at `txn_id`.
    /// Panics:
    ///  - if this [`TxnMapLock`] has already been finalized at `txn_id`
    ///  - if any new value to commit is still locked (for reading or writing)
    pub fn commit(&self, txn_id: I) {
        #[cfg(feature = "logging")]
        log::trace!("commit {self:?} at {txn_id:?}");

        let mut state = self.state_mut();

        if state.finalized.as_ref() >= Some(&txn_id) {
            panic!("tried to commit already-finalized version {:?}", txn_id);
        }

        state.commit(txn_id);

        self.semaphore.finalize(&txn_id, false);
    }

    /// Read and commit this [`TxnMapLock`] in a single operation.
    /// Also returns the set of changes committed, if any.
    ///
    /// Panics:
    ///  - if this [`TxnMapLock`] has already been finalized at `txn_id`
    ///  - if any new value to commit is still locked (for reading or writing)
    pub async fn read_and_commit(&self, txn_id: I) -> (Canon<K, V>, Option<Delta<K, V>>) {
        #[cfg(feature = "logging")]
        log::trace!("read and commit {self:?} at {txn_id:?}");

        let _permit = self
            .semaphore
            .read(txn_id, Range::All)
            .await
            .expect("permit");

        let (version, deltas) = {
            let mut state = self.state_mut();

            if state.finalized > Some(txn_id) {
                panic!("tried to commit already-finalized version {:?}", txn_id);
            }

            state.commit(txn_id);

            (state.canon(&txn_id), state.deltas.get(&txn_id).cloned())
        };

        self.semaphore.finalize(&txn_id, false);

        (version, deltas)
    }

    /// Roll back the state of this [`TxnMapLock`] at `txn_id`.
    ///
    /// Panics: if this [`TxnMapLock`] has already been committed or finalized at `txn_id`
    pub fn rollback(&self, txn_id: &I) {
        self.semaphore.finalize(txn_id, false);

        let mut state = self.state_mut();

        assert!(
            !state.commits.contains(txn_id),
            "cannot roll back committed transaction {:?}",
            txn_id
        );

        if state.finalized.as_ref() > Some(txn_id) {
            panic!("tried to roll back finalized version at {:?}", txn_id);
        }

        state.pending.remove(txn_id);

        self.semaphore.finalize(txn_id, false);
    }

    /// Read and roll back this [`TxnMapLock`] in a single operation.
    ///
    /// Panics:
    ///  - if this [`TxnMapLock`] has already been committed or finalized at `txn_id`
    ///  - if any updated value is still locked for reading or writing
    pub async fn read_and_rollback(&self, txn_id: I) -> (Canon<K, V>, Option<Delta<K, V>>) {
        let _permit = self
            .semaphore
            .read(txn_id, Range::All)
            .await
            .expect("permit");

        let (version, deltas) = {
            let mut state = self.state_mut();

            assert!(
                !state.commits.contains(&txn_id),
                "cannot roll back committed transaction {:?}",
                txn_id
            );

            if state.finalized > Some(txn_id) {
                panic!("tried to roll back finalized version at {:?}", txn_id);
            }

            // note: this removes the pending version, i.e. it returns the version that WOULD be committed
            let deltas = state.commit_version(&txn_id);

            let mut version = state.canon(&txn_id);

            if let Some(deltas) = &deltas {
                merge(&mut version, deltas);
            }

            (version, deltas)
        };

        self.semaphore.finalize(&txn_id, false);

        (version, deltas)
    }

    /// Finalize the state of this [`TxnMapLock`] at `txn_id`.
    /// This will finalize commits and prevent further reads of all versions earlier than `txn_id`.
    pub fn finalize(&self, txn_id: I) {
        self.semaphore.finalize(&txn_id, true);
        self.state_mut().finalize(txn_id);
    }

    /// Read and finalize the state of this [`TxnMapLock`] at `txn_id`, if not already finalized.
    /// This will finalize commits and prevent further reads of all versions earlier than `txn_id`.
    pub fn read_and_finalize(&self, txn_id: I) -> Option<Canon<K, V>> {
        self.semaphore.finalize(&txn_id, true);
        self.state_mut().finalize(txn_id).cloned()
    }
}

impl<I, K, V> fmt::Debug for TxnMapLock<I, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a transactional lock on a map of keys to values")
    }
}

/// A guard on a value in an [`Iter`]
#[derive(Debug)]
pub struct TxnMapLockIterGuard<V> {
    value: PendingValue<V>,
}

impl<V> From<PendingValue<V>> for TxnMapLockIterGuard<V> {
    fn from(value: PendingValue<V>) -> Self {
        Self { value }
    }
}

impl<V> Deref for TxnMapLockIterGuard<V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        match &self.value {
            PendingValue::Committed(value) => value.deref(),
            PendingValue::Pending(value) => value.deref(),
        }
    }
}

impl<V: PartialEq> PartialEq<V> for TxnMapLockIterGuard<V> {
    fn eq(&self, other: &V) -> bool {
        self.deref().eq(other)
    }
}

impl<V: PartialOrd> PartialOrd<V> for TxnMapLockIterGuard<V> {
    fn partial_cmp(&self, other: &V) -> Option<Ordering> {
        self.deref().partial_cmp(other)
    }
}

/// An iterator over the entries in a [`TxnMapLock`] as of a specific transaction
pub struct Iter<I, K, V> {
    lock_state: Arc<RwLockInner<State<I, K, V>>>,
    txn_id: I,
    permit: Option<PermitRead<Range<K>>>,
    keys: <HashSet<Key<K>> as IntoIterator>::IntoIter,
}

impl<I, K, V> Iter<I, K, V> {
    fn new(
        lock_state: Arc<RwLockInner<State<I, K, V>>>,
        txn_id: I,
        permit: Option<PermitRead<Range<K>>>,
        keys: HashSet<Key<K>>,
    ) -> Self {
        Self {
            lock_state,
            txn_id,
            permit,
            keys: keys.into_iter(),
        }
    }
}

impl<I, K, V> Iterator for Iter<I, K, V>
where
    I: Copy + Hash + Ord + fmt::Debug,
    K: Hash + Ord,
    V: fmt::Debug,
{
    type Item = (Key<K>, TxnMapLockIterGuard<V>);

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

#[inline]
fn get_key<I, K, V, Q>(
    state: &State<I, K, V>,
    txn_id: &I,
    key: &Q,
    committed: bool,
) -> Option<PendingValue<V>>
where
    I: Copy + Hash + Ord + fmt::Debug,
    K: Hash + Ord,
    V: fmt::Debug,
    Q: Eq + Hash + ?Sized,
    Key<K>: Borrow<Q>,
{
    if committed {
        state.get_canon(txn_id, key).map(PendingValue::Committed)
    } else {
        state.get_pending(txn_id, key)
    }
}

/// A guard on a mutable value in an [`IterMut`]
#[derive(Debug)]
pub struct TxnMapLockIterMutGuard<V> {
    guard: OwnedRwLockWriteGuard<V>,
}

impl<V> Deref for TxnMapLockIterMutGuard<V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

impl<V> DerefMut for TxnMapLockIterMutGuard<V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard.deref_mut()
    }
}

impl<V: PartialEq> PartialEq<V> for TxnMapLockIterMutGuard<V> {
    fn eq(&self, other: &V) -> bool {
        self.deref().eq(other)
    }
}

impl<V: PartialOrd> PartialOrd<V> for TxnMapLockIterMutGuard<V> {
    fn partial_cmp(&self, other: &V) -> Option<Ordering> {
        self.deref().partial_cmp(other)
    }
}

impl<V> From<OwnedRwLockWriteGuard<V>> for TxnMapLockIterMutGuard<V> {
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
    keys: <HashSet<Key<K>> as IntoIterator>::IntoIter,
}

impl<I, K, V> IterMut<I, K, V> {
    fn new(
        lock_state: Arc<RwLockInner<State<I, K, V>>>,
        txn_id: I,
        permit: PermitWrite<Range<K>>,
        keys: HashSet<Key<K>>,
    ) -> Self {
        Self {
            lock_state,
            txn_id,
            permit,
            keys: keys.into_iter(),
        }
    }
}

impl<I, K, V> Iterator for IterMut<I, K, V>
where
    I: Copy + Hash + Ord + fmt::Debug,
    K: Hash + Ord,
    V: Clone + fmt::Debug,
{
    type Item = (Arc<K>, TxnMapLockIterMutGuard<V>);

    fn next(&mut self) -> Option<Self::Item> {
        let mut state = self.lock_state.write().expect("lock state");

        loop {
            let key = self.keys.next()?;
            if let Some(guard) = state.get_mut(self.txn_id, key.clone()) {
                return Some((key.into(), TxnMapLockIterMutGuard::from(guard)));
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.keys.size_hint()
    }
}

#[inline]
fn merge<K: Eq + Hash, V>(version: &mut Canon<K, V>, deltas: &Delta<K, V>) {
    for (key, delta) in deltas {
        match delta {
            Some(value) => version.insert(key.clone(), value.clone()),
            None => version.remove(key),
        };
    }
}

#[inline]
fn merge_owned<K: Eq + Hash, V>(version: &mut Canon<K, V>, deltas: Delta<K, V>) {
    for (key, delta) in deltas {
        match delta {
            Some(value) => version.insert(key, value),
            None => version.remove(&key),
        };
    }
}
