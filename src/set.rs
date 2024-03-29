//! A futures-aware read-write lock on a [`HashSet`] which supports transactional versioning.
//!
//! Example usage:
//! ```
//! use std::collections::HashSet;
//! use std::sync::Arc;
//!
//! use futures::executor::block_on;
//!
//! use txn_lock::set::*;
//! use txn_lock::Error;
//!
//! let set = TxnSetLock::<u64, String>::new(0, []);
//!
//! let one = "one";
//! let two = "two";
//!
//! assert!(!block_on(set.contains(1, one)).expect("contains"));
//! block_on(set.insert(1, one.to_string())).expect("insert");
//! assert!(set.try_contains(1, one).expect("contains"));
//! assert_eq!(set.try_insert(2, one.to_string()).unwrap_err(), Error::WouldBlock);
//! set.commit(1);
//!
//! assert!(set.try_contains(2, one).expect("contains"));
//! assert_eq!(
//!     block_on(set.iter(2)).expect("iter").collect::<Vec<Arc<String>>>(),
//!     vec![Arc::new(one.to_string())]
//! );
//! assert!(block_on(set.remove(2, one)).expect("remove"));
//! assert!(!set.try_contains(2, one).expect("contains"));
//! assert!(!set.try_remove(2, one).expect("remove"));
//! assert!(!set.try_contains(2, one).expect("contains"));
//! assert!(!set.try_remove(2, two).expect("remove"));
//! set.try_insert(2, two.to_string()).expect("insert");
//! set.finalize(2);
//!
//! assert_eq!(set.try_contains(1, one).unwrap_err(), Error::Outdated);
//! assert!(set.try_contains(3, one).expect("contains"));
//! assert!(set.try_remove(3, one).expect("remove"));
//! assert!(!set.try_remove(3, one).expect("remove"));
//! assert!(!set.try_remove(3, two).expect("remove"));
//! set.commit(3);
//!
//! let new_values: HashSet<String> = ["one", "two", "three", "four"]
//!     .into_iter()
//!     .map(String::from)
//!     .collect();
//!
//! set.try_extend(4, new_values.clone()).expect("extend");
//! assert_eq!(
//!     new_values.into_iter().map(Arc::from).collect::<HashSet<Arc<String>>>(),
//!     set.try_clear(4).expect("clear").into_iter().map(Arc::from).collect::<HashSet<_>>()
//! );
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

use super::semaphore::{PermitRead, Semaphore};
use super::{Error, Result};

pub use super::range::{Key, Range};

type Delta<T> = HashMap<Key<T>, bool>;
type Canon<T> = HashSet<Key<T>>;

struct State<I, T> {
    canon: Canon<T>,
    deltas: OrdHashMap<I, Delta<T>>,
    commits: OrdHashSet<I>,
    pending: OrdHashMap<I, Delta<T>>,
    finalized: Option<I>,
}

impl<I: Ord + Hash + fmt::Debug, T: Eq + Hash> State<I, T> {
    fn new(txn_id: I, set: Canon<T>) -> Self {
        let delta = set.into_iter().map(|key| (key, true)).collect();

        State {
            canon: Canon::new(),
            deltas: OrdHashMap::new(),
            commits: OrdHashSet::new(),
            pending: OrdHashMap::from_iter(iter::once((txn_id, delta))),
            finalized: None,
        }
    }
}

impl<I: Copy + Hash + Ord + fmt::Debug, T: Hash + Ord> State<I, T> {
    #[inline]
    fn canon(&self, txn_id: &I) -> Canon<T> {
        let mut canon = self.canon.clone();

        for (_, delta) in self.deltas.iter().take_while(|(id, _)| *id <= &txn_id) {
            merge(&mut canon, delta);
        }

        canon
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
    fn commit(&mut self, txn_id: I) {
        if self.commits.contains(&txn_id) {
            #[cfg(feature = "logging")]
            log::warn!("duplicate commit at {:?}", txn_id);
        } else if let Some(delta) = self.pending.remove(&txn_id) {
            self.deltas.insert(txn_id, delta);
        }

        self.commits.insert(txn_id);
    }

    #[inline]
    fn contains_canon<Q>(&self, txn_id: &I, key: &Q) -> bool
    where
        Q: Eq + Hash + ?Sized,
        Key<T>: Borrow<Q>,
    {
        contains_canon(&self.canon, &self.deltas, txn_id, key)
    }

    #[inline]
    fn contains_committed<Q>(&self, txn_id: &I, key: &Q) -> Poll<Result<bool>>
    where
        Q: Eq + Hash + ?Sized,
        Key<T>: Borrow<Q>,
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
        Key<T>: Borrow<Q>,
    {
        if let Some(delta) = self.pending.get(txn_id) {
            if let Some(key_state) = delta.get(key) {
                return *key_state;
            }
        }

        self.contains_canon(txn_id, key)
    }

    #[inline]
    fn finalize(&mut self, txn_id: I) -> Option<&Canon<T>> {
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
    fn insert(&mut self, txn_id: I, key: Key<T>) {
        if let Some(pending) = self.pending.get_mut(&txn_id) {
            pending.insert(key, true);
        } else {
            let delta = iter::once((key, true)).collect();
            self.pending.insert(txn_id, delta);
        }
    }

    #[inline]
    fn key<Q>(&self, txn_id: &I, key: &Q) -> Option<&Key<T>>
    where
        Q: Eq + Hash + ?Sized,
        Key<T>: Borrow<Q>,
    {
        if let Some(key) = self.canon.get(key) {
            Some(key)
        } else if let Some(deltas) = self.pending.get(txn_id) {
            deltas.get_key_value(key).map(|(key, _present)| key)
        } else {
            None
        }
    }

    #[inline]
    fn remove(&mut self, txn_id: I, key: Key<T>) -> bool {
        if let Some(pending) = self.pending.get_mut(&txn_id) {
            match pending.entry(key) {
                hash_map::Entry::Occupied(mut entry) => entry.insert(false),
                hash_map::Entry::Vacant(entry) => {
                    if contains_canon(&self.canon, &self.deltas, &txn_id, entry.key()) {
                        entry.insert(false);
                        true
                    } else {
                        false
                    }
                }
            }
        } else if contains_canon(&self.canon, &self.deltas, &txn_id, &key) {
            let deltas = iter::once((key, false)).collect();
            self.pending.insert(txn_id, deltas);
            true
        } else {
            false
        }
    }
}

#[inline]
fn contains_canon<I, T, Q>(
    canon: &Canon<T>,
    deltas: &OrdHashMap<I, Delta<T>>,
    txn_id: &I,
    key: &Q,
) -> bool
where
    I: Hash + Ord + fmt::Debug,
    T: Hash + Ord,
    Q: Eq + Hash + ?Sized,
    Key<T>: Borrow<Q>,
{
    let deltas = deltas
        .iter()
        .rev()
        .skip_while(|(id, _)| *id > txn_id)
        .map(|(_, version)| version);

    for delta in deltas {
        if let Some(key_state) = delta.get(key) {
            return *key_state;
        }
    }

    canon.contains(key)
}

/// A futures-aware read-write lock on a [`HashSet`] which supports transactional versioning
pub struct TxnSetLock<I, T> {
    state: Arc<RwLockInner<State<I, T>>>,
    semaphore: Semaphore<I, Collator<T>, Range<T>>,
}

impl<I, T> Clone for TxnSetLock<I, T> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            semaphore: self.semaphore.clone(),
        }
    }
}

impl<I, T> TxnSetLock<I, T> {
    #[inline]
    fn state(&self) -> impl Deref<Target = State<I, T>> + '_ {
        self.state.read().expect("lock state")
    }

    #[inline]
    fn state_mut(&self) -> impl DerefMut<Target = State<I, T>> + '_ {
        self.state.write().expect("lock state")
    }
}

impl<I, T> TxnSetLock<I, T>
where
    I: Ord + Hash + fmt::Debug,
    T: Eq + Hash,
{
    /// Construct a new, empty [`TxnSetLock`].
    pub fn new<Contents>(txn_id: I, contents: Contents) -> Self
    where
        Contents: IntoIterator<Item = T>,
    {
        let contents = contents.into_iter().map(Key::from).collect();

        Self {
            state: Arc::new(RwLockInner::new(State::new(txn_id, contents))),
            semaphore: Semaphore::new(Collator::<T>::default()),
        }
    }
}

impl<I, T> TxnSetLock<I, T>
where
    I: Copy + Hash + Ord + fmt::Debug,
    T: Hash + Ord + fmt::Debug + Send + Sync,
    Range<T>: Send + Sync,
{
    /// Construct a new [`TxnSetLock`] with the given `contents`.
    pub fn with_contents<IT: IntoIterator<Item = T>>(txn_id: I, contents: IT) -> Self {
        let set = contents.into_iter().map(Key::new).collect();
        let collator = Collator::<T>::default();

        Self {
            state: Arc::new(RwLockInner::new(State::new(txn_id, set))),
            semaphore: Semaphore::with_reservation(txn_id, collator, Range::All),
        }
    }

    /// Check whether the given `key` is present in this [`TxnSetLock`] at `txn_id`.
    pub async fn contains<Q>(&self, txn_id: I, key: &Q) -> Result<bool>
    where
        Q: Eq + Hash + ToOwned<Owned = T> + ?Sized,
        Key<T>: Borrow<Q>,
    {
        let range: Range<T> = {
            let state = self.state();

            if let Poll::Ready(result) = state.contains_committed(&txn_id, key) {
                return result;
            } else {
                Key::<T>::from((key, state.key(&txn_id, key))).into()
            }
        };

        let _permit = self.semaphore.read(txn_id, range).await?;
        Ok(self.state().contains_pending(&txn_id, key))
    }

    /// Synchronously check whether the given `key` is present in this [`TxnSetLock`], if possible.
    pub fn try_contains<Q>(&self, txn_id: I, key: &Q) -> Result<bool>
    where
        Q: Eq + Hash + ToOwned<Owned = T> + ?Sized,
        Key<T>: Borrow<Q>,
    {
        let state = self.state();
        if let Poll::Ready(result) = state.contains_committed(&txn_id, key) {
            return result;
        }

        let range = Key::<T>::from((key, state.key(&txn_id, key))).into();
        let _permit = self.semaphore.try_read(txn_id, range)?;
        Ok(state.contains_pending(&txn_id, key))
    }

    /// Remove and return all keys in this [`TxnSetLock`] at `txn_id`.
    pub async fn clear(&self, txn_id: I) -> Result<Canon<T>> {
        let _permit = self.semaphore.write(txn_id, Range::All).await?;

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        let mut set = state.canon(&txn_id);

        if let Some(deltas) = state.pending.get(&txn_id) {
            merge(&mut set, deltas);
        }

        for key in set.iter().cloned() {
            state.remove(txn_id, key);
        }

        return Ok(set);
    }

    /// Remove and return all keys in this [`TxnSetLock`] at `txn_id` synchronously, if possible.
    pub fn try_clear(&self, txn_id: I) -> Result<Canon<T>> {
        let _permit = self.semaphore.try_write(txn_id, Range::All)?;

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        let mut set = state.canon(&txn_id);

        if let Some(delta) = state.pending.get(&txn_id) {
            merge(&mut set, delta);
        }

        for key in set.iter().cloned() {
            state.remove(txn_id, key);
        }

        return Ok(set);
    }

    /// Insert the `other` keys into this [`TxnSetLock`] at `txn_id`.
    pub async fn extend<K, E>(&self, txn_id: I, other: E) -> Result<()>
    where
        K: Into<Key<T>>,
        E: IntoIterator<Item = K>,
    {
        let _permit = self.semaphore.write(txn_id, Range::All).await?;

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        for key in other {
            state.insert(txn_id, key.into());
        }

        Ok(())
    }

    /// Insert the `other` keys into this [`TxnSetLock`] at `txn_id` synchronously, if possible.
    pub fn try_extend<K, E>(&self, txn_id: I, other: E) -> Result<()>
    where
        K: Into<Key<T>>,
        E: IntoIterator<Item = K>,
    {
        let _permit = self.semaphore.try_write(txn_id, Range::All)?;

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        for key in other {
            state.insert(txn_id, key.into());
        }

        Ok(())
    }

    /// Construct an iterator over this [`TxnSetLock`] at `txn_id`.
    pub async fn iter(&self, txn_id: I) -> Result<Iter<T>> {
        let permit = self.semaphore.read(txn_id, Range::All).await?;

        let state = self.state();

        let mut set = state.canon(&txn_id);
        if let Some(delta) = state.pending.get(&txn_id) {
            merge(&mut set, delta);
        }

        return Ok(Iter::new(permit, set));
    }

    /// Construct an iterator over this [`TxnSetLock`] at `txn_id` synchronously, if possible.
    pub fn try_iter(&self, txn_id: I) -> Result<Iter<T>> {
        let permit = self.semaphore.try_read(txn_id, Range::All)?;

        let state = self.state();

        let mut set = state.canon(&txn_id);
        if let Some(delta) = state.pending.get(&txn_id) {
            merge(&mut set, delta);
        }

        return Ok(Iter::new(permit, set));
    }

    /// Insert a new `key` into this [`TxnSetLock`] at `txn_id`.
    pub async fn insert<K: Into<Key<T>>>(&self, txn_id: I, key: K) -> Result<()> {
        let key = key.into();
        let _permit = self.semaphore.write(txn_id, key.clone().into()).await?;

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        Ok(state.insert(txn_id, key))
    }

    /// Insert a new `key` into this [`TxnSetLock`] at `txn_id` synchronously, if possible.
    pub fn try_insert<K: Into<Key<T>>>(&self, txn_id: I, key: K) -> Result<()> {
        let key = key.into();
        let _permit = self.semaphore.try_write(txn_id, key.clone().into())?;

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        Ok(state.insert(txn_id, key))
    }

    /// Return `true` if this [`TxnSetLock`] is empty at the given `txn_id`.
    pub async fn is_empty(&self, txn_id: I) -> Result<bool> {
        self.len(txn_id).map_ok(|len| len == 0).await
    }

    /// Get the size of this [`TxnSetLock`] at the given `txn_id`.
    pub async fn len(&self, txn_id: I) -> Result<usize> {
        let _permit = self.semaphore.read(txn_id, Range::All).await?;

        let state = self.state();

        let mut set = state.canon(&txn_id);
        if let Some(delta) = state.pending.get(&txn_id) {
            merge(&mut set, delta);
        }

        Ok(set.len())
    }

    /// Remove a `key` into this [`TxnSetLock`] at `txn_id` and return `true` if it was present.
    pub async fn remove<Q>(&self, txn_id: I, key: &Q) -> Result<bool>
    where
        Q: Eq + Hash + ToOwned<Owned = T> + ?Sized,
        Key<T>: Borrow<Q>,
    {
        let key: Key<T> = (key, self.state().key(&txn_id, key)).into();

        let _permit = self.semaphore.try_write(txn_id, key.clone().into())?;

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        Ok(state.remove(txn_id, key))
    }

    /// Remove a `key` into this [`TxnSetLock`] at `txn_id` and return `true` if it was present.
    pub fn try_remove<Q>(&self, txn_id: I, key: &Q) -> Result<bool>
    where
        Q: Eq + Hash + ToOwned<Owned = T> + ?Sized,
        Key<T>: Borrow<Q>,
    {
        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        let key: Key<T> = (key, state.key(&txn_id, key)).into();
        let _permit = self.semaphore.try_write(txn_id, key.clone().into())?;

        Ok(state.remove(txn_id, key))
    }
}

impl<I, T> TxnSetLock<I, T>
where
    I: Copy + Hash + Ord + fmt::Debug,
    T: Hash + Ord + fmt::Debug + Send + Sync,
{
    /// Commit the state of this [`TxnSetLock`] at `txn_id`.
    /// Panics: if this [`TxnSetLock`] has already been finalized at `txn_id`
    pub fn commit(&self, txn_id: I) {
        let mut state = self.state_mut();

        if state.finalized >= Some(txn_id) {
            panic!("tried to commit already-finalized version {:?}", txn_id);
        }

        state.commit(txn_id);

        self.semaphore.finalize(&txn_id, false);
    }

    /// Read and commit of this [`TxnSetLock`] in a single operation, if there is a version pending.
    /// Also returns the set of changes committed, if any.
    ///
    /// Panics: if this [`TxnSetLock`] has already been finalized at `txn_id`
    pub async fn read_and_commit(&self, txn_id: I) -> (Canon<T>, Option<Delta<T>>) {
        let _permit = self
            .semaphore
            .read(txn_id, Range::All)
            .await
            .expect("permit");

        let (version, delta) = {
            let mut state = self.state_mut();

            if state.finalized > Some(txn_id) {
                panic!("tried to commit already-finalized version {:?}", txn_id);
            }

            state.commit(txn_id);

            (state.canon(&txn_id), state.deltas.get(&txn_id).cloned())
        };

        self.semaphore.finalize(&txn_id, false);

        (version, delta)
    }

    /// Roll back the state of this [`TxnSetLock`] at `txn_id`.
    /// Panics: if this [`TxnSetLock`] has already been committed or finalized at `txn_id`
    pub fn rollback(&self, txn_id: &I) {
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

    /// Read and roll back this [`TxnSetLock`] in a single operation, if there is a version pending.
    /// Also returns the set of changes rolled back, if any.
    ///
    /// Panics: if this [`TxnSetLock`] has already been committed or finalized at `txn_id`
    pub async fn read_and_rollback(&self, txn_id: I) -> (Canon<T>, Option<Delta<T>>) {
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

            let mut version = state.canon(&txn_id);
            let deltas = state.pending.remove(&txn_id);

            if let Some(deltas) = &deltas {
                merge(&mut version, deltas);
            }

            (version, deltas)
        };

        self.semaphore.finalize(&txn_id, false);

        (version, deltas)
    }

    /// Finalize the state of this [`TxnSetLock`] at `txn_id`.
    /// This will merge in deltas and prevent further reads of versions earlier than `txn_id`.
    pub fn finalize(&self, txn_id: I) {
        self.semaphore.finalize(&txn_id, true);
        self.state_mut().finalize(txn_id);
    }

    /// Read and finalize the state of this [`TxnSetLock`] at `txn_id`, if not already finalized.
    /// This will merge in deltas and prevent further reads of versions earlier than `txn_id`.
    pub fn read_and_finalize(&self, txn_id: I) -> Option<Canon<T>> {
        self.semaphore.finalize(&txn_id, true);
        self.state_mut().finalize(txn_id).cloned()
    }
}

/// An iterator over the values of a [`TxnSetLock`] as of a specific transactional version
pub struct Iter<T> {
    #[allow(unused)]
    permit: PermitRead<Range<T>>,
    iter: <Canon<T> as IntoIterator>::IntoIter,
}

impl<T> Iter<T> {
    fn new(permit: PermitRead<Range<T>>, set: Canon<T>) -> Self {
        Self {
            permit,
            iter: set.into_iter(),
        }
    }
}

impl<T> Iterator for Iter<T> {
    type Item = Arc<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(Arc::from)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[inline]
fn merge<T: Hash + Ord>(version: &mut Canon<T>, delta: &Delta<T>) {
    for (key, key_state) in delta {
        match key_state {
            true => version.insert(key.clone()),
            false => version.remove(key),
        };
    }
}

#[inline]
fn merge_owned<T: Hash + Ord>(version: &mut Canon<T>, delta: Delta<T>) {
    for (key, key_state) in delta {
        match key_state {
            true => version.insert(key),
            false => version.remove(&key),
        };
    }
}

impl<I, T> fmt::Debug for TxnSetLock<I, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a transactional lock on a set of values")
    }
}
