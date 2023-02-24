//! A futures-aware read-write lock on a [`HashSet`] which supports transactional versioning.
//!
//! Example usage:
//! ```
//! use std::collections::HashSet;
//! use std::sync::Arc;
//! use futures::executor::block_on;
//!
//! use txn_lock::set::*;
//! use txn_lock::Error;
//!
//! let set = TxnSetLock::<u64, String>::new(0);
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

use ds_ext::OrdHashMap;

use super::guard::TxnCommitGuard;
use super::semaphore::{PermitRead, Semaphore};
use super::{Error, Result};

pub use super::range::{Key, Range};

type Delta<T> = HashMap<Key<T>, bool>;
type Canon<T> = HashSet<Key<T>>;
type Committed<I, T> = OrdHashMap<I, Option<Delta<T>>>;

/// A read guard on the committed state of a [`TxnSetLock`]
pub type TxnSetLockCommitGuard<I, T> = TxnCommitGuard<I, Range<T>, Canon<T>>;

struct State<I, T> {
    canon: Canon<T>,
    committed: OrdHashMap<I, Option<Delta<T>>>,
    pending: OrdHashMap<I, Delta<T>>,
    finalized: Option<I>,
}

impl<I: Copy + Hash + Ord + fmt::Debug, T: Hash + Ord> State<I, T> {
    fn new(txn_id: I, set: Canon<T>) -> Self {
        let delta = set.into_iter().map(|key| (key, true)).collect();

        State {
            canon: Canon::new(),
            committed: OrdHashMap::new(),
            pending: OrdHashMap::from_iter(iter::once((txn_id, delta))),
            finalized: None,
        }
    }

    #[inline]
    fn canon(&self, txn_id: &I) -> Canon<T> {
        let mut canon = self.canon.clone();
        for (_, version) in self.committed.iter().take_while(|(id, _)| *id <= &txn_id) {
            if let Some(delta) = version {
                merge(&mut canon, delta);
            }
        }

        canon
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
    fn contains_canon<Q>(&self, txn_id: &I, key: &Q) -> bool
    where
        Q: Eq + Hash + ?Sized,
        Key<T>: Borrow<Q>,
    {
        contains_canon(&self.canon, &self.committed, txn_id, key)
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
                if self.committed.contains_key(txn_id) {
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
                    if contains_canon(&self.canon, &self.committed, &txn_id, entry.key()) {
                        entry.insert(false);
                        true
                    } else {
                        false
                    }
                }
            }
        } else if contains_canon(&self.canon, &self.committed, &txn_id, &key) {
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
    committed: &Committed<I, T>,
    txn_id: &I,
    key: &Q,
) -> bool
where
    I: Hash + Ord + fmt::Debug,
    T: Hash + Ord,
    Q: Eq + Hash + ?Sized,
    Key<T>: Borrow<Q>,
{
    let committed = committed
        .iter()
        .rev()
        .skip_while(|(id, _)| *id > txn_id)
        .map(|(_, version)| version);

    for version in committed {
        if let Some(delta) = version {
            if let Some(key_state) = delta.get(key) {
                return *key_state;
            }
        }
    }

    canon.contains(key)
}

/// A futures-aware read-write lock on a [`HashSet`] which supports transactional versioning
pub struct TxnSetLock<I, T> {
    state: Arc<RwLockInner<State<I, T>>>,
    semaphore: Semaphore<I, Range<T>>,
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
    I: Copy + Hash + Ord + fmt::Debug,
    T: Hash + Ord + fmt::Debug,
{
    /// Construct a new, empty [`TxnSetLock`].
    pub fn new(txn_id: I) -> Self {
        Self {
            state: Arc::new(RwLockInner::new(State::new(txn_id, Canon::new()))),
            semaphore: Semaphore::new(),
        }
    }

    /// Construct a new [`TxnSetLock`] with the given `contents`.
    pub fn with_contents<C: IntoIterator<Item = T>>(txn_id: I, contents: C) -> Self {
        let set = contents.into_iter().map(Key::new).collect();

        Self {
            state: Arc::new(RwLockInner::new(State::new(txn_id, set))),
            semaphore: Semaphore::with_reservation(txn_id, Range::All),
        }
    }

    /// Check whether the given `key` is present in this [`TxnSetLock`] at `txn_id`.
    pub async fn contains<Q>(&self, txn_id: I, key: &Q) -> Result<bool>
    where
        Q: Eq + Hash + ToOwned<Owned = T> + ?Sized,
        Key<T>: Borrow<Q>,
    {
        // before acquiring a permit, check if this version has already been committed
        let range: Range<T> = {
            let state = self.state();
            if let Poll::Ready(result) = self.state().contains_committed(&txn_id, key) {
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
        // before acquiring a permit, check if this version has already been committed
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
        // before acquiring a permit, check if this version has already been committed
        let mut set = {
            let state = self.state();
            state.check_pending(&txn_id)?;
            state.canon(&txn_id)
        };

        let _permit = self.semaphore.write(txn_id, Range::All).await?;

        let mut state = self.state_mut();
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
        let mut state = self.state_mut();

        // before acquiring a permit, check if this version has already been committed
        let mut set = {
            state.check_pending(&txn_id)?;
            state.canon(&txn_id)
        };

        let _permit = self.semaphore.try_write(txn_id, Range::All)?;

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
        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let _permit = self.semaphore.write(txn_id, Range::All).await?;
        let mut state = self.state_mut();
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
        let mut state = self.state_mut();

        // before acquiring a permit, check if this version has already been committed
        state.check_pending(&txn_id)?;

        let _permit = self.semaphore.try_write(txn_id, Range::All)?;
        for key in other {
            state.insert(txn_id, key.into());
        }

        Ok(())
    }

    /// Construct an iterator over this [`TxnSetLock`] at `txn_id`.
    pub async fn iter(&self, txn_id: I) -> Result<Iter<T>> {
        let mut set = {
            let state = self.state();
            let set = state.canon(&txn_id);

            // before acquiring a permit, check if this version has already been committed
            if state.check_committed(&txn_id)? {
                return Ok(Iter::new(None, set));
            } else {
                set
            }
        };

        let permit = self.semaphore.read(txn_id, Range::All).await?;

        let state = self.state();
        if let Some(delta) = state.pending.get(&txn_id) {
            merge(&mut set, delta);
        }

        return Ok(Iter::new(Some(permit), set));
    }

    /// Construct an iterator over this [`TxnSetLock`] at `txn_id` synchronously, if possible.
    pub fn try_iter(&self, txn_id: I) -> Result<Iter<T>> {
        let state = self.state();
        let mut set = state.canon(&txn_id);

        // before acquiring a permit, check if this version has already been committed
        if state.check_committed(&txn_id)? {
            return Ok(Iter::new(None, set));
        }

        let permit = self.semaphore.try_read(txn_id, Range::All)?;

        if let Some(delta) = state.pending.get(&txn_id) {
            merge(&mut set, delta);
        }

        return Ok(Iter::new(Some(permit), set));
    }

    /// Insert a new `key` into this [`TxnSetLock`] at `txn_id`.
    pub async fn insert<K: Into<Key<T>>>(&self, txn_id: I, key: K) -> Result<()> {
        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let key = key.into();
        let _permit = self.semaphore.write(txn_id, key.clone().into()).await?;
        Ok(self.state_mut().insert(txn_id, key))
    }

    /// Insert a new `key` into this [`TxnSetLock`] at `txn_id` synchronously, if possible.
    pub fn try_insert<K: Into<Key<T>>>(&self, txn_id: I, key: K) -> Result<()> {
        let mut state = self.state_mut();

        // before acquiring a permit, check if this version has already been committed
        state.check_pending(&txn_id)?;

        let key = key.into();
        let _permit = self.semaphore.try_write(txn_id, key.clone().into())?;
        Ok(state.insert(txn_id, key))
    }

    /// Remove a `key` into this [`TxnSetLock`] at `txn_id` and return `true` if it was present.
    pub async fn remove<Q>(&self, txn_id: I, key: &Q) -> Result<bool>
    where
        Q: Eq + Hash + ToOwned<Owned = T> + ?Sized,
        Key<T>: Borrow<Q>,
    {
        // before acquiring a permit, check if this version has already been committed
        let key: Key<T> = {
            let state = self.state();
            state.check_pending(&txn_id)?;
            (key, state.key(&txn_id, key)).into()
        };

        let _permit = self.semaphore.try_write(txn_id, key.clone().into())?;
        Ok(self.state_mut().remove(txn_id, key))
    }

    /// Remove a `key` into this [`TxnSetLock`] at `txn_id` and return `true` if it was present.
    pub fn try_remove<Q>(&self, txn_id: I, key: &Q) -> Result<bool>
    where
        Q: Eq + Hash + ToOwned<Owned = T> + ?Sized,
        Key<T>: Borrow<Q>,
    {
        let mut state = self.state_mut();

        // before acquiring a permit, check if this version has already been committed
        state.check_pending(&txn_id)?;

        let key: Key<T> = (key, state.key(&txn_id, key)).into();
        let _permit = self.semaphore.try_write(txn_id, key.clone().into())?;
        Ok(state.remove(txn_id, key))
    }
}

impl<I, T> TxnSetLock<I, T>
where
    I: Copy + Hash + Ord + fmt::Debug,
    T: Hash + Ord + fmt::Debug,
{
    /// Commit the state of this [`TxnSetLock`] at `txn_id`.
    pub fn commit(&self, txn_id: I) {
        let mut state = self.state_mut();

        if state.finalized >= Some(txn_id) {
            #[cfg(feature = "logging")]
            log::warn!("committed already-finalized version {:?}", txn_id);
            return;
        }

        self.semaphore.finalize(&txn_id, false);

        let finalize = state.pending.keys().next() == Some(&txn_id);
        let version = state.pending.remove(&txn_id);

        if finalize {
            assert!(!state.committed.contains_key(&txn_id));
            let delta = version.expect("committed version");
            merge_owned(&mut state.canon, delta);
            state.finalized = Some(txn_id);
        } else if let Some(delta) = version {
            assert!(state.committed.insert(txn_id, Some(delta)).is_none());
        } else if let Some(prior_commit) = state.committed.insert(txn_id, None) {
            assert!(prior_commit.is_none());
            #[cfg(feature = "logging")]
            log::warn!("duplicate commit at {:?}", txn_id);
        }
    }

    /// Acquire a read lock on the contents of this [`TxnSetLock`] and commit in the same operation.
    /// Panics: if the state of this [`TxnSetLock`] has already been finalized at `txn_id`.
    pub async fn read_and_commit(&self, txn_id: I) -> TxnSetLockCommitGuard<I, T> {
        {
            let state = self.state();

            // before acquiring a permit, check if this version has already been committed
            if state.check_committed(&txn_id).expect("committed") {
                #[cfg(feature = "logging")]
                log::warn!("duplicate commit at {:?}", txn_id);

                return TxnSetLockCommitGuard::duplicate(state.canon(&txn_id));
            }
        };

        let permit = self
            .semaphore
            .read(txn_id, Range::All)
            .await
            .expect("permit");

        let mut state = self.state_mut();
        let finalize = state.pending.keys().next() == Some(&txn_id);
        let version = state.pending.remove(&txn_id);

        if finalize {
            assert!(!state.committed.contains_key(&txn_id));
            let delta = version.expect("committed version");
            merge_owned(&mut state.canon, delta);
            state.finalized = Some(txn_id);
        } else {
            // the case of a duplicate commit has already been handled
            assert!(state.committed.insert(txn_id, version).is_none());
        }

        TxnSetLockCommitGuard::commit(txn_id, self.semaphore.clone(), permit, state.canon.clone())
    }

    /// Roll back the state of this [`TxnSetLock`] at `txn_id`.
    pub fn rollback(&self, txn_id: &I) {
        let mut state = self.state_mut();

        assert!(
            !state.committed.contains_key(txn_id),
            "cannot roll back committed transaction {:?}",
            txn_id
        );

        self.semaphore.finalize(txn_id, false);
        state.pending.remove(txn_id);
    }

    /// Finalize the state of this [`TxnSetLock`] at `txn_id`.
    /// This will merge in deltas and prevent further reads of versions earlier than `txn_id`.
    pub fn finalize(&self, txn_id: I) {
        let mut state = self.state_mut();

        while let Some(version_id) = state.committed.keys().next().copied() {
            if version_id <= txn_id {
                if let Some(delta) = state.committed.remove(&version_id).expect("version") {
                    merge_owned::<T>(&mut state.canon, delta);
                }
            } else {
                break;
            }
        }

        if let Some(next_commit) = state.committed.keys().next() {
            assert!(next_commit > &txn_id);
        }

        self.semaphore.finalize(&txn_id, true);
        state.finalized = Some(txn_id);
    }
}

/// An iterator over the values of a [`TxnSetLock`] as of a specific transactional version
pub struct Iter<T> {
    #[allow(unused)]
    permit: Option<PermitRead<Range<T>>>,
    iter: <Canon<T> as IntoIterator>::IntoIter,
}

impl<T> Iter<T> {
    fn new(permit: Option<PermitRead<Range<T>>>, set: Canon<T>) -> Self {
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
fn merge_owned<T: Hash + Ord>(canon: &mut Canon<T>, delta: Delta<T>) {
    for (key, key_state) in delta {
        match key_state {
            true => canon.insert(key),
            false => canon.remove(&key),
        };
    }
}

#[inline]
fn merge<T: Hash + Ord>(canon: &mut Canon<T>, delta: &Delta<T>) {
    for (key, key_state) in delta {
        match key_state {
            true => canon.insert(key.clone()),
            false => canon.remove(key),
        };
    }
}

impl<I, T> fmt::Debug for TxnSetLock<I, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a transactional lock on a set of values")
    }
}
