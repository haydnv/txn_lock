//! A futures-aware read-write lock on a [`BTreeSet`] which supports transactional versioning.
//!
//! Example usage:
//! ```
//! use std::collections::BTreeSet;
//! use std::sync::Arc;
//! use futures::executor::block_on;
//!
//! use txn_lock::set::*;
//! use txn_lock::Error;
//!
//! let set = TxnSetLock::<u64, String>::new(0);
//!
//! let one = Arc::new("one".to_string());
//! let two = Arc::new("two".to_string());
//!
//! assert!(!block_on(set.contains(1, &one)).expect("contains"));
//! block_on(set.insert(1, one.clone())).expect("insert");
//! assert!(set.try_contains(1, &one).expect("contains"));
//! assert_eq!(set.try_insert(2, one.clone()).unwrap_err(), Error::WouldBlock);
//! set.commit(1);
//!
//! assert!(set.try_contains(2, &one).expect("contains"));
//! assert_eq!(block_on(set.iter(2)).expect("iter").collect::<Vec<Arc<String>>>(), vec![one.clone()]);
//! assert!(block_on(set.remove(2, one.clone())).expect("remove"));
//! assert!(!set.try_contains(2, &one).expect("contains"));
//! assert!(!set.try_remove(2, one.clone()).expect("remove"));
//! assert!(!set.try_contains(2, &one).expect("contains"));
//! assert!(!set.try_remove(2, two.clone()).expect("remove"));
//! set.try_insert(2, two.clone()).expect("insert");
//! set.finalize(2);
//!
//! assert_eq!(set.try_contains(1, &one).unwrap_err(), Error::Outdated);
//! assert!(set.try_contains(3, &one).expect("contains"));
//! assert!(set.try_remove(3, one.clone()).expect("remove"));
//! assert!(!set.try_remove(3, one).expect("remove"));
//! assert!(!set.try_remove(3, two).expect("remove"));
//! set.commit(3);
//!
//! let new_values: BTreeSet<Arc<String>> = ["one", "two", "three", "four"]
//!     .into_iter()
//!     .map(String::from)
//!     .map(Arc::from)
//!     .collect();
//!
//! set.try_extend(4, new_values.clone()).expect("extend");
//! assert_eq!(new_values, set.try_clear(4).expect("clear"));
//! ```

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, RwLock as RwLockInner};
use std::task::Poll;
use std::{fmt, iter};

use super::semaphore::{PermitRead, Semaphore};
use super::{Error, Result};

pub use super::range::Range;

#[derive(Debug, Eq, PartialEq)]
enum KeyState {
    Present,
    Absent,
}

impl KeyState {
    fn is_present(&self) -> bool {
        self == &KeyState::Present
    }
}

type Version<T> = BTreeMap<Arc<T>, KeyState>;
type Canon<T> = BTreeSet<Arc<T>>;
type Committed<I, T> = BTreeMap<I, Option<Version<T>>>;

struct State<I, T> {
    canon: BTreeSet<Arc<T>>,
    committed: BTreeMap<I, Option<Version<T>>>,
    pending: BTreeMap<I, Version<T>>,
    finalized: Option<I>,
}

impl<I: Copy + Ord, T: Ord> State<I, T> {
    fn new(txn_id: I, version: BTreeSet<Arc<T>>) -> Self {
        let version = version
            .into_iter()
            .map(|key| (key, KeyState::Present))
            .collect();

        State {
            canon: BTreeSet::new(),
            committed: BTreeMap::new(),
            pending: BTreeMap::from_iter(iter::once((txn_id, version))),
            finalized: None,
        }
    }

    #[inline]
    fn canon(&self, txn_id: &I) -> (Canon<T>, bool) {
        let mut canon = self.canon.clone();
        for (_, version) in self.committed.iter().take_while(|(id, _)| *id <= &txn_id) {
            if let Some(version) = version {
                merge(&mut canon, version);
            }
        }

        (canon, self.committed.contains_key(&txn_id))
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
    fn contains_canon(&self, txn_id: &I, key: &T) -> bool {
        contains_canon(&self.canon, &self.committed, txn_id, key)
    }

    #[inline]
    fn contains_committed(&self, txn_id: &I, key: &T) -> Poll<Result<bool>> {
        if self.finalized.as_ref() > Some(&txn_id) {
            Poll::Ready(Err(Error::Outdated))
        } else if self.committed.contains_key(&txn_id) {
            assert!(!self.pending.contains_key(&txn_id));
            Poll::Ready(Ok(self.contains_canon(&txn_id, &key)))
        } else {
            Poll::Pending
        }
    }

    #[inline]
    fn contains_pending(&self, txn_id: &I, key: &T) -> bool {
        if let Some(version) = self.pending.get(&txn_id) {
            if let Some(key_state) = version.get(key) {
                return key_state.is_present();
            }
        }

        self.contains_canon(txn_id, key)
    }

    #[inline]
    fn insert(&mut self, txn_id: I, key: Arc<T>) {
        match self.pending.entry(txn_id) {
            Entry::Occupied(mut entry) => {
                entry.get_mut().insert(key, KeyState::Present);
            }
            Entry::Vacant(entry) => {
                let version = iter::once((key, KeyState::Present)).collect();
                entry.insert(version);
            }
        }
    }

    #[inline]
    fn remove(&mut self, txn_id: I, key: Arc<T>) -> bool {
        match self.pending.entry(txn_id) {
            Entry::Occupied(mut pending) => match pending.get_mut().entry(key) {
                Entry::Occupied(mut entry) => entry.insert(KeyState::Absent).is_present(),
                Entry::Vacant(entry) => {
                    let present =
                        contains_canon(&self.canon, &self.committed, &txn_id, entry.key());

                    entry.insert(KeyState::Absent);
                    present
                }
            },
            Entry::Vacant(pending) => {
                let present = contains_canon(&self.canon, &self.committed, pending.key(), &key);
                let version = iter::once((key, KeyState::Absent)).collect();
                pending.insert(version);
                present
            }
        }
    }
}

#[inline]
fn contains_canon<I: Ord, T: Ord>(
    canon: &BTreeSet<Arc<T>>,
    committed: &Committed<I, T>,
    txn_id: &I,
    key: &T,
) -> bool {
    let committed = committed
        .iter()
        .rev()
        .skip_while(|(id, _)| *id > txn_id)
        .map(|(_, version)| version);

    for version in committed {
        if let Some(version) = version {
            if let Some(key_state) = version.get(key) {
                return key_state.is_present();
            }
        }
    }

    canon.contains(key)
}

/// A futures-aware read-write lock on a [`BTreeSet`] which supports transactional versioning.
// TODO: handle the case where a write permit is acquired and then dropped without committing
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

impl<I: Copy + Ord + fmt::Display, T: Ord + fmt::Debug> TxnSetLock<I, T> {
    /// Construct a new, empty [`TxnSetLock`].
    pub fn new(txn_id: I) -> Self {
        Self {
            state: Arc::new(RwLockInner::new(State::new(txn_id, BTreeSet::new()))),
            semaphore: Semaphore::new(),
        }
    }

    /// Commit the state of this [`TxnSetLock`] at `txn_id`.
    pub fn commit(&self, txn_id: I) {
        let mut state = self.state_mut();

        self.semaphore.finalize(&txn_id, false);

        let version = state.pending.remove(&txn_id);

        match state.committed.entry(txn_id) {
            Entry::Occupied(_) => {
                assert!(version.is_none());
                #[cfg(feature = "logging")]
                log::warn!("duplicate commit at {}", txn_id);
            }
            Entry::Vacant(entry) => {
                if let Some(version) = version {
                    entry.insert(Some(version));
                } else {
                    entry.insert(None);
                }
            }
        }
    }

    /// Finalize the state of this [`TxnSetLock`] at `txn_id`.
    /// This will merge in deltas and prevent further reads of versions earlier than `txn_id`.
    pub fn finalize(&self, txn_id: I) {
        let mut state = self.state_mut();

        while let Some(version_id) = state.committed.keys().next().copied() {
            if version_id <= txn_id {
                if let Some(version) = state.committed.remove(&version_id).expect("version") {
                    merge_owned::<I, T>(&mut state.canon, version);
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

    /// Construct a new [`TxnSetLock`] with the given `contents`.
    pub fn with_contents<C: IntoIterator<Item = T>>(txn_id: I, contents: C) -> Self {
        let version = contents.into_iter().map(Arc::new).collect();

        Self {
            state: Arc::new(RwLockInner::new(State::new(txn_id, version))),
            semaphore: Semaphore::with_reservation(txn_id, Range::All),
        }
    }

    /// Check whether the given `key` is present in this [`TxnSetLock`] at `txn_id`.
    pub async fn contains(&self, txn_id: I, key: &Arc<T>) -> Result<bool> {
        // before acquiring a permit, check if this version has already been committed
        if let Poll::Ready(result) = self.state().contains_committed(&txn_id, key) {
            return result;
        }

        let _permit = self.semaphore.read(txn_id, Range::One(key.clone())).await?;
        Ok(self.state().contains_pending(&txn_id, key))
    }

    /// Synchronously check whether the given `key` is present in this [`TxnSetLock`], if possible.
    pub fn try_contains(&self, txn_id: I, key: &Arc<T>) -> Result<bool> {
        // before acquiring a permit, check if this version has already been committed
        if let Poll::Ready(result) = self.state().contains_committed(&txn_id, key) {
            return result;
        }

        let _permit = self.semaphore.try_read(txn_id, Range::One(key.clone()))?;
        Ok(self.state().contains_pending(&txn_id, key))
    }

    /// Remove and return all keys in this [`TxnSetLock`] at `txn_id`.
    pub async fn clear(&self, txn_id: I) -> Result<BTreeSet<Arc<T>>> {
        let (mut set, committed) = self.state().canon(&txn_id);

        if committed {
            return Err(Error::Committed);
        }

        let _permit = self.semaphore.write(txn_id, Range::All).await?;

        let mut state = self.state_mut();
        if let Some(version) = state.pending.get(&txn_id) {
            merge(&mut set, version);
        }

        for key in set.iter().cloned() {
            state.remove(txn_id, key);
        }

        return Ok(set);
    }

    /// Remove and return all keys in this [`TxnSetLock`] at `txn_id` synchronously, if possible.
    pub fn try_clear(&self, txn_id: I) -> Result<BTreeSet<Arc<T>>> {
        let mut state = self.state_mut();
        let (mut set, committed) = state.canon(&txn_id);

        if committed {
            return Err(Error::Committed);
        }

        let _permit = self.semaphore.try_write(txn_id, Range::All)?;

        if let Some(version) = state.pending.get(&txn_id) {
            merge(&mut set, version);
        }

        for key in set.iter().cloned() {
            state.remove(txn_id, key);
        }

        return Ok(set);
    }

    /// Insert the `other` keys into this [`TxnSetLock`] at `txn_id`.
    pub async fn extend<K, E>(&self, txn_id: I, other: E) -> Result<()>
    where
        K: Into<Arc<T>>,
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
        K: Into<Arc<T>>,
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
        let (mut set, committed) = self.state().canon(&txn_id);

        if committed {
            return Ok(Iter::new(None, set));
        }

        let permit = self.semaphore.read(txn_id, Range::All).await?;

        let state = self.state();
        if let Some(version) = state.pending.get(&txn_id) {
            merge(&mut set, version);
        }

        return Ok(Iter::new(Some(permit), set));
    }

    /// Construct an iterator over this [`TxnSetLock`] at `txn_id` synchronously, if possible.
    pub fn try_iter(&self, txn_id: I) -> Result<Iter<T>> {
        let state = self.state();
        let (mut set, committed) = state.canon(&txn_id);

        if committed {
            return Ok(Iter::new(None, set));
        }

        let permit = self.semaphore.try_read(txn_id, Range::All)?;

        if let Some(version) = state.pending.get(&txn_id) {
            merge(&mut set, version);
        }

        return Ok(Iter::new(Some(permit), set));
    }

    /// Insert a new `key` into this [`TxnSetLock`] at `txn_id`.
    pub async fn insert<K: Into<Arc<T>>>(&self, txn_id: I, key: K) -> Result<()> {
        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let key = key.into();
        let range = Range::One(key.clone());
        let _permit = self.semaphore.write(txn_id, range).await?;
        Ok(self.state_mut().insert(txn_id, key))
    }

    /// Insert a new `key` into this [`TxnSetLock`] at `txn_id` synchronously, if possible.
    pub fn try_insert<K: Into<Arc<T>>>(&self, txn_id: I, key: K) -> Result<()> {
        let mut state = self.state_mut();

        // before acquiring a permit, check if this version has already been committed
        state.check_pending(&txn_id)?;

        let key = key.into();
        let _permit = self.semaphore.try_write(txn_id, Range::One(key.clone()))?;
        Ok(state.insert(txn_id, key))
    }

    /// Remove a `key` into this [`TxnSetLock`] at `txn_id` and return `true` if it was present.
    pub async fn remove<K: Into<Arc<T>>>(&self, txn_id: I, key: K) -> Result<bool> {
        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let key = key.into();
        let range = Range::One(key.clone());
        let _permit = self.semaphore.try_write(txn_id, range)?;
        Ok(self.state_mut().remove(txn_id, key))
    }

    /// Remove a `key` into this [`TxnSetLock`] at `txn_id` and return `true` if it was present.
    pub fn try_remove<K: Into<Arc<T>>>(&self, txn_id: I, key: K) -> Result<bool> {
        let mut state = self.state_mut();

        // before acquiring a permit, check if this version has already been committed
        state.check_pending(&txn_id)?;

        let key = key.into();
        let range = Range::One(key.clone());
        let _permit = self.semaphore.try_write(txn_id, range)?;
        Ok(state.remove(txn_id, key))
    }

    /// Roll back the state of this [`TxnSetLock`] at `txn_id`.
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

/// An iterator over the values of a [`TxnSetLock`] as of a specific transactional version
pub struct Iter<T> {
    #[allow(unused)]
    permit: Option<PermitRead<Range<T>>>,
    iter: <BTreeSet<Arc<T>> as IntoIterator>::IntoIter,
}

impl<T> Iter<T> {
    fn new(permit: Option<PermitRead<Range<T>>>, set: BTreeSet<Arc<T>>) -> Self {
        Self {
            permit,
            iter: set.into_iter(),
        }
    }
}

impl<T> Iterator for Iter<T> {
    type Item = Arc<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T> DoubleEndedIterator for Iter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

#[inline]
fn merge_owned<I: Ord, T: Ord>(canon: &mut Canon<T>, version: Version<T>) {
    for (key, key_state) in version {
        match key_state {
            KeyState::Present => canon.insert(key),
            KeyState::Absent => canon.remove(&key),
        };
    }
}

#[inline]
fn merge<T: Ord>(canon: &mut Canon<T>, version: &Version<T>) {
    for (key, key_state) in version {
        match key_state {
            KeyState::Present => canon.insert(key.clone()),
            KeyState::Absent => canon.remove(key),
        };
    }
}
