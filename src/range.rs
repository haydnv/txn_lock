use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

use collate::{Collate, Overlap, OverlapsRange};

/// A key in a transactional lock
pub struct Key<K> {
    key: Arc<K>,
}

impl<K> Clone for Key<K> {
    fn clone(&self) -> Self {
        Self {
            key: self.key.clone(),
        }
    }
}

impl<K> Key<K> {
    /// Construct a new [`Key`]
    pub fn new(key: K) -> Self {
        Self { key: Arc::new(key) }
    }
}

impl<K> From<K> for Key<K> {
    fn from(key: K) -> Self {
        Self { key: Arc::new(key) }
    }
}

impl<K> From<Arc<K>> for Key<K> {
    fn from(key: Arc<K>) -> Self {
        Self { key }
    }
}

impl<'a, K, Q: ToOwned<Owned = K> + ?Sized> From<(&'a Q, Option<&'a Key<K>>)> for Key<K> {
    fn from(known: (&'a Q, Option<&'a Key<K>>)) -> Self {
        let (key, maybe_key) = known;

        let key = if let Some(key) = maybe_key {
            key.key.clone()
        } else {
            Arc::new(key.to_owned())
        };

        Self { key }
    }
}

impl<K: PartialEq> PartialEq<K> for Key<K> {
    fn eq(&self, other: &K) -> bool {
        &*self.key == other
    }
}

impl<K: PartialEq> PartialEq<Arc<K>> for Key<K> {
    fn eq(&self, other: &Arc<K>) -> bool {
        &self.key == other
    }
}

impl<K: PartialEq> PartialEq<Self> for Key<K> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<K: Eq> Eq for Key<K> {}

impl<K: Hash> Hash for Key<K> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.hash(state)
    }
}

impl<K> Deref for Key<K> {
    type Target = K;

    fn deref(&self) -> &Self::Target {
        self.key.deref()
    }
}

impl<K> Borrow<Arc<K>> for Key<K> {
    fn borrow(&self) -> &Arc<K> {
        &self.key
    }
}

#[cfg(feature = "id")]
impl Borrow<str> for Key<hr_id::Id> {
    fn borrow(&self) -> &str {
        (&*self.key).borrow()
    }
}

#[cfg(feature = "id")]
impl Borrow<String> for Key<hr_id::Id> {
    fn borrow(&self) -> &String {
        (&*self.key).borrow()
    }
}

impl Borrow<str> for Key<String> {
    fn borrow(&self) -> &str {
        (&*self.key).borrow()
    }
}

impl<K> Borrow<K> for Key<K> {
    fn borrow(&self) -> &K {
        self.key.borrow()
    }
}

impl<K> From<Key<K>> for Arc<K> {
    fn from(key: Key<K>) -> Self {
        key.key
    }
}

impl<K> From<Key<K>> for Range<K> {
    fn from(key: Key<K>) -> Self {
        Range::One(key.key)
    }
}

impl<K: fmt::Debug> fmt::Debug for Key<K> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.key.fmt(f)
    }
}

impl<K: fmt::Display> fmt::Display for Key<K> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.key.fmt(f)
    }
}

/// A range used to reserve a permit to acquire a transactional lock
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

impl<K, C: Collate<Value = K>> OverlapsRange<Self, C> for Range<K> {
    fn overlaps(&self, other: &Self, collator: &C) -> Overlap {
        match self {
            Self::All => match other {
                Self::All => Overlap::Equal,
                _ => Overlap::Wide,
            },
            this => match other {
                Self::All => Overlap::Narrow,
                Self::One(that) => this.overlaps(&**that, collator),
            },
        }
    }
}

impl<K, C: Collate<Value = K>> OverlapsRange<K, C> for Range<K> {
    fn overlaps(&self, other: &K, collator: &C) -> Overlap {
        match self {
            Self::All => Overlap::Wide,
            Self::One(this) => match collator.cmp(&**this, other) {
                Ordering::Less => Overlap::Less,
                Ordering::Equal => Overlap::Equal,
                Ordering::Greater => Overlap::Greater,
            },
        }
    }
}
