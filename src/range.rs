use std::cmp::Ordering;
use std::sync::Arc;

use super::semaphore::{Overlap, Overlaps};

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
