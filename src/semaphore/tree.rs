use std::mem;
use std::ops::Range;
use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::Result;

const PERMITS: u32 = u32::MAX >> 3;

/// An [`Overlap`] is the result of a comparison between two ranges,
/// the equivalent of [`std::cmp::Ordering`] for hierarchical data.
#[derive(Eq, PartialEq, Copy, Clone, PartialOrd)]
pub enum Overlap {
    /// A lack of overlap where the compared range is entirely less than another
    Less,

    /// A lack of overlap where the compared range is entirely greater than another
    Greater,

    /// An overlap where the compared range is identical to another
    Equal,

    /// An overlap where the compared range is narrower than another
    Narrow,

    /// An overlap where the compared range is wider than another
    Wide,
}

/// A range supported by a transactional [`Semaphore`]
pub trait Overlaps<T> {
    /// A commutative method which returns `true` if `self` overlaps `other`.
    fn overlaps(&self, other: &T) -> Overlap;
}

impl<Idx: PartialOrd<Idx>> Overlaps<Range<Idx>> for Range<Idx> {
    fn overlaps(&self, other: &Self) -> Overlap {
        assert!(self.start <= self.end);
        assert!(other.start <= other.end);

        if other.end < self.start {
            Overlap::Greater
        } else if other.start > self.end {
            Overlap::Less
        } else if self.start == other.start && self.end == other.end {
            Overlap::Equal
        } else if self.start > other.start && self.end < other.end {
            Overlap::Narrow
        } else {
            Overlap::Wide
        }
    }
}

/// A transactional semaphore permit
pub struct Permit<R> {
    range: Arc<R>,
    permit: OwnedSemaphorePermit,
    left: Option<Box<Self>>,
    center: Option<Box<Self>>,
    right: Option<Box<Self>>,
}

/// A node in a 3-ary tree of overlapping ranges
struct Node<R> {
    range: Arc<R>,
    semaphore: Arc<Semaphore>,

    left: Option<Box<Self>>,
    center: Option<Box<Self>>,
    right: Option<Box<Self>>,
}

impl<R> Node<R> {
    fn new(range: R) -> Self {
        Self {
            range: Arc::new(range),
            semaphore: Arc::new(Semaphore::new(PERMITS as usize)),
            left: None,
            center: None,
            right: None,
        }
    }
}

impl<R> Node<R> {
    fn try_read(&self) -> Result<Permit<R>> {
        fn try_read<R>(node: Option<&Box<Node<R>>>) -> Result<Option<Box<Permit<R>>>> {
            if let Some(node) = node {
                node.try_read().map(Box::new).map(Some)
            } else {
                Ok(None)
            }
        }

        Ok(Permit {
            range: self.range.clone(),
            permit: self.semaphore.clone().try_acquire_owned()?,
            left: try_read(self.left.as_ref())?,
            center: try_read(self.center.as_ref())?,
            right: try_read(self.right.as_ref())?,
        })
    }
}

impl<R: Overlaps<R>> Node<R> {
    fn insert(&mut self, range: R) -> &Node<R> {
        fn insert_range<R: Overlaps<R>>(node: &mut Option<Box<Node<R>>>, range: R) -> &Node<R> {
            if let Some(node) = node {
                node.insert(range)
            } else {
                let child = Node::new(range);
                *node = Some(Box::new(child));
                node.as_ref().expect("new node")
            }
        }

        match self.range.overlaps(&range) {
            Overlap::Equal => self,
            Overlap::Greater => insert_range(&mut self.left, range),
            Overlap::Wide => insert_range(&mut self.center, range),
            Overlap::Less => insert_range(&mut self.right, range),
            Overlap::Narrow => {
                let mut child = Box::new(Node::new(range));
                mem::swap(self, &mut child);
                self.center = Some(child);
                self
            }
        }
    }
}
