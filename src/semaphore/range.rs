use std::mem;
use std::ops::Range;
use tokio::sync::Semaphore;

const PERMITS: u32 = u32::MAX >> 3;

/// An [`Overlap`] is the result of a comparison between two ranges
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

/// A node in a 3-ary tree of overlapping ranges
struct Node<R> {
    range: R,
    semaphore: Semaphore,

    left: Option<Box<Node<R>>>,
    center: Option<Box<Node<R>>>,
    right: Option<Box<Node<R>>>,
}

impl<R> Node<R> {
    fn new(range: R) -> Self {
        Self {
            range,
            semaphore: Semaphore::new(PERMITS as usize),
            left: None,
            center: None,
            right: None,
        }
    }
}

impl<R: Overlaps<R>> Node<R> {
    fn insert(self: &mut Box<Self>, range: R) {
        fn insert_range<R: Overlaps<R>>(node: &mut Option<Box<Node<R>>>, range: R) {
            if let Some(node) = node {
                node.insert(range)
            } else {
                let child = Node::new(range);
                *node = Some(Box::new(child));
            }
        }

        match self.range.overlaps(&range) {
            Overlap::Less => insert_range(&mut self.left, range),
            Overlap::Greater => insert_range(&mut self.right, range),
            Overlap::Wide => insert_range(&mut self.center, range),
            Overlap::Narrow => {
                let mut child = Box::new(Node::new(range));
                mem::swap(self, &mut child);
                self.center = Some(child);
            }
            Overlap::Equal => {}
        }
    }
}
