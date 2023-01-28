use std::ops::{Deref, Range};
use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

const PERMITS: u32 = u32::MAX >> 3;

/// An [`Overlap`] is the result of a comparison between two ranges,
/// the equivalent of [`Ordering`] for hierarchical data.
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

    /// An overlap where the compared range is wider than another on both sides
    Wide,

    /// An overlap where the compared range is wider than another with a lesser start and end point
    WideLess,

    /// An overlap where the compared range is wider than another with a greater start and end point
    WideGreater,
}

/// A range supported by a transactional [`Semaphore`]
pub trait Overlaps<T> {
    /// Check whether `self` overlaps `other`.
    ///
    /// Examples:
    /// ```
    /// // TODO: enable
    /// //assert_eq!((0..1).overlaps(&(2..5)), Overlap::Less);
    /// //assert_eq!((0..1).overlaps(&(0..1)), Overlap::Equal);
    /// //assert_eq!((3..5).overlaps(&(1..7)), Overlap::Narrow);
    /// //assert_eq!((1..7).overlaps(&(3..5)), Overlap::Wide);
    /// //assert_eq!((1..4).overlaps(&(3..5)), Overlap::WideLeft);
    /// //assert_eq!((3..5).overlaps(&(1..4)), Overlap::WideRight);
    /// ```
    fn overlaps(&self, other: &T) -> Overlap;
}

impl<Idx: PartialOrd<Idx>> Overlaps<Range<Idx>> for Range<Idx> {
    fn overlaps(&self, other: &Self) -> Overlap {
        if self.start == other.start && self.end == other.end {
            return Overlap::Equal;
        }

        match (self.contains(&other.start), self.contains(&other.end)) {
            (true, true) => Overlap::Wide,
            (true, false) => Overlap::WideLess,
            (false, true) => Overlap::WideGreater,
            (false, false) => {
                if self.start > other.end {
                    Overlap::Greater
                } else if self.end <= other.start {
                    Overlap::Less
                } else {
                    debug_assert!(self.start > other.start && self.end < other.end);
                    Overlap::Narrow
                }
            }
        }
    }
}

/// A transactional semaphore permit
pub struct Permit<R> {
    range: Arc<R>,
    permit: OwnedSemaphorePermit,
}

impl<R> Deref for Permit<R> {
    type Target = R;

    fn deref(&self) -> &Self::Target {
        self.range.deref()
    }
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

impl<R: Overlaps<R>> Node<R> {
    fn insert(&mut self, node: Self) -> &Self {
        #[inline]
        fn insert_node<R: Overlaps<R>>(
            extant: &mut Option<Box<Node<R>>>,
            new_node: Node<R>,
        ) -> &Node<R> {
            if let Some(extant) = extant {
                extant.insert(new_node)
            } else {
                *extant = Some(Box::new(new_node));
                extant.as_ref().expect("new node")
            }
        }

        match self.range.overlaps(&node.range) {
            Overlap::WideGreater => insert_node(&mut self.left, node),
            Overlap::Wide => insert_node(&mut self.center, node),
            Overlap::WideLess => insert_node(&mut self.right, node),
            _ => unreachable!(),
        }
    }
}

struct Version<R> {
    roots: Vec<Node<R>>,
}

impl<R: Overlaps<R>> Version<R> {
    fn bisect_left(&self, range: &R) -> usize {
        let mut lo = 0;
        let mut hi = self.roots.len();

        while hi > lo {
            let mid = (hi - lo) / 2;
            match self.roots[mid].range.overlaps(range) {
                Overlap::Less => lo = mid,
                _ => hi = mid,
            }
        }

        lo
    }

    fn bisect_right(&self, range: &R) -> usize {
        let mut lo = 0;
        let mut hi = self.roots.len();

        while hi > lo {
            let mid = (hi - lo) / 2;
            match self.roots[mid].range.overlaps(range) {
                Overlap::Greater => hi = mid,
                _ => lo = mid,
            }
        }

        hi
    }

    fn insert(&mut self, range: R) -> &Node<R> {
        let lo = self.bisect_left(&range);
        let hi = self.bisect_right(&range);

        match (lo, hi) {
            (l, r) if l == r => {
                self.roots.insert(l, Node::new(range));
            }
            (l, r) if l == r - 1 => match self.roots[l].range.overlaps(&range) {
                Overlap::Equal => {}
                Overlap::Wide => return self.roots[l].insert(Node::new(range)),
                _ => {
                    let mut root = Node::new(range);
                    let node = self.roots.remove(l);
                    root.insert(node);
                    self.roots.insert(l, root);
                }
            },
            (l, mut r) => {
                let mut root = Node::new(range);
                while r > l {
                    let node = self.roots.remove(l);
                    root.insert(node);
                    r -= 1;
                }
                self.roots.insert(l, root);
            }
        }

        &self.roots[lo]
    }
}
