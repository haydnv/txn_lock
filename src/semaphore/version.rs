use std::ops::{Deref, Range};
use std::pin::Pin;
use std::sync::Arc;

use futures::future::{self, Future, TryFutureExt};
use futures::try_join;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::Result;

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
    /// Check whether `other` lies entirely within `self`.
    fn contains(&self, other: &T) -> bool {
        self.overlaps(other) == Overlap::Wide
    }

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

struct NodePermit {
    permit: OwnedSemaphorePermit,
    children: [Option<Box<NodePermit>>; 3],
}

/// A transactional semaphore permit
pub struct Permit<R> {
    range: Arc<R>,
    permit: NodePermit,
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

    fn acquire(&self) -> Pin<Box<dyn Future<Output = Result<NodePermit>> + '_>> {
        Box::pin(async move {
            let permit = self.semaphore.clone().acquire_owned().await?;

            fn child_lock<R>(
                node: Option<&Box<Node<R>>>,
            ) -> Pin<Box<dyn Future<Output = Result<Option<Box<NodePermit>>>> + '_>> {
                if let Some(node) = node {
                    Box::pin(node.acquire().map_ok(Box::new).map_ok(Some))
                } else {
                    Box::pin(future::ready(Ok(None)))
                }
            }

            let (left, center, right) = try_join!(
                child_lock(self.left.as_ref()),
                child_lock(self.center.as_ref()),
                child_lock(self.right.as_ref()),
            )?;

            Ok(NodePermit {
                permit,
                children: [left, center, right],
            })
        })
    }

    fn try_acquire(&self) -> Result<NodePermit> {
        let permit = self.semaphore.clone().try_acquire_owned()?;

        fn child_lock<R>(node: Option<&Box<Node<R>>>) -> Result<Option<Box<NodePermit>>> {
            if let Some(node) = node {
                node.try_acquire().map(Box::new).map(Some)
            } else {
                Ok(None)
            }
        }

        Ok(NodePermit {
            permit,
            children: [
                child_lock(self.left.as_ref())?,
                child_lock(self.center.as_ref())?,
                child_lock(self.right.as_ref())?,
            ],
        })
    }

    fn acquire_many(&self, permits: u32) -> Pin<Box<dyn Future<Output = Result<NodePermit>> + '_>> {
        Box::pin(async move {
            let permit = self.semaphore.clone().acquire_many_owned(permits).await?;

            fn child_lock<R>(
                node: Option<&Box<Node<R>>>,
                permits: u32,
            ) -> Pin<Box<dyn Future<Output = Result<Option<Box<NodePermit>>>> + '_>> {
                if let Some(node) = node {
                    Box::pin(node.acquire_many(permits).map_ok(Box::new).map_ok(Some))
                } else {
                    Box::pin(future::ready(Ok(None)))
                }
            }

            let (left, center, right) = try_join!(
                child_lock(self.left.as_ref(), permits),
                child_lock(self.center.as_ref(), permits),
                child_lock(self.right.as_ref(), permits),
            )?;

            Ok(NodePermit {
                permit,
                children: [left, center, right],
            })
        })
    }

    fn try_acquire_many(&self, permits: u32) -> Result<NodePermit> {
        let permit = self.semaphore.clone().try_acquire_owned()?;

        fn child_lock<R>(
            node: Option<&Box<Node<R>>>,
            permits: u32,
        ) -> Result<Option<Box<NodePermit>>> {
            if let Some(node) = node {
                node.try_acquire_many(permits).map(Box::new).map(Some)
            } else {
                Ok(None)
            }
        }

        Ok(NodePermit {
            permit,
            children: [
                child_lock(self.left.as_ref(), permits)?,
                child_lock(self.center.as_ref(), permits)?,
                child_lock(self.right.as_ref(), permits)?,
            ],
        })
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

    fn read<'a>(&'a self, target: &'a R) -> Pin<Box<dyn Future<Output = Result<Permit<R>>> + 'a>> {
        Box::pin(async move {
            let overlap = self.range.overlaps(target);

            if overlap == Overlap::Equal {
                return self
                    .acquire()
                    .map_ok(|permit| Permit {
                        range: self.range.clone(),
                        permit,
                    })
                    .await;
            }

            // make sure this range is not locked
            let _permit = self.semaphore.acquire().await;

            match overlap {
                Overlap::WideGreater => self.left.as_ref().expect("left").read(target).await,
                Overlap::Wide => self.center.as_ref().expect("center").read(target).await,
                Overlap::WideLess => {
                    self.right
                        .as_ref()
                        .expect("branch right")
                        .read(target)
                        .await
                }
                _ => unreachable!(),
            }
        })
    }

    fn try_read(&self, target: &R) -> Result<Permit<R>> {
        let overlap = self.range.overlaps(target);

        if overlap == Overlap::Equal {
            return self.try_acquire().map(|permit| Permit {
                range: self.range.clone(),
                permit,
            });
        }

        // make sure this range is not locked
        let _permit = self.semaphore.try_acquire()?;

        match overlap {
            Overlap::WideGreater => self.left.as_ref().expect("left").try_read(target),
            Overlap::Wide => self.center.as_ref().expect("center").try_read(target),
            Overlap::WideLess => self.right.as_ref().expect("branch right").try_read(target),
            _ => unreachable!(),
        }
    }

    fn write<'a>(&'a self, target: &'a R) -> Pin<Box<dyn Future<Output = Result<Permit<R>>> + 'a>> {
        Box::pin(async move {
            let overlap = self.range.overlaps(target);

            if overlap == Overlap::Equal {
                return self
                    .acquire_many(PERMITS)
                    .map_ok(|permit| Permit {
                        range: self.range.clone(),
                        permit,
                    })
                    .await;
            }

            // make sure the target range is not locked
            let _permit = self.semaphore.acquire().await;

            match overlap {
                Overlap::WideGreater => self.left.as_ref().expect("left").write(target).await,
                Overlap::Wide => self.center.as_ref().expect("center").write(target).await,
                Overlap::WideLess => self.right.as_ref().expect("right").write(target).await,
                _ => unreachable!(),
            }
        })
    }

    fn try_write(&self, target: &R) -> Result<Permit<R>> {
        let overlap = self.range.overlaps(target);

        if overlap == Overlap::Equal {
            return self.try_acquire_many(PERMITS).map(|permit| Permit {
                range: self.range.clone(),
                permit,
            });
        }

        // make sure the target range is not locked
        let _permit = self.semaphore.try_acquire()?;

        match overlap {
            Overlap::WideGreater => self.left.as_ref().expect("left").try_write(target),
            Overlap::Wide => self.center.as_ref().expect("center").try_write(target),
            Overlap::WideLess => self.right.as_ref().expect("branch right").try_write(target),
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

    fn insert(&mut self, range: R) -> (usize, Arc<R>) {
        let lo = self.bisect_left(&range);
        let hi = self.bisect_right(&range);

        let range = match (lo, hi) {
            (l, r) if l == r => {
                let node = Node::new(range);
                let range = node.range.clone();
                self.roots.insert(l, node);
                range
            }
            (l, r) if l == r - 1 => match self.roots[l].range.overlaps(&range) {
                Overlap::Equal => self.roots[l].range.clone(),
                Overlap::Wide => {
                    let node = Node::new(range);
                    let range = node.range.clone();
                    self.roots[l].insert(node);
                    range
                }
                _ => {
                    let mut root = Node::new(range);
                    let range = root.range.clone();
                    let node = self.roots.remove(l);
                    root.insert(node);
                    self.roots.insert(l, root);
                    range
                }
            },
            (l, mut r) => {
                let mut root = Node::new(range);
                let range = root.range.clone();
                while r > l {
                    let node = self.roots.remove(l);
                    root.insert(node);
                    r -= 1;
                }
                self.roots.insert(l, root);
                range
            }
        };

        (lo, range)
    }

    fn try_read(&mut self, range: R) -> Result<Permit<R>> {
        let (i, range) = self.insert(range);
        let root = &self.roots[i];
        root.try_read(&range)
    }

    async fn read(&mut self, range: R) -> Result<Permit<R>> {
        let (i, range) = self.insert(range);
        let root = &self.roots[i];
        root.read(&range).await
    }

    fn try_write(&mut self, range: R) -> Result<Permit<R>> {
        let (i, range) = self.insert(range);
        let root = &self.roots[i];
        root.try_write(&range)
    }

    async fn write(&mut self, range: R) -> Result<Permit<R>> {
        let (i, range) = self.insert(range);
        let root = &self.roots[i];
        root.write(&range).await
    }
}
