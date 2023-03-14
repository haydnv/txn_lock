use std::marker::PhantomData;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::{fmt, mem};

use collate::{Collate, Overlap, OverlapsRange};
use ds_ext::List;
use futures::future::{self, Future, TryFutureExt};
use futures::try_join;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::Result;

const PERMITS: u32 = u32::MAX >> 3;

type BoxTryFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

struct NodePermit {
    #[allow(unused)]
    permit: OwnedSemaphorePermit,
    children: [Option<Box<NodePermit>>; 5],
}

impl fmt::Debug for NodePermit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let children = self.children.iter().filter_map(|c| c.as_ref()).count();
        write!(f, "permit for node with {} children", children)
    }
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

impl<R: fmt::Debug> fmt::Debug for Permit<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "permit for {:?} with {:?}", self.range, self.permit)
    }
}

/// A node in a 5-ary tree of overlapping ranges
pub struct RangeLock<C, R> {
    range: Arc<R>,
    semaphore: Arc<Semaphore>,
    write: Arc<Mutex<bool>>,

    left: Option<Box<Self>>,
    left_partial: Option<Box<Self>>,
    center: Option<Box<Self>>,
    right_partial: Option<Box<Self>>,
    right: Option<Box<Self>>,

    phantom: PhantomData<C>,
}

impl<C, R> Clone for RangeLock<C, R> {
    fn clone(&self) -> Self {
        Self {
            range: self.range.clone(),
            semaphore: self.semaphore.clone(),
            write: self.write.clone(),

            left: self.left.clone(),
            left_partial: self.left_partial.clone(),
            center: self.center.clone(),
            right_partial: self.right_partial.clone(),
            right: self.right.clone(),

            phantom: PhantomData,
        }
    }
}

impl<C: Send + Sync, R: Send + Sync> RangeLock<C, R> {
    fn new(range: Arc<R>, write: bool) -> Self {
        let semaphore = Arc::new(Semaphore::new(PERMITS as usize));

        Self {
            range,
            semaphore,
            write: Arc::new(Mutex::new(write)),
            left: None,
            left_partial: None,
            center: None,
            right_partial: None,
            right: None,
            phantom: PhantomData,
        }
    }

    #[inline]
    fn reserve(&self, write: bool) {
        let mut flag = self.write.lock().expect("write flag");
        *flag = *flag || write;
    }

    fn acquire(&self) -> BoxTryFuture<NodePermit> {
        Box::pin(async move {
            let permit = self.semaphore.clone().acquire_owned().await?;

            #[inline]
            fn child_lock<C, R>(
                node: Option<&Box<RangeLock<C, R>>>,
            ) -> BoxTryFuture<Option<Box<NodePermit>>>
            where
                C: Send + Sync,
                R: Send + Sync,
            {
                if let Some(node) = node {
                    Box::pin(node.acquire().map_ok(Box::new).map_ok(Some))
                } else {
                    Box::pin(future::ready(Ok(None)))
                }
            }

            let (left, left_partial, center, right_partial, right) = try_join!(
                child_lock(self.left.as_ref()),
                child_lock(self.left_partial.as_ref()),
                child_lock(self.center.as_ref()),
                child_lock(self.right_partial.as_ref()),
                child_lock(self.right.as_ref()),
            )?;

            Ok(NodePermit {
                permit,
                children: [left, left_partial, center, right_partial, right],
            })
        })
    }

    fn try_acquire(&self) -> Result<NodePermit> {
        let permit = self.semaphore.clone().try_acquire_owned()?;

        #[inline]
        fn child_lock<C, R>(node: Option<&Box<RangeLock<C, R>>>) -> Result<Option<Box<NodePermit>>>
        where
            C: Send + Sync,
            R: Send + Sync,
        {
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
                child_lock(self.left_partial.as_ref())?,
                child_lock(self.center.as_ref())?,
                child_lock(self.right_partial.as_ref())?,
                child_lock(self.right.as_ref())?,
            ],
        })
    }

    fn acquire_many(&self, permits: u32) -> BoxTryFuture<NodePermit> {
        Box::pin(async move {
            let permit = self.semaphore.clone().acquire_many_owned(permits).await?;

            #[inline]
            fn child_lock<C, R>(
                node: Option<&Box<RangeLock<C, R>>>,
                permits: u32,
            ) -> BoxTryFuture<Option<Box<NodePermit>>>
            where
                C: Send + Sync,
                R: Send + Sync,
            {
                if let Some(node) = node {
                    Box::pin(node.acquire_many(permits).map_ok(Box::new).map_ok(Some))
                } else {
                    Box::pin(future::ready(Ok(None)))
                }
            }

            let (left, left_partial, center, right_partial, right) = try_join!(
                child_lock(self.left.as_ref(), permits),
                child_lock(self.left_partial.as_ref(), permits),
                child_lock(self.center.as_ref(), permits),
                child_lock(self.right_partial.as_ref(), permits),
                child_lock(self.right.as_ref(), permits),
            )?;

            Ok(NodePermit {
                permit,
                children: [left, left_partial, center, right_partial, right],
            })
        })
    }

    fn try_acquire_many(&self, permits: u32) -> Result<NodePermit> {
        let permit = self.semaphore.clone().try_acquire_many_owned(permits)?;

        #[inline]
        fn child_lock<C, R>(
            node: Option<&Box<RangeLock<C, R>>>,
            permits: u32,
        ) -> Result<Option<Box<NodePermit>>>
        where
            C: Send + Sync,
            R: Send + Sync,
        {
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
                child_lock(self.left_partial.as_ref(), permits)?,
                child_lock(self.center.as_ref(), permits)?,
                child_lock(self.right_partial.as_ref(), permits)?,
                child_lock(self.right.as_ref(), permits)?,
            ],
        })
    }
}

impl<C, R> RangeLock<C, R>
where
    C: Collate + Send + Sync,
    R: OverlapsRange<R, C> + fmt::Debug + Send + Sync,
{
    fn insert<'a>(&'a mut self, collator: &'a C, node: Self) -> &'a Self {
        #[cfg(feature = "logging")]
        log::trace!("range {:?} is part of {:?}", node.range, self.range);

        #[inline]
        fn insert_node<'a, C, R>(
            extant: &'a mut Option<Box<RangeLock<C, R>>>,
            collator: &'a C,
            new_node: RangeLock<C, R>,
        ) -> &'a RangeLock<C, R>
        where
            C: Collate + Send + Sync,
            R: OverlapsRange<R, C> + fmt::Debug + Send + Sync,
        {
            if let Some(extant) = extant {
                extant.insert(collator, new_node)
            } else {
                *extant = Some(Box::new(new_node));
                extant.as_ref().expect("new node")
            }
        }

        match self.range.overlaps(&node.range, collator) {
            Overlap::Equal => {
                let write = node.write.lock().expect("write flag");
                self.reserve(*write);
                self
            }
            Overlap::Greater => insert_node(&mut self.left, collator, node),
            Overlap::WideGreater => insert_node(&mut self.left_partial, collator, node),
            Overlap::Wide => insert_node(&mut self.center, collator, node),
            Overlap::WideLess => insert_node(&mut self.right_partial, collator, node),
            Overlap::Less => insert_node(&mut self.right, collator, node),
            other => unreachable!("insert a range with overlap {:?}", other),
        }
    }

    #[inline]
    fn is_pending_write(&self) -> bool {
        if *self.write.lock().expect("write bit") {
            return true;
        }

        [
            &self.left,
            &self.left_partial,
            &self.center,
            &self.right_partial,
            &self.right,
        ]
        .into_iter()
        .filter_map(|node| node.as_ref())
        .any(|node| node.is_pending_write())
    }

    /// Acquire a read lock on this [`RangeLock`] and its children.
    pub fn read<'a>(&'a self, target: &'a R, collator: &'a C) -> BoxTryFuture<Permit<R>> {
        Box::pin(async move {
            let overlap = self.range.overlaps(target, collator);

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
                Overlap::Greater => {
                    self.left
                        .as_ref()
                        .expect("left")
                        .read(target, collator)
                        .await
                }
                Overlap::WideGreater => {
                    self.left_partial
                        .as_ref()
                        .expect("left")
                        .read(target, collator)
                        .await
                }
                Overlap::Wide => {
                    self.center
                        .as_ref()
                        .expect("center")
                        .read(target, collator)
                        .await
                }
                Overlap::WideLess => {
                    self.right_partial
                        .as_ref()
                        .expect("right")
                        .read(target, collator)
                        .await
                }
                Overlap::Less => {
                    self.right
                        .as_ref()
                        .expect("right")
                        .read(target, collator)
                        .await
                }
                overlap => unreachable!("lock a range with overlap {:?} for reading", overlap),
            }
        })
    }

    /// Acquire a read lock on this [`RangeLock`] and its children synchronously, if possible.
    pub fn try_read(&self, target: &R, collator: &C) -> Result<Permit<R>> {
        let overlap = self.range.overlaps(target, collator);

        if overlap == Overlap::Equal {
            return self.try_acquire().map(|permit| Permit {
                range: self.range.clone(),
                permit,
            });
        }

        // make sure this range is not locked
        let _permit = self.semaphore.try_acquire()?;

        match overlap {
            Overlap::Greater => self.left.as_ref().expect("left").try_read(target, collator),
            Overlap::WideGreater => self
                .left_partial
                .as_ref()
                .expect("left")
                .try_read(target, collator),

            Overlap::Wide => self
                .center
                .as_ref()
                .expect("center")
                .try_read(target, collator),

            Overlap::WideLess => self
                .right_partial
                .as_ref()
                .expect("right")
                .try_read(target, collator),

            Overlap::Less => self
                .right
                .as_ref()
                .expect("right")
                .try_read(target, collator),

            overlap => unreachable!("lock a range with overlap {:?} for reading", overlap),
        }
    }

    /// Acquire a write lock on this [`RangeLock`] and its children.
    pub fn write<'a>(&'a self, target: &'a R, collator: &'a C) -> BoxTryFuture<Permit<R>> {
        Box::pin(async move {
            let overlap = self.range.overlaps(target, collator);

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
                Overlap::Greater => {
                    self.left
                        .as_ref()
                        .expect("left")
                        .write(target, collator)
                        .await
                }
                Overlap::WideGreater => {
                    self.left_partial
                        .as_ref()
                        .expect("left")
                        .write(target, collator)
                        .await
                }
                Overlap::Wide => {
                    self.center
                        .as_ref()
                        .expect("center")
                        .write(target, collator)
                        .await
                }
                Overlap::WideLess => {
                    self.right_partial
                        .as_ref()
                        .expect("right")
                        .write(target, collator)
                        .await
                }
                Overlap::Less => {
                    self.right
                        .as_ref()
                        .expect("right")
                        .write(target, collator)
                        .await
                }
                overlap => unreachable!("lock a range with overlap {:?} for writing", overlap),
            }
        })
    }

    /// Acquire a write lock on this [`RangeLock`] and its children synchronously, if possible.
    pub fn try_write(&self, target: &R, collator: &C) -> Result<Permit<R>> {
        let overlap = self.range.overlaps(target, collator);

        if overlap == Overlap::Equal {
            return self.try_acquire_many(PERMITS).map(|permit| Permit {
                range: self.range.clone(),
                permit,
            });
        }

        // make sure the target range is not locked
        let _permit = self.semaphore.try_acquire()?;

        match overlap {
            Overlap::Greater => self
                .left
                .as_ref()
                .expect("left")
                .try_write(target, collator),

            Overlap::WideGreater => self
                .left_partial
                .as_ref()
                .expect("left")
                .try_write(target, collator),

            Overlap::Wide => self
                .center
                .as_ref()
                .expect("center")
                .try_write(target, collator),

            Overlap::WideLess => self
                .right_partial
                .as_ref()
                .expect("right")
                .try_write(target, collator),

            Overlap::Less => self
                .right
                .as_ref()
                .expect("right")
                .try_write(target, collator),

            overlap => unreachable!("lock a range with overlap {:?} for writing", overlap),
        }
    }
}

impl<C, R: fmt::Debug> fmt::Debug for RangeLock<C, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node {:?}", self.range)
    }
}

/// Semaphores for ranges within a single transactional version
pub struct Version<C, R> {
    collator: Arc<C>,
    roots: List<RangeLock<C, R>>,
}

impl<C, R> Version<C, R> {
    /// Create a new [`Version`] semaphore
    pub fn new(collator: Arc<C>) -> Self {
        Self {
            collator,
            roots: List::with_capacity(1),
        }
    }
}

impl<C: Collate + Send + Sync, R: OverlapsRange<R, C> + fmt::Debug + Send + Sync> Version<C, R> {
    /// Create a new `range` semaphore and return its root [`RangeLock`]
    pub fn insert(&mut self, range: Arc<R>, write: bool) -> RangeLock<C, R> {
        #[cfg(feature = "logging")]
        log::debug!("Version::insert {:?} (write: {})", range, write);

        let insert_at = bisect_left(&self.roots, &range, &self.collator);
        let take_until = bisect_right(&self.roots, &range, &self.collator);
        assert!(take_until >= insert_at);

        #[cfg(feature = "logging")]
        log::debug!(
            "{:?} covers {} existing ranges in this version",
            range,
            take_until - insert_at
        );

        if insert_at == take_until {
            let root = RangeLock::new(range, write);
            self.roots.insert(insert_at, root);
        } else if take_until - insert_at == 1 {
            let root = self.roots.get_mut(insert_at).expect("root range");
            match root.range.overlaps(&range, &self.collator) {
                Overlap::Equal => root.reserve(write),
                Overlap::WideLess | Overlap::Wide | Overlap::WideGreater => {
                    let node = RangeLock::new(range, write);
                    root.insert(&self.collator, node);
                }
                Overlap::Narrow => {
                    mem::drop(root);
                    let node = self.roots.remove(insert_at).expect("root");
                    let mut root = RangeLock::new(range, write);
                    root.insert(&self.collator, node);
                    self.roots.insert(insert_at, root);
                }
                _ => unreachable!(),
            }
        } else {
            let mut root = RangeLock::new(range, write);

            for _ in insert_at..take_until {
                let node = self.roots.remove(insert_at).expect("root");
                root.insert(&self.collator, node);
            }

            self.roots.insert(insert_at, root);
        }

        self.roots.get(insert_at).expect("range").clone()
    }

    /// Return `true` if any part of the given range has been reserved for reading.
    pub fn has_been_read_at(&self, target: &R) -> bool {
        for root in &self.roots {
            if root.range.contains_partial(target, &self.collator) {
                return true;
            }
        }

        false
    }

    /// Return `true` if any part of the given range has been reserved for writing.
    pub fn is_pending_write_at(&self, target: &R) -> bool {
        for root in &self.roots {
            if root.range.contains_partial(target, &self.collator) {
                if root.is_pending_write() {
                    return true;
                }
            }
        }

        false
    }
}

#[inline]
fn bisect_left<'a, C, R>(roots: &'a List<RangeLock<C, R>>, range: &'a R, collator: &'a C) -> usize
where
    C: Collate,
    R: OverlapsRange<R, C> + fmt::Debug + 'a,
{
    if roots.is_empty() {
        return 0;
    } else if roots.front().expect("root").range.overlaps(range, collator) == Overlap::Greater {
        return 0;
    } else if roots.back().expect("root").range.overlaps(range, collator) == Overlap::Less {
        return roots.len();
    }

    let mut start = 0;
    let mut end = roots.len();

    while start < end {
        let mid = (start + end) >> 1;

        let node = roots.get(mid).expect("range");
        if node.range.overlaps(range, collator) == Overlap::Less {
            start = mid + 1;
        } else {
            end = mid;
        }
    }

    start
}

#[inline]
fn bisect_right<'a, C, R>(roots: &'a List<RangeLock<C, R>>, range: &'a R, collator: &'a C) -> usize
where
    C: Collate,
    R: OverlapsRange<R, C> + fmt::Debug + 'a,
{
    if roots.is_empty() {
        return 0;
    } else if roots.front().expect("root").range.overlaps(range, collator) == Overlap::Greater {
        return 0;
    } else if roots.back().expect("root").range.overlaps(range, collator) == Overlap::Less {
        return roots.len();
    }

    let mut start = 0;
    let mut end = roots.len();

    while start < end {
        let mid = (start + end) >> 1;

        let node = roots.get(mid).expect("range");
        if node.range.overlaps(range, collator) == Overlap::Greater {
            end = mid;
        } else {
            start = mid + 1;
        }
    }

    end
}
