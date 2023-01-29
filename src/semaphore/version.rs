use std::fmt;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::Arc;

use futures::future::{self, Future, TryFutureExt};
use futures::try_join;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::Result;

use super::{Overlap, Overlaps};

const PERMITS: u32 = u32::MAX >> 3;

struct NodePermit {
    #[allow(unused)]
    permit: OwnedSemaphorePermit,
    children: [Option<Box<NodePermit>>; 3],
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

/// A node in a 3-ary tree of overlapping ranges
pub struct Node<R> {
    range: Arc<R>,
    semaphore: Arc<Semaphore>,

    left: Option<Box<Self>>,
    center: Option<Box<Self>>,
    right: Option<Box<Self>>,
}

impl<R> Clone for Node<R> {
    fn clone(&self) -> Self {
        Self {
            range: self.range.clone(),
            semaphore: self.semaphore.clone(),

            left: self.left.clone(),
            center: self.center.clone(),
            right: self.right.clone(),
        }
    }
}

impl<R> Node<R> {
    fn new(range: Arc<R>) -> Self {
        let semaphore = Arc::new(Semaphore::new(PERMITS as usize));

        Self {
            range,
            semaphore,
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
                permit: permit,
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
            permit: permit,
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
                permit: permit,
                children: [left, center, right],
            })
        })
    }

    fn try_acquire_many(&self, permits: u32) -> Result<NodePermit> {
        let permit = self.semaphore.clone().try_acquire_many_owned(permits)?;

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
            other => unreachable!("insert a range with overlap {:?}", other),
        }
    }

    /// Acquire a read lock on this [`Node`] and its children.
    pub fn read<'a>(
        &'a self,
        target: &'a R,
    ) -> Pin<Box<dyn Future<Output = Result<Permit<R>>> + 'a>> {
        Box::pin(async move {
            let overlap = self.range.overlaps(target);

            if overlap == Overlap::Equal {
                return self
                    .acquire()
                    .map_ok(|permit| Permit {
                        range: self.range.clone(),
                        permit: permit,
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

    /// Acquire a read lock on this [`Node`] and its children synchronously, if possible.
    pub fn try_read(&self, target: &R) -> Result<Permit<R>> {
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

    /// Acquire a write lock on this [`Node`] and its children.
    pub fn write<'a>(
        &'a self,
        target: &'a R,
    ) -> Pin<Box<dyn Future<Output = Result<Permit<R>>> + 'a>> {
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

    /// Acquire a write lock on this [`Node`] and its children synchronously, if possible.
    pub fn try_write(&self, target: &R) -> Result<Permit<R>> {
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

impl<R: fmt::Debug> fmt::Debug for Node<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node {:?}", self.range)
    }
}

/// Semaphores for ranges within a single transactional version
pub struct Version<R> {
    roots: Vec<Node<R>>,
}

impl<R> Version<R> {
    /// Create a new [`Version`] semaphore
    pub fn new() -> Self {
        Self { roots: Vec::new() }
    }

    /// Create a new [`Version`] semaphore with memory allocated for `capacity` roots
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            roots: Vec::with_capacity(capacity),
        }
    }
}

impl<R: Overlaps<R> + fmt::Debug> Version<R> {
    /// Create a new `range` semaphore and return its root [`Node`]
    pub fn insert(&mut self, range: Arc<R>) -> Node<R> {
        let insert_at = bisect_left(&self.roots, &range);
        let take_until = bisect_right(&self.roots, &range);
        assert!(take_until >= insert_at);

        if insert_at == take_until {
            let root = Node::new(range);
            self.roots.insert(insert_at, root);
        } else if take_until - insert_at == 1 {
            match self.roots[insert_at].range.overlaps(&range) {
                Overlap::Equal => {}
                Overlap::WideLess | Overlap::Wide | Overlap::WideGreater => {
                    let node = Node::new(range);
                    self.roots[insert_at].insert(node);
                }
                Overlap::Narrow => {
                    let node = self.roots.remove(insert_at);
                    let mut root = Node::new(range);
                    root.insert(node);
                    self.roots.insert(insert_at, root);
                }
                _ => unreachable!(),
            }
        } else {
            let mut root = Node::new(range);

            for _ in insert_at..take_until {
                let node = self.roots.remove(insert_at);
                root.insert(node);
            }

            self.roots.insert(insert_at, root);
        }

        self.roots[insert_at].clone()
    }
}

fn bisect_left<'a, R>(roots: &'a [Node<R>], range: &'a R) -> usize
where
    R: Overlaps<R> + fmt::Debug + 'a,
{
    let mut start = 0;
    let mut end = roots.len();

    while start < end {
        let mid = (start + end) / 2;

        let node = &roots[mid];
        if node.range.overlaps(range) == Overlap::Less {
            start = mid + 1;
        } else {
            end = mid;
        }
    }

    start
}

fn bisect_right<'a, R>(roots: &'a [Node<R>], range: &'a R) -> usize
where
    R: Overlaps<R> + fmt::Debug + 'a,
{
    let mut start = 0;
    let mut end = roots.len();

    while start < end {
        let mid = (end - start) / 2;

        let node = &roots[mid];
        if node.range.overlaps(range) == Overlap::Greater {
            println!("{:?} is greater than {:?}", node, range);
            end = mid;
        } else {
            println!("{:?} is less than or equal to than {:?}", node, range);
            start = mid + 1;
        }
    }

    end
}
