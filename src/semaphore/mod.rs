//! A futures-aware semaphore used to maintain the ACID compliance of a mutable data store.
//!
//! This differs from [`tokio::sync::Semaphore`] by tracking read and write reservations
//! rather than a single atomic integer. This is because of the additional logical constraints of
//! a transactional resource--
//! a write permit blocks future read permits and a read permit conflicts with past write permits.
//!
//! More information: [https://en.wikipedia.org/wiki/ACID](https://en.wikipedia.org/wiki/ACID)
//!
//! Example:
//! ```
//! use std::ops::Range;
//! use txn_lock::semaphore::*;
//! use txn_lock::Error;
//!
//! let semaphore = Semaphore::<u64, Range<usize>>::new();
//!
//! // Multiple overlapping read permits within a transaction are fine
//! let permit0_1 = semaphore.try_read(0, 0..1).expect("read 0..1");
//! let permit0_2 = semaphore.try_read(0, 0..1).expect("read 0..1");
//! // And non-overlapping write permits are fine
//! let permit0_3 = semaphore.try_write(0, 2..3).expect("write 2..3");
//! // But an overlapping write permit in the same transaction will block
//! assert_eq!(semaphore.try_write(0, 0..2).unwrap_err(), Error::WouldBlock);
//! std::mem::drop(permit0_1);
//! assert_eq!(semaphore.try_write(0, 0..2).unwrap_err(), Error::WouldBlock);
//! std::mem::drop(permit0_2);
//! // Until all overlapping permits are dropped
//! let permit0_3 = semaphore.try_write(0, 0..2).expect("write 0..2");
//!
//! // Finalizing a transaction will un-block permits for later transactions
//! // It's the caller's responsibility to make sure that data can't be mutated after finalizing
//! semaphore.finalize(&0, false);
//!
//! // Now permits for later transactions are un-blocked
//! let permit1 = semaphore.try_read(1, 1..2).expect("permit");
//! // Acquiring a write permit is fine even if there's a read permit in the past
//! let permit2 = semaphore.try_write(2, 1..3).expect("permit");
//! // But it will block all permits for later transactions
//! assert_eq!(semaphore.try_read(3, 1..4).unwrap_err(), Error::WouldBlock);
//!
//! // To prevent a memory leak, finalize all transactions earlier than the given ID
//! semaphore.finalize(&2, true);
//!
//! // Now later permits are un-blocked
//! let permit3 = semaphore.try_write(3, 1..4).expect("permit");
//! // And trying to write-lock the past will result in a conflict error
//! assert_eq!(semaphore.try_write(2, 2..3).unwrap_err(), Error::Conflict);
//!
//! // It's still allowed to acquire a write permit for later transactions
//! // if the range doesn't overlap with any other reservations
//! let permit4 = semaphore.try_write(4, 4..5).expect("permit");
//! ```
//!
//! [`tokio::sync::Semaphore`]: https://docs.rs/tokio/latest/tokio/sync/struct.Semaphore.html

mod version;

use std::cmp::Ordering;
use std::collections::{btree_map, BTreeMap};
use std::fmt;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use collate::Overlaps;
use tokio::sync::Notify;

use super::{Error, Result};

use version::{Permit as VersionPermit, RangeLock, Version};

/// A permit to read a specific section of a transactional resource
#[derive(Debug)]
pub struct PermitRead<R> {
    permit: VersionPermit<R>,
    notify: Arc<Notify>,
}

impl<R> Deref for PermitRead<R> {
    type Target = R;

    fn deref(&self) -> &Self::Target {
        self.permit.deref()
    }
}

impl<R> Drop for PermitRead<R> {
    fn drop(&mut self) {
        self.notify.notify_waiters()
    }
}

/// A permit to write to a specific section of a transactional resource
#[derive(Debug)]
pub struct PermitWrite<R> {
    permit: VersionPermit<R>,
    notify: Arc<Notify>,
}

impl<R> Deref for PermitWrite<R> {
    type Target = R;

    fn deref(&self) -> &Self::Target {
        self.permit.deref()
    }
}

impl<R> Drop for PermitWrite<R> {
    fn drop(&mut self) {
        self.notify.notify_waiters()
    }
}

enum VersionRead<I, R> {
    Version(Arc<R>, RangeLock<R>),
    Pending(Arc<R>, I),
}

/// A semaphore used to maintain the ACID compliance of transactional resource
pub struct Semaphore<I, R> {
    versions: Arc<Mutex<BTreeMap<I, Version<R>>>>,
    notify: Arc<Notify>,
}

impl<I, R> Clone for Semaphore<I, R> {
    fn clone(&self) -> Self {
        Self {
            versions: self.versions.clone(),
            notify: self.notify.clone(),
        }
    }
}

impl<I: Ord, R: Overlaps<R> + fmt::Debug> Semaphore<I, R> {
    /// Construct a new transactional [`Semaphore`].
    pub fn new() -> Self {
        Self {
            versions: Arc::new(Mutex::new(BTreeMap::new())),
            notify: Arc::new(Notify::new()),
        }
    }

    /// Construct a new transactional [`Semaphore`] with a write reservation for its initial value.
    pub fn with_reservation(txn_id: I, range: R) -> Self {
        let mut version = Version::new();
        version.insert(Arc::new(range), true);

        let mut versions: BTreeMap<I, Version<R>> = BTreeMap::new();
        versions.insert(txn_id, version);

        Self {
            versions: Arc::new(Mutex::new(BTreeMap::new())),
            notify: Arc::new(Notify::new()),
        }
    }

    fn read_inner(&self, txn_id: I, range: Arc<R>) -> VersionRead<I, R> {
        let mut versions = self.versions.lock().expect("versions");

        // check if there's an overlapping write lock in the past
        for (version_id, version) in versions.iter_mut() {
            match txn_id.cmp(version_id) {
                Ordering::Greater => {
                    if version.is_pending_write_at(&range) {
                        // if there's a write lock in the past, wait it out
                        return VersionRead::Pending(range, txn_id);
                    }
                }
                Ordering::Equal => {
                    let root = version.insert(range.clone(), false);
                    return VersionRead::Version(range, root);
                }
                Ordering::Less => break,
            }
        }

        let mut version = Version::new();
        let root = version.insert(range.clone(), false);
        versions.insert(txn_id, version);

        VersionRead::Version(range, root)
    }

    /// Acquire a permit to read a section of a transactional resource, if possible.
    pub async fn read(&self, mut txn_id: I, range: R) -> Result<PermitRead<R>> {
        let mut range = Arc::new(range);

        loop {
            match self.read_inner(txn_id, range) {
                VersionRead::Pending(r, id) => {
                    txn_id = id;
                    range = r;
                }
                VersionRead::Version(range, root) => {
                    return Ok(PermitRead {
                        permit: root.read(&range).await?,
                        notify: self.notify.clone(),
                    })
                }
            }

            self.notify.notified().await;
        }
    }

    /// Synchronously acquire a permit to read a section of transactional resource, if possible.
    pub fn try_read(&self, txn_id: I, range: R) -> Result<PermitRead<R>> {
        let range = Arc::new(range);

        match self.read_inner(txn_id, range) {
            VersionRead::Pending(_, _) => Err(Error::WouldBlock),
            VersionRead::Version(range, root) => root
                .try_read(&range)
                .map(|permit| PermitRead {
                    permit,
                    notify: self.notify.clone(),
                })
                .map_err(Error::from),
        }
    }

    fn write_inner(&self, txn_id: I, range: Arc<R>) -> Result<VersionRead<I, R>> {
        let mut versions = self.versions.lock().expect("versions");

        // handle creating a new version
        for (version_id, version) in versions.iter() {
            match txn_id.cmp(version_id) {
                Ordering::Greater => {
                    if version.is_pending_write_at(&range) {
                        // if there's a write lock in the past, wait it out
                        return Ok(VersionRead::Pending(range, txn_id));
                    }
                }
                Ordering::Equal => {
                    // this case is handled below
                }
                Ordering::Less => {
                    if version.has_been_read_at(&range) {
                        // can't allow writes to this range, there's already a future version
                        return Err(Error::Conflict);
                    }
                }
            }
        }

        match versions.entry(txn_id) {
            btree_map::Entry::Occupied(mut entry) => {
                let version = entry.get_mut();
                let root = version.insert(range.clone(), true);
                Ok(VersionRead::Version(range, root))
            }
            btree_map::Entry::Vacant(entry) => {
                let mut version = Version::new();
                let root = version.insert(range.clone(), true);
                entry.insert(version);

                Ok(VersionRead::Version(range, root))
            }
        }
    }

    /// Acquire a permit to write to a section of transactional resource, if possible.
    pub async fn write(&self, mut txn_id: I, range: R) -> Result<PermitWrite<R>> {
        let mut range = Arc::new(range);

        loop {
            match self.write_inner(txn_id, range)? {
                VersionRead::Pending(r, id) => {
                    txn_id = id;
                    range = r;
                }
                VersionRead::Version(range, root) => {
                    let permit = root.write(&range).await?;

                    return Ok(PermitWrite {
                        permit,
                        notify: self.notify.clone(),
                    });
                }
            }

            self.notify.notified().await;
        }
    }

    /// Synchronously acquire a permit to write to a section of transactional resource, if possible.
    pub fn try_write(&self, txn_id: I, range: R) -> Result<PermitWrite<R>> {
        let range = Arc::new(range);

        match self.write_inner(txn_id, range)? {
            VersionRead::Pending(_, _) => Err(Error::WouldBlock),
            VersionRead::Version(range, root) => root
                .try_write(&range)
                .map(|permit| PermitWrite {
                    permit,
                    notify: self.notify.clone(),
                })
                .map_err(Error::from),
        }
    }

    /// Mark a transaction completed, un-blocking waiting requests for future permits.
    ///
    /// Call this when the transactional resource is no longer writable at `txn_id`
    /// (due to a commit, rollback, timeout, or any other reason).
    ///
    /// Set `drop_past` to `true` to finalize all transactions older than `txn_id`.
    pub fn finalize(&self, txn_id: &I, drop_past: bool)
    where
        I: Copy,
    {
        let mut versions = self.versions.lock().expect("versions");

        let notify = if drop_past {
            let mut notify = false;

            while let Some(version_id) = versions.keys().next().copied() {
                if &version_id > txn_id {
                    break;
                } else {
                    assert!(versions.remove(&version_id).is_some());
                    notify = true;
                }
            }

            notify
        } else {
            versions.remove(txn_id).is_some()
        };

        if notify {
            self.notify.notify_waiters();
        }
    }
}
