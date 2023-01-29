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

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::{btree_map, BTreeMap};
use std::fmt;
use std::ops::{Deref, Range};
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;

use super::{Error, Result};

use version::{Node, Permit as VersionPermit, Version as VersionSemaphore};

/// An [`Overlap`] is the result of a comparison between two ranges,
/// the equivalent of [`Ordering`] for hierarchical data.
#[derive(Debug, Eq, PartialEq, Copy, Clone, PartialOrd)]
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
        match self.overlaps(other) {
            Overlap::Wide | Overlap::Equal => true,
            _ => false,
        }
    }

    /// Check whether `other` lies at least partially within `self`.
    fn contains_partial(&self, other: &T) -> bool {
        match self.overlaps(other) {
            Overlap::Narrow | Overlap::WideLess | Overlap::Wide | Overlap::WideGreater => true,
            _ => false,
        }
    }

    /// Check whether `self` overlaps `other`.
    ///
    /// Examples:
    /// ```
    /// use txn_lock::semaphore::{Overlap, Overlaps};
    /// assert_eq!((0..1).overlaps(&(2..5)), Overlap::Less);
    /// assert_eq!((0..1).overlaps(&(0..1)), Overlap::Equal);
    /// assert_eq!((2..3).overlaps(&(0..2)), Overlap::Greater);
    /// assert_eq!((3..5).overlaps(&(1..7)), Overlap::Narrow);
    /// assert_eq!((1..7).overlaps(&(3..5)), Overlap::Wide);
    /// assert_eq!((1..4).overlaps(&(3..5)), Overlap::WideLess);
    /// assert_eq!((3..5).overlaps(&(1..4)), Overlap::WideGreater);
    /// ```
    fn overlaps(&self, other: &T) -> Overlap;
}

impl<Idx: PartialOrd<Idx>> Overlaps<Range<Idx>> for Range<Idx> {
    fn overlaps(&self, other: &Self) -> Overlap {
        assert!(self.end >= self.start);
        assert!(other.end >= other.start);

        if self.start >= other.end {
            Overlap::Greater
        } else if self.end <= other.start {
            Overlap::Less
        } else if self.start == other.start && self.end == other.end {
            Overlap::Equal
        } else if self.start <= other.start && self.end >= other.end {
            Overlap::Wide
        } else if self.start >= other.start && self.end <= other.end {
            Overlap::Narrow
        } else if self.end > other.end {
            Overlap::WideGreater
        } else if self.start < other.start {
            Overlap::WideLess
        } else {
            unreachable!()
        }
    }
}

/// A permit to read a specific section of a transactional resource
#[derive(Debug)]
pub struct Permit<R> {
    permit: VersionPermit<R>,
    notify: Arc<Notify>,
}

impl<R> Deref for Permit<R> {
    type Target = R;

    fn deref(&self) -> &Self::Target {
        self.permit.deref()
    }
}

impl<R> Drop for Permit<R> {
    fn drop(&mut self) {
        self.notify.notify_waiters()
    }
}

enum Reservation<R> {
    Read(Arc<R>),
    Write(Arc<R>),
}

impl<R> Reservation<R> {
    fn range(&self) -> &Arc<R> {
        match self {
            Self::Read(range) => range,
            Self::Write(range) => range,
        }
    }
}

struct Version<R> {
    semaphore: VersionSemaphore<R>,
    reservations: Vec<Reservation<R>>,
}

impl<R> Version<R> {
    fn new(reservations: Vec<Reservation<R>>) -> Self {
        let semaphore = if reservations.is_empty() {
            VersionSemaphore::new()
        } else {
            VersionSemaphore::with_capacity(reservations.len())
        };

        Self {
            semaphore,
            reservations,
        }
    }

    fn with_reservation(reservation: Reservation<R>) -> Self {
        Self::new(vec![reservation])
    }
}

impl<R> FromIterator<R> for Version<R> {
    fn from_iter<T: IntoIterator<Item = R>>(reserve: T) -> Self {
        let reserved = reserve
            .into_iter()
            .map(|range| Reservation::Write(Arc::new(range)))
            .collect();

        Self::new(reserved)
    }
}

enum VersionRead<I, R> {
    Version(Arc<R>, Node<R>),
    Pending(Arc<R>, I),
}

enum RangeRead<I, R> {
    Range(Arc<R>),
    Pending(I),
    None,
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

    /// Construct a new transactional [`Semaphore`] with write reservations for its initial value.
    pub fn with_reservations<W: IntoIterator<Item = R>>(txn_id: I, reserve: W) -> Self {
        let mut versions: BTreeMap<I, Version<R>> = BTreeMap::new();
        versions.insert(txn_id, reserve.into_iter().collect());

        Self {
            versions: Arc::new(Mutex::new(BTreeMap::new())),
            notify: Arc::new(Notify::new()),
        }
    }

    fn maybe_read_inner<Q>(&self, txn_id: I, range: &Q) -> RangeRead<I, R>
    where
        R: Overlaps<Q> + PartialEq<Q>,
    {
        let versions = self.versions.lock().expect("versions");

        // check if there's already a reservation at the requested version
        if let Some(version) = versions.get(&txn_id) {
            for res in &version.reservations {
                if (&**res.range()).borrow() == range {
                    return RangeRead::Range(res.range().clone());
                }
            }
        }

        // if there's an overlapping write lock, wait it out
        // (i.e. don't leak information about an un-committed version)
        let mut found = RangeRead::None;
        for (_, version) in versions.iter().take_while(|(id, _)| *id < &txn_id) {
            for reservation in &version.reservations {
                match reservation {
                    Reservation::Read(locked) => {
                        if (&**locked).borrow() == range {
                            found = RangeRead::Range(locked.clone());
                        }
                    }
                    Reservation::Write(locked) if locked.contains_partial(range) => {
                        return RangeRead::Pending(txn_id)
                    }
                    Reservation::Write(_locked) => {}
                }
            }
        }

        found
    }

    /// Return a reserved range, if any is accessible at the given `txn_id`.
    pub async fn maybe_range<Q>(&self, mut txn_id: I, range: &Q) -> Option<Arc<R>>
    where
        R: Overlaps<Q> + PartialEq<Q>,
    {
        loop {
            txn_id = match self.maybe_read_inner(txn_id, range) {
                RangeRead::None => return None,
                RangeRead::Range(range) => return Some(range),
                RangeRead::Pending(id) => id,
            };

            self.notify.notified().await;
        }
    }

    fn read_inner(&self, txn_id: I, range: Arc<R>) -> VersionRead<I, R> {
        let mut versions = self.versions.lock().expect("versions");

        if let Some(version) = versions.get_mut(&txn_id) {
            let root = version.semaphore.insert(range.clone());
            return VersionRead::Version(range, root);
        }

        // handle creating a new version
        for (_, version) in versions.iter().take_while(|(id, _)| *id < &txn_id) {
            for reservation in &version.reservations {
                if let Reservation::Write(locked) = reservation {
                    if locked.contains_partial(&range) {
                        // if there's a write lock in the past, wait it out
                        return VersionRead::Pending(range, txn_id);
                    }
                }
            }
        }

        let mut version = Version::with_reservation(Reservation::Read(range.clone()));
        let root = version.semaphore.insert(range.clone());
        versions.insert(txn_id, version);

        VersionRead::Version(range, root)
    }

    /// Acquire a permit to read a section of a transactional resource, if possible.
    pub async fn read(&self, mut txn_id: I, range: R) -> Result<Permit<R>> {
        let mut range = Arc::new(range);

        loop {
            match self.read_inner(txn_id, range) {
                VersionRead::Pending(r, id) => {
                    txn_id = id;
                    range = r;
                }
                VersionRead::Version(range, root) => {
                    return Ok(Permit {
                        permit: root.read(&range).await?,
                        notify: self.notify.clone(),
                    })
                }
            }

            self.notify.notified().await;
        }
    }

    /// Synchronously acquire a permit to read a section of transactional resource, if possible.
    pub fn try_read(&self, txn_id: I, range: R) -> Result<Permit<R>> {
        let range = Arc::new(range);

        match self.read_inner(txn_id, range) {
            VersionRead::Pending(_, _) => Err(Error::WouldBlock),
            VersionRead::Version(range, root) => root
                .try_read(&range)
                .map(|permit| Permit {
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
                    for reservation in &version.reservations {
                        if let Reservation::Write(locked) = reservation {
                            if locked.contains_partial(&range) {
                                // if there's a write lock in the past, wait it out
                                return Ok(VersionRead::Pending(range, txn_id));
                            }
                        }
                    }
                }
                Ordering::Equal => {
                    // the Tokio semaphore will coordinate access within this version
                }
                Ordering::Less => {
                    for reservation in &version.reservations {
                        if reservation.range().contains_partial(&range) {
                            // can't allow writes to this range, there's already a future version
                            return Err(Error::Conflict);
                        }
                    }
                }
            }
        }

        match versions.entry(txn_id) {
            btree_map::Entry::Occupied(mut entry) => {
                let version = entry.get_mut();
                let root = version.semaphore.insert(range.clone());
                Ok(VersionRead::Version(range, root))
            }
            btree_map::Entry::Vacant(entry) => {
                let mut version = Version::with_reservation(Reservation::Write(range.clone()));
                let root = version.semaphore.insert(range.clone());
                entry.insert(version);

                Ok(VersionRead::Version(range, root))
            }
        }
    }

    /// Acquire a permit to write to a section of transactional resource, if possible.
    pub async fn write(&self, mut txn_id: I, range: R) -> Result<Permit<R>> {
        let mut range = Arc::new(range);

        loop {
            match self.write_inner(txn_id, range)? {
                VersionRead::Pending(r, id) => {
                    txn_id = id;
                    range = r;
                }
                VersionRead::Version(range, root) => {
                    let permit = root.write(&range).await?;

                    return Ok(Permit {
                        permit,
                        notify: self.notify.clone(),
                    });
                }
            }

            self.notify.notified().await;
        }
    }

    /// Synchronously acquire a permit to write to a section of transactional resource, if possible.
    pub fn try_write(&self, txn_id: I, range: R) -> Result<Permit<R>> {
        let range = Arc::new(range);

        match self.write_inner(txn_id, range)? {
            VersionRead::Pending(_, _) => Err(Error::WouldBlock),
            VersionRead::Version(range, root) => root
                .try_write(&range)
                .map(|permit| Permit {
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
