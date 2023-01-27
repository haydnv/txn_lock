//! A futures-aware semaphore used to maintain the ACID compliance of a mutable data store.
//!
//! This differs from [`tokio::sync::Semaphore`] by tracking read and write reservations
//! rather than an atomic integer. This is because of the additional logical constraints of
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
//! // Multiple read permits within a transaction are fine
//! let permit0_1 = semaphore.try_read(0, 0..1).expect("permit");
//! let permit0_2 = semaphore.try_read(0, 0..1).expect("permit");
//! // But they'll block a write permit in the same transaction
//! assert_eq!(semaphore.try_write(0, 0..2).unwrap_err(), Error::WouldBlock);
//! std::mem::drop(permit0_1);
//! assert_eq!(semaphore.try_write(0, 0..2).unwrap_err(), Error::WouldBlock);
//! std::mem::drop(permit0_2);
//! // Until they're all dropped
//! let permit0_3 = semaphore.try_write(0, 0..2).expect("permit");
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

use std::cmp::Ordering;
use std::collections::{btree_map, BTreeMap};
use std::ops::{Deref, Range};
use std::sync::{Arc, Mutex};

use tokio::sync::{Notify, OwnedSemaphorePermit};

use super::{Error, Result};

const PERMITS: u32 = u32::MAX >> 3;

/// A range supported by a transactional [`Semaphore`]
pub trait Overlap {
    /// A commutative method which returns `true` if `self` overlaps `other`.
    fn overlaps(&self, other: &Self) -> bool;
}

impl<Idx: PartialOrd<Idx>> Overlap for Range<Idx> {
    fn overlaps(&self, other: &Self) -> bool {
        if other.start >= self.start && other.start < self.end {
            // check if other.start is in self.start..self.end
            true
        } else if other.end >= self.start && other.end < self.end {
            // check if other.end is in self.start..self.end
            true
        } else {
            false
        }
    }
}

/// A permit to read a specific section of a transactional resource
#[derive(Debug)]
pub struct Permit<R> {
    _permit: OwnedSemaphorePermit,
    notify: Arc<Notify>,
    range: Arc<R>,
}

impl<R> Deref for Permit<R> {
    type Target = R;

    fn deref(&self) -> &Self::Target {
        self.range.deref()
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
    fn range(&self) -> &R {
        match self {
            Self::Read(range) => range,
            Self::Write(range) => range,
        }
    }
}

struct Version<R> {
    semaphore: Arc<tokio::sync::Semaphore>,
    reservations: Vec<Reservation<R>>,
}

impl<R> Version<R> {
    fn new(reservations: Vec<Reservation<R>>) -> Self {
        Self {
            semaphore: Arc::new(tokio::sync::Semaphore::new(PERMITS as usize)),
            reservations,
        }
    }

    fn with_reservation(reservation: Reservation<R>) -> Self {
        Self::new(vec![reservation])
    }

    fn with_reservations<W: IntoIterator<Item = R>>(reserve: W) -> Self {
        let reserved = reserve
            .into_iter()
            .map(|range| Reservation::Write(Arc::new(range)))
            .collect();

        Self::new(reserved)
    }
}

enum VersionResult<I, R> {
    Version(Arc<tokio::sync::Semaphore>, Arc<R>),
    Pending(I, Arc<R>),
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

impl<I: Ord, R: Overlap> Semaphore<I, R> {
    /// Construct a new transactional [`Semaphore`].
    pub fn new() -> Self {
        Self {
            versions: Arc::new(Mutex::new(BTreeMap::new())),
            notify: Arc::new(Notify::new()),
        }
    }

    /// Construct a new transactional [`Semaphore`] with write reservations for its initial value.
    pub fn with_reservations<W: IntoIterator<Item = R>>(txn_id: I, reserve: W) -> Self {
        let mut versions = BTreeMap::new();
        versions.insert(txn_id, Version::with_reservations(reserve));

        Self {
            versions: Arc::new(Mutex::new(BTreeMap::new())),
            notify: Arc::new(Notify::new()),
        }
    }

    fn read_inner(&self, txn_id: I, range: Arc<R>) -> VersionResult<I, R> {
        let mut versions = self.versions.lock().expect("versions");

        if let Some(version) = versions.get(&txn_id) {
            // this Tokio semaphore will coordinate access within this version
            return VersionResult::Version(version.semaphore.clone(), range);
        }

        // handle creating a new version
        for (_, version) in versions.iter().take_while(|(id, _)| *id < &txn_id) {
            for reservation in &version.reservations {
                if let Reservation::Write(locked) = reservation {
                    if locked.overlaps(&range) {
                        // if there's a write lock in the past, wait it out
                        return VersionResult::Pending(txn_id, range);
                    }
                }
            }
        }

        let version = Version::with_reservation(Reservation::Read(range.clone()));
        let semaphore = version.semaphore.clone();
        versions.insert(txn_id, version);

        VersionResult::Version(semaphore, range)
    }

    /// Acquire a permit to read a section of transactional resource, if possible.
    pub async fn read(&self, mut txn_id: I, range: R) -> Result<Permit<R>> {
        let mut range = Arc::new(range);

        loop {
            match self.read_inner(txn_id, range) {
                VersionResult::Pending(id, r) => {
                    txn_id = id;
                    range = r;
                }
                VersionResult::Version(semaphore, range) => {
                    return Ok(Permit {
                        _permit: semaphore.acquire_owned().await?,
                        notify: self.notify.clone(),
                        range,
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
            VersionResult::Pending(_, _) => Err(Error::WouldBlock),
            VersionResult::Version(semaphore, range) => semaphore
                .try_acquire_owned()
                .map(|_permit| Permit {
                    _permit,
                    notify: self.notify.clone(),
                    range,
                })
                .map_err(Error::from),
        }
    }

    fn write_inner(&self, txn_id: I, range: Arc<R>) -> Result<VersionResult<I, R>> {
        let mut versions = self.versions.lock().expect("versions");

        // handle creating a new version
        for (version_id, version) in versions.iter() {
            match txn_id.cmp(version_id) {
                Ordering::Greater => {
                    for reservation in &version.reservations {
                        if let Reservation::Write(locked) = reservation {
                            if locked.overlaps(&range) {
                                // if there's a write lock in the past, wait it out
                                return Ok(VersionResult::Pending(txn_id, range));
                            }
                        }
                    }
                }
                Ordering::Equal => {
                    // the Tokio semaphore will coordinate access within this version
                }
                Ordering::Less => {
                    for reservation in &version.reservations {
                        if reservation.range().overlaps(&range) {
                            // can't allow writes to this range, there's already a future version
                            return Err(Error::Conflict);
                        }
                    }
                }
            }
        }

        match versions.entry(txn_id) {
            btree_map::Entry::Occupied(entry) => {
                let version = entry.get();
                Ok(VersionResult::Version(version.semaphore.clone(), range))
            }
            btree_map::Entry::Vacant(entry) => {
                let version = Version::with_reservation(Reservation::Write(range.clone()));
                let semaphore = version.semaphore.clone();
                entry.insert(version);

                Ok(VersionResult::Version(semaphore, range))
            }
        }
    }

    /// Acquire a permit to write to a section of transactional resource, if possible.
    pub async fn write(&self, mut txn_id: I, range: R) -> Result<Permit<R>> {
        let mut range = Arc::new(range);

        loop {
            match self.write_inner(txn_id, range)? {
                VersionResult::Pending(id, r) => {
                    txn_id = id;
                    range = r;
                }
                VersionResult::Version(semaphore, range) => {
                    return Ok(Permit {
                        _permit: semaphore.acquire_many_owned(PERMITS).await?,
                        notify: self.notify.clone(),
                        range,
                    })
                }
            }

            self.notify.notified().await;
        }
    }

    /// Synchronously acquire a permit to write to a section of transactional resource, if possible.
    pub fn try_write(&self, txn_id: I, range: R) -> Result<Permit<R>> {
        let range = Arc::new(range);

        match self.write_inner(txn_id, range)? {
            VersionResult::Pending(_, _) => Err(Error::WouldBlock),
            VersionResult::Version(semaphore, range) => semaphore
                .try_acquire_many_owned(PERMITS)
                .map(|_permit| Permit {
                    _permit,
                    notify: self.notify.clone(),
                    range,
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
