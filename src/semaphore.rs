//! A semaphore used to maintain the ACID compliance of a mutable data store.
//!
//! This differs from a traditional semaphore by tracking read and write reservations
//! rather than an [`AtomicUsize`]. This is because of the additional logical constraints of
//! a transactional resource--
//! a write permit blocks future read permits and a read permit conflicts with past write permits.
//!
//! More information: [https://en.wikipedia.org/wiki/ACID](https://en.wikipedia.org/wiki/ACID)

use std::cmp::Ordering;
use std::collections::btree_map;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use tokio::sync::{Notify, OwnedSemaphorePermit};

use super::{Error, Result};

const PERMITS: u32 = u32::MAX >> 3;

pub trait Overlap {
    fn overlaps(&self, other: &Self) -> bool;
}

/// A permit to read a specific section of a transactional resource
pub struct Permit<Range> {
    _permit: OwnedSemaphorePermit,
    notify: Arc<Notify>,
    range: Arc<Range>,
}

impl<Range> Deref for Permit<Range> {
    type Target = Range;

    fn deref(&self) -> &Self::Target {
        self.range.deref()
    }
}

impl<Range> Drop for Permit<Range> {
    fn drop(&mut self) {
        self.notify.notify_waiters()
    }
}

enum Reservation<Range> {
    Read(Arc<Range>),
    Write(Arc<Range>),
}

impl<Range> Reservation<Range> {
    fn range(&self) -> &Range {
        match self {
            Self::Read(range) => range,
            Self::Write(range) => range,
        }
    }
}

struct Version<Range> {
    semaphore: Arc<tokio::sync::Semaphore>,
    reservations: Vec<Reservation<Range>>,
}

impl<Range> Version<Range> {
    fn new(reservations: Vec<Reservation<Range>>) -> Self {
        Self {
            semaphore: Arc::new(tokio::sync::Semaphore::new(PERMITS as usize)),
            reservations,
        }
    }

    fn reserve(reservation: Reservation<Range>) -> Self {
        Self::new(vec![reservation])
    }
}

enum VersionResult<TxnId, Range> {
    Version(Arc<tokio::sync::Semaphore>, Arc<Range>),
    Pending(TxnId, Arc<Range>),
}

/// A semaphore used to maintain the ACID compliance of transactional resource
pub struct Semaphore<TxnId, Range> {
    versions: Arc<Mutex<btree_map::BTreeMap<TxnId, Version<Range>>>>,
    notify: Arc<Notify>,
}

impl<TxnId: Ord, Range: Overlap> Semaphore<TxnId, Range> {
    /// Construct a new transactional [`Semaphore`].
    pub fn new() -> Self {
        Self {
            versions: Arc::new(Mutex::new(btree_map::BTreeMap::new())),
            notify: Arc::new(Notify::new()),
        }
    }

    fn read_inner(&self, txn_id: TxnId, range: Arc<Range>) -> VersionResult<TxnId, Range> {
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

        let version = Version::reserve(Reservation::Read(range.clone()));
        let semaphore = version.semaphore.clone();
        versions.insert(txn_id, version);

        VersionResult::Version(semaphore, range)
    }

    /// Acquire a permit to read a section of transactional resource, if possible.
    pub async fn read(&self, mut txn_id: TxnId, range: Range) -> Result<Permit<Range>> {
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

    fn write_inner(&self, txn_id: TxnId, range: Arc<Range>) -> Result<VersionResult<TxnId, Range>> {
        let mut versions = self.versions.lock().expect("versions");

        // handle creating a new version
        for (version_id, version) in versions.iter() {
            match txn_id.cmp(version_id) {
                Ordering::Less => {
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
                Ordering::Greater => {
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
                let version = Version::reserve(Reservation::Write(range.clone()));
                let semaphore = version.semaphore.clone();
                entry.insert(version);

                Ok(VersionResult::Version(semaphore, range))
            }
        }
    }

    /// Acquire a permit to write to a section of transactional resource, if possible.
    pub async fn write(&self, mut txn_id: TxnId, range: Range) -> Result<Permit<Range>> {
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

    /// Mark a transaction completed, un-blocking waiting requests for future permits.
    ///
    /// Call this when the transactional resource is no longer writable at `txn_id`
    /// (due to a commit, rollback, timeout, or any other reason).
    ///
    /// Set `drop_past` to `true` to finalize all transactions older than `txn_id`.
    pub fn finalize(&self, txn_id: &TxnId, drop_past: bool)
    where
        TxnId: Copy,
    {
        let mut versions = self.versions.lock().expect("versions");

        let notify = if drop_past {
            let mut notify = false;

            while let Some(version_id) = versions.keys().next().copied() {
                assert!(versions.remove(&version_id).is_some());
                notify = true;
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
