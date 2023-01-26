//! A semaphore used to maintain the ACID compliance of a mutable data store.
//!
//! This differs from a traditional semaphore by tracking read and write permits
//! rather than an [`AtomicUsize`]. This is because of the additional logical constraints of
//! a transactional resource--
//! a write permit blocks future read permits and a read permit blocks past write permits.
//!
//! More information: [https://en.wikipedia.org/wiki/ACID](https://en.wikipedia.org/wiki/ACID)

use std::collections::btree_map;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use tokio::sync::{Notify, OwnedSemaphorePermit};

use super::Result;

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
            // this semaphore will take care of permits within this version
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
}
