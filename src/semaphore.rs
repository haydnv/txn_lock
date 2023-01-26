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

use super::{Error, Result};

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
    fn new(permits: usize) -> Self {
        Self {
            semaphore: Arc::new(tokio::sync::Semaphore::new(permits)),
            reservations: Vec::new(),
        }
    }
}

enum ReadResult<TxnId> {
    Version(Arc<tokio::sync::Semaphore>),
    Pending(TxnId),
}

/// A semaphore used to maintain the ACID compliance of transactional resource
pub struct Semaphore<TxnId, Range> {
    versions: Arc<Mutex<btree_map::BTreeMap<TxnId, Arc<Mutex<Version<Range>>>>>>,
    notify: Arc<Notify>,
}

impl<TxnId, Range: Overlap> Semaphore<TxnId, Range> {
    fn read_inner(&self, txn_id: TxnId, range: &Arc<Range>) -> Result<ReadResult<TxnId>> {
        todo!()
    }

    /// Acquire a permit to read a transactional resource, if possible.
    pub async fn read(&self, mut txn_id: TxnId, range: Range) -> Result<Permit<Range>> {
        let range = Arc::new(range);

        loop {
            match self.read_inner(txn_id, &range)? {
                ReadResult::Pending(id) => {
                    txn_id = id;
                }
                ReadResult::Version(semaphore) => {
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
