//! A transactional message queue.
//!
//! Example:
//! ```
//! use futures::executor::block_on;
//! use txn_lock::queue::message::*;
//! use txn_lock::Error;
//!
//! let queue = MessageQueue::<u64, &'static str>::new();
//!
//! assert_eq!(queue.commit(0), Vec::<&'static str>::new());
//!
//! queue.push(1, "first message").expect("send");
//! queue.push(2, "second message").expect("send");
//! queue.push(3, "third message").expect("send");
//!
//! assert_eq!(queue.commit(2), vec!["second message"]);
//! assert_eq!(queue.commit(1), vec!["first message"]);
//!
//! queue.finalize(2);
//!
//! assert_eq!(queue.commit(3), vec!["third message"]);
//!
//! queue.push(4, "fourth message").expect("send");
//! ```

use std::fmt;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use crate::Error;

use super::{Entry, State};

/// A transactional message queue
pub struct MessageQueue<I, M> {
    state: Arc<Mutex<State<I, Vec<M>>>>,
}

impl<I, M> Clone for MessageQueue<I, M> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

impl<I, M> MessageQueue<I, M> {
    /// Construct a new [`MessageQueue`].
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(State::new())),
        }
    }
}

impl<I: Copy + Eq + Ord + Hash, M> MessageQueue<I, M> {
    /// Push the given `message` onto the queue at `txn_id`.
    pub fn push(&self, txn_id: I, message: M) -> Result<(), Error> {
        let mut state = self.state.lock().expect("state");

        match state.check_pending(txn_id)? {
            Entry::Occupied(mut entry) => entry.get_mut().push(message),
            Entry::Vacant(entry) => {
                entry.insert(vec![message]);
            }
        };

        Ok(())
    }
}

impl<I: Eq + Hash + Ord + fmt::Debug, M> MessageQueue<I, M> {
    /// Close and return the queue for the given `txn_id`, if any.
    ///
    /// If `commit` is called multiple times, only the first will return a filled `Vec`.
    ///
    /// Panics:
    ///  - if this [`MessageQueue`] has already been finalized at the given `txn_id`
    pub fn commit(&self, txn_id: I) -> Vec<M> {
        let mut state = self.state.lock().expect("state");
        state.commit(txn_id).unwrap_or_default()
    }

    /// Close the queue at `txn_id`, if any.
    ///
    /// Panics:
    ///  - if this [`MessageQueue`] has already been committed at the given `txn_id`
    ///  - if this [`MessageQueue`] has already been finalized at the given `txn_id`
    pub fn rollback(&self, txn_id: &I) {
        let mut state = self.state.lock().expect("state");
        state.rollback(txn_id);
    }

    /// Finalize this [`MessageQueue`], preventing any further actions prior to the given `txn_id`.
    pub fn finalize(&self, txn_id: I)
    where
        I: PartialOrd,
    {
        let mut state = self.state.lock().expect("state");
        state.finalize(txn_id)
    }
}
