//! A transactional message queue.
//!
//! Example:
//! ```
//! use futures::executor::block_on;
//! use txn_lock::queue::*;
//! use txn_lock::Error;
//!
//! let queue = Queue::<u64, &'static str>::new();
//!
//! assert_eq!(queue.commit(0), Vec::<&'static str>::new());
//!
//! queue.send(1, "first message").expect("send");
//! queue.send(2, "second message").expect("send");
//! queue.send(3, "third message").expect("send");
//!
//! assert_eq!(queue.commit(2), vec!["second message"]);
//! assert_eq!(queue.commit(1), vec!["first message"]);
//!
//! queue.finalize(2);
//!
//! assert_eq!(queue.commit(3), vec!["third message"]);
//!
//! queue.send(4, "fourth message").expect("send");
//!
//! ```

use std::fmt;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use ds_ext::{OrdHashMap, OrdHashSet};

use super::Error;

struct State<I, M> {
    pending: OrdHashMap<I, Vec<M>>,
    commits: OrdHashSet<I>,
    finalized: Option<I>,
}

/// A transactional message queue
pub struct Queue<I, M> {
    state: Arc<Mutex<State<I, M>>>,
}

impl<I, M> Queue<I, M> {
    /// Construct a new [`Queue`].
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                pending: OrdHashMap::new(),
                commits: OrdHashSet::new(),
                finalized: None,
            })),
        }
    }
}

impl<I: Eq + Ord + Hash, M> Queue<I, M> {
    /// Send the given `message` at `txn_id`.
    pub fn send(&self, txn_id: I, message: M) -> Result<(), Error> {
        let mut state = self.state.lock().expect("state");

        if Some(&txn_id) < state.finalized.as_ref() {
            return Err(Error::Outdated);
        } else if state.commits.contains(&txn_id) {
            return Err(Error::Committed);
        }

        if let Some(queue) = state.pending.get_mut(&txn_id) {
            queue.push(message);
        } else {
            state.pending.insert(txn_id, vec![message]);
        }

        Ok(())
    }
}

impl<I: Eq + Hash + Ord + fmt::Debug, M> Queue<I, M> {
    /// Close the channel and return the queue for the given `txn_id`, if any.
    ///
    /// If `commit` is called multiple times, only the first will return a filled `Vec`.
    ///
    /// Panics:
    ///  - if the [`Queue`] has already been finalized at the given `txn_id`
    pub fn commit(&self, txn_id: I) -> Vec<M> {
        let mut state = self.state.lock().expect("state");

        assert!(
            state.finalized.as_ref() < Some(&txn_id),
            "queue is already finalized at {txn_id:?}"
        );

        let queue = state.pending.remove(&txn_id).unwrap_or_default();
        state.commits.insert(txn_id);
        queue
    }

    /// Close the channel and return the queue for the given `txn_id`, if any.
    /// Because a [`Queue`] has no canonical state, `rollback` is an alias of `commit`
    /// provided for the sake of code clarity.
    ///
    /// If `rollback` is called multiple times, only the first will return a filled `Vec`.
    ///
    /// Panics:
    ///  - if the [`Queue`] has already been finalized at the given `txn_id`
    pub fn rollback(&self, txn_id: I) -> Vec<M> {
        self.commit(txn_id)
    }

    /// Finalize the [`Queue`], preventing any further actions prior to the given `txn_id`.
    pub fn finalize(&self, txn_id: I)
    where
        I: PartialOrd,
    {
        let mut state = self.state.lock().expect("state");

        while state
            .commits
            .first()
            .map(|version_id| &**version_id <= &txn_id)
            .unwrap_or_default()
        {
            state.commits.pop_first();
        }

        while state
            .pending
            .keys()
            .next()
            .map(|version_id| version_id <= &txn_id)
            .unwrap_or_default()
        {
            state.pending.pop_first();
        }

        if state.finalized.as_ref() > Some(&txn_id) {
            // no-op
        } else {
            state.finalized = Some(txn_id);
        }
    }
}
