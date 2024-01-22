//! A transactional task queue.
//!
//! Example:
//! ```
//! use txn_lock::queue::task::*;
//! use txn_lock::Error;
//!
//! let task: Task<u64, u64> = Box::pin(|x| Box::pin(async move { x * 2 }));
//! let queue = TaskQueue::<u64, _, _>::new(task);
//!
//! // this can only execute when a tokio reactor is running
//! // queue.push(1, 1).expect("push");
//! ```

use std::fmt;
use std::hash::Hash;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures::future::Future;
use tokio::task::JoinHandle;

use crate::Error;

use super::{Entry, State};

/// The return type of a [`Task`] function
pub type BoxFuture<Out> = Pin<Box<dyn Future<Output = Out> + Send + Sync>>;

/// A task for a [`TaskQueue`] to run
pub type Task<I, O> = Pin<Box<dyn Fn(I) -> BoxFuture<O>>>;

/// A transactional task queue
pub struct TaskQueue<I, In, Out> {
    task: Task<In, Out>,
    state: Arc<Mutex<State<I, Vec<JoinHandle<Out>>>>>,
}

impl<I, In, Out> TaskQueue<I, In, Out> {
    /// Construct a new transactional task queue.
    pub fn new(task: Task<In, Out>) -> Self {
        Self {
            task,
            state: Arc::new(Mutex::new(State::new())),
        }
    }
}

impl<I, In, Out> TaskQueue<I, In, Out>
where
    I: Eq + Hash + Ord,
    Out: Send + Sync + 'static,
{
    /// Push a new input onto the queue at `txn_id`.
    pub fn push(&self, txn_id: I, input: In) -> Result<(), Error> {
        let mut state = self.state.lock().expect("state");

        let task = tokio::spawn((self.task)(input));

        match state.check_pending(txn_id)? {
            Entry::Occupied(mut entry) => entry.get_mut().push(task),
            Entry::Vacant(entry) => {
                entry.insert(Vec::with_capacity(1)).push(task);
            }
        };

        Ok(())
    }
}

impl<I, In, Out> TaskQueue<I, In, Out>
where
    I: Eq + Hash + Ord + fmt::Debug,
{
    /// Close and return the queue at `txn_id`.
    pub fn commit(&self, txn_id: I) -> Option<Vec<JoinHandle<Out>>> {
        let mut state = self.state.lock().expect("state");
        state.commit(txn_id)
    }

    /// Close the queue at `txn_id`.
    pub fn rollback(&self, txn_id: &I) {
        let mut state = self.state.lock().expect("state");
        state.rollback(txn_id);
    }

    /// Finalize the queue at `txn_id`, preventing futher operations prior to `txn_id`.
    pub fn finalize(&self, txn_id: I) {
        let mut state = self.state.lock().expect("state");
        state.finalize(txn_id);
    }
}
