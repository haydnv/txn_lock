//! A transactional task queue.
//!
//! Example:
//! ```
//! # use std::sync::Arc;
//! # use std::time::Duration;
//! use txn_lock::queue::task::*;
//! use txn_lock::Error;
//!
//! let task: Task<Duration, ()> = Arc::pin(|d| Box::pin(tokio::time::sleep(d)));
//! let queue = TaskQueue::<u64, _, _>::new(task);
//!
//! // this can only execute when a tokio reactor is running
//! // queue.push(1, 1).expect("push");
//! ```

use std::collections::VecDeque;
use std::fmt;
use std::hash::Hash;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures::future::{Future, FutureExt};
use tokio::sync::{OwnedRwLockReadGuard, RwLock};
use tokio::task::JoinHandle;

use crate::Error;

use super::{Entry, State};

/// The return type of a [`Task`] function
pub type BoxFuture<Out> = Pin<Box<dyn Future<Output = Out> + Send + Sync>>;

/// A task for a [`TaskQueue`] to run
pub type Task<I, O> = Pin<Arc<dyn Fn(I) -> BoxFuture<O> + Send + Sync>>;

struct Queue<O> {
    tasks: VecDeque<JoinHandle<O>>,
    results: Arc<RwLock<Vec<O>>>,
}

impl<O> Queue<O> {
    fn new() -> Self {
        Self {
            tasks: VecDeque::with_capacity(1),
            results: Arc::new(RwLock::new(Vec::with_capacity(1))),
        }
    }
}

impl<O: Send + Sync + fmt::Debug + 'static> Queue<O> {
    async fn commit(mut self) -> Vec<O> {
        let mut results = Arc::try_unwrap(self.results).expect("results").into_inner();

        while let Some(handle) = self.tasks.pop_front() {
            let result = handle.await.expect("join");
            results.push(result);
        }

        results
    }

    async fn peek(&mut self) -> OwnedRwLockReadGuard<Vec<O>> {
        let mut results = self.results.clone().write_owned().await;

        while let Some(handle) = self.tasks.pop_front() {
            let result = handle.await.expect("join");
            results.push(result);
        }

        results.downgrade()
    }

    fn push(&mut self, task: BoxFuture<O>) {
        self.tasks.push_back(tokio::spawn(task))
    }
}

/// A transactional task queue
pub struct TaskQueue<I, In, Out> {
    task: Task<In, Out>,
    state: Arc<Mutex<State<I, Queue<Out>>>>,
}

impl<I, In, Out> Clone for TaskQueue<I, In, Out> {
    fn clone(&self) -> Self {
        Self {
            task: self.task.clone(),
            state: self.state.clone(),
        }
    }
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
    Out: Send + Sync + fmt::Debug + 'static,
{
    /// Wait for all queued tasks to complete, then borrow them for inspection.
    pub async fn peek(&self, txn_id: &I) -> Result<Option<OwnedRwLockReadGuard<Vec<Out>>>, Error> {
        let mut state = self.state.lock().expect("state");

        if let Some(queue) = state.check_finalized(txn_id)? {
            queue.peek().map(Some).map(Ok).await
        } else {
            Ok(None)
        }
    }

    /// Push a new input onto the queue at `txn_id`.
    pub fn push(&self, txn_id: I, input: In) -> Result<(), Error> {
        let mut state = self.state.lock().expect("state");

        let task = (self.task)(input);

        match state.check_pending(txn_id)? {
            Entry::Occupied(mut entry) => entry.get_mut().push(task),
            Entry::Vacant(entry) => {
                entry.insert(Queue::new()).push(task);
            }
        };

        Ok(())
    }
}

impl<I, In, Out> TaskQueue<I, In, Out>
where
    I: Eq + Hash + Ord + fmt::Debug,
    Out: Send + Sync + fmt::Debug + 'static,
{
    /// Close and return the queue at `txn_id`.
    ///
    /// Panics:
    ///  - if there is an active lock on the message queue at `txn_id`
    ///  - if the queue has already been finalized at `txn_id`
    pub async fn commit(&self, txn_id: I) -> Vec<Out> {
        let mut state = self.state.lock().expect("state");
        if let Some(queue) = state.commit(txn_id) {
            queue.commit().await
        } else {
            vec![]
        }
    }

    /// Close the queue at `txn_id`.
    ///
    /// Panics:
    ///  - if the queue has already been committed at `txn_id`
    ///  - if the queue has already been finalized at `txn_id`
    pub fn rollback(&self, txn_id: &I) {
        let mut state = self.state.lock().expect("state");
        state.rollback(txn_id);
    }

    /// Finalize the queue at `txn_id`, preventing further operations prior to `txn_id`.
    pub fn finalize(&self, txn_id: I) {
        let mut state = self.state.lock().expect("state");
        state.finalize(txn_id);
    }
}
