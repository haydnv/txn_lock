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

use futures::future::Future;
use tokio::sync::{mpsc, OwnedRwLockReadGuard, RwLock};
use tokio::task::JoinHandle;

use crate::Error;

use super::{Entry, State};

/// The return type of a [`Task`] function
pub type BoxFuture<Out> = Pin<Box<dyn Future<Output = Out> + Send>>;

/// A task for a [`TaskQueue`] to run
pub type Task<I, O> = Pin<Arc<dyn Fn(I) -> BoxFuture<O> + Send + Sync>>;

struct Queue<O> {
    tx: mpsc::UnboundedSender<JoinHandle<O>>,
    rx: mpsc::UnboundedReceiver<JoinHandle<O>>,
    results: Arc<RwLock<Vec<O>>>,
}

impl<O> Queue<O> {
    fn new() -> Self {
        let (tx, rx) = mpsc::unbounded_channel();

        Self {
            tx,
            rx,
            results: Arc::new(RwLock::new(Vec::with_capacity(1))),
        }
    }
}

impl<O: Send + Sync + fmt::Debug + 'static> Queue<O> {
    async fn commit(mut self) -> Vec<O> {
        std::mem::drop(self.tx);

        let mut results = Arc::try_unwrap(self.results).expect("results").into_inner();

        while let Some(handle) = self.rx.recv().await {
            let result = handle.await.expect("join");
            results.push(result);
        }

        results
    }

    fn peek(&mut self) -> (VecDeque<JoinHandle<O>>, Arc<RwLock<Vec<O>>>) {
        let mut pending = VecDeque::with_capacity(0);

        while let Ok(handle) = self.rx.try_recv() {
            pending.push_back(handle);
        }

        (pending, self.results.clone())
    }

    fn push(&mut self, task: BoxFuture<O>) -> Result<(), mpsc::error::SendError<JoinHandle<O>>> {
        self.tx.send(tokio::spawn(task))
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
        let (mut pending, results) = {
            let mut state = self.state.lock().expect("state");

            if let Some(queue) = state.check_finalized(txn_id)? {
                queue.peek()
            } else {
                return Ok(None);
            }
        };

        let mut results = results.write_owned().await;

        while let Some(handle) = pending.pop_front() {
            let result = handle.await?;
            results.push(result);
        }

        Ok(Some(results.downgrade()))
    }

    /// Push a new input onto the queue at `txn_id`.
    pub fn push(&self, txn_id: I, input: In) -> Result<(), Error> {
        let mut state = self.state.lock().expect("state");

        let task = (self.task)(input);

        match state.check_pending(txn_id)? {
            Entry::Occupied(mut entry) => entry.get_mut().push(task),
            Entry::Vacant(entry) => entry.insert(Queue::new()).push(task),
        }
        .expect("send"); // at this point the channel can't possibly be closed

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
        let queue = {
            let mut state = self.state.lock().expect("state");
            state.commit(txn_id)
        };

        if let Some(queue) = queue {
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
