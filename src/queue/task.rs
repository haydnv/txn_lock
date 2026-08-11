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
//! let queue = TaskQueue::<u64, _, _>::new(task, 16);
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
    accepted: usize,
    capacity: usize,
    tx: mpsc::Sender<JoinHandle<O>>,
    rx: mpsc::Receiver<JoinHandle<O>>,
    results: Arc<RwLock<Vec<O>>>,
}

impl<O> Queue<O> {
    fn new(capacity: usize) -> Self {
        let (tx, rx) = mpsc::channel(capacity);

        Self {
            accepted: 0,
            capacity,
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

    fn push<In>(&mut self, task: &Task<In, O>, input: In) -> Result<(), Error> {
        if self.accepted >= self.capacity {
            return Err(Error::Saturated);
        }

        let permit = self.tx.try_reserve().map_err(|cause| match cause {
            mpsc::error::TrySendError::Full(_) => Error::Saturated,
            mpsc::error::TrySendError::Closed(_) => Error::Outdated,
        })?;

        permit.send(tokio::spawn(task(input)));
        self.accepted += 1;
        Ok(())
    }
}

/// A transactional task queue
pub struct TaskQueue<I, In, Out> {
    capacity: usize,
    task: Task<In, Out>,
    state: Arc<Mutex<State<I, Queue<Out>>>>,
}

impl<I, In, Out> Clone for TaskQueue<I, In, Out> {
    fn clone(&self) -> Self {
        Self {
            capacity: self.capacity,
            task: self.task.clone(),
            state: self.state.clone(),
        }
    }
}

impl<I, In, Out> TaskQueue<I, In, Out> {
    /// Construct a new transactional task queue.
    pub fn new(task: Task<In, Out>, capacity: usize) -> Self {
        assert!(capacity > 0, "task queue capacity must be positive");

        Self {
            capacity,
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

        match state.check_pending(txn_id)? {
            Entry::Occupied(mut entry) => entry.get_mut().push(&self.task, input),
            Entry::Vacant(entry) => entry
                .insert(Queue::new(self.capacity))
                .push(&self.task, input),
        }?;

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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{Task, TaskQueue};
    use crate::Error;

    #[tokio::test]
    async fn applies_backpressure_at_capacity() {
        let task: Task<(), ()> = Arc::pin(|()| Box::pin(async {}));
        let queue = TaskQueue::new(task, 1);

        queue.push(1, ()).expect("first task");
        let results = queue.peek(&1).await.expect("peek").expect("task queue");
        assert_eq!(results.as_slice(), &[()]);
        drop(results);

        assert_eq!(queue.push(1, ()), Err(Error::Saturated));
        assert_eq!(queue.commit(1).await, vec![()]);
    }
}
