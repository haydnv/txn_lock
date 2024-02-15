/// Transactional queue structures.
use std::fmt;
use std::hash::Hash;

use ds_ext::map::{Entry, OrdHashMap};
use ds_ext::set::OrdHashSet;

use super::Error;

pub mod message;
pub mod task;

struct State<I, S> {
    pending: OrdHashMap<I, S>,
    commits: OrdHashSet<I>,
    finalized: Option<I>,
}

impl<I, S> State<I, S> {
    fn new() -> Self {
        Self {
            pending: OrdHashMap::new(),
            commits: OrdHashSet::new(),
            finalized: None,
        }
    }
}

impl<I, S> State<I, S>
where
    I: Eq + Hash + Ord,
{
    fn check_finalized(&mut self, txn_id: &I) -> Result<Option<&mut S>, Error> {
        if Some(txn_id) <= self.finalized.as_ref() {
            Err(Error::Outdated)
        } else {
            Ok(self.pending.get_mut(txn_id))
        }
    }

    fn check_pending(&mut self, txn_id: I) -> Result<Entry<I, S>, Error> {
        if Some(&txn_id) <= self.finalized.as_ref() {
            Err(Error::Outdated)
        } else if self.commits.contains(&txn_id) {
            Err(Error::Committed)
        } else {
            Ok(self.pending.entry(txn_id))
        }
    }
}

impl<I, S> State<I, S>
where
    I: Eq + Hash + Ord + fmt::Debug,
{
    fn commit(&mut self, txn_id: I) -> Option<S> {
        assert!(
            self.finalized.as_ref() < Some(&txn_id),
            "queue is already finalized at {txn_id:?}"
        );

        let queue = self.pending.remove(&txn_id);
        self.commits.insert(txn_id);
        queue
    }

    fn rollback(&mut self, txn_id: &I) -> Option<S> {
        assert!(
            self.finalized.as_ref() < Some(&txn_id),
            "queue is already finalized at {txn_id:?}"
        );

        assert!(
            !self.commits.contains(txn_id),
            "queue is already committed at {txn_id:?}"
        );

        self.pending.remove(txn_id)
    }

    fn finalize(&mut self, txn_id: I) {
        while self
            .commits
            .first()
            .map(|version_id| version_id <= &txn_id)
            .unwrap_or_default()
        {
            self.commits.pop_first();
        }

        while self
            .pending
            .keys()
            .next()
            .map(|version_id| version_id <= &txn_id)
            .unwrap_or_default()
        {
            self.pending.pop_first();
        }

        if self.finalized.as_ref() > Some(&txn_id) {
            // no-op
        } else {
            self.finalized = Some(txn_id);
        }
    }
}
