//! Lock-free in-order iterator for the learned index tree.
#![allow(unsafe_code)]

use std::sync::atomic::Ordering;

use crossbeam_epoch::Guard;

use crate::key::Key;
use crate::node::{Node, SlotInner};

/// An iterator over the key-value pairs in a learned index.
///
/// Yields references tied to the lifetime of the epoch guard. The traversal
/// order follows slot indices (DFS), which may not be perfectly sorted across
/// child boundaries. Use [`sorted_pairs`] for guaranteed sorted output.
pub struct Iter<'g, K, V> {
    /// Stack of (node, `next_slot_index`) for DFS traversal.
    stack: Vec<(&'g Node<K, V>, usize)>,
    /// The epoch guard that keeps referenced data alive.
    guard: &'g Guard,
}

impl<'g, K: Key, V> Iter<'g, K, V> {
    /// Create a new iterator starting from the root node.
    pub fn new(root: &'g Node<K, V>, guard: &'g Guard) -> Self {
        Self {
            stack: vec![(root, 0)],
            guard,
        }
    }
}

impl<'g, K: Key, V> Iterator for Iter<'g, K, V> {
    type Item = (&'g K, &'g V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (node, slot_idx) = self.stack.last_mut()?;

            if *slot_idx >= node.capacity() {
                // Exhausted this node, pop and continue parent
                self.stack.pop();
                continue;
            }

            let current_idx = *slot_idx;
            *slot_idx += 1;

            let shared = node.slot(current_idx).load(Ordering::Acquire, self.guard);
            if shared.is_null() {
                continue;
            }

            // SAFETY: shared is not null and valid for the lifetime of the guard.
            match unsafe { shared.deref() } {
                SlotInner::Data { key, value } => return Some((key, value)),
                SlotInner::Child(child) => {
                    self.stack.push((child, 0));
                }
            }
        }
    }
}

/// Collect all key-value pairs from a tree in sorted order.
///
/// This performs a full traversal, cloning all entries, and sorts the results.
/// The DFS traversal order follows slot indices which are sorted for well-fit
/// models but may not be perfectly sorted across child boundaries.
pub fn sorted_pairs<K: Key, V: Clone>(root: &Node<K, V>, guard: &Guard) -> Vec<(K, V)> {
    let iter = Iter::new(root, guard);
    let mut pairs: Vec<(K, V)> = iter.map(|(k, v)| (*k, v.clone())).collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    use crossbeam_epoch as epoch;

    fn guard() -> epoch::Guard {
        epoch::pin()
    }

    #[test]
    fn iter_empty_tree() {
        let g = guard();
        let node = Node::<u64, ()>::with_capacity(crate::model::LinearModel::constant(), 5);
        assert!(Iter::new(&node, &g).next().is_none());
    }

    #[test]
    fn iter_single_element() {
        let g = guard();
        let pairs = vec![(42u64, "answer")];
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        let items: Vec<_> = Iter::new(&node, &g).collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0], (&42u64, &"answer"));
    }

    #[test]
    fn iter_all_elements() {
        let g = guard();
        let pairs: Vec<(u64, u64)> = (0..100).map(|i| (i, i * 10)).collect();
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        assert_eq!(Iter::new(&node, &g).count(), 100);
    }

    #[test]
    fn sorted_pairs_in_order() {
        let g = guard();
        let pairs: Vec<(u64, u64)> = (0..50).map(|i| (i * 3 + 1, i)).collect();
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        let sorted = sorted_pairs(&node, &g);
        assert_eq!(sorted.len(), 50);
        for window in sorted.windows(2) {
            assert!(
                window[0].0 < window[1].0,
                "not sorted: {} >= {}",
                window[0].0,
                window[1].0
            );
        }
    }

    #[test]
    fn sorted_pairs_match_input() {
        let g = guard();
        let pairs: Vec<(u64, &str)> = vec![(5, "e"), (10, "j"), (15, "o"), (20, "t")];
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        let sorted = sorted_pairs(&node, &g);
        assert_eq!(sorted, pairs);
    }

    #[test]
    fn iter_after_inserts() {
        let g = guard();
        let pairs: Vec<(u64, u64)> = vec![(10, 1), (30, 3), (50, 5)];
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();

        crate::insert::insert(&node, 20, &2, &g);
        crate::insert::insert(&node, 40, &4, &g);

        let sorted = sorted_pairs(&node, &g);
        assert_eq!(sorted.len(), 5);
        let keys: Vec<u64> = sorted.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![10, 20, 30, 40, 50]);
    }
}
