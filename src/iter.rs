//! In-order iterator for the learned index tree.

use crate::key::Key;
use crate::node::{Node, Slot};

/// An iterator over the key-value pairs in a learned index, in sorted order.
pub struct Iter<'a, K, V> {
    /// Stack of (node, `next_slot_index`) for DFS traversal.
    stack: Vec<(&'a Node<K, V>, usize)>,
}

impl<'a, K: Key, V> Iter<'a, K, V> {
    /// Create a new iterator starting from the root node.
    pub fn new(root: &'a Node<K, V>) -> Self {
        Self {
            stack: vec![(root, 0)],
        }
    }
}

impl<'a, K: Key, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (node, slot_idx) = self.stack.last_mut()?;

            if *slot_idx >= node.slots.len() {
                // Exhausted this node, pop and continue parent
                self.stack.pop();
                continue;
            }

            let current_idx = *slot_idx;
            *slot_idx += 1;

            match &node.slots[current_idx] {
                Slot::Empty => {}
                Slot::Data(k, v) => return Some((k, v)),
                Slot::Child(child) => {
                    self.stack.push((child, 0));
                }
            }
        }
    }
}

/// Collect all key-value pairs from a tree in sorted order.
///
/// This performs a full traversal and sorts the results, since the DFS
/// traversal order follows slot indices (model-predicted positions),
/// which are sorted for well-fit models but may not be perfectly sorted
/// across child boundaries.
pub fn sorted_pairs<K: Key, V: Clone>(root: &Node<K, V>) -> Vec<(K, V)> {
    let iter = Iter::new(root);
    let mut pairs: Vec<(K, V)> = iter.map(|(k, v)| (*k, v.clone())).collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn iter_empty_tree() {
        let node = Node::<u64, ()>::with_capacity(crate::model::LinearModel::constant(), 5);
        let items: Vec<_> = Iter::new(&node).collect();
        assert!(items.is_empty());
    }

    #[test]
    fn iter_single_element() {
        let pairs = vec![(42u64, "answer")];
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        let items: Vec<_> = Iter::new(&node).collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0], (&42u64, &"answer"));
    }

    #[test]
    fn iter_all_elements() {
        let pairs: Vec<(u64, u64)> = (0..100).map(|i| (i, i * 10)).collect();
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        let items: Vec<_> = Iter::new(&node).collect();
        assert_eq!(items.len(), 100);
    }

    #[test]
    fn sorted_pairs_in_order() {
        let pairs: Vec<(u64, u64)> = (0..50).map(|i| (i * 3 + 1, i)).collect();
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        let sorted = sorted_pairs(&node);
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
        let pairs: Vec<(u64, &str)> = vec![(5, "e"), (10, "j"), (15, "o"), (20, "t")];
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        let sorted = sorted_pairs(&node);
        assert_eq!(sorted, pairs);
    }

    #[test]
    fn iter_after_inserts() {
        let pairs: Vec<(u64, u64)> = vec![(10, 1), (30, 3), (50, 5)];
        let mut node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();

        crate::insert::insert(&mut node, 20, 2);
        crate::insert::insert(&mut node, 40, 4);

        let sorted = sorted_pairs(&node);
        assert_eq!(sorted.len(), 5);
        let keys: Vec<u64> = sorted.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![10, 20, 30, 40, 50]);
    }
}
