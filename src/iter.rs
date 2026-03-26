//! Lock-free in-order iterator and range queries for the learned index tree.
#![allow(unsafe_code)]

use std::ops::{Bound, RangeBounds};

use crossbeam_epoch::Guard;

use crate::key::Key;
use crate::node::{is_child, Node, SLOT_DATA, SLOT_WRITING};

/// An iterator over the key-value pairs in a learned index in sorted order.
///
/// Yields references tied to the lifetime of the epoch guard. The left-to-right
/// DFS traversal produces keys in ascending order because the linear model is
/// monotonic (non-negative slope fitted from sorted keys) and children at
/// slot `s` contain only keys whose predicted position is `s`.
pub struct Iter<'g, K, V> {
    /// Stack of (node, `next_slot_index`) for DFS traversal.
    stack: Vec<(&'g Node<K, V>, usize)>,
    /// The epoch guard that keeps referenced data alive.
    guard: &'g Guard,
    /// Approximate remaining entries, used for `size_hint`.
    remaining: Option<usize>,
}

impl<'g, K: Key, V> Iter<'g, K, V> {
    /// Create a new iterator starting from the root node.
    pub fn new(root: &'g Node<K, V>, guard: &'g Guard) -> Self {
        Self {
            stack: vec![(root, 0)],
            guard,
            remaining: None,
        }
    }

    /// Create a new iterator with an approximate entry count hint.
    ///
    /// The hint is used by [`size_hint`](Iterator::size_hint) to help
    /// callers like `collect()` pre-allocate. It does not need to be exact.
    pub fn with_hint(root: &'g Node<K, V>, guard: &'g Guard, count: usize) -> Self {
        Self {
            stack: vec![(root, 0)],
            guard,
            remaining: Some(count),
        }
    }
}

impl<'g, K: Key, V> Iterator for Iter<'g, K, V> {
    type Item = (&'g K, &'g V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (node, slot_idx) = self.stack.last_mut()?;
            if *slot_idx >= node.capacity() {
                self.stack.pop();
                continue;
            }
            let current_idx = *slot_idx;
            *slot_idx += 1;

            let state = node.slot_state(current_idx);
            match state {
                SLOT_DATA => {
                    if let Some(r) = &mut self.remaining {
                        *r = r.saturating_sub(1);
                    }
                    // SAFETY: state is DATA, inline data is valid.
                    let key = unsafe { node.read_key(current_idx) };
                    let value = unsafe { node.read_value(current_idx) };
                    return Some((key, value));
                }
                s if is_child(s) => {
                    let child_shared = node.load_child(current_idx, self.guard);
                    if !child_shared.is_null() {
                        let child = unsafe { child_shared.deref() };
                        self.stack.push((child, 0));
                    }
                }
                SLOT_WRITING => {
                    // A concurrent insert is claiming this slot. Spin briefly
                    // so rebuild snapshots don't miss in-flight writes.
                    *slot_idx -= 1; // re-visit this slot
                    std::hint::spin_loop();
                    continue;
                }
                _ => continue, // EMPTY, TOMBSTONE
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.remaining.map_or((0, None), |r| (r, Some(r)))
    }
}

/// Collect all key-value pairs from a tree in sorted order.
///
/// This performs a full DFS traversal and clones all entries. The traversal
/// naturally produces sorted output (see [`Iter`] docs).
pub fn sorted_pairs<K: Key, V: Clone>(root: &Node<K, V>, guard: &Guard) -> Vec<(K, V)> {
    let iter = Iter::new(root, guard);
    iter.map(|(k, v)| (k.clone(), v.clone())).collect()
}

/// A range iterator over key-value pairs in a learned index.
///
/// Yields only entries whose keys fall within the specified bounds, in
/// ascending key order. Uses model-guided seek for O(depth) initialization
/// when the start bound is specified.
pub struct Range<'g, K, V> {
    /// Stack of (node, `next_slot_index`) for DFS traversal.
    stack: Vec<(&'g Node<K, V>, usize)>,
    /// The epoch guard that keeps referenced data alive.
    guard: &'g Guard,
    /// Lower bound of the range.
    start: Bound<K>,
    /// Upper bound of the range.
    end: Bound<K>,
    /// Whether we have yielded at least one entry (past the start bound).
    started: bool,
    /// Whether we have passed the end bound (iterator exhausted).
    done: bool,
}

impl<'g, K: Key, V> Range<'g, K, V> {
    /// Create a new range iterator over the given bounds.
    pub fn new<R: RangeBounds<K>>(root: &'g Node<K, V>, range: R, guard: &'g Guard) -> Self {
        let start = match range.start_bound() {
            Bound::Included(k) => Bound::Included(k.clone()),
            Bound::Excluded(k) => Bound::Excluded(k.clone()),
            Bound::Unbounded => Bound::Unbounded,
        };
        let end = match range.end_bound() {
            Bound::Included(k) => Bound::Included(k.clone()),
            Bound::Excluded(k) => Bound::Excluded(k.clone()),
            Bound::Unbounded => Bound::Unbounded,
        };

        let is_unbounded = matches!(&start, Bound::Unbounded);

        let mut iter = Self {
            stack: Vec::new(),
            guard,
            start,
            end,
            started: is_unbounded,
            done: false,
        };

        let seek_key = match &iter.start {
            Bound::Included(k) | Bound::Excluded(k) => Some(k.clone()),
            Bound::Unbounded => None,
        };
        if let Some(ref k) = seek_key {
            iter.seek_to(root, k);
        } else {
            iter.stack.push((root, 0));
        }

        iter
    }

    /// Seek the DFS stack to the predicted position of `key`.
    ///
    /// At each level, predicts the slot for `key` and pushes the node starting
    /// at that slot. If the slot contains a child, pushes a continuation for the
    /// parent at `slot + 1` and recurses into the child.
    fn seek_to(&mut self, node: &'g Node<K, V>, key: &K) {
        let p = node.predict_slot(key);
        let state = node.slot_state(p);
        if is_child(state) {
            let child_shared = node.load_child(p, self.guard);
            if !child_shared.is_null() {
                let child = unsafe { child_shared.deref() };
                self.stack.push((node, p + 1));
                self.seek_to(child, key);
                return;
            }
        }
        self.stack.push((node, p));
    }

    /// Check if a key is past the end bound.
    fn past_end(&self, key: &K) -> bool {
        match &self.end {
            Bound::Included(end) => key > end,
            Bound::Excluded(end) => key >= end,
            Bound::Unbounded => false,
        }
    }

    /// Check if a key is before the start bound.
    fn before_start(&self, key: &K) -> bool {
        match &self.start {
            Bound::Included(start) => key < start,
            Bound::Excluded(start) => key <= start,
            Bound::Unbounded => false,
        }
    }
}

impl<'g, K: Key, V> Iterator for Range<'g, K, V> {
    type Item = (&'g K, &'g V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        loop {
            let (node, slot_idx) = self.stack.last_mut()?;

            if *slot_idx >= node.capacity() {
                self.stack.pop();
                continue;
            }

            let current_idx = *slot_idx;
            *slot_idx += 1;

            let state = node.slot_state(current_idx);
            match state {
                SLOT_DATA => {
                    // SAFETY: state is DATA, inline data is valid.
                    let key = unsafe { node.read_key(current_idx) };
                    let value = unsafe { node.read_value(current_idx) };
                    if self.past_end(key) {
                        self.done = true;
                        return None;
                    }
                    if !self.started {
                        if self.before_start(key) {
                            continue;
                        }
                        self.started = true;
                    }
                    return Some((key, value));
                }
                s if is_child(s) => {
                    let child_shared = node.load_child(current_idx, self.guard);
                    if !child_shared.is_null() {
                        let child = unsafe { child_shared.deref() };
                        self.stack.push((child, 0));
                    }
                }
                SLOT_WRITING => {
                    // A concurrent insert is claiming this slot. Spin briefly
                    // so rebuild snapshots don't miss in-flight writes.
                    *slot_idx -= 1; // re-visit this slot
                    std::hint::spin_loop();
                    continue;
                }
                _ => continue, // EMPTY, TOMBSTONE
            }
        }
    }
}

/// Return the first (minimum) key-value pair in the tree.
///
/// Returns `None` if the tree is empty. O(depth) typical.
pub fn first_entry<'g, K: Key, V>(
    root: &'g Node<K, V>,
    guard: &'g Guard,
) -> Option<(&'g K, &'g V)> {
    Iter::new(root, guard).next()
}

/// Return the last (maximum) key-value pair in the tree.
///
/// Scans slots right-to-left at each level, following the rightmost non-empty
/// slot. Returns `None` if the tree is empty. O(depth) typical.
pub fn last_entry<'g, K: Key, V>(root: &'g Node<K, V>, guard: &'g Guard) -> Option<(&'g K, &'g V)> {
    let mut node = root;
    loop {
        let mut child_found = None;
        for i in (0..node.capacity()).rev() {
            let state = node.slot_state(i);
            match state {
                SLOT_DATA => {
                    // SAFETY: state is DATA, inline data is valid.
                    let key = unsafe { node.read_key(i) };
                    let value = unsafe { node.read_value(i) };
                    return Some((key, value));
                }
                s if is_child(s) => {
                    let child_shared = node.load_child(i, guard);
                    if !child_shared.is_null() {
                        child_found = Some(unsafe { child_shared.deref() });
                        break;
                    }
                }
                _ => continue,
            }
        }
        node = child_found?;
    }
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

        crate::insert::insert(&node, 20, &2, &Config::default(), &g);
        crate::insert::insert(&node, 40, &4, &Config::default(), &g);

        let sorted = sorted_pairs(&node, &g);
        assert_eq!(sorted.len(), 5);
        let keys: Vec<u64> = sorted.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![10, 20, 30, 40, 50]);
    }

    // -----------------------------------------------------------------------
    // Iter sortedness (Part A validation)
    // -----------------------------------------------------------------------

    #[test]
    fn iter_raw_is_sorted_bulk_load() {
        let g = guard();
        let pairs: Vec<(u64, u64)> = (0..500).map(|i| (i * 3 + 1, i)).collect();
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        let keys: Vec<u64> = Iter::new(&node, &g).map(|(k, _)| *k).collect();
        assert_eq!(keys.len(), 500);
        for w in keys.windows(2) {
            assert!(w[0] < w[1], "not sorted: {} >= {}", w[0], w[1]);
        }
    }

    #[test]
    fn iter_raw_is_sorted_after_inserts() {
        let g = guard();
        let pairs: Vec<(u64, u64)> = (0..100).map(|i| (i * 4, i)).collect();
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        for i in 0..100u64 {
            crate::insert::insert(&node, i * 4 + 2, &(i + 1000), &Config::default(), &g);
        }
        let keys: Vec<u64> = Iter::new(&node, &g).map(|(k, _)| *k).collect();
        assert_eq!(keys.len(), 200);
        for w in keys.windows(2) {
            assert!(w[0] < w[1], "not sorted: {} >= {}", w[0], w[1]);
        }
    }

    #[test]
    fn iter_raw_is_sorted_reverse_inserts() {
        let g = guard();
        let node = Node::<u64, u64>::with_capacity(crate::model::LinearModel::new(0.01, 0.0), 16);
        for i in (0..200u64).rev() {
            crate::insert::insert(&node, i, &i, &Config::default(), &g);
        }
        let keys: Vec<u64> = Iter::new(&node, &g).map(|(k, _)| *k).collect();
        assert_eq!(keys.len(), 200);
        for w in keys.windows(2) {
            assert!(w[0] < w[1], "not sorted: {} >= {}", w[0], w[1]);
        }
    }

    // -----------------------------------------------------------------------
    // Range iterator
    // -----------------------------------------------------------------------

    fn make_0_to_99() -> Node<u64, u64> {
        let pairs: Vec<(u64, u64)> = (0..100).map(|i| (i, i * 10)).collect();
        crate::build::bulk_load(&pairs, &Config::default()).unwrap()
    }

    #[test]
    fn range_inclusive_both() {
        let g = guard();
        let node = make_0_to_99();
        let items: Vec<u64> = Range::new(&node, 10..=20, &g).map(|(k, _)| *k).collect();
        assert_eq!(items.len(), 11);
        assert_eq!(items, (10..=20).collect::<Vec<_>>());
    }

    #[test]
    fn range_exclusive_end() {
        let g = guard();
        let node = make_0_to_99();
        let items: Vec<u64> = Range::new(&node, 10..20, &g).map(|(k, _)| *k).collect();
        assert_eq!(items.len(), 10);
        assert_eq!(items, (10..20).collect::<Vec<_>>());
    }

    #[test]
    fn range_from() {
        let g = guard();
        let node = make_0_to_99();
        let items: Vec<u64> = Range::new(&node, 90.., &g).map(|(k, _)| *k).collect();
        assert_eq!(items.len(), 10);
        assert_eq!(items, (90..100).collect::<Vec<_>>());
    }

    #[test]
    fn range_to() {
        let g = guard();
        let node = make_0_to_99();
        let items: Vec<u64> = Range::new(&node, ..5, &g).map(|(k, _)| *k).collect();
        assert_eq!(items.len(), 5);
        assert_eq!(items, (0..5).collect::<Vec<_>>());
    }

    #[test]
    fn range_to_inclusive() {
        let g = guard();
        let node = make_0_to_99();
        let items: Vec<u64> = Range::new(&node, ..=5, &g).map(|(k, _)| *k).collect();
        assert_eq!(items.len(), 6);
        assert_eq!(items, (0..=5).collect::<Vec<_>>());
    }

    #[test]
    fn range_empty_result() {
        let g = guard();
        let node = make_0_to_99();
        let items: Vec<u64> = Range::new(&node, 200..300, &g).map(|(k, _)| *k).collect();
        assert!(items.is_empty());
    }

    #[test]
    fn range_single_element() {
        let g = guard();
        let node = make_0_to_99();
        let items: Vec<u64> = Range::new(&node, 50..=50, &g).map(|(k, _)| *k).collect();
        assert_eq!(items, vec![50]);
    }

    // -----------------------------------------------------------------------
    // first_entry / last_entry
    // -----------------------------------------------------------------------

    #[test]
    fn first_entry_basic() {
        let g = guard();
        let node = make_0_to_99();
        let (k, v) = first_entry(&node, &g).unwrap();
        assert_eq!(*k, 0);
        assert_eq!(*v, 0);
    }

    #[test]
    fn last_entry_basic() {
        let g = guard();
        let node = make_0_to_99();
        let (k, v) = last_entry(&node, &g).unwrap();
        assert_eq!(*k, 99);
        assert_eq!(*v, 990);
    }

    #[test]
    fn first_entry_empty() {
        let g = guard();
        let node = Node::<u64, u64>::with_capacity(crate::model::LinearModel::constant(), 5);
        assert!(first_entry(&node, &g).is_none());
    }

    #[test]
    fn last_entry_empty() {
        let g = guard();
        let node = Node::<u64, u64>::with_capacity(crate::model::LinearModel::constant(), 5);
        assert!(last_entry(&node, &g).is_none());
    }

    #[test]
    fn size_hint_without_hint() {
        let g = guard();
        let pairs: Vec<(u64, u64)> = (0..10).map(|i| (i, i)).collect();
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        let iter = Iter::new(&node, &g);
        assert_eq!(iter.size_hint(), (0, None));
    }

    #[test]
    fn size_hint_with_hint() {
        let g = guard();
        let pairs: Vec<(u64, u64)> = (0..10).map(|i| (i, i)).collect();
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        let mut iter = Iter::with_hint(&node, &g, 10);
        assert_eq!(iter.size_hint(), (10, Some(10)));
        iter.next();
        assert_eq!(iter.size_hint(), (9, Some(9)));
    }
}
