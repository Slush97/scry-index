//! Lock-free insert algorithm using LIPP's chain method with CAS retry loops.
//!
//! When inserting a key into a node:
//! - If the predicted slot is empty, CAS null → Data.
//! - If the slot contains the same key, CAS old → new Data (update).
//! - If the slot contains a different key, build a child and CAS old → Child.
//! - If the slot contains a child, recurse into it.
//! - On CAS failure, retry from the slot load.
#![allow(unsafe_code)]

use std::sync::atomic::Ordering;

use crossbeam_epoch::{self as epoch, Guard, Owned};

use crate::key::Key;
use crate::model::LinearModel;
use crate::node::{Node, SlotInner};

/// Result of an insert operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertResult {
    /// A new key was inserted.
    Inserted,
    /// An existing key was updated with a new value.
    Updated,
}

/// Insert a key-value pair into the tree rooted at `node`.
///
/// Returns [`InsertResult::Inserted`] if the key was new, or
/// [`InsertResult::Updated`] if an existing key's value was replaced.
pub fn insert<K: Key, V: Clone + Send + Sync>(
    node: &Node<K, V>,
    key: K,
    value: &V,
    guard: &Guard,
) -> InsertResult {
    let mut current_node = node;
    loop {
        let slot_idx = current_node.predict_slot(key);
        let slot = current_node.slot(slot_idx);

        let current = slot.load(Ordering::Acquire, guard);

        if current.is_null() {
            // Empty slot: CAS null → Data
            let new = Owned::new(SlotInner::Data {
                key,
                value: value.clone(),
            });
            if slot
                .compare_exchange(current, new, Ordering::AcqRel, Ordering::Acquire, guard)
                .is_ok()
            {
                current_node.inc_keys();
                return InsertResult::Inserted;
            }
            continue;
        }

        // SAFETY: current is not null and valid for the lifetime of the guard.
        let inner = unsafe { current.deref() };

        match inner {
            SlotInner::Data {
                key: existing_key,
                value: existing_value,
            } => {
                if *existing_key == key {
                    // Same key: CAS old Data → new Data (update)
                    let new = Owned::new(SlotInner::Data {
                        key,
                        value: value.clone(),
                    });
                    if slot
                        .compare_exchange(
                            current, new, Ordering::AcqRel, Ordering::Acquire, guard,
                        )
                        .is_ok()
                    {
                        // SAFETY: We successfully CAS'd out `current`. No new reader
                        // will load it. Existing readers are protected by their guards.
                        unsafe {
                            guard.defer_destroy(current);
                        }
                        return InsertResult::Updated;
                    }
                    // CAS failed — slot changed, retry
                    continue;
                }
                // Collision: build child containing both entries, CAS old → Child
                let ek = *existing_key;
                let ev = existing_value.clone();
                let child = build_conflict_node(ek, ev, key, value.clone());
                let new = Owned::new(SlotInner::Child(child));
                if slot
                    .compare_exchange(current, new, Ordering::AcqRel, Ordering::Acquire, guard)
                    .is_ok()
                {
                    // SAFETY: CAS succeeded; old Data is unreachable to new readers.
                    unsafe {
                        guard.defer_destroy(current);
                    }
                    current_node.dec_keys(); // existing data moved to child
                    return InsertResult::Inserted;
                }
                // CAS failed — slot changed, retry
            }
            SlotInner::Child(child) => {
                current_node = child;
            }
        }
    }
}

/// Build a small child node from two conflicting key-value pairs.
///
/// Uses direct placement into a 4-slot node instead of full FMCD fitting.
/// For just 2 keys, FMCD is overkill — a simple linear interpolation into
/// a tiny array is sufficient and avoids the allocation + scan overhead.
fn build_conflict_node<K: Key, V: Clone + Send + Sync>(
    k1: K,
    v1: V,
    k2: K,
    v2: V,
) -> Node<K, V> {
    let (lo_k, lo_v, hi_k, hi_v) = if k1 < k2 {
        (k1, v1, k2, v2)
    } else {
        (k2, v2, k1, v1)
    };

    let lo_f = lo_k.to_model_input();
    let hi_f = hi_k.to_model_input();
    let key_range = hi_f - lo_f;

    // Use a 4-slot array: map lo to slot 0, hi to slot 3
    let array_size = 4;

    let (slope, intercept) = if key_range.abs() < f64::EPSILON {
        // Same model input (shouldn't happen for distinct integer keys) — stack them
        (0.0, 0.0)
    } else {
        let s = (array_size - 1) as f64 / key_range;
        (s, -s * lo_f)
    };

    let model = LinearModel::new(slope, intercept);
    let node = Node::with_capacity(model, array_size);

    let s1 = node.predict_slot(lo_k);
    let s2 = node.predict_slot(hi_k);

    node.store_slot(
        s1,
        SlotInner::Data {
            key: lo_k,
            value: lo_v,
        },
    );
    node.inc_keys();

    if s1 == s2 {
        // Still collide — insert the second key via the concurrent insert path
        // (with unprotected guard since this is single-threaded construction).
        // SAFETY: Exclusive access during construction. The unprotected guard
        // is safe because no concurrent readers exist for this node yet.
        unsafe {
            let guard = epoch::unprotected();
            insert(&node, hi_k, &hi_v, guard);
        }
    } else {
        node.store_slot(
            s2,
            SlotInner::Data {
                key: hi_k,
                value: hi_v,
            },
        );
        node.inc_keys();
    }

    node
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    use crossbeam_epoch as epoch;

    fn guard() -> epoch::Guard {
        epoch::pin()
    }

    fn empty_root() -> Node<u64, u64> {
        let model = LinearModel::new(0.1, 0.0);
        Node::with_capacity(model, 100)
    }

    #[test]
    fn insert_into_empty_slot() {
        let g = guard();
        let node = empty_root();
        let result = insert(&node, 50, &500, &g);
        assert_eq!(result, InsertResult::Inserted);
        assert_eq!(node.total_keys(&g), 1);
    }

    #[test]
    fn insert_duplicate_returns_updated() {
        let g = guard();
        let node = empty_root();
        insert(&node, 50, &500, &g);
        let result = insert(&node, 50, &5000, &g);
        assert_eq!(result, InsertResult::Updated);
        assert_eq!(node.total_keys(&g), 1);
    }

    #[test]
    fn insert_conflict_creates_child() {
        let g = guard();
        let pairs: Vec<(u64, &str)> = vec![(10, "a"), (20, "b")];
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        let initial_keys = node.total_keys(&g);

        insert(&node, 15, &"c", &g);
        assert_eq!(node.total_keys(&g), initial_keys + 1);

        assert_eq!(crate::lookup::get(&node, &10, &g), Some(&"a"));
        assert_eq!(crate::lookup::get(&node, &15, &g), Some(&"c"));
        assert_eq!(crate::lookup::get(&node, &20, &g), Some(&"b"));
    }

    #[test]
    fn insert_many_sequential() {
        let g = guard();
        let node = empty_root();
        for i in 0..100u64 {
            insert(&node, i, &i, &g);
        }
        assert_eq!(node.total_keys(&g), 100);
        for i in 0..100u64 {
            assert_eq!(
                crate::lookup::get(&node, &i, &g),
                Some(&i),
                "key {i} not found after sequential insert"
            );
        }
    }

    #[test]
    fn insert_reverse_order() {
        let g = guard();
        let node = empty_root();
        for i in (0..50u64).rev() {
            insert(&node, i, &(i * 10), &g);
        }
        assert_eq!(node.total_keys(&g), 50);
        for i in 0..50u64 {
            assert_eq!(crate::lookup::get(&node, &i, &g), Some(&(i * 10)));
        }
    }

    #[test]
    fn insert_update_preserves_count() {
        let g = guard();
        let node = empty_root();
        insert(&node, 1, &10, &g);
        insert(&node, 2, &20, &g);
        insert(&node, 3, &30, &g);
        assert_eq!(node.total_keys(&g), 3);

        insert(&node, 2, &200, &g);
        assert_eq!(node.total_keys(&g), 3);
        assert_eq!(crate::lookup::get(&node, &2, &g), Some(&200));
    }

    #[test]
    fn insert_into_bulk_loaded_tree() {
        let g = guard();
        let pairs: Vec<(u64, u64)> = (0..100).map(|i| (i * 2, i)).collect();
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();

        for i in 0..100u64 {
            insert(&node, i * 2 + 1, &(i + 1000), &g);
        }

        assert_eq!(node.total_keys(&g), 200);

        for i in 0..200u64 {
            assert!(
                crate::lookup::get(&node, &i, &g).is_some(),
                "key {i} not found after mixed bulk_load + insert"
            );
        }
    }

    #[test]
    fn conflict_node_is_small() {
        let g = guard();
        let node = build_conflict_node(10u64, "a", 20u64, "b");
        assert_eq!(node.capacity(), 4);
        assert_eq!(node.total_keys(&g), 2);
    }

    #[test]
    fn conflict_node_both_findable() {
        let g = guard();
        let node = build_conflict_node(100u64, 1, 200u64, 2);
        assert_eq!(crate::lookup::get(&node, &100, &g), Some(&1));
        assert_eq!(crate::lookup::get(&node, &200, &g), Some(&2));
    }

    #[test]
    fn insert_update_returns_correct_result() {
        let g = guard();
        let node = empty_root();
        assert_eq!(insert(&node, 1, &10, &g), InsertResult::Inserted);
        assert_eq!(insert(&node, 1, &20, &g), InsertResult::Updated);
        assert_eq!(insert(&node, 2, &30, &g), InsertResult::Inserted);
    }

    #[test]
    fn insert_value_is_updated() {
        let g = guard();
        let node = empty_root();
        insert(&node, 42, &100, &g);
        assert_eq!(crate::lookup::get(&node, &42, &g), Some(&100));
        insert(&node, 42, &999, &g);
        assert_eq!(crate::lookup::get(&node, &42, &g), Some(&999));
    }
}
