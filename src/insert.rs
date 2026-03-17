//! Insert algorithm using LIPP's chain method.
//!
//! When inserting a key into a node:
//! - If the predicted slot is empty, place the key-value there.
//! - If the slot contains a different key, create a child node containing both.
//! - If the slot contains the same key, update the value.
//! - If the slot contains a child, recurse into it.

use crate::key::Key;
use crate::model::LinearModel;
use crate::node::{Node, Slot};

/// Insert a key-value pair into the tree rooted at `node`.
///
/// Returns `Some(old_value)` if the key already existed, `None` otherwise.
pub fn insert<K: Key, V: Clone>(node: &mut Node<K, V>, key: K, value: V) -> Option<V> {
    let slot_idx = node.predict_slot(key);

    match &mut node.slots[slot_idx] {
        Slot::Empty => {
            node.slots[slot_idx] = Slot::Data(key, value);
            node.num_keys += 1;
            None
        }
        Slot::Data(existing_key, existing_value) => {
            if *existing_key == key {
                // Key already exists — update value
                let old = existing_value.clone();
                *existing_value = value;
                Some(old)
            } else {
                // Conflict — create a small child node for both entries
                let ek = *existing_key;
                let ev = existing_value.clone();
                let child = build_conflict_node(ek, ev, key, value);
                node.slots[slot_idx] = Slot::Child(Box::new(child));
                node.num_keys -= 1; // the existing data entry moved to child
                None
            }
        }
        Slot::Child(child) => insert(child, key, value),
    }
}

/// Build a small child node from two conflicting key-value pairs.
///
/// Uses direct placement into a 4-slot node instead of full FMCD fitting.
/// For just 2 keys, FMCD is overkill — a simple linear interpolation into
/// a tiny array is sufficient and avoids the allocation + scan overhead.
fn build_conflict_node<K: Key, V: Clone>(k1: K, v1: V, k2: K, v2: V) -> Node<K, V> {
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
        // Same key value (shouldn't happen — caller checks equality) — stack them
        (0.0, 0.0)
    } else {
        let s = (array_size - 1) as f64 / key_range;
        (s, -s * lo_f)
    };

    let model = LinearModel::new(slope, intercept);
    let mut node = Node::with_capacity(model, array_size);

    let s1 = node.predict_slot(lo_k);
    let s2 = node.predict_slot(hi_k);

    if s1 == s2 {
        // Still collide (e.g., identical model inputs) — put one and chain the other
        node.slots[s1] = Slot::Data(lo_k, lo_v);
        node.num_keys += 1;
        insert(&mut node, hi_k, hi_v);
    } else {
        node.slots[s1] = Slot::Data(lo_k, lo_v);
        node.slots[s2] = Slot::Data(hi_k, hi_v);
        node.num_keys += 2;
    }

    node
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn empty_root() -> Node<u64, u64> {
        let model = LinearModel::new(0.1, 0.0);
        Node::with_capacity(model, 100)
    }

    #[test]
    fn insert_into_empty_slot() {
        let mut node = empty_root();
        let old = insert(&mut node, 50, 500);
        assert!(old.is_none());
        assert_eq!(node.total_keys(), 1);
    }

    #[test]
    fn insert_duplicate_returns_old() {
        let mut node = empty_root();
        insert(&mut node, 50, 500);
        let old = insert(&mut node, 50, 5000);
        assert_eq!(old, Some(500));
        assert_eq!(node.total_keys(), 1);
    }

    #[test]
    fn insert_conflict_creates_child() {
        let pairs: Vec<(u64, &str)> = vec![(10, "a"), (20, "b")];
        let mut node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        let initial_keys = node.total_keys();

        insert(&mut node, 15, "c");
        assert_eq!(node.total_keys(), initial_keys + 1);

        assert_eq!(crate::lookup::get(&node, &10), Some(&"a"));
        assert_eq!(crate::lookup::get(&node, &15), Some(&"c"));
        assert_eq!(crate::lookup::get(&node, &20), Some(&"b"));
    }

    #[test]
    fn insert_many_sequential() {
        let mut node = empty_root();
        for i in 0..100u64 {
            insert(&mut node, i, i);
        }
        assert_eq!(node.total_keys(), 100);
        for i in 0..100u64 {
            assert_eq!(
                crate::lookup::get(&node, &i),
                Some(&i),
                "key {i} not found after sequential insert"
            );
        }
    }

    #[test]
    fn insert_reverse_order() {
        let mut node = empty_root();
        for i in (0..50u64).rev() {
            insert(&mut node, i, i * 10);
        }
        assert_eq!(node.total_keys(), 50);
        for i in 0..50u64 {
            assert_eq!(crate::lookup::get(&node, &i), Some(&(i * 10)));
        }
    }

    #[test]
    fn insert_update_preserves_count() {
        let mut node = empty_root();
        insert(&mut node, 1, 10);
        insert(&mut node, 2, 20);
        insert(&mut node, 3, 30);
        assert_eq!(node.total_keys(), 3);

        insert(&mut node, 2, 200);
        assert_eq!(node.total_keys(), 3);
        assert_eq!(crate::lookup::get(&node, &2), Some(&200));
    }

    #[test]
    fn insert_into_bulk_loaded_tree() {
        let pairs: Vec<(u64, u64)> = (0..100).map(|i| (i * 2, i)).collect();
        let mut node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();

        for i in 0..100u64 {
            insert(&mut node, i * 2 + 1, i + 1000);
        }

        assert_eq!(node.total_keys(), 200);

        for i in 0..200u64 {
            assert!(
                crate::lookup::get(&node, &i).is_some(),
                "key {i} not found after mixed bulk_load + insert"
            );
        }
    }

    #[test]
    fn conflict_node_is_small() {
        let node = build_conflict_node(10u64, "a", 20u64, "b");
        assert_eq!(node.capacity(), 4);
        assert_eq!(node.total_keys(), 2);
    }

    #[test]
    fn conflict_node_both_findable() {
        let node = build_conflict_node(100u64, 1, 200u64, 2);
        assert_eq!(crate::lookup::get(&node, &100), Some(&1));
        assert_eq!(crate::lookup::get(&node, &200), Some(&2));
    }
}
