//! Insert algorithm using LIPP's chain method.
//!
//! When inserting a key into a node:
//! - If the predicted slot is empty, place the key-value there.
//! - If the slot contains a different key, create a child node containing both.
//! - If the slot contains the same key, update the value.
//! - If the slot contains a child, recurse into it.

use crate::config::Config;
use crate::key::Key;
use crate::model::fit_fmcd;
use crate::node::{Node, Slot};

/// Insert a key-value pair into the tree rooted at `node`.
///
/// Returns `Some(old_value)` if the key already existed, `None` otherwise.
pub fn insert<K: Key, V: Clone>(node: &mut Node<K, V>, key: K, value: V, config: &Config) -> Option<V> {
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
                // Conflict — create a child node containing both entries
                let ek = *existing_key;
                let ev = existing_value.clone();
                let child = build_conflict_node(ek, ev, key, value, config);
                node.slots[slot_idx] = Slot::Child(Box::new(child));
                node.num_keys -= 1; // the existing data entry moved to child
                None
            }
        }
        Slot::Child(child) => insert(child, key, value, config),
    }
}

/// Build a new child node from two conflicting key-value pairs.
fn build_conflict_node<K: Key, V: Clone>(
    k1: K,
    v1: V,
    k2: K,
    v2: V,
    config: &Config,
) -> Node<K, V> {
    let (first_k, first_v, second_k, second_v) = if k1 < k2 {
        (k1, v1, k2, v2)
    } else {
        (k2, v2, k1, v1)
    };

    let keys = [first_k, second_k];
    let result = fit_fmcd(&keys, config.expansion_factor);
    let mut node = Node::with_capacity(result.model, result.array_size);

    let s1 = node.predict_slot(first_k);
    let s2 = node.predict_slot(second_k);

    if s1 == s2 {
        // Even after FMCD, they still collide — create another level
        node.slots[s1] = Slot::Data(first_k, first_v);
        node.num_keys += 1;
        // Insert the second key, which will recurse and create a child
        insert(&mut node, second_k, second_v, config);
    } else {
        node.slots[s1] = Slot::Data(first_k, first_v);
        node.slots[s2] = Slot::Data(second_k, second_v);
        node.num_keys += 2;
    }

    node
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> Config {
        Config::default()
    }

    fn empty_root() -> Node<u64, u64> {
        let model = crate::model::LinearModel::new(0.1, 0.0);
        Node::with_capacity(model, 100)
    }

    #[test]
    fn insert_into_empty_slot() {
        let mut node = empty_root();
        let old = insert(&mut node, 50, 500, &default_config());
        assert!(old.is_none());
        assert_eq!(node.total_keys(), 1);
    }

    #[test]
    fn insert_duplicate_returns_old() {
        let mut node = empty_root();
        insert(&mut node, 50, 500, &default_config());
        let old = insert(&mut node, 50, 5000, &default_config());
        assert_eq!(old, Some(500));
        assert_eq!(node.total_keys(), 1);
    }

    #[test]
    fn insert_conflict_creates_child() {
        let pairs: Vec<(u64, &str)> = vec![(10, "a"), (20, "b")];
        let mut node = crate::build::bulk_load(&pairs, &default_config()).unwrap();
        let initial_keys = node.total_keys();

        // Insert a key that might conflict
        insert(&mut node, 15, "c", &default_config());
        assert_eq!(node.total_keys(), initial_keys + 1);

        // All three keys should be findable
        assert_eq!(crate::lookup::get(&node, &10), Some(&"a"));
        assert_eq!(crate::lookup::get(&node, &15), Some(&"c"));
        assert_eq!(crate::lookup::get(&node, &20), Some(&"b"));
    }

    #[test]
    fn insert_many_sequential() {
        let mut node = empty_root();
        for i in 0..100u64 {
            insert(&mut node, i, i, &default_config());
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
            insert(&mut node, i, i * 10, &default_config());
        }
        assert_eq!(node.total_keys(), 50);
        for i in 0..50u64 {
            assert_eq!(crate::lookup::get(&node, &i), Some(&(i * 10)));
        }
    }

    #[test]
    fn insert_update_preserves_count() {
        let mut node = empty_root();
        insert(&mut node, 1, 10, &default_config());
        insert(&mut node, 2, 20, &default_config());
        insert(&mut node, 3, 30, &default_config());
        assert_eq!(node.total_keys(), 3);

        // Update existing key
        insert(&mut node, 2, 200, &default_config());
        assert_eq!(node.total_keys(), 3); // count unchanged
        assert_eq!(crate::lookup::get(&node, &2), Some(&200));
    }

    #[test]
    fn insert_into_bulk_loaded_tree() {
        let pairs: Vec<(u64, u64)> = (0..100).map(|i| (i * 2, i)).collect();
        let mut node = crate::build::bulk_load(&pairs, &default_config()).unwrap();

        // Insert keys in the gaps
        for i in 0..100u64 {
            insert(&mut node, i * 2 + 1, i + 1000, &default_config());
        }

        assert_eq!(node.total_keys(), 200);

        // Verify all 200 keys are findable
        for i in 0..200u64 {
            assert!(
                crate::lookup::get(&node, &i).is_some(),
                "key {i} not found after mixed bulk_load + insert"
            );
        }
    }
}
