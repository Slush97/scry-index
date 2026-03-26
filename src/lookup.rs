//! Lock-free lookup algorithm for the learned index.
#![allow(unsafe_code)]

use crossbeam_epoch::Guard;

use crate::key::Key;
use crate::node::{is_child, Node, SLOT_DATA};

/// Look up a key in the tree, returning a reference to the value if found.
///
/// The returned reference is valid for the lifetime of the guard.
pub fn get<'g, K: Key, V>(node: &'g Node<K, V>, key: &K, guard: &'g Guard) -> Option<&'g V> {
    let mut current_node = node;
    loop {
        let slot_idx = current_node.predict_slot(key);
        let state = current_node.slot_state(slot_idx);

        match state {
            SLOT_DATA => {
                // SAFETY: state is DATA so inline data is valid and immutable.
                let k = unsafe { current_node.read_key(slot_idx) };
                return if k == key {
                    Some(unsafe { current_node.read_value(slot_idx) })
                } else {
                    None
                };
            }
            s if is_child(s) => {
                let child_shared = current_node.load_child(slot_idx, guard);
                if child_shared.is_null() {
                    return None;
                }
                // SAFETY: child_shared is valid for guard lifetime.
                current_node = unsafe { child_shared.deref() };
            }
            _ => return None, // EMPTY, WRITING, TOMBSTONE
        }
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

    fn build_test_tree() -> Node<u64, &'static str> {
        let pairs: Vec<(u64, &str)> = vec![
            (10, "ten"),
            (20, "twenty"),
            (30, "thirty"),
            (40, "forty"),
            (50, "fifty"),
        ];
        crate::build::bulk_load(&pairs, &Config::default()).unwrap()
    }

    #[test]
    fn find_existing_keys() {
        let g = guard();
        let tree = build_test_tree();
        assert_eq!(get(&tree, &10, &g), Some(&"ten"));
        assert_eq!(get(&tree, &20, &g), Some(&"twenty"));
        assert_eq!(get(&tree, &30, &g), Some(&"thirty"));
        assert_eq!(get(&tree, &40, &g), Some(&"forty"));
        assert_eq!(get(&tree, &50, &g), Some(&"fifty"));
    }

    #[test]
    fn missing_keys_return_none() {
        let g = guard();
        let tree = build_test_tree();
        assert_eq!(get(&tree, &0, &g), None);
        assert_eq!(get(&tree, &15, &g), None);
        assert_eq!(get(&tree, &25, &g), None);
        assert_eq!(get(&tree, &100, &g), None);
    }

    #[test]
    fn contains_key_works() {
        let g = guard();
        let tree = build_test_tree();
        assert!(get(&tree, &10, &g).is_some());
        assert!(get(&tree, &50, &g).is_some());
        assert!(get(&tree, &99, &g).is_none());
    }

    #[test]
    fn single_element_tree() {
        let g = guard();
        let pairs = vec![(42u64, "answer")];
        let tree = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        assert_eq!(get(&tree, &42, &g), Some(&"answer"));
        assert_eq!(get(&tree, &0, &g), None);
    }

    #[test]
    fn large_tree_all_found() {
        let g = guard();
        let pairs: Vec<(u64, u64)> = (0..1000).map(|i| (i * 3, i)).collect();
        let tree = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        for (k, v) in &pairs {
            assert_eq!(get(&tree, k, &g), Some(v), "key {k} not found in tree");
        }
    }

    #[test]
    fn large_tree_gaps_not_found() {
        let g = guard();
        let pairs: Vec<(u64, u64)> = (0..100).map(|i| (i * 10, i)).collect();
        let tree = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        for i in 0..100 {
            let gap_key = i * 10 + 5;
            assert_eq!(
                get(&tree, &gap_key, &g),
                None,
                "gap key {gap_key} should not be found"
            );
        }
    }
}
