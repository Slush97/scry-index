//! Lookup algorithm for the learned index.

use crate::key::Key;
use crate::node::{Node, Slot};

/// Look up a key in the tree, returning a reference to the value if found.
pub fn get<'a, K: Key, V>(node: &'a Node<K, V>, key: &K) -> Option<&'a V> {
    let slot_idx = node.predict_slot(*key);
    match &node.slots[slot_idx] {
        Slot::Empty => None,
        Slot::Data(k, v) => {
            if k == key {
                Some(v)
            } else {
                None
            }
        }
        Slot::Child(child) => get(child, key),
    }
}

/// Look up a key in the tree, returning a mutable reference to the value.
pub fn get_mut<'a, K: Key, V>(node: &'a mut Node<K, V>, key: &K) -> Option<&'a mut V> {
    let slot_idx = node.predict_slot(*key);
    match &mut node.slots[slot_idx] {
        Slot::Empty => None,
        Slot::Data(k, v) => {
            if k == key {
                Some(v)
            } else {
                None
            }
        }
        Slot::Child(child) => get_mut(child, key),
    }
}

/// Check whether a key exists in the tree.
pub fn contains_key<K: Key, V>(node: &Node<K, V>, key: &K) -> bool {
    get(node, key).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

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
        let tree = build_test_tree();
        assert_eq!(get(&tree, &10), Some(&"ten"));
        assert_eq!(get(&tree, &20), Some(&"twenty"));
        assert_eq!(get(&tree, &30), Some(&"thirty"));
        assert_eq!(get(&tree, &40), Some(&"forty"));
        assert_eq!(get(&tree, &50), Some(&"fifty"));
    }

    #[test]
    fn missing_keys_return_none() {
        let tree = build_test_tree();
        assert_eq!(get(&tree, &0), None);
        assert_eq!(get(&tree, &15), None);
        assert_eq!(get(&tree, &25), None);
        assert_eq!(get(&tree, &100), None);
    }

    #[test]
    fn contains_key_works() {
        let tree = build_test_tree();
        assert!(contains_key(&tree, &10));
        assert!(contains_key(&tree, &50));
        assert!(!contains_key(&tree, &99));
    }

    #[test]
    fn get_mut_modifies_value() {
        let mut tree = build_test_tree();
        if let Some(v) = get_mut(&mut tree, &20) {
            *v = "TWENTY";
        }
        assert_eq!(get(&tree, &20), Some(&"TWENTY"));
    }

    #[test]
    fn single_element_tree() {
        let pairs = vec![(42u64, "answer")];
        let tree = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        assert_eq!(get(&tree, &42), Some(&"answer"));
        assert_eq!(get(&tree, &0), None);
    }

    #[test]
    fn large_tree_all_found() {
        let pairs: Vec<(u64, u64)> = (0..1000).map(|i| (i * 3, i)).collect();
        let tree = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        for (k, v) in &pairs {
            assert_eq!(
                get(&tree, k),
                Some(v),
                "key {k} not found in tree"
            );
        }
    }

    #[test]
    fn large_tree_gaps_not_found() {
        let pairs: Vec<(u64, u64)> = (0..100).map(|i| (i * 10, i)).collect();
        let tree = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        // Keys between multiples of 10 should not be found
        for i in 0..100 {
            let gap_key = i * 10 + 5;
            assert_eq!(get(&tree, &gap_key), None, "gap key {gap_key} should not be found");
        }
    }
}
