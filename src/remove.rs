//! Remove algorithm for the learned index.

use crate::key::Key;
use crate::node::{Node, Slot};

/// Remove a key from the tree, returning the value if it existed.
pub fn remove<K: Key, V>(node: &mut Node<K, V>, key: &K) -> Option<V> {
    let slot_idx = node.predict_slot(*key);

    match &mut node.slots[slot_idx] {
        Slot::Empty => None,
        Slot::Data(k, _) => {
            if k == key {
                // Take the slot, replacing with Empty
                let old = std::mem::replace(&mut node.slots[slot_idx], Slot::Empty);
                node.num_keys -= 1;
                match old {
                    Slot::Data(_, v) => Some(v),
                    _ => unreachable!(),
                }
            } else {
                None
            }
        }
        Slot::Child(child) => remove(child, key),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn remove_existing_key() {
        let pairs = vec![(10u64, "a"), (20, "b"), (30, "c")];
        let mut node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();

        let removed = remove(&mut node, &20);
        assert_eq!(removed, Some("b"));
        assert_eq!(node.total_keys(), 2);
        assert_eq!(crate::lookup::get(&node, &20), None);
    }

    #[test]
    fn remove_missing_key() {
        let pairs = vec![(10u64, "a"), (20, "b")];
        let mut node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();

        let removed = remove(&mut node, &99);
        assert!(removed.is_none());
        assert_eq!(node.total_keys(), 2);
    }

    #[test]
    fn remove_all_keys() {
        let pairs = vec![(1u64, 'a'), (2, 'b'), (3, 'c')];
        let mut node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();

        assert_eq!(remove(&mut node, &1), Some('a'));
        assert_eq!(remove(&mut node, &2), Some('b'));
        assert_eq!(remove(&mut node, &3), Some('c'));
        assert_eq!(node.total_keys(), 0);
    }

    #[test]
    fn remove_then_reinsert() {
        let pairs = vec![(10u64, "a"), (20, "b")];
        let mut node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();

        remove(&mut node, &10);
        assert_eq!(crate::lookup::get(&node, &10), None);

        crate::insert::insert(&mut node, 10, "A", &Config::default());
        assert_eq!(crate::lookup::get(&node, &10), Some(&"A"));
    }

    #[test]
    fn remove_idempotent() {
        let pairs = vec![(5u64, "x")];
        let mut node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();

        assert_eq!(remove(&mut node, &5), Some("x"));
        assert_eq!(remove(&mut node, &5), None);
    }
}
