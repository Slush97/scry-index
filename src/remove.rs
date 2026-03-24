//! Lock-free remove algorithm with CAS retry loop.
#![allow(unsafe_code)]

use std::sync::atomic::Ordering;

use crossbeam_epoch::{Guard, Shared};

use crate::key::Key;
use crate::node::{Node, SlotInner};

/// Remove a key from the tree, returning `true` if it was found and removed.
pub fn remove<K: Key, V>(node: &Node<K, V>, key: &K, guard: &Guard) -> bool {
    let mut current_node = node;
    loop {
        let slot_idx = current_node.predict_slot(key);
        let slot = current_node.slot(slot_idx);
        let current = slot.load(Ordering::Acquire, guard);

        if current.is_null() {
            return false;
        }

        // SAFETY: current is not null and valid for the lifetime of the guard.
        let inner = unsafe { current.deref() };

        match inner {
            SlotInner::Data { key: k, .. } => {
                if k != key {
                    return false; // different key
                }
                // Found: CAS current → null
                if slot
                    .compare_exchange(
                        current,
                        Shared::null(),
                        Ordering::AcqRel,
                        Ordering::Acquire,
                        guard,
                    )
                    .is_ok()
                {
                    // SAFETY: CAS succeeded; the old Data is unreachable to
                    // new readers. Defer destruction until all existing guards
                    // that may have loaded it are dropped.
                    unsafe {
                        guard.defer_destroy(current);
                    }
                    current_node.dec_keys();
                    return true;
                }
                // CAS failed — slot changed, retry
            }
            SlotInner::Child(child) => {
                current_node = child;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::insert;

    use crossbeam_epoch as epoch;

    fn guard() -> epoch::Guard {
        epoch::pin()
    }

    #[test]
    fn remove_existing_key() {
        let g = guard();
        let pairs = vec![(10u64, "a"), (20, "b"), (30, "c")];
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();

        let removed = remove(&node, &20, &g);
        assert!(removed);
        assert_eq!(node.total_keys(&g), 2);
        assert_eq!(crate::lookup::get(&node, &20, &g), None);
    }

    #[test]
    fn remove_missing_key() {
        let g = guard();
        let pairs = vec![(10u64, "a"), (20, "b")];
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();

        let removed = remove(&node, &99, &g);
        assert!(!removed);
        assert_eq!(node.total_keys(&g), 2);
    }

    #[test]
    fn remove_all_keys() {
        let g = guard();
        let pairs = vec![(1u64, 'a'), (2, 'b'), (3, 'c')];
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();

        assert!(remove(&node, &1, &g));
        assert!(remove(&node, &2, &g));
        assert!(remove(&node, &3, &g));
        assert_eq!(node.total_keys(&g), 0);
    }

    #[test]
    fn remove_then_reinsert() {
        let g = guard();
        let pairs = vec![(10u64, "a"), (20, "b")];
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();

        remove(&node, &10, &g);
        assert_eq!(crate::lookup::get(&node, &10, &g), None);

        insert::insert(&node, 10, &"A", &Config::default(), &g);
        assert_eq!(crate::lookup::get(&node, &10, &g), Some(&"A"));
    }

    #[test]
    fn remove_idempotent() {
        let g = guard();
        let pairs = vec![(5u64, "x")];
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();

        assert!(remove(&node, &5, &g));
        assert!(!remove(&node, &5, &g));
    }

    #[test]
    fn remove_from_child_node() {
        let g = guard();
        let pairs: Vec<(u64, &str)> = vec![(10, "a"), (20, "b")];
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();

        // Insert a key that may create a child
        insert::insert(&node, 15, &"c", &Config::default(), &g);
        assert_eq!(node.total_keys(&g), 3);

        // Remove from within the child structure
        assert!(remove(&node, &15, &g));
        assert_eq!(crate::lookup::get(&node, &15, &g), None);
    }
}
