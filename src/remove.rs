//! Lock-free remove algorithm with CAS retry loop and tombstone compaction.
//!
//! When a key is removed, the slot state is CAS'd from `DATA` to `TOMBSTONE`.
//! If the tombstone ratio in the node exceeds the configured threshold, a
//! localized rebuild of the parent subtree is triggered to reclaim space.
#![allow(unsafe_code)]

use crossbeam_epoch::{Guard, Shared};

use crate::config::Config;
use crate::key::Key;
use crate::node::{is_child, Node, SLOT_DATA, SLOT_EMPTY, SLOT_TOMBSTONE, SLOT_WRITING};

/// Remove a key from the tree, returning `true` if it was found and removed.
///
/// When `config.auto_rebuild` is enabled and the tombstone ratio in the node
/// exceeds `config.tombstone_ratio_threshold`, triggers a localized subtree
/// rebuild to compact tombstones.
///
/// If a concurrent rebuild replaces a subtree on the remove's descent path,
/// the remove detects the change via a post-CAS validation and retries from
/// the root of `node`.
pub fn remove<K: Key, V: Clone + Send + Sync>(
    node: &Node<K, V>,
    key: &K,
    config: &Config,
    guard: &Guard,
) -> bool {
    // Track whether any retry iteration successfully CAS'd the key to tombstone.
    // If so, the key was genuinely removed, even if a later retry finds it
    // absent (it was excluded from the rebuild snapshot). Persists across retries.
    let mut was_removed = false;

    // Outer retry loop: if a concurrent rebuild orphans the subtree we
    // removed from, we restart the remove from the root.
    'retry: loop {
        let mut current_node = node;
        let mut rebuild_candidate: Option<(&Node<K, V>, usize)> = None;
        // Track the first child descent so we can detect if a concurrent
        // rebuild replaced the subtree after our remove completed.
        #[allow(clippy::type_complexity)]
        let mut descent_snapshot: Option<(
            &Node<K, V>,
            usize,
            Shared<'_, Node<K, V>>,
        )> = None;

        loop {
            let slot_idx = current_node.predict_slot(key);
            let state = current_node.slot_state(slot_idx);

            match state {
                SLOT_EMPTY | SLOT_TOMBSTONE => return was_removed,
                SLOT_WRITING => {
                    std::hint::spin_loop();
                    continue;
                }
                SLOT_DATA => {
                    // SAFETY: state is DATA, so the key slot is initialized.
                    let k = unsafe { current_node.read_key(slot_idx) };
                    if k != key {
                        return was_removed;
                    }
                    // Found: CAS DATA -> TOMBSTONE
                    if current_node.cas_data_to_tombstone(slot_idx) {
                        current_node.dec_keys();
                        current_node.inc_tombstones();

                        // Validate: if we descended into a child, verify the
                        // subtree is still reachable. A concurrent rebuild may
                        // have replaced it, orphaning our remove.
                        if let Some((parent, pidx, expected)) = descent_snapshot {
                            if parent.load_child(pidx, guard) != expected {
                                was_removed = true;
                                continue 'retry;
                            }
                        }

                        // Tombstone compaction: if the ratio exceeds threshold,
                        // trigger a localized subtree rebuild.
                        if config.auto_rebuild
                            && current_node.tombstone_ratio()
                                > config.tombstone_ratio_threshold
                        {
                            if let Some((parent, idx)) = rebuild_candidate {
                                crate::rebuild::try_rebuild_subtree(
                                    parent, idx, config, guard,
                                );
                            }
                        }

                        return true;
                    }
                    // CAS failed -- slot changed, retry inner loop
                }
                s if is_child(s) => {
                    let child_shared = current_node.load_child(slot_idx, guard);
                    // If the slot is tagged, a rebuild is in progress on this
                    // subtree. Spin until the rebuild completes, then re-predict
                    // (the slot now points to the rebuilt child).
                    if child_shared.tag() != 0 {
                        std::hint::spin_loop();
                        continue;
                    }
                    if child_shared.is_null() {
                        return was_removed;
                    }
                    if rebuild_candidate.is_none() {
                        rebuild_candidate = Some((current_node, slot_idx));
                    }
                    // Track first descent for post-remove validation.
                    if descent_snapshot.is_none() {
                        descent_snapshot = Some((current_node, slot_idx, child_shared));
                    }
                    // SAFETY: child_shared is not null and valid for the
                    // lifetime of the guard.
                    current_node = unsafe { child_shared.deref() };
                }
                _ => return was_removed, // unknown state
            }
        }
    } // 'retry
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::insert;
    use crate::node::Node;

    use crossbeam_epoch as epoch;

    fn guard() -> epoch::Guard {
        epoch::pin()
    }

    fn cfg() -> Config {
        Config::default()
    }

    #[test]
    fn remove_existing_key() {
        let g = guard();
        let c = cfg();
        let pairs = vec![(10u64, "a"), (20, "b"), (30, "c")];
        let node = crate::build::bulk_load(&pairs, &c).unwrap();

        let removed = remove(&node, &20, &c, &g);
        assert!(removed);
        assert_eq!(node.total_keys(&g), 2);
        assert_eq!(crate::lookup::get(&node, &20, &g), None);
    }

    #[test]
    fn remove_missing_key() {
        let g = guard();
        let c = cfg();
        let pairs = vec![(10u64, "a"), (20, "b")];
        let node = crate::build::bulk_load(&pairs, &c).unwrap();

        let removed = remove(&node, &99, &c, &g);
        assert!(!removed);
        assert_eq!(node.total_keys(&g), 2);
    }

    #[test]
    fn remove_all_keys() {
        let g = guard();
        let c = cfg();
        let pairs = vec![(1u64, 'a'), (2, 'b'), (3, 'c')];
        let node = crate::build::bulk_load(&pairs, &c).unwrap();

        assert!(remove(&node, &1, &c, &g));
        assert!(remove(&node, &2, &c, &g));
        assert!(remove(&node, &3, &c, &g));
        assert_eq!(node.total_keys(&g), 0);
    }

    #[test]
    fn remove_then_reinsert() {
        let g = guard();
        let c = cfg();
        let pairs = vec![(10u64, "a"), (20, "b")];
        let node = crate::build::bulk_load(&pairs, &c).unwrap();

        remove(&node, &10, &c, &g);
        assert_eq!(crate::lookup::get(&node, &10, &g), None);

        insert::insert(&node, 10, &"A", &c, &g);
        assert_eq!(crate::lookup::get(&node, &10, &g), Some(&"A"));
    }

    #[test]
    fn remove_idempotent() {
        let g = guard();
        let c = cfg();
        let pairs = vec![(5u64, "x")];
        let node = crate::build::bulk_load(&pairs, &c).unwrap();

        assert!(remove(&node, &5, &c, &g));
        assert!(!remove(&node, &5, &c, &g));
    }

    #[test]
    fn remove_from_child_node() {
        let g = guard();
        let c = cfg();
        let pairs: Vec<(u64, &str)> = vec![(10, "a"), (20, "b")];
        let node = crate::build::bulk_load(&pairs, &c).unwrap();

        // Insert a key that may create a child
        insert::insert(&node, 15, &"c", &c, &g);
        assert_eq!(node.total_keys(&g), 3);

        // Remove from within the child structure
        assert!(remove(&node, &15, &c, &g));
        assert_eq!(crate::lookup::get(&node, &15, &g), None);
    }

    #[test]
    fn tombstone_compaction_preserves_remaining_keys() {
        let g = guard();
        let c = Config::new().auto_rebuild(true).tombstone_ratio_threshold(0.3);
        let root = Node::<u64, u64>::with_capacity(
            crate::model::LinearModel::new(0.01, 0.0),
            16,
        );
        // Insert keys that create child subtrees
        for i in 0..40u64 {
            insert::insert(&root, i, &i, &c, &g);
        }
        assert_eq!(root.total_keys(&g), 40);

        // Remove most keys -- should trigger tombstone compaction
        for i in 0..30u64 {
            remove(&root, &i, &c, &g);
        }

        // Remaining keys must still be findable
        for i in 30..40u64 {
            assert_eq!(
                crate::lookup::get(&root, &i, &g),
                Some(&i),
                "key {i} lost after tombstone compaction"
            );
        }
    }
}
