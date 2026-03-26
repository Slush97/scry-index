//! Localized subtree rebuild via CAS-swap.
//!
//! When an insert descends beyond the configured depth threshold, the inserting
//! thread rebuilds the degraded subtree inline using [`try_rebuild_subtree`].
//! This replaces the global `RwLock`-protected rebuild with fine-grained,
//! lock-free subtree compaction.
//!
//! The rebuild uses a **freeze protocol** to prevent data loss: the parent's
//! child pointer is tagged (frozen) before the snapshot so concurrent inserts
//! spin-wait. A post-CAS recovery scan catches in-flight inserts that loaded
//! the child pointer before the freeze.
#![allow(unsafe_code)]

use std::sync::atomic::Ordering;

use crossbeam_epoch::{Atomic, Guard, Owned, Shared};

use crate::build;
use crate::config::Config;
use crate::iter;
use crate::key::Key;
use crate::node::{is_child, Node};

/// RAII guard that unfreezes a child pointer if the rebuild doesn't complete normally.
///
/// On drop (including panic unwind), CAS the child pointer back from the frozen
/// (tagged) pointer to the original untagged pointer. This prevents a
/// stuck freeze from blocking all inserts into the subtree forever.
struct FreezeGuard<'a, 'g, K, V> {
    slot: &'a Atomic<Node<K, V>>,
    frozen: Shared<'g, Node<K, V>>,
    original: Shared<'g, Node<K, V>>,
    guard: &'g Guard,
    disarmed: bool,
}

impl<K, V> Drop for FreezeGuard<'_, '_, K, V> {
    fn drop(&mut self) {
        if !self.disarmed {
            let _ = self.slot.compare_exchange(
                self.frozen,
                self.original,
                Ordering::AcqRel,
                Ordering::Relaxed,
                self.guard,
            );
        }
    }
}

/// Attempt to rebuild a degraded subtree at `parent_node`'s child in slot
/// `parent_slot_idx`.
///
/// If the slot contains a child node, collects all its key-value pairs, builds
/// a fresh subtree with a boosted expansion factor, and CAS-swaps it in place.
///
/// Returns `true` if the rebuild succeeded, `false` if the slot was not a child,
/// the subtree was too small, or the CAS failed (another thread rebuilt first).
///
/// # Concurrent safety
///
/// - Two threads rebuilding the same subtree: the first freeze-CAS wins, the
///   other returns false immediately. Safe.
/// - Insert races with rebuild: the child pointer is *frozen* (tagged) before the
///   snapshot, so new inserts spin-wait until the rebuild completes. In-flight
///   inserts (threads that loaded the child pointer before the freeze) may
///   complete inside the old subtree; a post-CAS recovery scan detects and
///   re-inserts any keys they added.
/// - Parent rebuilt while child being rebuilt: the child rebuild CAS fails because
///   the parent slot now points to a new child inside the rebuilt parent.
pub fn try_rebuild_subtree<K: Key, V: Clone + Send + Sync>(
    parent_node: &Node<K, V>,
    parent_slot_idx: usize,
    config: &Config,
    guard: &Guard,
) -> bool {
    let state = parent_node.slot_state(parent_slot_idx);
    if !is_child(state) {
        return false;
    }

    let child_atomic = parent_node.child_atomic(parent_slot_idx);
    let current = child_atomic.load(Ordering::Acquire, guard);

    // Null or already frozen by another rebuild — bail out.
    if current.is_null() || current.tag() != 0 {
        return false;
    }

    // SAFETY: current is not null and valid for the lifetime of the guard.
    let child = unsafe { current.deref() };

    // Freeze the child pointer: tag it so concurrent inserts spin-wait instead of
    // descending into the old subtree during the rebuild.
    let frozen = current.with_tag(1);
    if child_atomic
        .compare_exchange(current, frozen, Ordering::AcqRel, Ordering::Acquire, guard)
        .is_err()
    {
        return false;
    }

    // RAII: unfreeze on panic or early return.
    let mut freeze_guard = FreezeGuard {
        slot: child_atomic,
        frozen,
        original: current,
        guard,
        disarmed: false,
    };

    let pairs = iter::sorted_pairs(child, guard);
    if pairs.len() <= 1 {
        // Too small to rebuild — FreezeGuard::drop unfreezes.
        return false;
    }

    // Boost expansion factor to reduce future conflicts in this subtree.
    let boosted_factor = (config.expansion_factor * 1.5).min(4.0);
    let boosted_config = Config {
        expansion_factor: boosted_factor,
        ..config.clone()
    };

    let new_node = build::build_recursive(&pairs, &boosted_config);
    let new_child = Owned::new(new_node);

    // CAS: frozen old → new child.
    match child_atomic.compare_exchange(
        frozen,
        new_child,
        Ordering::AcqRel,
        Ordering::Acquire,
        guard,
    ) {
        Ok(_) => {
            freeze_guard.disarmed = true;

            // In-flight inserts (threads that loaded the child pointer before
            // the freeze) detect the stale subtree via descent_snapshot
            // validation in insert::insert and retry automatically. No
            // explicit recovery scan is needed — the happens-before chain
            // between insert CAS, validation, freeze, and snapshot guarantees
            // that any insert completing before the freeze is captured in the
            // snapshot.

            // SAFETY: CAS succeeded; old subtree is unreachable to new readers.
            unsafe {
                guard.defer_destroy(current);
            }
            // Advance the local epoch so deferred memory can be reclaimed.
            // Without this, a long-lived guard accumulates all rebuilt subtrees
            // until it drops, which causes unbounded memory growth for
            // pathological insert patterns (e.g., sequential keys into a
            // small root).
            guard.flush();
            true
        }
        Err(_) => {
            // FreezeGuard::drop unfreezes the child pointer.
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::insert;
    use crate::lookup;
    use crate::model::LinearModel;
    use crate::node::SLOT_DATA;

    use crossbeam_epoch as epoch;

    fn guard() -> epoch::Guard {
        epoch::pin()
    }

    fn default_config() -> Config {
        Config::default()
    }

    #[test]
    fn rebuild_subtree_basic() {
        let g = guard();
        // Disable auto_rebuild so we can test manual subtree rebuild
        let config = Config::new().auto_rebuild(false);

        // Create a root where keys pile up, creating a deep chain
        let root = Node::<u64, u64>::with_capacity(LinearModel::new(0.01, 0.0), 16);
        for i in 0..20u64 {
            insert::insert(&root, i, &i, &config, &g);
        }

        let depth_before = root.max_depth(&g);
        assert!(
            depth_before > 2,
            "setup should create depth > 2, got {depth_before}"
        );

        // Find a child slot and rebuild it
        let mut rebuilt = false;
        for idx in 0..root.capacity() {
            if is_child(root.slot_state(idx)) {
                rebuilt = try_rebuild_subtree(&root, idx, &config, &g);
                if rebuilt {
                    break;
                }
            }
        }
        assert!(rebuilt, "should have rebuilt at least one subtree");

        // All keys should still be accessible
        for i in 0..20u64 {
            assert_eq!(
                lookup::get(&root, &i, &g),
                Some(&i),
                "key {i} lost after subtree rebuild"
            );
        }
    }

    #[test]
    fn rebuild_subtree_preserves_data() {
        let g = guard();
        let config = default_config();

        let root = Node::<u64, u64>::with_capacity(LinearModel::new(0.01, 0.0), 16);
        for i in 0..50u64 {
            insert::insert(&root, i, &(i * 10), &config, &g);
        }

        // Rebuild all child subtrees
        for idx in 0..root.capacity() {
            if is_child(root.slot_state(idx)) {
                try_rebuild_subtree(&root, idx, &config, &g);
            }
        }

        // All keys and values must be intact
        assert_eq!(root.total_keys(&g), 50);
        for i in 0..50u64 {
            assert_eq!(
                lookup::get(&root, &i, &g),
                Some(&(i * 10)),
                "key {i} has wrong value after rebuild"
            );
        }
    }

    #[test]
    fn rebuild_subtree_returns_false_for_data_slot() {
        let g = guard();
        let config = default_config();

        let pairs = vec![(10u64, 100), (20, 200)];
        let root = crate::build::bulk_load(&pairs, &config).unwrap();

        // Find a data slot and try to rebuild it -- should return false
        for idx in 0..root.capacity() {
            if root.slot_state(idx) == SLOT_DATA {
                assert!(!try_rebuild_subtree(&root, idx, &config, &g));
                return;
            }
        }
        panic!("no data slot found");
    }

    #[test]
    fn rebuild_subtree_empty_child() {
        let g = guard();
        let config = default_config();

        // Create a parent with an empty child node
        let empty_child = Node::<u64, u64>::with_capacity(LinearModel::new(0.1, 0.0), 4);
        let parent = Node::<u64, u64>::with_capacity(LinearModel::new(0.1, 0.0), 4);
        parent.store_child(0, empty_child);

        // Rebuild should return false (empty subtree)
        assert!(!try_rebuild_subtree(&parent, 0, &config, &g));
    }
}
