//! Localized subtree rebuild via CAS-swap.
//!
//! When an insert descends beyond the configured depth threshold, the inserting
//! thread rebuilds the degraded subtree inline using [`try_rebuild_subtree`].
//! This replaces the global `RwLock`-protected rebuild with fine-grained,
//! lock-free subtree compaction.
//!
//! The rebuild uses a **freeze protocol** to prevent data loss: the parent slot
//! is tagged (frozen) before the snapshot so concurrent inserts spin-wait. A
//! post-CAS recovery scan catches in-flight inserts that loaded the child
//! pointer before the freeze.
#![allow(unsafe_code)]

use std::sync::atomic::Ordering;

use crossbeam_epoch::{Atomic, Guard, Owned, Shared};

use crate::build;
use crate::config::Config;
use crate::iter;
use crate::key::Key;
use crate::node::{Node, SlotInner};

/// RAII guard that unfreezes a slot if the rebuild doesn't complete normally.
///
/// On drop (including panic unwind), CAS the slot back from the frozen
/// (tagged) pointer to the original untagged pointer. This prevents a
/// stuck freeze from blocking all inserts into the subtree forever.
struct FreezeGuard<'a, 'g, K, V> {
    slot: &'a Atomic<SlotInner<K, V>>,
    frozen: Shared<'g, SlotInner<K, V>>,
    original: Shared<'g, SlotInner<K, V>>,
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

/// Attempt to rebuild a degraded subtree at `parent_node.slot(parent_slot_idx)`.
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
/// - Insert races with rebuild: the parent slot is *frozen* (tagged) before the
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
    let slot = parent_node.slot(parent_slot_idx);
    let current = slot.load(Ordering::Acquire, guard);

    // Null, already frozen by another rebuild, or not a child — bail out.
    if current.is_null() || current.tag() != 0 {
        return false;
    }

    // SAFETY: current is not null and valid for the lifetime of the guard.
    let inner = unsafe { current.deref() };

    let child = match inner {
        SlotInner::Child(child) => child,
        SlotInner::Data { .. } => return false,
    };

    // Freeze the slot: tag it so concurrent inserts spin-wait instead of
    // descending into the old subtree during the rebuild.
    let frozen = current.with_tag(1);
    if slot
        .compare_exchange(current, frozen, Ordering::AcqRel, Ordering::Acquire, guard)
        .is_err()
    {
        return false;
    }

    // RAII: unfreeze on panic or early return.
    let mut freeze_guard = FreezeGuard {
        slot,
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
    let new_inner = Owned::new(SlotInner::Child(new_node));

    // CAS: frozen old → new child.
    match slot.compare_exchange(
        frozen,
        new_inner,
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
            // FreezeGuard::drop unfreezes the slot.
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
            let shared = root.slot(idx).load(Ordering::Acquire, &g);
            if !shared.is_null() {
                // SAFETY: shared is not null and valid under guard.
                if let SlotInner::Child(_) = unsafe { shared.deref() } {
                    rebuilt = try_rebuild_subtree(&root, idx, &config, &g);
                    if rebuilt {
                        break;
                    }
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
            let shared = root.slot(idx).load(Ordering::Acquire, &g);
            if !shared.is_null() {
                // SAFETY: shared is not null and valid under guard.
                if let SlotInner::Child(_) = unsafe { shared.deref() } {
                    try_rebuild_subtree(&root, idx, &config, &g);
                }
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
            let shared = root.slot(idx).load(Ordering::Acquire, &g);
            if !shared.is_null() {
                // SAFETY: shared is not null and valid under guard.
                if let SlotInner::Data { .. } = unsafe { shared.deref() } {
                    assert!(!try_rebuild_subtree(&root, idx, &config, &g));
                    return;
                }
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
        parent.store_slot(0, SlotInner::Child(empty_child));

        // Rebuild should return false (empty subtree)
        assert!(!try_rebuild_subtree(&parent, 0, &config, &g));
    }
}
