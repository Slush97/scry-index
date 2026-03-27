//! Lock-free insert algorithm using LIPP's chain method with CAS retry loops.
//!
//! When inserting a key into a node:
//! - If the predicted slot is empty, CAS `EMPTY` → `WRITING` → `DATA` (inline).
//! - If the slot contains the same key (`DATA`), build a 1-entry child with the
//!   new value and CAS `DATA` → `CHILD_STALE`. The old inline data stays stale.
//! - If the slot contains a different key (`DATA`), clone the existing key+value,
//!   build a 2-entry conflict child, and CAS `DATA` → `CHILD_STALE`.
//! - If the slot is `CHILD` or `CHILD_STALE`, follow the child pointer.
//! - If the slot is `TOMBSTONE`, CAS `TOMBSTONE` → `WRITING` → `DATA` to reuse it.
//! - If the slot is `WRITING`, spin (another thread is claiming it).
//! - On CAS failure, retry from the slot state read.
#![allow(unsafe_code)]

use crossbeam_epoch::{self as epoch, Guard, Shared};

use crate::config::Config;
use crate::key::Key;
use crate::model::LinearModel;
use crate::node::{is_child, Node, SLOT_DATA, SLOT_TOMBSTONE, SLOT_WRITING};

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
///
/// When `config.auto_rebuild` is enabled, tracks descent depth and triggers
/// a localized subtree rebuild if the depth exceeds
/// `config.rebuild_depth_threshold`.
///
/// If a concurrent rebuild replaces a subtree on the insert's descent path,
/// the insert detects the change via a post-CAS validation and retries from
/// the root of `node`.
#[allow(clippy::too_many_lines, clippy::needless_pass_by_value)]
pub fn insert<K: Key, V: Clone + Send + Sync>(
    node: &Node<K, V>,
    key: K,
    value: &V,
    config: &Config,
    guard: &Guard,
) -> InsertResult {
    // Track whether any retry iteration produced an Inserted result. If so,
    // the key was genuinely new, even if a later retry finds it already
    // present (placed by the rebuild snapshot). Persists across retries.
    let mut retry_was_new = false;

    // Outer retry loop: if a concurrent rebuild orphans the subtree we
    // inserted into, we restart the insert from the root.
    'retry: loop {
        let mut current_node = node;
        let mut depth: usize = 0;
        let mut rebuild_candidate: Option<(&Node<K, V>, usize)> = None;
        // Track the first child descent so we can detect if a concurrent
        // rebuild replaced the subtree after our insert completed.
        // Tuple: (parent_node, slot_idx, child_shared).
        #[allow(clippy::type_complexity)]
        let mut descent_snapshot: Option<(&Node<K, V>, usize, Shared<'_, Node<K, V>>)> = None;

        loop {
            let slot_idx = current_node.predict_slot(&key);
            let state = current_node.slot_state(slot_idx);

            if state == SLOT_WRITING {
                // Another thread is claiming this slot. Spin and re-read.
                std::hint::spin_loop();
                continue;
            }

            if state == crate::node::SLOT_EMPTY {
                // Empty slot: CAS EMPTY → WRITING → DATA (inline).
                if current_node.cas_empty_to_data(slot_idx, key.clone(), value.clone()) {
                    current_node.inc_keys();
                    // Validate: if we descended into a child, verify the subtree
                    // is still reachable. A concurrent rebuild may have replaced
                    // it, orphaning our insert.
                    if let Some((parent, pidx, expected)) = descent_snapshot {
                        if parent.load_child(pidx, guard) != expected {
                            retry_was_new = true;
                            continue 'retry;
                        }
                    }
                    // Only rebuild if depth exceeded the threshold — the subtree
                    // actually degraded beyond the capture point. Rebuilding when
                    // depth == threshold wastes work on already-compact subtrees.
                    if config.auto_rebuild && depth > config.rebuild_depth_threshold {
                        if let Some((parent, idx)) = rebuild_candidate {
                            crate::rebuild::try_rebuild_subtree(parent, idx, config, guard);
                        }
                    }
                    return InsertResult::Inserted;
                }
                // CAS failed — slot changed, retry from state read.
                continue;
            }

            if state == SLOT_TOMBSTONE {
                // Reuse a tombstone slot.
                if current_node.cas_tombstone_to_data(slot_idx, key.clone(), value.clone()) {
                    current_node.inc_keys();
                    if let Some((parent, pidx, expected)) = descent_snapshot {
                        if parent.load_child(pidx, guard) != expected {
                            retry_was_new = true;
                            continue 'retry;
                        }
                    }
                    if config.auto_rebuild && depth > config.rebuild_depth_threshold {
                        if let Some((parent, idx)) = rebuild_candidate {
                            crate::rebuild::try_rebuild_subtree(parent, idx, config, guard);
                        }
                    }
                    return InsertResult::Inserted;
                }
                // CAS failed — slot changed, retry.
                continue;
            }

            if state == SLOT_DATA {
                // SAFETY: state is DATA so inline key is valid and immutable.
                let existing_key = unsafe { current_node.read_key(slot_idx) };

                if existing_key == &key {
                    // Same key: build a single-entry child with the new value,
                    // then CAS DATA → CHILD_STALE. The old inline data stays stale.
                    let child = build_single_entry_child(&key, value.clone());
                    if current_node.cas_data_to_child_stale(slot_idx, child, guard) {
                        if let Some((parent, pidx, expected)) = descent_snapshot {
                            if parent.load_child(pidx, guard) != expected {
                                continue 'retry;
                            }
                        }
                        return if retry_was_new {
                            InsertResult::Inserted
                        } else {
                            InsertResult::Updated
                        };
                    }
                    // CAS failed — slot changed, retry.
                    continue;
                }

                // Collision: clone the existing key+value and build a 2-entry
                // conflict child node.
                let ek = existing_key.clone();
                // SAFETY: state is DATA so inline value is valid and immutable.
                let ev = unsafe { current_node.read_value(slot_idx) }.clone();
                let child = build_conflict_node(ek, ev, key.clone(), value.clone(), config);

                if current_node.cas_data_to_child_stale(slot_idx, child, guard) {
                    current_node.dec_keys(); // existing data moved to child
                    if let Some((parent, pidx, expected)) = descent_snapshot {
                        if parent.load_child(pidx, guard) != expected {
                            retry_was_new = true;
                            continue 'retry;
                        }
                    }
                    if config.auto_rebuild && depth > config.rebuild_depth_threshold {
                        if let Some((parent, idx)) = rebuild_candidate {
                            crate::rebuild::try_rebuild_subtree(parent, idx, config, guard);
                        }
                    }
                    return InsertResult::Inserted;
                }
                // CAS failed — slot changed, retry.
                continue;
            }

            if is_child(state) {
                // CHILD or CHILD_STALE: follow the child pointer.
                let child_shared = current_node.load_child(slot_idx, guard);

                // If the child pointer is tagged, a rebuild is in progress on this
                // subtree. Spin until the rebuild completes, then re-predict
                // (the slot now points to the rebuilt child).
                if child_shared.tag() != 0 {
                    std::hint::spin_loop();
                    continue;
                }

                if child_shared.is_null() {
                    // Should not happen for a CHILD/CHILD_STALE slot, but be safe.
                    continue;
                }

                depth += 1;
                // Capture the shallowest child on the descent path. If depth
                // eventually exceeds the threshold, we rebuild from this point,
                // flattening the entire degraded chain — not just a small
                // subtree deep in the tree.
                if rebuild_candidate.is_none() {
                    rebuild_candidate = Some((current_node, slot_idx));
                }
                // Track first descent for post-insert validation.
                if descent_snapshot.is_none() {
                    descent_snapshot = Some((current_node, slot_idx, child_shared));
                }
                // SAFETY: child_shared is not null and valid for the guard lifetime.
                current_node = unsafe { child_shared.deref() };
                continue;
            }

            // Unknown state — spin and re-read (defensive).
            std::hint::spin_loop();
        }
    } // 'retry
}

/// Atomically get an existing value or insert a new one.
///
/// If the key already exists, returns a reference to the existing value and
/// [`InsertResult::Updated`] (signalling "already existed — no insert performed").
/// If the key is absent, inserts the key-value pair and returns a reference to
/// the new value with [`InsertResult::Inserted`].
///
/// Unlike [`insert`], this function never overwrites an existing value.
///
/// The returned reference is valid for the lifetime of the `guard`.
#[allow(clippy::too_many_lines, clippy::needless_pass_by_value)]
pub fn get_or_insert<'g, K: Key, V: Clone + Send + Sync>(
    node: &'g Node<K, V>,
    key: K,
    value: &V,
    config: &Config,
    guard: &'g Guard,
) -> (&'g V, InsertResult) {
    let mut retry_was_new = false;

    'retry: loop {
        let mut current_node: &'g Node<K, V> = node;
        let mut depth: usize = 0;
        let mut rebuild_candidate: Option<(&'g Node<K, V>, usize)> = None;
        #[allow(clippy::type_complexity)]
        let mut descent_snapshot: Option<(&'g Node<K, V>, usize, Shared<'g, Node<K, V>>)> = None;

        loop {
            let slot_idx = current_node.predict_slot(&key);
            let state = current_node.slot_state(slot_idx);

            if state == SLOT_WRITING {
                std::hint::spin_loop();
                continue;
            }

            if state == crate::node::SLOT_EMPTY {
                // Empty slot: CAS EMPTY → WRITING → DATA (inline).
                if current_node.cas_empty_to_data(slot_idx, key.clone(), value.clone()) {
                    current_node.inc_keys();
                    if let Some((parent, pidx, expected)) = descent_snapshot {
                        if parent.load_child(pidx, guard) != expected {
                            retry_was_new = true;
                            continue 'retry;
                        }
                    }
                    if config.auto_rebuild && depth > config.rebuild_depth_threshold {
                        if let Some((parent, idx)) = rebuild_candidate {
                            crate::rebuild::try_rebuild_subtree(parent, idx, config, guard);
                        }
                    }
                    // SAFETY: We just wrote the value via cas_empty_to_data and
                    // the state is now DATA. The inline value is valid for 'g.
                    let val = unsafe { current_node.read_value(slot_idx) };
                    return (val, InsertResult::Inserted);
                }
                // CAS failed — slot changed, retry.
                continue;
            }

            if state == SLOT_TOMBSTONE {
                // Reuse a tombstone slot.
                if current_node.cas_tombstone_to_data(slot_idx, key.clone(), value.clone()) {
                    current_node.inc_keys();
                    if let Some((parent, pidx, expected)) = descent_snapshot {
                        if parent.load_child(pidx, guard) != expected {
                            retry_was_new = true;
                            continue 'retry;
                        }
                    }
                    if config.auto_rebuild && depth > config.rebuild_depth_threshold {
                        if let Some((parent, idx)) = rebuild_candidate {
                            crate::rebuild::try_rebuild_subtree(parent, idx, config, guard);
                        }
                    }
                    // SAFETY: We just wrote the value. State is DATA.
                    let val = unsafe { current_node.read_value(slot_idx) };
                    return (val, InsertResult::Inserted);
                }
                // CAS failed — slot changed, retry.
                continue;
            }

            if state == SLOT_DATA {
                // SAFETY: state is DATA so inline key+value are valid.
                let existing_key = unsafe { current_node.read_key(slot_idx) };

                if existing_key == &key {
                    // Same key: return existing value, don't update.
                    // SAFETY: state is DATA so inline value is valid and immutable.
                    let existing_value = unsafe { current_node.read_value(slot_idx) };
                    let result = if retry_was_new {
                        InsertResult::Inserted
                    } else {
                        InsertResult::Updated
                    };
                    return (existing_value, result);
                }

                // Collision: clone the existing key+value, build a 2-entry
                // conflict child.
                let ek = existing_key.clone();
                let ev = unsafe { current_node.read_value(slot_idx) }.clone();
                let child = build_conflict_node(ek, ev, key.clone(), value.clone(), config);

                if current_node.cas_data_to_child_stale(slot_idx, child, guard) {
                    current_node.dec_keys(); // existing data moved to child
                    if let Some((parent, pidx, expected)) = descent_snapshot {
                        if parent.load_child(pidx, guard) != expected {
                            retry_was_new = true;
                            continue 'retry;
                        }
                    }
                    if config.auto_rebuild && depth > config.rebuild_depth_threshold {
                        if let Some((parent, idx)) = rebuild_candidate {
                            crate::rebuild::try_rebuild_subtree(parent, idx, config, guard);
                        }
                    }
                    // Look up the value we just inserted in the new child.
                    let child_shared = current_node.load_child(slot_idx, guard);
                    if !child_shared.is_null() {
                        // SAFETY: child_shared is the child we just stored.
                        let child_node = unsafe { child_shared.deref() };
                        if let Some(val) = crate::lookup::get(child_node, &key, guard) {
                            return (val, InsertResult::Inserted);
                        }
                    }
                    // Should not happen by construction; loop retries.
                }
                // CAS failed — slot changed, retry.
                continue;
            }

            if is_child(state) {
                let child_shared = current_node.load_child(slot_idx, guard);

                if child_shared.tag() != 0 {
                    std::hint::spin_loop();
                    continue;
                }

                if child_shared.is_null() {
                    continue;
                }

                depth += 1;
                if rebuild_candidate.is_none() {
                    rebuild_candidate = Some((current_node, slot_idx));
                }
                if descent_snapshot.is_none() {
                    descent_snapshot = Some((current_node, slot_idx, child_shared));
                }
                // SAFETY: child_shared is not null and valid for guard lifetime.
                current_node = unsafe { child_shared.deref() };
                continue;
            }

            // Unknown state — spin and re-read (defensive).
            std::hint::spin_loop();
        }
    } // 'retry
}

/// Build a single-entry child node containing one key-value pair.
///
/// Used when updating an existing key: the new value goes into a child node
/// so the parent slot can be CAS'd from `DATA` → `CHILD_STALE`.
fn build_single_entry_child<K: Key, V: Clone + Send + Sync>(key: &K, value: V) -> Node<K, V> {
    let f = key.to_model_input();
    let node = Node::with_capacity(LinearModel::new(1.0, -f), 2);
    let slot = node.predict_slot(key);
    node.store_data(slot, key.clone(), value);
    node.inc_keys();
    node
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
    config: &Config,
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

    let node = if key_range.abs() < f64::EPSILON {
        // Keys have the same f64 representation (precision loss for large
        // integers). Use exact ordinal comparison to separate them.
        let lo_ord = lo_k.to_exact_ordinal();
        let hi_ord = hi_k.to_exact_ordinal();
        if lo_ord == hi_ord {
            // Ordinals also collide (e.g., variable-length keys sharing a
            // 16-byte prefix). Fall back to Ord-based split.
            Node::with_split_key(lo_k.clone(), array_size)
        } else {
            let midpoint = lo_ord + (hi_ord - lo_ord) / 2;
            Node::with_capacity(LinearModel::binary_split(midpoint), array_size)
        }
    } else {
        let s = (array_size - 1) as f64 / key_range;
        Node::with_capacity(LinearModel::new(s, -s * lo_f), array_size)
    };

    let s1 = node.predict_slot(&lo_k);
    let s2 = node.predict_slot(&hi_k);

    node.store_data(s1, lo_k, lo_v);
    node.inc_keys();

    if s1 == s2 {
        // Still collide — insert the second key via the concurrent insert path
        // (with unprotected guard since this is single-threaded construction).
        // SAFETY: Exclusive access during construction. The unprotected guard
        // is safe because no concurrent readers exist for this node yet.
        // Depth is 1 at most, so the rebuild threshold never triggers.
        unsafe {
            let guard = epoch::unprotected();
            insert(&node, hi_k, &hi_v, config, guard);
        }
    } else {
        node.store_data(s2, hi_k, hi_v);
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

    fn cfg() -> Config {
        Config::default()
    }

    fn empty_root() -> Node<u64, u64> {
        let model = LinearModel::new(0.1, 0.0);
        Node::with_capacity(model, 100)
    }

    #[test]
    fn insert_into_empty_slot() {
        let g = guard();
        let node = empty_root();
        let result = insert(&node, 50, &500, &cfg(), &g);
        assert_eq!(result, InsertResult::Inserted);
        assert_eq!(node.total_keys(&g), 1);
    }

    #[test]
    fn insert_duplicate_returns_updated() {
        let g = guard();
        let node = empty_root();
        insert(&node, 50, &500, &cfg(), &g);
        let result = insert(&node, 50, &5000, &cfg(), &g);
        assert_eq!(result, InsertResult::Updated);
        assert_eq!(node.total_keys(&g), 1);
    }

    #[test]
    fn insert_conflict_creates_child() {
        let g = guard();
        let pairs: Vec<(u64, &str)> = vec![(10, "a"), (20, "b")];
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        let initial_keys = node.total_keys(&g);

        insert(&node, 15, &"c", &cfg(), &g);
        assert_eq!(node.total_keys(&g), initial_keys + 1);

        assert_eq!(crate::lookup::get(&node, &10, &g), Some(&"a"));
        assert_eq!(crate::lookup::get(&node, &15, &g), Some(&"c"));
        assert_eq!(crate::lookup::get(&node, &20, &g), Some(&"b"));
    }

    #[test]
    fn insert_many_sequential() {
        let g = guard();
        let c = cfg();
        let node = empty_root();
        for i in 0..100u64 {
            insert(&node, i, &i, &c, &g);
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
        let c = cfg();
        let node = empty_root();
        for i in (0..50u64).rev() {
            insert(&node, i, &(i * 10), &c, &g);
        }
        assert_eq!(node.total_keys(&g), 50);
        for i in 0..50u64 {
            assert_eq!(crate::lookup::get(&node, &i, &g), Some(&(i * 10)));
        }
    }

    #[test]
    fn insert_update_preserves_count() {
        let g = guard();
        let c = cfg();
        let node = empty_root();
        insert(&node, 1, &10, &c, &g);
        insert(&node, 2, &20, &c, &g);
        insert(&node, 3, &30, &c, &g);
        assert_eq!(node.total_keys(&g), 3);

        insert(&node, 2, &200, &c, &g);
        assert_eq!(node.total_keys(&g), 3);
        assert_eq!(crate::lookup::get(&node, &2, &g), Some(&200));
    }

    #[test]
    fn insert_into_bulk_loaded_tree() {
        let g = guard();
        let c = cfg();
        let pairs: Vec<(u64, u64)> = (0..100).map(|i| (i * 2, i)).collect();
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();

        for i in 0..100u64 {
            insert(&node, i * 2 + 1, &(i + 1000), &c, &g);
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
        let node = build_conflict_node(10u64, "a", 20u64, "b", &cfg());
        assert_eq!(node.capacity(), 4);
        assert_eq!(node.total_keys(&g), 2);
    }

    #[test]
    fn conflict_node_both_findable() {
        let g = guard();
        let node = build_conflict_node(100u64, 1, 200u64, 2, &cfg());
        assert_eq!(crate::lookup::get(&node, &100, &g), Some(&1));
        assert_eq!(crate::lookup::get(&node, &200, &g), Some(&2));
    }

    #[test]
    fn insert_update_returns_correct_result() {
        let g = guard();
        let c = cfg();
        let node = empty_root();
        assert_eq!(insert(&node, 1, &10, &c, &g), InsertResult::Inserted);
        assert_eq!(insert(&node, 1, &20, &c, &g), InsertResult::Updated);
        assert_eq!(insert(&node, 2, &30, &c, &g), InsertResult::Inserted);
    }

    #[test]
    fn insert_value_is_updated() {
        let g = guard();
        let c = cfg();
        let node = empty_root();
        insert(&node, 42, &100, &c, &g);
        assert_eq!(crate::lookup::get(&node, &42, &g), Some(&100));
        insert(&node, 42, &999, &c, &g);
        assert_eq!(crate::lookup::get(&node, &42, &g), Some(&999));
    }

    #[test]
    fn conflict_node_same_f64_keys() {
        let g = guard();
        let base: u64 = 1_700_000_000_000_000_000;
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(base as f64, (base + 1) as f64, "precondition: same f64");
        }
        let node = build_conflict_node(base, "a", base + 1, "b", &cfg());
        assert_eq!(node.total_keys(&g), 2);
        assert_eq!(crate::lookup::get(&node, &base, &g), Some(&"a"));
        assert_eq!(crate::lookup::get(&node, &(base + 1), &g), Some(&"b"));
    }

    #[test]
    fn conflict_node_many_same_f64_keys() {
        let g = guard();
        let c = cfg();
        let base: u64 = 1_700_000_000_000_000_000;
        let node = empty_root();
        for i in 0..20u64 {
            insert(&node, base + i, &(base + i), &c, &g);
        }
        assert_eq!(node.total_keys(&g), 20);
        for i in 0..20u64 {
            assert_eq!(
                crate::lookup::get(&node, &(base + i), &g),
                Some(&(base + i)),
                "key base+{i} not found"
            );
        }
    }

    #[test]
    fn get_or_insert_empty_slot() {
        let g = guard();
        let node = empty_root();
        let (val, result) = get_or_insert(&node, 50, &500, &cfg(), &g);
        assert_eq!(result, InsertResult::Inserted);
        assert_eq!(*val, 500);
        assert_eq!(node.total_keys(&g), 1);
    }

    #[test]
    fn get_or_insert_existing_no_update() {
        let g = guard();
        let c = cfg();
        let node = empty_root();
        insert(&node, 50, &500, &c, &g);
        let (val, result) = get_or_insert(&node, 50, &999, &c, &g);
        assert_eq!(result, InsertResult::Updated);
        assert_eq!(*val, 500); // original value, not 999
        assert_eq!(node.total_keys(&g), 1);
    }

    #[test]
    fn get_or_insert_conflict_creates_child() {
        let g = guard();
        let c = cfg();
        let pairs: Vec<(u64, &str)> = vec![(10, "a"), (20, "b")];
        let node = crate::build::bulk_load(&pairs, &Config::default()).unwrap();
        let initial_keys = node.total_keys(&g);

        let (val, result) = get_or_insert(&node, 15, &"c", &c, &g);
        assert_eq!(result, InsertResult::Inserted);
        assert_eq!(*val, "c");
        assert_eq!(node.total_keys(&g), initial_keys + 1);

        // All keys still findable
        assert_eq!(crate::lookup::get(&node, &10, &g), Some(&"a"));
        assert_eq!(crate::lookup::get(&node, &15, &g), Some(&"c"));
        assert_eq!(crate::lookup::get(&node, &20, &g), Some(&"b"));
    }

    #[test]
    fn get_or_insert_idempotent() {
        let g = guard();
        let c = cfg();
        let node = empty_root();
        let (v1, r1) = get_or_insert(&node, 42, &100, &c, &g);
        let (v2, r2) = get_or_insert(&node, 42, &999, &c, &g);
        assert_eq!(r1, InsertResult::Inserted);
        assert_eq!(r2, InsertResult::Updated);
        assert_eq!(*v1, 100);
        assert_eq!(*v2, 100); // same value as first call
        assert_eq!(node.total_keys(&g), 1);
    }

    #[test]
    fn get_or_insert_many_sequential() {
        let g = guard();
        let c = cfg();
        let node = empty_root();
        for i in 0..100u64 {
            let (val, result) = get_or_insert(&node, i, &i, &c, &g);
            assert_eq!(result, InsertResult::Inserted);
            assert_eq!(*val, i);
        }
        assert_eq!(node.total_keys(&g), 100);
        for i in 0..100u64 {
            let (val, result) = get_or_insert(&node, i, &(i + 1000), &c, &g);
            assert_eq!(result, InsertResult::Updated);
            assert_eq!(*val, i); // original value
        }
    }

    #[test]
    fn insert_triggers_localized_rebuild() {
        let g = guard();
        let config = Config::new().rebuild_depth_threshold(4);
        let node = empty_root();

        // Insert keys that pile up in the 100-slot node, creating deep chains
        for i in 0..50u64 {
            insert(&node, i, &(i * 10), &config, &g);
        }

        // With threshold 4, localized rebuilds should keep depth bounded
        let depth = node.max_depth(&g);
        assert!(
            depth <= 10,
            "depth {depth} too high with rebuild_depth_threshold=4"
        );

        // All keys should still be present
        for i in 0..50u64 {
            assert_eq!(
                crate::lookup::get(&node, &i, &g),
                Some(&(i * 10)),
                "key {i} missing after localized rebuild"
            );
        }
    }
}
