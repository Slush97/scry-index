//! Bulk-load construction for building a learned index from sorted data.
#![allow(unsafe_code)]

use crate::config::Config;
use crate::error::{Error, Result};
use crate::key::Key;
use crate::model::{fit_fmcd, LinearModel};
use crate::node::Node;

/// Build a learned index tree from sorted key-value pairs.
///
/// The input must be sorted by key in ascending order with no duplicates.
///
/// # Errors
///
/// Returns [`Error::EmptyData`] if the input is empty.
/// Returns [`Error::NotSorted`] if the input is not sorted.
pub fn bulk_load<K: Key, V: Clone>(pairs: &[(K, V)], config: &Config) -> Result<Node<K, V>> {
    if pairs.is_empty() {
        return Err(Error::EmptyData);
    }

    // Verify sorted order
    for window in pairs.windows(2) {
        if window[0].0 >= window[1].0 {
            return Err(Error::NotSorted);
        }
    }

    Ok(build_recursive(pairs, config))
}

/// Recursively build a subtree from a sorted slice of key-value pairs.
pub(crate) fn build_recursive<K: Key, V: Clone>(pairs: &[(K, V)], config: &Config) -> Node<K, V> {
    let n = pairs.len();

    // Check for degenerate case: all keys share the same f64 model input.
    if n > 1 {
        let first_f = pairs[0].0.to_model_input();
        let last_f = pairs[n - 1].0.to_model_input();
        if (last_f - first_f).abs() < f64::EPSILON {
            return build_degenerate(pairs);
        }
    }

    // Pass pairs directly — fit_fmcd extracts keys via closure, avoiding a
    // separate Vec<K> allocation.
    let result = fit_fmcd(
        n,
        |i| pairs[i].0.to_model_input(),
        config.expansion_factor,
        config.range_headroom,
    );

    if result.conflicts == 0 {
        // No conflicts — use a leaf node (no children array allocation).
        let node = Node::with_capacity_leaf(result.model, result.array_size);
        for (key, value) in pairs {
            let slot = node.predict_slot(key);
            node.store_data(slot, key.clone(), value.clone());
            node.inc_keys();
        }
        return node;
    }

    // Conflicts exist — sort by predicted slot, then process runs.
    let node = Node::with_capacity(result.model, result.array_size);

    let mut assignments: Vec<(usize, usize)> = pairs
        .iter()
        .enumerate()
        .map(|(i, (k, _))| (node.predict_slot(k), i))
        .collect();
    assignments.sort_unstable_by_key(|&(slot, _)| slot);

    let mut i = 0;
    while i < assignments.len() {
        let slot_idx = assignments[i].0;
        let start = i;
        while i < assignments.len() && assignments[i].0 == slot_idx {
            i += 1;
        }
        let run = &assignments[start..i];
        if run.len() == 1 {
            let (k, v) = &pairs[run[0].1];
            node.store_data(slot_idx, k.clone(), v.clone());
            node.inc_keys();
        } else {
            let child_pairs: Vec<(K, V)> = run
                .iter()
                .map(|&(_, idx)| {
                    let (k, v) = &pairs[idx];
                    (k.clone(), v.clone())
                })
                .collect();
            let child = if is_degenerate_group(&child_pairs) {
                build_degenerate(&child_pairs)
            } else {
                build_recursive(&child_pairs, config)
            };
            node.store_child(slot_idx, child);
        }
    }

    node
}

/// Check if all keys in a group have the same f64 model input.
fn is_degenerate_group<K: Key, V>(pairs: &[(K, V)]) -> bool {
    if pairs.len() <= 1 {
        return false;
    }
    let first_f = pairs[0].0.to_model_input();
    let last_f = pairs[pairs.len() - 1].0.to_model_input();
    (last_f - first_f).abs() < f64::EPSILON
}

/// Build a balanced binary tree for keys that share the same f64 representation.
fn build_degenerate<K: Key, V: Clone>(pairs: &[(K, V)]) -> Node<K, V> {
    debug_assert!(!pairs.is_empty());

    if pairs.len() == 1 {
        let node = Node::with_capacity(LinearModel::constant(), 1);
        node.store_data(0, pairs[0].0.clone(), pairs[0].1.clone());
        node.inc_keys();
        return node;
    }

    let mid_idx = pairs.len() / 2;
    let lo_half = &pairs[..mid_idx];
    let hi_half = &pairs[mid_idx..];

    let lo_last_ord = lo_half.last().unwrap().0.to_exact_ordinal();
    let hi_first_ord = hi_half.first().unwrap().0.to_exact_ordinal();

    let node = if lo_last_ord == hi_first_ord {
        Node::with_split_key(lo_half.last().unwrap().0.clone(), 2)
    } else {
        let midpoint = lo_last_ord + (hi_first_ord - lo_last_ord) / 2;
        let model = LinearModel::binary_split(midpoint);
        Node::with_capacity(model, 2)
    };

    if lo_half.len() == 1 {
        node.store_data(0, lo_half[0].0.clone(), lo_half[0].1.clone());
        node.inc_keys();
    } else {
        let child = build_degenerate(lo_half);
        node.store_child(0, child);
    }

    if hi_half.len() == 1 {
        node.store_data(1, hi_half[0].0.clone(), hi_half[0].1.clone());
        node.inc_keys();
    } else {
        let child = build_degenerate(hi_half);
        node.store_child(1, child);
    }

    node
}

#[cfg(test)]
mod tests {
    use super::*;

    use crossbeam_epoch as epoch;

    fn default_config() -> Config {
        Config::default()
    }

    fn guard() -> epoch::Guard {
        epoch::pin()
    }

    #[test]
    fn bulk_load_empty() {
        let result = bulk_load::<u64, ()>(&[], &default_config());
        assert!(matches!(result, Err(Error::EmptyData)));
    }

    #[test]
    fn bulk_load_not_sorted() {
        let pairs = vec![(3u64, "c"), (1, "a"), (2, "b")];
        let result = bulk_load(&pairs, &default_config());
        assert!(matches!(result, Err(Error::NotSorted)));
    }

    #[test]
    fn bulk_load_duplicates_rejected() {
        let pairs = vec![(1u64, "a"), (1, "b"), (2, "c")];
        let result = bulk_load(&pairs, &default_config());
        assert!(matches!(result, Err(Error::NotSorted)));
    }

    #[test]
    fn bulk_load_single() {
        let g = guard();
        let pairs = vec![(42u64, "hello")];
        let node = bulk_load(&pairs, &default_config()).unwrap();
        assert_eq!(node.total_keys(&g), 1);
    }

    #[test]
    fn bulk_load_sequential() {
        let g = guard();
        let pairs: Vec<(u64, usize)> = (0..100).map(|i| (i, i as usize)).collect();
        let node = bulk_load(&pairs, &default_config()).unwrap();
        assert_eq!(node.total_keys(&g), 100);
    }

    #[test]
    fn bulk_load_preserves_all_keys() {
        let g = guard();
        let pairs: Vec<(u64, u64)> = (0..50).map(|i| (i * 7 + 3, i)).collect();
        let node = bulk_load(&pairs, &default_config()).unwrap();
        assert_eq!(node.total_keys(&g), 50);
    }

    #[test]
    fn bulk_load_sparse_keys() {
        let g = guard();
        let pairs = vec![(1u64, 'a'), (1000, 'b'), (1_000_000, 'c')];
        let node = bulk_load(&pairs, &default_config()).unwrap();
        assert_eq!(node.total_keys(&g), 3);
    }

    #[test]
    fn bulk_load_signed_keys() {
        let g = guard();
        let pairs: Vec<(i64, &str)> = vec![(-100, "neg"), (0, "zero"), (100, "pos")];
        let node = bulk_load(&pairs, &default_config()).unwrap();
        assert_eq!(node.total_keys(&g), 3);
    }

    #[test]
    fn bulk_load_same_f64_keys() {
        let g = guard();
        let base: u64 = 1_700_000_000_000_000_000;
        let pairs: Vec<(u64, u64)> = (0..20).map(|i| (base + i, i)).collect();
        let node = bulk_load(&pairs, &default_config()).unwrap();
        assert_eq!(node.total_keys(&g), 20);
        for (k, v) in &pairs {
            assert_eq!(
                crate::lookup::get(&node, k, &g),
                Some(v),
                "missing key {k}"
            );
        }
    }

    #[test]
    fn build_degenerate_two_keys() {
        let g = guard();
        let base: u64 = 1_700_000_000_000_000_000;
        let pairs = vec![(base, 1u64), (base + 1, 2)];
        let node = build_degenerate(&pairs);
        assert_eq!(node.total_keys(&g), 2);
        assert_eq!(crate::lookup::get(&node, &base, &g), Some(&1));
        assert_eq!(crate::lookup::get(&node, &(base + 1), &g), Some(&2));
    }

    #[test]
    fn build_degenerate_many_keys() {
        let g = guard();
        let base: u64 = 1_700_000_000_000_000_000;
        let pairs: Vec<(u64, u64)> = (0..50).map(|i| (base + i, i)).collect();
        let node = build_degenerate(&pairs);
        assert_eq!(node.total_keys(&g), 50);
        for (k, v) in &pairs {
            assert_eq!(
                crate::lookup::get(&node, k, &g),
                Some(v),
                "missing key {k}"
            );
        }
    }
}
