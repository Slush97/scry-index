//! Bulk-load construction for building a learned index from sorted data.
#![allow(unsafe_code)]

use crate::config::Config;
use crate::error::{Error, Result};
use crate::key::Key;
use crate::model::fit_fmcd;
use crate::node::{Node, SlotInner};

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
    let keys: Vec<K> = pairs.iter().map(|(k, _)| *k).collect();
    let result = fit_fmcd(&keys, config.expansion_factor);

    let node = Node::with_capacity(result.model, result.array_size);

    if result.conflicts == 0 {
        // No conflicts — place each key-value directly in its predicted slot
        for (key, value) in pairs {
            let slot = node.predict_slot(*key);
            node.store_slot(slot, SlotInner::Data { key: *key, value: value.clone() });
            node.inc_keys();
        }
    } else {
        // Conflicts exist — sort by predicted slot, then process runs.
        // This avoids allocating one Vec per slot (which can be huge for large
        // array_size with few actual conflicts).
        let mut assignments: Vec<(usize, usize)> = pairs
            .iter()
            .enumerate()
            .map(|(i, (k, _))| (node.predict_slot(*k), i))
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
                node.store_slot(slot_idx, SlotInner::Data { key: *k, value: v.clone() });
                node.inc_keys();
            } else {
                let child_pairs: Vec<(K, V)> =
                    run.iter().map(|&(_, idx)| {
                        let (k, v) = &pairs[idx];
                        (*k, v.clone())
                    }).collect();
                let child = build_recursive(&child_pairs, config);
                node.store_slot(slot_idx, SlotInner::Child(child));
            }
        }
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
}
