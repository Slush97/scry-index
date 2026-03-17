//! Bulk-load construction for building a learned index from sorted data.

use crate::config::Config;
use crate::error::{Error, Result};
use crate::key::Key;
use crate::model::fit_fmcd;
use crate::node::{Node, Slot};

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
fn build_recursive<K: Key, V: Clone>(pairs: &[(K, V)], config: &Config) -> Node<K, V> {
    let keys: Vec<K> = pairs.iter().map(|(k, _)| *k).collect();
    let result = fit_fmcd(&keys, config.expansion_factor);

    let mut node = Node::with_capacity(result.model, result.array_size);

    if result.conflicts == 0 {
        // No conflicts — place each key-value directly in its predicted slot
        for (key, value) in pairs {
            let slot = node.predict_slot(*key);
            node.slots[slot] = Slot::Data(*key, value.clone());
            node.num_keys += 1;
        }
    } else {
        // Conflicts exist — group keys by predicted slot, recurse on groups
        let mut groups: Vec<Vec<(K, V)>> = vec![Vec::new(); result.array_size];
        for (key, value) in pairs {
            let slot = node.predict_slot(*key);
            groups[slot].push((*key, value.clone()));
        }

        for (slot_idx, group) in groups.into_iter().enumerate() {
            match group.len() {
                0 => {} // Slot::Empty (already initialized)
                1 => {
                    let (k, v) = group.into_iter().next().expect("checked len == 1");
                    node.slots[slot_idx] = Slot::Data(k, v);
                    node.num_keys += 1;
                }
                _ => {
                    // Multiple keys map to this slot — create a child node
                    let child = build_recursive(&group, config);
                    node.slots[slot_idx] = Slot::Child(Box::new(child));
                }
            }
        }
    }

    node
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> Config {
        Config::default()
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
        let pairs = vec![(42u64, "hello")];
        let node = bulk_load(&pairs, &default_config()).unwrap();
        assert_eq!(node.total_keys(), 1);
    }

    #[test]
    fn bulk_load_sequential() {
        let pairs: Vec<(u64, usize)> = (0..100).map(|i| (i, i as usize)).collect();
        let node = bulk_load(&pairs, &default_config()).unwrap();
        assert_eq!(node.total_keys(), 100);
    }

    #[test]
    fn bulk_load_preserves_all_keys() {
        let pairs: Vec<(u64, u64)> = (0..50).map(|i| (i * 7 + 3, i)).collect();
        let node = bulk_load(&pairs, &default_config()).unwrap();
        assert_eq!(node.total_keys(), 50);
    }

    #[test]
    fn bulk_load_sparse_keys() {
        let pairs = vec![(1u64, 'a'), (1000, 'b'), (1_000_000, 'c')];
        let node = bulk_load(&pairs, &default_config()).unwrap();
        assert_eq!(node.total_keys(), 3);
    }

    #[test]
    fn bulk_load_signed_keys() {
        let pairs: Vec<(i64, &str)> = vec![(-100, "neg"), (0, "zero"), (100, "pos")];
        let node = bulk_load(&pairs, &default_config()).unwrap();
        assert_eq!(node.total_keys(), 3);
    }
}
