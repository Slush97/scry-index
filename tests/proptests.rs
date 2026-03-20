//! Property-based tests using proptest for core scry-index invariants.

use proptest::prelude::*;
use std::collections::BTreeMap;

use scry_index::{Config, LearnedMap, LearnedSet};

/// Generate a random range bound pair (lo, hi) with lo <= hi within [0, max].
fn random_bounds(max: u64) -> impl Strategy<Value = (u64, u64)> {
    (0..max, 0..max).prop_map(|(a, b)| if a <= b { (a, b) } else { (b, a) })
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// Generate a vec of unique u64 keys (unordered).
fn unique_keys(max_len: usize) -> impl Strategy<Value = Vec<u64>> {
    prop::collection::hash_set(0..100_000u64, 1..max_len)
        .prop_map(|s| s.into_iter().collect::<Vec<_>>())
}

/// Generate a vec of unique u64 keys, sorted (for `bulk_load`).
fn sorted_unique_keys(max_len: usize) -> impl Strategy<Value = Vec<u64>> {
    unique_keys(max_len).prop_map(|mut v| {
        v.sort_unstable();
        v
    })
}

// ---------------------------------------------------------------------------
// Insert / lookup roundtrip
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn insert_lookup_roundtrip(keys in unique_keys(200)) {
        let map = LearnedMap::new();
        let g = map.guard();
        for &k in &keys {
            map.insert(k, k * 10, &g);
        }

        prop_assert_eq!(map.len(), keys.len());

        for &k in &keys {
            prop_assert_eq!(map.get(&k, &g), Some(&(k * 10)));
        }
    }

    #[test]
    fn insert_overwrite_keeps_latest(keys in unique_keys(100)) {
        let map = LearnedMap::new();
        let g = map.guard();
        for &k in &keys {
            map.insert(k, 1u64, &g);
        }
        for &k in &keys {
            map.insert(k, 2, &g);
        }

        prop_assert_eq!(map.len(), keys.len());

        for &k in &keys {
            prop_assert_eq!(map.get(&k, &g), Some(&2u64));
        }
    }
}

// ---------------------------------------------------------------------------
// Remove invariants
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn remove_makes_key_unfindable(keys in unique_keys(100)) {
        let map = LearnedMap::new();
        let g = map.guard();
        for &k in &keys {
            map.insert(k, k, &g);
        }

        // Remove the first half
        let to_remove = &keys[..keys.len() / 2];
        let to_keep = &keys[keys.len() / 2..];

        for &k in to_remove {
            prop_assert!(map.remove(&k, &g));
        }

        prop_assert_eq!(map.len(), to_keep.len());

        for &k in to_remove {
            prop_assert_eq!(map.get(&k, &g), None);
        }
        for &k in to_keep {
            prop_assert_eq!(map.get(&k, &g), Some(&k));
        }
    }

    #[test]
    fn remove_missing_is_noop(keys in unique_keys(50)) {
        let map = LearnedMap::new();
        let g = map.guard();
        for &k in &keys {
            map.insert(k, k, &g);
        }

        // Remove keys that were never inserted
        for i in 100_001..100_051u64 {
            prop_assert!(!map.remove(&i, &g));
        }
        prop_assert_eq!(map.len(), keys.len());
    }
}

// ---------------------------------------------------------------------------
// Sorted iteration
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn iter_sorted_always_sorted(keys in unique_keys(200)) {
        let map = LearnedMap::new();
        let g = map.guard();
        for &k in &keys {
            map.insert(k, k, &g);
        }

        let items: Vec<(u64, u64)> = map.iter_sorted(&g);
        prop_assert_eq!(items.len(), keys.len());

        for w in items.windows(2) {
            prop_assert!(w[0].0 < w[1].0, "not sorted: {} >= {}", w[0].0, w[1].0);
        }
    }

    #[test]
    fn iter_sorted_matches_insert_set(keys in unique_keys(150)) {
        let map = LearnedMap::new();
        let g = map.guard();
        let mut expected: BTreeMap<u64, u64> = BTreeMap::new();
        for &k in &keys {
            map.insert(k, k, &g);
            expected.insert(k, k);
        }

        let items: Vec<(u64, u64)> = map.iter_sorted(&g);
        let expected_items: Vec<(u64, u64)> = expected.into_iter().collect();
        prop_assert_eq!(items, expected_items);
    }
}

// ---------------------------------------------------------------------------
// Bulk load
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn bulk_load_roundtrip(keys in sorted_unique_keys(200)) {
        let pairs: Vec<(u64, u64)> = keys.iter().map(|&k| (k, k * 3)).collect();
        let map = LearnedMap::bulk_load(&pairs).unwrap();
        let g = map.guard();

        prop_assert_eq!(map.len(), keys.len());
        for &k in &keys {
            prop_assert_eq!(map.get(&k, &g), Some(&(k * 3)));
        }
    }

    #[test]
    fn bulk_load_sorted_iteration(keys in sorted_unique_keys(200)) {
        let pairs: Vec<(u64, u64)> = keys.iter().map(|&k| (k, k)).collect();
        let map = LearnedMap::bulk_load(&pairs).unwrap();
        let g = map.guard();

        let items: Vec<(u64, u64)> = map.iter_sorted(&g);
        prop_assert_eq!(items, pairs);
    }
}

// ---------------------------------------------------------------------------
// Len tracking
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn len_matches_oracle(
        inserts in prop::collection::vec(0..10_000u64, 1..200),
        removes in prop::collection::vec(0..10_000u64, 0..100),
    ) {
        let map = LearnedMap::new();
        let g = map.guard();
        let mut oracle = BTreeMap::new();

        for &k in &inserts {
            map.insert(k, k, &g);
            oracle.insert(k, k);
        }
        for &k in &removes {
            let m = map.remove(&k, &g);
            let o = oracle.remove(&k).is_some();
            prop_assert_eq!(m, o);
        }

        prop_assert_eq!(map.len(), oracle.len());

        // Spot check contents
        for (&k, &v) in &oracle {
            prop_assert_eq!(map.get(&k, &g), Some(&v));
        }
    }
}

// ---------------------------------------------------------------------------
// Rebuild preserves data
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn rebuild_preserves_all_entries(keys in unique_keys(200)) {
        let map = LearnedMap::new();
        let g = map.guard();
        for &k in &keys {
            map.insert(k, k, &g);
        }

        let before: Vec<(u64, u64)> = map.iter_sorted(&g);
        map.rebuild(&g);
        let after: Vec<(u64, u64)> = map.iter_sorted(&g);

        prop_assert_eq!(before, after);
        prop_assert_eq!(map.len(), keys.len());
    }
}

// ---------------------------------------------------------------------------
// LearnedSet mirrors BTreeSet
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn set_matches_btreeset(
        inserts in prop::collection::vec(0..10_000u64, 1..200),
        removes in prop::collection::vec(0..10_000u64, 0..100),
    ) {
        let set = LearnedSet::new();
        let g = set.guard();
        let mut oracle = std::collections::BTreeSet::new();

        for &k in &inserts {
            let s = set.insert(k, &g);
            let o = oracle.insert(k);
            prop_assert_eq!(s, o);
        }
        for &k in &removes {
            let s = set.remove(&k, &g);
            let o = oracle.remove(&k);
            prop_assert_eq!(s, o);
        }

        prop_assert_eq!(set.len(), oracle.len());
        for &k in &oracle {
            prop_assert!(set.contains(&k, &g));
        }
    }
}

// ---------------------------------------------------------------------------
// Expansion factor stress
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn various_expansion_factors(
        keys in sorted_unique_keys(100),
        factor in 1.0f64..5.0,
    ) {
        let pairs: Vec<(u64, u64)> = keys.iter().map(|&k| (k, k)).collect();
        let config = Config::new().expansion_factor(factor);
        let map = LearnedMap::bulk_load_with_config(&pairs, config).unwrap();
        let g = map.guard();

        prop_assert_eq!(map.len(), keys.len());
        for &k in &keys {
            prop_assert_eq!(map.get(&k, &g), Some(&k));
        }
    }
}

// ---------------------------------------------------------------------------
// Mixed operations converge with oracle
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn mixed_ops_match_btreemap(
        ops in prop::collection::vec(
            prop_oneof![
                (0..10_000u64).prop_map(|k| (true, k)),
                (0..10_000u64).prop_map(|k| (false, k)),
            ],
            1..300,
        )
    ) {
        let map = LearnedMap::new();
        let g = map.guard();
        let mut oracle = BTreeMap::new();

        for &(is_insert, key) in &ops {
            if is_insert {
                map.insert(key, key, &g);
                oracle.insert(key, key);
            } else {
                let m = map.remove(&key, &g);
                let o = oracle.remove(&key).is_some();
                prop_assert_eq!(m, o);
            }
        }

        prop_assert_eq!(map.len(), oracle.len());
        for (&k, &v) in &oracle {
            prop_assert_eq!(map.get(&k, &g), Some(&v));
        }
    }
}

// ---------------------------------------------------------------------------
// Iter sortedness (Phase 3)
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn iter_is_sorted_prop(keys in unique_keys(300)) {
        let map = LearnedMap::new();
        let g = map.guard();
        for &k in &keys {
            map.insert(k, k, &g);
        }

        let items: Vec<u64> = map.iter(&g).map(|(k, _)| *k).collect();
        prop_assert_eq!(items.len(), keys.len());
        for w in items.windows(2) {
            prop_assert!(w[0] < w[1], "not sorted: {} >= {}", w[0], w[1]);
        }
    }
}

// ---------------------------------------------------------------------------
// Range matches BTreeMap
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn range_matches_btreemap_range(
        keys in unique_keys(200),
        bounds in random_bounds(100_000),
    ) {
        let map = LearnedMap::new();
        let g = map.guard();
        let mut oracle = BTreeMap::new();
        for &k in &keys {
            map.insert(k, k, &g);
            oracle.insert(k, k);
        }

        let (lo, hi) = bounds;
        let map_keys: Vec<u64> = map.range(lo..=hi, &g).map(|(k, _)| *k).collect();
        let btree_keys: Vec<u64> = oracle.range(lo..=hi).map(|(k, _)| *k).collect();
        prop_assert_eq!(map_keys, btree_keys);
    }

    #[test]
    fn first_last_match_btreemap(keys in unique_keys(200)) {
        let map = LearnedMap::new();
        let g = map.guard();
        let mut oracle = BTreeMap::new();
        for &k in &keys {
            map.insert(k, k, &g);
            oracle.insert(k, k);
        }

        let map_first = map.first_key_value(&g).map(|(k, _)| *k);
        let btree_first = oracle.keys().next().copied();
        prop_assert_eq!(map_first, btree_first);

        let map_last = map.last_key_value(&g).map(|(k, _)| *k);
        let btree_last = oracle.keys().next_back().copied();
        prop_assert_eq!(map_last, btree_last);
    }
}
