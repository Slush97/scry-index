//! End-to-end integration tests for the scry-index Phase 2 concurrent API.

use std::collections::BTreeMap;

use scry_index::{Config, Error, LearnedMap, LearnedSet};

// ---------------------------------------------------------------------------
// LearnedMap: construction
// ---------------------------------------------------------------------------

#[test]
fn empty_map_has_no_entries() {
    let map = LearnedMap::<u64, i32>::new();
    let g = map.guard();
    assert!(map.is_empty());
    assert_eq!(map.len(), 0);
    assert_eq!(map.get(&0, &g), None);
}

#[test]
fn bulk_load_rejects_empty() {
    let result = LearnedMap::<u64, ()>::bulk_load(&[]);
    assert!(matches!(result, Err(Error::EmptyData)));
}

#[test]
fn bulk_load_rejects_unsorted() {
    let pairs = vec![(5u64, 'a'), (3, 'b'), (7, 'c')];
    let result = LearnedMap::bulk_load(&pairs);
    assert!(matches!(result, Err(Error::NotSorted)));
}

#[test]
fn bulk_load_rejects_duplicates() {
    let pairs = vec![(1u64, "x"), (1, "y")];
    let result = LearnedMap::bulk_load(&pairs);
    assert!(matches!(result, Err(Error::NotSorted)));
}

#[test]
fn bulk_load_all_keys_retrievable() {
    let pairs: Vec<(u64, u64)> = (0..500).map(|i| (i * 7, i)).collect();
    let map = LearnedMap::bulk_load(&pairs).unwrap();
    let g = map.guard();
    assert_eq!(map.len(), 500);
    for &(k, v) in &pairs {
        assert_eq!(map.get(&k, &g), Some(&v), "missing key {k}");
    }
}

#[test]
fn bulk_load_with_custom_config() {
    let pairs: Vec<(u64, u64)> = (0..100).map(|i| (i, i)).collect();
    let config = Config::new().expansion_factor(3.0);
    let map = LearnedMap::bulk_load_with_config(&pairs, config).unwrap();
    let g = map.guard();
    assert_eq!(map.len(), 100);
    for i in 0..100u64 {
        assert_eq!(map.get(&i, &g), Some(&i));
    }
}

// ---------------------------------------------------------------------------
// LearnedMap: insert / get / remove lifecycle
// ---------------------------------------------------------------------------

#[test]
fn insert_get_remove_cycle() {
    let map = LearnedMap::new();
    let g = map.guard();
    for i in 0..100u64 {
        assert!(map.insert(i, i * 10, &g));
    }
    assert_eq!(map.len(), 100);

    // All present
    for i in 0..100u64 {
        assert_eq!(map.get(&i, &g), Some(&(i * 10)));
    }

    // Remove odds
    for i in (1..100u64).step_by(2) {
        assert!(map.remove(&i, &g));
    }
    assert_eq!(map.len(), 50);

    // Evens still present, odds gone
    for i in 0..100u64 {
        if i % 2 == 0 {
            assert_eq!(map.get(&i, &g), Some(&(i * 10)));
        } else {
            assert_eq!(map.get(&i, &g), None);
        }
    }
}

#[test]
fn insert_updates_existing_key() {
    let map = LearnedMap::new();
    let g = map.guard();
    assert!(map.insert(1u64, "first", &g));
    assert!(!map.insert(1, "second", &g));
    assert_eq!(map.get(&1, &g), Some(&"second"));
    assert_eq!(map.len(), 1);
}

#[test]
fn contains_key_matches_get() {
    let map = LearnedMap::new();
    let g = map.guard();
    map.insert(10u64, (), &g);
    assert!(map.contains_key(&10, &g));
    assert!(!map.contains_key(&11, &g));
    map.remove(&10, &g);
    assert!(!map.contains_key(&10, &g));
}

#[test]
fn remove_returns_false_for_missing() {
    let map = LearnedMap::<u64, u64>::new();
    let g = map.guard();
    map.insert(1, 1, &g);
    assert!(!map.remove(&999, &g));
    assert_eq!(map.len(), 1);
}

#[test]
fn remove_then_reinsert() {
    let map = LearnedMap::new();
    let g = map.guard();
    map.insert(42u64, "original", &g);
    map.remove(&42, &g);
    map.insert(42, "reinserted", &g);
    assert_eq!(map.get(&42, &g), Some(&"reinserted"));
    assert_eq!(map.len(), 1);
}

// ---------------------------------------------------------------------------
// LearnedMap: iteration
// ---------------------------------------------------------------------------

#[test]
fn iter_sorted_produces_ascending_keys() {
    let map = LearnedMap::new();
    let g = map.guard();
    // Insert in reverse order
    for i in (0..200u64).rev() {
        map.insert(i, i, &g);
    }
    let items: Vec<(u64, u64)> = map.iter_sorted(&g);
    assert_eq!(items.len(), 200);
    for w in items.windows(2) {
        assert!(w[0].0 < w[1].0, "not sorted: {} >= {}", w[0].0, w[1].0);
    }
}

#[test]
fn iter_sorted_after_mixed_ops() {
    let map = LearnedMap::new();
    let g = map.guard();
    for i in 0..50u64 {
        map.insert(i * 2, i, &g);
    }
    // Remove some
    for i in (0..50u64).step_by(3) {
        map.remove(&(i * 2), &g);
    }
    // Insert odds
    for i in 0..25u64 {
        map.insert(i * 2 + 1, i + 1000, &g);
    }

    let items: Vec<(u64, u64)> = map.iter_sorted(&g);
    assert_eq!(items.len(), map.len());
    for w in items.windows(2) {
        assert!(w[0].0 < w[1].0);
    }
}

// ---------------------------------------------------------------------------
// LearnedMap: FromIterator / Extend
// ---------------------------------------------------------------------------

#[test]
fn from_iterator_collects_all() {
    let map: LearnedMap<u64, u64> = (0..75).map(|i| (i, i)).collect();
    let g = map.guard();
    assert_eq!(map.len(), 75);
    for i in 0..75u64 {
        assert!(map.contains_key(&i, &g));
    }
}

#[test]
fn extend_adds_to_existing() {
    let mut map = LearnedMap::new();
    let g = map.guard();
    map.insert(1u64, 10, &g);
    map.extend(vec![(2, 20), (3, 30), (4, 40)]);
    assert_eq!(map.len(), 4);
    assert_eq!(map.get(&3, &g), Some(&30));
}

#[test]
fn extend_with_overlap_updates() {
    let mut map = LearnedMap::new();
    let g = map.guard();
    map.insert(1u64, 10, &g);
    map.extend(vec![(1, 99), (2, 20)]);
    assert_eq!(map.len(), 2);
    assert_eq!(map.get(&1, &g), Some(&99));
}

// ---------------------------------------------------------------------------
// LearnedMap: rebuild
// ---------------------------------------------------------------------------

#[test]
fn rebuild_preserves_all_data() {
    let map = LearnedMap::new();
    let g = map.guard();
    for i in (0..300u64).rev() {
        map.insert(i, i * 7, &g);
    }
    let len_before = map.len();
    map.rebuild(&g);
    assert_eq!(map.len(), len_before);
    for i in 0..300u64 {
        assert_eq!(
            map.get(&i, &g),
            Some(&(i * 7)),
            "key {i} lost after rebuild"
        );
    }
}

#[test]
fn rebuild_reduces_depth() {
    let map = LearnedMap::new();
    let g = map.guard();
    // Reverse insert creates worst-case chaining
    for i in (0..200u64).rev() {
        map.insert(i, i, &g);
    }
    let depth_before = map.max_depth(&g);
    map.rebuild(&g);
    assert!(
        map.max_depth(&g) <= depth_before,
        "rebuild worsened depth: {} -> {}",
        depth_before,
        map.max_depth(&g)
    );
}

#[test]
fn rebuild_empty_is_noop() {
    let map = LearnedMap::<u64, u64>::new();
    let g = map.guard();
    map.rebuild(&g);
    assert!(map.is_empty());
}

// ---------------------------------------------------------------------------
// LearnedMap: mixed bulk_load + insert
// ---------------------------------------------------------------------------

#[test]
fn bulk_load_then_incremental_inserts() {
    let pairs: Vec<(u64, u64)> = (0..100).map(|i| (i * 2, i)).collect();
    let map = LearnedMap::bulk_load(&pairs).unwrap();
    let g = map.guard();

    // Fill in the gaps
    for i in 0..100u64 {
        map.insert(i * 2 + 1, i + 1000, &g);
    }
    assert_eq!(map.len(), 200);

    for i in 0..200u64 {
        assert!(map.get(&i, &g).is_some(), "key {i} not found");
    }
}

// ---------------------------------------------------------------------------
// LearnedMap: different key types
// ---------------------------------------------------------------------------

#[test]
fn works_with_i64_keys() {
    let pairs: Vec<(i64, &str)> = vec![(-100, "neg"), (-1, "almost"), (0, "zero"), (50, "pos")];
    let map = LearnedMap::bulk_load(&pairs).unwrap();
    let g = map.guard();
    assert_eq!(map.get(&-100, &g), Some(&"neg"));
    assert_eq!(map.get(&0, &g), Some(&"zero"));
    assert_eq!(map.get(&50, &g), Some(&"pos"));
    assert_eq!(map.get(&1, &g), None);
}

#[test]
fn works_with_u32_keys() {
    let map = LearnedMap::new();
    let g = map.guard();
    map.insert(u32::MIN, "min", &g);
    map.insert(u32::MAX, "max", &g);
    map.insert(1000u32, "mid", &g);
    assert_eq!(map.len(), 3);
    assert_eq!(map.get(&u32::MIN, &g), Some(&"min"));
    assert_eq!(map.get(&u32::MAX, &g), Some(&"max"));
}

#[test]
fn works_with_i32_keys() {
    let map = LearnedMap::new();
    let g = map.guard();
    map.insert(i32::MIN, 0, &g);
    map.insert(0i32, 1, &g);
    map.insert(i32::MAX, 2, &g);
    assert_eq!(map.len(), 3);
    assert_eq!(map.get(&i32::MIN, &g), Some(&0));
}

// ---------------------------------------------------------------------------
// LearnedMap: depth invariants
// ---------------------------------------------------------------------------

#[test]
fn bulk_load_depth_bounded_1k() {
    let pairs: Vec<(u64, u64)> = (0..1000).map(|i| (i, i)).collect();
    let map = LearnedMap::bulk_load(&pairs).unwrap();
    let g = map.guard();
    assert!(
        map.max_depth(&g) <= 5,
        "depth {} too high for 1000 sequential keys",
        map.max_depth(&g)
    );
}

#[test]
fn bulk_load_depth_bounded_sparse() {
    // Highly non-uniform: powers of 2
    let pairs: Vec<(u64, u64)> = (0..20).map(|i| (1u64 << i, i)).collect();
    let map = LearnedMap::bulk_load(&pairs).unwrap();
    let g = map.guard();
    assert!(
        map.max_depth(&g) <= 10,
        "depth {} too high for 20 power-of-2 keys",
        map.max_depth(&g)
    );
}

// ---------------------------------------------------------------------------
// LearnedMap: stress
// ---------------------------------------------------------------------------

#[test]
fn stress_1000_random_pattern() {
    use std::collections::BTreeMap;

    let map = LearnedMap::new();
    let g = map.guard();
    let mut oracle = BTreeMap::new();

    // Insert keys with gaps
    for i in 0..1000u64 {
        let key = i.wrapping_mul(2_654_435_761) % 100_000;
        map.insert(key, i, &g);
        oracle.insert(key, i);
    }

    assert_eq!(map.len(), oracle.len());

    // Verify all oracle entries
    for (&k, &v) in &oracle {
        assert_eq!(map.get(&k, &g), Some(&v), "mismatch at key {k}");
    }

    // Remove half
    let keys_to_remove: Vec<u64> = oracle.keys().step_by(2).copied().collect();
    for k in &keys_to_remove {
        map.remove(k, &g);
        oracle.remove(k);
    }

    assert_eq!(map.len(), oracle.len());
    for (&k, &v) in &oracle {
        assert_eq!(
            map.get(&k, &g),
            Some(&v),
            "mismatch after removal at key {k}"
        );
    }
}

// ---------------------------------------------------------------------------
// LearnedSet
// ---------------------------------------------------------------------------

#[test]
fn set_empty() {
    let set = LearnedSet::<u64>::new();
    let g = set.guard();
    assert!(set.is_empty());
    assert_eq!(set.len(), 0);
    assert!(!set.contains(&0, &g));
}

#[test]
fn set_insert_contains_remove() {
    let set = LearnedSet::new();
    let g = set.guard();
    assert!(set.insert(10u64, &g));
    assert!(set.insert(20, &g));
    assert!(!set.insert(10, &g)); // duplicate
    assert_eq!(set.len(), 2);

    assert!(set.contains(&10, &g));
    assert!(set.contains(&20, &g));
    assert!(!set.contains(&30, &g));

    assert!(set.remove(&10, &g));
    assert!(!set.remove(&10, &g)); // already removed
    assert_eq!(set.len(), 1);
    assert!(!set.contains(&10, &g));
}

#[test]
fn set_bulk_load() {
    let keys: Vec<u64> = (0..200).collect();
    let set = LearnedSet::bulk_load(&keys).unwrap();
    let g = set.guard();
    assert_eq!(set.len(), 200);
    for k in &keys {
        assert!(set.contains(k, &g), "missing key {k}");
    }
    assert!(!set.contains(&200, &g));
}

#[test]
fn set_bulk_load_rejects_unsorted() {
    let keys = vec![5u64, 3, 7];
    assert!(LearnedSet::bulk_load(&keys).is_err());
}

#[test]
fn set_from_iterator() {
    let set: LearnedSet<u64> = (0..50).collect();
    let g = set.guard();
    assert_eq!(set.len(), 50);
    for i in 0..50u64 {
        assert!(set.contains(&i, &g));
    }
}

#[test]
fn set_extend() {
    let mut set = LearnedSet::new();
    let g = set.guard();
    set.insert(1u64, &g);
    set.extend(vec![2, 3, 4, 5]);
    assert_eq!(set.len(), 5);
}

#[test]
fn set_extend_with_duplicates() {
    let mut set: LearnedSet<u64> = (0..10).collect();
    set.extend(0..10); // all duplicates
    assert_eq!(set.len(), 10);
}

#[test]
fn set_default() {
    let set = LearnedSet::<u64>::default();
    assert!(set.is_empty());
}

#[test]
fn set_with_config() {
    let config = Config::new().expansion_factor(3.0);
    let set = LearnedSet::with_config(config);
    let g = set.guard();
    set.insert(1u64, &g);
    set.insert(2, &g);
    assert_eq!(set.len(), 2);
}

#[test]
fn set_large_insert_remove() {
    let set = LearnedSet::new();
    let g = set.guard();
    for i in 0..500u64 {
        set.insert(i, &g);
    }
    assert_eq!(set.len(), 500);

    for i in (0..500u64).step_by(2) {
        set.remove(&i, &g);
    }
    assert_eq!(set.len(), 250);

    for i in 0..500u64 {
        if i % 2 == 0 {
            assert!(!set.contains(&i, &g));
        } else {
            assert!(set.contains(&i, &g));
        }
    }
}

// ---------------------------------------------------------------------------
// Range queries (Phase 3)
// ---------------------------------------------------------------------------

#[test]
fn range_on_empty_map() {
    let map = LearnedMap::<u64, u64>::new();
    let g = map.guard();
    let items: Vec<_> = map.range(0..100, &g).collect();
    assert!(items.is_empty());
}

#[test]
fn range_full_matches_iter_sorted() {
    let map = LearnedMap::new();
    let g = map.guard();
    for i in (0..200u64).rev() {
        map.insert(i, i * 10, &g);
    }
    let from_range: Vec<(u64, u64)> = map.range(.., &g).map(|(&k, &v)| (k, v)).collect();
    let from_sorted = map.iter_sorted(&g);
    assert_eq!(from_range, from_sorted);
}

#[test]
fn range_bounded_matches_btreemap() {
    let map = LearnedMap::new();
    let g = map.guard();
    let mut oracle = BTreeMap::new();
    for i in 0..1000u64 {
        let key = i.wrapping_mul(2_654_435_761) % 50_000;
        map.insert(key, key, &g);
        oracle.insert(key, key);
    }

    // Test several range bounds
    let ranges: Vec<std::ops::Range<u64>> =
        vec![0..100, 1000..5000, 10_000..20_000, 49_000..50_001];
    for r in ranges {
        let map_keys: Vec<u64> = map.range(r.clone(), &g).map(|(k, _)| *k).collect();
        let btree_keys: Vec<u64> = oracle.range(r.clone()).map(|(k, _)| *k).collect();
        assert_eq!(map_keys, btree_keys, "mismatch for range {r:?}");
    }
}

#[test]
fn range_with_removed_keys() {
    let map = LearnedMap::new();
    let g = map.guard();
    for i in 0..100u64 {
        map.insert(i, i, &g);
    }
    // Remove even keys in range 20..40
    for i in (20..40u64).step_by(2) {
        map.remove(&i, &g);
    }

    let items: Vec<u64> = map.range(20..40, &g).map(|(k, _)| *k).collect();
    // Only odd keys 21,23,...,39 should remain
    let expected: Vec<u64> = (21..40).step_by(2).collect();
    assert_eq!(items, expected);
}

#[test]
fn range_various_bound_types() {
    let pairs: Vec<(u64, u64)> = (0..50).map(|i| (i, i)).collect();
    let map = LearnedMap::bulk_load(&pairs).unwrap();
    let g = map.guard();

    // ..
    assert_eq!(map.range(.., &g).count(), 50);
    // a..b
    assert_eq!(map.range(10..20, &g).count(), 10);
    // a..=b
    assert_eq!(map.range(10..=20, &g).count(), 11);
    // a..
    assert_eq!(map.range(40.., &g).count(), 10);
    // ..b
    assert_eq!(map.range(..10, &g).count(), 10);
    // ..=b
    assert_eq!(map.range(..=10, &g).count(), 11);
}

#[test]
fn first_last_key_value() {
    let map = LearnedMap::new();
    let g = map.guard();

    // Empty map
    assert!(map.first_key_value(&g).is_none());
    assert!(map.last_key_value(&g).is_none());

    for i in (0..100u64).rev() {
        map.insert(i * 3, i, &g);
    }

    let sorted = map.iter_sorted(&g);
    let first = map.first_key_value(&g).unwrap();
    let last = map.last_key_value(&g).unwrap();

    assert_eq!(*first.0, sorted.first().unwrap().0);
    assert_eq!(*last.0, sorted.last().unwrap().0);
}

#[test]
fn set_range_and_first_last() {
    let keys: Vec<u64> = (0..100).collect();
    let set = LearnedSet::bulk_load(&keys).unwrap();
    let g = set.guard();

    // first / last
    assert_eq!(set.first(&g), Some(&0u64));
    assert_eq!(set.last(&g), Some(&99u64));

    // range
    let range_keys: Vec<u64> = set.range(10..=20, &g).copied().collect();
    assert_eq!(range_keys, (10..=20).collect::<Vec<_>>());
}

#[test]
fn map_ref_range_and_first_last() {
    let map = LearnedMap::new();
    let m = map.pin();
    for i in 0..50u64 {
        m.insert(i, i * 10);
    }

    let first = m.first_key_value().unwrap();
    assert_eq!(*first.0, 0);
    let last = m.last_key_value().unwrap();
    assert_eq!(*last.0, 49);

    let range_keys: Vec<u64> = m.range(10..20).map(|(k, _)| *k).collect();
    assert_eq!(range_keys, (10..20).collect::<Vec<_>>());

    assert_eq!(m.range_count(10..20), 10);
}

#[test]
fn set_ref_range_and_first_last() {
    let set = LearnedSet::new();
    let s = set.pin();
    for i in 0..50u64 {
        s.insert(i);
    }

    assert_eq!(s.first(), Some(&0u64));
    assert_eq!(s.last(), Some(&49u64));

    let range_keys: Vec<u64> = s.range(10..20).copied().collect();
    assert_eq!(range_keys, (10..20).collect::<Vec<_>>());
}

// ---------------------------------------------------------------------------
// f64 precision: keys with identical f64 representation (regression tests)
// ---------------------------------------------------------------------------

#[test]
fn insert_u64_keys_same_f64() {
    let base: u64 = 1_700_000_000_000_000_000;
    let k1 = base;
    let k2 = base + 1;
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(k1 as f64, k2 as f64, "precondition: keys must share f64");
    }

    let map = LearnedMap::new();
    let g = map.guard();
    map.insert(k1, "first", &g);
    map.insert(k2, "second", &g);

    assert_eq!(map.len(), 2);
    assert_eq!(map.get(&k1, &g), Some(&"first"));
    assert_eq!(map.get(&k2, &g), Some(&"second"));
}

#[test]
fn insert_many_u64_keys_same_f64() {
    let base: u64 = 1_700_000_000_000_000_000;
    let n = 100u64;
    let map = LearnedMap::new();
    let g = map.guard();
    for i in 0..n {
        map.insert(base + i, i, &g);
    }
    assert_eq!(map.len(), n as usize);
    for i in 0..n {
        assert_eq!(map.get(&(base + i), &g), Some(&i), "missing key base+{i}");
    }
}

#[test]
fn bulk_load_u64_keys_same_f64() {
    let base: u64 = 1_700_000_000_000_000_000;
    let pairs: Vec<(u64, u64)> = (0..50).map(|i| (base + i, i)).collect();
    let map = LearnedMap::bulk_load(&pairs).unwrap();
    let g = map.guard();
    assert_eq!(map.len(), 50);
    for (k, v) in &pairs {
        assert_eq!(map.get(k, &g), Some(v), "missing key {k}");
    }
}

#[test]
fn remove_from_same_f64_keys() {
    let base: u64 = 1_700_000_000_000_000_000;
    let map = LearnedMap::new();
    let g = map.guard();
    map.insert(base, 1u64, &g);
    map.insert(base + 1, 2, &g);
    assert!(map.remove(&base, &g));
    assert_eq!(map.get(&base, &g), None);
    assert_eq!(map.get(&(base + 1), &g), Some(&2));
    assert_eq!(map.len(), 1);
}

#[test]
fn iter_sorted_with_same_f64_keys() {
    let base: u64 = 1_700_000_000_000_000_000;
    let pairs: Vec<(u64, u64)> = (0..20).map(|i| (base + i, i)).collect();
    let map = LearnedMap::bulk_load(&pairs).unwrap();
    let g = map.guard();
    let sorted = map.iter_sorted(&g);
    assert_eq!(sorted.len(), 20);
    for w in sorted.windows(2) {
        assert!(w[0].0 < w[1].0, "not sorted: {} >= {}", w[0].0, w[1].0);
    }
}

#[test]
fn rebuild_with_same_f64_keys() {
    let base: u64 = 1_700_000_000_000_000_000;
    let map = LearnedMap::new();
    let g = map.guard();
    for i in 0..50u64 {
        map.insert(base + i, i, &g);
    }
    map.rebuild(&g);
    let g2 = map.guard();
    for i in 0..50u64 {
        assert_eq!(
            map.get(&(base + i), &g2),
            Some(&i),
            "key base+{i} lost after rebuild"
        );
    }
}

#[test]
fn mixed_normal_and_degenerate_keys() {
    let base: u64 = 1_700_000_000_000_000_000;
    let map = LearnedMap::new();
    let g = map.guard();
    for i in 0..100u64 {
        map.insert(i, i, &g);
    }
    for i in 0..20u64 {
        map.insert(base + i, i + 1000, &g);
    }
    assert_eq!(map.len(), 120);
    for i in 0..100u64 {
        assert_eq!(map.get(&i, &g), Some(&i));
    }
    for i in 0..20u64 {
        assert_eq!(map.get(&(base + i), &g), Some(&(i + 1000)));
    }
}

#[test]
fn update_value_same_f64_keys() {
    let base: u64 = 1_700_000_000_000_000_000;
    let map = LearnedMap::new();
    let g = map.guard();
    map.insert(base, 1u64, &g);
    map.insert(base + 1, 2, &g);
    // Update existing key
    assert!(!map.insert(base, 99, &g));
    assert_eq!(map.get(&base, &g), Some(&99));
    assert_eq!(map.get(&(base + 1), &g), Some(&2));
    assert_eq!(map.len(), 2);
}

// ---------------------------------------------------------------------------
// Model quality: FMCD + range headroom
// ---------------------------------------------------------------------------

#[test]
fn headroom_prevents_edge_pileup() {
    // Insert 200 keys to trigger root rebuilds, then insert 200 more
    // beyond the original range. With headroom, the second batch should
    // NOT all pile up in the last slot.
    let map = LearnedMap::new();
    let g = map.guard();
    for i in 0..200u64 {
        map.insert(i, i, &g);
    }
    // Root has been rebuilt (threshold 64, 128). Now insert beyond range.
    let g2 = map.guard();
    for i in 200..400u64 {
        map.insert(i, i, &g2);
    }
    let g3 = map.guard();
    assert_eq!(map.len(), 400);
    let depth = map.max_depth(&g3);
    assert!(
        depth <= 12,
        "depth {depth} too high — headroom should prevent edge pile-up"
    );
    for i in 0..400u64 {
        assert_eq!(map.get(&i, &g3), Some(&i), "key {i} missing");
    }
}

#[test]
fn clustered_keys_bulk_load_shallow() {
    // Two clusters far apart. FMCD should fit a model that separates them
    // cleanly, producing a shallow tree.
    let mut pairs: Vec<(u64, u64)> = (0..50).map(|i| (i, i)).collect();
    pairs.extend((0..50).map(|i| (1_000_000 + i, i + 50)));
    let map = LearnedMap::bulk_load(&pairs).unwrap();
    let g = map.guard();
    let depth = map.max_depth(&g);
    assert!(
        depth <= 3,
        "depth {depth} too high for clustered bulk-loaded data"
    );
    assert_eq!(map.len(), 100);
}

#[test]
fn incremental_insert_1000_from_empty() {
    let map = LearnedMap::new();
    let g = map.guard();
    for i in 0..1000u64 {
        map.insert(i, i, &g);
    }
    let g2 = map.guard();
    assert_eq!(map.len(), 1000);
    let depth = map.max_depth(&g2);
    assert!(
        depth <= 12,
        "depth {depth} too high for 1000 sequential inserts from empty"
    );
    for i in 0..1000u64 {
        assert_eq!(map.get(&i, &g2), Some(&i), "key {i} missing");
    }
}
