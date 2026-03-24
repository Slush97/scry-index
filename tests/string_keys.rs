//! Integration tests for `String` and `Vec<u8>` key support.

use scry_index::{Config, LearnedMap};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Basic insert / get / remove roundtrip
// ---------------------------------------------------------------------------

#[test]
fn string_insert_get_roundtrip() {
    let map = LearnedMap::new();
    let guard = map.guard();

    map.insert("hello".to_string(), 1, &guard);
    map.insert("world".to_string(), 2, &guard);

    assert_eq!(map.get(&"hello".to_string(), &guard), Some(&1));
    assert_eq!(map.get(&"world".to_string(), &guard), Some(&2));
    assert_eq!(map.get(&"missing".to_string(), &guard), None);
    assert_eq!(map.len(), 2);
}

#[test]
fn string_update_existing() {
    let map = LearnedMap::new();
    let guard = map.guard();

    assert!(map.insert("key".to_string(), 1, &guard));
    assert!(!map.insert("key".to_string(), 2, &guard));
    assert_eq!(map.get(&"key".to_string(), &guard), Some(&2));
    assert_eq!(map.len(), 1);
}

#[test]
fn string_remove() {
    let map = LearnedMap::new();
    let guard = map.guard();

    map.insert("a".to_string(), 1, &guard);
    map.insert("b".to_string(), 2, &guard);
    map.insert("c".to_string(), 3, &guard);

    assert!(map.remove(&"b".to_string(), &guard));
    assert_eq!(map.get(&"b".to_string(), &guard), None);
    assert_eq!(map.get(&"a".to_string(), &guard), Some(&1));
    assert_eq!(map.get(&"c".to_string(), &guard), Some(&3));
}

#[test]
fn string_get_or_insert() {
    let map = LearnedMap::new();
    let guard = map.guard();

    let v = map.get_or_insert("key".to_string(), 10, &guard);
    assert_eq!(*v, 10);

    let v = map.get_or_insert("key".to_string(), 99, &guard);
    assert_eq!(*v, 10); // original value, not 99
}

// ---------------------------------------------------------------------------
// Strings with shared prefixes — collision handling
// ---------------------------------------------------------------------------

#[test]
fn shared_8byte_prefix_same_model_input() {
    // Strings sharing their first 8 bytes have the same to_model_input().
    // This triggers the degenerate path in build, which uses binary_split
    // or split_key depending on whether ordinals also collide.
    let map = LearnedMap::new();
    let guard = map.guard();

    let prefix = "AAAAAAAA"; // 8 bytes
    for i in 0..20u32 {
        let key = format!("{prefix}{i:04}");
        map.insert(key, i, &guard);
    }

    assert_eq!(map.len(), 20);
    for i in 0..20u32 {
        let key = format!("{prefix}{i:04}");
        assert_eq!(
            map.get(&key, &guard),
            Some(&i),
            "key {key:?} not found"
        );
    }
}

#[test]
fn shared_16byte_prefix_ordinal_collision() {
    // Strings sharing their first 16 bytes have the same to_exact_ordinal().
    // This triggers the Ord-based split_key fallback in nodes.
    let map = LearnedMap::new();
    let guard = map.guard();

    let prefix = "0123456789abcdef"; // exactly 16 bytes
    for i in 0..30u32 {
        let key = format!("{prefix}_{i:04}");
        map.insert(key, i, &guard);
    }

    assert_eq!(map.len(), 30);
    for i in 0..30u32 {
        let key = format!("{prefix}_{i:04}");
        assert_eq!(
            map.get(&key, &guard),
            Some(&i),
            "key {key:?} not found after ordinal-collision insert"
        );
    }
}

#[test]
fn shared_long_prefix_many_keys() {
    // Stress test: 100 keys with a 32-byte shared prefix.
    let map = LearnedMap::new();
    let guard = map.guard();

    let prefix = "abcdefghijklmnopqrstuvwxyz012345"; // 32 bytes
    for i in 0..100u32 {
        let key = format!("{prefix}{i:06}");
        map.insert(key, i, &guard);
    }

    assert_eq!(map.len(), 100);
    for i in 0..100u32 {
        let key = format!("{prefix}{i:06}");
        assert_eq!(
            map.get(&key, &guard),
            Some(&i),
            "key {key:?} not found"
        );
    }
}

// ---------------------------------------------------------------------------
// Bulk load
// ---------------------------------------------------------------------------

#[test]
fn string_bulk_load() {
    let mut pairs: Vec<(String, usize)> = (0..100)
        .map(|i| (format!("key_{i:04}"), i))
        .collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));

    let map = LearnedMap::bulk_load(&pairs).unwrap();
    let guard = map.guard();

    assert_eq!(map.len(), 100);
    for (k, v) in &pairs {
        assert_eq!(map.get(k, &guard), Some(v), "key {k:?} not found");
    }
}

#[test]
fn string_bulk_load_shared_prefix() {
    // Bulk load strings that share a long prefix.
    let prefix = "0123456789abcdef_long_prefix_";
    let mut pairs: Vec<(String, usize)> = (0..50)
        .map(|i| (format!("{prefix}{i:04}"), i))
        .collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));

    let map = LearnedMap::bulk_load(&pairs).unwrap();
    let guard = map.guard();

    assert_eq!(map.len(), 50);
    for (k, v) in &pairs {
        assert_eq!(map.get(k, &guard), Some(v), "key {k:?} not found");
    }
}

// ---------------------------------------------------------------------------
// Range queries — lexicographic order
// ---------------------------------------------------------------------------

#[test]
fn string_range_queries() {
    let mut pairs: Vec<(String, usize)> = (0..100)
        .map(|i| (format!("item_{i:04}"), i))
        .collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));

    let map = LearnedMap::bulk_load(&pairs).unwrap();
    let guard = map.guard();

    // Range from "item_0010" to "item_0020" (exclusive)
    let start = "item_0010".to_string();
    let end = "item_0020".to_string();
    let range_items: Vec<_> = map.range(start..end, &guard).collect();
    assert_eq!(range_items.len(), 10);

    // Verify order
    for pair in range_items.windows(2) {
        assert!(pair[0].0 < pair[1].0, "range not sorted");
    }
}

#[test]
fn string_first_last() {
    let mut pairs: Vec<(String, usize)> = vec![
        ("banana".into(), 1),
        ("apple".into(), 2),
        ("cherry".into(), 3),
    ];
    pairs.sort_by(|a, b| a.0.cmp(&b.0));

    let map = LearnedMap::bulk_load(&pairs).unwrap();
    let guard = map.guard();

    let (first_k, _) = map.first_key_value(&guard).unwrap();
    let (last_k, _) = map.last_key_value(&guard).unwrap();
    assert_eq!(first_k, &"apple".to_string());
    assert_eq!(last_k, &"cherry".to_string());
}

// ---------------------------------------------------------------------------
// Iteration — sorted order
// ---------------------------------------------------------------------------

#[test]
fn string_iter_sorted() {
    let map = LearnedMap::new();
    let guard = map.guard();

    let words = ["zebra", "alpha", "mango", "delta", "kiwi"];
    for (i, w) in words.iter().enumerate() {
        map.insert((*w).to_string(), i, &guard);
    }

    let items: Vec<_> = map.iter(&guard).collect();
    for pair in items.windows(2) {
        assert!(
            pair[0].0 < pair[1].0,
            "iteration not sorted: {:?} >= {:?}",
            pair[0].0,
            pair[1].0
        );
    }
}

// ---------------------------------------------------------------------------
// Rebuild and clear/drain
// ---------------------------------------------------------------------------

#[test]
fn string_rebuild() {
    let map = LearnedMap::new();
    let guard = map.guard();

    for i in 0..50u32 {
        map.insert(format!("key_{i:04}"), i, &guard);
    }

    map.rebuild(&guard);

    for i in 0..50u32 {
        let key = format!("key_{i:04}");
        assert_eq!(map.get(&key, &guard), Some(&i), "key {key:?} lost after rebuild");
    }
}

#[test]
fn string_clear_and_drain() {
    let map = LearnedMap::new();
    let guard = map.guard();

    for i in 0..10u32 {
        map.insert(format!("k{i}"), i, &guard);
    }

    let drained = map.drain(&guard);
    assert_eq!(drained.len(), 10);
    assert!(map.is_empty());

    // Drain should be sorted
    for pair in drained.windows(2) {
        assert!(pair[0].0 < pair[1].0);
    }
}

// ---------------------------------------------------------------------------
// Concurrent read/write with String keys
// ---------------------------------------------------------------------------

#[test]
fn string_concurrent_insert_get() {
    use std::sync::Arc;
    use std::thread;

    let map = Arc::new(LearnedMap::new());
    let n = 200;

    // Writer threads
    let mut handles = Vec::new();
    for t in 0..4u32 {
        let map = Arc::clone(&map);
        handles.push(thread::spawn(move || {
            let guard = map.guard();
            for i in 0..n {
                let key = format!("t{t}_k{i:04}");
                map.insert(key, t * 1000 + i, &guard);
            }
        }));
    }

    // Reader threads
    for t in 0..2u32 {
        let map = Arc::clone(&map);
        handles.push(thread::spawn(move || {
            let guard = map.guard();
            for i in 0..n {
                let key = format!("t{t}_k{i:04}");
                let _ = map.get(&key, &guard);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Verify all keys present
    let guard = map.guard();
    for t in 0..4u32 {
        for i in 0..n {
            let key = format!("t{t}_k{i:04}");
            assert!(
                map.get(&key, &guard).is_some(),
                "key {key:?} missing after concurrent inserts"
            );
        }
    }
}

#[test]
fn string_concurrent_shared_prefix() {
    use std::sync::Arc;
    use std::thread;

    let map = Arc::new(LearnedMap::new());
    let prefix = "shared_prefix_16"; // 16 bytes
    let n = 100;

    let mut handles = Vec::new();
    for t in 0..4u32 {
        let map = Arc::clone(&map);
        let prefix = prefix.to_string();
        handles.push(thread::spawn(move || {
            let guard = map.guard();
            for i in 0..n {
                let key = format!("{prefix}_t{t}_{i:04}");
                map.insert(key, i, &guard);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let guard = map.guard();
    for t in 0..4u32 {
        for i in 0..n {
            let key = format!("{prefix}_t{t}_{i:04}");
            assert!(
                map.get(&key, &guard).is_some(),
                "key {key:?} missing after concurrent shared-prefix inserts"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Vec<u8> keys — basic roundtrip
// ---------------------------------------------------------------------------

#[test]
fn vec_u8_insert_get_roundtrip() {
    let map = LearnedMap::new();
    let guard = map.guard();

    map.insert(vec![1, 2, 3], "a", &guard);
    map.insert(vec![4, 5, 6], "b", &guard);
    map.insert(vec![], "empty", &guard);

    assert_eq!(map.get(&vec![1, 2, 3], &guard), Some(&"a"));
    assert_eq!(map.get(&vec![4, 5, 6], &guard), Some(&"b"));
    assert_eq!(map.get(&vec![], &guard), Some(&"empty"));
    assert_eq!(map.get(&vec![7, 8, 9], &guard), None);
}

#[test]
fn vec_u8_bulk_load() {
    let mut pairs: Vec<(Vec<u8>, usize)> = (0..50u8)
        .map(|i| (vec![0, 0, i], i as usize))
        .collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));

    let map = LearnedMap::bulk_load(&pairs).unwrap();
    let guard = map.guard();

    for (k, v) in &pairs {
        assert_eq!(map.get(k, &guard), Some(v));
    }
}

// ---------------------------------------------------------------------------
// Property test: String LearnedMap matches BTreeMap
// ---------------------------------------------------------------------------

#[test]
fn string_matches_btreemap() {
    let map = LearnedMap::new();
    let mut btree = BTreeMap::new();
    let guard = map.guard();

    // Insert phase
    let keys: Vec<String> = (0..200)
        .map(|i| format!("key_{i:05}"))
        .collect();

    for (i, k) in keys.iter().enumerate() {
        map.insert(k.clone(), i, &guard);
        btree.insert(k.clone(), i);
    }

    // Lookup phase
    for k in &keys {
        assert_eq!(map.get(k, &guard), btree.get(k), "mismatch for {k:?}");
    }

    // Missing keys
    for i in 0..50 {
        let k = format!("missing_{i}");
        assert_eq!(map.get(&k, &guard), btree.get(&k));
    }

    // Remove phase
    for k in &keys[..50] {
        map.remove(k, &guard);
        btree.remove(k);
    }

    for k in &keys {
        assert_eq!(
            map.get(k, &guard),
            btree.get(k),
            "mismatch after remove for {k:?}"
        );
    }
}

#[test]
fn string_matches_btreemap_shared_prefix() {
    // Stress the ordinal collision path.
    let map = LearnedMap::new();
    let mut btree = BTreeMap::new();
    let guard = map.guard();

    let prefix = "0123456789ABCDEF"; // 16 bytes — ordinals collide
    for i in 0..100u32 {
        let k = format!("{prefix}_suffix_{i:06}");
        map.insert(k.clone(), i, &guard);
        btree.insert(k, i);
    }

    for (k, v) in &btree {
        assert_eq!(
            map.get(k, &guard),
            Some(v),
            "mismatch for {k:?}"
        );
    }

    // Iteration order must match BTreeMap
    let learned_keys: Vec<String> = map.iter(&guard).map(|(k, _)| k.clone()).collect();
    let btree_keys: Vec<String> = btree.keys().cloned().collect();
    assert_eq!(learned_keys, btree_keys, "iteration order mismatch");
}

// ---------------------------------------------------------------------------
// MapRef convenience API with String keys
// ---------------------------------------------------------------------------

#[test]
fn string_mapref() {
    let map = LearnedMap::new();
    let m = map.pin();

    m.insert("alpha".to_string(), 1);
    m.insert("beta".to_string(), 2);

    assert_eq!(m.get(&"alpha".to_string()), Some(&1));
    assert_eq!(m.get(&"beta".to_string()), Some(&2));
    assert!(m.contains_key(&"alpha".to_string()));
    assert!(!m.contains_key(&"gamma".to_string()));
    assert_eq!(m.len(), 2);
}

// ---------------------------------------------------------------------------
// Empty string key
// ---------------------------------------------------------------------------

#[test]
fn empty_string_key() {
    let map = LearnedMap::new();
    let guard = map.guard();

    map.insert(String::new(), 42, &guard);
    assert_eq!(map.get(&String::new(), &guard), Some(&42));
    assert_eq!(map.len(), 1);

    map.insert("a".to_string(), 1, &guard);
    assert_eq!(map.get(&String::new(), &guard), Some(&42));
    assert_eq!(map.get(&"a".to_string(), &guard), Some(&1));
}

// ---------------------------------------------------------------------------
// Bulk load with config
// ---------------------------------------------------------------------------

#[test]
fn string_bulk_load_with_config() {
    let config = Config::new().expansion_factor(3.0);
    let mut pairs: Vec<(String, usize)> = (0..50)
        .map(|i| (format!("item_{i:04}"), i))
        .collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));

    let map = LearnedMap::bulk_load_with_config(&pairs, config).unwrap();
    let guard = map.guard();

    for (k, v) in &pairs {
        assert_eq!(map.get(k, &guard), Some(v));
    }
}
