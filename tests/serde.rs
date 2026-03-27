#![allow(missing_docs)]
#![cfg(feature = "serde")]

use scry_index::{Config, Error, LearnedMap, LearnedSet};

#[test]
fn map_round_trip_json() {
    let pairs: Vec<(u64, u64)> = (0..100).map(|i| (i, i * 10)).collect();
    let map = LearnedMap::bulk_load(&pairs).unwrap();

    let json = serde_json::to_string(&map).unwrap();
    let restored: LearnedMap<u64, u64> = serde_json::from_str(&json).unwrap();

    let g = restored.guard();
    assert_eq!(restored.len(), 100);
    for (k, v) in &pairs {
        assert_eq!(restored.get(k, &g), Some(v));
    }
}

#[test]
fn map_empty_round_trip() {
    let map = LearnedMap::<u64, u64>::new();
    let json = serde_json::to_string(&map).unwrap();
    assert_eq!(json, "[]");

    let restored: LearnedMap<u64, u64> = serde_json::from_str(&json).unwrap();
    assert!(restored.is_empty());
}

#[test]
fn map_deserialized_supports_lookups_and_ranges() {
    let pairs: Vec<(u64, &str)> = vec![(10, "a"), (20, "b"), (30, "c"), (40, "d"), (50, "e")];
    let map = LearnedMap::bulk_load(&pairs).unwrap();

    let json = serde_json::to_string(&map).unwrap();
    let restored: LearnedMap<u64, &str> = serde_json::from_str(&json).unwrap();

    let g = restored.guard();
    // Point lookups
    assert_eq!(restored.get(&20, &g), Some(&"b"));
    assert_eq!(restored.get(&99, &g), None);

    // Range query
    let range_items: Vec<_> = restored.range(15..=35, &g).collect();
    assert_eq!(range_items, vec![(&20, &"b"), (&30, &"c")]);

    // First/last
    assert_eq!(restored.first_key_value(&g), Some((&10, &"a")));
    assert_eq!(restored.last_key_value(&g), Some((&50, &"e")));
}

#[test]
fn map_string_keys_round_trip() {
    let map = LearnedMap::new();
    let g = map.guard();
    map.insert("alpha".to_string(), 1u64, &g);
    map.insert("beta".to_string(), 2, &g);
    map.insert("gamma".to_string(), 3, &g);

    let json = serde_json::to_string(&map).unwrap();
    let restored: LearnedMap<String, u64> = serde_json::from_str(&json).unwrap();

    let g2 = restored.guard();
    assert_eq!(restored.len(), 3);
    assert_eq!(restored.get(&"alpha".to_string(), &g2), Some(&1));
    assert_eq!(restored.get(&"beta".to_string(), &g2), Some(&2));
    assert_eq!(restored.get(&"gamma".to_string(), &g2), Some(&3));
}

#[test]
fn set_round_trip_json() {
    let keys: Vec<u64> = (0..50).collect();
    let set = LearnedSet::bulk_load(&keys).unwrap();

    let json = serde_json::to_string(&set).unwrap();
    let restored: LearnedSet<u64> = serde_json::from_str(&json).unwrap();

    let g = restored.guard();
    assert_eq!(restored.len(), 50);
    for k in &keys {
        assert!(restored.contains(k, &g));
    }
}

#[test]
fn set_empty_round_trip() {
    let set = LearnedSet::<u64>::new();
    let json = serde_json::to_string(&set).unwrap();
    assert_eq!(json, "[]");

    let restored: LearnedSet<u64> = serde_json::from_str(&json).unwrap();
    assert!(restored.is_empty());
}

#[test]
fn config_round_trip_json() {
    let config = Config::new()
        .expansion_factor(3.0)
        .auto_rebuild(false)
        .rebuild_depth_threshold(12)
        .tombstone_ratio_threshold(0.75)
        .range_headroom(0.5);

    let json = serde_json::to_string(&config).unwrap();
    let restored: Config = serde_json::from_str(&json).unwrap();

    assert!((restored.expansion_factor - 3.0).abs() < f64::EPSILON);
    assert!(!restored.auto_rebuild);
    assert_eq!(restored.rebuild_depth_threshold, 12);
    assert!((restored.tombstone_ratio_threshold - 0.75).abs() < f64::EPSILON);
    assert!((restored.range_headroom - 0.5).abs() < f64::EPSILON);
}

#[test]
fn error_round_trip_json() {
    let errors = vec![
        Error::EmptyData,
        Error::NotSorted,
        Error::InvalidKey {
            detail: "overflow".into(),
        },
    ];

    for err in &errors {
        let json = serde_json::to_string(err).unwrap();
        let restored: Error = serde_json::from_str(&json).unwrap();
        assert_eq!(&restored, err);
    }
}

#[test]
fn map_incremental_inserts_round_trip() {
    let map = LearnedMap::new();
    let g = map.guard();
    for i in (0..50u64).rev() {
        map.insert(i, format!("val-{i}"), &g);
    }

    let json = serde_json::to_string(&map).unwrap();
    let restored: LearnedMap<u64, String> = serde_json::from_str(&json).unwrap();

    let g2 = restored.guard();
    assert_eq!(restored.len(), 50);
    for i in 0..50u64 {
        assert_eq!(
            restored.get(&i, &g2),
            Some(&format!("val-{i}")),
            "key {i} mismatch"
        );
    }
}
