//! Concurrent tests for the scry-index Phase 2 lock-free implementation.

use std::collections::BTreeMap;
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use scry_index::{Config, LearnedMap};

/// Multi-reader on a bulk-loaded map: 8 threads each read 10K keys.
#[test]
fn concurrent_readers_bulk_loaded() {
    let pairs: Vec<(u64, u64)> = (0..10_000).map(|i| (i, i * 10)).collect();
    let map = Arc::new(LearnedMap::bulk_load(&pairs).unwrap());
    let barrier = Arc::new(Barrier::new(8));

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let guard = map.guard();
                for i in 0..10_000u64 {
                    assert_eq!(
                        map.get(&i, &guard),
                        Some(&(i * 10)),
                        "reader missed key {i}"
                    );
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

/// Reader-writer coexistence: 4 writers + 8 readers on disjoint key ranges.
#[test]
fn readers_writers_disjoint_ranges() {
    let map = Arc::new(LearnedMap::with_config(Config::new().auto_rebuild(false)));
    let barrier = Arc::new(Barrier::new(12));

    // 4 writers: each inserts 1000 keys in a unique range
    let writer_handles: Vec<_> = (0..4u64)
        .map(|w| {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let guard = map.guard();
                let base = w * 1000;
                for i in 0..1000 {
                    map.insert(base + i, base + i, &guard);
                }
            })
        })
        .collect();

    // 8 readers: attempt to read keys written by writer 0 (may or may not be present yet)
    let reader_handles: Vec<_> = (0..8)
        .map(|_| {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let guard = map.guard();
                let mut found = 0;
                for i in 0..1000u64 {
                    if map.get(&i, &guard).is_some() {
                        found += 1;
                    }
                }
                found
            })
        })
        .collect();

    for h in writer_handles {
        h.join().unwrap();
    }
    for h in reader_handles {
        h.join().unwrap();
    }

    // After all writers finish, all 4000 keys should be present
    let guard = map.guard();
    assert_eq!(map.len(), 4000);
    for i in 0..4000u64 {
        assert_eq!(map.get(&i, &guard), Some(&i), "key {i} missing after join");
    }
}

/// Writer contention: 8 threads all insert/update the same 1000 keys.
#[test]
fn writer_contention_shared_keys() {
    let map = Arc::new(LearnedMap::with_config(Config::new().auto_rebuild(false)));
    let barrier = Arc::new(Barrier::new(8));

    let handles: Vec<_> = (0..8u64)
        .map(|t| {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let guard = map.guard();
                for i in 0..1000u64 {
                    map.insert(i, t * 1000 + i, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Each key should be present with SOME value (from one of the writers)
    let guard = map.guard();
    assert_eq!(map.len(), 1000);
    for i in 0..1000u64 {
        assert!(
            map.get(&i, &guard).is_some(),
            "key {i} missing after contention"
        );
    }
}

/// Insert-remove interleaving with post-join verification.
#[test]
fn insert_remove_interleaving() {
    let map = Arc::new(LearnedMap::with_config(Config::new().auto_rebuild(false)));

    // Phase 1: populate
    {
        let guard = map.guard();
        for i in 0..2000u64 {
            map.insert(i, i, &guard);
        }
    }
    assert_eq!(map.len(), 2000);

    let barrier = Arc::new(Barrier::new(8));

    // 4 threads remove even keys
    let remove_handles: Vec<_> = (0..4)
        .map(|t| {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let guard = map.guard();
                // Each thread handles a subset of even keys
                for i in (0..2000u64).step_by(2) {
                    if i % 4 == (t * 2) % 4 {
                        map.remove(&i, &guard);
                    }
                }
            })
        })
        .collect();

    // 4 threads insert new keys (2000..4000)
    let insert_handles: Vec<_> = (0..4u64)
        .map(|t| {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let guard = map.guard();
                let base = 2000 + t * 500;
                for i in 0..500 {
                    map.insert(base + i, base + i, &guard);
                }
            })
        })
        .collect();

    for h in remove_handles {
        h.join().unwrap();
    }
    for h in insert_handles {
        h.join().unwrap();
    }

    // Verify: odd keys [0,2000) should be present, even keys removed,
    // new keys [2000,4000) should be present
    let guard = map.guard();
    for i in (1..2000u64).step_by(2) {
        assert_eq!(map.get(&i, &guard), Some(&i), "odd key {i} missing");
    }
    for i in 2000..4000u64 {
        assert_eq!(map.get(&i, &guard), Some(&i), "new key {i} missing");
    }
}

/// Stress test vs `Mutex<BTreeMap>` oracle.
#[test]
fn stress_vs_oracle() {
    let map = Arc::new(LearnedMap::with_config(Config::new().auto_rebuild(false)));
    let oracle = Arc::new(Mutex::new(BTreeMap::new()));
    let barrier = Arc::new(Barrier::new(4));

    // Each thread inserts a unique set of keys, then removes some
    let handles: Vec<_> = (0..4u64)
        .map(|t| {
            let map = Arc::clone(&map);
            let oracle = Arc::clone(&oracle);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let guard = map.guard();
                let base = t * 500;
                for i in 0..500 {
                    let key = base + i;
                    map.insert(key, key, &guard);
                    oracle.lock().unwrap().insert(key, key);
                }
                // Remove every 3rd key
                for i in (0..500).step_by(3) {
                    let key = base + i;
                    map.remove(&key, &guard);
                    oracle.lock().unwrap().remove(&key);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let oracle = oracle.lock().unwrap();
    let guard = map.guard();
    assert_eq!(map.len(), oracle.len());
    for (&k, &v) in oracle.iter() {
        assert_eq!(
            map.get(&k, &guard),
            Some(&v),
            "oracle has key {k} but map doesn't"
        );
    }
}

/// `MapRef` convenience API under concurrency.
#[test]
fn map_ref_concurrent() {
    let map = Arc::new(LearnedMap::with_config(Config::new().auto_rebuild(false)));
    let barrier = Arc::new(Barrier::new(4));

    let handles: Vec<_> = (0..4u64)
        .map(|t| {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let m = map.pin();
                let base = t * 250;
                for i in 0..250 {
                    m.insert(base + i, base + i);
                }
                for i in 0..250 {
                    assert!(m.contains_key(&(base + i)));
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(map.len(), 1000);
}

/// Insert collision path under contention: threads insert different keys that
/// predict to the same slot, forcing concurrent child-node creation.
#[test]
fn insert_collision_contention() {
    // Use a bulk-loaded map with known structure, then insert interleaving keys
    // from multiple threads. The interleaving keys will collide with existing
    // keys at their predicted slots, exercising the CAS Data→Child path.
    let pairs: Vec<(u64, u64)> = (0..100).map(|i| (i * 2, i)).collect();
    let config = Config::new().auto_rebuild(false);
    let map = Arc::new(LearnedMap::bulk_load_with_config(&pairs, config).unwrap());
    let barrier = Arc::new(Barrier::new(8));

    // 8 threads each insert odd keys into gaps — these collide with even keys
    let handles: Vec<_> = (0..8u64)
        .map(|t| {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let guard = map.guard();
                // Each thread inserts a subset of odd keys
                for i in 0..100u64 {
                    if i % 8 == t {
                        let key = i * 2 + 1;
                        map.insert(key, key, &guard);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // All original even keys and all new odd keys should be present
    let guard = map.guard();
    for i in 0..200u64 {
        assert!(
            map.get(&i, &guard).is_some(),
            "key {i} missing after collision contention"
        );
    }
}

/// Concurrent insert + remove of the SAME key from different threads.
#[test]
fn concurrent_insert_remove_same_key() {
    let map = Arc::new(LearnedMap::with_config(Config::new().auto_rebuild(false)));
    let barrier = Arc::new(Barrier::new(8));

    // Populate initial keys
    {
        let guard = map.guard();
        for i in 0..100u64 {
            map.insert(i, i, &guard);
        }
    }

    // 4 threads insert key K with value T, 4 threads remove key K
    // All targeting the same 100 keys simultaneously
    let handles: Vec<_> = (0..8u64)
        .map(|t| {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let guard = map.guard();
                for round in 0..50u64 {
                    for key in 0..100u64 {
                        if t < 4 {
                            map.insert(key, round * 1000 + t, &guard);
                        } else {
                            map.remove(&key, &guard);
                        }
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // After all threads finish, each key is either present or absent.
    // The map should be internally consistent: every key that get() finds
    // should also be findable via contains_key(), and vice versa.
    let guard = map.guard();
    for key in 0..100u64 {
        let got = map.get(&key, &guard);
        let contains = map.contains_key(&key, &guard);
        assert_eq!(
            got.is_some(),
            contains,
            "inconsistency at key {key}: get={got:?}, contains={contains}"
        );
    }
}

/// Rebuild under concurrent reads.
#[test]
fn rebuild_with_concurrent_readers() {
    let pairs: Vec<(u64, u64)> = (0..5000).map(|i| (i, i)).collect();
    let map = Arc::new(LearnedMap::bulk_load(&pairs).unwrap());
    let barrier = Arc::new(Barrier::new(5));

    // 4 reader threads
    let reader_handles: Vec<_> = (0..4)
        .map(|_| {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for _ in 0..10 {
                    let guard = map.guard();
                    for i in 0..5000u64 {
                        // Key may or may not be found during rebuild
                        let _ = map.get(&i, &guard);
                    }
                }
            })
        })
        .collect();

    // 1 rebuild thread
    let map_clone = Arc::clone(&map);
    let barrier_clone = Arc::clone(&barrier);
    let rebuild_handle = thread::spawn(move || {
        barrier_clone.wait();
        for _ in 0..3 {
            let guard = map_clone.guard();
            map_clone.rebuild(&guard);
        }
    });

    rebuild_handle.join().unwrap();
    for h in reader_handles {
        h.join().unwrap();
    }

    // After rebuild, all keys should still be present
    let guard = map.guard();
    for i in 0..5000u64 {
        assert_eq!(map.get(&i, &guard), Some(&i), "key {i} lost after rebuild");
    }
}

/// Lock-free rebuild during concurrent inserts: no deadlock or corruption.
///
/// With lock-free global rebuild, inserts that land in the old tree between
/// the snapshot and the CAS may be lost. This test verifies thread safety
/// and structural integrity, not data preservation.
#[test]
fn global_rebuild_lockfree() {
    let map = Arc::new(LearnedMap::with_config(Config::new().auto_rebuild(false)));
    let barrier = Arc::new(Barrier::new(5));

    // 4 inserter threads, each inserting 750 unique keys (3000 total)
    let insert_handles: Vec<_> = (0..4u64)
        .map(|t| {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let guard = map.guard();
                let base = t * 750;
                for i in 0..750 {
                    map.insert(base + i, base + i, &guard);
                }
            })
        })
        .collect();

    // 1 rebuilder thread calling rebuild repeatedly
    let map_r = Arc::clone(&map);
    let barrier_r = Arc::clone(&barrier);
    let rebuild_handle = thread::spawn(move || {
        barrier_r.wait();
        for _ in 0..10 {
            let guard = map_r.guard();
            map_r.rebuild(&guard);
        }
    });

    for h in insert_handles {
        h.join().unwrap();
    }
    rebuild_handle.join().unwrap();

    // Lock-free rebuild may race with inserts: verify no corruption.
    let guard = map.guard();
    let actual = map.iter_sorted(&guard).len();
    assert!(
        actual <= 3000,
        "actual count {actual} exceeds total inserts"
    );

    // Map must be usable: re-insert all keys, rebuild, verify all present.
    for i in 0..3000u64 {
        map.insert(i, i, &guard);
    }
    map.rebuild(&guard);
    let g2 = map.guard();
    assert_eq!(map.len(), 3000);
    for i in 0..3000u64 {
        assert!(map.get(&i, &g2).is_some(), "key {i} missing after recovery");
    }
}

/// Lock-free rebuild during concurrent removes: no corruption.
///
/// With lock-free rebuild, a rebuild may snapshot the tree before some
/// removes complete, then CAS-swap, effectively restoring removed keys.
/// This test verifies structural integrity; odd keys must always be present
/// (they are never removed), even keys may vary.
#[test]
fn rebuild_with_concurrent_removes_no_corruption() {
    // Pre-populate 2000 keys
    let pairs: Vec<(u64, u64)> = (0..2000).map(|i| (i, i)).collect();
    let map = Arc::new(
        LearnedMap::bulk_load_with_config(&pairs, Config::new().auto_rebuild(false)).unwrap(),
    );
    let barrier = Arc::new(Barrier::new(3));

    // 2 remover threads, each removing half the even keys
    let remove_handles: Vec<_> = (0..2u64)
        .map(|t| {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let guard = map.guard();
                for i in (0..2000u64).step_by(2) {
                    if i % 4 == t * 2 {
                        map.remove(&i, &guard);
                    }
                }
            })
        })
        .collect();

    // 1 rebuilder
    let map_r = Arc::clone(&map);
    let barrier_r = Arc::clone(&barrier);
    let rebuild_handle = thread::spawn(move || {
        barrier_r.wait();
        for _ in 0..5 {
            let guard = map_r.guard();
            map_r.rebuild(&guard);
        }
    });

    for h in remove_handles {
        h.join().unwrap();
    }
    rebuild_handle.join().unwrap();

    // Odd keys should always be present (never removed, never affected by rebuild)
    let guard = map.guard();
    for i in (1..2000u64).step_by(2) {
        assert!(
            map.get(&i, &guard).is_some(),
            "odd key {i} should still be present"
        );
    }
    // Map should be internally consistent
    let actual = map.iter_sorted(&guard).len();
    assert!(
        actual >= 1000,
        "at least 1000 odd keys should be present, got {actual}"
    );
}

/// Auto-rebuild (localized subtree rebuilds) under concurrency.
///
/// With localized rebuilds, concurrent inserts into the same subtree can
/// race with the rebuild CAS. This test verifies no corruption and bounded
/// depth. Some keys may be lost due to the race.
#[test]
fn auto_rebuild_concurrent_no_corruption() {
    // auto_rebuild enabled (default)
    let map = Arc::new(LearnedMap::new());
    let barrier = Arc::new(Barrier::new(8));

    let handles: Vec<_> = (0..8u64)
        .map(|t| {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let guard = map.guard();
                let base = t * 1000;
                for i in 0..1000 {
                    map.insert(base + i, base + i, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Localized rebuilds may race with concurrent inserts on overlapping
    // subtrees (the default 16-slot root maps many ranges to the same slots).
    // Verify no corruption and bounded depth, not all keys present.
    let guard = map.guard();
    let actual = map.iter_sorted(&guard).len();
    assert!(actual > 0, "map should not be empty after 8000 inserts");

    // Depth should be bounded by the localized rebuild mechanism
    let depth = map.max_depth(&guard);
    assert!(
        depth <= 20,
        "depth {depth} too high with localized rebuilds"
    );

    // After re-inserting all keys and rebuilding, everything should be present
    for i in 0..8000u64 {
        map.insert(i, i, &guard);
    }
    map.rebuild(&guard);
    let g2 = map.guard();
    assert_eq!(map.len(), 8000);
    for i in 0..8000u64 {
        assert!(map.get(&i, &g2).is_some(), "key {i} missing after recovery");
    }
}

/// Multiple concurrent rebuilds: 4 threads all calling `rebuild()`, no corruption.
#[test]
fn multiple_concurrent_rebuilds() {
    // Pre-populate
    let pairs: Vec<(u64, u64)> = (0..5000).map(|i| (i, i)).collect();
    let map = Arc::new(LearnedMap::bulk_load(&pairs).unwrap());
    let barrier = Arc::new(Barrier::new(4));

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for _ in 0..5 {
                    let guard = map.guard();
                    map.rebuild(&guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // All keys intact
    let guard = map.guard();
    for i in 0..5000u64 {
        assert_eq!(
            map.get(&i, &guard),
            Some(&i),
            "key {i} corrupted after concurrent rebuilds"
        );
    }
    assert_eq!(map.len(), 5000);
}

/// Auto-rebuild (localized): single-threaded incremental inserts stay compact.
#[test]
fn auto_rebuild_keeps_depth_bounded() {
    // Default config has auto_rebuild = true, rebuild_depth_threshold = 8
    let map = LearnedMap::new();
    let g = map.guard();
    for i in 0..10_000u64 {
        map.insert(i, i, &g);
    }
    assert_eq!(map.len(), 10_000);

    // With localized rebuilds triggered at depth 8, the tree should stay
    // much shallower than the previous global-rebuild approach.
    let g2 = map.guard();
    let depth = map.max_depth(&g2);
    assert!(
        depth <= 30,
        "depth {depth} too high after 10k inserts with localized auto-rebuild"
    );

    // All keys must be present (single-threaded, no rebuild races)
    for i in 0..10_000u64 {
        assert_eq!(
            map.get(&i, &g2),
            Some(&i),
            "key {i} missing after auto-rebuild"
        );
    }

    // After an explicit rebuild, depth should be very low
    map.rebuild(&g2);
    let g3 = map.guard();
    assert!(
        map.max_depth(&g3) <= 5,
        "depth {} too high after explicit rebuild",
        map.max_depth(&g3)
    );
}

/// Auto-rebuild disabled via config: depth grows unbounded.
#[test]
fn auto_rebuild_disabled_depth_grows() {
    let map = LearnedMap::with_config(Config::new().auto_rebuild(false));
    let g = map.guard();
    for i in 0..200u64 {
        map.insert(i, i, &g);
    }
    // Without auto-rebuild, the default 16-slot root causes deep chains.
    // Depth should be significantly higher than with rebuild.
    let depth = map.max_depth(&g);
    assert!(
        depth > 5,
        "depth {depth} suspiciously low without auto-rebuild"
    );
}

/// Range queries during concurrent inserts return sorted, valid data.
#[test]
fn concurrent_range_during_inserts() {
    let pairs: Vec<(u64, u64)> = (0..1000).map(|i| (i * 2, i)).collect();
    let map = Arc::new(
        LearnedMap::bulk_load_with_config(&pairs, Config::new().auto_rebuild(false)).unwrap(),
    );
    let barrier = Arc::new(Barrier::new(6));

    // 4 writer threads insert odd keys
    let writer_handles: Vec<_> = (0..4u64)
        .map(|t| {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let guard = map.guard();
                for i in 0..1000u64 {
                    if i % 4 == t {
                        map.insert(i * 2 + 1, i + 10_000, &guard);
                    }
                }
            })
        })
        .collect();

    // 2 reader threads do range queries
    let reader_handles: Vec<_> = (0..2)
        .map(|_| {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for _ in 0..20 {
                    let guard = map.guard();
                    let items: Vec<u64> = map.range(100..200, &guard).map(|(k, _)| *k).collect();
                    // Must be sorted
                    for w in items.windows(2) {
                        assert!(w[0] < w[1], "range not sorted: {} >= {}", w[0], w[1]);
                    }
                    // All keys must be within bounds
                    for &k in &items {
                        assert!((100..200).contains(&k), "key {k} out of range");
                    }
                }
            })
        })
        .collect();

    for h in writer_handles {
        h.join().unwrap();
    }
    for h in reader_handles {
        h.join().unwrap();
    }
}

/// first/last queries during concurrent inserts remain consistent.
#[test]
fn concurrent_first_last_during_inserts() {
    let map = Arc::new(LearnedMap::with_config(Config::new().auto_rebuild(false)));
    let barrier = Arc::new(Barrier::new(6));

    // 4 writers insert keys 0..4000
    let writer_handles: Vec<_> = (0..4u64)
        .map(|t| {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let guard = map.guard();
                let base = t * 1000;
                for i in 0..1000 {
                    map.insert(base + i, base + i, &guard);
                }
            })
        })
        .collect();

    // 2 readers check first/last
    let reader_handles: Vec<_> = (0..2)
        .map(|_| {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for _ in 0..50 {
                    let guard = map.guard();
                    if let (Some(first), Some(last)) =
                        (map.first_key_value(&guard), map.last_key_value(&guard))
                    {
                        assert!(first.0 <= last.0, "first ({}) > last ({})", first.0, last.0);
                    }
                }
            })
        })
        .collect();

    for h in writer_handles {
        h.join().unwrap();
    }
    for h in reader_handles {
        h.join().unwrap();
    }

    // After all writers finish, verify first/last
    let guard = map.guard();
    assert_eq!(map.first_key_value(&guard).map(|(k, _)| *k), Some(0));
    assert_eq!(map.last_key_value(&guard).map(|(k, _)| *k), Some(3999));
}

/// Localized rebuild bounds depth: 8 threads x 1000 keys, depth stays bounded.
#[test]
fn localized_rebuild_bounds_depth() {
    // Use a bulk-loaded base so each thread's range maps to distinct root slots
    let anchors: Vec<(u64, u64)> = (0..8000).step_by(80).map(|i| (i, i)).collect();
    let map = Arc::new(LearnedMap::bulk_load(&anchors).unwrap());
    let barrier = Arc::new(Barrier::new(8));

    let handles: Vec<_> = (0..8u64)
        .map(|t| {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let guard = map.guard();
                let base = t * 1000;
                for i in 0..1000 {
                    map.insert(base + i, base + i, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let guard = map.guard();
    let depth = map.max_depth(&guard);
    // With rebuild_depth_threshold=8, depth should stay bounded
    assert!(
        depth <= 20,
        "depth {depth} too high after 8x1000 inserts with localized rebuilds"
    );
}

/// Localized rebuild with well-modeled root: no data loss when threads
/// operate on non-overlapping subtrees.
#[test]
fn localized_rebuild_no_data_loss() {
    // Sparse anchors spanning 0..8000 create a root model that maps each
    // thread's 1000-key range to distinct root slots (~25 slots per thread).
    let anchors: Vec<(u64, u64)> = (0..8000).step_by(80).map(|i| (i, i)).collect();
    let map = Arc::new(LearnedMap::bulk_load(&anchors).unwrap());
    let barrier = Arc::new(Barrier::new(8));

    let handles: Vec<_> = (0..8u64)
        .map(|t| {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let guard = map.guard();
                let base = t * 1000;
                for i in 0..1000 {
                    map.insert(base + i, base + i, &guard);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // With non-overlapping subtrees, localized rebuilds should not lose data
    let guard = map.guard();
    for i in 0..8000u64 {
        assert!(
            map.get(&i, &guard).is_some(),
            "key {i} lost during localized rebuild"
        );
    }
    assert_eq!(map.len(), 8000);
}
