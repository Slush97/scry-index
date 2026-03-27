#![no_main]

use std::collections::BTreeMap;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use scry_index::LearnedMap;

/// An operation the fuzzer can perform on the map.
#[derive(Arbitrary, Debug)]
enum Op {
    Insert { key: u64, value: u64 },
    Get { key: u64 },
    Remove { key: u64 },
    GetOrInsert { key: u64, value: u64 },
    ContainsKey { key: u64 },
    Len,
    RangeCount { lo: u64, hi: u64 },
    FirstKeyValue,
    LastKeyValue,
    Iter,
}

fuzz_target!(|ops: Vec<Op>| {
    // Cap sequence length to keep each run fast.
    if ops.len() > 512 {
        return;
    }

    let learned = LearnedMap::new();
    let guard = learned.guard();
    let mut oracle = BTreeMap::new();

    for op in &ops {
        match op {
            Op::Insert { key, value } => {
                let l_new = learned.insert(*key, *value, &guard);
                let b_prev = oracle.insert(*key, *value);
                // Both should agree on whether the key was new.
                assert_eq!(l_new, b_prev.is_none());
            }
            Op::Get { key } => {
                let l_val = learned.get(key, &guard);
                let b_val = oracle.get(key);
                assert_eq!(l_val, b_val);
            }
            Op::Remove { key } => {
                let l_removed = learned.remove(key, &guard);
                let b_removed = oracle.remove(key).is_some();
                assert_eq!(l_removed, b_removed);
            }
            Op::GetOrInsert { key, value } => {
                let l_val = learned.get_or_insert(*key, *value, &guard);
                let b_val = oracle.entry(*key).or_insert(*value);
                assert_eq!(l_val, b_val);
            }
            Op::ContainsKey { key } => {
                let l = learned.contains_key(key, &guard);
                let b = oracle.contains_key(key);
                assert_eq!(l, b);
            }
            Op::Len => {
                // len() is documented as approximate under concurrency,
                // but single-threaded it should be exact.
                assert_eq!(learned.len(), oracle.len());
            }
            Op::RangeCount { lo, hi } => {
                if lo <= hi {
                    let l = learned.range_count(*lo..=*hi, &guard);
                    let b = oracle.range(*lo..=*hi).count();
                    assert_eq!(l, b);
                }
            }
            Op::FirstKeyValue => {
                let l = learned.first_key_value(&guard);
                let b = oracle.iter().next().map(|(k, v)| (k, v));
                match (l, b) {
                    (Some((lk, lv)), Some((bk, bv))) => {
                        assert_eq!(lk, bk);
                        assert_eq!(lv, bv);
                    }
                    (None, None) => {}
                    _ => panic!("first_key_value mismatch"),
                }
            }
            Op::LastKeyValue => {
                let l = learned.last_key_value(&guard);
                let b = oracle.iter().next_back().map(|(k, v)| (k, v));
                match (l, b) {
                    (Some((lk, lv)), Some((bk, bv))) => {
                        assert_eq!(lk, bk);
                        assert_eq!(lv, bv);
                    }
                    (None, None) => {}
                    _ => panic!("last_key_value mismatch"),
                }
            }
            Op::Iter => {
                let learned_sorted: Vec<_> = learned
                    .iter(&guard)
                    .map(|(k, v)| (*k, *v))
                    .collect();
                let oracle_sorted: Vec<_> = oracle
                    .iter()
                    .map(|(k, v)| (*k, *v))
                    .collect();
                assert_eq!(learned_sorted, oracle_sorted);
            }
        }
    }
});
