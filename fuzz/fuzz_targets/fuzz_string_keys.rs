#![no_main]

use std::collections::BTreeMap;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use scry_index::LearnedMap;

/// Operations on a `LearnedMap<String, u64>` to exercise the variable-length
/// key path (prefix-based model input, ordinal fallback for shared prefixes).
#[derive(Arbitrary, Debug)]
enum Op {
    Insert { key: String, value: u64 },
    Get { key: String },
    Remove { key: String },
    Iter,
}

fuzz_target!(|ops: Vec<Op>| {
    if ops.len() > 256 {
        return;
    }

    let learned: LearnedMap<String, u64> = LearnedMap::new();
    let guard = learned.guard();
    let mut oracle = BTreeMap::new();

    for op in &ops {
        match op {
            Op::Insert { key, value } => {
                let l_new = learned.insert(key.clone(), *value, &guard);
                let b_prev = oracle.insert(key.clone(), *value);
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
            Op::Iter => {
                let learned_sorted: Vec<_> = learned
                    .iter(&guard)
                    .map(|(k, v)| (k.clone(), *v))
                    .collect();
                let oracle_sorted: Vec<_> = oracle
                    .iter()
                    .map(|(k, v)| (k.clone(), *v))
                    .collect();
                assert_eq!(learned_sorted, oracle_sorted);
            }
        }
    }
});
