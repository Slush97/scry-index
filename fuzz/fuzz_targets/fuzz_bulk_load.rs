#![no_main]

use std::collections::BTreeMap;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use scry_index::LearnedMap;

#[derive(Arbitrary, Debug)]
struct Input {
    pairs: Vec<(u64, u64)>,
    use_dedup: bool,
}

fuzz_target!(|input: Input| {
    // Cap input size to keep each run fast.
    if input.pairs.len() > 1024 {
        return;
    }

    // Sort the pairs by key (required for bulk_load).
    let mut pairs = input.pairs;
    pairs.sort_by_key(|(k, _)| *k);

    let result = if input.use_dedup {
        // Dedup keeps last value for each key.
        LearnedMap::bulk_load_dedup(&pairs)
    } else {
        LearnedMap::bulk_load(&pairs)
    };

    match result {
        Ok(map) => {
            let guard = map.guard();

            // Build the oracle. For dedup, last writer wins (matching bulk_load_dedup).
            let oracle: BTreeMap<u64, u64> = if input.use_dedup {
                pairs.into_iter().collect()
            } else {
                // bulk_load succeeded, so there are no duplicate keys.
                pairs.into_iter().collect()
            };

            // Every key in the oracle must be in the map with the right value.
            for (k, v) in &oracle {
                let got = map.get(k, &guard);
                assert_eq!(got, Some(v), "missing or wrong value for key {k}");
            }

            // len should match.
            assert_eq!(map.len(), oracle.len());

            // Iteration order must match.
            let learned_sorted: Vec<_> = map
                .iter(&guard)
                .map(|(k, v)| (*k, *v))
                .collect();
            let oracle_sorted: Vec<_> = oracle.into_iter().collect();
            assert_eq!(learned_sorted, oracle_sorted);
        }
        Err(_) => {
            // bulk_load (non-dedup) rejects duplicates — verify that duplicates exist.
            if !input.use_dedup {
                let has_dupes = pairs.windows(2).any(|w| w[0].0 == w[1].0);
                assert!(
                    has_dupes || pairs.is_empty(),
                    "bulk_load failed but input has no duplicates"
                );
            }
        }
    }
});
