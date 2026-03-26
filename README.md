# scry-index

A concurrent sorted key-value map backed by learned index structures.

Uses piecewise linear models (LIPP chain method) to predict key positions,
giving lock-free reads and CAS-based writes with no global lock. Based on
the [SALI paper](https://people.iiis.tsinghua.edu.cn/~huanchen/publications/sali-sigmod24.pdf) (SIGMOD 2024).

## Usage

```rust
use scry_index::LearnedMap;

let map = LearnedMap::new();
let m = map.pin();

m.insert(42u64, "the answer");
m.insert(1u64, "first");
m.insert(100u64, "last");

assert_eq!(m.get(&42), Some(&"the answer"));

for (k, v) in m.range(1u64..50) {
    println!("{k}: {v}");
}
```

## Tradeoffs

Read-optimized. Lookups are dramatically faster; inserts are moderately slower
than `BTreeMap`.

| Operation | LearnedMap | BTreeMap | |
|---|---|---|---|
| Point lookup (100K, sequential) | 0.99 ms | 6.27 ms | **6.3x faster** |
| Point lookup (100K, random) | 1.13 ms | 11.94 ms | **10.5x faster** |
| Point lookup (500K, random) | 11.4 ms | 86.0 ms | **7.5x faster** |
| Sequential insert (100K) | 14.7 ms | 9.7 ms | 1.5x slower |
| Bulk load (100K) | 3.5 ms | 1.9 ms | 1.9x slower |

The O(1) model-predicted lookup vs O(log n) tree walk advantage grows with
scale and is especially pronounced under random access patterns where
BTreeMap suffers cache misses.

Good for read-heavy, concurrent workloads with sorted keys (time-series
queries, lookup tables, analytics indexes). Not the right choice if writes
dominate.

```sh
cargo bench                                      # criterion microbenchmarks
cargo run --example mini_bench --release         # quick comparison
cargo run --example simulate --release           # time-series workload
```

## Features

- Lock-free reads via epoch-based reclamation
- CAS-based writes, per-slot contention only
- Sorted iteration and range queries (`a..b`, `a..=b`, `a..`, `..b`, `..`)
- Bulk loading from sorted data
- Depth-triggered localized subtree rebuilds
- Tombstone compaction for remove-heavy workloads
- `get_or_insert` / `get_or_insert_with` entry API
- `LearnedSet` wrapper for key-only use
- Optional serde support (`features = ["serde"]`)

## Key types

Integers, `[u8; N]`, `String`, and `Vec<u8>` implement `Key` out of the box.
Implement the `Key` trait for custom types.

## Configuration

```rust
use scry_index::{Config, LearnedMap};

let config = Config::new()
    .expansion_factor(2.5)
    .rebuild_depth_threshold(10)
    .tombstone_ratio_threshold(0.3);

let map: LearnedMap<u64, String> = LearnedMap::with_config(config);
```

See `Config` docs for defaults and details.

## Minimum supported Rust version

1.83.0

## License

MIT OR Apache-2.0
