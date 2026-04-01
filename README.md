# scry-index

[![Crates.io](https://img.shields.io/crates/v/scry-index)](https://crates.io/crates/scry-index)
[![docs.rs](https://img.shields.io/docsrs/scry-index)](https://docs.rs/scry-index)
[![CI](https://github.com/slush97/scry-index/actions/workflows/ci.yml/badge.svg)](https://github.com/slush97/scry-index/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![MSRV](https://img.shields.io/badge/MSRV-1.83-orange.svg)](https://www.rust-lang.org)

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

All operations take `&self` — share across threads with `Arc`:

```rust
use std::sync::Arc;
use scry_index::LearnedMap;

let map = Arc::new(LearnedMap::new());

std::thread::scope(|s| {
    // Writers
    for t in 0..4 {
        let map = Arc::clone(&map);
        s.spawn(move || {
            let m = map.pin();
            for i in 0..1000u64 {
                m.insert(t * 1000 + i, i);
            }
        });
    }
    // Concurrent readers — no locks, no contention
    for _ in 0..8 {
        let map = Arc::clone(&map);
        s.spawn(move || {
            let m = map.pin();
            let _ = m.get(&42);
        });
    }
});
```

## When to use scry-index

Read-heavy concurrent workloads with sorted keys — time-series queries,
lookup tables, caches, analytics indexes, in-memory stores where multiple
threads read and a few threads write. See [Design trade-offs](#design-trade-offs)
for where other structures are a better fit.

## Benchmarks

100K sequential keys, bulk-loaded. Measured on the same machine; see
`examples/simulate.rs` for the full workload.

### Point lookups (single-threaded)

| Operation | LearnedMap | BTreeMap | SkipMap |
|---|---|---|---|
| Sequential keys | **0.6 ms** | 2.8 ms (4.7x) | 4.5 ms (7.6x) |
| Random keys | **0.9 ms** | 5.4 ms (6.2x) | 15.6 ms (18x) |

### Concurrent (100K keys, 8 reader + 4 writer threads)

| Operation | LearnedMap | RwLock\<BTreeMap\> | SkipMap |
|---|---|---|---|
| 8T point reads | **1.6 ms** | 9.8 ms (6.2x) | 9.2 ms (5.9x) |
| 4T writes | **1.5 ms** | 3.4 ms (2.2x) | 2.3 ms (1.5x) |
| 8R + 4W mixed | **3.6 ms** | 27.3 ms (7.5x) | 11.5 ms (3.2x) |

### Writes (single-threaded)

| Operation | LearnedMap | BTreeMap |
|---|---|---|
| Incremental insert (10K into 100K) | **0.2 ms** | 0.3 ms (1.5x) |
| Bulk load | 1.4 ms | **0.7 ms** (1.9x faster) |

```sh
cargo run --example simulate --release      # full 3-way workload comparison
cargo run --example mini_bench --release     # single-threaded breakdown
cargo bench                                  # criterion microbenchmarks
```

## Design trade-offs

These are structural properties of concurrent learned indexes
([SALI](https://people.iiis.tsinghua.edu.cn/~huanchen/publications/sali-sigmod24.pdf),
LIPP, ALEX), not implementation gaps.

**O(1) lookups vs O(log n).** The model-predicted lookup advantage grows
with scale and is largest under random access, where B-tree variants
suffer cache misses at depth. This is the core benefit of learned indexes.

**Range scans are slower than BTreeMap** (~30x for pure sequential scans).
The LIPP chain method places keys at model-predicted slots across a tree
of nodes; in-order traversal is a DFS with pointer chasing. BTreeMap's
contiguous sorted leaves make range scans essentially sequential memory
reads. This is the fundamental trade-off of the chain method — it is what
enables lock-free concurrent inserts without data shifting. Under
concurrent read/write workloads the gap narrows significantly because
BTreeMap requires a write lock while `LearnedMap` readers never block.

**Random-key insert is slower than BTreeMap** (~3-5x from empty). Keys
that don't match the node's linear model collide into the same slot,
creating deeper chains until a localized rebuild fires. All learned
indexes assume some structure in the key distribution — sequential,
time-series, and bulk-loaded keys fit the model well. Use `bulk_load`
when data is pre-sorted.

**Per-operation epoch overhead.** Lock-free reclamation (crossbeam-epoch)
adds a small fixed cost to every operation. This is shared by all
epoch-based concurrent structures (including `SkipMap`). The cost is
amortized under actual concurrency — single-threaded insert is ~1.3x
slower than BTreeMap, but concurrent insert is 2.2x faster.

## Features

- Lock-free reads via epoch-based reclamation
- CAS-based writes, per-slot contention only
- Sorted iteration and range queries (`a..b`, `a..=b`, `a..`, `..b`, `..`)
- Bulk loading from sorted data (`bulk_load`, `bulk_load_dedup`)
- Depth-triggered localized subtree rebuilds
- Tombstone compaction for remove-heavy workloads
- `get_or_insert` / `get_or_insert_with` entry API
- `clear()` / `drain()` for bulk removal
- `allocated_bytes()` for memory introspection
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
