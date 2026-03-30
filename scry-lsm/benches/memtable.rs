//! Three-way memtable benchmark: LearnedMap vs SkipMap vs RwLock<BTreeMap>.
//!
//! These benchmarks simulate realistic LSM engine memtable workloads to
//! demonstrate where the learned index advantage appears.
//!
//! Run with: `cargo bench -p scry-lsm`

#![allow(clippy::doc_markdown, missing_docs)]

use std::sync::{Arc, Barrier};
use std::thread;

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use rand::Rng;
use scry_lsm::memtable::{
    BTreeMemtable, LearnedMemtable, Memtable, SkipListMemtable,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a key from a u64 counter (8-byte big-endian for sort order).
fn make_key(i: u64) -> Vec<u8> {
    i.to_be_bytes().to_vec()
}

/// Generate a value of the given size.
fn make_value(i: u64, size: usize) -> Vec<u8> {
    let seed = i.to_le_bytes();
    let mut v = vec![0u8; size];
    for (j, byte) in v.iter_mut().enumerate() {
        *byte = seed[j % 8];
    }
    v
}

/// Prefill a memtable with `n` sequential entries.
fn prefill(mt: &dyn Memtable, n: u64) {
    for i in 0..n {
        mt.insert(make_key(i), make_value(i, 64));
    }
}

/// Names for the three implementations, used as benchmark IDs.
const NAMES: [&str; 3] = ["learned", "skiplist", "btree"];

/// Create one instance of each memtable backend.
fn make_memtables() -> [Box<dyn Memtable>; 3] {
    [
        Box::new(LearnedMemtable::new()),
        Box::new(SkipListMemtable::new()),
        Box::new(BTreeMemtable::new()),
    ]
}

// ---------------------------------------------------------------------------
// Benchmark: Sequential fill (single-threaded)
// ---------------------------------------------------------------------------

fn bench_fill_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("memtable_fill_sequential");

    for &n in &[10_000u64, 50_000, 100_000] {
        group.throughput(Throughput::Elements(n));

        for (idx, name) in NAMES.iter().enumerate() {
            group.bench_with_input(BenchmarkId::new(*name, n), &n, |b, &n| {
                b.iter(|| {
                    let mts = make_memtables();
                    let mt = &mts[idx];
                    for i in 0..n {
                        mt.insert(make_key(i), make_value(i, 64));
                    }
                });
            });
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Concurrent read (prefilled, read-only, multiple threads)
// ---------------------------------------------------------------------------

fn bench_concurrent_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("memtable_concurrent_read");
    let n = 100_000u64;
    let num_threads = 8usize;
    let reads_per_thread = 50_000u64;
    group.throughput(Throughput::Elements(
        reads_per_thread * num_threads as u64,
    ));

    for (idx, name) in NAMES.iter().enumerate() {
        group.bench_function(BenchmarkId::new(*name, num_threads), |b| {
            // Prefill once per benchmark iteration.
            let mts = make_memtables();
            let mt = &mts[idx];
            prefill(mt.as_ref(), n);

            b.iter(|| {
                // We need to share the memtable across threads.
                // Since the trait is Send + Sync, we can use a raw pointer
                // wrapped in a newtype. But for safety, let's just prefill
                // fresh each time and use scoped threads.
                thread::scope(|s| {
                    let barrier = Arc::new(Barrier::new(num_threads));
                    for _ in 0..num_threads {
                        let barrier = Arc::clone(&barrier);
                        s.spawn(move || {
                            let mut rng = rand::thread_rng();
                            barrier.wait();
                            for _ in 0..reads_per_thread {
                                let key = make_key(rng.gen_range(0..n));
                                std::hint::black_box(mt.get(&key));
                            }
                        });
                    }
                });
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Mixed read/write (80% read, 20% write, multiple threads)
// ---------------------------------------------------------------------------

fn bench_mixed_rw(c: &mut Criterion) {
    let mut group = c.benchmark_group("memtable_mixed_rw_80_20");
    let prefill_n = 50_000u64;
    let num_threads = 8usize;
    let ops_per_thread = 20_000u64;
    group.throughput(Throughput::Elements(
        ops_per_thread * num_threads as u64,
    ));

    for (idx, name) in NAMES.iter().enumerate() {
        group.bench_function(BenchmarkId::new(*name, num_threads), |b| {
            b.iter(|| {
                let mts = make_memtables();
                let mt = &mts[idx];
                prefill(mt.as_ref(), prefill_n);

                thread::scope(|s| {
                    let barrier = Arc::new(Barrier::new(num_threads));
                    for t in 0..num_threads {
                        let barrier = Arc::clone(&barrier);
                        s.spawn(move || {
                            let mut rng = rand::thread_rng();
                            barrier.wait();
                            for i in 0..ops_per_thread {
                                if i % 5 == 0 {
                                    // 20% writes — each thread writes to its own range.
                                    let key_id = prefill_n
                                        + (t as u64) * ops_per_thread
                                        + i;
                                    mt.insert(
                                        make_key(key_id),
                                        make_value(key_id, 64),
                                    );
                                } else {
                                    // 80% reads.
                                    let key =
                                        make_key(rng.gen_range(0..prefill_n));
                                    std::hint::black_box(mt.get(&key));
                                }
                            }
                        });
                    }
                });
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Concurrent write (multiple threads, non-overlapping ranges)
// ---------------------------------------------------------------------------

fn bench_concurrent_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("memtable_concurrent_write");
    let num_threads = 8usize;
    let writes_per_thread = 10_000u64;
    group.throughput(Throughput::Elements(
        writes_per_thread * num_threads as u64,
    ));

    for (idx, name) in NAMES.iter().enumerate() {
        group.bench_function(BenchmarkId::new(*name, num_threads), |b| {
            b.iter(|| {
                let mts = make_memtables();
                let mt = &mts[idx];

                thread::scope(|s| {
                    let barrier = Arc::new(Barrier::new(num_threads));
                    for t in 0..num_threads {
                        let barrier = Arc::clone(&barrier);
                        s.spawn(move || {
                            let base = (t as u64) * writes_per_thread;
                            barrier.wait();
                            for i in 0..writes_per_thread {
                                mt.insert(
                                    make_key(base + i),
                                    make_value(base + i, 64),
                                );
                            }
                        });
                    }
                });
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Drain (simulates memtable flush to SSTable)
// ---------------------------------------------------------------------------

fn bench_drain(c: &mut Criterion) {
    let mut group = c.benchmark_group("memtable_drain");
    let n = 100_000u64;
    group.throughput(Throughput::Elements(n));

    for (idx, name) in NAMES.iter().enumerate() {
        group.bench_function(BenchmarkId::new(*name, n), |b| {
            b.iter(|| {
                let mts = make_memtables();
                let mt = &mts[idx];
                prefill(mt.as_ref(), n);
                std::hint::black_box(mt.drain_sorted());
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Fill-drain cycle (steady-state LSM operation)
// ---------------------------------------------------------------------------

fn bench_fill_drain_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("memtable_fill_drain_cycle");
    let n = 50_000u64;
    let cycles = 3u64;
    group.throughput(Throughput::Elements(n * cycles));

    for (idx, name) in NAMES.iter().enumerate() {
        group.bench_function(BenchmarkId::new(*name, cycles), |b| {
            b.iter(|| {
                let mts = make_memtables();
                let mt = &mts[idx];
                for cycle in 0..cycles {
                    let base = cycle * n;
                    for i in 0..n {
                        mt.insert(make_key(base + i), make_value(base + i, 64));
                    }
                    std::hint::black_box(mt.drain_sorted());
                }
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_fill_sequential,
    bench_concurrent_read,
    bench_mixed_rw,
    bench_concurrent_write,
    bench_drain,
    bench_fill_drain_cycle,
);
criterion_main!(benches);
