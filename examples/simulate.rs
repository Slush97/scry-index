//! Time-series workload simulator for scry-index.
//!
//! Simulates a metrics ingestion service to compare `LearnedMap` against
//! `RwLock<BTreeMap>` and crossbeam `SkipMap` across realistic workloads.
//!
#![allow(
    clippy::too_many_lines,
    clippy::from_iter_instead_of_collect,
    clippy::needless_pass_by_value,
    dead_code,
    clippy::uninlined_format_args,
    clippy::collection_is_never_read
)]
//! Run with:
//! ```sh
//! cargo run --example simulate --release
//! ```

use std::collections::BTreeMap;
use std::hint::black_box;
use std::sync::{Arc, Barrier, RwLock};
use std::time::{Duration, Instant};

use crossbeam_skiplist::SkipMap;
use rand::Rng;
use scry_index::LearnedMap;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const HISTORICAL_KEYS: usize = 100_000;
const LOOKUP_SAMPLE: usize = 100_000;
const RANGE_SIZE: usize = 10_000;
const RANGE_ITERS: usize = 100;
const WRITE_PER_THREAD: usize = 10_000;
const READ_PER_THREAD: usize = 50_000;
const SCAN_PER_THREAD: usize = 500;
const READER_THREADS: usize = 8;
const WRITER_THREADS: usize = 4;

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

struct WorkloadResult {
    name: &'static str,
    detail: String,
    learned: Duration,
    btree: Duration,
    skipmap: Duration,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn fmt_dur(d: Duration) -> String {
    let us = d.as_micros();
    if us < 1_000 {
        format!("{us}us")
    } else if us < 1_000_000 {
        format!("{:.1}ms", us as f64 / 1_000.0)
    } else {
        format!("{:.2}s", us as f64 / 1_000_000.0)
    }
}

fn speedup(baseline: Duration, candidate: Duration) -> String {
    if candidate.is_zero() || baseline.is_zero() {
        return "N/A".into();
    }
    let ratio = baseline.as_nanos() as f64 / candidate.as_nanos() as f64;
    if ratio >= 1.0 {
        format!("\x1b[32m{ratio:.2}x\x1b[0m")
    } else {
        format!("\x1b[31m{:.2}x\x1b[0m", ratio)
    }
}

/// 1M sequential timestamps at microsecond intervals from a fixed epoch.
fn sequential_timestamps(n: usize) -> Vec<(u64, u64)> {
    let base: u64 = 1_700_000_000_000_000;
    (0..n as u64)
        .map(|i| (base + i, i.wrapping_mul(7).wrapping_add(42)))
        .collect()
}

// ---------------------------------------------------------------------------
// Workloads
// ---------------------------------------------------------------------------

fn bench_bulk_load(data: &[(u64, u64)]) -> WorkloadResult {
    let n = data.len();

    let start = Instant::now();
    let _ = black_box(LearnedMap::bulk_load(data).unwrap());
    let learned = start.elapsed();

    let start = Instant::now();
    let _ = black_box(BTreeMap::from_iter(data.iter().copied()));
    let btree = start.elapsed();

    let skip = SkipMap::new();
    let start = Instant::now();
    for &(k, v) in data {
        skip.insert(k, v);
    }
    black_box(&skip);
    let skipmap = start.elapsed();

    WorkloadResult {
        name: "Bulk load",
        detail: format!("{n} seq keys"),
        learned,
        btree,
        skipmap,
    }
}

fn bench_seq_lookup(data: &[(u64, u64)]) -> WorkloadResult {
    let n = data.len();
    let sample_n = LOOKUP_SAMPLE.min(n);

    let lm = LearnedMap::bulk_load(data).unwrap();
    let bt: BTreeMap<u64, u64> = data.iter().copied().collect();
    let sm = SkipMap::new();
    for &(k, v) in data {
        sm.insert(k, v);
    }
    let keys: Vec<u64> = data[..sample_n].iter().map(|(k, _)| *k).collect();

    let guard = lm.guard();
    let start = Instant::now();
    for &k in &keys {
        black_box(lm.get(&k, &guard));
    }
    let learned = start.elapsed();

    let start = Instant::now();
    for &k in &keys {
        black_box(bt.get(&k));
    }
    let btree = start.elapsed();

    let start = Instant::now();
    for &k in &keys {
        black_box(sm.get(&k));
    }
    let skipmap = start.elapsed();

    WorkloadResult {
        name: "Point lookup (seq)",
        detail: format!("{sample_n} lookups"),
        learned,
        btree,
        skipmap,
    }
}

fn bench_rand_lookup(data: &[(u64, u64)]) -> WorkloadResult {
    let n = data.len();
    let mut rng = rand::thread_rng();

    let lm = LearnedMap::bulk_load(data).unwrap();
    let bt: BTreeMap<u64, u64> = data.iter().copied().collect();
    let sm = SkipMap::new();
    for &(k, v) in data {
        sm.insert(k, v);
    }
    let keys: Vec<u64> = data.iter().map(|(k, _)| *k).collect();
    let sample: Vec<u64> = (0..LOOKUP_SAMPLE)
        .map(|_| keys[rng.gen_range(0..n)])
        .collect();

    let guard = lm.guard();
    let start = Instant::now();
    for &k in &sample {
        black_box(lm.get(&k, &guard));
    }
    let learned = start.elapsed();

    let start = Instant::now();
    for &k in &sample {
        black_box(bt.get(&k));
    }
    let btree = start.elapsed();

    let start = Instant::now();
    for &k in &sample {
        black_box(sm.get(&k));
    }
    let skipmap = start.elapsed();

    WorkloadResult {
        name: "Point lookup (rand)",
        detail: format!("{LOOKUP_SAMPLE} lookups"),
        learned,
        btree,
        skipmap,
    }
}

fn bench_range_scan(data: &[(u64, u64)]) -> WorkloadResult {
    let n = data.len();

    let lm = LearnedMap::bulk_load(data).unwrap();
    let bt: BTreeMap<u64, u64> = data.iter().copied().collect();
    let sm = SkipMap::new();
    for &(k, v) in data {
        sm.insert(k, v);
    }

    let lo = data[n - RANGE_SIZE].0;
    let hi = data[n - 1].0;

    let guard = lm.guard();
    let start = Instant::now();
    for _ in 0..RANGE_ITERS {
        black_box(lm.range(lo..=hi, &guard).count());
    }
    let learned = start.elapsed();

    let start = Instant::now();
    for _ in 0..RANGE_ITERS {
        black_box(bt.range(lo..=hi).count());
    }
    let btree = start.elapsed();

    let start = Instant::now();
    for _ in 0..RANGE_ITERS {
        black_box(sm.range(lo..=hi).count());
    }
    let skipmap = start.elapsed();

    WorkloadResult {
        name: "Range scan",
        detail: format!("{RANGE_SIZE} keys x{RANGE_ITERS}"),
        learned,
        btree,
        skipmap,
    }
}

fn bench_incremental_insert(data: &[(u64, u64)]) -> WorkloadResult {
    let insert_n: usize = 10_000;
    let base = data.last().unwrap().0 + 1;

    let lm = LearnedMap::bulk_load(data).unwrap();
    let guard = lm.guard();
    let start = Instant::now();
    for i in 0..insert_n as u64 {
        lm.insert(base + i, i, &guard);
    }
    let learned = start.elapsed();

    let mut bt: BTreeMap<u64, u64> = data.iter().copied().collect();
    let start = Instant::now();
    for i in 0..insert_n as u64 {
        bt.insert(base + i, i);
    }
    let btree = start.elapsed();

    let sm = SkipMap::new();
    for &(k, v) in data {
        sm.insert(k, v);
    }
    let start = Instant::now();
    for i in 0..insert_n as u64 {
        sm.insert(base + i, i);
    }
    let skipmap = start.elapsed();

    WorkloadResult {
        name: "Incremental insert",
        detail: format!("{insert_n} appends into {}", data.len()),
        learned,
        btree,
        skipmap,
    }
}

// ---------------------------------------------------------------------------
// Concurrent workloads
// ---------------------------------------------------------------------------

fn bench_concurrent_read(data: &[(u64, u64)]) -> WorkloadResult {
    let n = data.len();
    let threads = READER_THREADS;
    let reads = READ_PER_THREAD;
    let keys: Arc<Vec<u64>> = Arc::new(data.iter().map(|(k, _)| *k).collect());

    // LearnedMap — lock-free
    let lm = Arc::new(LearnedMap::bulk_load(data).unwrap());
    let bar = Arc::new(Barrier::new(threads + 1));
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let m = Arc::clone(&lm);
            let k = Arc::clone(&keys);
            let b = Arc::clone(&bar);
            std::thread::spawn(move || {
                let mut rng = rand::thread_rng();
                b.wait();
                let g = m.guard();
                for _ in 0..reads {
                    black_box(m.get(&k[rng.gen_range(0..n)], &g));
                }
            })
        })
        .collect();
    bar.wait();
    let start = Instant::now();
    for h in handles {
        h.join().unwrap();
    }
    let learned = start.elapsed();

    // RwLock<BTreeMap>
    let bt = Arc::new(RwLock::new(BTreeMap::from_iter(data.iter().copied())));
    let bar = Arc::new(Barrier::new(threads + 1));
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let m = Arc::clone(&bt);
            let k = Arc::clone(&keys);
            let b = Arc::clone(&bar);
            std::thread::spawn(move || {
                let mut rng = rand::thread_rng();
                b.wait();
                for _ in 0..reads {
                    let g = m.read().unwrap();
                    black_box(g.get(&k[rng.gen_range(0..n)]));
                }
            })
        })
        .collect();
    bar.wait();
    let start = Instant::now();
    for h in handles {
        h.join().unwrap();
    }
    let btree = start.elapsed();

    // SkipMap — lock-free
    let sm = Arc::new(SkipMap::new());
    for &(k, v) in data {
        sm.insert(k, v);
    }
    let bar = Arc::new(Barrier::new(threads + 1));
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let m = Arc::clone(&sm);
            let k = Arc::clone(&keys);
            let b = Arc::clone(&bar);
            std::thread::spawn(move || {
                let mut rng = rand::thread_rng();
                b.wait();
                for _ in 0..reads {
                    black_box(m.get(&k[rng.gen_range(0..n)]));
                }
            })
        })
        .collect();
    bar.wait();
    let start = Instant::now();
    for h in handles {
        h.join().unwrap();
    }
    let skipmap = start.elapsed();

    WorkloadResult {
        name: "Concurrent read",
        detail: format!("{threads}T x {reads} lookups"),
        learned,
        btree,
        skipmap,
    }
}

fn bench_concurrent_write(data: &[(u64, u64)]) -> WorkloadResult {
    let threads = WRITER_THREADS;
    let writes = WRITE_PER_THREAD;
    let base = data.last().unwrap().0 + 1;

    // LearnedMap
    let lm = Arc::new(LearnedMap::bulk_load(data).unwrap());
    let bar = Arc::new(Barrier::new(threads + 1));
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let m = Arc::clone(&lm);
            let b = Arc::clone(&bar);
            std::thread::spawn(move || {
                let off = t as u64 * writes as u64;
                b.wait();
                let g = m.guard();
                for i in 0..writes as u64 {
                    m.insert(base + off + i, i, &g);
                }
            })
        })
        .collect();
    bar.wait();
    let start = Instant::now();
    for h in handles {
        h.join().unwrap();
    }
    let learned = start.elapsed();

    // RwLock<BTreeMap>
    let bt = Arc::new(RwLock::new(BTreeMap::from_iter(data.iter().copied())));
    let bar = Arc::new(Barrier::new(threads + 1));
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let m = Arc::clone(&bt);
            let b = Arc::clone(&bar);
            std::thread::spawn(move || {
                let off = t as u64 * writes as u64;
                b.wait();
                for i in 0..writes as u64 {
                    m.write().unwrap().insert(base + off + i, i);
                }
            })
        })
        .collect();
    bar.wait();
    let start = Instant::now();
    for h in handles {
        h.join().unwrap();
    }
    let btree = start.elapsed();

    // SkipMap
    let sm = Arc::new(SkipMap::new());
    for &(k, v) in data {
        sm.insert(k, v);
    }
    let bar = Arc::new(Barrier::new(threads + 1));
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let m = Arc::clone(&sm);
            let b = Arc::clone(&bar);
            std::thread::spawn(move || {
                let off = t as u64 * writes as u64;
                b.wait();
                for i in 0..writes as u64 {
                    m.insert(base + off + i, i);
                }
            })
        })
        .collect();
    bar.wait();
    let start = Instant::now();
    for h in handles {
        h.join().unwrap();
    }
    let skipmap = start.elapsed();

    WorkloadResult {
        name: "Concurrent write",
        detail: format!("{threads}T x {writes} inserts"),
        learned,
        btree,
        skipmap,
    }
}

fn bench_mixed_rw(data: &[(u64, u64)]) -> WorkloadResult {
    let n = data.len();
    let readers = READER_THREADS;
    let writers = WRITER_THREADS;
    let total = readers + writers;
    let reads = READ_PER_THREAD;
    let writes = WRITE_PER_THREAD;
    let base = data.last().unwrap().0 + 1;
    let keys: Arc<Vec<u64>> = Arc::new(data.iter().map(|(k, _)| *k).collect());

    // LearnedMap
    let lm = Arc::new(LearnedMap::bulk_load(data).unwrap());
    let bar = Arc::new(Barrier::new(total + 1));
    let mut handles = Vec::with_capacity(total);
    for _ in 0..readers {
        let m = Arc::clone(&lm);
        let k = Arc::clone(&keys);
        let b = Arc::clone(&bar);
        handles.push(std::thread::spawn(move || {
            let mut rng = rand::thread_rng();
            b.wait();
            let g = m.guard();
            for _ in 0..reads {
                black_box(m.get(&k[rng.gen_range(0..n)], &g));
            }
        }));
    }
    for t in 0..writers {
        let m = Arc::clone(&lm);
        let b = Arc::clone(&bar);
        handles.push(std::thread::spawn(move || {
            let off = t as u64 * writes as u64;
            b.wait();
            let g = m.guard();
            for i in 0..writes as u64 {
                m.insert(base + off + i, i, &g);
            }
        }));
    }
    bar.wait();
    let start = Instant::now();
    for h in handles {
        h.join().unwrap();
    }
    let learned = start.elapsed();

    // RwLock<BTreeMap>
    let bt = Arc::new(RwLock::new(BTreeMap::from_iter(data.iter().copied())));
    let bar = Arc::new(Barrier::new(total + 1));
    let mut handles = Vec::with_capacity(total);
    for _ in 0..readers {
        let m = Arc::clone(&bt);
        let k = Arc::clone(&keys);
        let b = Arc::clone(&bar);
        handles.push(std::thread::spawn(move || {
            let mut rng = rand::thread_rng();
            b.wait();
            for _ in 0..reads {
                let g = m.read().unwrap();
                black_box(g.get(&k[rng.gen_range(0..n)]));
            }
        }));
    }
    for t in 0..writers {
        let m = Arc::clone(&bt);
        let b = Arc::clone(&bar);
        handles.push(std::thread::spawn(move || {
            let off = t as u64 * writes as u64;
            b.wait();
            for i in 0..writes as u64 {
                m.write().unwrap().insert(base + off + i, i);
            }
        }));
    }
    bar.wait();
    let start = Instant::now();
    for h in handles {
        h.join().unwrap();
    }
    let btree = start.elapsed();

    // SkipMap
    let sm = Arc::new(SkipMap::new());
    for &(k, v) in data {
        sm.insert(k, v);
    }
    let bar = Arc::new(Barrier::new(total + 1));
    let mut handles = Vec::with_capacity(total);
    for _ in 0..readers {
        let m = Arc::clone(&sm);
        let k = Arc::clone(&keys);
        let b = Arc::clone(&bar);
        handles.push(std::thread::spawn(move || {
            let mut rng = rand::thread_rng();
            b.wait();
            for _ in 0..reads {
                black_box(m.get(&k[rng.gen_range(0..n)]));
            }
        }));
    }
    for t in 0..writers {
        let m = Arc::clone(&sm);
        let b = Arc::clone(&bar);
        handles.push(std::thread::spawn(move || {
            let off = t as u64 * writes as u64;
            b.wait();
            for i in 0..writes as u64 {
                m.insert(base + off + i, i);
            }
        }));
    }
    bar.wait();
    let start = Instant::now();
    for h in handles {
        h.join().unwrap();
    }
    let skipmap = start.elapsed();

    WorkloadResult {
        name: "Mixed read/write",
        detail: format!("{readers}R + {writers}W"),
        learned,
        btree,
        skipmap,
    }
}

fn bench_concurrent_range(data: &[(u64, u64)]) -> WorkloadResult {
    let n = data.len();
    let readers = READER_THREADS;
    let writers = WRITER_THREADS;
    let total = readers + writers;
    let writes = WRITE_PER_THREAD;
    let scans = SCAN_PER_THREAD;
    let base = data.last().unwrap().0 + 1;
    let mid = data[n / 2].0;
    let range_end = mid + RANGE_SIZE as u64;

    // LearnedMap
    let lm = Arc::new(LearnedMap::bulk_load(data).unwrap());
    let bar = Arc::new(Barrier::new(total + 1));
    let mut handles = Vec::with_capacity(total);
    for _ in 0..readers {
        let m = Arc::clone(&lm);
        let b = Arc::clone(&bar);
        handles.push(std::thread::spawn(move || {
            b.wait();
            let g = m.guard();
            for _ in 0..scans {
                black_box(m.range(mid..range_end, &g).count());
            }
        }));
    }
    for t in 0..writers {
        let m = Arc::clone(&lm);
        let b = Arc::clone(&bar);
        handles.push(std::thread::spawn(move || {
            let off = t as u64 * writes as u64;
            b.wait();
            let g = m.guard();
            for i in 0..writes as u64 {
                m.insert(base + off + i, i, &g);
            }
        }));
    }
    bar.wait();
    let start = Instant::now();
    for h in handles {
        h.join().unwrap();
    }
    let learned = start.elapsed();

    // RwLock<BTreeMap>
    let bt = Arc::new(RwLock::new(BTreeMap::from_iter(data.iter().copied())));
    let bar = Arc::new(Barrier::new(total + 1));
    let mut handles = Vec::with_capacity(total);
    for _ in 0..readers {
        let m = Arc::clone(&bt);
        let b = Arc::clone(&bar);
        handles.push(std::thread::spawn(move || {
            b.wait();
            for _ in 0..scans {
                let g = m.read().unwrap();
                black_box(g.range(mid..range_end).count());
            }
        }));
    }
    for t in 0..writers {
        let m = Arc::clone(&bt);
        let b = Arc::clone(&bar);
        handles.push(std::thread::spawn(move || {
            let off = t as u64 * writes as u64;
            b.wait();
            for i in 0..writes as u64 {
                m.write().unwrap().insert(base + off + i, i);
            }
        }));
    }
    bar.wait();
    let start = Instant::now();
    for h in handles {
        h.join().unwrap();
    }
    let btree = start.elapsed();

    // SkipMap
    let sm = Arc::new(SkipMap::new());
    for &(k, v) in data {
        sm.insert(k, v);
    }
    let bar = Arc::new(Barrier::new(total + 1));
    let mut handles = Vec::with_capacity(total);
    for _ in 0..readers {
        let m = Arc::clone(&sm);
        let b = Arc::clone(&bar);
        handles.push(std::thread::spawn(move || {
            b.wait();
            for _ in 0..scans {
                black_box(m.range(mid..range_end).count());
            }
        }));
    }
    for t in 0..writers {
        let m = Arc::clone(&sm);
        let b = Arc::clone(&bar);
        handles.push(std::thread::spawn(move || {
            let off = t as u64 * writes as u64;
            b.wait();
            for i in 0..writes as u64 {
                m.insert(base + off + i, i);
            }
        }));
    }
    bar.wait();
    let start = Instant::now();
    for h in handles {
        h.join().unwrap();
    }
    let skipmap = start.elapsed();

    WorkloadResult {
        name: "Range scan + writes",
        detail: format!("{readers}R(scan) + {writers}W"),
        learned,
        btree,
        skipmap,
    }
}

// ---------------------------------------------------------------------------
// Tree diagnostics
// ---------------------------------------------------------------------------

fn print_diagnostics(data: &[(u64, u64)]) {
    let lm = LearnedMap::bulk_load(data).unwrap();
    let guard = lm.guard();
    let depth_bulk = lm.max_depth(&guard);

    // Insert 100K more sequential keys and check depth
    let base = data.last().unwrap().0 + 1;
    for i in 0..100_000u64 {
        lm.insert(base + i, i, &guard);
    }
    let depth_after = lm.max_depth(&guard);

    println!("  Tree diagnostics ({} bulk-loaded keys):", data.len());
    println!("    Max depth after bulk load:       {depth_bulk}");
    println!("    Max depth after +100K inserts:   {depth_after}");
    println!("    Total keys:                      {}", lm.len());
    println!();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!();
    println!("  ╔═══════════════════════════════════════════════════════╗");
    println!("  ║  scry-index workload simulator                       ║");
    println!("  ║  Scenario: time-series metrics ingestion service     ║");
    println!("  ╚═══════════════════════════════════════════════════════╝");
    println!();
    println!("  Generating {HISTORICAL_KEYS} sequential timestamps...");

    let data = sequential_timestamps(HISTORICAL_KEYS);

    println!("  Running workloads... (use --release for accurate numbers)");
    println!();

    let mut results = Vec::new();

    eprintln!("  [1/9] bulk_load...");
    results.push(bench_bulk_load(&data));
    eprintln!("  [2/9] seq_lookup...");
    results.push(bench_seq_lookup(&data));
    eprintln!("  [3/9] rand_lookup...");
    results.push(bench_rand_lookup(&data));
    eprintln!("  [4/9] range_scan...");
    results.push(bench_range_scan(&data));
    eprintln!("  [5/9] incremental_insert...");
    results.push(bench_incremental_insert(&data));
    eprintln!("  [6/9] concurrent_read...");
    results.push(bench_concurrent_read(&data));
    eprintln!("  [7/9] concurrent_write...");
    results.push(bench_concurrent_write(&data));
    eprintln!("  [8/9] mixed_rw...");
    results.push(bench_mixed_rw(&data));
    eprintln!("  [9/9] concurrent_range...");
    results.push(bench_concurrent_range(&data));

    // --- Results table ---
    let nw = 22;
    let dw = 26;
    let tw = 12;

    println!(
        "  {:<nw$} {:<dw$} {:>tw$} {:>tw$} {:>tw$}",
        "Workload", "Detail", "Learned", "BTreeMap", "SkipMap"
    );
    println!("  {}", "─".repeat(nw + dw + tw * 3 + 4));

    for r in &results {
        println!(
            "  {:<nw$} {:<dw$} {:>tw$} {:>tw$} {:>tw$}",
            r.name,
            r.detail,
            fmt_dur(r.learned),
            fmt_dur(r.btree),
            fmt_dur(r.skipmap),
        );
    }

    // --- Speedup vs BTreeMap ---
    println!();
    println!("  Speedup vs BTreeMap (>1 = learned is faster):");
    println!("  {}", "─".repeat(nw + 14));
    for r in &results {
        println!("  {:<nw$} {}", r.name, speedup(r.btree, r.learned));
    }

    // --- Speedup vs SkipMap ---
    println!();
    println!("  Speedup vs SkipMap (>1 = learned is faster):");
    println!("  {}", "─".repeat(nw + 14));
    for r in &results {
        println!("  {:<nw$} {}", r.name, speedup(r.skipmap, r.learned));
    }

    // --- Diagnostics disabled: sequential inserts into a bulk-loaded tree
    // funnel all keys into one root slot, causing O(n^2) rebuild work and
    // unbounded deferred-memory growth.  This needs a root-level refit or
    // adaptive model extension before it can run safely.
    // println!();
    // print_diagnostics(&data);
}
