//! Quick mini-benchmark for write performance vs BTreeMap.
//! Run with: cargo run --example mini_bench --release
#![allow(clippy::uninlined_format_args, clippy::too_many_lines)]

use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Instant;

use rand::Rng;
use scry_index::LearnedMap;

fn median(times: &mut Vec<f64>) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = times.len() / 2;
    if times.len() % 2 == 0 {
        (times[mid - 1] + times[mid]) / 2.0
    } else {
        times[mid]
    }
}

fn bench<F: FnMut()>(name: &str, iters: usize, mut f: F) -> f64 {
    // warmup
    f();
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let med = median(&mut times);
    let min = times.first().unwrap();
    println!("  {:<45} med={:.3}ms  min={:.3}ms", name, med, min);
    med
}

fn ratio(learned: f64, btree: f64) {
    let r = learned / btree;
    let tag = if r < 1.0 { "FASTER" } else { "slower" };
    println!("    => learned is {:.2}x {}\n", if r < 1.0 { 1.0 / r } else { r }, tag);
}

fn main() {
    let iters = 5;

    // ── Sequential insert (from empty) ──────────────────────────────
    println!("=== Sequential Insert (from empty) ===");

    for &n in &[1_000u64, 10_000, 100_000] {
        println!("  n={n}");
        let l = bench(&format!("  learned insert {n}"), iters, || {
            let map = LearnedMap::new();
            let guard = map.guard();
            for i in 0..n {
                black_box(map.insert(i, i, &guard));
            }
        });
        let b = bench(&format!("  btree insert {n}"), iters, || {
            let mut map = BTreeMap::new();
            for i in 0..n {
                black_box(map.insert(i, i));
            }
        });
        ratio(l, b);
    }

    // ── Bulk load ───────────────────────────────────────────────────
    println!("=== Bulk Load ===");

    for &n in &[10_000usize, 100_000, 500_000] {
        let pairs: Vec<(u64, u64)> = (0..n as u64).map(|i| (i, i)).collect();
        println!("  n={n}");
        let l = bench(&format!("  learned bulk_load {n}"), iters, || {
            black_box(LearnedMap::bulk_load(&pairs).unwrap());
        });
        let b = bench(&format!("  btree from_iter {n}"), iters, || {
            let m: BTreeMap<u64, u64> = pairs.iter().copied().collect();
            black_box(m);
        });
        ratio(l, b);
    }

    // ── Random insert ───────────────────────────────────────────────
    println!("=== Random Insert ===");

    for &n in &[10_000u64, 100_000] {
        let mut rng = rand::thread_rng();
        let keys: Vec<u64> = (0..n).map(|_| rng.gen_range(0..n * 100)).collect();
        println!("  n={n}");
        let l = bench(&format!("  learned random insert {n}"), iters, || {
            let map = LearnedMap::new();
            let guard = map.guard();
            for &k in &keys {
                black_box(map.insert(k, k, &guard));
            }
        });
        let b = bench(&format!("  btree random insert {n}"), iters, || {
            let mut map = BTreeMap::new();
            for &k in &keys {
                black_box(map.insert(k, k));
            }
        });
        ratio(l, b);
    }

    // ── Insert into bulk-loaded (gaps) ──────────────────────────────
    println!("=== Insert into Bulk-Loaded (gap fill) ===");

    for &n in &[10_000u64, 100_000] {
        let base: Vec<(u64, u64)> = (0..n).map(|i| (i * 2, i)).collect();
        println!("  n={n}");
        let l = bench(&format!("  learned gap-fill {n}"), iters, || {
            let map = LearnedMap::bulk_load(&base).unwrap();
            let guard = map.guard();
            for i in 0..n {
                black_box(map.insert(i * 2 + 1, i, &guard));
            }
        });
        let b = bench(&format!("  btree gap-fill {n}"), iters, || {
            let mut map: BTreeMap<u64, u64> = base.iter().copied().collect();
            for i in 0..n {
                black_box(map.insert(i * 2 + 1, i));
            }
        });
        ratio(l, b);
    }

    // ── Append-only (monotonic, simulating time-series) ─────────────
    println!("=== Append-Only (monotonic timestamps) ===");

    let base_ts: u64 = 1_700_000_000;
    for &n in &[10_000u64, 100_000] {
        println!("  n={n}");
        let l = bench(&format!("  learned append {n}"), iters, || {
            let map = LearnedMap::new();
            let guard = map.guard();
            for i in 0..n {
                let ts = base_ts + i;
                black_box(map.insert(ts, i, &guard));
            }
        });
        let b = bench(&format!("  btree append {n}"), iters, || {
            let mut map = BTreeMap::new();
            for i in 0..n {
                let ts = base_ts + i;
                black_box(map.insert(ts, i));
            }
        });
        ratio(l, b);
    }

    // ── Point lookup after bulk load (sequential keys) ───────────
    println!("=== Point Lookup (sequential keys, bulk-loaded) ===");

    for &n in &[10_000usize, 100_000, 500_000] {
        let pairs: Vec<(u64, u64)> = (0..n as u64).map(|i| (i, i)).collect();
        let learned = LearnedMap::bulk_load(&pairs).unwrap();
        let btree: BTreeMap<u64, u64> = pairs.iter().copied().collect();
        println!("  n={n}");
        let l = bench(&format!("  learned get {n}"), iters, || {
            let guard = learned.guard();
            for i in 0..n as u64 {
                black_box(learned.get(&i, &guard));
            }
        });
        let b = bench(&format!("  btree get {n}"), iters, || {
            for i in 0..n as u64 {
                black_box(btree.get(&i));
            }
        });
        ratio(l, b);
    }

    // ── Point lookup (random access pattern) ─────────────────────
    println!("=== Point Lookup (random access, bulk-loaded) ===");

    for &n in &[10_000usize, 100_000, 500_000] {
        let pairs: Vec<(u64, u64)> = (0..n as u64).map(|i| (i, i)).collect();
        let learned = LearnedMap::bulk_load(&pairs).unwrap();
        let btree: BTreeMap<u64, u64> = pairs.iter().copied().collect();
        let mut rng = rand::thread_rng();
        let queries: Vec<u64> = (0..n).map(|_| rng.gen_range(0..n as u64)).collect();
        println!("  n={n}");
        let l = bench(&format!("  learned random get {n}"), iters, || {
            let guard = learned.guard();
            for &k in &queries {
                black_box(learned.get(&k, &guard));
            }
        });
        let b = bench(&format!("  btree random get {n}"), iters, || {
            for &k in &queries {
                black_box(btree.get(&k));
            }
        });
        ratio(l, b);
    }

    // ── Point lookup (miss rate — keys not in map) ───────────────
    println!("=== Point Lookup (50% miss rate) ===");

    for &n in &[10_000usize, 100_000] {
        let pairs: Vec<(u64, u64)> = (0..n as u64).map(|i| (i * 2, i)).collect();
        let learned = LearnedMap::bulk_load(&pairs).unwrap();
        let btree: BTreeMap<u64, u64> = pairs.iter().copied().collect();
        // Query all keys 0..2n — half are hits, half are misses
        println!("  n={n}");
        let l = bench(&format!("  learned mixed hit/miss {n}"), iters, || {
            let guard = learned.guard();
            for i in 0..n as u64 * 2 {
                black_box(learned.get(&i, &guard));
            }
        });
        let b = bench(&format!("  btree mixed hit/miss {n}"), iters, || {
            for i in 0..n as u64 * 2 {
                black_box(btree.get(&i));
            }
        });
        ratio(l, b);
    }
}
