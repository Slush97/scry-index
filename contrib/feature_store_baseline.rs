//! # ML Feature Store Baseline (no scry-index)
//!
//! Same pipeline as `feature_store.rs` but using std collections:
//! - `HashMap` for point lookups
//! - `BTreeMap` for sorted/range lookups
//! - `RwLock` for concurrent access
//!
//! Run with:
//! ```sh
//! cargo run --example feature_store_baseline --release
//! ```
#![allow(clippy::too_many_lines)]

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use scry_learn::dataset::Dataset;
use scry_learn::metrics::accuracy;
use scry_learn::tree::RandomForestClassifier;

const N_USERS: usize = 10_000;
const N_FEATURES: usize = 5;
const N_THREADS: usize = 8;
const REQUESTS_PER_THREAD: usize = 5_000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Phase 1: Generate synthetic user data (identical to feature_store.rs) ──

    println!("=== ML Feature Store Baseline ===");
    println!("    scry-learn + HashMap / BTreeMap\n");

    let mut rng = StdRng::seed_from_u64(42);

    let feature_names = vec![
        "session_duration".into(),
        "pages_viewed".into(),
        "scroll_depth".into(),
        "time_of_day".into(),
        "return_visits".into(),
    ];

    let mut user_features: Vec<Vec<f64>> = Vec::with_capacity(N_USERS);
    let mut columns: Vec<Vec<f64>> = (0..N_FEATURES)
        .map(|_| Vec::with_capacity(N_USERS))
        .collect();

    for _ in 0..N_USERS {
        let row: Vec<f64> = (0..N_FEATURES).map(|_| rng.gen::<f64>()).collect();
        for (col, &val) in columns.iter_mut().zip(row.iter()) {
            col.push(val);
        }
        user_features.push(row);
    }

    let weights = [0.3, 0.2, 0.35, -0.1, 0.25];
    let target: Vec<f64> = user_features
        .iter()
        .map(|feats| {
            let score: f64 = feats.iter().zip(weights.iter()).map(|(f, w)| f * w).sum();
            if score > 0.35 {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    let n_positive = target
        .iter()
        .filter(|&&t| (t - 1.0).abs() < f64::EPSILON)
        .count();
    println!(
        "Generated {N_USERS} users, {N_FEATURES} features, {:.1}% positive class",
        n_positive as f64 / N_USERS as f64 * 100.0
    );

    // ── Phase 2: Train classifier (identical) ──────────────────────────

    let dataset = Dataset::new(columns, target, feature_names, "will_convert");

    let mut model = RandomForestClassifier::new()
        .n_estimators(20)
        .max_depth(6)
        .seed(42);

    let train_start = Instant::now();
    model.fit(&dataset)?;
    println!(
        "Trained RandomForest (20 trees, depth 6) in {:?}",
        train_start.elapsed()
    );

    let preds = model.predict(&dataset.feature_matrix())?;
    println!(
        "Training accuracy: {:.4}",
        accuracy(&dataset.target, &preds)
    );

    // ── Phase 3a: Build HashMap store ──────────────────────────────────

    let load_start = Instant::now();
    let hashmap: HashMap<u64, Vec<f64>> = user_features
        .iter()
        .enumerate()
        .map(|(i, feats)| (i as u64, feats.clone()))
        .collect();
    let hashmap_time = load_start.elapsed();
    let hashmap = Arc::new(RwLock::new(hashmap));
    println!("\nHashMap:  {N_USERS} vectors loaded in {hashmap_time:?}");

    // ── Phase 3b: Build BTreeMap store ─────────────────────────────────

    let load_start = Instant::now();
    let btree: BTreeMap<u64, Vec<f64>> = user_features
        .iter()
        .enumerate()
        .map(|(i, feats)| (i as u64, feats.clone()))
        .collect();
    let btree_time = load_start.elapsed();
    let btree = Arc::new(RwLock::new(btree));
    println!("BTreeMap: {N_USERS} vectors loaded in {btree_time:?}");

    // ── Phase 4a: Concurrent serving — HashMap + RwLock ────────────────

    let total_requests = N_THREADS * REQUESTS_PER_THREAD;
    println!("\n--- HashMap + RwLock ({total_requests} predictions, {N_THREADS} threads) ---");

    let model = Arc::new(model);
    let serve_start = Instant::now();

    let handles: Vec<_> = (0..N_THREADS)
        .map(|t| {
            let store = Arc::clone(&hashmap);
            let model = Arc::clone(&model);
            thread::spawn(move || {
                let mut served: u64 = 0;
                let mut positive: u64 = 0;
                for i in 0..REQUESTS_PER_THREAD {
                    let user_id = ((t * REQUESTS_PER_THREAD + i) % N_USERS) as u64;
                    let features = {
                        let map = store.read().unwrap();
                        map.get(&user_id).cloned()
                    };
                    if let Some(features) = features {
                        let pred = model.predict(&[features]).unwrap();
                        if (pred[0] - 1.0).abs() < f64::EPSILON {
                            positive += 1;
                        }
                        served += 1;
                    }
                }
                (served, positive)
            })
        })
        .collect();

    let mut total_served: u64 = 0;
    let mut total_positive: u64 = 0;
    for h in handles {
        let (s, p) = h.join().unwrap();
        total_served += s;
        total_positive += p;
    }

    let hashmap_elapsed = serve_start.elapsed();
    let hashmap_throughput = total_served as f64 / hashmap_elapsed.as_secs_f64();
    println!("Done in {hashmap_elapsed:?}");
    println!("  Throughput:  {hashmap_throughput:.0} predictions/sec");
    println!(
        "  Predicted:   {total_positive} positive / {total_served} total ({:.1}%)",
        total_positive as f64 / total_served as f64 * 100.0
    );

    // ── Phase 4b: Concurrent serving — BTreeMap + RwLock ───────────────

    println!("\n--- BTreeMap + RwLock ({total_requests} predictions, {N_THREADS} threads) ---");

    let serve_start = Instant::now();

    let handles: Vec<_> = (0..N_THREADS)
        .map(|t| {
            let store = Arc::clone(&btree);
            let model = Arc::clone(&model);
            thread::spawn(move || {
                let mut served: u64 = 0;
                let mut positive: u64 = 0;
                for i in 0..REQUESTS_PER_THREAD {
                    let user_id = ((t * REQUESTS_PER_THREAD + i) % N_USERS) as u64;
                    let features = {
                        let map = store.read().unwrap();
                        map.get(&user_id).cloned()
                    };
                    if let Some(features) = features {
                        let pred = model.predict(&[features]).unwrap();
                        if (pred[0] - 1.0).abs() < f64::EPSILON {
                            positive += 1;
                        }
                        served += 1;
                    }
                }
                (served, positive)
            })
        })
        .collect();

    total_served = 0;
    total_positive = 0;
    for h in handles {
        let (s, p) = h.join().unwrap();
        total_served += s;
        total_positive += p;
    }

    let btree_elapsed = serve_start.elapsed();
    let btree_throughput = total_served as f64 / btree_elapsed.as_secs_f64();
    println!("Done in {btree_elapsed:?}");
    println!("  Throughput:  {btree_throughput:.0} predictions/sec");
    println!(
        "  Predicted:   {total_positive} positive / {total_served} total ({:.1}%)",
        total_positive as f64 / total_served as f64 * 100.0
    );

    // ── Phase 5: Range queries ─────────────────────────────────────────

    println!("\n--- Range Queries ---");

    // HashMap: no range support — must filter all keys
    let range_start = Instant::now();
    let hashmap_guard = hashmap.read().unwrap();
    let hash_range_count = hashmap_guard
        .iter()
        .filter(|(&k, _)| (1000..1100).contains(&k))
        .count();
    let hash_range_time = range_start.elapsed();
    println!(
        "HashMap  scan [1000..1100]: {hash_range_count} users in {hash_range_time:?} (full scan, no sorted order)",
    );
    drop(hashmap_guard);

    // BTreeMap: native range support
    let range_start = Instant::now();
    let btree_range: Vec<(u64, Vec<f64>)> = {
        let guard = btree.read().unwrap();
        guard
            .range(1000u64..1100u64)
            .map(|(&k, v)| (k, v.clone()))
            .collect()
    };
    let btree_range_time = range_start.elapsed();
    println!(
        "BTreeMap range [1000..1100]: {} users in {:?}",
        btree_range.len(),
        btree_range_time,
    );

    // Batch predict on the BTreeMap range
    let batch_features: Vec<Vec<f64>> = btree_range.iter().map(|(_, v)| v.clone()).collect();
    let batch_preds = model.predict(&batch_features)?;
    let batch_positive = batch_preds
        .iter()
        .filter(|&&p| (p - 1.0).abs() < f64::EPSILON)
        .count();
    println!(
        "  {batch_positive}/{} predicted to convert",
        btree_range.len()
    );

    // ── Summary ────────────────────────────────────────────────────────

    println!("\n=== Summary ===");
    println!("HashMap + RwLock:  {hashmap_throughput:>12.0} pred/sec");
    println!("BTreeMap + RwLock: {btree_throughput:>12.0} pred/sec");
    println!("\nNote: RwLock allows concurrent reads but blocks all readers on write.");
    println!("Run `cargo run --example feature_store --release` to compare with scry-index.");

    Ok(())
}
