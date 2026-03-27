//! # ML Feature Store Demo
//!
//! Demonstrates scry-index as a concurrent feature store for ML inference.
//!
//! Pipeline:
//! 1. Generate synthetic user data (10K users, 5 features)
//! 2. Train a Random Forest classifier with scry-learn
//! 3. Bulk-load user features into a `LearnedMap`
//! 4. Serve predictions from 8 threads concurrently
//!
//! Run with:
//! ```sh
//! cargo run --example feature_store --release
//! ```
#![allow(clippy::too_many_lines)]

use std::sync::Arc;
use std::thread;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use scry_index::LearnedMap;

use scry_learn::dataset::Dataset;
use scry_learn::metrics::accuracy;
use scry_learn::tree::RandomForestClassifier;

const N_USERS: usize = 10_000;
const N_FEATURES: usize = 5;
const N_THREADS: usize = 8;
const REQUESTS_PER_THREAD: usize = 5_000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Phase 1: Generate synthetic user data ──────────────────────────
    //
    // Simulates a recommendation system: each user has engagement features,
    // and we predict whether they'll convert (click/buy/subscribe).

    println!("=== ML Feature Store Demo ===");
    println!("    scry-learn + scry-index\n");

    let mut rng = StdRng::seed_from_u64(42);

    // Feature meanings (synthetic):
    //   0: session_duration  (0-1, normalized)
    //   1: pages_viewed      (0-1)
    //   2: scroll_depth      (0-1)
    //   3: time_of_day       (0-1)
    //   4: return_visits      (0-1)

    let feature_names = vec![
        "session_duration".into(),
        "pages_viewed".into(),
        "scroll_depth".into(),
        "time_of_day".into(),
        "return_visits".into(),
    ];

    // Generate row-major (for scry-index storage) and column-major (for scry-learn)
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

    // Label: convert if weighted engagement score exceeds threshold.
    // This gives the model a learnable signal.
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

    // ── Phase 2: Train classifier ──────────────────────────────────────

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

    // ── Phase 3: Build feature store ───────────────────────────────────
    //
    // In production, this would be: "load latest features from warehouse
    // into an in-memory store for low-latency serving."

    let pairs: Vec<(u64, Vec<f64>)> = user_features
        .iter()
        .enumerate()
        .map(|(i, feats)| (i as u64, feats.clone()))
        .collect();

    let load_start = Instant::now();
    let store = LearnedMap::bulk_load(&pairs)?;
    let store = Arc::new(store);
    println!(
        "\nFeature store: {N_USERS} vectors loaded in {:?}",
        load_start.elapsed()
    );

    let guard = store.guard();
    println!(
        "Memory: ~{:.1} KB ({:.1} bytes/user)",
        store.allocated_bytes(&guard) as f64 / 1024.0,
        store.allocated_bytes(&guard) as f64 / N_USERS as f64,
    );
    drop(guard);

    // ── Phase 4: Concurrent inference serving ──────────────────────────
    //
    // 8 threads, each handling 5K requests. Each request:
    //   1. Look up user features from scry-index (lock-free)
    //   2. Run the trained model to predict conversion

    let total_requests = N_THREADS * REQUESTS_PER_THREAD;
    println!("\nServing {total_requests} predictions across {N_THREADS} threads...");

    let model = Arc::new(model);
    let serve_start = Instant::now();

    let handles: Vec<_> = (0..N_THREADS)
        .map(|t| {
            let store = Arc::clone(&store);
            let model = Arc::clone(&model);
            thread::spawn(move || {
                let map = store.pin();
                let mut served: u64 = 0;
                let mut positive: u64 = 0;
                for i in 0..REQUESTS_PER_THREAD {
                    let user_id = ((t * REQUESTS_PER_THREAD + i) % N_USERS) as u64;
                    if let Some(features) = map.get(&user_id) {
                        let pred = model.predict(std::slice::from_ref(features)).unwrap();
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
        let (served, positive) = h.join().unwrap();
        total_served += served;
        total_positive += positive;
    }

    let serve_elapsed = serve_start.elapsed();
    let throughput = total_served as f64 / serve_elapsed.as_secs_f64();

    println!("Done in {serve_elapsed:?}");
    println!("  Throughput:  {throughput:.0} predictions/sec");
    println!(
        "  Predicted:   {total_positive} positive / {total_served} total ({:.1}%)",
        total_positive as f64 / total_served as f64 * 100.0
    );

    // ── Phase 5: Feature store queries ─────────────────────────────────
    //
    // Show range queries — "find all users in an ID range" is free because
    // scry-index maintains sorted order.

    let guard = store.guard();

    println!("\n--- Feature Store Queries ---");

    // Point lookup
    if let Some(feats) = store.get(&42, &guard) {
        let pred = model.predict(std::slice::from_ref(feats))?;
        let label = if (pred[0] - 1.0).abs() < f64::EPSILON {
            "will convert"
        } else {
            "won't convert"
        };
        println!("User 42: {label}");
        println!(
            "  features: [{}]",
            feats
                .iter()
                .map(|f| format!("{f:.3}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // Range query: fetch a batch of users for bulk scoring
    let range_start = Instant::now();
    let batch: Vec<_> = store.range(1000u64..1100u64, &guard).collect();
    let range_time = range_start.elapsed();
    println!(
        "Range [1000..1100]: {} users fetched in {:?}",
        batch.len(),
        range_time,
    );

    // Batch predict on the range result
    let batch_features: Vec<Vec<f64>> = batch.iter().map(|(_, v)| (*v).clone()).collect();
    let batch_preds = model.predict(&batch_features)?;
    let batch_positive = batch_preds
        .iter()
        .filter(|&&p| (p - 1.0).abs() < f64::EPSILON)
        .count();
    println!("  {batch_positive}/{} predicted to convert", batch.len());

    println!("\nDone.");
    Ok(())
}
