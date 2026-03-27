//! Basic usage of `LearnedMap` and `LearnedSet`.
//!
//! Run with:
//! ```sh
//! cargo run --example basic
//! ```

use scry_index::{Config, LearnedMap, LearnedSet};

fn main() {
    // ── Basic insert / get / remove ──────────────────────────────────────

    let map = LearnedMap::new();
    let m = map.pin(); // pin an epoch guard for convenience

    m.insert(10u64, "ten");
    m.insert(20u64, "twenty");
    m.insert(30u64, "thirty");
    m.insert(5u64, "five");

    println!("get(20) = {:?}", m.get(&20)); // Some(&"twenty")
    println!("get(99) = {:?}", m.get(&99)); // None
    println!("len     = {}", m.len());      // 4

    // Update an existing key
    m.insert(10u64, "TEN");
    println!("get(10) = {:?}", m.get(&10)); // Some(&"TEN")

    // Remove
    m.remove(&5);
    println!("contains(5) = {}", m.contains_key(&5)); // false

    // ── Sorted iteration ─────────────────────────────────────────────────

    println!("\nAll entries (sorted):");
    for (k, v) in m.iter() {
        println!("  {k}: {v}");
    }

    // ── Range queries ────────────────────────────────────────────────────

    println!("\nRange 10..=20:");
    for (k, v) in m.range(10u64..=20) {
        println!("  {k}: {v}");
    }

    println!(
        "first = {:?}, last = {:?}",
        m.first_key_value(),
        m.last_key_value(),
    );

    // ── Bulk loading (efficient for pre-sorted data) ─────────────────────

    let pairs: Vec<(u64, &str)> = (0..100).map(|i| (i, "bulk")).collect();
    let bulk = LearnedMap::bulk_load(&pairs).expect("sorted, non-empty, no dups");
    let b = bulk.pin();
    println!("\nbulk-loaded {} entries", b.len());
    println!("bulk get(50) = {:?}", b.get(&50));

    // ── Entry API (atomic check-and-insert) ──────────────────────────────

    let inserted = m.get_or_insert(42u64, "new");
    println!("\nget_or_insert(42) = {inserted:?}"); // "new"

    let existing = m.get_or_insert(42u64, "ignored");
    println!("get_or_insert(42) = {existing:?}");   // "new" (not replaced)

    // ── Custom configuration ─────────────────────────────────────────────

    let config = Config::new()
        .expansion_factor(3.0)
        .rebuild_depth_threshold(6);

    let configured: LearnedMap<i32, String> = LearnedMap::with_config(config);
    let c = configured.pin();
    for i in 0..50 {
        c.insert(i, format!("val-{i}"));
    }
    println!("\nconfigured map len = {}", c.len());

    // ── LearnedSet (keys only) ───────────────────────────────────────────

    let set = LearnedSet::new();
    let s = set.pin();
    s.insert(1u64);
    s.insert(2u64);
    s.insert(3u64);
    println!("\nset contains(2) = {}", s.contains(&2));
    println!("set len = {}", s.len());

    // ── String keys ──────────────────────────────────────────────────────

    let names = LearnedMap::new();
    let n = names.pin();
    n.insert("alice".to_string(), 30);
    n.insert("bob".to_string(), 25);
    n.insert("charlie".to_string(), 35);

    println!("\nString key lookup: alice = {:?}", n.get(&"alice".to_string()));
    println!("Sorted names:");
    for (name, age) in n.iter() {
        println!("  {name}: {age}");
    }

    println!("\nDone.");
}
