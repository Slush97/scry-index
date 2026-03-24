#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use scry_index::{Config, LearnedMap};
use std::collections::BTreeMap;

fn bench_insert_learned_10k(c: &mut Criterion) {
    c.bench_function("learned_insert_10k_seq", |b| {
        b.iter(|| {
            let map = LearnedMap::new();
            let guard = map.guard();
            for i in 0..10_000u64 {
                black_box(map.insert(i, i, &guard));
            }
        });
    });
}

fn bench_insert_learned_10k_no_rebuild(c: &mut Criterion) {
    c.bench_function("learned_insert_10k_seq_no_rebuild", |b| {
        b.iter(|| {
            let map = LearnedMap::with_config(Config::new().auto_rebuild(false));
            let guard = map.guard();
            for i in 0..10_000u64 {
                black_box(map.insert(i, i, &guard));
            }
        });
    });
}

fn bench_insert_btree_10k(c: &mut Criterion) {
    c.bench_function("btree_insert_10k_seq", |b| {
        b.iter(|| {
            let mut map = BTreeMap::new();
            for i in 0..10_000u64 {
                black_box(map.insert(i, i));
            }
        });
    });
}

fn bench_insert_learned_100k(c: &mut Criterion) {
    c.bench_function("learned_insert_100k_seq", |b| {
        b.iter(|| {
            let map = LearnedMap::new();
            let guard = map.guard();
            for i in 0..100_000u64 {
                black_box(map.insert(i, i, &guard));
            }
        });
    });
}

fn bench_insert_btree_100k(c: &mut Criterion) {
    c.bench_function("btree_insert_100k_seq", |b| {
        b.iter(|| {
            let mut map = BTreeMap::new();
            for i in 0..100_000u64 {
                black_box(map.insert(i, i));
            }
        });
    });
}

fn bench_bulk_load_learned(c: &mut Criterion) {
    let pairs: Vec<(u64, u64)> = (0..10_000).map(|i| (i, i * 10)).collect();

    c.bench_function("learned_bulk_load_10k", |b| {
        b.iter(|| {
            black_box(LearnedMap::bulk_load(&pairs).unwrap());
        });
    });
}

fn bench_bulk_load_btree(c: &mut Criterion) {
    let pairs: Vec<(u64, u64)> = (0..10_000).map(|i| (i, i * 10)).collect();

    c.bench_function("btree_from_iter_10k", |b| {
        b.iter(|| {
            let map: BTreeMap<u64, u64> = pairs.iter().copied().collect();
            black_box(map);
        });
    });
}

fn bench_bulk_load_100k(c: &mut Criterion) {
    let pairs: Vec<(u64, u64)> = (0..100_000).map(|i| (i, i * 10)).collect();

    c.bench_function("learned_bulk_load_100k", |b| {
        b.iter(|| {
            black_box(LearnedMap::bulk_load(&pairs).unwrap());
        });
    });

    c.bench_function("btree_from_iter_100k", |b| {
        b.iter(|| {
            let map: BTreeMap<u64, u64> = pairs.iter().copied().collect();
            black_box(map);
        });
    });
}

fn bench_insert_into_bulkloaded(c: &mut Criterion) {
    let base: Vec<(u64, u64)> = (0..10_000).map(|i| (i * 2, i)).collect();

    c.bench_function("learned_insert_10k_into_bulkloaded", |b| {
        b.iter(|| {
            let map = LearnedMap::bulk_load(&base).unwrap();
            let guard = map.guard();
            // Insert into gaps — model already fits, no pile-up
            for i in 0..10_000u64 {
                black_box(map.insert(i * 2 + 1, i + 1000, &guard));
            }
        });
    });

    c.bench_function("btree_insert_10k_into_existing", |b| {
        b.iter(|| {
            let mut map: BTreeMap<u64, u64> = base.iter().copied().collect();
            for i in 0..10_000u64 {
                black_box(map.insert(i * 2 + 1, i + 1000));
            }
        });
    });
}

fn bench_insert_learned_100_seq(c: &mut Criterion) {
    c.bench_function("learned_insert_100_seq_rampup", |b| {
        b.iter(|| {
            let map = LearnedMap::new();
            let guard = map.guard();
            for i in 0..100u64 {
                black_box(map.insert(i, i, &guard));
            }
        });
    });
}

fn bench_insert_btree_100_seq(c: &mut Criterion) {
    c.bench_function("btree_insert_100_seq_rampup", |b| {
        b.iter(|| {
            let mut map = BTreeMap::new();
            for i in 0..100u64 {
                black_box(map.insert(i, i));
            }
        });
    });
}

criterion_group!(
    benches,
    bench_insert_learned_100_seq,
    bench_insert_btree_100_seq,
    bench_insert_learned_10k,
    bench_insert_learned_10k_no_rebuild,
    bench_insert_btree_10k,
    bench_insert_into_bulkloaded,
    bench_insert_learned_100k,
    bench_insert_btree_100k,
    bench_bulk_load_learned,
    bench_bulk_load_btree,
    bench_bulk_load_100k,
);
criterion_main!(benches);
