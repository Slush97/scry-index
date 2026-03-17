use criterion::{black_box, criterion_group, criterion_main, Criterion};
use scry_index::LearnedMap;
use std::collections::BTreeMap;

fn bench_insert_learned_10k(c: &mut Criterion) {
    c.bench_function("learned_map_insert_10k_seq", |b| {
        b.iter(|| {
            let mut map = LearnedMap::new();
            for i in 0..10_000u64 {
                black_box(map.insert(i, i));
            }
        });
    });
}

fn bench_insert_btree_10k(c: &mut Criterion) {
    c.bench_function("btree_map_insert_10k_seq", |b| {
        b.iter(|| {
            let mut map = BTreeMap::new();
            for i in 0..10_000u64 {
                black_box(map.insert(i, i));
            }
        });
    });
}

fn bench_bulk_load_learned(c: &mut Criterion) {
    let pairs: Vec<(u64, u64)> = (0..10_000).map(|i| (i, i * 10)).collect();

    c.bench_function("learned_map_bulk_load_10k", |b| {
        b.iter(|| {
            black_box(LearnedMap::bulk_load(&pairs).unwrap());
        });
    });
}

fn bench_bulk_load_btree(c: &mut Criterion) {
    let pairs: Vec<(u64, u64)> = (0..10_000).map(|i| (i, i * 10)).collect();

    c.bench_function("btree_map_from_iter_10k", |b| {
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

criterion_group!(
    benches,
    bench_insert_learned_10k,
    bench_insert_btree_10k,
    bench_bulk_load_learned,
    bench_bulk_load_btree,
    bench_bulk_load_100k,
);
criterion_main!(benches);
