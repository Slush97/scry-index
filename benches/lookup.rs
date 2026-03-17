use criterion::{black_box, criterion_group, criterion_main, Criterion};
use scry_index::LearnedMap;
use std::collections::BTreeMap;

fn bench_lookup_learned_10k(c: &mut Criterion) {
    let pairs: Vec<(u64, u64)> = (0..10_000).map(|i| (i, i * 10)).collect();
    let map = LearnedMap::bulk_load(&pairs).unwrap();

    c.bench_function("learned_get_10k_seq", |b| {
        b.iter(|| {
            for i in 0..10_000u64 {
                black_box(map.get(&i));
            }
        });
    });
}

fn bench_lookup_btree_10k(c: &mut Criterion) {
    let map: BTreeMap<u64, u64> = (0..10_000).map(|i| (i, i * 10)).collect();

    c.bench_function("btree_get_10k_seq", |b| {
        b.iter(|| {
            for i in 0..10_000u64 {
                black_box(map.get(&i));
            }
        });
    });
}

fn bench_lookup_100k(c: &mut Criterion) {
    let pairs: Vec<(u64, u64)> = (0..100_000).map(|i| (i, i * 10)).collect();
    let learned = LearnedMap::bulk_load(&pairs).unwrap();
    let btree: BTreeMap<u64, u64> = pairs.iter().copied().collect();

    c.bench_function("learned_get_100k_seq", |b| {
        b.iter(|| {
            for i in 0..100_000u64 {
                black_box(learned.get(&i));
            }
        });
    });

    c.bench_function("btree_get_100k_seq", |b| {
        b.iter(|| {
            for i in 0..100_000u64 {
                black_box(btree.get(&i));
            }
        });
    });
}

fn bench_lookup_sparse(c: &mut Criterion) {
    // Sparse keys: every 1000th integer
    let pairs: Vec<(u64, u64)> = (0..10_000).map(|i| (i * 1000, i)).collect();
    let learned = LearnedMap::bulk_load(&pairs).unwrap();
    let btree: BTreeMap<u64, u64> = pairs.iter().copied().collect();

    c.bench_function("learned_get_10k_sparse", |b| {
        b.iter(|| {
            for i in 0..10_000u64 {
                black_box(learned.get(&(i * 1000)));
            }
        });
    });

    c.bench_function("btree_get_10k_sparse", |b| {
        b.iter(|| {
            for i in 0..10_000u64 {
                black_box(btree.get(&(i * 1000)));
            }
        });
    });
}

criterion_group!(
    benches,
    bench_lookup_learned_10k,
    bench_lookup_btree_10k,
    bench_lookup_100k,
    bench_lookup_sparse,
);
criterion_main!(benches);
