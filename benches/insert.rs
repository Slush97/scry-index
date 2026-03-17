use criterion::{black_box, criterion_group, criterion_main, Criterion};
use scry_index::LearnedMap;
use std::collections::BTreeMap;

fn bench_insert_learned(c: &mut Criterion) {
    c.bench_function("learned_map_insert_10k", |b| {
        b.iter(|| {
            let mut map = LearnedMap::new();
            for i in 0..10_000u64 {
                black_box(map.insert(i, i));
            }
        });
    });
}

fn bench_insert_btree(c: &mut Criterion) {
    c.bench_function("btree_map_insert_10k", |b| {
        b.iter(|| {
            let mut map = BTreeMap::new();
            for i in 0..10_000u64 {
                black_box(map.insert(i, i));
            }
        });
    });
}

fn bench_bulk_load(c: &mut Criterion) {
    let pairs: Vec<(u64, u64)> = (0..10_000).map(|i| (i, i * 10)).collect();

    c.bench_function("learned_map_bulk_load_10k", |b| {
        b.iter(|| {
            black_box(LearnedMap::bulk_load(&pairs).unwrap());
        });
    });
}

criterion_group!(benches, bench_insert_learned, bench_insert_btree, bench_bulk_load);
criterion_main!(benches);
