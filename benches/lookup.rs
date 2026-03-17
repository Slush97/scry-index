use criterion::{black_box, criterion_group, criterion_main, Criterion};
use scry_index::LearnedMap;
use std::collections::BTreeMap;

fn bench_lookup_learned(c: &mut Criterion) {
    let pairs: Vec<(u64, u64)> = (0..10_000).map(|i| (i, i * 10)).collect();
    let map = LearnedMap::bulk_load(&pairs).unwrap();

    c.bench_function("learned_map_get_10k", |b| {
        b.iter(|| {
            for i in 0..10_000u64 {
                black_box(map.get(&i));
            }
        });
    });
}

fn bench_lookup_btree(c: &mut Criterion) {
    let map: BTreeMap<u64, u64> = (0..10_000).map(|i| (i, i * 10)).collect();

    c.bench_function("btree_map_get_10k", |b| {
        b.iter(|| {
            for i in 0..10_000u64 {
                black_box(map.get(&i));
            }
        });
    });
}

criterion_group!(benches, bench_lookup_learned, bench_lookup_btree);
criterion_main!(benches);
