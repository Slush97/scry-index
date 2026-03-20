#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use scry_index::LearnedMap;
use std::collections::BTreeMap;
use std::sync::{Arc, Barrier, RwLock};
use std::thread;

fn bench_concurrent_read(c: &mut Criterion) {
    let pairs: Vec<(u64, u64)> = (0..100_000).map(|i| (i, i * 10)).collect();
    let learned = Arc::new(LearnedMap::bulk_load(&pairs).unwrap());
    let btree = Arc::new(RwLock::new(
        pairs.iter().copied().collect::<BTreeMap<u64, u64>>(),
    ));

    c.bench_function("learned_parallel_read_100k_8t", |b| {
        b.iter(|| {
            let barrier = Arc::new(Barrier::new(8));
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let map = Arc::clone(&learned);
                    let barrier = Arc::clone(&barrier);
                    thread::spawn(move || {
                        barrier.wait();
                        let guard = map.guard();
                        for i in 0..100_000u64 {
                            black_box(map.get(&i, &guard));
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
        });
    });

    c.bench_function("rwlock_btree_parallel_read_100k_8t", |b| {
        b.iter(|| {
            let barrier = Arc::new(Barrier::new(8));
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let map = Arc::clone(&btree);
                    let barrier = Arc::clone(&barrier);
                    thread::spawn(move || {
                        barrier.wait();
                        let guard = map.read().unwrap();
                        for i in 0..100_000u64 {
                            black_box(guard.get(&i));
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
        });
    });
}

fn bench_concurrent_mixed(c: &mut Criterion) {
    c.bench_function("learned_mixed_rw_4t", |b| {
        b.iter(|| {
            let map = Arc::new(LearnedMap::new());
            let barrier = Arc::new(Barrier::new(4));

            // 2 writers + 2 readers
            let handles: Vec<_> = (0..4)
                .map(|t| {
                    let map = Arc::clone(&map);
                    let barrier = Arc::clone(&barrier);
                    thread::spawn(move || {
                        barrier.wait();
                        let guard = map.guard();
                        if t < 2 {
                            // writer
                            let base = t as u64 * 5000;
                            for i in 0..5000u64 {
                                map.insert(base + i, base + i, &guard);
                            }
                        } else {
                            // reader
                            for i in 0..10_000u64 {
                                black_box(map.get(&i, &guard));
                            }
                        }
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }
        });
    });

    c.bench_function("rwlock_btree_mixed_rw_4t", |b| {
        b.iter(|| {
            let map = Arc::new(RwLock::new(BTreeMap::new()));
            let barrier = Arc::new(Barrier::new(4));

            let handles: Vec<_> = (0..4)
                .map(|t| {
                    let map = Arc::clone(&map);
                    let barrier = Arc::clone(&barrier);
                    thread::spawn(move || {
                        barrier.wait();
                        if t < 2 {
                            let base = t as u64 * 5000;
                            for i in 0..5000u64 {
                                map.write().unwrap().insert(base + i, base + i);
                            }
                        } else {
                            for i in 0..10_000u64 {
                                let guard = map.read().unwrap();
                                black_box(guard.get(&i));
                            }
                        }
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }
        });
    });
}

fn bench_concurrent_write(c: &mut Criterion) {
    c.bench_function("learned_parallel_write_10k_8t", |b| {
        b.iter(|| {
            let map = Arc::new(LearnedMap::new());
            let barrier = Arc::new(Barrier::new(8));

            let handles: Vec<_> = (0..8u64)
                .map(|t| {
                    let map = Arc::clone(&map);
                    let barrier = Arc::clone(&barrier);
                    thread::spawn(move || {
                        barrier.wait();
                        let guard = map.guard();
                        let base = t * 10_000;
                        for i in 0..10_000u64 {
                            map.insert(base + i, base + i, &guard);
                        }
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }
        });
    });
}

criterion_group!(
    benches,
    bench_concurrent_read,
    bench_concurrent_mixed,
    bench_concurrent_write,
);
criterion_main!(benches);
