//! Integration tests for the LSM engine.

use scry_lsm::engine::{EngineConfig, LsmEngine};
use scry_lsm::memtable::LearnedMemtable;

fn open_engine(dir: &std::path::Path) -> LsmEngine {
    LsmEngine::open(
        dir,
        Box::new(LearnedMemtable::new()),
        EngineConfig::default(),
    )
    .unwrap()
}

fn open_engine_with_threshold(dir: &std::path::Path, threshold: usize) -> LsmEngine {
    LsmEngine::open(
        dir,
        Box::new(LearnedMemtable::new()),
        EngineConfig {
            flush_threshold: threshold,
        },
    )
    .unwrap()
}

#[test]
fn put_get_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let engine = open_engine(dir.path());

    engine.put(b"hello", b"world").unwrap();
    engine.put(b"foo", b"bar").unwrap();

    assert_eq!(engine.get(b"hello").unwrap(), Some(b"world".to_vec()));
    assert_eq!(engine.get(b"foo").unwrap(), Some(b"bar".to_vec()));
    assert_eq!(engine.get(b"missing").unwrap(), None);
}

#[test]
fn delete_returns_none() {
    let dir = tempfile::tempdir().unwrap();
    let engine = open_engine(dir.path());

    engine.put(b"key", b"value").unwrap();
    assert_eq!(engine.get(b"key").unwrap(), Some(b"value".to_vec()));

    engine.delete(b"key").unwrap();
    assert_eq!(engine.get(b"key").unwrap(), None);
}

#[test]
fn overwrite_value() {
    let dir = tempfile::tempdir().unwrap();
    let engine = open_engine(dir.path());

    engine.put(b"key", b"v1").unwrap();
    engine.put(b"key", b"v2").unwrap();
    assert_eq!(engine.get(b"key").unwrap(), Some(b"v2".to_vec()));
}

#[test]
fn flush_and_read_from_sstable() {
    let dir = tempfile::tempdir().unwrap();
    let engine = open_engine(dir.path());

    engine.put(b"aaa", b"111").unwrap();
    engine.put(b"bbb", b"222").unwrap();
    engine.put(b"ccc", b"333").unwrap();

    // Force flush to SSTable.
    engine.flush().unwrap();

    // Data should still be readable from SSTable.
    assert_eq!(engine.get(b"aaa").unwrap(), Some(b"111".to_vec()));
    assert_eq!(engine.get(b"bbb").unwrap(), Some(b"222".to_vec()));
    assert_eq!(engine.get(b"ccc").unwrap(), Some(b"333".to_vec()));
    assert_eq!(engine.get(b"missing").unwrap(), None);
}

#[test]
fn tombstone_propagates_through_flush() {
    let dir = tempfile::tempdir().unwrap();
    let engine = open_engine(dir.path());

    // Put a key, flush it to SSTable.
    engine.put(b"key", b"value").unwrap();
    engine.flush().unwrap();

    // Delete the key (tombstone goes to memtable).
    engine.delete(b"key").unwrap();

    // Tombstone in memtable should shadow the SSTable value.
    assert_eq!(engine.get(b"key").unwrap(), None);

    // Flush the tombstone to SSTable.
    engine.flush().unwrap();

    // Still deleted.
    assert_eq!(engine.get(b"key").unwrap(), None);
}

#[test]
fn wal_recovery() {
    let dir = tempfile::tempdir().unwrap();

    // Write some data.
    {
        let engine = open_engine(dir.path());
        engine.put(b"key1", b"val1").unwrap();
        engine.put(b"key2", b"val2").unwrap();
        engine.delete(b"key1").unwrap();
        // Drop engine without flushing — data is only in WAL.
    }

    // Reopen — WAL should be replayed.
    {
        let engine = open_engine(dir.path());
        assert_eq!(engine.get(b"key1").unwrap(), None); // Was deleted.
        assert_eq!(engine.get(b"key2").unwrap(), Some(b"val2".to_vec()));
    }
}

#[test]
fn auto_flush_on_threshold() {
    let dir = tempfile::tempdir().unwrap();
    // Tiny threshold to force auto-flush.
    let engine = open_engine_with_threshold(dir.path(), 100);

    // Insert enough data to exceed 100 bytes.
    for i in 0u32..20 {
        let key = format!("key-{i:04}");
        let val = format!("value-{i:04}");
        engine.put(key.as_bytes(), val.as_bytes()).unwrap();
    }

    // After auto-flush, data should be readable (either from memtable
    // or from SSTable).
    for i in 0u32..20 {
        let key = format!("key-{i:04}");
        let val = format!("value-{i:04}");
        assert_eq!(
            engine.get(key.as_bytes()).unwrap(),
            Some(val.into_bytes()),
            "missing key: {key}"
        );
    }
}

#[test]
fn multiple_flushes_and_reads() {
    let dir = tempfile::tempdir().unwrap();
    let engine = open_engine(dir.path());

    // Batch 1.
    for i in 0u32..100 {
        let key = format!("a-{i:05}");
        engine.put(key.as_bytes(), b"batch1").unwrap();
    }
    engine.flush().unwrap();

    // Batch 2.
    for i in 0u32..100 {
        let key = format!("b-{i:05}");
        engine.put(key.as_bytes(), b"batch2").unwrap();
    }
    engine.flush().unwrap();

    // Both batches readable.
    assert_eq!(engine.get(b"a-00050").unwrap(), Some(b"batch1".to_vec()));
    assert_eq!(engine.get(b"b-00050").unwrap(), Some(b"batch2".to_vec()));
}

// -------------------------------------------------------------------------
// Streaming ingestion tests
// -------------------------------------------------------------------------

#[test]
fn ingest_iter_basic() {
    let dir = tempfile::tempdir().unwrap();
    let engine = open_engine(dir.path());

    let count = engine
        .ingest_iter((0u32..50).map(|i| {
            (
                format!("k-{i:04}").into_bytes(),
                format!("v-{i:04}").into_bytes(),
            )
        }))
        .unwrap();
    assert_eq!(count, 50);

    assert_eq!(engine.get(b"k-0000").unwrap(), Some(b"v-0000".to_vec()));
    assert_eq!(engine.get(b"k-0049").unwrap(), Some(b"v-0049".to_vec()));
    assert_eq!(engine.get(b"k-0050").unwrap(), None);
}

#[test]
fn ingest_iter_triggers_flush() {
    let dir = tempfile::tempdir().unwrap();
    // Tiny threshold so ingestion triggers multiple flushes.
    let engine = open_engine_with_threshold(dir.path(), 200);

    let count = engine
        .ingest_iter((0u32..100).map(|i| {
            (
                format!("k-{i:04}").into_bytes(),
                b"some-value-data".to_vec(),
            )
        }))
        .unwrap();
    assert_eq!(count, 100);

    // All entries readable across memtable + SSTables.
    for i in 0u32..100 {
        let key = format!("k-{i:04}");
        assert!(
            engine.get(key.as_bytes()).unwrap().is_some(),
            "missing after ingest: {key}"
        );
    }
}

#[test]
fn ingest_reader_tsv() {
    let dir = tempfile::tempdir().unwrap();
    let engine = open_engine(dir.path());

    let tsv = b"alpha\t111\nbravo\t222\ncharlie\t333\n";
    let count = engine.ingest_reader(&tsv[..]).unwrap();
    assert_eq!(count, 3);

    assert_eq!(engine.get(b"alpha").unwrap(), Some(b"111".to_vec()));
    assert_eq!(engine.get(b"bravo").unwrap(), Some(b"222".to_vec()));
    assert_eq!(engine.get(b"charlie").unwrap(), Some(b"333".to_vec()));
}

#[test]
fn ingest_reader_skips_malformed_lines() {
    let dir = tempfile::tempdir().unwrap();
    let engine = open_engine(dir.path());

    let tsv = b"good\tdata\nbad line no tab\n\nalso_good\tmore_data\n";
    let count = engine.ingest_reader(&tsv[..]).unwrap();
    assert_eq!(count, 2); // Only the two valid lines.

    assert_eq!(engine.get(b"good").unwrap(), Some(b"data".to_vec()));
    assert_eq!(
        engine.get(b"also_good").unwrap(),
        Some(b"more_data".to_vec())
    );
}

#[test]
fn ingest_sorted_bypasses_memtable() {
    let dir = tempfile::tempdir().unwrap();
    let engine = open_engine(dir.path());

    let count = engine
        .ingest_sorted((0u32..200).map(|i| {
            (
                format!("s-{i:05}").into_bytes(),
                format!("v-{i:05}").into_bytes(),
            )
        }))
        .unwrap();
    assert_eq!(count, 200);

    // Data is readable from SSTable (not memtable).
    assert_eq!(engine.get(b"s-00000").unwrap(), Some(b"v-00000".to_vec()));
    assert_eq!(engine.get(b"s-00100").unwrap(), Some(b"v-00100".to_vec()));
    assert_eq!(engine.get(b"s-00199").unwrap(), Some(b"v-00199".to_vec()));
    assert_eq!(engine.get(b"s-00200").unwrap(), None);
}

#[test]
fn ingest_sorted_rejects_unsorted() {
    let dir = tempfile::tempdir().unwrap();
    let engine = open_engine(dir.path());

    let bad_data = vec![
        (b"bbb".to_vec(), b"2".to_vec()),
        (b"aaa".to_vec(), b"1".to_vec()), // Out of order.
    ];

    let result = engine.ingest_sorted(bad_data.into_iter());
    assert!(result.is_err());
}

#[test]
fn ingest_sorted_empty_is_noop() {
    let dir = tempfile::tempdir().unwrap();
    let engine = open_engine(dir.path());

    let count = engine.ingest_sorted(std::iter::empty()).unwrap();
    assert_eq!(count, 0);
}

#[test]
fn mixed_ingest_and_put() {
    let dir = tempfile::tempdir().unwrap();
    let engine = open_engine(dir.path());

    // Bulk-ingest sorted data directly to SSTable.
    let sorted: Vec<(Vec<u8>, Vec<u8>)> = vec![
        (b"a".to_vec(), b"from-sorted".to_vec()),
        (b"c".to_vec(), b"from-sorted".to_vec()),
    ];
    engine.ingest_sorted(sorted.into_iter()).unwrap();

    // Regular puts go through memtable.
    engine.put(b"b", b"from-put").unwrap();

    // Overwrite a sorted key via put.
    engine.put(b"a", b"overwritten").unwrap();

    // Reads merge both sources correctly.
    assert_eq!(engine.get(b"a").unwrap(), Some(b"overwritten".to_vec()));
    assert_eq!(engine.get(b"b").unwrap(), Some(b"from-put".to_vec()));
    assert_eq!(engine.get(b"c").unwrap(), Some(b"from-sorted".to_vec()));
}
