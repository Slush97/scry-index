//! Memtable backed by [`crossbeam_skiplist::SkipMap`].

use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_skiplist::SkipMap;

use super::Memtable;

/// A memtable using a lock-free skip list — the industry standard.
///
/// This is the data structure used by RocksDB, LevelDB, and most production
/// LSM engines for their memtables. Provides O(log n) lookups and concurrent
/// access without a global lock.
pub struct SkipListMemtable {
    map: SkipMap<Vec<u8>, Vec<u8>>,
    byte_count: AtomicUsize,
}

impl SkipListMemtable {
    /// Create a new empty skip-list memtable.
    pub fn new() -> Self {
        Self {
            map: SkipMap::new(),
            byte_count: AtomicUsize::new(0),
        }
    }
}

impl Default for SkipListMemtable {
    fn default() -> Self {
        Self::new()
    }
}

impl Memtable for SkipListMemtable {
    fn insert(&self, key: Vec<u8>, value: Vec<u8>) {
        let added_bytes = key.len() + value.len();
        self.map.insert(key, value);
        self.byte_count.fetch_add(added_bytes, Ordering::Relaxed);
    }

    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.map.get(key).map(|e| e.value().clone())
    }

    fn delete(&self, key: &[u8]) {
        self.insert(key.to_vec(), Vec::new());
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn approximate_bytes(&self) -> usize {
        self.byte_count.load(Ordering::Relaxed)
    }

    fn drain_sorted(&self) -> Vec<(Vec<u8>, Vec<u8>)> {
        let entries: Vec<_> = self
            .map
            .iter()
            .map(|e| (e.key().clone(), e.value().clone()))
            .collect();
        for (key, _) in &entries {
            self.map.remove(key);
        }
        self.byte_count.store(0, Ordering::Relaxed);
        entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get() {
        let mt = SkipListMemtable::new();
        mt.insert(b"hello".to_vec(), b"world".to_vec());
        assert_eq!(mt.get(b"hello"), Some(b"world".to_vec()));
        assert_eq!(mt.get(b"missing"), None);
        assert_eq!(mt.len(), 1);
    }

    #[test]
    fn upsert_overwrites() {
        let mt = SkipListMemtable::new();
        mt.insert(b"key".to_vec(), b"v1".to_vec());
        mt.insert(b"key".to_vec(), b"v2".to_vec());
        assert_eq!(mt.get(b"key"), Some(b"v2".to_vec()));
    }

    #[test]
    fn delete_inserts_tombstone() {
        let mt = SkipListMemtable::new();
        mt.insert(b"key".to_vec(), b"value".to_vec());
        mt.delete(b"key");
        assert_eq!(mt.get(b"key"), Some(Vec::new()));
    }

    #[test]
    fn drain_returns_sorted_and_empties() {
        let mt = SkipListMemtable::new();
        mt.insert(b"c".to_vec(), b"3".to_vec());
        mt.insert(b"a".to_vec(), b"1".to_vec());
        mt.insert(b"b".to_vec(), b"2".to_vec());

        let entries = mt.drain_sorted();
        let keys: Vec<&[u8]> = entries.iter().map(|(k, _)| k.as_slice()).collect();
        assert_eq!(keys, vec![b"a".as_slice(), b"b", b"c"]);
        assert!(mt.is_empty());
    }
}
