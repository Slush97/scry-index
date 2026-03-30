//! Memtable backed by `RwLock<BTreeMap>`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;

use super::Memtable;

/// A memtable using `RwLock<BTreeMap>` — the simplest baseline.
///
/// Readers can proceed concurrently with each other but are blocked by
/// writers, and only one writer can proceed at a time. This is the
/// worst-case for concurrent workloads and serves as a lower bound
/// in benchmarks.
pub struct BTreeMemtable {
    map: RwLock<BTreeMap<Vec<u8>, Vec<u8>>>,
    byte_count: AtomicUsize,
}

impl BTreeMemtable {
    /// Create a new empty `BTreeMap`-backed memtable.
    pub fn new() -> Self {
        Self {
            map: RwLock::new(BTreeMap::new()),
            byte_count: AtomicUsize::new(0),
        }
    }
}

impl Default for BTreeMemtable {
    fn default() -> Self {
        Self::new()
    }
}

impl Memtable for BTreeMemtable {
    fn insert(&self, key: Vec<u8>, value: Vec<u8>) {
        let added_bytes = key.len() + value.len();
        self.map.write().unwrap_or_else(std::sync::PoisonError::into_inner).insert(key, value);
        self.byte_count.fetch_add(added_bytes, Ordering::Relaxed);
    }

    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.map.read().unwrap_or_else(std::sync::PoisonError::into_inner).get(key).cloned()
    }

    fn delete(&self, key: &[u8]) {
        self.insert(key.to_vec(), Vec::new());
    }

    fn len(&self) -> usize {
        self.map.read().unwrap_or_else(std::sync::PoisonError::into_inner).len()
    }

    fn approximate_bytes(&self) -> usize {
        self.byte_count.load(Ordering::Relaxed)
    }

    fn drain_sorted(&self) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut map = self.map.write().unwrap_or_else(std::sync::PoisonError::into_inner);
        let old = std::mem::take(&mut *map);
        self.byte_count.store(0, Ordering::Relaxed);
        old.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get() {
        let mt = BTreeMemtable::new();
        mt.insert(b"hello".to_vec(), b"world".to_vec());
        assert_eq!(mt.get(b"hello"), Some(b"world".to_vec()));
        assert_eq!(mt.get(b"missing"), None);
        assert_eq!(mt.len(), 1);
    }

    #[test]
    fn upsert_overwrites() {
        let mt = BTreeMemtable::new();
        mt.insert(b"key".to_vec(), b"v1".to_vec());
        mt.insert(b"key".to_vec(), b"v2".to_vec());
        assert_eq!(mt.get(b"key"), Some(b"v2".to_vec()));
    }

    #[test]
    fn delete_inserts_tombstone() {
        let mt = BTreeMemtable::new();
        mt.insert(b"key".to_vec(), b"value".to_vec());
        mt.delete(b"key");
        assert_eq!(mt.get(b"key"), Some(Vec::new()));
    }

    #[test]
    fn drain_returns_sorted_and_empties() {
        let mt = BTreeMemtable::new();
        mt.insert(b"c".to_vec(), b"3".to_vec());
        mt.insert(b"a".to_vec(), b"1".to_vec());
        mt.insert(b"b".to_vec(), b"2".to_vec());

        let entries = mt.drain_sorted();
        let keys: Vec<&[u8]> = entries.iter().map(|(k, _)| k.as_slice()).collect();
        assert_eq!(keys, vec![b"a".as_slice(), b"b", b"c"]);
        assert!(mt.is_empty());
    }
}
