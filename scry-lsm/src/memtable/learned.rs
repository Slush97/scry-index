//! Memtable backed by [`scry_index::LearnedMap`].

use std::sync::atomic::{AtomicUsize, Ordering};

use scry_index::LearnedMap;

use super::Memtable;

/// A memtable using a learned index for O(1) expected lookups.
///
/// Backed by [`scry_index::LearnedMap`], which provides:
/// - Lock-free concurrent reads (no contention between readers)
/// - CAS-based concurrent writes (writers only contend on the same slot)
/// - Sorted iteration via in-order DFS (needed for flush to SSTable)
/// - Automatic subtree rebuilds to maintain model quality
pub struct LearnedMemtable {
    map: LearnedMap<Vec<u8>, Vec<u8>>,
    byte_count: AtomicUsize,
}

impl LearnedMemtable {
    /// Create a new empty learned memtable.
    pub fn new() -> Self {
        Self {
            map: LearnedMap::new(),
            byte_count: AtomicUsize::new(0),
        }
    }
}

impl Default for LearnedMemtable {
    fn default() -> Self {
        Self::new()
    }
}

impl Memtable for LearnedMemtable {
    fn insert(&self, key: Vec<u8>, value: Vec<u8>) {
        let added_bytes = key.len() + value.len();
        let guard = self.map.guard();

        if self.map.insert(key.clone(), value.clone(), &guard) {
            // New key — count the bytes.
            self.byte_count.fetch_add(added_bytes, Ordering::Relaxed);
        } else {
            // Key exists — remove then re-insert for upsert semantics.
            // Byte count is approximate: we add the new size but don't
            // subtract the old value's size (acceptable for flush threshold).
            self.map.remove(&key, &guard);
            self.map.insert(key, value, &guard);
        }
    }

    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let owned = key.to_vec();
        let guard = self.map.guard();
        self.map.get(&owned, &guard).cloned()
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
        let guard = self.map.guard();
        let entries = self.map.drain(&guard);
        self.byte_count.store(0, Ordering::Relaxed);
        entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get() {
        let mt = LearnedMemtable::new();
        mt.insert(b"hello".to_vec(), b"world".to_vec());
        assert_eq!(mt.get(b"hello"), Some(b"world".to_vec()));
        assert_eq!(mt.get(b"missing"), None);
        assert_eq!(mt.len(), 1);
    }

    #[test]
    fn upsert_overwrites() {
        let mt = LearnedMemtable::new();
        mt.insert(b"key".to_vec(), b"v1".to_vec());
        mt.insert(b"key".to_vec(), b"v2".to_vec());
        assert_eq!(mt.get(b"key"), Some(b"v2".to_vec()));
    }

    #[test]
    fn delete_inserts_tombstone() {
        let mt = LearnedMemtable::new();
        mt.insert(b"key".to_vec(), b"value".to_vec());
        mt.delete(b"key");
        assert_eq!(mt.get(b"key"), Some(Vec::new()));
    }

    #[test]
    fn drain_returns_sorted_and_empties() {
        let mt = LearnedMemtable::new();
        mt.insert(b"c".to_vec(), b"3".to_vec());
        mt.insert(b"a".to_vec(), b"1".to_vec());
        mt.insert(b"b".to_vec(), b"2".to_vec());

        let entries = mt.drain_sorted();
        let keys: Vec<&[u8]> = entries.iter().map(|(k, _)| k.as_slice()).collect();
        assert_eq!(keys, vec![b"a".as_slice(), b"b", b"c"]);
        assert!(mt.is_empty());
    }
}
