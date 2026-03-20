//! A sorted set backed by a learned index.

use crate::config::Config;
use crate::error::Result;
use crate::key::Key;
use crate::map::{Guard, LearnedMap, MapRef};

/// A sorted set backed by a learned index.
///
/// This is a thin wrapper around [`LearnedMap<K, ()>`].
///
/// All operations take `&self` and are safe to call from multiple threads.
#[derive(Debug)]
pub struct LearnedSet<K: Key> {
    inner: LearnedMap<K, ()>,
}

/// A convenience handle that bundles a set reference with an epoch guard.
pub struct SetRef<'a, K: Key> {
    inner: MapRef<'a, K, ()>,
}

impl<K: Key> SetRef<'_, K> {
    /// Insert a key. Returns `true` if the key was newly inserted.
    pub fn insert(&self, key: K) -> bool {
        self.inner.insert(key, ())
    }

    /// Remove a key. Returns `true` if the key was present.
    pub fn remove(&self, key: &K) -> bool {
        self.inner.remove(key)
    }

    /// Check whether the set contains a key.
    pub fn contains(&self, key: &K) -> bool {
        self.inner.contains_key(key)
    }

    /// Return the number of elements (approximate).
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Return `true` if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<K: Key> LearnedSet<K> {
    /// Create a new empty set.
    pub fn new() -> Self {
        Self {
            inner: LearnedMap::new(),
        }
    }

    /// Create a new set with the given configuration.
    pub fn with_config(config: Config) -> Self {
        Self {
            inner: LearnedMap::with_config(config),
        }
    }

    /// Create a set from sorted keys.
    ///
    /// # Errors
    ///
    /// Returns an error if `keys` is empty or not sorted.
    pub fn bulk_load(keys: &[K]) -> Result<Self> {
        let pairs: Vec<(K, ())> = keys.iter().map(|&k| (k, ())).collect();
        Ok(Self {
            inner: LearnedMap::bulk_load(&pairs)?,
        })
    }

    /// Acquire an epoch guard.
    pub fn guard(&self) -> Guard {
        self.inner.guard()
    }

    /// Pin the current epoch and return a [`SetRef`] convenience handle.
    pub fn pin(&self) -> SetRef<'_, K> {
        SetRef {
            inner: self.inner.pin(),
        }
    }

    /// Insert a key. Returns `true` if the key was newly inserted.
    pub fn insert(&self, key: K, guard: &Guard) -> bool {
        self.inner.insert(key, (), guard)
    }

    /// Remove a key. Returns `true` if the key was present.
    pub fn remove(&self, key: &K, guard: &Guard) -> bool {
        self.inner.remove(key, guard)
    }

    /// Check whether the set contains a key.
    pub fn contains(&self, key: &K, guard: &Guard) -> bool {
        self.inner.contains_key(key, guard)
    }

    /// Return the number of elements (approximate).
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Return `true` if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<K: Key> Default for LearnedSet<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key> FromIterator<K> for LearnedSet<K> {
    fn from_iter<I: IntoIterator<Item = K>>(iter: I) -> Self {
        let set = Self::new();
        let guard = set.guard();
        for k in iter {
            set.insert(k, &guard);
        }
        set
    }
}

impl<K: Key> Extend<K> for LearnedSet<K> {
    fn extend<I: IntoIterator<Item = K>>(&mut self, iter: I) {
        let guard = self.guard();
        for k in iter {
            self.insert(k, &guard);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_set_ops() {
        let set = LearnedSet::new();
        let g = set.guard();
        assert!(set.insert(1u64, &g));
        assert!(set.insert(2, &g));
        assert!(!set.insert(1, &g)); // duplicate
        assert_eq!(set.len(), 2);
        assert!(set.contains(&1, &g));
        assert!(set.remove(&1, &g));
        assert!(!set.contains(&1, &g));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn from_iterator() {
        let set: LearnedSet<u64> = vec![3, 1, 2].into_iter().collect();
        let g = set.guard();
        assert_eq!(set.len(), 3);
        assert!(set.contains(&1, &g));
        assert!(set.contains(&2, &g));
        assert!(set.contains(&3, &g));
    }

    #[test]
    fn bulk_load_set() {
        let keys: Vec<u64> = (0..100).collect();
        let set = LearnedSet::bulk_load(&keys).unwrap();
        let g = set.guard();
        assert_eq!(set.len(), 100);
        for k in &keys {
            assert!(set.contains(k, &g));
        }
    }

    #[test]
    fn set_ref_convenience() {
        let set = LearnedSet::new();
        let s = set.pin();
        assert!(s.insert(10u64));
        assert!(s.insert(20));
        assert!(!s.insert(10));
        assert_eq!(s.len(), 2);
        assert!(s.contains(&10));
        assert!(s.remove(&10));
        assert!(!s.contains(&10));
        assert_eq!(s.len(), 1);
    }
}
