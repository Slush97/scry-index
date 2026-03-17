//! A sorted set backed by a learned index.

use crate::config::Config;
use crate::error::Result;
use crate::key::Key;
use crate::map::LearnedMap;

/// A sorted set backed by a learned index.
///
/// This is a thin wrapper around [`LearnedMap<K, ()>`].
#[derive(Debug)]
pub struct LearnedSet<K: Key> {
    inner: LearnedMap<K, ()>,
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

    /// Insert a key. Returns `true` if the key was newly inserted.
    pub fn insert(&mut self, key: K) -> bool {
        self.inner.insert(key, ()).is_none()
    }

    /// Remove a key. Returns `true` if the key was present.
    pub fn remove(&mut self, key: &K) -> bool {
        self.inner.remove(key).is_some()
    }

    /// Check whether the set contains a key.
    pub fn contains(&self, key: &K) -> bool {
        self.inner.contains_key(key)
    }

    /// Return the number of elements in the set.
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
        let mut set = Self::new();
        for k in iter {
            set.insert(k);
        }
        set
    }
}

impl<K: Key> Extend<K> for LearnedSet<K> {
    fn extend<I: IntoIterator<Item = K>>(&mut self, iter: I) {
        for k in iter {
            self.insert(k);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_set_ops() {
        let mut set = LearnedSet::new();
        assert!(set.insert(1u64));
        assert!(set.insert(2));
        assert!(!set.insert(1)); // duplicate
        assert_eq!(set.len(), 2);
        assert!(set.contains(&1));
        assert!(set.remove(&1));
        assert!(!set.contains(&1));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn from_iterator() {
        let set: LearnedSet<u64> = vec![3, 1, 2].into_iter().collect();
        assert_eq!(set.len(), 3);
        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert!(set.contains(&3));
    }

    #[test]
    fn bulk_load_set() {
        let keys: Vec<u64> = (0..100).collect();
        let set = LearnedSet::bulk_load(&keys).unwrap();
        assert_eq!(set.len(), 100);
        for k in &keys {
            assert!(set.contains(k));
        }
    }
}
