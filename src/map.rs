//! The primary public type: [`LearnedMap`].

use crate::build;
use crate::config::Config;
use crate::error::Result;
use crate::insert;
use crate::iter::{self, Iter};
use crate::key::Key;
use crate::lookup;
use crate::model::LinearModel;
use crate::node::Node;
use crate::remove;

/// A sorted key-value map backed by a learned index.
///
/// Uses piecewise linear models to predict key positions, achieving O(1)
/// expected lookup time for keys matching the data distribution.
///
/// # Phase 1 (Current)
///
/// This is a single-threaded implementation. All mutating operations
/// require `&mut self`. Phase 2 will make all operations take `&self`
/// for concurrent access.
///
/// # Example
///
/// ```
/// use scry_index::LearnedMap;
///
/// let mut map = LearnedMap::new();
/// map.insert(3u64, "three");
/// map.insert(1, "one");
/// map.insert(2, "two");
///
/// assert_eq!(map.get(&1), Some(&"one"));
/// assert_eq!(map.len(), 3);
/// ```
#[derive(Debug)]
pub struct LearnedMap<K: Key, V> {
    root: Node<K, V>,
    len: usize,
    config: Config,
}

impl<K: Key, V: Clone> LearnedMap<K, V> {
    /// Create a new empty learned map with default configuration.
    pub fn new() -> Self {
        Self::with_config(Config::default())
    }

    /// Create a new empty learned map with the given configuration.
    pub fn with_config(config: Config) -> Self {
        // Start with a small root node. The model doesn't matter yet
        // because all slots are empty — any key will predict some slot.
        let root = Node::with_capacity(LinearModel::new(0.01, 0.0), 16);
        Self {
            root,
            len: 0,
            config,
        }
    }

    /// Create a learned map from sorted key-value pairs.
    ///
    /// This is significantly faster than inserting one-by-one because it
    /// builds the tree structure optimally using FMCD model fitting.
    ///
    /// # Errors
    ///
    /// Returns an error if `pairs` is empty or not sorted by key.
    pub fn bulk_load(pairs: &[(K, V)]) -> Result<Self> {
        Self::bulk_load_with_config(pairs, Config::default())
    }

    /// Create a learned map from sorted key-value pairs with configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if `pairs` is empty or not sorted by key.
    pub fn bulk_load_with_config(pairs: &[(K, V)], config: Config) -> Result<Self> {
        let root = build::bulk_load(pairs, &config)?;
        Ok(Self {
            len: pairs.len(),
            root,
            config,
        })
    }

    /// Look up a key, returning a reference to the value if found.
    pub fn get(&self, key: &K) -> Option<&V> {
        lookup::get(&self.root, key)
    }

    /// Look up a key, returning a mutable reference to the value.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        lookup::get_mut(&mut self.root, key)
    }

    /// Insert a key-value pair.
    ///
    /// Returns the previous value if the key already existed.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let old = insert::insert(&mut self.root, key, value, &self.config);
        if old.is_none() {
            self.len += 1;
        }
        old
    }

    /// Remove a key, returning the value if it existed.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let removed = remove::remove(&mut self.root, key);
        if removed.is_some() {
            self.len -= 1;
        }
        removed
    }

    /// Check whether the map contains a key.
    pub fn contains_key(&self, key: &K) -> bool {
        lookup::contains_key(&self.root, key)
    }

    /// Return the number of key-value pairs in the map.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Return `true` if the map contains no entries.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Iterate over all key-value pairs in sorted order.
    ///
    /// Note: this collects and sorts internally because the tree's DFS
    /// traversal may not be perfectly sorted across child node boundaries.
    pub fn iter_sorted(&self) -> impl Iterator<Item = (K, V)> + '_ {
        iter::sorted_pairs(&self.root).into_iter()
    }

    /// Return the maximum depth of the tree.
    ///
    /// Useful for diagnostics — a well-fit model should keep depth low.
    pub fn max_depth(&self) -> usize {
        self.root.max_depth()
    }
}

impl<K: Key, V: Clone> Default for LearnedMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key, V: Clone> Extend<(K, V)> for LearnedMap<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K: Key, V: Clone> FromIterator<(K, V)> for LearnedMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut map = Self::new();
        map.extend(iter);
        map
    }
}

/// Iterator over references — delegates to `Iter`.
impl<'a, K: Key, V> IntoIterator for &'a LearnedMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(&self.root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_map_is_empty() {
        let map = LearnedMap::<u64, ()>::new();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn insert_and_get() {
        let mut map = LearnedMap::new();
        map.insert(42u64, "hello");
        assert_eq!(map.get(&42), Some(&"hello"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn insert_duplicate_updates() {
        let mut map = LearnedMap::new();
        assert!(map.insert(1u64, "one").is_none());
        assert_eq!(map.insert(1, "ONE"), Some("one"));
        assert_eq!(map.get(&1), Some(&"ONE"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn remove_existing() {
        let mut map = LearnedMap::new();
        map.insert(1u64, "a");
        map.insert(2, "b");
        assert_eq!(map.remove(&1), Some("a"));
        assert_eq!(map.len(), 1);
        assert!(!map.contains_key(&1));
        assert!(map.contains_key(&2));
    }

    #[test]
    fn remove_missing() {
        let mut map = LearnedMap::new();
        map.insert(1u64, "a");
        assert!(map.remove(&99).is_none());
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn bulk_load_basic() {
        let pairs: Vec<(u64, u64)> = (0..100).map(|i| (i, i * 10)).collect();
        let map = LearnedMap::bulk_load(&pairs).unwrap();
        assert_eq!(map.len(), 100);
        for (k, v) in &pairs {
            assert_eq!(map.get(k), Some(v));
        }
    }

    #[test]
    fn bulk_load_then_insert() {
        let pairs: Vec<(u64, u64)> = vec![(10, 1), (20, 2), (30, 3)];
        let mut map = LearnedMap::bulk_load(&pairs).unwrap();
        map.insert(15, 15);
        map.insert(25, 25);
        assert_eq!(map.len(), 5);
        assert_eq!(map.get(&15), Some(&15));
        assert_eq!(map.get(&25), Some(&25));
    }

    #[test]
    fn from_iterator() {
        let map: LearnedMap<u64, &str> =
            vec![(1, "a"), (2, "b"), (3, "c")].into_iter().collect();
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&2), Some(&"b"));
    }

    #[test]
    fn extend_map() {
        let mut map = LearnedMap::new();
        map.insert(1u64, 10);
        map.extend(vec![(2, 20), (3, 30)]);
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn iter_sorted_order() {
        let mut map = LearnedMap::new();
        map.insert(30u64, "c");
        map.insert(10, "a");
        map.insert(20, "b");

        let items: Vec<(u64, &str)> = map.iter_sorted().collect();
        assert_eq!(items, vec![(10, "a"), (20, "b"), (30, "c")]);
    }

    #[test]
    fn into_iter_ref() {
        let mut map = LearnedMap::new();
        map.insert(1u64, "one");
        map.insert(2, "two");

        let items: Vec<_> = (&map).into_iter().collect();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn get_mut_modifies() {
        let mut map = LearnedMap::new();
        map.insert(1u64, 100);
        if let Some(v) = map.get_mut(&1) {
            *v = 999;
        }
        assert_eq!(map.get(&1), Some(&999));
    }

    #[test]
    fn max_depth_bounded() {
        let pairs: Vec<(u64, u64)> = (0..1000).map(|i| (i, i)).collect();
        let map = LearnedMap::bulk_load(&pairs).unwrap();
        // With expansion_factor 2.0 and sequential keys, depth should be very low
        assert!(
            map.max_depth() <= 5,
            "depth {} is too high for 1000 sequential keys",
            map.max_depth()
        );
    }

    #[test]
    fn stress_insert_lookup_remove() {
        let mut map = LearnedMap::new();
        let n = 500u64;

        // Insert
        for i in 0..n {
            map.insert(i * 3, i);
        }
        assert_eq!(map.len(), n as usize);

        // Verify all present
        for i in 0..n {
            assert_eq!(map.get(&(i * 3)), Some(&i), "key {} missing", i * 3);
        }

        // Remove even-indexed
        for i in (0..n).filter(|i| i % 2 == 0) {
            map.remove(&(i * 3));
        }
        assert_eq!(map.len(), (n / 2) as usize);

        // Verify odd-indexed still present
        for i in (0..n).filter(|i| i % 2 != 0) {
            assert_eq!(map.get(&(i * 3)), Some(&i));
        }
    }
}
