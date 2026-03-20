//! The primary public types: [`LearnedMap`], [`Guard`], and [`MapRef`].
#![allow(unsafe_code)]

use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_epoch::{self as epoch, Atomic, Owned};

use crate::build;
use crate::config::Config;
use crate::error::Result;
use crate::insert::{self, InsertResult};
use std::ops::RangeBounds;

use crate::iter::{self, Iter, Range};
use crate::key::Key;
use crate::lookup;
use crate::model::LinearModel;
use crate::node::Node;
use crate::remove;

/// An epoch guard that keeps the current thread pinned.
///
/// While a guard exists, any memory retired during this epoch will not be
/// reclaimed. Guards should be short-lived to avoid delaying reclamation.
///
/// Obtain a guard via [`LearnedMap::guard`] or use [`LearnedMap::pin`] for
/// the convenience [`MapRef`] wrapper.
pub struct Guard {
    inner: epoch::Guard,
}

impl Guard {
    fn new(inner: epoch::Guard) -> Self {
        Self { inner }
    }
}

/// A convenience handle that bundles a map reference with an epoch guard.
///
/// All operations on `MapRef` are forwarded to the underlying [`LearnedMap`]
/// using the guard owned by this handle. This avoids passing a guard to
/// every method call.
///
/// # Example
///
/// ```
/// use scry_index::LearnedMap;
///
/// let map = LearnedMap::new();
/// let m = map.pin();
/// m.insert(1u64, "hello");
/// assert_eq!(m.get(&1), Some(&"hello"));
/// ```
pub struct MapRef<'a, K: Key, V> {
    map: &'a LearnedMap<K, V>,
    guard: Guard,
}

impl<K: Key, V: Clone + Send + Sync> MapRef<'_, K, V> {
    /// Look up a key, returning a reference to the value if found.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key, &self.guard)
    }

    /// Insert a key-value pair. Returns `true` if the key was newly inserted.
    pub fn insert(&self, key: K, value: V) -> bool {
        self.map.insert(key, value, &self.guard)
    }

    /// Remove a key. Returns `true` if the key was present and removed.
    pub fn remove(&self, key: &K) -> bool {
        self.map.remove(key, &self.guard)
    }

    /// Check whether the map contains a key.
    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key, &self.guard)
    }

    /// Return the number of key-value pairs (approximate).
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Return `true` if the map contains no entries.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Iterate over all key-value pairs in sorted order.
    #[allow(clippy::iter_without_into_iter)]
    pub fn iter(&self) -> Iter<'_, K, V> {
        self.map.iter(&self.guard)
    }

    /// Collect all key-value pairs in sorted order (cloned).
    pub fn iter_sorted(&self) -> Vec<(K, V)> {
        self.map.iter_sorted(&self.guard)
    }

    /// Return an iterator over key-value pairs within the given range.
    pub fn range<R: RangeBounds<K>>(&self, range: R) -> Range<'_, K, V> {
        self.map.range(range, &self.guard)
    }

    /// Return the first (minimum) key-value pair.
    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        self.map.first_key_value(&self.guard)
    }

    /// Return the last (maximum) key-value pair.
    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        self.map.last_key_value(&self.guard)
    }

    /// Count the number of entries within the given range.
    pub fn range_count<R: RangeBounds<K>>(&self, range: R) -> usize {
        self.map.range_count(range, &self.guard)
    }

    /// Return the maximum depth of the tree.
    pub fn max_depth(&self) -> usize {
        self.map.max_depth(&self.guard)
    }

    /// Rebuild the tree from scratch using bulk load.
    pub fn rebuild(&self) {
        self.map.rebuild(&self.guard);
    }
}

/// A sorted key-value map backed by a learned index.
///
/// Uses piecewise linear models to predict key positions, achieving O(1)
/// expected lookup time for keys matching the data distribution.
///
/// # Concurrency
///
/// All operations take `&self` and are safe to call from multiple threads.
/// Reads are lock-free (atomic loads under an epoch guard). Writes use
/// compare-and-swap retry loops on individual slots — no global lock.
///
/// # Example
///
/// ```
/// use scry_index::LearnedMap;
///
/// let map = LearnedMap::new();
/// let guard = map.guard();
///
/// map.insert(42u64, "hello", &guard);
/// map.insert(17, "world", &guard);
///
/// assert_eq!(map.get(&42, &guard), Some(&"hello"));
/// assert_eq!(map.get(&99, &guard), None);
/// assert_eq!(map.len(), 2);
/// ```
pub struct LearnedMap<K: Key, V> {
    root: Atomic<Node<K, V>>,
    len: AtomicUsize,
    config: Config,
}

impl<K: Key, V> std::fmt::Debug for LearnedMap<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LearnedMap")
            .field("len", &self.len.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl<K: Key, V: Clone + Send + Sync> LearnedMap<K, V> {
    /// Create a new empty learned map with default configuration.
    pub fn new() -> Self {
        Self::with_config(Config::default())
    }

    /// Create a new empty learned map with the given configuration.
    pub fn with_config(config: Config) -> Self {
        let root = Node::with_capacity(LinearModel::new(0.01, 0.0), 16);
        let root_atomic = Atomic::new(root);
        Self {
            root: root_atomic,
            len: AtomicUsize::new(0),
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
        let root_atomic = Atomic::new(root);
        Ok(Self {
            len: AtomicUsize::new(pairs.len()),
            root: root_atomic,
            config,
        })
    }

    /// Acquire an epoch guard for use with operations on this map.
    ///
    /// The guard pins the current thread to an epoch, preventing any
    /// concurrently retired memory from being reclaimed while the guard
    /// is held. Keep guards short-lived.
    pub fn guard(&self) -> Guard {
        Guard::new(epoch::pin())
    }

    /// Pin the current epoch and return a [`MapRef`] convenience handle.
    ///
    /// This is equivalent to `guard()` + passing the guard to every method,
    /// but more ergonomic for sequences of operations.
    pub fn pin(&self) -> MapRef<'_, K, V> {
        MapRef {
            map: self,
            guard: self.guard(),
        }
    }

    /// Look up a key, returning a reference to the value if found.
    ///
    /// The returned reference is valid for the lifetime of the guard.
    pub fn get<'g>(&self, key: &K, guard: &'g Guard) -> Option<&'g V> {
        let root_shared = self.root.load(Ordering::Acquire, &guard.inner);
        // SAFETY: root is always non-null (set during construction, never nulled).
        let root = unsafe { root_shared.deref() };
        lookup::get(root, key, &guard.inner)
    }

    /// Insert a key-value pair. Returns `true` if the key was newly inserted,
    /// `false` if an existing key's value was updated.
    ///
    /// When `auto_rebuild` is enabled, the insert path tracks descent depth
    /// and triggers a localized subtree rebuild if the depth exceeds the
    /// configured threshold. No global lock is required.
    pub fn insert(&self, key: K, value: V, guard: &Guard) -> bool {
        let root_shared = self.root.load(Ordering::Acquire, &guard.inner);
        // SAFETY: root is always non-null.
        let root = unsafe { root_shared.deref() };
        let result = insert::insert(root, key, &value, &self.config, &guard.inner);
        if result == InsertResult::Inserted {
            self.len.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Remove a key. Returns `true` if the key was present and removed.
    pub fn remove(&self, key: &K, guard: &Guard) -> bool {
        let root_shared = self.root.load(Ordering::Acquire, &guard.inner);
        // SAFETY: root is always non-null.
        let root = unsafe { root_shared.deref() };
        let removed = remove::remove(root, key, &guard.inner);
        if removed {
            self.len.fetch_sub(1, Ordering::Relaxed);
        }
        removed
    }

    /// Check whether the map contains a key.
    pub fn contains_key(&self, key: &K, guard: &Guard) -> bool {
        self.get(key, guard).is_some()
    }

    /// Return the approximate number of key-value pairs in the map.
    ///
    /// This is a relaxed atomic load and may be slightly stale under
    /// concurrent modification.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// Return `true` if the map contains no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over all key-value pairs in sorted order.
    ///
    /// The returned references are valid for the lifetime of the guard.
    pub fn iter<'g>(&self, guard: &'g Guard) -> Iter<'g, K, V> {
        let root_shared = self.root.load(Ordering::Acquire, &guard.inner);
        // SAFETY: root is always non-null.
        let root = unsafe { root_shared.deref() };
        Iter::new(root, &guard.inner)
    }

    /// Collect all key-value pairs in sorted order (cloned).
    ///
    /// Performs a full traversal and clones all entries.
    pub fn iter_sorted(&self, guard: &Guard) -> Vec<(K, V)> {
        let root_shared = self.root.load(Ordering::Acquire, &guard.inner);
        // SAFETY: root is always non-null.
        let root = unsafe { root_shared.deref() };
        iter::sorted_pairs(root, &guard.inner)
    }

    /// Return an iterator over key-value pairs within the given range.
    ///
    /// The iterator yields entries in ascending key order. Uses model-guided
    /// seek for efficient initialization when a start bound is provided.
    ///
    /// Accepts any range syntax: `a..b`, `a..=b`, `a..`, `..b`, `..=b`, `..`.
    pub fn range<'g, R: RangeBounds<K>>(&self, range: R, guard: &'g Guard) -> Range<'g, K, V> {
        let root_shared = self.root.load(Ordering::Acquire, &guard.inner);
        // SAFETY: root is always non-null.
        let root = unsafe { root_shared.deref() };
        Range::new(root, range, &guard.inner)
    }

    /// Return the first (minimum) key-value pair in the map.
    pub fn first_key_value<'g>(&self, guard: &'g Guard) -> Option<(&'g K, &'g V)> {
        let root_shared = self.root.load(Ordering::Acquire, &guard.inner);
        // SAFETY: root is always non-null.
        let root = unsafe { root_shared.deref() };
        iter::first_entry(root, &guard.inner)
    }

    /// Return the last (maximum) key-value pair in the map.
    pub fn last_key_value<'g>(&self, guard: &'g Guard) -> Option<(&'g K, &'g V)> {
        let root_shared = self.root.load(Ordering::Acquire, &guard.inner);
        // SAFETY: root is always non-null.
        let root = unsafe { root_shared.deref() };
        iter::last_entry(root, &guard.inner)
    }

    /// Count the number of entries within the given range.
    pub fn range_count<R: RangeBounds<K>>(&self, range: R, guard: &Guard) -> usize {
        self.range(range, guard).count()
    }

    /// Return the maximum depth of the tree.
    ///
    /// Useful for diagnostics — a well-fit model should keep depth low.
    pub fn max_depth(&self, guard: &Guard) -> usize {
        let root_shared = self.root.load(Ordering::Acquire, &guard.inner);
        // SAFETY: root is always non-null.
        let root = unsafe { root_shared.deref() };
        root.max_depth(&guard.inner)
    }

    /// Rebuild the tree from scratch using bulk load.
    ///
    /// Collects all key-value pairs, sorts them, and rebuilds with optimal
    /// FMCD model fitting. This compacts the tree and restores O(1) lookups
    /// after many incremental inserts.
    ///
    /// This is lock-free: it snapshots the current tree, builds a new one,
    /// and CAS-swaps the root. Concurrent mutations that land in the old tree
    /// between the snapshot and the CAS may be lost (same as any lock-free
    /// compaction). Use for manual compaction when write quiescence is
    /// acceptable, or rely on the automatic localized rebuilds for online use.
    pub fn rebuild(&self, guard: &Guard) {
        let root_shared = self.root.load(Ordering::Acquire, &guard.inner);
        if root_shared.is_null() {
            return;
        }
        // SAFETY: root is not null.
        let root = unsafe { root_shared.deref() };

        let pairs = iter::sorted_pairs(root, &guard.inner);
        if pairs.is_empty() {
            return;
        }

        let Ok(new_root) = build::bulk_load(&pairs, &self.config) else {
            return;
        };
        let new_owned = Owned::new(new_root);
        if self
            .root
            .compare_exchange(
                root_shared,
                new_owned,
                Ordering::AcqRel,
                Ordering::Acquire,
                &guard.inner,
            )
            .is_ok()
        {
            // SAFETY: CAS succeeded; old root is unreachable to new readers.
            unsafe {
                guard.inner.defer_destroy(root_shared);
            }
            self.len.store(pairs.len(), Ordering::Relaxed);
        }
        // On CAS failure: concurrent modification — our rebuilt tree is
        // discarded (the Owned<Node> in the Err is dropped automatically).
    }
}

impl<K: Key, V: Clone + Send + Sync> Default for LearnedMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key, V: Clone + Send + Sync> Extend<(K, V)> for LearnedMap<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        let guard = self.guard();
        for (k, v) in iter {
            self.insert(k, v, &guard);
        }
    }
}

impl<K: Key, V: Clone + Send + Sync> FromIterator<(K, V)> for LearnedMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let map = Self::new();
        let guard = map.guard();
        for (k, v) in iter {
            map.insert(k, v, &guard);
        }
        map
    }
}

impl<K: Key, V> Drop for LearnedMap<K, V> {
    fn drop(&mut self) {
        // We must defer destruction of the root rather than freeing immediately.
        // Our Guard type does not borrow the map, so a caller can hold a Guard
        // (and references from get()) after the map is dropped. Using
        // defer_destroy ensures the tree lives until all such guards are gone.
        //
        // SAFETY: We pin the current epoch and schedule the root for deferred
        // destruction. Crossbeam guarantees the root won't be freed until all
        // guards active at this epoch are dropped. Node::drop (which uses
        // unprotected + into_owned to free children) is safe at that point
        // because no guard can still reference the tree.
        unsafe {
            let guard = epoch::pin();
            let shared = self.root.load(Ordering::Relaxed, &guard);
            if !shared.is_null() {
                guard.defer_destroy(shared);
            }
        }
    }
}

// SAFETY: LearnedMap is Send+Sync when K and V are Send+Sync. All interior
// mutation goes through atomic operations and epoch-based reclamation.
unsafe impl<K: Key, V: Send + Sync> Send for LearnedMap<K, V> {}
unsafe impl<K: Key, V: Send + Sync> Sync for LearnedMap<K, V> {}

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
        let map = LearnedMap::new();
        let g = map.guard();
        assert!(map.insert(42u64, "hello", &g));
        assert_eq!(map.get(&42, &g), Some(&"hello"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn insert_duplicate_updates() {
        let map = LearnedMap::new();
        let g = map.guard();
        assert!(map.insert(1u64, "one", &g));
        assert!(!map.insert(1, "ONE", &g));
        assert_eq!(map.get(&1, &g), Some(&"ONE"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn remove_existing() {
        let map = LearnedMap::new();
        let g = map.guard();
        map.insert(1u64, "a", &g);
        map.insert(2, "b", &g);
        assert!(map.remove(&1, &g));
        assert_eq!(map.len(), 1);
        assert!(!map.contains_key(&1, &g));
        assert!(map.contains_key(&2, &g));
    }

    #[test]
    fn remove_missing() {
        let map = LearnedMap::new();
        let g = map.guard();
        map.insert(1u64, "a", &g);
        assert!(!map.remove(&99, &g));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn bulk_load_basic() {
        let pairs: Vec<(u64, u64)> = (0..100).map(|i| (i, i * 10)).collect();
        let map = LearnedMap::bulk_load(&pairs).unwrap();
        let g = map.guard();
        assert_eq!(map.len(), 100);
        for (k, v) in &pairs {
            assert_eq!(map.get(k, &g), Some(v));
        }
    }

    #[test]
    fn bulk_load_then_insert() {
        let pairs: Vec<(u64, u64)> = vec![(10, 1), (20, 2), (30, 3)];
        let map = LearnedMap::bulk_load(&pairs).unwrap();
        let g = map.guard();
        map.insert(15, 15, &g);
        map.insert(25, 25, &g);
        assert_eq!(map.len(), 5);
        assert_eq!(map.get(&15, &g), Some(&15));
        assert_eq!(map.get(&25, &g), Some(&25));
    }

    #[test]
    fn from_iterator() {
        let map: LearnedMap<u64, &str> = vec![(1, "a"), (2, "b"), (3, "c")].into_iter().collect();
        let g = map.guard();
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&2, &g), Some(&"b"));
    }

    #[test]
    fn extend_map() {
        let mut map = LearnedMap::new();
        {
            let g = map.guard();
            map.insert(1u64, 10, &g);
        }
        map.extend(vec![(2, 20), (3, 30)]);
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn iter_sorted_order() {
        let map = LearnedMap::new();
        let g = map.guard();
        map.insert(30u64, "c", &g);
        map.insert(10, "a", &g);
        map.insert(20, "b", &g);

        let items: Vec<(u64, &str)> = map.iter_sorted(&g);
        assert_eq!(items, vec![(10, "a"), (20, "b"), (30, "c")]);
    }

    #[test]
    fn max_depth_bounded() {
        let pairs: Vec<(u64, u64)> = (0..1000).map(|i| (i, i)).collect();
        let map = LearnedMap::bulk_load(&pairs).unwrap();
        let g = map.guard();
        assert!(
            map.max_depth(&g) <= 5,
            "depth {} is too high for 1000 sequential keys",
            map.max_depth(&g)
        );
    }

    #[test]
    fn stress_insert_lookup_remove() {
        let map = LearnedMap::new();
        let g = map.guard();
        let n = 500u64;

        for i in 0..n {
            map.insert(i * 3, i, &g);
        }
        assert_eq!(map.len(), n as usize);

        for i in 0..n {
            assert_eq!(map.get(&(i * 3), &g), Some(&i), "key {} missing", i * 3);
        }

        for i in (0..n).filter(|i| i % 2 == 0) {
            map.remove(&(i * 3), &g);
        }
        assert_eq!(map.len(), (n / 2) as usize);

        for i in (0..n).filter(|i| i % 2 != 0) {
            assert_eq!(map.get(&(i * 3), &g), Some(&i));
        }
    }

    #[test]
    fn manual_rebuild() {
        let map = LearnedMap::new();
        let g = map.guard();
        for i in (0..100u64).rev() {
            map.insert(i, i * 10, &g);
        }
        let depth_before = map.max_depth(&g);
        map.rebuild(&g);
        let depth_after = map.max_depth(&g);
        assert!(
            depth_after <= depth_before,
            "rebuild didn't help: {depth_before} -> {depth_after}"
        );
        // All keys still present (need fresh guard after rebuild)
        let g2 = map.guard();
        for i in 0..100u64 {
            assert_eq!(map.get(&i, &g2), Some(&(i * 10)));
        }
    }

    #[test]
    fn rebuild_empty_is_noop() {
        let map = LearnedMap::<u64, u64>::new();
        let g = map.guard();
        map.rebuild(&g);
        assert!(map.is_empty());
    }

    #[test]
    fn large_incremental_insert() {
        let map = LearnedMap::new();
        let g = map.guard();
        for i in 0..1000u64 {
            map.insert(i, i, &g);
        }
        assert_eq!(map.len(), 1000);
        for i in 0..1000u64 {
            assert_eq!(map.get(&i, &g), Some(&i));
        }
    }

    #[test]
    fn pin_convenience() {
        let map = LearnedMap::new();
        let m = map.pin();
        m.insert(1u64, "one");
        m.insert(2, "two");
        assert_eq!(m.get(&1), Some(&"one"));
        assert_eq!(m.get(&2), Some(&"two"));
        assert_eq!(m.len(), 2);
        assert!(!m.is_empty());
    }

    #[test]
    fn map_ref_remove() {
        let map = LearnedMap::new();
        let m = map.pin();
        m.insert(10u64, 100);
        m.insert(20, 200);
        assert!(m.remove(&10));
        assert!(!m.remove(&10));
        assert_eq!(m.len(), 1);
        assert!(m.contains_key(&20));
    }

    #[test]
    fn map_ref_iter_sorted() {
        let map = LearnedMap::new();
        let m = map.pin();
        m.insert(3u64, "c");
        m.insert(1, "a");
        m.insert(2, "b");
        let items = m.iter_sorted();
        assert_eq!(items, vec![(1, "a"), (2, "b"), (3, "c")]);
    }
}
