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

/// Number of entries at which the first automatic root rebuild triggers.
/// 64 entries gives FMCD enough data points for a well-fitted model,
/// reducing the number of rebuilds during the critical ramp-up phase
/// (first ~1000 inserts). The slightly longer initial bad-model phase
/// is offset by fewer total rebuilds and better model quality.
const INITIAL_ROOT_REBUILD_THRESHOLD: usize = 64;

/// Growth factor between successive root rebuild thresholds.
/// Schedule: 64, 128, 256, 512, 1024, 2048, ...
/// Total amortized rebuild cost: sum of geometric series ≈ 2N = O(N).
/// Using 2x (not 4x) keeps the root model fresh — critical because keys
/// outside the model's fitted range all clamp to the last slot.
const ROOT_REBUILD_GROWTH_FACTOR: usize = 2;

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

    /// Atomically get an existing value or insert a new one.
    ///
    /// See [`LearnedMap::get_or_insert`] for details.
    pub fn get_or_insert(&self, key: K, value: V) -> &V {
        self.map.get_or_insert(key, value, &self.guard)
    }

    /// Atomically get an existing value or insert a computed one.
    ///
    /// See [`LearnedMap::get_or_insert_with`] for details.
    pub fn get_or_insert_with(&self, key: K, f: impl FnOnce() -> V) -> &V {
        self.map.get_or_insert_with(key, f, &self.guard)
    }

    /// Check whether the map contains a key.
    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key, &self.guard)
    }

    /// Return the approximate number of key-value pairs in the map.
    ///
    /// See [`LearnedMap::len`] for details on relaxed-atomic staleness
    /// under concurrency.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Return `true` if the map contains no entries.
    ///
    /// Subject to the same relaxed-atomic staleness as [`len`](Self::len).
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

    /// Estimate the total heap memory allocated by this map, in bytes.
    ///
    /// See [`LearnedMap::allocated_bytes`] for details.
    pub fn allocated_bytes(&self) -> usize {
        self.map.allocated_bytes(&self.guard)
    }

    /// Return the maximum depth of the tree.
    pub fn max_depth(&self) -> usize {
        self.map.max_depth(&self.guard)
    }

    /// Rebuild the tree from scratch using bulk load.
    pub fn rebuild(&self) {
        self.map.rebuild(&self.guard);
    }

    /// Remove all entries from the map and return them as a sorted `Vec`.
    pub fn drain(&self) -> Vec<(K, V)> {
        self.map.drain(&self.guard)
    }

    /// Remove all entries from the map.
    pub fn clear(&self) {
        self.map.clear(&self.guard);
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
    /// Entry count at which the next automatic root rebuild triggers.
    next_root_rebuild: AtomicUsize,
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
        let root = Node::with_capacity(LinearModel::new(1.0, 0.0), 64);
        let root_atomic = Atomic::new(root);
        Self {
            root: root_atomic,
            len: AtomicUsize::new(0),
            next_root_rebuild: AtomicUsize::new(INITIAL_ROOT_REBUILD_THRESHOLD),
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
        // Build with headroom so appends beyond the loaded range don't all
        // clamp to the last slot. This is the most common pattern: bulk_load
        // historical data, then stream new entries.
        let build_config = Config {
            range_headroom: config.range_headroom.max(1.0),
            ..config
        };
        let root = build::bulk_load(pairs, &build_config)?;
        let root_atomic = Atomic::new(root);
        let next_threshold = pairs.len().saturating_mul(ROOT_REBUILD_GROWTH_FACTOR);
        Ok(Self {
            len: AtomicUsize::new(pairs.len()),
            root: root_atomic,
            next_root_rebuild: AtomicUsize::new(next_threshold),
            config,
        })
    }

    /// Create a learned map from sorted key-value pairs, deduplicating keys.
    ///
    /// When duplicate keys are present, the **last** value for each key is kept,
    /// matching the semantics of `BTreeMap::from_iter`. The input must be sorted
    /// by key in ascending order but may contain duplicates.
    ///
    /// This is equivalent to deduplicating and then calling [`bulk_load`](Self::bulk_load).
    ///
    /// # Errors
    ///
    /// Returns an error if `pairs` is empty (after dedup) or not sorted by key.
    pub fn bulk_load_dedup(pairs: &[(K, V)]) -> Result<Self> {
        Self::bulk_load_dedup_with_config(pairs, Config::default())
    }

    /// Create a learned map from sorted key-value pairs with dedup and config.
    ///
    /// See [`bulk_load_dedup`](Self::bulk_load_dedup) for semantics.
    ///
    /// # Errors
    ///
    /// Returns an error if `pairs` is empty (after dedup) or not sorted by key.
    pub fn bulk_load_dedup_with_config(pairs: &[(K, V)], config: Config) -> Result<Self> {
        if pairs.is_empty() {
            return Err(crate::error::Error::EmptyData);
        }

        // Verify sorted (non-strictly: duplicates allowed).
        for window in pairs.windows(2) {
            if window[0].0 > window[1].0 {
                return Err(crate::error::Error::NotSorted);
            }
        }

        // Dedup: keep last value per key (scan right-to-left, take first unseen).
        let mut deduped = Vec::with_capacity(pairs.len());
        for window in pairs.windows(2) {
            if window[0].0 != window[1].0 {
                deduped.push(window[0].clone());
            }
        }
        // Always include the last element.
        if let Some(last) = pairs.last() {
            deduped.push(last.clone());
        }

        if deduped.is_empty() {
            return Err(crate::error::Error::EmptyData);
        }

        Self::bulk_load_with_config(&deduped, config)
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
    ///
    /// If a concurrent root rebuild replaces the tree, the insert detects the
    /// change and retries against the new root. A `was_new` flag tracks
    /// whether the key was newly inserted across retries so `len` is
    /// incremented exactly once.
    #[allow(clippy::needless_pass_by_value)]
    pub fn insert(&self, key: K, value: V, guard: &Guard) -> bool {
        let mut was_new = false;
        loop {
            let root_shared = self.root.load(Ordering::Acquire, &guard.inner);
            // If root is frozen (tagged), a global rebuild is in progress.
            if root_shared.tag() != 0 {
                std::hint::spin_loop();
                continue;
            }
            // SAFETY: root is always non-null.
            let root = unsafe { root_shared.deref() };
            let result = insert::insert(root, key.clone(), &value, &self.config, &guard.inner);
            // Validate: root wasn't replaced or frozen by a concurrent rebuild.
            if self.root.load(Ordering::Acquire, &guard.inner) != root_shared {
                if result == InsertResult::Inserted {
                    was_new = true;
                }
                continue;
            }
            let is_new = result == InsertResult::Inserted || was_new;
            if is_new {
                let new_len = self.len.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
                self.maybe_rebuild_root(new_len, guard);
            }
            return is_new;
        }
    }

    /// Check if the root should be rebuilt and, if so, claim the rebuild via CAS.
    ///
    /// Only one thread wins the CAS on `next_root_rebuild`. The winner performs
    /// the rebuild inline; all other threads proceed without blocking.
    fn maybe_rebuild_root(&self, current_len: usize, guard: &Guard) {
        if !self.config.auto_rebuild {
            return;
        }

        let threshold = self.next_root_rebuild.load(Ordering::Relaxed);
        if current_len < threshold {
            return;
        }

        // Try to claim the rebuild by CAS-advancing the threshold.
        let next_threshold = threshold.saturating_mul(ROOT_REBUILD_GROWTH_FACTOR);
        if self
            .next_root_rebuild
            .compare_exchange(
                threshold,
                next_threshold,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            self.rebuild(guard);
        }
    }

    /// Remove a key. Returns `true` if the key was present and removed.
    ///
    /// If a concurrent root rebuild replaces the tree, the remove detects the
    /// change and retries against the new root. A `was_removed` flag tracks
    /// whether the key was successfully removed across retries so `len` is
    /// decremented exactly once.
    pub fn remove(&self, key: &K, guard: &Guard) -> bool {
        let mut was_removed = false;
        loop {
            let root_shared = self.root.load(Ordering::Acquire, &guard.inner);
            // If root is frozen (tagged), a global rebuild is in progress.
            if root_shared.tag() != 0 {
                std::hint::spin_loop();
                continue;
            }
            // SAFETY: root is always non-null.
            let root = unsafe { root_shared.deref() };
            let removed = remove::remove(root, key, &self.config, &guard.inner);
            // Validate: root wasn't replaced or frozen by a concurrent rebuild.
            if self.root.load(Ordering::Acquire, &guard.inner) != root_shared {
                if removed {
                    was_removed = true;
                }
                continue;
            }
            let did_remove = removed || was_removed;
            if did_remove {
                self.len.fetch_sub(1, Ordering::Relaxed);
            }
            return did_remove;
        }
    }

    /// Atomically get an existing value or insert a new one.
    ///
    /// If the key already exists, returns a reference to the existing value
    /// without modifying it. If the key is absent, inserts the key-value pair
    /// and returns a reference to the newly inserted value.
    ///
    /// This is atomic with respect to concurrent operations — there is no
    /// TOCTOU race between checking and inserting.
    #[allow(clippy::needless_pass_by_value)]
    pub fn get_or_insert<'g>(&self, key: K, value: V, guard: &'g Guard) -> &'g V {
        let mut was_new = false;
        loop {
            let root_shared = self.root.load(Ordering::Acquire, &guard.inner);
            if root_shared.tag() != 0 {
                std::hint::spin_loop();
                continue;
            }
            // SAFETY: root is always non-null.
            let root = unsafe { root_shared.deref() };
            let (val, result) =
                insert::get_or_insert(root, key.clone(), &value, &self.config, &guard.inner);
            // Validate: root wasn't replaced or frozen by a concurrent rebuild.
            if self.root.load(Ordering::Acquire, &guard.inner) != root_shared {
                if result == InsertResult::Inserted {
                    was_new = true;
                }
                continue;
            }
            let is_new = result == InsertResult::Inserted || was_new;
            if is_new {
                let new_len = self.len.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
                self.maybe_rebuild_root(new_len, guard);
            }
            return val;
        }
    }

    /// Atomically get an existing value or insert a computed one.
    ///
    /// Like [`get_or_insert`](Self::get_or_insert), but the value is lazily
    /// computed by `f` only if the key is absent. If the key is present, `f`
    /// is never called.
    pub fn get_or_insert_with<'g>(&self, key: K, f: impl FnOnce() -> V, guard: &'g Guard) -> &'g V {
        // Fast path: key already exists.
        if let Some(val) = self.get(&key, guard) {
            return val;
        }
        // Slow path: compute value and atomically insert-or-get.
        let value = f();
        self.get_or_insert(key, value, guard)
    }

    /// Check whether the map contains a key.
    pub fn contains_key(&self, key: &K, guard: &Guard) -> bool {
        self.get(key, guard).is_some()
    }

    /// Return the approximate number of key-value pairs in the map.
    ///
    /// Uses a relaxed atomic load internally. Under concurrent inserts or
    /// removes the returned value may be slightly stale — it is **not**
    /// linearizable with respect to other operations. For an exact count,
    /// call [`iter`](Self::iter) and count the entries, which gives a
    /// consistent snapshot under the epoch guard.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// Return `true` if the map contains no entries.
    ///
    /// Subject to the same relaxed-atomic staleness as [`len`](Self::len).
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
        Iter::with_hint(root, &guard.inner, self.len())
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

    /// Estimate the total heap memory allocated by this map, in bytes.
    ///
    /// Walks the entire tree and sums up node structs, slot arrays, and
    /// per-entry allocations. This is an approximation — it does not account
    /// for allocator overhead, alignment padding, or epoch-deferred garbage.
    ///
    /// Useful for monitoring memory usage in long-running processes.
    pub fn allocated_bytes(&self, guard: &Guard) -> usize {
        let root_shared = self.root.load(Ordering::Acquire, &guard.inner);
        // SAFETY: root is always non-null.
        let root = unsafe { root_shared.deref() };
        root.allocated_bytes(&guard.inner)
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
    /// The root is *frozen* (tagged) during the rebuild so concurrent inserts
    /// spin-wait and then retry against the new root. In-flight inserts that
    /// loaded the old root before the freeze detect the change via post-insert
    /// validation in [`LearnedMap::insert`] and retry automatically.
    pub fn rebuild(&self, guard: &Guard) {
        let root_shared = self.root.load(Ordering::Acquire, &guard.inner);
        if root_shared.is_null() || root_shared.tag() != 0 {
            return;
        }
        // SAFETY: root is not null.
        let root = unsafe { root_shared.deref() };

        // Freeze the root: tag it so concurrent inserts spin-wait instead
        // of operating on the old tree during the rebuild.
        let frozen = root_shared.with_tag(1);
        if self
            .root
            .compare_exchange(
                root_shared,
                frozen,
                Ordering::AcqRel,
                Ordering::Acquire,
                &guard.inner,
            )
            .is_err()
        {
            return;
        }

        let pairs = iter::sorted_pairs(root, &guard.inner);
        if pairs.is_empty() {
            // Unfreeze.
            let _ = self.root.compare_exchange(
                frozen,
                root_shared,
                Ordering::AcqRel,
                Ordering::Relaxed,
                &guard.inner,
            );
            return;
        }

        // Root rebuilds use range headroom so the model covers beyond the
        // current key range. Without this, all keys inserted after the rebuild
        // that exceed the training range clamp to the last slot.
        let rebuild_config = Config {
            range_headroom: 1.0,
            ..self.config.clone()
        };
        let Ok(new_root) = build::bulk_load(&pairs, &rebuild_config) else {
            // Unfreeze.
            let _ = self.root.compare_exchange(
                frozen,
                root_shared,
                Ordering::AcqRel,
                Ordering::Relaxed,
                &guard.inner,
            );
            return;
        };
        let new_owned = Owned::new(new_root);
        if self
            .root
            .compare_exchange(
                frozen,
                new_owned,
                Ordering::AcqRel,
                Ordering::Acquire,
                &guard.inner,
            )
            .is_ok()
        {
            // SAFETY: CAS succeeded; old root is unreachable to new readers.
            // In-flight inserts detect the change via map::insert validation.
            unsafe {
                guard.inner.defer_destroy(root_shared);
            }
            // Don't reset len — it's maintained solely by map::insert
            // (fetch_add) and map::remove (fetch_sub). Resetting via store
            // would race with concurrent fetch_adds.
            let count = pairs.len();
            // Reset the rebuild schedule relative to the new tree's size.
            self.next_root_rebuild.store(
                count.saturating_mul(ROOT_REBUILD_GROWTH_FACTOR),
                Ordering::Relaxed,
            );
        } else {
            // CAS failed after freeze — shouldn't happen, but unfreeze.
            let _ = self.root.compare_exchange(
                frozen,
                root_shared,
                Ordering::AcqRel,
                Ordering::Relaxed,
                &guard.inner,
            );
        }
    }

    /// Remove all entries from the map and return them as a sorted `Vec`.
    ///
    /// Uses the same freeze protocol as [`rebuild`](Self::rebuild) to coordinate
    /// with concurrent inserts. After draining, the map has a fresh root identical
    /// to one created by [`new`](Self::new).
    ///
    /// Returns an empty `Vec` if the map is already empty or if the freeze CAS
    /// fails (another thread is rebuilding concurrently).
    pub fn drain(&self, guard: &Guard) -> Vec<(K, V)> {
        let root_shared = self.root.load(Ordering::Acquire, &guard.inner);
        if root_shared.is_null() || root_shared.tag() != 0 {
            return Vec::new();
        }

        // Freeze the root so concurrent inserts spin-wait.
        let frozen = root_shared.with_tag(1);
        if self
            .root
            .compare_exchange(
                root_shared,
                frozen,
                Ordering::AcqRel,
                Ordering::Acquire,
                &guard.inner,
            )
            .is_err()
        {
            return Vec::new();
        }

        // SAFETY: root is not null (checked above) and frozen by us.
        let root = unsafe { root_shared.deref() };
        let pairs = iter::sorted_pairs(root, &guard.inner);

        let new_root = Node::with_capacity(LinearModel::new(1.0, 0.0), 64);
        let new_owned = Owned::new(new_root);
        if self
            .root
            .compare_exchange(
                frozen,
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
            self.len.store(0, Ordering::Relaxed);
            self.next_root_rebuild
                .store(INITIAL_ROOT_REBUILD_THRESHOLD, Ordering::Relaxed);
        } else {
            let _ = self.root.compare_exchange(
                frozen,
                root_shared,
                Ordering::AcqRel,
                Ordering::Relaxed,
                &guard.inner,
            );
        }

        pairs
    }

    /// Remove all entries from the map, resetting it to an empty state.
    ///
    /// Uses the same freeze protocol as [`rebuild`](Self::rebuild) to coordinate
    /// with concurrent inserts. After clearing, the map has a fresh root identical
    /// to one created by [`new`](Self::new).
    pub fn clear(&self, guard: &Guard) {
        let root_shared = self.root.load(Ordering::Acquire, &guard.inner);
        if root_shared.is_null() || root_shared.tag() != 0 {
            return;
        }

        // Freeze the root so concurrent inserts spin-wait.
        let frozen = root_shared.with_tag(1);
        if self
            .root
            .compare_exchange(
                root_shared,
                frozen,
                Ordering::AcqRel,
                Ordering::Acquire,
                &guard.inner,
            )
            .is_err()
        {
            return;
        }

        let new_root = Node::with_capacity(LinearModel::new(1.0, 0.0), 64);
        let new_owned = Owned::new(new_root);
        if self
            .root
            .compare_exchange(
                frozen,
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
            self.len.store(0, Ordering::Relaxed);
            self.next_root_rebuild
                .store(INITIAL_ROOT_REBUILD_THRESHOLD, Ordering::Relaxed);
        } else {
            let _ = self.root.compare_exchange(
                frozen,
                root_shared,
                Ordering::AcqRel,
                Ordering::Relaxed,
                &guard.inner,
            );
        }
    }
}

#[cfg(feature = "serde")]
impl<K, V> serde::Serialize for LearnedMap<K, V>
where
    K: Key + serde::Serialize,
    V: Clone + Send + Sync + serde::Serialize,
{
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;

        let guard = self.guard();
        let len = self.len();
        let mut seq = serializer.serialize_seq(Some(len))?;
        for (k, v) in self.iter(&guard) {
            seq.serialize_element(&(k, v))?;
        }
        seq.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, K, V> serde::Deserialize<'de> for LearnedMap<K, V>
where
    K: Key + serde::Deserialize<'de>,
    V: Clone + Send + Sync + serde::Deserialize<'de>,
{
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let pairs: Vec<(K, V)> = Vec::deserialize(deserializer)?;
        if pairs.is_empty() {
            return Ok(Self::new());
        }
        Self::bulk_load_dedup(&pairs).map_err(serde::de::Error::custom)
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
    fn bulk_load_dedup_keeps_last() {
        let pairs: Vec<(u64, &str)> = vec![(1, "a"), (1, "A"), (2, "b"), (3, "c"), (3, "C")];
        let map = LearnedMap::bulk_load_dedup(&pairs).unwrap();
        let g = map.guard();
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&1, &g), Some(&"A"));
        assert_eq!(map.get(&2, &g), Some(&"b"));
        assert_eq!(map.get(&3, &g), Some(&"C"));
    }

    #[test]
    fn bulk_load_dedup_no_duplicates() {
        let pairs: Vec<(u64, u64)> = (0..50).map(|i| (i, i * 10)).collect();
        let map = LearnedMap::bulk_load_dedup(&pairs).unwrap();
        let g = map.guard();
        assert_eq!(map.len(), 50);
        for (k, v) in &pairs {
            assert_eq!(map.get(k, &g), Some(v));
        }
    }

    #[test]
    fn bulk_load_dedup_all_same_key() {
        let pairs: Vec<(u64, u64)> = (0..10).map(|i| (42, i)).collect();
        let map = LearnedMap::bulk_load_dedup(&pairs).unwrap();
        let g = map.guard();
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&42, &g), Some(&9));
    }

    #[test]
    fn bulk_load_dedup_empty() {
        let result = LearnedMap::<u64, u64>::bulk_load_dedup(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn bulk_load_dedup_not_sorted() {
        let pairs: Vec<(u64, u64)> = vec![(3, 0), (1, 0), (2, 0)];
        let result = LearnedMap::bulk_load_dedup(&pairs);
        assert!(result.is_err());
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

    #[test]
    fn auto_root_rebuild_from_empty() {
        let map = LearnedMap::new();
        let g = map.guard();
        for i in 0..200u64 {
            map.insert(i, i, &g);
        }
        let g2 = map.guard();
        let depth = map.max_depth(&g2);
        assert!(
            depth <= 12,
            "depth {depth} too high after auto root rebuild"
        );
        for i in 0..200u64 {
            assert_eq!(map.get(&i, &g2), Some(&i), "key {i} missing");
        }
    }

    #[test]
    fn auto_root_rebuild_disabled() {
        let map = LearnedMap::with_config(Config::new().auto_rebuild(false));
        let g = map.guard();
        for i in 0..200u64 {
            map.insert(i, i, &g);
        }
        let depth = map.max_depth(&g);
        assert!(depth > 5, "depth {depth} too low without auto rebuild");
    }

    #[test]
    fn bulk_load_no_early_rebuild() {
        let pairs: Vec<(u64, u64)> = (0..100).map(|i| (i, i)).collect();
        let map = LearnedMap::bulk_load(&pairs).unwrap();
        let g = map.guard();
        let depth = map.max_depth(&g);
        assert!(depth <= 3, "bulk-loaded tree depth {depth} too high");
        assert_eq!(map.len(), 100);
    }

    #[test]
    fn manual_rebuild_resets_threshold() {
        let map = LearnedMap::new();
        let g = map.guard();
        for i in 0..50u64 {
            map.insert(i, i, &g);
        }
        map.rebuild(&g);
        let g2 = map.guard();
        for i in 50..150u64 {
            map.insert(i, i, &g2);
        }
        assert_eq!(map.len(), 150);
        for i in 0..150u64 {
            assert_eq!(map.get(&i, &g2), Some(&i));
        }
    }

    #[test]
    fn clear_empties_map() {
        let map = LearnedMap::new();
        let g = map.guard();
        for i in 0..100u64 {
            map.insert(i, i, &g);
        }
        assert_eq!(map.len(), 100);
        map.clear(&g);
        let g2 = map.guard();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        for i in 0..100u64 {
            assert_eq!(map.get(&i, &g2), None);
        }
    }

    #[test]
    fn clear_then_reinsert() {
        let map = LearnedMap::new();
        let g = map.guard();
        for i in 0..50u64 {
            map.insert(i, i * 10, &g);
        }
        map.clear(&g);
        let g2 = map.guard();
        for i in 0..30u64 {
            map.insert(i + 100, i, &g2);
        }
        assert_eq!(map.len(), 30);
        assert_eq!(map.get(&100, &g2), Some(&0));
        assert_eq!(map.get(&0, &g2), None);
    }

    #[test]
    fn clear_empty_is_noop() {
        let map = LearnedMap::<u64, u64>::new();
        let g = map.guard();
        map.clear(&g);
        assert!(map.is_empty());
    }

    #[test]
    fn map_ref_clear() {
        let map = LearnedMap::new();
        let m = map.pin();
        m.insert(1u64, "a");
        m.insert(2, "b");
        assert_eq!(m.len(), 2);
        m.clear();
        assert!(m.is_empty());
        assert_eq!(m.get(&1), None);
    }

    #[test]
    fn drain_returns_sorted_entries() {
        let map = LearnedMap::new();
        let g = map.guard();
        for i in (0..50u64).rev() {
            map.insert(i, i * 10, &g);
        }
        assert_eq!(map.len(), 50);
        let drained = map.drain(&g);
        assert_eq!(drained.len(), 50);
        // Verify sorted order
        for w in drained.windows(2) {
            assert!(w[0].0 < w[1].0);
        }
        // Verify contents
        for (i, (k, v)) in drained.iter().enumerate() {
            assert_eq!(*k, i as u64);
            assert_eq!(*v, (i as u64) * 10);
        }
        // Map should be empty after drain
        let g2 = map.guard();
        assert!(map.is_empty());
        assert_eq!(map.get(&0, &g2), None);
    }

    #[test]
    fn drain_empty_returns_empty() {
        let map = LearnedMap::<u64, u64>::new();
        let g = map.guard();
        let drained = map.drain(&g);
        assert!(drained.is_empty());
        assert!(map.is_empty());
    }

    #[test]
    fn drain_then_reinsert() {
        let map = LearnedMap::new();
        let g = map.guard();
        for i in 0..30u64 {
            map.insert(i, i, &g);
        }
        let drained = map.drain(&g);
        assert_eq!(drained.len(), 30);
        let g2 = map.guard();
        for i in 100..110u64 {
            map.insert(i, i, &g2);
        }
        assert_eq!(map.len(), 10);
        assert_eq!(map.get(&100, &g2), Some(&100));
        assert_eq!(map.get(&0, &g2), None);
    }

    #[test]
    fn map_ref_drain() {
        let map = LearnedMap::new();
        let m = map.pin();
        m.insert(3u64, "c");
        m.insert(1, "a");
        m.insert(2, "b");
        let drained = m.drain();
        assert_eq!(drained, vec![(1, "a"), (2, "b"), (3, "c")]);
        assert!(m.is_empty());
    }

    #[test]
    fn allocated_bytes_empty() {
        let map = LearnedMap::<u64, u64>::new();
        let g = map.guard();
        let bytes = map.allocated_bytes(&g);
        // An empty map still has the root node + slot array.
        assert!(bytes > 0, "empty map should have non-zero allocation");
    }

    #[test]
    fn allocated_bytes_grows_with_entries() {
        let map = LearnedMap::new();
        let g = map.guard();
        let empty_bytes = map.allocated_bytes(&g);

        for i in 0..100u64 {
            map.insert(i, i, &g);
        }
        let g2 = map.guard();
        let full_bytes = map.allocated_bytes(&g2);
        assert!(
            full_bytes > empty_bytes,
            "100 entries should use more memory than empty: {full_bytes} vs {empty_bytes}"
        );
    }

    #[test]
    fn allocated_bytes_bulk_load() {
        let pairs: Vec<(u64, u64)> = (0..500).map(|i| (i, i)).collect();
        let map = LearnedMap::bulk_load(&pairs).unwrap();
        let g = map.guard();
        let bytes = map.allocated_bytes(&g);
        // Sanity: at minimum each entry occupies size_of key + value in a SlotInner.
        let min_data_bytes = 500 * std::mem::size_of::<u64>() * 2;
        assert!(
            bytes > min_data_bytes,
            "allocated_bytes {bytes} is less than minimum data size {min_data_bytes}"
        );
    }

    #[test]
    fn map_ref_allocated_bytes() {
        let map = LearnedMap::new();
        let m = map.pin();
        m.insert(1u64, 1u64);
        m.insert(2, 2);
        let bytes = m.allocated_bytes();
        assert!(bytes > 0);
    }
}
