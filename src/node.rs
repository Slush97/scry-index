//! Node types for the learned index tree.
//!
//! Each node contains a linear model and a fixed-size array of inline slots.
//! Slots store keys and values directly in contiguous arrays (no per-entry heap
//! allocation), with child pointers in a separate array for conflict chains.
//!
//! Slot states are tracked via per-slot [`AtomicU8`] bytes:
//! - [`SLOT_EMPTY`][]: unused
//! - [`SLOT_WRITING`]: being claimed by a concurrent insert (transient)
//! - [`SLOT_DATA`]: contains an inline key-value pair (immutable once published)
//! - [`SLOT_CHILD`]: contains a child node pointer (no inline data)
//! - [`SLOT_CHILD_STALE`]: contains a child pointer + stale inline data from
//!   a DATA→CHILD transition
//! - [`SLOT_TOMBSTONE`]: logically removed, inline data is stale
#![allow(unsafe_code)]

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::OnceLock;

use crossbeam_epoch::{self as epoch, Atomic, Guard, Owned};

use crate::key::Key;
use crate::model::LinearModel;

/// Lazily-initialized array of epoch-protected child pointers.
type ChildArray<K, V> = OnceLock<Box<[Atomic<Node<K, V>>]>>;

/// Slot is unused. No inline data, no child.
pub const SLOT_EMPTY: u8 = 0;
/// Slot is being claimed by a concurrent insert (transient).
/// Other threads seeing this state should spin-retry.
pub const SLOT_WRITING: u8 = 1;
/// Slot contains a valid inline key-value pair.
pub const SLOT_DATA: u8 = 2;
/// Slot contains a child node pointer. No inline data was ever written
/// (set during bulk-load construction).
pub const SLOT_CHILD: u8 = 3;
/// Slot contains a child node pointer AND stale inline key-value data
/// from a concurrent DATA→CHILD transition. During drop, both the inline
/// data and the child are cleaned up.
pub const SLOT_CHILD_STALE: u8 = 4;
/// Slot was logically removed. Inline key-value data is stale.
pub const SLOT_TOMBSTONE: u8 = 5;

/// Returns `true` if the state indicates a child node is present.
#[inline]
pub fn is_child(state: u8) -> bool {
    state == SLOT_CHILD || state == SLOT_CHILD_STALE
}

/// A node in the learned index tree.
///
/// Contains a linear model for position prediction and fixed-size arrays
/// for inline key-value storage. Conflicts are resolved by creating child
/// nodes stored via epoch-protected [`Atomic`] pointers.
///
/// Keys and values are stored **inline** in contiguous arrays with no per-entry
/// heap allocation, reducing allocation overhead compared to a
/// pointer-per-entry design.
pub struct Node<K, V> {
    /// The linear model for this node (immutable after construction).
    model: LinearModel,
    /// Per-slot state byte (see `SLOT_*` constants).
    states: Box<[AtomicU8]>,
    /// Inline key storage. Valid when state is DATA, `CHILD_STALE`, or TOMBSTONE.
    keys: Box<[UnsafeCell<MaybeUninit<K>>]>,
    /// Inline value storage. Valid when state is DATA, `CHILD_STALE`, or TOMBSTONE.
    values: Box<[UnsafeCell<MaybeUninit<V>>]>,
    /// Child node pointers. Valid when state is CHILD or `CHILD_STALE`.
    /// Lazily initialized via [`OnceLock`] to avoid allocating a large array
    /// for zero-conflict bulk-loaded nodes that never need children.
    children: ChildArray<K, V>,
    /// Approximate number of data entries in this node (not counting children).
    num_keys: AtomicUsize,
    /// Approximate number of tombstones in this node.
    num_tombstones: AtomicUsize,
    /// Optional boundary key for `Ord`-based binary splits.
    split_key: Option<K>,
}

impl<K, V> std::fmt::Debug for Node<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("model", &self.model)
            .field("capacity", &self.states.len())
            .field("num_keys", &self.num_keys.load(Ordering::Relaxed))
            .field(
                "num_tombstones",
                &self.num_tombstones.load(Ordering::Relaxed),
            )
            .field("has_split_key", &self.split_key.is_some())
            .finish_non_exhaustive()
    }
}

impl<K: Key, V> Node<K, V> {
    /// Create a new node with the given model and array size.
    ///
    /// All slots are initialized to empty. A full children array is allocated
    /// eagerly for conflict resolution. Use [`with_capacity_leaf`] when no
    /// conflicts are expected (e.g., zero-conflict bulk-load nodes) to defer
    /// the children allocation until a child is actually stored.
    pub fn with_capacity(model: LinearModel, array_size: usize) -> Self {
        // Use a single zeroed allocation for states, keys, and values.
        // MaybeUninit and AtomicU8(0) are both zero-initialized, so we can
        // use vec![...; n] which the compiler can lower to calloc/memset.
        let children = OnceLock::new();
        // Eagerly initialize the children array for nodes that expect conflicts.
        let _ = children.set(
            (0..array_size)
                .map(|_| Atomic::null())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        Self {
            model,
            states: (0..array_size)
                .map(|_| AtomicU8::new(SLOT_EMPTY))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            keys: (0..array_size)
                .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            values: (0..array_size)
                .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            children,
            num_keys: AtomicUsize::new(0),
            num_tombstones: AtomicUsize::new(0),
            split_key: None,
        }
    }

    /// Create a leaf node with the given model and array size.
    ///
    /// Like [`with_capacity`] but defers the children array allocation until
    /// a child is actually stored. This saves one heap allocation of
    /// `array_size * size_of::<Atomic<Node>>()` bytes, which is significant
    /// for large zero-conflict bulk-loaded nodes.
    ///
    /// If a concurrent insert later creates a conflict on this node, the
    /// children array is lazily allocated via [`OnceLock`] on first use.
    pub fn with_capacity_leaf(model: LinearModel, array_size: usize) -> Self {
        Self {
            model,
            states: (0..array_size)
                .map(|_| AtomicU8::new(SLOT_EMPTY))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            keys: (0..array_size)
                .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            values: (0..array_size)
                .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            children: OnceLock::new(),
            num_keys: AtomicUsize::new(0),
            num_tombstones: AtomicUsize::new(0),
            split_key: None,
        }
    }

    /// Create a node that splits keys using `Ord` comparison against `boundary`.
    ///
    /// Keys <= `boundary` predict to slot 0; keys > `boundary` predict to the
    /// last slot.
    pub fn with_split_key(boundary: K, array_size: usize) -> Self {
        let mut node = Self::with_capacity(LinearModel::constant(), array_size);
        node.split_key = Some(boundary);
        node
    }

    /// Predict the slot index for a key.
    #[inline]
    pub fn predict_slot(&self, key: &K) -> usize {
        if let Some(ref sk) = self.split_key {
            return if key <= sk {
                0
            } else {
                self.states.len().saturating_sub(1)
            };
        }
        self.model.predict(key, self.states.len())
    }

    /// Return the number of slots in this node.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.states.len()
    }

    // -----------------------------------------------------------------------
    // Children array lazy initialization
    // -----------------------------------------------------------------------

    /// Lazily initialize and return the children array.
    ///
    /// On the first call, allocates an array of `capacity()` null [`Atomic`]
    /// pointers. Subsequent calls return the same array. Thread-safe via
    /// [`OnceLock`].
    #[inline]
    fn ensure_children(&self) -> &[Atomic<Self>] {
        self.children.get_or_init(|| {
            let cap = self.states.len();
            (0..cap)
                .map(|_| Atomic::null())
                .collect::<Vec<_>>()
                .into_boxed_slice()
        })
    }

    // -----------------------------------------------------------------------
    // Construction-time writes (no concurrent access)
    // -----------------------------------------------------------------------

    /// Store an inline key-value pair during construction.
    ///
    /// The slot must be empty. Uses relaxed ordering (no concurrent readers).
    pub fn store_data(&self, idx: usize, key: K, value: V) {
        debug_assert_eq!(
            self.states[idx].load(Ordering::Relaxed),
            SLOT_EMPTY,
            "store_data called on non-empty slot {idx}"
        );
        unsafe {
            (*self.keys[idx].get()) = MaybeUninit::new(key);
            (*self.values[idx].get()) = MaybeUninit::new(value);
        }
        self.states[idx].store(SLOT_DATA, Ordering::Relaxed);
    }

    /// Store a child node during construction.
    ///
    /// The slot must be empty. No inline data is written. Lazily allocates
    /// the children array if this is the first child stored.
    pub fn store_child(&self, idx: usize, child: Self) {
        debug_assert_eq!(
            self.states[idx].load(Ordering::Relaxed),
            SLOT_EMPTY,
            "store_child called on non-empty slot {idx}"
        );
        let children = self.ensure_children();
        unsafe {
            let guard = epoch::unprotected();
            children[idx].store(Owned::new(child).into_shared(guard), Ordering::Relaxed);
        }
        self.states[idx].store(SLOT_CHILD, Ordering::Relaxed);
    }

    // -----------------------------------------------------------------------
    // Concurrent slot access
    // -----------------------------------------------------------------------

    /// Load the state of a slot with Acquire ordering.
    #[inline]
    pub fn slot_state(&self, idx: usize) -> u8 {
        self.states[idx].load(Ordering::Acquire)
    }

    /// Read the inline key at the given slot.
    ///
    /// # Safety
    ///
    /// Caller must ensure the slot has inline data (state is DATA, `CHILD_STALE`,
    /// or TOMBSTONE) and that the data will not be concurrently modified.
    #[inline]
    pub unsafe fn read_key(&self, idx: usize) -> &K {
        (*self.keys[idx].get()).assume_init_ref()
    }

    /// Read the inline value at the given slot.
    ///
    /// # Safety
    ///
    /// Caller must ensure the slot has inline data (state is DATA) and that
    /// the data will not be concurrently modified.
    #[inline]
    pub unsafe fn read_value(&self, idx: usize) -> &V {
        (*self.values[idx].get()).assume_init_ref()
    }

    /// Load the child pointer at the given slot.
    ///
    /// Returns `Shared::null()` if the children array has not been initialized
    /// (e.g., for a leaf node created via [`with_capacity_leaf`]).
    #[inline]
    pub fn load_child<'g>(
        &self,
        idx: usize,
        guard: &'g Guard,
    ) -> crossbeam_epoch::Shared<'g, Self> {
        self.children
            .get()
            .map_or_else(crossbeam_epoch::Shared::null, |children| {
                children[idx].load(Ordering::Acquire, guard)
            })
    }

    /// Get a reference to the child's [`Atomic`] for CAS operations (freeze, etc.).
    ///
    /// Lazily allocates the children array if it has not been initialized.
    #[inline]
    pub fn child_atomic(&self, idx: usize) -> &Atomic<Self> {
        &self.ensure_children()[idx]
    }

    /// Get a reference to the state [`AtomicU8`] for CAS operations.
    #[inline]
    pub fn state_atomic(&self, idx: usize) -> &AtomicU8 {
        &self.states[idx]
    }

    // -----------------------------------------------------------------------
    // Concurrent writes
    // -----------------------------------------------------------------------

    /// Atomically claim an EMPTY slot and write inline key-value data.
    ///
    /// Uses an intermediate WRITING state so that only one thread can claim
    /// a slot. Returns `true` on success, `false` if the slot was not EMPTY.
    pub fn cas_empty_to_data(&self, idx: usize, key: K, value: V) -> bool {
        // CAS EMPTY → WRITING to claim exclusive write access to this slot.
        if self.states[idx]
            .compare_exchange(
                SLOT_EMPTY,
                SLOT_WRITING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return false;
        }
        // We own this slot. Write key+value, then publish.
        unsafe {
            (*self.keys[idx].get()) = MaybeUninit::new(key);
            (*self.values[idx].get()) = MaybeUninit::new(value);
        }
        self.states[idx].store(SLOT_DATA, Ordering::Release);
        true
    }

    /// Atomically claim a TOMBSTONE slot and write new inline key-value data.
    ///
    /// Returns `true` on success. The old stale data is overwritten (safe for
    /// types without Drop; for Drop types the old data leaks until node rebuild).
    pub fn cas_tombstone_to_data(&self, idx: usize, key: K, value: V) -> bool {
        if self.states[idx]
            .compare_exchange(
                SLOT_TOMBSTONE,
                SLOT_WRITING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return false;
        }
        // Overwrite inline storage. The old stale data is not explicitly dropped
        // here. For Copy types this is fine; for Drop types the old data's
        // destructor is skipped (acceptable since the node rebuild will reclaim).
        unsafe {
            (*self.keys[idx].get()) = MaybeUninit::new(key);
            (*self.values[idx].get()) = MaybeUninit::new(value);
        }
        self.states[idx].store(SLOT_DATA, Ordering::Release);
        true
    }

    /// Transition a DATA slot to a `CHILD_STALE` slot with the given child node.
    ///
    /// Clones the existing inline key+value (for the child to use), then stores
    /// the child pointer and transitions the state. The inline data becomes stale.
    ///
    /// Returns `true` on success, `false` if the slot was no longer DATA.
    pub fn cas_data_to_child_stale(&self, idx: usize, child: Self, guard: &Guard) -> bool {
        if self.states[idx]
            .compare_exchange(SLOT_DATA, SLOT_WRITING, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        // We own the slot. Store the child pointer (lazily allocates children array).
        self.ensure_children()[idx].store(Owned::new(child).into_shared(guard), Ordering::Release);
        // Publish: the inline data is now stale but stays for drop correctness.
        self.states[idx].store(SLOT_CHILD_STALE, Ordering::Release);
        true
    }

    /// Transition a DATA slot to TOMBSTONE (logical removal).
    ///
    /// Returns `true` on success.
    pub fn cas_data_to_tombstone(&self, idx: usize) -> bool {
        self.states[idx]
            .compare_exchange(
                SLOT_DATA,
                SLOT_TOMBSTONE,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    // -----------------------------------------------------------------------
    // Counters
    // -----------------------------------------------------------------------

    /// Increment the approximate key count.
    pub fn inc_keys(&self) {
        self.num_keys.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the approximate key count.
    pub fn dec_keys(&self) {
        self.num_keys.fetch_sub(1, Ordering::Relaxed);
    }

    /// Increment the approximate tombstone count.
    pub fn inc_tombstones(&self) {
        self.num_tombstones.fetch_add(1, Ordering::Relaxed);
    }

    /// Return the approximate ratio of tombstones to total capacity.
    pub fn tombstone_ratio(&self) -> f64 {
        let cap = self.states.len();
        if cap == 0 {
            return 0.0;
        }
        self.num_tombstones.load(Ordering::Relaxed) as f64 / cap as f64
    }

    /// Count total keys stored in this node and all descendants.
    pub fn total_keys(&self, guard: &Guard) -> usize {
        let children = self.children.get();
        let mut count = 0;
        for i in 0..self.states.len() {
            let state = self.states[i].load(Ordering::Acquire);
            match state {
                SLOT_DATA => count += 1,
                s if is_child(s) => {
                    if let Some(c) = children {
                        let child_shared = c[i].load(Ordering::Acquire, guard);
                        if !child_shared.is_null() {
                            // SAFETY: child_shared is valid for the guard lifetime.
                            count += unsafe { child_shared.deref() }.total_keys(guard);
                        }
                    }
                }
                _ => {}
            }
        }
        count
    }

    /// Estimate the total heap memory used by this node and all descendants.
    pub fn allocated_bytes(&self, guard: &Guard) -> usize {
        let node_size = std::mem::size_of::<Self>();
        let cap = self.states.len();
        let children = self.children.get();
        let children_cap = children.map_or(0, |c| c.len());
        let arrays_size = cap
            * (std::mem::size_of::<AtomicU8>()
                + std::mem::size_of::<UnsafeCell<MaybeUninit<K>>>()
                + std::mem::size_of::<UnsafeCell<MaybeUninit<V>>>())
            + children_cap * std::mem::size_of::<Atomic<Self>>();

        let mut total = node_size + arrays_size;

        if let Some(c) = children {
            for i in 0..cap {
                let state = self.states[i].load(Ordering::Acquire);
                if is_child(state) {
                    let child_shared = c[i].load(Ordering::Acquire, guard);
                    if !child_shared.is_null() {
                        // SAFETY: child_shared is valid for the guard lifetime.
                        total += unsafe { child_shared.deref() }.allocated_bytes(guard);
                    }
                }
            }
        }

        total
    }

    /// Return the depth of the deepest path from this node.
    pub fn max_depth(&self, guard: &Guard) -> usize {
        let mut max_child_depth = 0;
        if let Some(children) = self.children.get() {
            for i in 0..self.states.len() {
                let state = self.states[i].load(Ordering::Acquire);
                if is_child(state) {
                    let child_shared = children[i].load(Ordering::Acquire, guard);
                    if !child_shared.is_null() {
                        // SAFETY: child_shared is valid for the guard lifetime.
                        let depth = unsafe { child_shared.deref() }.max_depth(guard);
                        max_child_depth = max_child_depth.max(depth);
                    }
                }
            }
        }
        1 + max_child_depth
    }
}

impl<K, V> Drop for Node<K, V> {
    fn drop(&mut self) {
        // SAFETY: We have exclusive access during drop. No other thread can
        // reference this node.
        unsafe {
            let guard = epoch::unprotected();
            let children = self.children.get();
            for i in 0..self.states.len() {
                let state = *self.states[i].get_mut();
                // Drop inline data if it was initialized.
                let has_inline =
                    state == SLOT_DATA || state == SLOT_CHILD_STALE || state == SLOT_TOMBSTONE;
                if has_inline && std::mem::needs_drop::<K>() {
                    std::ptr::drop_in_place((*self.keys[i].get()).as_mut_ptr());
                }
                if has_inline && std::mem::needs_drop::<V>() {
                    std::ptr::drop_in_place((*self.values[i].get()).as_mut_ptr());
                }
                // Drop child node if present.
                if let Some(c) = children {
                    if is_child(state) {
                        let shared = c[i].load(Ordering::Relaxed, guard);
                        if !shared.is_null() {
                            drop(shared.into_owned());
                        }
                    }
                }
            }
        }
    }
}

// SAFETY: Node is Send+Sync when K and V are Send+Sync. All interior mutation
// goes through atomic operations (AtomicU8, AtomicUsize, Atomic<Node>) or
// UnsafeCell guarded by state CAS. The recursive type prevents auto-derivation.
unsafe impl<K: Send + Sync, V: Send + Sync> Send for Node<K, V> {}
unsafe impl<K: Send + Sync, V: Send + Sync> Sync for Node<K, V> {}

#[cfg(test)]
mod tests {
    use super::*;

    fn guard() -> epoch::Guard {
        epoch::pin()
    }

    #[test]
    fn new_node_all_empty() {
        let g = guard();
        let node = Node::<u64, String>::with_capacity(LinearModel::constant(), 10);
        assert_eq!(node.capacity(), 10);
        assert_eq!(node.total_keys(&g), 0);
    }

    #[test]
    fn total_keys_empty() {
        let g = guard();
        let node = Node::<u64, ()>::with_capacity(LinearModel::constant(), 5);
        assert_eq!(node.total_keys(&g), 0);
    }

    #[test]
    fn total_keys_with_data() {
        let g = guard();
        let node = Node::<u64, &str>::with_capacity(LinearModel::constant(), 5);
        node.store_data(0, 1, "a");
        node.inc_keys();
        node.store_data(2, 2, "b");
        node.inc_keys();
        assert_eq!(node.total_keys(&g), 2);
    }

    #[test]
    fn total_keys_with_children() {
        let g = guard();

        let child = Node::<u64, &str>::with_capacity(LinearModel::constant(), 3);
        child.store_data(0, 10, "x");
        child.inc_keys();
        child.store_data(1, 20, "y");
        child.inc_keys();

        let parent = Node::<u64, &str>::with_capacity(LinearModel::constant(), 5);
        parent.store_data(0, 1, "a");
        parent.inc_keys();
        parent.store_child(1, child);

        assert_eq!(parent.total_keys(&g), 3);
    }

    #[test]
    fn max_depth_leaf() {
        let g = guard();
        let node = Node::<u64, ()>::with_capacity(LinearModel::constant(), 5);
        assert_eq!(node.max_depth(&g), 1);
    }

    #[test]
    fn max_depth_nested() {
        let g = guard();
        let leaf = Node::<u64, ()>::with_capacity(LinearModel::constant(), 2);
        let mid = Node::<u64, ()>::with_capacity(LinearModel::constant(), 2);
        mid.store_child(0, leaf);
        let root = Node::<u64, ()>::with_capacity(LinearModel::constant(), 2);
        root.store_child(0, mid);
        assert_eq!(root.max_depth(&g), 3);
    }

    #[test]
    fn store_and_read_data() {
        let node = Node::<u64, i32>::with_capacity(LinearModel::constant(), 4);
        node.store_data(1, 42, 100);
        node.inc_keys();

        assert_eq!(node.slot_state(1), SLOT_DATA);
        unsafe {
            assert_eq!(*node.read_key(1), 42);
            assert_eq!(*node.read_value(1), 100);
        }
    }

    #[test]
    fn cas_empty_to_data_success() {
        let g = guard();
        let node = Node::<u64, &str>::with_capacity(LinearModel::constant(), 4);
        assert!(node.cas_empty_to_data(0, 10, "hello"));
        assert_eq!(node.slot_state(0), SLOT_DATA);
        unsafe {
            assert_eq!(*node.read_key(0), 10);
            assert_eq!(*node.read_value(0), "hello");
        }
        // total_keys scans actual DATA slots, so it's 1 even without inc_keys
        assert_eq!(node.total_keys(&g), 1);
    }

    #[test]
    fn cas_empty_to_data_fails_on_occupied() {
        let node = Node::<u64, u64>::with_capacity(LinearModel::constant(), 4);
        assert!(node.cas_empty_to_data(0, 1, 10));
        // Second attempt should fail
        assert!(!node.cas_empty_to_data(0, 2, 20));
        // Original data preserved
        unsafe {
            assert_eq!(*node.read_key(0), 1);
            assert_eq!(*node.read_value(0), 10);
        }
    }

    #[test]
    fn cas_data_to_child_stale() {
        let g = guard();
        let node = Node::<u64, u64>::with_capacity(LinearModel::constant(), 4);
        node.store_data(0, 10, 100);
        node.inc_keys();

        let child = Node::<u64, u64>::with_capacity(LinearModel::constant(), 2);
        child.store_data(0, 10, 200);
        child.inc_keys();

        assert!(node.cas_data_to_child_stale(0, child, &g));
        assert_eq!(node.slot_state(0), SLOT_CHILD_STALE);

        let child_shared = node.load_child(0, &g);
        assert!(!child_shared.is_null());
    }

    #[test]
    fn cas_data_to_tombstone() {
        let node = Node::<u64, u64>::with_capacity(LinearModel::constant(), 4);
        node.store_data(0, 10, 100);
        assert!(node.cas_data_to_tombstone(0));
        assert_eq!(node.slot_state(0), SLOT_TOMBSTONE);
    }

    #[test]
    fn drop_with_inline_data() {
        // Verify that nodes with String values correctly drop inline data.
        let node = Node::<u64, String>::with_capacity(LinearModel::constant(), 4);
        node.store_data(0, 1, "hello".to_string());
        node.store_data(1, 2, "world".to_string());
        drop(node);
        // If Drop is implemented correctly, no leak or double-free.
    }
}
