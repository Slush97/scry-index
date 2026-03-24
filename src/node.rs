//! Node types for the learned index tree.
//!
//! Each node contains a linear model and a fixed-size array of atomic slots.
//! Slots are either empty (null) or point to a [`SlotInner`] which is either
//! a key-value pair or a child node.
#![allow(unsafe_code)]

use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_epoch::{self as epoch, Atomic, Guard, Owned};

use crate::key::Key;
use crate::model::LinearModel;

/// The content of a non-empty slot: either data or a child node.
pub enum SlotInner<K, V> {
    /// A key-value pair stored in this slot.
    Data { key: K, value: V },
    /// A child node created by conflict resolution.
    Child(Node<K, V>),
}

/// A node in the learned index tree.
///
/// Contains a linear model for position prediction and a fixed-size array
/// of atomic slots. The model maps keys to slot indices; conflicts are resolved
/// by creating child nodes.
pub struct Node<K, V> {
    /// The linear model for this node (immutable after construction).
    model: LinearModel,
    /// Slot array. Each atomic is null (empty) or points to a `SlotInner`.
    slots: Box<[Atomic<SlotInner<K, V>>]>,
    /// Approximate number of data entries in this node (not counting children).
    num_keys: AtomicUsize,
}

impl<K, V> std::fmt::Debug for Node<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("model", &self.model)
            .field("capacity", &self.slots.len())
            .field("num_keys", &self.num_keys.load(Ordering::Relaxed))
            .finish()
    }
}

impl<K: Key, V> Node<K, V> {
    /// Create a new node with the given model and array size.
    ///
    /// All slots are initialized to null (empty).
    pub fn with_capacity(model: LinearModel, array_size: usize) -> Self {
        let slots: Vec<Atomic<SlotInner<K, V>>> = (0..array_size).map(|_| Atomic::null()).collect();
        Self {
            model,
            slots: slots.into_boxed_slice(),
            num_keys: AtomicUsize::new(0),
        }
    }

    /// Predict the slot index for a key.
    #[inline]
    pub fn predict_slot(&self, key: &K) -> usize {
        self.model.predict(key, self.slots.len())
    }

    /// Return the number of slots in this node.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Get a reference to the atomic at the given slot index.
    #[inline]
    pub fn slot(&self, idx: usize) -> &Atomic<SlotInner<K, V>> {
        &self.slots[idx]
    }

    /// Store a value into a slot during construction (no concurrent access).
    ///
    /// This uses relaxed ordering since no other thread can observe the node yet.
    /// The slot must be empty (null); storing into an occupied slot leaks the old
    /// value. A debug assertion guards against this.
    pub fn store_slot(&self, idx: usize, inner: SlotInner<K, V>) {
        // SAFETY: Called only during construction when no concurrent access exists.
        // Using unprotected() is safe because we only convert Owned to Shared for
        // storage; no concurrent readers can observe this data yet.
        unsafe {
            let guard = epoch::unprotected();
            debug_assert!(
                self.slots[idx].load(Ordering::Relaxed, guard).is_null(),
                "store_slot called on occupied slot {idx} — would leak memory"
            );
            self.slots[idx].store(Owned::new(inner).into_shared(guard), Ordering::Relaxed);
        }
    }

    /// Increment the approximate key count.
    pub fn inc_keys(&self) {
        self.num_keys.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the approximate key count.
    pub fn dec_keys(&self) {
        self.num_keys.fetch_sub(1, Ordering::Relaxed);
    }

    /// Count total keys stored in this node and all descendants.
    pub fn total_keys(&self, guard: &Guard) -> usize {
        let mut count = 0;
        for slot in &*self.slots {
            let shared = slot.load(Ordering::Acquire, guard);
            if shared.is_null() {
                continue;
            }
            // SAFETY: shared is not null and is valid for the lifetime of the guard.
            match unsafe { shared.deref() } {
                SlotInner::Data { .. } => count += 1,
                SlotInner::Child(child) => count += child.total_keys(guard),
            }
        }
        count
    }

    /// Estimate the total heap memory used by this node and all descendants.
    ///
    /// The estimate includes:
    /// - The `Node` struct itself (`size_of::<Node<K, V>>()`)
    /// - The boxed slot array (`capacity * size_of::<Atomic<SlotInner<K, V>>>()`)
    /// - Each non-null slot's heap-allocated `SlotInner` (`size_of::<SlotInner<K, V>>()`)
    /// - Recursive child nodes
    ///
    /// This is an approximation — it does not account for allocator overhead,
    /// alignment padding, or epoch-deferred garbage.
    pub fn allocated_bytes(&self, guard: &Guard) -> usize {
        let node_size = std::mem::size_of::<Self>();
        let slots_size =
            self.slots.len() * std::mem::size_of::<Atomic<SlotInner<K, V>>>();
        let inner_size = std::mem::size_of::<SlotInner<K, V>>();

        let mut total = node_size + slots_size;

        for slot in &*self.slots {
            let shared = slot.load(Ordering::Acquire, guard);
            if shared.is_null() {
                continue;
            }
            total += inner_size;
            // SAFETY: shared is not null and is valid for the lifetime of the guard.
            if let SlotInner::Child(child) = unsafe { shared.deref() } {
                total += child.allocated_bytes(guard);
            }
        }

        total
    }

    /// Return the depth of the deepest path from this node.
    pub fn max_depth(&self, guard: &Guard) -> usize {
        let mut max_child_depth = 0;
        for slot in &*self.slots {
            let shared = slot.load(Ordering::Acquire, guard);
            if shared.is_null() {
                continue;
            }
            // SAFETY: shared is not null and is valid for the lifetime of the guard.
            if let SlotInner::Child(child) = unsafe { shared.deref() } {
                max_child_depth = max_child_depth.max(child.max_depth(guard));
            }
        }
        1 + max_child_depth
    }
}

impl<K, V> Drop for Node<K, V> {
    fn drop(&mut self) {
        // SAFETY: We have exclusive access during drop — no other thread can
        // reference this node. Using unprotected() and Relaxed ordering is safe.
        unsafe {
            let guard = epoch::unprotected();
            for slot in &*self.slots {
                let shared = slot.load(Ordering::Relaxed, guard);
                if !shared.is_null() {
                    drop(shared.into_owned());
                }
            }
        }
    }
}

// SAFETY: Node is Send+Sync when K and V are Send+Sync. All interior mutation
// goes through atomic operations (Atomic<SlotInner> and AtomicUsize), which are
// inherently thread-safe. The recursive type prevents auto-derivation, so we
// implement these traits manually.
unsafe impl<K: Send + Sync, V: Send + Sync> Send for Node<K, V> {}
unsafe impl<K: Send + Sync, V: Send + Sync> Sync for Node<K, V> {}

#[cfg(test)]
mod tests {
    use super::*;

    fn guard() -> Guard {
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
        node.store_slot(0, SlotInner::Data { key: 1, value: "a" });
        node.inc_keys();
        node.store_slot(2, SlotInner::Data { key: 2, value: "b" });
        node.inc_keys();
        assert_eq!(node.total_keys(&g), 2);
    }

    #[test]
    fn total_keys_with_children() {
        let g = guard();

        let child = Node::<u64, &str>::with_capacity(LinearModel::constant(), 3);
        child.store_slot(
            0,
            SlotInner::Data {
                key: 10,
                value: "x",
            },
        );
        child.inc_keys();
        child.store_slot(
            1,
            SlotInner::Data {
                key: 20,
                value: "y",
            },
        );
        child.inc_keys();

        let parent = Node::<u64, &str>::with_capacity(LinearModel::constant(), 5);
        parent.store_slot(0, SlotInner::Data { key: 1, value: "a" });
        parent.inc_keys();
        parent.store_slot(1, SlotInner::Child(child));

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
        mid.store_slot(0, SlotInner::Child(leaf));
        let root = Node::<u64, ()>::with_capacity(LinearModel::constant(), 2);
        root.store_slot(0, SlotInner::Child(mid));
        assert_eq!(root.max_depth(&g), 3);
    }

    #[test]
    fn store_and_load_slot() {
        let g = guard();
        let node = Node::<u64, i32>::with_capacity(LinearModel::constant(), 4);
        node.store_slot(
            1,
            SlotInner::Data {
                key: 42,
                value: 100,
            },
        );
        node.inc_keys();

        let shared = node.slot(1).load(Ordering::Acquire, &g);
        assert!(!shared.is_null());

        // SAFETY: shared is not null and valid under guard
        match unsafe { shared.deref() } {
            SlotInner::Data { key, value } => {
                assert_eq!(*key, 42);
                assert_eq!(*value, 100);
            }
            SlotInner::Child(_) => panic!("expected Data"),
        }
    }
}
