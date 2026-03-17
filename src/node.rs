//! Node types for the learned index tree.
//!
//! Each node contains a linear model and a fixed-size array of slots.
//! Slots are either empty, contain a key-value pair, or point to a child node.

use crate::key::Key;
use crate::model::LinearModel;

/// A slot in a node's array.
#[derive(Debug)]
pub enum Slot<K, V> {
    /// The slot is empty (available for insertion).
    Empty,
    /// The slot contains a key-value pair.
    Data(K, V),
    /// The slot points to a child node (created by conflict resolution).
    Child(Box<Node<K, V>>),
}

impl<K, V> Slot<K, V> {
    /// Returns `true` if this slot is empty.
    pub const fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Returns `true` if this slot contains data.
    pub const fn is_data(&self) -> bool {
        matches!(self, Self::Data(..))
    }

    /// Returns `true` if this slot points to a child.
    pub const fn is_child(&self) -> bool {
        matches!(self, Self::Child(_))
    }
}

/// A node in the learned index tree.
///
/// Contains a linear model for position prediction and a fixed-size array
/// of slots. The model maps keys to slot indices; conflicts are resolved
/// by creating child nodes.
#[derive(Debug)]
pub struct Node<K, V> {
    /// The linear model for this node.
    pub model: LinearModel,
    /// The slot array. Length is determined at build time.
    pub slots: Vec<Slot<K, V>>,
    /// Number of data entries in this node (not counting children).
    pub num_keys: usize,
}

impl<K: Key, V> Node<K, V> {
    /// Create a new node with the given model and array size.
    ///
    /// All slots are initialized to [`Slot::Empty`].
    pub fn with_capacity(model: LinearModel, array_size: usize) -> Self {
        let mut slots = Vec::with_capacity(array_size);
        slots.resize_with(array_size, || Slot::Empty);
        Self {
            model,
            slots,
            num_keys: 0,
        }
    }

    /// Predict the slot index for a key.
    #[inline]
    pub fn predict_slot(&self, key: K) -> usize {
        self.model.predict(key, self.slots.len())
    }

    /// Return the number of slots in this node.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Count total keys stored in this node and all descendants.
    pub fn total_keys(&self) -> usize {
        let mut count = 0;
        for slot in &self.slots {
            match slot {
                Slot::Empty => {}
                Slot::Data(..) => count += 1,
                Slot::Child(child) => count += child.total_keys(),
            }
        }
        count
    }

    /// Return the depth of the deepest path from this node.
    pub fn max_depth(&self) -> usize {
        let mut max_child_depth = 0;
        for slot in &self.slots {
            if let Slot::Child(child) = slot {
                max_child_depth = max_child_depth.max(child.max_depth());
            }
        }
        1 + max_child_depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_node_all_empty() {
        let node = Node::<u64, String>::with_capacity(LinearModel::constant(), 10);
        assert_eq!(node.capacity(), 10);
        assert_eq!(node.num_keys, 0);
        assert!(node.slots.iter().all(Slot::is_empty));
    }

    #[test]
    fn total_keys_empty() {
        let node = Node::<u64, ()>::with_capacity(LinearModel::constant(), 5);
        assert_eq!(node.total_keys(), 0);
    }

    #[test]
    fn total_keys_with_data() {
        let mut node = Node::<u64, &str>::with_capacity(LinearModel::constant(), 5);
        node.slots[0] = Slot::Data(1, "a");
        node.slots[2] = Slot::Data(2, "b");
        assert_eq!(node.total_keys(), 2);
    }

    #[test]
    fn total_keys_with_children() {
        let mut child = Node::<u64, &str>::with_capacity(LinearModel::constant(), 3);
        child.slots[0] = Slot::Data(10, "x");
        child.slots[1] = Slot::Data(20, "y");

        let mut parent = Node::<u64, &str>::with_capacity(LinearModel::constant(), 5);
        parent.slots[0] = Slot::Data(1, "a");
        parent.slots[1] = Slot::Child(Box::new(child));

        assert_eq!(parent.total_keys(), 3);
    }

    #[test]
    fn max_depth_leaf() {
        let node = Node::<u64, ()>::with_capacity(LinearModel::constant(), 5);
        assert_eq!(node.max_depth(), 1);
    }

    #[test]
    fn max_depth_nested() {
        let leaf = Node::<u64, ()>::with_capacity(LinearModel::constant(), 2);
        let mut mid = Node::<u64, ()>::with_capacity(LinearModel::constant(), 2);
        mid.slots[0] = Slot::Child(Box::new(leaf));
        let mut root = Node::<u64, ()>::with_capacity(LinearModel::constant(), 2);
        root.slots[0] = Slot::Child(Box::new(mid));
        assert_eq!(root.max_depth(), 3);
    }

    #[test]
    fn slot_type_checks() {
        let empty: Slot<u64, ()> = Slot::Empty;
        let data: Slot<u64, ()> = Slot::Data(1, ());
        let child: Slot<u64, ()> = Slot::Child(Box::new(
            Node::with_capacity(LinearModel::constant(), 1),
        ));

        assert!(empty.is_empty());
        assert!(!empty.is_data());
        assert!(!empty.is_child());

        assert!(!data.is_empty());
        assert!(data.is_data());
        assert!(!data.is_child());

        assert!(!child.is_empty());
        assert!(!child.is_data());
        assert!(child.is_child());
    }
}
