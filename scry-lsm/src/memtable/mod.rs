//! Pluggable memtable backends for the LSM engine.
//!
//! The memtable is the in-memory write buffer — the first destination for all
//! writes and the first place checked on reads. It must support concurrent
//! access from multiple threads without external synchronization.
//!
//! Three implementations are provided for benchmarking:
//!
//! - [`LearnedMemtable`] — backed by [`scry_index::LearnedMap`], O(1) expected
//!   lookups via learned index models, fully lock-free reads
//! - [`SkipListMemtable`] — backed by [`crossbeam_skiplist::SkipMap`], the
//!   industry standard used in RocksDB and LevelDB
//! - [`BTreeMemtable`] — backed by `RwLock<BTreeMap>`, a simple baseline

mod btree;
mod learned;
mod skiplist;

pub use btree::BTreeMemtable;
pub use learned::LearnedMemtable;
pub use skiplist::SkipListMemtable;

/// A concurrent in-memory sorted write buffer for the LSM engine.
///
/// All operations take `&self` — implementations handle their own
/// synchronization internally. Tombstones (deletions) are represented
/// as entries with an empty value (`Vec<u8>` with length 0).
pub trait Memtable: Send + Sync {
    /// Insert a key-value pair, overwriting any existing value for the key.
    fn insert(&self, key: Vec<u8>, value: Vec<u8>);

    /// Look up a key. Returns the value if found, `None` if absent.
    ///
    /// An empty `Vec<u8>` return indicates a tombstone (the key was deleted).
    fn get(&self, key: &[u8]) -> Option<Vec<u8>>;

    /// Mark a key as deleted by inserting a tombstone (empty value).
    fn delete(&self, key: &[u8]);

    /// Approximate number of entries (including tombstones).
    fn len(&self) -> usize;

    /// Whether the memtable contains no entries.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Approximate memory usage in bytes (key + value data, not structural overhead).
    ///
    /// Used by the engine to decide when to flush the memtable to disk.
    fn approximate_bytes(&self) -> usize;

    /// Extract all entries in sorted key order, emptying the memtable.
    ///
    /// Tombstones are included (with empty values) so they propagate to
    /// SSTables for correct deletion semantics across LSM levels.
    ///
    /// This is called during flush. Callers should ensure no concurrent
    /// writers are active (the engine swaps in a fresh memtable first).
    fn drain_sorted(&self) -> Vec<(Vec<u8>, Vec<u8>)>;
}
