//! A minimal LSM-tree storage engine showcasing [`scry-index`] as the memtable.
//!
//! Every production LSM engine (RocksDB, LevelDB, Pebble) uses a concurrent
//! sorted map as its in-memory write buffer. They all use skip-lists.
//! `scry-lsm` demonstrates that a learned index — specifically
//! [`scry_index::LearnedMap`] — is a drop-in replacement with O(1) expected
//! lookups instead of O(log n), and fully lock-free concurrent reads.
//!
//! # Architecture
//!
//! ```text
//! writes ──→ WAL (append-only log) ──→ Memtable (LearnedMap)
//!                                            │
//!                                       flush when full
//!                                            ▼
//!                                   SSTable (sorted file on disk)
//!                                            │
//!                                       compaction merges
//!                                            ▼
//!                                   Merged SSTables (higher levels)
//!
//! reads ──→ memtable → L0 SSTables → L1 → ... (first match wins)
//! ```

pub mod compaction;
pub mod engine;
pub mod error;
pub mod manifest;
pub mod memtable;
pub mod sstable;
pub mod wal;
