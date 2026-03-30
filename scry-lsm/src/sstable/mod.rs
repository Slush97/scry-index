//! Sorted String Table (SSTable) format — immutable sorted files on disk.
//!
//! When the memtable is full, it is flushed to an SSTable. Each SSTable
//! contains entries sorted by key and is immutable once written. Point
//! lookups use a block index for binary search, and sequential scans
//! read entries in order for compaction merges.
//!
//! # File Format
//!
//! ```text
//! [Data Block 0][Data Block 1] ... [Data Block N]
//! [Index Block]
//! [Footer]
//! ```
//!
//! **Data Block**: sequence of length-prefixed key-value entries.
//! Each entry: `[key_len: u32 LE][key][value_len: u32 LE][value]`
//!
//! **Index Block**: one entry per data block, storing the first key and
//! the block's file offset.
//! Each entry: `[key_len: u32 LE][first_key][offset: u64 LE][block_len: u64 LE]`
//!
//! **Footer** (fixed 20 bytes):
//! `[index_offset: u64 LE][entry_count: u64 LE][magic: u32 LE]`

pub mod reader;
pub mod writer;

/// Magic number at the end of every SSTable file.
const MAGIC: u32 = 0x5353_544C; // "SSTL"

/// Default target size for data blocks (4 KiB).
const DEFAULT_BLOCK_SIZE: usize = 4096;

/// Footer size in bytes: index_offset(8) + entry_count(8) + magic(4).
const FOOTER_SIZE: usize = 20;

/// Metadata about an SSTable file, used by the manifest.
#[derive(Debug, Clone)]
pub struct SsTableMeta {
    /// Unique identifier for this SSTable.
    pub id: u64,
    /// LSM level (0 = freshly flushed from memtable).
    pub level: usize,
    /// Path to the SSTable file on disk.
    pub path: std::path::PathBuf,
    /// Total number of entries (including tombstones).
    pub entry_count: u64,
    /// Smallest key in the SSTable.
    pub min_key: Vec<u8>,
    /// Largest key in the SSTable.
    pub max_key: Vec<u8>,
}
