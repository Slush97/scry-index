//! Manifest — tracks SSTable metadata across LSM levels.
//!
//! The manifest is an in-memory registry of all live SSTable files, organized
//! by level. Level 0 contains freshly-flushed SSTables (may have overlapping
//! key ranges). Higher levels contain merged, non-overlapping SSTables.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::sstable::SsTableMeta;

/// In-memory registry of SSTable files organized by level.
pub struct Manifest {
    levels: Vec<Vec<SsTableMeta>>,
    next_id: AtomicU64,
    dir: PathBuf,
}

impl Manifest {
    /// Create a new empty manifest rooted at the given directory.
    pub fn new(dir: &Path, num_levels: usize) -> Self {
        Self {
            levels: (0..num_levels).map(|_| Vec::new()).collect(),
            next_id: AtomicU64::new(0),
            dir: dir.to_path_buf(),
        }
    }

    /// Allocate a unique SSTable ID and return the file path for it.
    pub fn next_sstable_path(&self, level: usize) -> (u64, PathBuf) {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let path = self.dir.join(format!("L{level}_{id:06}.sst"));
        (id, path)
    }

    /// Register a new SSTable at the given level.
    pub fn add(&mut self, meta: SsTableMeta) {
        let level = meta.level;
        if level >= self.levels.len() {
            self.levels.resize_with(level + 1, Vec::new);
        }
        self.levels[level].push(meta);
    }

    /// Remove SSTables by ID from a given level.
    pub fn remove(&mut self, level: usize, ids: &[u64]) {
        if level < self.levels.len() {
            self.levels[level].retain(|m| !ids.contains(&m.id));
        }
    }

    /// Get all SSTables at a given level, ordered newest-first.
    pub fn level(&self, level: usize) -> &[SsTableMeta] {
        self.levels.get(level).map_or(&[], Vec::as_slice)
    }

    /// Number of SSTables at a given level.
    pub fn level_count(&self, level: usize) -> usize {
        self.levels.get(level).map_or(0, Vec::len)
    }

    /// Total number of levels that have at least one SSTable.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Iterate all levels with their SSTables.
    pub fn levels(&self) -> impl Iterator<Item = (usize, &[SsTableMeta])> {
        self.levels
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.as_slice()))
    }
}
