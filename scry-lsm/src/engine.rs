//! LSM engine — coordinates memtable, WAL, SSTables, and compaction.
//!
//! The engine is the public interface for the key-value store. Writes go to
//! the WAL and memtable. When the memtable exceeds a size threshold, it is
//! drained to an SSTable on disk and the WAL is truncated. Reads check the
//! memtable first, then SSTables from newest to oldest.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::compaction;
use crate::error::{self, Error, Result};
use crate::manifest::Manifest;
use crate::memtable::Memtable;
use crate::sstable::reader::SsTableReader;
use crate::sstable::writer::SsTableWriter;
use crate::sstable::SsTableMeta;
use crate::wal::{Wal, WalRecord};

/// Default flush threshold: 4 MiB of key-value data.
const DEFAULT_FLUSH_THRESHOLD: usize = 4 * 1024 * 1024;

/// Number of LSM levels.
const NUM_LEVELS: usize = 4;

/// Configuration for the LSM engine.
#[derive(Clone, Copy)]
pub struct EngineConfig {
    /// Memtable flush threshold in bytes.
    pub flush_threshold: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            flush_threshold: DEFAULT_FLUSH_THRESHOLD,
        }
    }
}

/// A minimal LSM-tree key-value engine.
///
/// Writes go through the WAL and memtable. When the memtable reaches
/// the flush threshold, it is drained to a sorted SSTable on disk.
/// Reads check the memtable first, then SSTables newest-to-oldest.
pub struct LsmEngine {
    #[allow(dead_code)] // Retained for future use (e.g., listing SSTables).
    dir: PathBuf,
    memtable: Box<dyn Memtable>,
    wal: Mutex<Wal>,
    manifest: Mutex<Manifest>,
    flush_threshold: usize,
}

impl LsmEngine {
    /// Open or create an LSM engine at the given directory.
    ///
    /// If a WAL exists from a previous run, its records are replayed into the
    /// memtable for crash recovery.
    pub fn open(dir: &Path, memtable: Box<dyn Memtable>, config: EngineConfig) -> Result<Self> {
        std::fs::create_dir_all(dir)?;

        let wal_path = dir.join("wal.log");

        // Recover from WAL if it exists.
        let records = Wal::recover(&wal_path)?;
        for record in records {
            match record {
                WalRecord::Put { key, value } => memtable.insert(key, value),
                WalRecord::Delete { key } => memtable.delete(&key),
            }
        }

        let wal = Wal::open(&wal_path)?;

        Ok(Self {
            dir: dir.to_path_buf(),
            memtable,
            wal: Mutex::new(wal),
            manifest: Mutex::new(Manifest::new(dir, NUM_LEVELS)),
            flush_threshold: config.flush_threshold,
        })
    }

    /// Insert a key-value pair.
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        // WAL first for durability.
        error::lock(&self.wal)?.append_put(key, value)?;
        self.memtable.insert(key.to_vec(), value.to_vec());
        self.maybe_flush()?;
        Ok(())
    }

    /// Look up a key. Returns `None` if not found or deleted.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        // 1. Check memtable (freshest data).
        if let Some(value) = self.memtable.get(key) {
            return if value.is_empty() {
                Ok(None) // Tombstone — key was deleted.
            } else {
                Ok(Some(value))
            };
        }

        // 2. Check SSTables from newest (L0) to oldest (Ln).
        let manifest = error::lock(&self.manifest)?;
        for (_, tables) in manifest.levels() {
            // Within a level, search newest (last added) first.
            for meta in tables.iter().rev() {
                if key < meta.min_key.as_slice() || key > meta.max_key.as_slice() {
                    continue; // Key not in this SSTable's range.
                }
                let mut reader = SsTableReader::open(&meta.path)?;
                if let Some(value) = reader.get(key)? {
                    return if value.is_empty() {
                        Ok(None) // Tombstone.
                    } else {
                        Ok(Some(value))
                    };
                }
            }
        }

        Ok(None)
    }

    /// Delete a key by inserting a tombstone.
    pub fn delete(&self, key: &[u8]) -> Result<()> {
        error::lock(&self.wal)?.append_delete(key)?;
        self.memtable.delete(key);
        Ok(())
    }

    /// Force the memtable to flush to an SSTable, regardless of size.
    pub fn flush(&self) -> Result<()> {
        self.flush_memtable()
    }

    // -----------------------------------------------------------------
    // Streaming ingestion
    // -----------------------------------------------------------------

    /// Ingest key-value pairs from any iterator.
    ///
    /// Pairs are inserted through the normal write path (WAL + memtable),
    /// with automatic flushes when the memtable fills. This handles
    /// datasets of any size — only one memtable's worth of data is held
    /// in RAM at a time.
    ///
    /// Returns the number of entries ingested.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use scry_lsm::engine::{LsmEngine, EngineConfig};
    /// # use scry_lsm::memtable::LearnedMemtable;
    /// # let dir = tempfile::tempdir().unwrap();
    /// # let engine = LsmEngine::open(dir.path(), Box::new(LearnedMemtable::new()), EngineConfig::default()).unwrap();
    /// let data = vec![
    ///     (b"key1".to_vec(), b"val1".to_vec()),
    ///     (b"key2".to_vec(), b"val2".to_vec()),
    /// ];
    /// let count = engine.ingest_iter(data.into_iter()).unwrap();
    /// assert_eq!(count, 2);
    /// ```
    pub fn ingest_iter<I>(&self, iter: I) -> Result<u64>
    where
        I: Iterator<Item = (Vec<u8>, Vec<u8>)>,
    {
        let mut count = 0u64;
        for (key, value) in iter {
            self.put(&key, &value)?;
            count += 1;
        }
        Ok(count)
    }

    /// Ingest newline-delimited records from any [`Read`] source
    /// (file, stdin, network socket, etc.).
    ///
    /// Each line is parsed as `key\tvalue` (tab-separated). Lines without
    /// a tab are skipped with a warning. The reader is consumed one line
    /// at a time — memory usage is bounded by the memtable flush threshold,
    /// not the input size.
    ///
    /// Returns the number of entries ingested.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use scry_lsm::engine::{LsmEngine, EngineConfig};
    /// # use scry_lsm::memtable::LearnedMemtable;
    /// # let dir = tempfile::tempdir().unwrap();
    /// # let engine = LsmEngine::open(dir.path(), Box::new(LearnedMemtable::new()), EngineConfig::default()).unwrap();
    /// let tsv = b"key1\tvalue1\nkey2\tvalue2\n";
    /// let count = engine.ingest_reader(&tsv[..]).unwrap();
    /// assert_eq!(count, 2);
    /// ```
    pub fn ingest_reader<R: Read>(&self, reader: R) -> Result<u64> {
        let buf_reader = BufReader::new(reader);
        let mut count = 0u64;

        for line_result in buf_reader.lines() {
            let line = line_result?;
            if line.is_empty() {
                continue;
            }
            let Some((key, value)) = line.split_once('\t') else {
                continue; // Skip malformed lines.
            };
            self.put(key.as_bytes(), value.as_bytes())?;
            count += 1;
        }

        Ok(count)
    }

    /// Ingest a pre-sorted file directly as an L0 SSTable, bypassing the
    /// memtable entirely.
    ///
    /// This is the fastest path for bulk loading sorted data (e.g., from
    /// an external sort, another database export, or a previous SSTable).
    /// No WAL is written — the SSTable *is* the durable copy.
    ///
    /// The input iterator **must** yield entries in ascending key order.
    /// Returns the number of entries written.
    pub fn ingest_sorted<I>(&self, iter: I) -> Result<u64>
    where
        I: Iterator<Item = (Vec<u8>, Vec<u8>)>,
    {
        let mut manifest = error::lock(&self.manifest)?;
        let (id, path) = manifest.next_sstable_path(0);

        let mut writer = SsTableWriter::new(&path)?;
        let mut prev_key: Option<Vec<u8>> = None;

        for (key, value) in iter {
            // Verify sort order.
            if let Some(ref prev) = prev_key {
                if key.as_slice() <= prev.as_slice() {
                    // Clean up partial file.
                    drop(writer);
                    let _ = std::fs::remove_file(&path);
                    return Err(Error::Corruption(
                        "ingest_sorted: keys not in ascending order".into(),
                    ));
                }
            }
            writer.add(&key, &value)?;
            prev_key = Some(key);
        }

        let (entry_count, min_key, max_key) = writer.finish()?;

        if entry_count > 0 {
            manifest.add(SsTableMeta {
                id,
                level: 0,
                path,
                entry_count,
                min_key,
                max_key,
            });
            compaction::maybe_compact(&mut manifest)?;
        } else {
            // Empty input — remove the file.
            let _ = std::fs::remove_file(&path);
        }

        Ok(entry_count)
    }

    /// Check if the memtable exceeds the threshold and flush if needed.
    fn maybe_flush(&self) -> Result<()> {
        if self.memtable.approximate_bytes() >= self.flush_threshold {
            self.flush_memtable()?;
        }
        Ok(())
    }

    /// Drain the memtable to a new SSTable and truncate the WAL.
    fn flush_memtable(&self) -> Result<()> {
        let entries = self.memtable.drain_sorted();
        if entries.is_empty() {
            return Ok(());
        }

        let mut manifest = error::lock(&self.manifest)?;

        // Write a new L0 SSTable.
        let (id, path) = manifest.next_sstable_path(0);
        let mut writer = SsTableWriter::new(&path)?;
        for (k, v) in &entries {
            writer.add(k, v)?;
        }
        let (entry_count, min_key, max_key) = writer.finish()?;

        manifest.add(SsTableMeta {
            id,
            level: 0,
            path,
            entry_count,
            min_key,
            max_key,
        });

        // Truncate WAL after successful flush.
        error::lock(&self.wal)?.truncate()?;

        // Check for compaction.
        compaction::maybe_compact(&mut manifest)?;

        Ok(())
    }
}
