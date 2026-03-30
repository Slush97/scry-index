//! Size-tiered compaction with streaming k-way merge.
//!
//! When level L has more than `max_tables_per_level` SSTables, they are
//! merged into a single SSTable at level L+1 using a streaming merge that
//! reads one block at a time from each input SSTable.
//!
//! Memory usage during compaction: O(num_sstables × block_size), not
//! O(total entries). A 4-way compaction of 100 MB SSTables uses ~16 KiB
//! of block buffers, not 400 MB.

use std::collections::BinaryHeap;
use std::fs;

use crate::manifest::Manifest;
use crate::sstable::reader::SsTableReader;
use crate::sstable::writer::SsTableWriter;
use crate::sstable::SsTableMeta;

/// Maximum SSTables at any level before compaction triggers.
const MAX_TABLES_PER_LEVEL: usize = 4;

/// Check all levels and compact any that exceed the threshold.
pub fn maybe_compact(manifest: &mut Manifest) -> std::io::Result<()> {
    let num_levels = manifest.num_levels();
    for level in 0..num_levels {
        if manifest.level_count(level) >= MAX_TABLES_PER_LEVEL {
            compact_level(manifest, level)?;
        }
    }
    Ok(())
}

/// An entry in the merge heap: (key, value, source_index).
/// Wrapped in `Reverse` so the smallest key is popped first.
struct MergeEntry {
    key: Vec<u8>,
    value: Vec<u8>,
    source: usize,
}

impl PartialEq for MergeEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.source == other.source
    }
}

impl Eq for MergeEntry {}

impl PartialOrd for MergeEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering: smallest key first, then newest source first
        // (higher source index = newer SSTable).
        other
            .key
            .cmp(&self.key)
            .then(other.source.cmp(&self.source))
    }
}

/// Merge all SSTables at `level` into one SSTable at `level + 1`.
///
/// Uses a streaming k-way merge: only one block per input SSTable is held
/// in memory at a time. Duplicate keys are resolved by keeping the newest
/// (highest source index) value.
fn compact_level(manifest: &mut Manifest, level: usize) -> std::io::Result<()> {
    let tables = manifest.level(level);
    if tables.is_empty() {
        return Ok(());
    }

    let old_ids: Vec<u64> = tables.iter().map(|m| m.id).collect();
    let old_paths: Vec<_> = tables.iter().map(|m| m.path.clone()).collect();

    // Open streaming iterators for each SSTable.
    let mut readers: Vec<SsTableReader> = Vec::with_capacity(tables.len());
    for meta in tables {
        readers.push(SsTableReader::open(&meta.path)?);
    }

    // Seed the min-heap with the first entry from each SSTable.
    let mut heap: BinaryHeap<MergeEntry> = BinaryHeap::new();
    let mut scans: Vec<_> = readers.iter_mut().map(SsTableReader::scan).collect();

    for (i, scan) in scans.iter_mut().enumerate() {
        if let Some((key, value)) = scan.next_entry()? {
            heap.push(MergeEntry {
                key,
                value,
                source: i,
            });
        }
    }

    // Drop tombstones only at the highest level.
    let is_highest = level + 1 >= manifest.num_levels() - 1;

    let target_level = level + 1;
    let (id, path) = manifest.next_sstable_path(target_level);
    let mut writer = SsTableWriter::new(&path)?;
    let mut last_key: Option<Vec<u8>> = None;
    let mut wrote_any = false;

    // Stream entries out in sorted order.
    while let Some(entry) = heap.pop() {
        // Advance the source scan.
        if let Some((next_key, next_value)) = scans[entry.source].next_entry()? {
            heap.push(MergeEntry {
                key: next_key,
                value: next_value,
                source: entry.source,
            });
        }

        // Deduplicate: skip if same key as previous (we already wrote
        // the newer version since higher source index sorts first).
        if last_key.as_ref() == Some(&entry.key) {
            continue;
        }
        last_key = Some(entry.key.clone());

        // Drop tombstones at the highest level.
        if is_highest && entry.value.is_empty() {
            continue;
        }

        writer.add(&entry.key, &entry.value)?;
        wrote_any = true;
    }

    // Remove old SSTables from manifest and disk.
    manifest.remove(level, &old_ids);
    for p in &old_paths {
        let _ = fs::remove_file(p);
    }

    // Only register the new SSTable if it has entries.
    if wrote_any {
        let (entry_count, min_key, max_key) = writer.finish()?;
        manifest.add(SsTableMeta {
            id,
            level: target_level,
            path,
            entry_count,
            min_key,
            max_key,
        });
    }

    Ok(())
}
