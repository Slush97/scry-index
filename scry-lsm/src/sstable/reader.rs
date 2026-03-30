//! SSTable reader — point lookups and sequential scans over SSTable files.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use super::{FOOTER_SIZE, MAGIC};

/// Convert a slice to a fixed-size array, returning `io::Error` on mismatch.
fn to_array<const N: usize>(slice: &[u8]) -> std::io::Result<[u8; N]> {
    slice.try_into().map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "unexpected byte slice length",
        )
    })
}

/// An index entry pointing to a data block.
#[derive(Debug, Clone)]
struct IndexEntry {
    first_key: Vec<u8>,
    offset: u64,
    block_len: u64,
}

/// Reader for a single SSTable file.
pub struct SsTableReader {
    file: File,
    index: Vec<IndexEntry>,
    entry_count: u64,
}

impl SsTableReader {
    /// Open an SSTable file and read its index into memory.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();

        if file_len < FOOTER_SIZE as u64 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "file too small to be an SSTable",
            ));
        }

        // Read footer.
        #[allow(clippy::cast_possible_wrap)] // FOOTER_SIZE is 20, never wraps.
        file.seek(SeekFrom::End(-(FOOTER_SIZE as i64)))?;
        let mut footer_buf = [0u8; FOOTER_SIZE];
        file.read_exact(&mut footer_buf)?;

        let index_offset = u64::from_le_bytes(to_array(&footer_buf[0..8])?);
        let entry_count = u64::from_le_bytes(to_array(&footer_buf[8..16])?);
        let magic = u32::from_le_bytes(to_array(&footer_buf[16..20])?);

        if magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bad SSTable magic: {magic:#x}, expected {MAGIC:#x}"),
            ));
        }

        // Read the index block.
        let index_len = file_len - FOOTER_SIZE as u64 - index_offset;
        file.seek(SeekFrom::Start(index_offset))?;
        let mut index_buf = vec![0u8; index_len as usize];
        file.read_exact(&mut index_buf)?;

        let index = parse_index(&index_buf)?;

        Ok(Self {
            file,
            index,
            entry_count,
        })
    }

    /// Number of entries in this SSTable.
    pub fn entry_count(&self) -> u64 {
        self.entry_count
    }

    /// Point lookup: find the value for a key, or `None` if absent.
    pub fn get(&mut self, key: &[u8]) -> std::io::Result<Option<Vec<u8>>> {
        let Some(block_idx) = find_block(&self.index, key) else {
            return Ok(None);
        };

        let entry = &self.index[block_idx];
        let block = self.read_block(entry.offset, entry.block_len)?;

        // Scan the block for the key.
        let mut cursor = 0;
        while cursor < block.len() {
            let (k, v, next) = decode_entry(&block, cursor)?;
            match k.cmp(key) {
                std::cmp::Ordering::Equal => return Ok(Some(v.to_vec())),
                std::cmp::Ordering::Greater => return Ok(None), // Past the key.
                std::cmp::Ordering::Less => cursor = next,
            }
        }

        Ok(None)
    }

    /// Read all entries in sorted order. Used for compaction merges.
    pub fn read_all(&mut self) -> std::io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut entries = Vec::with_capacity(self.entry_count as usize);

        for idx in 0..self.index.len() {
            let ie = self.index[idx].clone();
            let block = self.read_block(ie.offset, ie.block_len)?;
            let mut cursor = 0;
            while cursor < block.len() {
                let (k, v, next) = decode_entry(&block, cursor)?;
                entries.push((k.to_vec(), v.to_vec()));
                cursor = next;
            }
        }

        Ok(entries)
    }

    /// Create a streaming iterator that reads one block at a time.
    ///
    /// Memory usage is O(block\_size) regardless of SSTable size.
    pub fn scan(&mut self) -> SsTableScan<'_> {
        SsTableScan {
            reader: self,
            block_idx: 0,
            block_buf: Vec::new(),
            cursor: 0,
        }
    }

    /// Read a data block from disk.
    fn read_block(&mut self, offset: u64, len: u64) -> std::io::Result<Vec<u8>> {
        self.file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; len as usize];
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }
}

/// A streaming iterator over SSTable entries, reading one block at a time.
///
/// Only one 4 KiB block is held in memory at any point, regardless of the
/// total SSTable size. Entries are yielded in sorted key order.
pub struct SsTableScan<'a> {
    reader: &'a mut SsTableReader,
    block_idx: usize,
    block_buf: Vec<u8>,
    cursor: usize,
}

impl SsTableScan<'_> {
    /// Yield the next key-value pair, or `None` at end of file.
    ///
    /// This is not `Iterator::next` because it returns `io::Result`.
    pub fn next_entry(&mut self) -> std::io::Result<Option<(Vec<u8>, Vec<u8>)>> {
        loop {
            // Try to decode from current block buffer.
            if self.cursor < self.block_buf.len() {
                let (k, v, next) = decode_entry(&self.block_buf, self.cursor)?;
                self.cursor = next;
                return Ok(Some((k.to_vec(), v.to_vec())));
            }

            // Load the next block.
            if self.block_idx >= self.reader.index.len() {
                return Ok(None); // End of SSTable.
            }

            let ie = self.reader.index[self.block_idx].clone();
            self.block_buf = self.reader.read_block(ie.offset, ie.block_len)?;
            self.cursor = 0;
            self.block_idx += 1;
        }
    }
}

/// Binary search the index to find which data block might contain the key.
fn find_block(index: &[IndexEntry], key: &[u8]) -> Option<usize> {
    if index.is_empty() {
        return None;
    }

    // Find the last block whose first_key <= key.
    let pos = index.partition_point(|e| e.first_key.as_slice() <= key);
    if pos == 0 {
        // Key is before all blocks — could still be in the first block
        // if it's >= the first block's first key (handled by partition_point
        // returning 1 in that case). If partition_point returns 0, the key
        // is less than the first block's first key, so it's not in the SSTable.

        // Actually, let's check: if key < first block's first_key, not found.
        if key < index[0].first_key.as_slice() {
            return None;
        }
        return Some(0);
    }

    Some(pos - 1)
}

/// Parse the index block into index entries.
fn parse_index(buf: &[u8]) -> std::io::Result<Vec<IndexEntry>> {
    let mut entries = Vec::new();
    let mut cursor = 0;

    while cursor < buf.len() {
        if cursor + 4 > buf.len() {
            break;
        }
        let key_len = u32::from_le_bytes(to_array(&buf[cursor..cursor + 4])?) as usize;
        cursor += 4;

        if cursor + key_len + 16 > buf.len() {
            break;
        }
        let first_key = buf[cursor..cursor + key_len].to_vec();
        cursor += key_len;

        let offset = u64::from_le_bytes(to_array(&buf[cursor..cursor + 8])?);
        cursor += 8;

        let block_len = u64::from_le_bytes(to_array(&buf[cursor..cursor + 8])?);
        cursor += 8;

        entries.push(IndexEntry {
            first_key,
            offset,
            block_len,
        });
    }

    Ok(entries)
}

/// Decode one key-value entry from a data block. Returns (key, value, next_cursor).
fn decode_entry(block: &[u8], cursor: usize) -> std::io::Result<(&[u8], &[u8], usize)> {
    let err = || {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "truncated entry in data block",
        )
    };

    if cursor + 4 > block.len() {
        return Err(err());
    }
    let key_len = u32::from_le_bytes(to_array(&block[cursor..cursor + 4])?) as usize;
    let pos = cursor + 4;

    if pos + key_len + 4 > block.len() {
        return Err(err());
    }
    let key = &block[pos..pos + key_len];
    let pos = pos + key_len;

    let val_len = u32::from_le_bytes(to_array(&block[pos..pos + 4])?) as usize;
    let pos = pos + 4;

    if pos + val_len > block.len() {
        return Err(err());
    }
    let value = &block[pos..pos + val_len];
    let next = pos + val_len;

    Ok((key, value, next))
}

#[cfg(test)]
mod tests {
    use super::super::writer::SsTableWriter;
    use super::*;

    fn write_test_sstable(path: &Path, entries: &[(&[u8], &[u8])]) {
        let mut writer = SsTableWriter::new(path).unwrap();
        for (k, v) in entries {
            writer.add(k, v).unwrap();
        }
        writer.finish().unwrap();
    }

    #[test]
    fn roundtrip_point_lookup() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.sst");
        write_test_sstable(
            &path,
            &[
                (b"alpha", b"1"),
                (b"bravo", b"2"),
                (b"charlie", b"3"),
                (b"delta", b"4"),
            ],
        );

        let mut reader = SsTableReader::open(&path).unwrap();
        assert_eq!(reader.get(b"alpha").unwrap(), Some(b"1".to_vec()));
        assert_eq!(reader.get(b"charlie").unwrap(), Some(b"3".to_vec()));
        assert_eq!(reader.get(b"delta").unwrap(), Some(b"4".to_vec()));
        assert_eq!(reader.get(b"missing").unwrap(), None);
        assert_eq!(reader.get(b"aaa").unwrap(), None); // Before all keys.
        assert_eq!(reader.get(b"zzz").unwrap(), None); // After all keys.
    }

    #[test]
    fn roundtrip_iteration() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.sst");
        let input: Vec<(&[u8], &[u8])> =
            vec![(b"a", b"1"), (b"b", b"2"), (b"c", b"3"), (b"d", b"4")];
        write_test_sstable(&path, &input);

        let mut reader = SsTableReader::open(&path).unwrap();
        let entries = reader.read_all().unwrap();
        let expected: Vec<(Vec<u8>, Vec<u8>)> = input
            .iter()
            .map(|(k, v)| (k.to_vec(), v.to_vec()))
            .collect();
        assert_eq!(entries, expected);
    }

    #[test]
    fn empty_sstable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.sst");
        write_test_sstable(&path, &[]);

        let mut reader = SsTableReader::open(&path).unwrap();
        assert_eq!(reader.entry_count(), 0);
        assert_eq!(reader.get(b"anything").unwrap(), None);
        assert!(reader.read_all().unwrap().is_empty());
    }

    #[test]
    fn multi_block_sstable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi.sst");

        // Use tiny blocks to force multiple data blocks.
        let mut writer = SsTableWriter::with_block_size(&path, 32).unwrap();
        for i in 0u32..100 {
            let key = format!("key-{i:05}");
            let val = format!("val-{i:05}");
            writer.add(key.as_bytes(), val.as_bytes()).unwrap();
        }
        writer.finish().unwrap();

        let mut reader = SsTableReader::open(&path).unwrap();
        assert_eq!(reader.entry_count(), 100);

        // Spot-check lookups.
        assert_eq!(
            reader.get(b"key-00000").unwrap(),
            Some(b"val-00000".to_vec())
        );
        assert_eq!(
            reader.get(b"key-00050").unwrap(),
            Some(b"val-00050".to_vec())
        );
        assert_eq!(
            reader.get(b"key-00099").unwrap(),
            Some(b"val-00099".to_vec())
        );
        assert_eq!(reader.get(b"key-00100").unwrap(), None);

        // Full iteration.
        let entries = reader.read_all().unwrap();
        assert_eq!(entries.len(), 100);
        assert_eq!(entries[0].0, b"key-00000");
        assert_eq!(entries[99].0, b"key-00099");
    }
}
