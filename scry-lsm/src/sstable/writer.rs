//! SSTable writer — builds sorted immutable files from memtable drain output.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use super::{DEFAULT_BLOCK_SIZE, MAGIC};

/// Builds an SSTable file from a sorted sequence of key-value pairs.
pub struct SsTableWriter {
    writer: BufWriter<File>,
    block_size: usize,
    // Current data block accumulation.
    block_buf: Vec<u8>,
    block_first_key: Option<Vec<u8>>,
    block_offset: u64,
    // Index entries: (first_key, offset, block_len).
    index: Vec<(Vec<u8>, u64, u64)>,
    entry_count: u64,
    min_key: Option<Vec<u8>>,
    max_key: Option<Vec<u8>>,
}

impl SsTableWriter {
    /// Create a new writer that will produce an SSTable at the given path.
    pub fn new(path: &Path) -> std::io::Result<Self> {
        Self::with_block_size(path, DEFAULT_BLOCK_SIZE)
    }

    /// Create a new writer with a custom target block size.
    pub fn with_block_size(path: &Path, block_size: usize) -> std::io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            block_size,
            block_buf: Vec::with_capacity(block_size),
            block_first_key: None,
            block_offset: 0,
            index: Vec::new(),
            entry_count: 0,
            min_key: None,
            max_key: None,
        })
    }

    /// Add a key-value entry. Entries **must** be added in sorted key order.
    pub fn add(&mut self, key: &[u8], value: &[u8]) -> std::io::Result<()> {
        // Track min/max keys.
        if self.min_key.is_none() {
            self.min_key = Some(key.to_vec());
        }
        self.max_key = Some(key.to_vec());

        // Track first key of the current block.
        if self.block_first_key.is_none() {
            self.block_first_key = Some(key.to_vec());
        }

        // Encode entry into block buffer.
        self.block_buf
            .extend_from_slice(&(key.len() as u32).to_le_bytes());
        self.block_buf.extend_from_slice(key);
        self.block_buf
            .extend_from_slice(&(value.len() as u32).to_le_bytes());
        self.block_buf.extend_from_slice(value);
        self.entry_count += 1;

        // Flush block if it exceeds the target size.
        if self.block_buf.len() >= self.block_size {
            self.flush_block()?;
        }

        Ok(())
    }

    /// Finish writing the SSTable. Returns (entry_count, min_key, max_key).
    pub fn finish(mut self) -> std::io::Result<(u64, Vec<u8>, Vec<u8>)> {
        // Flush any remaining block data.
        if !self.block_buf.is_empty() {
            self.flush_block()?;
        }

        let index_offset = self.block_offset;

        // Write the index block.
        for (first_key, offset, block_len) in &self.index {
            self.writer
                .write_all(&(first_key.len() as u32).to_le_bytes())?;
            self.writer.write_all(first_key)?;
            self.writer.write_all(&offset.to_le_bytes())?;
            self.writer.write_all(&block_len.to_le_bytes())?;
        }

        // Write the footer.
        self.writer.write_all(&index_offset.to_le_bytes())?;
        self.writer.write_all(&self.entry_count.to_le_bytes())?;
        self.writer.write_all(&MAGIC.to_le_bytes())?;

        self.writer.flush()?;

        let min_key = self.min_key.unwrap_or_default();
        let max_key = self.max_key.unwrap_or_default();
        Ok((self.entry_count, min_key, max_key))
    }

    /// Flush the current block buffer to disk and record an index entry.
    fn flush_block(&mut self) -> std::io::Result<()> {
        let block_len = self.block_buf.len() as u64;
        self.writer.write_all(&self.block_buf)?;

        if let Some(first_key) = self.block_first_key.take() {
            self.index.push((first_key, self.block_offset, block_len));
        }

        self.block_offset += block_len;
        self.block_buf.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_empty_sstable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.sst");
        let writer = SsTableWriter::new(&path).unwrap();
        let (count, _, _) = writer.finish().unwrap();
        assert_eq!(count, 0);

        // File should contain only the index block (empty) and footer.
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(meta.len(), crate::sstable::FOOTER_SIZE as u64);
    }

    #[test]
    fn write_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.sst");
        let mut writer = SsTableWriter::new(&path).unwrap();
        writer.add(b"aaa", b"111").unwrap();
        writer.add(b"bbb", b"222").unwrap();
        writer.add(b"ccc", b"333").unwrap();
        let (count, min, max) = writer.finish().unwrap();

        assert_eq!(count, 3);
        assert_eq!(min, b"aaa");
        assert_eq!(max, b"ccc");
    }
}
