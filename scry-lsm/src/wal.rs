//! Write-ahead log for crash recovery.
//!
//! Every write is appended to the WAL before being applied to the memtable.
//! If the process crashes, the WAL is replayed on startup to reconstruct the
//! memtable state.
//!
//! # Format
//!
//! Each record is:
//! ```text
//! [type: u8][key_len: u32 LE][key: [u8; key_len]][value_len: u32 LE][value: [u8; value_len]]
//! ```
//!
//! Where type is `0x01` for Put and `0x02` for Delete (value_len = 0 for deletes).

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

const RECORD_PUT: u8 = 0x01;
const RECORD_DELETE: u8 = 0x02;

/// A record recovered from the WAL during replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WalRecord {
    /// A put operation.
    Put {
        /// The key.
        key: Vec<u8>,
        /// The value.
        value: Vec<u8>,
    },
    /// A delete operation.
    Delete {
        /// The key to delete.
        key: Vec<u8>,
    },
}

/// An append-only write-ahead log.
pub struct Wal {
    path: PathBuf,
    writer: BufWriter<File>,
}

impl Wal {
    /// Open or create a WAL at the given path.
    pub fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            writer: BufWriter::new(file),
        })
    }

    /// Append a put record to the WAL.
    pub fn append_put(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        self.writer.write_all(&[RECORD_PUT])?;
        self.writer.write_all(&(key.len() as u32).to_le_bytes())?;
        self.writer.write_all(key)?;
        self.writer.write_all(&(value.len() as u32).to_le_bytes())?;
        self.writer.write_all(value)?;
        Ok(())
    }

    /// Append a delete record to the WAL.
    pub fn append_delete(&mut self, key: &[u8]) -> Result<()> {
        self.writer.write_all(&[RECORD_DELETE])?;
        self.writer.write_all(&(key.len() as u32).to_le_bytes())?;
        self.writer.write_all(key)?;
        // Delete records have zero-length value.
        self.writer.write_all(&0_u32.to_le_bytes())?;
        Ok(())
    }

    /// Flush buffered writes to the OS.
    pub fn sync(&mut self) -> Result<()> {
        use std::io::Write as _;
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(())
    }

    /// Truncate the WAL (called after a successful memtable flush to SSTable).
    pub fn truncate(&mut self) -> Result<()> {
        use std::io::Write as _;
        self.writer.flush()?;
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        self.writer = BufWriter::new(file);
        Ok(())
    }

    /// Replay all records from the WAL file.
    ///
    /// Returns records in order. Incomplete trailing records (from a crash
    /// mid-write) are silently skipped.
    pub fn recover(path: &Path) -> Result<Vec<WalRecord>> {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };

        let mut reader = BufReader::new(file);
        let mut records = Vec::new();

        loop {
            // Read record type.
            let mut type_buf = [0u8; 1];
            if reader.read_exact(&mut type_buf).is_err() {
                break; // EOF or incomplete record.
            }

            // Read key.
            let Ok(key) = read_length_prefixed(&mut reader) else {
                break; // Incomplete record.
            };

            // Read value.
            let Ok(value) = read_length_prefixed(&mut reader) else {
                break; // Incomplete record.
            };

            match type_buf[0] {
                RECORD_PUT => records.push(WalRecord::Put { key, value }),
                RECORD_DELETE => records.push(WalRecord::Delete { key }),
                unknown => {
                    return Err(Error::Corruption(format!(
                        "unknown WAL record type: {unknown:#x}"
                    )));
                }
            }
        }

        Ok(records)
    }
}

/// Read a length-prefixed byte sequence: `[len: u32 LE][data: [u8; len]]`.
fn read_length_prefixed(reader: &mut impl Read) -> std::io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut data = vec![0u8; len];
    reader.read_exact(&mut data)?;
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_put_and_delete() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        // Write some records.
        {
            let mut wal = Wal::open(&wal_path).unwrap();
            wal.append_put(b"key1", b"value1").unwrap();
            wal.append_put(b"key2", b"value2").unwrap();
            wal.append_delete(b"key1").unwrap();
            wal.sync().unwrap();
        }

        // Recover and verify.
        let records = Wal::recover(&wal_path).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(
            records[0],
            WalRecord::Put {
                key: b"key1".to_vec(),
                value: b"value1".to_vec()
            }
        );
        assert_eq!(
            records[1],
            WalRecord::Put {
                key: b"key2".to_vec(),
                value: b"value2".to_vec()
            }
        );
        assert_eq!(
            records[2],
            WalRecord::Delete {
                key: b"key1".to_vec()
            }
        );
    }

    #[test]
    fn truncate_clears_wal() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut wal = Wal::open(&wal_path).unwrap();
        wal.append_put(b"key", b"value").unwrap();
        wal.sync().unwrap();
        wal.truncate().unwrap();

        let records = Wal::recover(&wal_path).unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn recover_nonexistent_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let records = Wal::recover(&dir.path().join("nope.wal")).unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn incomplete_record_is_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        // Write a complete record followed by a partial one.
        {
            let mut wal = Wal::open(&wal_path).unwrap();
            wal.append_put(b"good", b"data").unwrap();
            wal.sync().unwrap();
        }
        // Append garbage (incomplete record).
        {
            use std::io::Write as _;
            let mut file = OpenOptions::new().append(true).open(&wal_path).unwrap();
            file.write_all(&[RECORD_PUT, 0xFF, 0xFF]).unwrap();
        }

        let records = Wal::recover(&wal_path).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0],
            WalRecord::Put {
                key: b"good".to_vec(),
                value: b"data".to_vec()
            }
        );
    }
}
