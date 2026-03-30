//! Error types for the LSM engine.

use std::fmt;
use std::io;

/// Errors that can occur in the LSM engine.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// An I/O error occurred (WAL, SSTable, or filesystem).
    Io(io::Error),
    /// Data corruption was detected during recovery or read.
    Corruption(String),
    /// A lock was poisoned (a thread panicked while holding it).
    Poisoned,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Corruption(msg) => write!(f, "corruption: {msg}"),
            Self::Poisoned => write!(f, "lock poisoned"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Corruption(_) | Self::Poisoned => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// A specialized `Result` type for LSM engine operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Lock a `Mutex`, converting `PoisonError` to [`Error::Poisoned`].
pub fn lock<T>(mutex: &std::sync::Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>> {
    mutex.lock().map_err(|_| Error::Poisoned)
}

/// Read-lock an `RwLock`, converting `PoisonError` to [`Error::Poisoned`].
pub fn read_lock<T>(rw: &std::sync::RwLock<T>) -> Result<std::sync::RwLockReadGuard<'_, T>> {
    rw.read().map_err(|_| Error::Poisoned)
}

/// Write-lock an `RwLock`, converting `PoisonError` to [`Error::Poisoned`].
pub fn write_lock<T>(rw: &std::sync::RwLock<T>) -> Result<std::sync::RwLockWriteGuard<'_, T>> {
    rw.write().map_err(|_| Error::Poisoned)
}
