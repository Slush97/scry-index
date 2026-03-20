//! Error types for scry-index.

use std::fmt;

/// Result type alias for scry-index operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during index operations.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The input data is empty.
    EmptyData,
    /// The input data is not sorted.
    NotSorted,
    /// A key could not be converted to a model input.
    InvalidKey {
        /// Description of what went wrong.
        detail: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyData => write!(f, "input data is empty"),
            Self::NotSorted => write!(f, "input data is not sorted"),
            Self::InvalidKey { detail } => write!(f, "invalid key: {detail}"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        assert_eq!(Error::EmptyData.to_string(), "input data is empty");
        assert_eq!(Error::NotSorted.to_string(), "input data is not sorted");
        assert_eq!(
            Error::InvalidKey {
                detail: "overflow".into()
            }
            .to_string(),
            "invalid key: overflow"
        );
    }

    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Error>();
    }
}
