//! # scry-index
//!
//! A concurrent sorted key-value map backed by learned index structures.
//!
//! This crate provides [`LearnedMap`], a sorted map that uses piecewise linear
//! models to predict key positions, achieving O(1) expected lookup time for
//! keys matching the data distribution. A [`LearnedSet`] wrapper is also
//! available for key-only use cases.
//!
//! # Supported key types
//!
//! Any type implementing [`Key`] can be used. Built-in implementations cover:
//!
//! - Integer types: `u8`..`u128`, `i8`..`i128`
//! - Byte arrays: `[u8; N]` for any `N` (UUIDs, hashes, etc.)
//! - Heap-allocated: `String`, `Vec<u8>`
//!
//! # Concurrency
//!
//! All operations take `&self` and are safe to call from multiple threads.
//! Reads are lock-free (atomic loads under an epoch guard). Writes use
//! CAS retry loops on individual slots with no global lock.
//!
//! # Algorithm
//!
//! Based on LIPP's chain method: each node contains a linear model
//! `f(key) = slope * key + intercept` that maps keys to unique slots in a
//! fixed-size array. Collisions are resolved by creating child nodes
//! (chaining), not by shifting data. This yields zero prediction error and
//! enables fine-grained concurrent access.
//!
//! # Example
//!
//! ```
//! use scry_index::LearnedMap;
//!
//! let map = LearnedMap::new();
//! let m = map.pin();
//!
//! m.insert(42u64, "hello");
//! m.insert(17u64, "world");
//!
//! assert_eq!(m.get(&42), Some(&"hello"));
//! assert_eq!(m.get(&99), None);
//! assert_eq!(m.len(), 2);
//! ```
//!
//! For pre-sorted data, [`LearnedMap::bulk_load`] builds an optimal tree in
//! one pass:
//!
//! ```
//! use scry_index::LearnedMap;
//!
//! let data: Vec<(u64, &str)> = vec![(1, "a"), (2, "b"), (3, "c")];
//! let map = LearnedMap::bulk_load(&data).unwrap();
//! assert_eq!(map.pin().get(&2), Some(&"b"));
//! ```
//!
//! # Feature flags
//!
//! - **`serde`**: Enables `Serialize`/`Deserialize` for [`LearnedMap`],
//!   [`LearnedSet`], [`Config`], and [`Error`].

mod build;
mod config;
mod error;
mod insert;
mod iter;
mod key;
mod lookup;
mod map;
mod model;
mod node;
mod rebuild;
mod remove;
mod set;

pub use config::Config;
pub use error::{Error, Result};
pub use iter::{Iter, Range};
pub use key::Key;
pub use map::{Guard, LearnedMap, MapRef};
pub use set::{LearnedSet, SetRef};
