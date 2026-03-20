//! # scry-index
//!
//! A concurrent sorted key-value map backed by learned index structures.
//!
//! This crate provides [`LearnedMap`], a sorted map that uses piecewise linear
//! models to predict key positions, achieving O(1) expected lookup time for
//! keys matching the data distribution.
//!
//! # Concurrency
//!
//! All operations take `&self` and are safe to call from multiple threads.
//! Reads are lock-free (atomic loads under an epoch guard). Writes use
//! CAS retry loops on individual slots — no global lock.
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
//! let guard = map.guard();
//!
//! map.insert(42u64, "hello", &guard);
//! map.insert(17u64, "world", &guard);
//!
//! assert_eq!(map.get(&42, &guard), Some(&"hello"));
//! assert_eq!(map.get(&99, &guard), None);
//! assert_eq!(map.len(), 2);
//! ```

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
