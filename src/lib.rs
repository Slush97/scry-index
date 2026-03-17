//! # scry-index
//!
//! A concurrent sorted key-value map backed by learned index structures.
//!
//! This crate provides [`LearnedMap`], a sorted map that uses piecewise linear
//! models to predict key positions, achieving O(1) expected lookup time for
//! keys matching the data distribution.
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
//! let mut map = LearnedMap::new();
//! map.insert(42u64, "hello");
//! map.insert(17u64, "world");
//!
//! assert_eq!(map.get(&42), Some(&"hello"));
//! assert_eq!(map.get(&99), None);
//! assert_eq!(map.len(), 2);
//!
//! for (k, v) in &map {
//!     println!("{k}: {v}");
//! }
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
mod remove;
mod set;

pub use config::Config;
pub use error::{Error, Result};
pub use key::Key;
pub use map::LearnedMap;
pub use set::LearnedSet;
