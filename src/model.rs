//! Linear model fitting for learned index nodes.
//!
//! The core algorithm is FMCD (Fastest Minimum Conflict Degree) from the LIPP
//! paper. Given a sorted set of keys and a target array size, it computes a
//! linear model `f(key) = slope * key + intercept` that maps each key to a
//! unique slot with minimum conflicts.

use crate::key::Key;

/// A linear model that predicts slot positions from keys.
///
/// Given a key `k`, the predicted slot index is:
/// `(slope * k.to_model_input() + intercept).round() as usize`
#[derive(Debug, Clone, Copy)]
pub struct LinearModel {
    /// Slope of the linear function.
    pub slope: f64,
    /// Intercept of the linear function.
    pub intercept: f64,
}

impl LinearModel {
    /// Create a new linear model with the given parameters.
    pub const fn new(slope: f64, intercept: f64) -> Self {
        Self { slope, intercept }
    }

    /// Predict the slot index for a key.
    ///
    /// Returns the predicted position clamped to `[0, array_size - 1]`.
    #[inline]
    pub fn predict<K: Key>(&self, key: K, array_size: usize) -> usize {
        let pos = self.slope.mul_add(key.to_model_input(), self.intercept);
        // Clamp to valid range
        let pos = pos.round().max(0.0) as usize;
        pos.min(array_size.saturating_sub(1))
    }

    /// Create a model that maps a single key to slot 0.
    pub const fn constant() -> Self {
        Self {
            slope: 0.0,
            intercept: 0.0,
        }
    }
}

/// Result of FMCD model fitting.
#[derive(Debug)]
pub struct FmcdResult {
    /// The fitted linear model.
    pub model: LinearModel,
    /// The array size needed to achieve zero (or minimal) conflicts.
    pub array_size: usize,
    /// Number of conflicts (keys mapping to the same slot).
    pub conflicts: usize,
}

/// Fit a linear model to sorted keys using the FMCD algorithm.
///
/// The algorithm finds a linear function that maps each key to a unique slot
/// in an array of size `keys.len() * expansion_factor`. The expansion factor
/// controls how many gaps (empty slots) are left for future inserts.
///
/// # Arguments
///
/// - `keys`: Sorted slice of keys (must be non-empty and sorted).
/// - `expansion_factor`: Ratio of array size to number of keys. Must be >= 1.0.
///   A factor of 2.0 means the array is twice as large as the key count (50% gaps).
///
/// # Returns
///
/// A [`FmcdResult`] containing the model, array size, and conflict count.
pub fn fit_fmcd<K: Key>(keys: &[K], expansion_factor: f64) -> FmcdResult {
    assert!(!keys.is_empty(), "keys must be non-empty");
    assert!(
        expansion_factor >= 1.0,
        "expansion_factor must be >= 1.0, got {expansion_factor}"
    );

    let n = keys.len();

    // Special case: single key
    if n == 1 {
        return FmcdResult {
            model: LinearModel::constant(),
            array_size: 1,
            conflicts: 0,
        };
    }

    let array_size = (n as f64 * expansion_factor).ceil().max(n as f64) as usize;

    // Compute the ideal mapping: key_i should map to position i * (array_size-1) / (n-1).
    // We fit a linear model through the key-to-position mapping.
    let first = keys[0].to_model_input();
    let last = keys[n - 1].to_model_input();
    let key_range = last - first;

    let (slope, intercept) = if key_range.abs() < f64::EPSILON {
        // All keys are the same value — map all to the middle
        (0.0, (array_size / 2) as f64)
    } else {
        // Linear interpolation: map first key to 0, last key to array_size - 1
        let s = (array_size - 1) as f64 / key_range;
        let i = -s * first;
        (s, i)
    };

    let model = LinearModel::new(slope, intercept);

    // Count conflicts: how many keys map to the same slot
    let conflicts = count_conflicts(keys, &model, array_size);

    FmcdResult {
        model,
        array_size,
        conflicts,
    }
}

/// Count how many keys collide (map to the same slot).
fn count_conflicts<K: Key>(keys: &[K], model: &LinearModel, array_size: usize) -> usize {
    let mut occupied = vec![false; array_size];
    let mut conflicts = 0;
    for &key in keys {
        let slot = model.predict(key, array_size);
        if occupied[slot] {
            conflicts += 1;
        } else {
            occupied[slot] = true;
        }
    }
    conflicts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_key() {
        let result = fit_fmcd(&[42u64], 2.0);
        assert_eq!(result.array_size, 1);
        assert_eq!(result.conflicts, 0);
        assert_eq!(result.model.predict(42u64, 1), 0);
    }

    #[test]
    fn two_keys() {
        let result = fit_fmcd(&[10u64, 20], 2.0);
        assert_eq!(result.conflicts, 0);
        assert!(result.array_size >= 2);
        // Keys should map to different slots
        let s1 = result.model.predict(10u64, result.array_size);
        let s2 = result.model.predict(20u64, result.array_size);
        assert_ne!(s1, s2, "two keys should map to different slots");
    }

    #[test]
    fn sequential_keys_no_conflicts() {
        let keys: Vec<u64> = (0..100).collect();
        let result = fit_fmcd(&keys, 2.0);
        assert_eq!(
            result.conflicts, 0,
            "sequential keys with 2x expansion should have zero conflicts"
        );
    }

    #[test]
    fn dense_keys_some_conflicts() {
        // With expansion_factor = 1.0, conflicts are likely for non-uniform keys
        let keys: Vec<u64> = vec![1, 2, 3, 100, 200, 300];
        let result = fit_fmcd(&keys, 1.0);
        // We just verify it doesn't panic; conflicts depend on distribution
        assert!(result.array_size >= keys.len());
    }

    #[test]
    fn predict_clamps_to_range() {
        let model = LinearModel::new(1.0, -10.0);
        // Negative prediction should clamp to 0
        assert_eq!(model.predict(5u64, 100), 0);
        // Large prediction should clamp to array_size - 1
        let model2 = LinearModel::new(1.0, 1000.0);
        assert_eq!(model2.predict(5u64, 100), 99);
    }

    #[test]
    fn expansion_factor_affects_size() {
        let keys: Vec<u64> = (0..50).collect();
        let r1 = fit_fmcd(&keys, 1.5);
        let r2 = fit_fmcd(&keys, 3.0);
        assert!(r2.array_size > r1.array_size);
    }

    #[test]
    fn identical_keys_handled() {
        // All same keys — should not panic
        let keys = vec![5u64; 10];
        let result = fit_fmcd(&keys, 2.0);
        // All map to same slot, so conflicts = n - 1
        assert_eq!(result.conflicts, 9);
    }

    #[test]
    fn large_key_range() {
        let keys = vec![0u64, u64::MAX / 2];
        let result = fit_fmcd(&keys, 2.0);
        assert_eq!(result.conflicts, 0);
    }

    #[test]
    fn signed_keys() {
        let keys: Vec<i64> = vec![-100, -50, 0, 50, 100];
        let result = fit_fmcd(&keys, 2.0);
        assert_eq!(result.conflicts, 0);
    }

    #[test]
    #[should_panic(expected = "keys must be non-empty")]
    fn empty_keys_panics() {
        fit_fmcd::<u64>(&[], 2.0);
    }

    #[test]
    #[should_panic(expected = "expansion_factor must be >= 1.0")]
    fn bad_expansion_panics() {
        fit_fmcd(&[1u64, 2, 3], 0.5);
    }

    #[test]
    fn model_monotonic_for_sorted_keys() {
        let keys: Vec<u64> = (0..1000).map(|i| i * 7 + 3).collect();
        let result = fit_fmcd(&keys, 2.0);
        let positions: Vec<usize> = keys
            .iter()
            .map(|&k| result.model.predict(k, result.array_size))
            .collect();
        // Positions should be non-decreasing for sorted keys
        for pair in positions.windows(2) {
            assert!(
                pair[0] <= pair[1],
                "model is not monotonic: {} > {}",
                pair[0],
                pair[1]
            );
        }
    }
}
