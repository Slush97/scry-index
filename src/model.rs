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
///
/// When `midpoint` is set, the linear formula is overridden with a binary
/// split: keys with `to_exact_ordinal() <= midpoint` predict to slot 0,
/// keys above predict to the last slot. This handles the degenerate case
/// where `f64` cannot distinguish two distinct keys.
#[derive(Debug, Clone, Copy)]
pub struct LinearModel {
    /// Slope of the linear function.
    pub slope: f64,
    /// Intercept of the linear function.
    pub intercept: f64,
    /// When set, overrides linear prediction with a binary split.
    midpoint: Option<i128>,
}

impl LinearModel {
    /// Create a new linear model with the given parameters.
    pub const fn new(slope: f64, intercept: f64) -> Self {
        Self {
            slope,
            intercept,
            midpoint: None,
        }
    }

    /// Create a binary-split model that separates keys at the given midpoint.
    ///
    /// Keys with `to_exact_ordinal() <= midpoint` predict to slot 0;
    /// keys above predict to `array_size - 1`.
    pub const fn binary_split(midpoint: i128) -> Self {
        Self {
            slope: 0.0,
            intercept: 0.0,
            midpoint: Some(midpoint),
        }
    }

    /// Predict the slot index for a key.
    ///
    /// Returns the predicted position clamped to `[0, array_size - 1]`.
    #[inline]
    pub fn predict<K: Key>(&self, key: &K, array_size: usize) -> usize {
        if let Some(mid) = self.midpoint {
            return if key.to_exact_ordinal() <= mid {
                0
            } else {
                array_size.saturating_sub(1)
            };
        }
        let pos = self.slope.mul_add(key.to_model_input(), self.intercept);
        let pos = pos.round().max(0.0) as usize;
        pos.min(array_size.saturating_sub(1))
    }

    /// Predict the slot index from a raw `f64` model input value.
    ///
    /// This avoids needing a `Key` reference, useful in FMCD fitting where
    /// only the `f64` representation is available via a closure.
    ///
    /// Returns the predicted position clamped to `[0, array_size - 1]`.
    #[inline]
    pub fn predict_raw(&self, value: f64, array_size: usize) -> usize {
        let pos = self.slope.mul_add(value, self.intercept);
        let pos = pos.round().max(0.0) as usize;
        pos.min(array_size.saturating_sub(1))
    }

    /// Create a model that maps a single key to slot 0.
    pub const fn constant() -> Self {
        Self {
            slope: 0.0,
            intercept: 0.0,
            midpoint: None,
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

/// Maximum number of candidate slopes to try in FMCD.
const MAX_FMCD_CANDIDATES: usize = 32;

/// Fit a linear model to sorted keys using the FMCD algorithm.
///
/// Finds a linear function `f(key) = slope * key + intercept` that maps each
/// key to a slot in an array, minimizing the number of collisions (conflicts).
///
/// The algorithm tries candidate slopes derived from the maximum inter-key gap
/// (the FMCD heuristic from the LIPP paper) and selects the slope with the
/// fewest conflicts. Linear interpolation is always tried as a baseline.
///
/// # Arguments
///
/// - `n`: Number of keys (must be > 0).
/// - `key_input`: Closure returning the `f64` model input for the key at index `i`.
/// - `expansion_factor`: Ratio of array size to number of keys. Must be >= 1.0.
///   A factor of 2.0 means the array is twice as large as the key count (50% gaps).
/// - `range_headroom`: Extra key range coverage beyond the current data. When > 0,
///   the model covers `(1 + range_headroom)` times the actual key range and the
///   array grows proportionally, leaving space for future keys beyond the current
///   max without clamping. Must be >= 0.0.
///
/// # Returns
///
/// A [`FmcdResult`] containing the model, array size, and conflict count.
pub fn fit_fmcd(
    n: usize,
    key_input: impl Fn(usize) -> f64,
    expansion_factor: f64,
    range_headroom: f64,
) -> FmcdResult {
    assert!(n > 0, "keys must be non-empty");
    assert!(
        expansion_factor >= 1.0,
        "expansion_factor must be >= 1.0, got {expansion_factor}"
    );

    // Special case: single key
    if n == 1 {
        return FmcdResult {
            model: LinearModel::constant(),
            array_size: 1,
            conflicts: 0,
        };
    }

    let headroom_mult = 1.0 + range_headroom;
    let array_size = (n as f64 * expansion_factor * headroom_mult)
        .ceil()
        .max(n as f64) as usize;

    let first = key_input(0);
    let last = key_input(n - 1);
    let key_range = last - first;

    if key_range.abs() < f64::EPSILON {
        // All keys are the same value. Map all to the middle.
        return FmcdResult {
            model: LinearModel::new(0.0, (array_size / 2) as f64),
            array_size,
            conflicts: n - 1,
        };
    }

    let effective_range = key_range * headroom_mult;

    // --- Candidate 1: Linear interpolation (baseline) ---
    let lin_slope = (array_size - 1) as f64 / effective_range;
    let lin_intercept = -lin_slope * first;
    let lin_model = LinearModel::new(lin_slope, lin_intercept);
    let lin_conflicts = count_conflicts_fast(n, &key_input, &lin_model, array_size);

    if lin_conflicts == 0 {
        return FmcdResult {
            model: lin_model,
            array_size,
            conflicts: 0,
        };
    }

    let mut best_model = lin_model;
    let mut best_conflicts = lin_conflicts;

    // --- FMCD: try slopes derived from the maximum inter-key gap ---
    // The insight: the optimal slope often places the widest-gap pair at
    // consecutive slots. Trying multiples of 1/max_gap searches
    // for the minimum-conflict model.
    let mut max_gap = 0.0_f64;
    for i in 0..n - 1 {
        let gap = key_input(i + 1) - key_input(i);
        max_gap = max_gap.max(gap);
    }

    if max_gap > f64::EPSILON {
        // Upper bound: slope where predict(last) = array_size - 1
        let max_slope = (array_size - 1) as f64 / key_range;

        for j in 1..=MAX_FMCD_CANDIDATES {
            let slope = j as f64 / max_gap;
            if slope > max_slope {
                break;
            }

            let intercept = -slope * first;
            let model = LinearModel::new(slope, intercept);
            let conflicts = count_conflicts_fast(n, &key_input, &model, array_size);

            if conflicts < best_conflicts {
                best_conflicts = conflicts;
                best_model = model;
            }
            if best_conflicts == 0 {
                break;
            }
        }
    }

    FmcdResult {
        model: best_model,
        array_size,
        conflicts: best_conflicts,
    }
}

/// Fast conflict counting for sorted keys.
///
/// Since keys are sorted and the model is monotonic, we only need to check
/// if adjacent keys map to the same slot (no allocation needed).
fn count_conflicts_fast(
    n: usize,
    key_input: &impl Fn(usize) -> f64,
    model: &LinearModel,
    array_size: usize,
) -> usize {
    let mut conflicts = 0;
    let mut prev_slot = model.predict_raw(key_input(0), array_size);
    for i in 1..n {
        let slot = model.predict_raw(key_input(i), array_size);
        if slot == prev_slot {
            conflicts += 1;
        }
        prev_slot = slot;
    }
    conflicts
}

/// Count how many keys collide (map to the same slot).
///
/// Full version: allocates a boolean array. Works for unsorted keys.
#[cfg(test)]
fn count_conflicts(
    n: usize,
    key_input: &impl Fn(usize) -> f64,
    model: &LinearModel,
    array_size: usize,
) -> usize {
    let mut occupied = vec![false; array_size];
    let mut conflicts = 0;
    for i in 0..n {
        let slot = model.predict_raw(key_input(i), array_size);
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
        let keys = [42u64];
        let result = fit_fmcd(keys.len(), |i| keys[i].to_model_input(), 2.0, 0.0);
        assert_eq!(result.array_size, 1);
        assert_eq!(result.conflicts, 0);
        assert_eq!(result.model.predict(&42u64, 1), 0);
    }

    #[test]
    fn two_keys() {
        let keys = [10u64, 20];
        let result = fit_fmcd(keys.len(), |i| keys[i].to_model_input(), 2.0, 0.0);
        assert_eq!(result.conflicts, 0);
        assert!(result.array_size >= 2);
        let s1 = result.model.predict(&10u64, result.array_size);
        let s2 = result.model.predict(&20u64, result.array_size);
        assert_ne!(s1, s2, "two keys should map to different slots");
    }

    #[test]
    fn sequential_keys_no_conflicts() {
        let keys: Vec<u64> = (0..100).collect();
        let result = fit_fmcd(keys.len(), |i| keys[i].to_model_input(), 2.0, 0.0);
        assert_eq!(
            result.conflicts, 0,
            "sequential keys with 2x expansion should have zero conflicts"
        );
    }

    #[test]
    fn dense_keys_some_conflicts() {
        let keys: Vec<u64> = vec![1, 2, 3, 100, 200, 300];
        let result = fit_fmcd(keys.len(), |i| keys[i].to_model_input(), 1.0, 0.0);
        assert!(result.array_size >= keys.len());
    }

    #[test]
    fn predict_clamps_to_range() {
        let model = LinearModel::new(1.0, -10.0);
        assert_eq!(model.predict(&5u64, 100), 0);
        let model2 = LinearModel::new(1.0, 1000.0);
        assert_eq!(model2.predict(&5u64, 100), 99);
    }

    #[test]
    fn expansion_factor_affects_size() {
        let keys: Vec<u64> = (0..50).collect();
        let r1 = fit_fmcd(keys.len(), |i| keys[i].to_model_input(), 1.5, 0.0);
        let r2 = fit_fmcd(keys.len(), |i| keys[i].to_model_input(), 3.0, 0.0);
        assert!(r2.array_size > r1.array_size);
    }

    #[test]
    fn identical_keys_handled() {
        let keys = [5u64; 10];
        let result = fit_fmcd(keys.len(), |i| keys[i].to_model_input(), 2.0, 0.0);
        assert_eq!(result.conflicts, 9);
    }

    #[test]
    fn large_key_range() {
        let keys = [0u64, u64::MAX / 2];
        let result = fit_fmcd(keys.len(), |i| keys[i].to_model_input(), 2.0, 0.0);
        assert_eq!(result.conflicts, 0);
    }

    #[test]
    fn signed_keys() {
        let keys: Vec<i64> = vec![-100, -50, 0, 50, 100];
        let result = fit_fmcd(keys.len(), |i| keys[i].to_model_input(), 2.0, 0.0);
        assert_eq!(result.conflicts, 0);
    }

    #[test]
    #[should_panic(expected = "keys must be non-empty")]
    fn empty_keys_panics() {
        fit_fmcd(0, |i: usize| i as f64, 2.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "expansion_factor must be >= 1.0")]
    fn bad_expansion_panics() {
        let keys = [1u64, 2, 3];
        fit_fmcd(keys.len(), |i| keys[i].to_model_input(), 0.5, 0.0);
    }

    #[test]
    fn model_monotonic_for_sorted_keys() {
        let keys: Vec<u64> = (0..1000).map(|i| i * 7 + 3).collect();
        let result = fit_fmcd(keys.len(), |i| keys[i].to_model_input(), 2.0, 0.0);
        let positions: Vec<usize> = keys
            .iter()
            .map(|k| result.model.predict(k, result.array_size))
            .collect();
        for pair in positions.windows(2) {
            assert!(
                pair[0] <= pair[1],
                "model is not monotonic: {} > {}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn binary_split_separates_keys() {
        let base: u64 = 1_700_000_000_000_000_000;
        let lo_ord = base.to_exact_ordinal();
        let hi_ord = (base + 1).to_exact_ordinal();
        let midpoint = lo_ord + (hi_ord - lo_ord) / 2;
        let model = LinearModel::binary_split(midpoint);

        // array_size = 2: should map lo to 0, hi to 1
        assert_eq!(model.predict(&base, 2), 0);
        assert_eq!(model.predict(&(base + 1), 2), 1);

        // array_size = 4: should map lo to 0, hi to 3
        assert_eq!(model.predict(&base, 4), 0);
        assert_eq!(model.predict(&(base + 1), 4), 3);
    }

    #[test]
    fn binary_split_many_keys() {
        let base: u64 = 1_700_000_000_000_000_000;
        // Split at midpoint between base+4 and base+5
        let mid = base.to_exact_ordinal() + 4;
        let model = LinearModel::binary_split(mid);

        for i in 0..=4u64 {
            assert_eq!(
                model.predict(&(base + i), 2),
                0,
                "base+{i} should go to slot 0"
            );
        }
        for i in 5..10u64 {
            assert_eq!(
                model.predict(&(base + i), 2),
                1,
                "base+{i} should go to slot 1"
            );
        }
    }

    #[test]
    fn fast_and_full_conflict_count_agree() {
        // Non-uniform keys where conflicts are likely with tight expansion
        let keys: Vec<u64> = vec![1, 2, 3, 4, 5, 100, 200, 300, 400, 500];
        let array_size = 15;
        let first = keys[0].to_model_input();
        let last = keys[keys.len() - 1].to_model_input();
        let slope = (array_size - 1) as f64 / (last - first);
        let intercept = -slope * first;
        let model = LinearModel::new(slope, intercept);

        let ki = |i: usize| keys[i].to_model_input();
        let fast = count_conflicts_fast(keys.len(), &ki, &model, array_size);
        let full = count_conflicts(keys.len(), &ki, &model, array_size);
        assert_eq!(
            fast, full,
            "fast ({fast}) and full ({full}) conflict counts disagree"
        );
    }

    #[test]
    fn fmcd_reduces_conflicts_for_nonuniform_keys() {
        // Non-uniform spacing where FMCD can find a better slope than
        // linear interpolation. Keys with varying gaps benefit from
        // slope optimization.
        let keys: Vec<u64> = vec![0, 1, 2, 3, 10, 11, 12, 13, 20, 21];
        let result = fit_fmcd(keys.len(), |i| keys[i].to_model_input(), 2.0, 0.0);
        assert!(
            result.conflicts <= 2,
            "FMCD should achieve <= 2 conflicts for mildly clustered keys, got {}",
            result.conflicts
        );
    }

    #[test]
    fn headroom_increases_array_size() {
        let keys: Vec<u64> = (0..100).collect();
        let r0 = fit_fmcd(keys.len(), |i| keys[i].to_model_input(), 2.0, 0.0);
        let r1 = fit_fmcd(keys.len(), |i| keys[i].to_model_input(), 2.0, 1.0);
        assert!(
            r1.array_size > r0.array_size,
            "headroom=1.0 should produce larger array: {} vs {}",
            r1.array_size,
            r0.array_size
        );
    }

    #[test]
    fn headroom_keys_not_at_edge() {
        let keys: Vec<u64> = (0..100).collect();
        let result = fit_fmcd(keys.len(), |i| keys[i].to_model_input(), 2.0, 1.0);
        // With headroom=1.0, the last key should map to approximately
        // the middle of the array, not the end.
        let last_slot = result.model.predict(&99u64, result.array_size);
        let midpoint = result.array_size / 2;
        assert!(
            last_slot <= midpoint + midpoint / 4,
            "last key at slot {last_slot}, expected near {midpoint} (array_size={})",
            result.array_size
        );
    }

    #[test]
    fn uniform_still_zero_conflicts_with_fmcd() {
        // Regression guard: sequential keys with good expansion should
        // still get 0 conflicts after the FMCD rewrite.
        let keys: Vec<u64> = (0..1000).collect();
        let result = fit_fmcd(keys.len(), |i| keys[i].to_model_input(), 2.0, 0.0);
        assert_eq!(
            result.conflicts, 0,
            "sequential keys should have 0 conflicts"
        );
    }
}
