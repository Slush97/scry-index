//! Configuration for the learned index.

/// Configuration parameters for [`LearnedMap`](crate::LearnedMap).
#[derive(Debug, Clone)]
pub struct Config {
    /// Expansion factor for node arrays.
    ///
    /// Controls the ratio of array size to key count. A factor of 2.0 means
    /// the array is twice as large as the number of keys (50% gaps), leaving
    /// room for future inserts without conflicts.
    ///
    /// Higher values reduce conflicts but use more memory.
    ///
    /// Default: `2.0`. Must be `>= 1.0`.
    pub expansion_factor: f64,

    /// Whether to automatically rebuild the tree after a threshold of inserts.
    ///
    /// When enabled, the map periodically rebuilds with optimal FMCD model
    /// fitting to keep the tree shallow and lookups fast. The rebuild threshold
    /// is `(len / 4).clamp(16, 10_000)`.
    ///
    /// Default: `true`.
    pub auto_rebuild: bool,

    /// Maximum subtree depth before a localized rebuild is triggered.
    ///
    /// When an insert descends through more child nodes than this threshold,
    /// the inserting thread rebuilds the degraded subtree inline. Only applies
    /// when `auto_rebuild` is `true`. Set to `usize::MAX` to disable.
    ///
    /// Default: `8`.
    pub rebuild_depth_threshold: usize,
}

impl Config {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the expansion factor.
    ///
    /// # Panics
    ///
    /// Panics if `factor < 1.0`.
    pub fn expansion_factor(mut self, factor: f64) -> Self {
        assert!(factor >= 1.0, "expansion_factor must be >= 1.0");
        self.expansion_factor = factor;
        self
    }

    /// Enable or disable automatic rebuilds.
    pub fn auto_rebuild(mut self, enabled: bool) -> Self {
        self.auto_rebuild = enabled;
        self
    }

    /// Set the maximum subtree depth before localized rebuild triggers.
    pub fn rebuild_depth_threshold(mut self, threshold: usize) -> Self {
        self.rebuild_depth_threshold = threshold;
        self
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            expansion_factor: 2.0,
            auto_rebuild: true,
            rebuild_depth_threshold: 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = Config::default();
        assert!((config.expansion_factor - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn builder_pattern() {
        let config = Config::new().expansion_factor(3.0);
        assert!((config.expansion_factor - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "expansion_factor must be >= 1.0")]
    fn reject_low_expansion() {
        Config::new().expansion_factor(0.5);
    }
}
