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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            expansion_factor: 2.0,
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
