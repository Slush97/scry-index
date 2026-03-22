//! The [`Key`] trait for types usable as learned index keys.
//!
//! Keys must be ordered, copyable, and convertible to `f64` for linear model
//! prediction. This conversion must be monotonic: if `a < b` then
//! `a.to_model_input() < b.to_model_input()`.

/// A key type usable in a learned index.
///
/// # Contract
///
/// - `to_model_input` must be a **monotonic** function: if `a < b` then
///   `a.to_model_input() <= b.to_model_input()`. Note: the mapping is
///   non-strict — distinct keys may produce the same `f64` due to precision
///   loss (e.g., `u64` keys above 2^53). The index handles this correctly
///   via the `to_exact_ordinal` fallback.
/// - The returned `f64` must be finite (not NaN or infinity).
/// - The mapping should preserve relative distances where possible, so that
///   linear models can fit the key distribution effectively.
/// - `to_exact_ordinal` must be a **strictly monotonic, injective** function:
///   if `a < b` then `a.to_exact_ordinal() < b.to_exact_ordinal()`.
pub trait Key: Copy + Ord + Send + Sync + std::fmt::Debug + 'static {
    /// Convert this key to a `f64` value for model prediction.
    ///
    /// The conversion must be monotonic and return a finite value. It may
    /// be non-injective for large integer keys (precision loss above 2^53).
    fn to_model_input(self) -> f64;

    /// Convert this key to a lossless `i128` for exact comparison.
    ///
    /// This must be strictly monotonic and injective: if `a < b` then
    /// `a.to_exact_ordinal() < b.to_exact_ordinal()`. Used as a fallback
    /// for conflict resolution when `to_model_input` cannot distinguish keys.
    fn to_exact_ordinal(self) -> i128;
}

macro_rules! impl_key_unsigned {
    ($($t:ty),*) => {
        $(
            impl Key for $t {
                #[inline]
                fn to_model_input(self) -> f64 {
                    self as f64
                }

                #[inline]
                fn to_exact_ordinal(self) -> i128 {
                    self as i128
                }
            }
        )*
    };
}

macro_rules! impl_key_signed {
    ($($t:ty),*) => {
        $(
            impl Key for $t {
                #[inline]
                fn to_model_input(self) -> f64 {
                    self as f64
                }

                #[inline]
                fn to_exact_ordinal(self) -> i128 {
                    self as i128
                }
            }
        )*
    };
}

impl_key_unsigned!(u8, u16, u32, u64);
impl_key_signed!(i8, i16, i32, i64);

// u128/i128 lose precision in to_model_input beyond 2^53, but
// to_exact_ordinal is fully injective for all values.
impl Key for u128 {
    #[inline]
    fn to_model_input(self) -> f64 {
        self as f64
    }

    #[inline]
    #[allow(clippy::cast_possible_wrap)]
    fn to_exact_ordinal(self) -> i128 {
        // Order-preserving bijection: flip the sign bit so that
        // 0u128 -> i128::MIN and u128::MAX -> i128::MAX.
        (self as i128) ^ i128::MIN
    }
}

impl Key for i128 {
    #[inline]
    fn to_model_input(self) -> f64 {
        self as f64
    }

    #[inline]
    fn to_exact_ordinal(self) -> i128 {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u64_monotonic() {
        let keys: Vec<u64> = vec![0, 1, 100, 1000, u64::MAX / 2];
        for pair in keys.windows(2) {
            assert!(
                pair[0].to_model_input() < pair[1].to_model_input(),
                "monotonicity violated: {} >= {}",
                pair[0].to_model_input(),
                pair[1].to_model_input()
            );
        }
    }

    #[test]
    fn i64_monotonic() {
        let keys: Vec<i64> = vec![i64::MIN, -1000, -1, 0, 1, 1000, i64::MAX / 2];
        for pair in keys.windows(2) {
            assert!(
                pair[0].to_model_input() < pair[1].to_model_input(),
                "monotonicity violated for i64: {} >= {}",
                pair[0].to_model_input(),
                pair[1].to_model_input()
            );
        }
    }

    #[test]
    fn u32_finite() {
        for &k in &[0u32, 1, u32::MAX] {
            let v = k.to_model_input();
            assert!(v.is_finite(), "{k} produced non-finite model input {v}");
        }
    }

    #[test]
    fn key_is_send_sync() {
        fn assert_send_sync<T: Key>() {}
        assert_send_sync::<u64>();
        assert_send_sync::<i32>();
    }

    #[test]
    fn exact_ordinal_injective_near_precision_boundary() {
        // u64 keys near 2^53 where f64 loses precision
        let base: u64 = 1 << 53;
        assert_eq!(base as f64, (base + 1) as f64, "precondition: same f64");
        let o1 = base.to_exact_ordinal();
        let o2 = (base + 1).to_exact_ordinal();
        assert_ne!(o1, o2, "to_exact_ordinal must be injective");
        assert!(o1 < o2, "to_exact_ordinal must be monotonic");
    }

    #[test]
    fn exact_ordinal_injective_nanosecond_timestamps() {
        let base: u64 = 1_700_000_000_000_000_000;
        for i in 0..256u64 {
            let o1 = (base + i).to_exact_ordinal();
            let o2 = (base + i + 1).to_exact_ordinal();
            assert!(o1 < o2, "monotonicity violated at offset {i}");
        }
    }

    #[test]
    fn u128_exact_ordinal_preserves_order() {
        let vals: Vec<u128> = vec![0, 1, u128::MAX / 2, u128::MAX / 2 + 1, u128::MAX];
        for pair in vals.windows(2) {
            assert!(
                pair[0].to_exact_ordinal() < pair[1].to_exact_ordinal(),
                "u128 order violated: {} vs {}",
                pair[0],
                pair[1]
            );
        }
    }
}
