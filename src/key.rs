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
///   `a.to_model_input() < b.to_model_input()`.
/// - The returned `f64` must be finite (not NaN or infinity).
/// - The mapping should preserve relative distances where possible, so that
///   linear models can fit the key distribution effectively.
pub trait Key: Copy + Ord + Send + Sync + std::fmt::Debug + 'static {
    /// Convert this key to a `f64` value for model prediction.
    ///
    /// The conversion must be monotonic and return a finite value.
    fn to_model_input(self) -> f64;
}

macro_rules! impl_key_unsigned {
    ($($t:ty),*) => {
        $(
            impl Key for $t {
                #[inline]
                fn to_model_input(self) -> f64 {
                    self as f64
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
            }
        )*
    };
}

impl_key_unsigned!(u8, u16, u32, u64);
impl_key_signed!(i8, i16, i32, i64);

// u128/i128 lose precision beyond 2^53 but the monotonicity is preserved
// for values that fit. We provide the impl with that caveat.
impl Key for u128 {
    #[inline]
    fn to_model_input(self) -> f64 {
        self as f64
    }
}

impl Key for i128 {
    #[inline]
    fn to_model_input(self) -> f64 {
        self as f64
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
}
