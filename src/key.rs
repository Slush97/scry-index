//! The [`Key`] trait for types usable as learned index keys.
//!
//! Keys must be ordered, clonable, and convertible to `f64` for linear model
//! prediction. This conversion must be monotonic: if `a < b` then
//! `a.to_model_input() < b.to_model_input()`.

/// A key type usable in a learned index.
///
/// # Contract
///
/// - `to_model_input` must be a **monotonic** function: if `a < b` then
///   `a.to_model_input() <= b.to_model_input()`. Note: the mapping is
///   non-strict: distinct keys may produce the same `f64` due to precision
///   loss (e.g., `u64` keys above 2^53). The index handles this correctly
///   via the `to_exact_ordinal` fallback.
/// - The returned `f64` must be finite (not NaN or infinity).
/// - The mapping should preserve relative distances where possible, so that
///   linear models can fit the key distribution effectively.
/// - `to_exact_ordinal` must be a **strictly monotonic, injective** function:
///   if `a < b` then `a.to_exact_ordinal() < b.to_exact_ordinal()`.
pub trait Key: Clone + Ord + Send + Sync + std::fmt::Debug + 'static {
    /// Convert this key to a `f64` value for model prediction.
    ///
    /// The conversion must be monotonic and return a finite value. It may
    /// be non-injective for large integer keys (precision loss above 2^53).
    fn to_model_input(&self) -> f64;

    /// Convert this key to a lossless `i128` for exact comparison.
    ///
    /// This must be strictly monotonic and injective: if `a < b` then
    /// `a.to_exact_ordinal() < b.to_exact_ordinal()`. Used as a fallback
    /// for conflict resolution when `to_model_input` cannot distinguish keys.
    fn to_exact_ordinal(&self) -> i128;
}

macro_rules! impl_key_unsigned {
    ($($t:ty),*) => {
        $(
            impl Key for $t {
                #[inline]
                fn to_model_input(&self) -> f64 {
                    *self as f64
                }

                #[inline]
                fn to_exact_ordinal(&self) -> i128 {
                    *self as i128
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
                fn to_model_input(&self) -> f64 {
                    *self as f64
                }

                #[inline]
                fn to_exact_ordinal(&self) -> i128 {
                    *self as i128
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
    fn to_model_input(&self) -> f64 {
        *self as f64
    }

    #[inline]
    #[allow(clippy::cast_possible_wrap)]
    fn to_exact_ordinal(&self) -> i128 {
        // Order-preserving bijection: flip the sign bit so that
        // 0u128 -> i128::MIN and u128::MAX -> i128::MAX.
        (*self as i128) ^ i128::MIN
    }
}

impl Key for i128 {
    #[inline]
    fn to_model_input(&self) -> f64 {
        *self as f64
    }

    #[inline]
    fn to_exact_ordinal(&self) -> i128 {
        *self
    }
}

/// Interpret the first 8 bytes of a byte slice as a big-endian `u64` -> `f64`.
///
/// Shorter slices are zero-padded on the right. Provides a monotonic mapping
/// with respect to lexicographic byte order.
///
/// This helper is useful for implementing [`Key::to_model_input`] on custom
/// types that have a byte representation.
#[inline]
pub fn bytes_to_model_input(bytes: &[u8]) -> f64 {
    let mut buf = [0u8; 8];
    let len = bytes.len().min(8);
    buf[..len].copy_from_slice(&bytes[..len]);
    u64::from_be_bytes(buf) as f64
}

/// Interpret the first 16 bytes of a byte slice as a big-endian `u128`,
/// then apply the sign-flip bijection to produce an order-preserving `i128`.
///
/// Shorter slices are zero-padded on the right. For slices longer than 16
/// bytes, only the first 16 bytes are used. Keys that share a 16-byte
/// prefix will produce the same ordinal (non-injective). The index handles
/// these collisions via `Ord`-based splits in the node.
///
/// This helper is useful for implementing [`Key::to_exact_ordinal`] on custom
/// types that have a byte representation.
#[inline]
#[allow(clippy::cast_possible_wrap)]
pub fn bytes_to_exact_ordinal(bytes: &[u8]) -> i128 {
    let mut buf = [0u8; 16];
    let len = bytes.len().min(16);
    buf[..len].copy_from_slice(&bytes[..len]);
    // Order-preserving bijection: XOR with i128::MIN
    // maps 0x00..00 → i128::MIN and 0xFF..FF → i128::MAX.
    (u128::from_be_bytes(buf) as i128) ^ i128::MIN
}

/// Key impl for fixed-size byte arrays.
///
/// Enables byte-array keys like `[u8; 16]` (UUIDs), `[u8; 32]` (hashes), etc.
///
/// - `to_model_input`: interprets the first 8 bytes as a big-endian `u64` and
///   casts to `f64`. Shorter arrays are zero-padded on the right. This provides
///   a monotonic mapping with respect to lexicographic order.
/// - `to_exact_ordinal`: interprets the first 16 bytes as a big-endian `u128`
///   (with the same sign-flip trick as `u128`). This is injective for `N <= 16`.
///   For `N > 16`, it is a best-effort prefix: distinct keys that share their
///   first 16 bytes will collide. The index resolves these via `Ord`-based
///   splits in the node.
impl<const N: usize> Key for [u8; N] {
    #[inline]
    fn to_model_input(&self) -> f64 {
        bytes_to_model_input(self)
    }

    #[inline]
    fn to_exact_ordinal(&self) -> i128 {
        bytes_to_exact_ordinal(self)
    }
}

/// Key impl for `String`.
///
/// Enables heap-allocated string keys. Uses the UTF-8 byte representation
/// for model input and ordinal computation.
///
/// - `to_model_input`: interprets the first 8 bytes as a big-endian `u64` →
///   `f64`. Strings shorter than 8 bytes are zero-padded. Monotonic with
///   respect to lexicographic (byte) order.
/// - `to_exact_ordinal`: interprets the first 16 bytes as a big-endian `u128`
///   with sign-flip. Non-injective for strings sharing a 16-byte prefix;
///   the index uses `Ord`-based node splits to handle these collisions.
impl Key for String {
    #[inline]
    fn to_model_input(&self) -> f64 {
        bytes_to_model_input(self.as_bytes())
    }

    #[inline]
    fn to_exact_ordinal(&self) -> i128 {
        bytes_to_exact_ordinal(self.as_bytes())
    }
}

/// Key impl for `Vec<u8>`.
///
/// Enables heap-allocated byte-vector keys. Semantics are identical to the
/// fixed-size `[u8; N]` implementation but for dynamically sized vectors.
///
/// - `to_model_input`: first 8 bytes as big-endian `u64` → `f64`.
/// - `to_exact_ordinal`: first 16 bytes as `i128` with sign-flip.
///   Non-injective for vectors sharing a 16-byte prefix.
impl Key for Vec<u8> {
    #[inline]
    fn to_model_input(&self) -> f64 {
        bytes_to_model_input(self)
    }

    #[inline]
    fn to_exact_ordinal(&self) -> i128 {
        bytes_to_exact_ordinal(self)
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
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(base as f64, (base + 1) as f64, "precondition: same f64");
        }
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

    // -----------------------------------------------------------------------
    // [u8; N] byte array keys
    // -----------------------------------------------------------------------

    #[test]
    fn byte_array_key_is_send_sync() {
        fn assert_key<T: Key>() {}
        assert_key::<[u8; 4]>();
        assert_key::<[u8; 8]>();
        assert_key::<[u8; 16]>();
        assert_key::<[u8; 32]>();
    }

    #[test]
    fn byte4_model_input_monotonic() {
        let keys: Vec<[u8; 4]> = vec![[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]];
        for pair in keys.windows(2) {
            assert!(
                pair[0].to_model_input() <= pair[1].to_model_input(),
                "monotonicity violated for [u8; 4]: {:?} > {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn byte8_model_input_monotonic() {
        let keys: Vec<[u8; 8]> = vec![
            [0; 8],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [255; 8],
        ];
        for pair in keys.windows(2) {
            assert!(
                pair[0].to_model_input() <= pair[1].to_model_input(),
                "monotonicity violated for [u8; 8]: {:?} > {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn byte16_exact_ordinal_injective() {
        let keys: Vec<[u8; 16]> = vec![
            [0; 16],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [255; 16],
        ];
        for pair in keys.windows(2) {
            assert!(
                pair[0].to_exact_ordinal() < pair[1].to_exact_ordinal(),
                "exact_ordinal not injective for [u8; 16]: {:?} >= {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn byte32_model_input_finite() {
        let keys: Vec<[u8; 32]> = vec![[0; 32], [128; 32], [255; 32]];
        for k in &keys {
            let v = k.to_model_input();
            assert!(v.is_finite(), "{k:?} produced non-finite model input {v}");
        }
    }

    #[test]
    fn byte4_exact_ordinal_monotonic() {
        // For N < 16, zero-padded; still strictly monotonic for lex order
        let keys: Vec<[u8; 4]> = vec![[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]];
        for pair in keys.windows(2) {
            assert!(
                pair[0].to_exact_ordinal() < pair[1].to_exact_ordinal(),
                "exact_ordinal not monotonic for [u8; 4]: {:?} >= {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn byte32_exact_ordinal_prefix_only() {
        // For N > 16, only the first 16 bytes are used. Keys differing
        // after byte 16 will have the same ordinal.
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        a[20] = 1;
        b[20] = 2;
        // Same first 16 bytes → same ordinal (best-effort limitation)
        assert_eq!(a.to_exact_ordinal(), b.to_exact_ordinal());
        // But Ord comparison still distinguishes them
        assert!(a < b);
    }

    // -----------------------------------------------------------------------
    // String keys
    // -----------------------------------------------------------------------

    #[test]
    fn string_key_is_send_sync() {
        fn assert_key<T: Key>() {}
        assert_key::<String>();
    }

    #[test]
    fn string_model_input_monotonic() {
        let keys: Vec<String> = vec![
            String::new(),
            "a".into(),
            "b".into(),
            "hello".into(),
            "world".into(),
            "zzzzzzzz".into(),
        ];
        for pair in keys.windows(2) {
            assert!(
                pair[0].to_model_input() <= pair[1].to_model_input(),
                "monotonicity violated for String: {:?} > {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn string_model_input_finite() {
        let keys = vec![String::new(), "hello".into(), "\u{ffff}".into()];
        for k in &keys {
            let v = k.to_model_input();
            assert!(v.is_finite(), "{k:?} produced non-finite model input {v}");
        }
    }

    #[test]
    fn string_exact_ordinal_monotonic() {
        let keys: Vec<String> = vec![
            String::new(),
            "a".into(),
            "aa".into(),
            "ab".into(),
            "b".into(),
        ];
        for pair in keys.windows(2) {
            assert!(
                pair[0].to_exact_ordinal() <= pair[1].to_exact_ordinal(),
                "exact_ordinal not monotonic for String: {:?} >= {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn string_shared_prefix_same_ordinal() {
        // Strings sharing a 16+ byte prefix have the same ordinal.
        let a = "1234567890abcdefXXX".to_string();
        let b = "1234567890abcdefYYY".to_string();
        assert_eq!(a.to_exact_ordinal(), b.to_exact_ordinal());
        assert!(a < b);
    }

    #[test]
    fn string_different_in_first_16_bytes() {
        let a = "1234567890abcdeX".to_string();
        let b = "1234567890abcdeY".to_string();
        assert!(a.to_exact_ordinal() < b.to_exact_ordinal());
    }

    #[test]
    fn string_matches_byte_array() {
        // String and [u8; N] should produce the same model input/ordinal
        // for identical byte content.
        let s = "hello".to_string();
        let arr: [u8; 5] = *b"hello";
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(s.to_model_input(), arr.to_model_input());
        }
        assert_eq!(s.to_exact_ordinal(), arr.to_exact_ordinal());
    }

    // -----------------------------------------------------------------------
    // Vec<u8> keys
    // -----------------------------------------------------------------------

    #[test]
    fn vec_u8_key_is_send_sync() {
        fn assert_key<T: Key>() {}
        assert_key::<Vec<u8>>();
    }

    #[test]
    fn vec_u8_model_input_monotonic() {
        let keys: Vec<Vec<u8>> = vec![vec![], vec![0], vec![0, 1], vec![1], vec![1, 0], vec![255]];
        for pair in keys.windows(2) {
            assert!(
                pair[0].to_model_input() <= pair[1].to_model_input(),
                "monotonicity violated for Vec<u8>: {:?} > {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn vec_u8_exact_ordinal_monotonic() {
        let keys: Vec<Vec<u8>> = vec![
            vec![],
            vec![0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            vec![1],
        ];
        for pair in keys.windows(2) {
            assert!(
                pair[0].to_exact_ordinal() <= pair[1].to_exact_ordinal(),
                "exact_ordinal not monotonic for Vec<u8>: {:?} >= {:?}",
                pair[0],
                pair[1]
            );
        }
    }
}
