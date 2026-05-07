//! Fast Fourier transforms for the dihedral group over NTT-friendly
//! coefficient rings.
//!
//! We use the presentation
//!
//! ```text
//! D_{2n} = <r, s | r^n = s^2 = e, srs^{-1} = r^{-1}>.
//! ```
//!
//! A function `f: D_{2n} -> R` is represented by two length-`n` slices:
//!
//! ```text
//! rotations[k]   = f(r^k)
//! reflections[k] = f(s r^k)
//! ```
//!
//! With the normalized finite-group DFT convention
//!
//! ```text
//! fhat(rho) = (1 / |D_{2n}|) sum_g f(g) rho(g),
//! ```
//!
//! each two-dimensional Fourier coefficient is assembled from four cyclic
//! transforms:
//!
//! ```text
//! fhat(rho_j) = 1/(2n) * [
//!   [DFT_omega(rotations)[j],    DFT_omega^{-1}(reflections)[j]],
//!   [DFT_omega(reflections)[j],  DFT_omega^{-1}(rotations)[j]]
//! ]
//! ```
//!
//! The fast path uses the [`ntt`](https://crates.io/crates/ntt) crate, so it
//! currently requires `n` to be a power of two and a suitable `n`th root of
//! unity in the coefficient ring.
//!
//! The inverse transform rebuilds the two cyclic spectra for the rotation and
//! reflection coefficients, then applies two inverse NTTs. Multiplication in
//! the group algebra is implemented by transforming both inputs, multiplying
//! each Fourier block, scaling by `|D_{2n}|` because the transform is
//! normalized, and applying the inverse transform.
//!
//! # Example
//!
//! ```
//! use fft_dihedral::{
//!     DEFAULT_MODULUS, dihedral_fft, dihedral_invert_fft, dihedral_inverse_fft,
//!     dihedral_multiply_fft, flatten_transform, root_of_unity,
//! };
//!
//! let n = 16;
//! let omega = root_of_unity(n, DEFAULT_MODULUS)?;
//! let rotations: Vec<u64> = (0..n).map(|k| k as u64).collect();
//! let reflections: Vec<u64> = (0..n).map(|k| (2 * k) as u64).collect();
//! let transform = dihedral_fft(&rotations, &reflections, DEFAULT_MODULUS, omega)?;
//!
//! assert_eq!(flatten_transform(&transform).len(), 2 * n);
//! assert_eq!(
//!     dihedral_inverse_fft(&transform, omega)?,
//!     (rotations.clone(), reflections.clone())
//! );
//!
//! let mut unit_rotations = vec![0; n];
//! unit_rotations[3] = 7;
//! let unit_reflections = vec![0; n];
//! let inverse = dihedral_invert_fft(&unit_rotations, &unit_reflections, DEFAULT_MODULUS, omega)?;
//! let product = dihedral_multiply_fft(
//!     &unit_rotations,
//!     &unit_reflections,
//!     &inverse.0,
//!     &inverse.1,
//!     DEFAULT_MODULUS,
//!     omega,
//! )?;
//! assert_eq!(product.0[0], 1);
//! # Ok::<(), fft_dihedral::Error>(())
//! ```
//!
//! # Coefficient Rings
//!
//! Prime fields `GF(p)` are supported whenever the required roots of unity
//! exist. The `ntt` backend also supports suitable integer quotient rings
//! `Z/mZ`, including some prime-power and composite moduli. These are rings:
//! `Z/p^eZ` is not the finite field `GF(p^e)` when `e > 1`.

/// Default NTT-friendly prime modulus.
///
/// `2013265921 = 15 * 2^27 + 1`, so this prime supports radix-2 transforms of
/// length up to `2^27`.
pub const DEFAULT_MODULUS: u64 = 2_013_265_921;

/// A primitive generator of `GF(2013265921)^*`.
pub const DEFAULT_PRIMITIVE_ROOT: u64 = 31;

/// The current `ntt` backend uses `i64` multiplication internally.
///
/// Keeping the modulus at or below `floor(sqrt(i64::MAX))` avoids overflow in
/// those products. The default NTT prime is below this bound.
pub const MAX_SAFE_NTT_MODULUS: u64 = 3_037_000_499;

/// Errors returned by checked transform constructors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// The coefficient modulus was zero.
    ModulusIsZero,
    /// A zero-length transform was requested.
    EmptyInput,
    /// The rotation and reflection arrays had different lengths.
    MismatchedInputLengths,
    /// Dihedral groups with `n < 3` are excluded from this implementation.
    RotationOrderTooSmall,
    /// The group order `2n` is not a unit modulo the coefficient modulus.
    GroupOrderNotInvertible,
    /// The supplied root does not satisfy `omega^n = 1`.
    RootIsNotNthRoot,
    /// For radix-2 NTTs, the supplied root must have exact order `n`.
    RootIsNotPrimitivePowerOfTwo,
    /// The fast NTT backend currently requires power-of-two transform length.
    NttLengthNotPowerOfTwo,
    /// `n` does not divide `p - 1` for the prime-field root helper.
    RotationOrderDoesNotDivideModulusMinusOne,
    /// The modulus is too large for the current `ntt` backend's `i64`
    /// multiplication strategy.
    ModulusTooLargeForNttBackend,
    /// The backend could not construct a suitable root of unity.
    RootUnavailable,
    /// Two Fourier transforms had different rotation orders.
    TransformLengthMismatch,
    /// Two Fourier transforms used different coefficient moduli.
    TransformModulusMismatch,
    /// The supplied Fourier data is missing coefficients, has duplicate
    /// coefficients, or has coefficients with impossible labels.
    InvalidFourierData,
    /// A group-algebra element is not invertible over the coefficient ring.
    NonInvertibleElement,
}

/// Labels for the one-dimensional irreducible representations.
///
/// For odd `n`, only [`OneDimensionalRep::Trivial`] and
/// [`OneDimensionalRep::ReflectionSign`] occur. For even `n`, the two extra
/// signs involving the parity of the rotation exponent are also present.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OneDimensionalRep {
    /// Sends both `r` and `s` to `1`.
    Trivial,
    /// Sends `r` to `1` and `s` to `-1`.
    ReflectionSign,
    /// Sends `r` to `-1` and `s` to `1`; only present for even `n`.
    RotationSign,
    /// Sends both `r` and `s` to `-1`; only present for even `n`.
    TotalSign,
}

/// A `2 x 2` Fourier coefficient matrix.
///
/// The entries are stored in row-major order:
///
/// ```text
/// [[a00, a01],
///  [a10, a11]]
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Matrix2 {
    /// Row 0, column 0.
    pub a00: u64,
    /// Row 0, column 1.
    pub a01: u64,
    /// Row 1, column 0.
    pub a10: u64,
    /// Row 1, column 1.
    pub a11: u64,
}

/// Fourier data for a function on `D_{2n}`.
///
/// The one-dimensional coefficients are scalar values. The two-dimensional
/// coefficients are stored as `(j, Matrix2)` pairs corresponding to the irrep
/// `rho_j`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DihedralDft {
    /// Rotation order. The group has order `2n`.
    pub n: usize,
    /// Coefficient modulus.
    pub modulus: u64,
    /// Scalar coefficients for the one-dimensional irreducible representations.
    pub one_dimensional: Vec<(OneDimensionalRep, u64)>,
    /// Matrix coefficients for the two-dimensional irreducible representations.
    pub two_dimensional: Vec<(usize, Matrix2)>,
}

/// Return whether `n` is a positive power of two.
pub fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Add two residues modulo `modulus`.
pub fn add_mod(a: u64, b: u64, modulus: u64) -> u64 {
    let sum = a + b;
    if sum >= modulus { sum - modulus } else { sum }
}

/// Subtract two residues modulo `modulus`.
pub fn sub_mod(a: u64, b: u64, modulus: u64) -> u64 {
    if a >= b { a - b } else { a + modulus - b }
}

/// Multiply two residues modulo `modulus`.
pub fn mul_mod(a: u64, b: u64, modulus: u64) -> u64 {
    ((a as u128 * b as u128) % modulus as u128) as u64
}

/// Modular exponentiation.
pub fn pow_mod(mut base: u64, mut exponent: u64, modulus: u64) -> u64 {
    let mut result = 1;
    base %= modulus;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = mul_mod(result, base, modulus);
        }
        base = mul_mod(base, base, modulus);
        exponent >>= 1;
    }
    result
}

/// Greatest common divisor.
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn extended_gcd(a: i128, b: i128) -> (i128, i128, i128) {
    if b == 0 {
        (a, 1, 0)
    } else {
        let (g, x, y) = extended_gcd(b, a % b);
        (g, y, x - (a / b) * y)
    }
}

/// Multiplicative inverse modulo `modulus`, returning `None` for non-units.
pub fn try_inv_mod(value: u64, modulus: u64) -> Option<u64> {
    if modulus == 0 {
        return None;
    }
    let value = value % modulus;
    let (g, x, _) = extended_gcd(value as i128, modulus as i128);
    if g != 1 {
        return None;
    }
    Some(x.rem_euclid(modulus as i128) as u64)
}

/// Multiplicative inverse modulo `modulus`.
///
/// # Panics
///
/// Panics if `value` is not a unit modulo `modulus`.
pub fn inv_mod(value: u64, modulus: u64) -> u64 {
    try_inv_mod(value, modulus)
        .unwrap_or_else(|| panic!("{value} is not invertible modulo {modulus}"))
}

fn validate_ntt_backend_modulus(modulus: u64) -> Result<(), Error> {
    if modulus == 0 {
        return Err(Error::ModulusIsZero);
    }
    if modulus > MAX_SAFE_NTT_MODULUS {
        return Err(Error::ModulusTooLargeForNttBackend);
    }
    Ok(())
}

/// Construct a primitive `n`th root in a prime field from a known primitive
/// generator.
///
/// This helper assumes `modulus` is prime and `primitive_root` generates
/// `GF(modulus)^*`.
pub fn primitive_nth_root(n: usize, modulus: u64, primitive_root: u64) -> Result<u64, Error> {
    if modulus == 0 {
        return Err(Error::ModulusIsZero);
    }
    if n == 0 {
        return Err(Error::EmptyInput);
    }
    let n_u64 = n as u64;
    if !(modulus - 1).is_multiple_of(n_u64) {
        return Err(Error::RotationOrderDoesNotDivideModulusMinusOne);
    }
    let omega = pow_mod(primitive_root, (modulus - 1) / n_u64, modulus);
    if pow_mod(omega, n_u64, modulus) != 1 {
        return Err(Error::RootIsNotNthRoot);
    }
    Ok(omega)
}

/// Compute an `n`th root of unity using the `ntt` crate.
///
/// This supports the moduli supported by `ntt::omega`, including suitable prime
/// powers and some composite moduli. This is arithmetic in `Z/mZ`; note that
/// `Z/p^eZ` is not the finite field `GF(p^e)` when `e > 1`.
pub fn root_of_unity(n: usize, modulus: u64) -> Result<u64, Error> {
    if n == 0 {
        return Err(Error::EmptyInput);
    }
    validate_ntt_backend_modulus(modulus)?;

    std::panic::catch_unwind(|| ::ntt::omega(modulus as i64, n))
        .map(|root| root as u64)
        .map_err(|_| Error::RootUnavailable)
}

fn normalize(values: &[u64], modulus: u64) -> Vec<u64> {
    values.iter().map(|value| value % modulus).collect()
}

fn validate_dihedral_input(
    rotations: &[u64],
    reflections: &[u64],
    modulus: u64,
    omega: u64,
) -> Result<usize, Error> {
    if modulus == 0 {
        return Err(Error::ModulusIsZero);
    }
    let n = rotations.len();
    if n == 0 {
        return Err(Error::EmptyInput);
    }
    if n != reflections.len() {
        return Err(Error::MismatchedInputLengths);
    }
    if n < 3 {
        return Err(Error::RotationOrderTooSmall);
    }
    if gcd(2 * n as u64, modulus) != 1 {
        return Err(Error::GroupOrderNotInvertible);
    }
    if pow_mod(omega, n as u64, modulus) != 1 {
        return Err(Error::RootIsNotNthRoot);
    }
    Ok(n)
}

fn validate_power_of_two_root(n: usize, root: u64, modulus: u64) -> Result<(), Error> {
    validate_ntt_backend_modulus(modulus)?;
    if !is_power_of_two(n) {
        return Err(Error::NttLengthNotPowerOfTwo);
    }
    if pow_mod(root, n as u64, modulus) != 1 {
        return Err(Error::RootIsNotNthRoot);
    }
    if n > 1 && pow_mod(root, (n / 2) as u64, modulus) == 1 {
        return Err(Error::RootIsNotPrimitivePowerOfTwo);
    }
    Ok(())
}

fn bit_reverse_index(mut index: usize, bits: u32) -> usize {
    let mut reversed = 0;
    for _ in 0..bits {
        reversed = (reversed << 1) | (index & 1);
        index >>= 1;
    }
    reversed
}

fn from_bit_reversed(values: &[i64]) -> Vec<u64> {
    let n = values.len();
    let bits = n.trailing_zeros();
    let mut natural = vec![0; n];
    for (bit_reversed_index, value) in values.iter().enumerate() {
        let natural_index = bit_reverse_index(bit_reversed_index, bits);
        natural[natural_index] = *value as u64;
    }
    natural
}

fn to_bit_reversed(values: &[u64]) -> Vec<i64> {
    let n = values.len();
    let bits = n.trailing_zeros();
    let mut bit_reversed = vec![0; n];
    for (natural_index, value) in values.iter().enumerate() {
        let bit_reversed_index = bit_reverse_index(natural_index, bits);
        bit_reversed[bit_reversed_index] = *value as i64;
    }
    bit_reversed
}

/// Unnormalized radix-2 number-theoretic transform.
///
/// The `j`-th output is `sum_k values[k] * root^(j*k)`.
///
/// The underlying `ntt` crate returns the forward transform in bit-reversed
/// order; this wrapper converts the output back to natural frequency order.
pub fn ntt(values: &[u64], root: u64, modulus: u64) -> Result<Vec<u64>, Error> {
    let n = values.len();
    validate_power_of_two_root(n, root, modulus)?;
    let input: Vec<_> = normalize(values, modulus)
        .into_iter()
        .map(|value| value as i64)
        .collect();
    let bit_reversed = ::ntt::ntt(&input, root as i64, n, modulus as i64);
    Ok(from_bit_reversed(&bit_reversed))
}

/// Inverse radix-2 number-theoretic transform.
///
/// This is the inverse of [`ntt()`] for the same `root` and `modulus`.
pub fn inverse_ntt(values: &[u64], root: u64, modulus: u64) -> Result<Vec<u64>, Error> {
    let n = values.len();
    validate_power_of_two_root(n, root, modulus)?;
    let input = to_bit_reversed(&normalize(values, modulus));
    Ok(::ntt::intt(&input, root as i64, n, modulus as i64)
        .into_iter()
        .map(|value| value as u64)
        .collect())
}

/// Direct unnormalized cyclic DFT with the same convention as [`ntt()`].
///
/// This is quadratic time and intended mainly for tests and small reference
/// computations.
pub fn cyclic_dft_naive(values: &[u64], root: u64, modulus: u64) -> Vec<u64> {
    let n = values.len();
    let values = normalize(values, modulus);
    let mut result = Vec::with_capacity(n);
    for j in 0..n {
        let step = pow_mod(root, j as u64, modulus);
        let mut power = 1;
        let mut total = 0;
        for value in &values {
            total = add_mod(total, mul_mod(*value, power, modulus), modulus);
            power = mul_mod(power, step, modulus);
        }
        result.push(total);
    }
    result
}

fn one_dimensional_coefficients(
    rotations: &[u64],
    reflections: &[u64],
    modulus: u64,
) -> Vec<(OneDimensionalRep, u64)> {
    let n = rotations.len();
    let inv_group_order = inv_mod(2 * n as u64, modulus);
    let rotation_sum = rotations
        .iter()
        .fold(0, |acc, value| add_mod(acc, value % modulus, modulus));
    let reflection_sum = reflections
        .iter()
        .fold(0, |acc, value| add_mod(acc, value % modulus, modulus));

    let mut coefficients = vec![
        (
            OneDimensionalRep::Trivial,
            mul_mod(
                add_mod(rotation_sum, reflection_sum, modulus),
                inv_group_order,
                modulus,
            ),
        ),
        (
            OneDimensionalRep::ReflectionSign,
            mul_mod(
                sub_mod(rotation_sum, reflection_sum, modulus),
                inv_group_order,
                modulus,
            ),
        ),
    ];

    if n.is_multiple_of(2) {
        let alternating_rotation_sum =
            rotations.iter().enumerate().fold(0, |acc, (index, value)| {
                if index % 2 == 0 {
                    add_mod(acc, value % modulus, modulus)
                } else {
                    sub_mod(acc, value % modulus, modulus)
                }
            });
        let alternating_reflection_sum =
            reflections
                .iter()
                .enumerate()
                .fold(0, |acc, (index, value)| {
                    if index % 2 == 0 {
                        add_mod(acc, value % modulus, modulus)
                    } else {
                        sub_mod(acc, value % modulus, modulus)
                    }
                });
        coefficients.extend([
            (
                OneDimensionalRep::RotationSign,
                mul_mod(
                    add_mod(
                        alternating_rotation_sum,
                        alternating_reflection_sum,
                        modulus,
                    ),
                    inv_group_order,
                    modulus,
                ),
            ),
            (
                OneDimensionalRep::TotalSign,
                mul_mod(
                    sub_mod(
                        alternating_rotation_sum,
                        alternating_reflection_sum,
                        modulus,
                    ),
                    inv_group_order,
                    modulus,
                ),
            ),
        ]);
    }

    coefficients
}

fn two_dimensional_range(n: usize) -> std::ops::RangeInclusive<usize> {
    if n.is_multiple_of(2) {
        1..=(n / 2 - 1)
    } else {
        1..=((n - 1) / 2)
    }
}

const ODD_ONE_DIMENSIONAL_REPS: [OneDimensionalRep; 2] = [
    OneDimensionalRep::Trivial,
    OneDimensionalRep::ReflectionSign,
];

const EVEN_ONE_DIMENSIONAL_REPS: [OneDimensionalRep; 4] = [
    OneDimensionalRep::Trivial,
    OneDimensionalRep::ReflectionSign,
    OneDimensionalRep::RotationSign,
    OneDimensionalRep::TotalSign,
];

fn expected_one_dimensional_reps(n: usize) -> &'static [OneDimensionalRep] {
    if n.is_multiple_of(2) {
        &EVEN_ONE_DIMENSIONAL_REPS
    } else {
        &ODD_ONE_DIMENSIONAL_REPS
    }
}

fn one_dimensional_value(transform: &DihedralDft, rep: OneDimensionalRep) -> Result<u64, Error> {
    let mut found = None;
    for (label, value) in &transform.one_dimensional {
        if *label == rep {
            if found.is_some() {
                return Err(Error::InvalidFourierData);
            }
            found = Some(*value % transform.modulus);
        }
    }
    found.ok_or(Error::InvalidFourierData)
}

fn two_dimensional_matrices_by_index(
    transform: &DihedralDft,
) -> Result<Vec<Option<Matrix2>>, Error> {
    let n = transform.n;
    let mut matrices = vec![None; n];
    let mut count = 0;
    let valid_range = two_dimensional_range(n);

    for (j, matrix) in &transform.two_dimensional {
        if !valid_range.contains(j) {
            return Err(Error::InvalidFourierData);
        }
        if matrices[*j].replace(*matrix).is_some() {
            return Err(Error::InvalidFourierData);
        }
        count += 1;
    }

    if count != two_dimensional_range(n).count() {
        return Err(Error::InvalidFourierData);
    }

    Ok(matrices)
}

fn validate_fourier_shape(transform: &DihedralDft) -> Result<(), Error> {
    if transform.modulus == 0 {
        return Err(Error::ModulusIsZero);
    }
    if transform.n == 0 {
        return Err(Error::EmptyInput);
    }
    if transform.n < 3 {
        return Err(Error::RotationOrderTooSmall);
    }
    if gcd(2 * transform.n as u64, transform.modulus) != 1 {
        return Err(Error::GroupOrderNotInvertible);
    }
    if transform.one_dimensional.len() != expected_one_dimensional_reps(transform.n).len() {
        return Err(Error::InvalidFourierData);
    }
    for rep in expected_one_dimensional_reps(transform.n) {
        one_dimensional_value(transform, *rep)?;
    }
    two_dimensional_matrices_by_index(transform)?;
    Ok(())
}

fn assemble_dihedral_transform(
    rotations: &[u64],
    reflections: &[u64],
    positive_rotations: &[u64],
    negative_rotations: &[u64],
    positive_reflections: &[u64],
    negative_reflections: &[u64],
    modulus: u64,
) -> DihedralDft {
    let n = rotations.len();
    let inv_group_order = inv_mod(2 * n as u64, modulus);
    let mut two_dimensional = Vec::new();

    for j in two_dimensional_range(n) {
        two_dimensional.push((
            j,
            Matrix2 {
                a00: mul_mod(positive_rotations[j], inv_group_order, modulus),
                a01: mul_mod(negative_reflections[j], inv_group_order, modulus),
                a10: mul_mod(positive_reflections[j], inv_group_order, modulus),
                a11: mul_mod(negative_rotations[j], inv_group_order, modulus),
            },
        ));
    }

    DihedralDft {
        n,
        modulus,
        one_dimensional: one_dimensional_coefficients(rotations, reflections, modulus),
        two_dimensional,
    }
}

/// Compute the normalized dihedral DFT directly from the representation
/// formulas.
///
/// This is the quadratic-time reference implementation used to check
/// [`dihedral_fft`]. The inputs are:
///
/// - `rotations[k] = f(r^k)`
/// - `reflections[k] = f(s r^k)`
///
/// The result uses the normalization `1 / |D_{2n}| = 1 / (2n)`.
pub fn dihedral_dft_naive(
    rotations: &[u64],
    reflections: &[u64],
    modulus: u64,
    omega: u64,
) -> Result<DihedralDft, Error> {
    validate_dihedral_input(rotations, reflections, modulus, omega)?;

    let rotations = normalize(rotations, modulus);
    let reflections = normalize(reflections, modulus);
    let inv_omega = inv_mod(omega, modulus);

    let positive_rotations = cyclic_dft_naive(&rotations, omega, modulus);
    let negative_rotations = cyclic_dft_naive(&rotations, inv_omega, modulus);
    let positive_reflections = cyclic_dft_naive(&reflections, omega, modulus);
    let negative_reflections = cyclic_dft_naive(&reflections, inv_omega, modulus);

    Ok(assemble_dihedral_transform(
        &rotations,
        &reflections,
        &positive_rotations,
        &negative_rotations,
        &positive_reflections,
        &negative_reflections,
        modulus,
    ))
}

/// Compute the normalized fast dihedral DFT.
///
/// The input convention is:
///
/// - `rotations[k] = f(r^k)`
/// - `reflections[k] = f(s r^k)`
///
/// This function computes the one-dimensional character sums directly, then
/// computes the two-dimensional coefficients from four length-`n` NTTs:
///
/// ```text
/// DFT_omega(rotations)
/// DFT_omega^{-1}(rotations)
/// DFT_omega(reflections)
/// DFT_omega^{-1}(reflections)
/// ```
///
/// The result uses the normalization `1 / |D_{2n}| = 1 / (2n)`.
pub fn dihedral_fft(
    rotations: &[u64],
    reflections: &[u64],
    modulus: u64,
    omega: u64,
) -> Result<DihedralDft, Error> {
    let n = validate_dihedral_input(rotations, reflections, modulus, omega)?;
    validate_power_of_two_root(n, omega, modulus)?;

    let rotations = normalize(rotations, modulus);
    let reflections = normalize(reflections, modulus);
    let inv_omega = inv_mod(omega, modulus);

    let positive_rotations = ntt(&rotations, omega, modulus)?;
    let negative_rotations = ntt(&rotations, inv_omega, modulus)?;
    let positive_reflections = ntt(&reflections, omega, modulus)?;
    let negative_reflections = ntt(&reflections, inv_omega, modulus)?;

    Ok(assemble_dihedral_transform(
        &rotations,
        &reflections,
        &positive_rotations,
        &negative_rotations,
        &positive_reflections,
        &negative_reflections,
        modulus,
    ))
}

/// Invert a normalized fast dihedral DFT.
///
/// The input must have been computed with the same `omega` and modulus stored
/// in `transform`. The returned pair uses the original coefficient convention:
///
/// - `rotations[k] = f(r^k)`
/// - `reflections[k] = f(s r^k)`
///
/// Internally this reconstructs the two full cyclic Fourier vectors
/// `DFT_omega(rotations)` and `DFT_omega(reflections)` from the scalar and
/// matrix coefficients, then applies two inverse NTTs.
pub fn dihedral_inverse_fft(
    transform: &DihedralDft,
    omega: u64,
) -> Result<(Vec<u64>, Vec<u64>), Error> {
    validate_fourier_shape(transform)?;
    let n = transform.n;
    let modulus = transform.modulus;
    validate_power_of_two_root(n, omega, modulus)?;

    let n_scale = n as u64 % modulus;
    let group_order = 2 * n as u64 % modulus;
    let mut rotation_frequencies = vec![0; n];
    let mut reflection_frequencies = vec![0; n];

    let trivial = one_dimensional_value(transform, OneDimensionalRep::Trivial)?;
    let reflection_sign = one_dimensional_value(transform, OneDimensionalRep::ReflectionSign)?;
    rotation_frequencies[0] = mul_mod(n_scale, add_mod(trivial, reflection_sign, modulus), modulus);
    reflection_frequencies[0] =
        mul_mod(n_scale, sub_mod(trivial, reflection_sign, modulus), modulus);

    if n.is_multiple_of(2) {
        let rotation_sign = one_dimensional_value(transform, OneDimensionalRep::RotationSign)?;
        let total_sign = one_dimensional_value(transform, OneDimensionalRep::TotalSign)?;
        rotation_frequencies[n / 2] = mul_mod(
            n_scale,
            add_mod(rotation_sign, total_sign, modulus),
            modulus,
        );
        reflection_frequencies[n / 2] = mul_mod(
            n_scale,
            sub_mod(rotation_sign, total_sign, modulus),
            modulus,
        );
    }

    let matrices = two_dimensional_matrices_by_index(transform)?;
    for j in two_dimensional_range(n) {
        let matrix = matrices[j].ok_or(Error::InvalidFourierData)?;
        let mirror = n - j;

        rotation_frequencies[j] = mul_mod(matrix.a00, group_order, modulus);
        reflection_frequencies[j] = mul_mod(matrix.a10, group_order, modulus);
        rotation_frequencies[mirror] = mul_mod(matrix.a11, group_order, modulus);
        reflection_frequencies[mirror] = mul_mod(matrix.a01, group_order, modulus);
    }

    Ok((
        inverse_ntt(&rotation_frequencies, omega, modulus)?,
        inverse_ntt(&reflection_frequencies, omega, modulus)?,
    ))
}

fn matrix_inverse_scaled(matrix: Matrix2, scale: u64, modulus: u64) -> Result<Matrix2, Error> {
    let determinant = sub_mod(
        mul_mod(matrix.a00, matrix.a11, modulus),
        mul_mod(matrix.a01, matrix.a10, modulus),
        modulus,
    );
    let inverse_determinant =
        try_inv_mod(determinant, modulus).ok_or(Error::NonInvertibleElement)?;
    let scale = mul_mod(scale, inverse_determinant, modulus);

    Ok(Matrix2 {
        a00: mul_mod(scale, matrix.a11, modulus),
        a01: mul_mod(scale, sub_mod(0, matrix.a01, modulus), modulus),
        a10: mul_mod(scale, sub_mod(0, matrix.a10, modulus), modulus),
        a11: mul_mod(scale, matrix.a00, modulus),
    })
}

/// Invert a normalized dihedral Fourier transform block-by-block.
///
/// The crate uses the normalized convention
/// `fhat(rho) = |D_{2n}|^{-1} sum_g f(g) rho(g)`. Since multiplication in
/// Fourier space is `(xy)^hat = |D_{2n}| xhat yhat`, the transform of a
/// group-algebra inverse has blocks `|D_{2n}|^{-2} xhat^{-1}`.
pub fn invert_fourier_transform(transform: &DihedralDft) -> Result<DihedralDft, Error> {
    validate_fourier_shape(transform)?;
    let n = transform.n;
    let modulus = transform.modulus;
    let group_order = (2 * n as u64) % modulus;
    let inverse_group_order =
        try_inv_mod(group_order, modulus).ok_or(Error::GroupOrderNotInvertible)?;
    let inverse_scale = mul_mod(inverse_group_order, inverse_group_order, modulus);

    let mut one_dimensional = Vec::with_capacity(expected_one_dimensional_reps(n).len());
    for rep in expected_one_dimensional_reps(n) {
        let value = one_dimensional_value(transform, *rep)?;
        let inverse = try_inv_mod(value, modulus).ok_or(Error::NonInvertibleElement)?;
        one_dimensional.push((*rep, mul_mod(inverse_scale, inverse, modulus)));
    }

    let matrices = two_dimensional_matrices_by_index(transform)?;
    let mut two_dimensional = Vec::with_capacity(two_dimensional_range(n).count());
    for j in two_dimensional_range(n) {
        let matrix = matrices[j].ok_or(Error::InvalidFourierData)?;
        two_dimensional.push((j, matrix_inverse_scaled(matrix, inverse_scale, modulus)?));
    }

    Ok(DihedralDft {
        n,
        modulus,
        one_dimensional,
        two_dimensional,
    })
}

fn matrix_multiply_scaled(lhs: Matrix2, rhs: Matrix2, scale: u64, modulus: u64) -> Matrix2 {
    let entry = |a: u64, b: u64, c: u64, d: u64| {
        mul_mod(
            scale,
            add_mod(mul_mod(a, b, modulus), mul_mod(c, d, modulus), modulus),
            modulus,
        )
    };

    Matrix2 {
        a00: entry(lhs.a00, rhs.a00, lhs.a01, rhs.a10),
        a01: entry(lhs.a00, rhs.a01, lhs.a01, rhs.a11),
        a10: entry(lhs.a10, rhs.a00, lhs.a11, rhs.a10),
        a11: entry(lhs.a10, rhs.a01, lhs.a11, rhs.a11),
    }
}

/// Multiply two normalized dihedral Fourier transforms.
///
/// With the crate's normalized convention,
/// `fhat(rho) = (1 / |D_{2n}|) sum_g f(g) rho(g)`, convolution in the group
/// algebra becomes
///
/// ```text
/// (f * h)^hat(rho) = |D_{2n}| fhat(rho) hhat(rho).
/// ```
///
/// This function performs exactly that scalar or matrix multiplication on each
/// irreducible block.
pub fn multiply_fourier_transforms(
    lhs: &DihedralDft,
    rhs: &DihedralDft,
) -> Result<DihedralDft, Error> {
    validate_fourier_shape(lhs)?;
    validate_fourier_shape(rhs)?;
    if lhs.n != rhs.n {
        return Err(Error::TransformLengthMismatch);
    }
    if lhs.modulus != rhs.modulus {
        return Err(Error::TransformModulusMismatch);
    }

    let n = lhs.n;
    let modulus = lhs.modulus;
    let group_order = 2 * n as u64 % modulus;
    let mut one_dimensional = Vec::with_capacity(expected_one_dimensional_reps(n).len());

    for rep in expected_one_dimensional_reps(n) {
        one_dimensional.push((
            *rep,
            mul_mod(
                group_order,
                mul_mod(
                    one_dimensional_value(lhs, *rep)?,
                    one_dimensional_value(rhs, *rep)?,
                    modulus,
                ),
                modulus,
            ),
        ));
    }

    let lhs_matrices = two_dimensional_matrices_by_index(lhs)?;
    let rhs_matrices = two_dimensional_matrices_by_index(rhs)?;
    let mut two_dimensional = Vec::with_capacity(two_dimensional_range(n).count());

    for j in two_dimensional_range(n) {
        let lhs_matrix = lhs_matrices[j].ok_or(Error::InvalidFourierData)?;
        let rhs_matrix = rhs_matrices[j].ok_or(Error::InvalidFourierData)?;
        two_dimensional.push((
            j,
            matrix_multiply_scaled(lhs_matrix, rhs_matrix, group_order, modulus),
        ));
    }

    Ok(DihedralDft {
        n,
        modulus,
        one_dimensional,
        two_dimensional,
    })
}

fn validate_group_algebra_inputs(
    lhs_rotations: &[u64],
    lhs_reflections: &[u64],
    rhs_rotations: &[u64],
    rhs_reflections: &[u64],
    modulus: u64,
) -> Result<usize, Error> {
    if modulus == 0 {
        return Err(Error::ModulusIsZero);
    }
    let n = lhs_rotations.len();
    if n == 0 {
        return Err(Error::EmptyInput);
    }
    if n != lhs_reflections.len() || n != rhs_rotations.len() || n != rhs_reflections.len() {
        return Err(Error::MismatchedInputLengths);
    }
    if n < 3 {
        return Err(Error::RotationOrderTooSmall);
    }
    Ok(n)
}

/// Direct quadratic multiplication in the group algebra of `D_{2n}`.
///
/// This is mostly useful as a reference implementation for tests. The basis is
/// the same as the transform input:
///
/// ```text
/// rotations[k]   = coefficient of r^k
/// reflections[k] = coefficient of s r^k
/// ```
pub fn dihedral_multiply_naive(
    lhs_rotations: &[u64],
    lhs_reflections: &[u64],
    rhs_rotations: &[u64],
    rhs_reflections: &[u64],
    modulus: u64,
) -> Result<(Vec<u64>, Vec<u64>), Error> {
    let n = validate_group_algebra_inputs(
        lhs_rotations,
        lhs_reflections,
        rhs_rotations,
        rhs_reflections,
        modulus,
    )?;
    let lhs_rotations = normalize(lhs_rotations, modulus);
    let lhs_reflections = normalize(lhs_reflections, modulus);
    let rhs_rotations = normalize(rhs_rotations, modulus);
    let rhs_reflections = normalize(rhs_reflections, modulus);
    let mut rotations = vec![0; n];
    let mut reflections = vec![0; n];

    for i in 0..n {
        for j in 0..n {
            let sum_index = (i + j) % n;
            let difference_index = (j + n - i) % n;

            rotations[sum_index] = add_mod(
                rotations[sum_index],
                mul_mod(lhs_rotations[i], rhs_rotations[j], modulus),
                modulus,
            );
            rotations[difference_index] = add_mod(
                rotations[difference_index],
                mul_mod(lhs_reflections[i], rhs_reflections[j], modulus),
                modulus,
            );
            reflections[difference_index] = add_mod(
                reflections[difference_index],
                mul_mod(lhs_rotations[i], rhs_reflections[j], modulus),
                modulus,
            );
            reflections[sum_index] = add_mod(
                reflections[sum_index],
                mul_mod(lhs_reflections[i], rhs_rotations[j], modulus),
                modulus,
            );
        }
    }

    Ok((rotations, reflections))
}

/// Multiply two elements of the group algebra of `D_{2n}` using the FFT.
///
/// The inputs are coefficient arrays in the basis `r^k` and `s r^k`. The
/// function computes two dihedral FFTs, multiplies the Fourier blocks, and
/// applies [`dihedral_inverse_fft`] to return the product coefficients.
pub fn dihedral_multiply_fft(
    lhs_rotations: &[u64],
    lhs_reflections: &[u64],
    rhs_rotations: &[u64],
    rhs_reflections: &[u64],
    modulus: u64,
    omega: u64,
) -> Result<(Vec<u64>, Vec<u64>), Error> {
    validate_group_algebra_inputs(
        lhs_rotations,
        lhs_reflections,
        rhs_rotations,
        rhs_reflections,
        modulus,
    )?;
    let lhs = dihedral_fft(lhs_rotations, lhs_reflections, modulus, omega)?;
    let rhs = dihedral_fft(rhs_rotations, rhs_reflections, modulus, omega)?;
    let product = multiply_fourier_transforms(&lhs, &rhs)?;
    dihedral_inverse_fft(&product, omega)
}

/// Invert an element of the group algebra of `D_{2n}` using the FFT.
///
/// The inputs are coefficient arrays in the basis `r^k` and `s r^k`. The
/// function transforms the element, inverts every Fourier scalar or `2 x 2`
/// block, and applies [`dihedral_inverse_fft`] to return the inverse
/// coefficients.
pub fn dihedral_invert_fft(
    rotations: &[u64],
    reflections: &[u64],
    modulus: u64,
    omega: u64,
) -> Result<(Vec<u64>, Vec<u64>), Error> {
    validate_group_algebra_inputs(rotations, reflections, rotations, reflections, modulus)?;
    let transform = dihedral_fft(rotations, reflections, modulus, omega)?;
    let inverse = invert_fourier_transform(&transform)?;
    dihedral_inverse_fft(&inverse, omega)
}

/// Flatten Fourier data to a vector of length `2n`.
///
/// The order is: all one-dimensional coefficients first, followed by the
/// two-dimensional matrices in increasing `j`, each in row-major order.
pub fn flatten_transform(transform: &DihedralDft) -> Vec<u64> {
    let mut flattened = Vec::with_capacity(2 * transform.n);
    for (_, value) in &transform.one_dimensional {
        flattened.push(*value);
    }
    for (_, matrix) in &transform.two_dimensional {
        flattened.extend([matrix.a00, matrix.a01, matrix.a10, matrix.a11]);
    }
    flattened
}

/// Generate deterministic pseudo-random input data for examples and benchmarks.
pub fn deterministic_dihedral_function(n: usize, seed: u64, modulus: u64) -> (Vec<u64>, Vec<u64>) {
    let mut rng = SplitMix64::new(seed);
    let rotations = (0..n).map(|_| rng.next_u64() % modulus).collect();
    let reflections = (0..n).map(|_| rng.next_u64() % modulus).collect();
    (rotations, reflections)
}

/// Small deterministic pseudo-random generator used for tests and examples.
///
/// This is not cryptographically secure.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a new generator from `seed`.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Return the next pseudo-random `u64`.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(n: usize) -> u64 {
        primitive_nth_root(n, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT).unwrap()
    }

    #[test]
    fn crate_root_of_unity_is_valid_for_default_prime() {
        for n in [4, 8, 16, 32, 64, 128] {
            let omega = root_of_unity(n, DEFAULT_MODULUS).unwrap();
            assert_eq!(pow_mod(omega, n as u64, DEFAULT_MODULUS), 1);
            assert_ne!(pow_mod(omega, (n / 2) as u64, DEFAULT_MODULUS), 1);
        }
    }

    #[test]
    fn ntt_backend_handles_prime_power_modulus() {
        let modulus = 17 * 17;
        let n = 8;
        let omega = root_of_unity(n, modulus).unwrap();
        let rotations: Vec<_> = (0..n).map(|k| (3 * k as u64 + 1) % modulus).collect();
        let reflections: Vec<_> = (0..n).map(|k| (5 * k as u64 + 2) % modulus).collect();

        assert_eq!(
            ntt(&rotations, omega, modulus).unwrap(),
            cyclic_dft_naive(&rotations, omega, modulus)
        );
        assert_eq!(
            flatten_transform(&dihedral_fft(&rotations, &reflections, modulus, omega).unwrap()),
            flatten_transform(
                &dihedral_dft_naive(&rotations, &reflections, modulus, omega).unwrap()
            )
        );
    }

    #[test]
    fn dihedral_fft_accepts_other_compatible_primes() {
        for (modulus, n) in [(17, 8), (97, 16), (257, 64)] {
            let omega = root_of_unity(n, modulus).unwrap();
            let (rotations, reflections) =
                deterministic_dihedral_function(n, modulus + n as u64, modulus);

            assert_eq!(
                ntt(&rotations, omega, modulus).unwrap(),
                cyclic_dft_naive(&rotations, omega, modulus)
            );
            assert_eq!(
                flatten_transform(&dihedral_fft(&rotations, &reflections, modulus, omega).unwrap()),
                flatten_transform(
                    &dihedral_dft_naive(&rotations, &reflections, modulus, omega).unwrap()
                )
            );
        }
    }

    #[test]
    fn ntt_matches_naive_cyclic_dft() {
        for n in [4, 8, 16, 32, 64, 128] {
            let omega = root(n);
            let values: Vec<_> = (0..n)
                .map(|k| (17 * k as u64 * k as u64 + 5 * k as u64 + 11) % DEFAULT_MODULUS)
                .collect();

            assert_eq!(
                ntt(&values, omega, DEFAULT_MODULUS).unwrap(),
                cyclic_dft_naive(&values, omega, DEFAULT_MODULUS)
            );
        }
    }

    #[test]
    fn inverse_ntt_round_trips() {
        for n in [4, 8, 16, 32, 64, 128] {
            let omega = root(n);
            let values: Vec<_> = (0..n)
                .map(|k| (31 * k as u64 + 7) % DEFAULT_MODULUS)
                .collect();
            let transformed = ntt(&values, omega, DEFAULT_MODULUS).unwrap();

            assert_eq!(
                inverse_ntt(&transformed, omega, DEFAULT_MODULUS).unwrap(),
                values
            );
        }
    }

    #[test]
    fn dihedral_inverse_fft_round_trips() {
        for n in [4, 8, 16, 32, 64, 128] {
            let omega = root(n);
            for seed in 0..5 {
                let (rotations, reflections) =
                    deterministic_dihedral_function(n, 30_000 * n as u64 + seed, DEFAULT_MODULUS);
                let transform =
                    dihedral_fft(&rotations, &reflections, DEFAULT_MODULUS, omega).unwrap();
                let (inverse_rotations, inverse_reflections) =
                    dihedral_inverse_fft(&transform, omega).unwrap();

                assert_eq!(inverse_rotations, rotations);
                assert_eq!(inverse_reflections, reflections);
            }
        }
    }

    #[test]
    fn dihedral_fft_matches_naive_dft_for_random_inputs() {
        for n in [4, 8, 16, 32, 64, 128] {
            let omega = root(n);
            for seed in 0..5 {
                let (rotations, reflections) =
                    deterministic_dihedral_function(n, 10_000 * n as u64 + seed, DEFAULT_MODULUS);
                let fast = flatten_transform(
                    &dihedral_fft(&rotations, &reflections, DEFAULT_MODULUS, omega).unwrap(),
                );
                let naive = flatten_transform(
                    &dihedral_dft_naive(&rotations, &reflections, DEFAULT_MODULUS, omega).unwrap(),
                );

                assert_eq!(fast, naive);
            }
        }
    }

    #[test]
    fn dihedral_fft_matches_naive_dft_for_structured_inputs() {
        for n in [4, 8, 16, 32] {
            let omega = root(n);
            let mut rotations = vec![0; n];
            let mut reflections = vec![0; n];
            rotations[0] = 1;
            reflections[1] = 1;

            assert_eq!(
                flatten_transform(
                    &dihedral_fft(&rotations, &reflections, DEFAULT_MODULUS, omega).unwrap()
                ),
                flatten_transform(
                    &dihedral_dft_naive(&rotations, &reflections, DEFAULT_MODULUS, omega).unwrap()
                )
            );
        }
    }

    #[test]
    fn naive_multiplication_uses_dihedral_group_law() {
        let n = 8;
        let modulus = 97;
        let zero = vec![0; n];

        let mut r = vec![0; n];
        r[1] = 1;
        let mut s = vec![0; n];
        s[0] = 1;

        let (_, r_times_s_reflections) =
            dihedral_multiply_naive(&r, &zero, &zero, &s, modulus).unwrap();
        let (_, s_times_r_reflections) =
            dihedral_multiply_naive(&zero, &s, &r, &zero, modulus).unwrap();

        let mut expected_r_times_s = vec![0; n];
        expected_r_times_s[n - 1] = 1;
        let mut expected_s_times_r = vec![0; n];
        expected_s_times_r[1] = 1;

        assert_eq!(r_times_s_reflections, expected_r_times_s);
        assert_eq!(s_times_r_reflections, expected_s_times_r);
    }

    #[test]
    fn multiplying_fourier_transforms_matches_transform_of_product() {
        for n in [4, 8, 16, 32, 64] {
            let omega = root(n);
            let (lhs_rotations, lhs_reflections) =
                deterministic_dihedral_function(n, 40_000 * n as u64, DEFAULT_MODULUS);
            let (rhs_rotations, rhs_reflections) =
                deterministic_dihedral_function(n, 50_000 * n as u64, DEFAULT_MODULUS);
            let (product_rotations, product_reflections) = dihedral_multiply_naive(
                &lhs_rotations,
                &lhs_reflections,
                &rhs_rotations,
                &rhs_reflections,
                DEFAULT_MODULUS,
            )
            .unwrap();

            let lhs_transform =
                dihedral_fft(&lhs_rotations, &lhs_reflections, DEFAULT_MODULUS, omega).unwrap();
            let rhs_transform =
                dihedral_fft(&rhs_rotations, &rhs_reflections, DEFAULT_MODULUS, omega).unwrap();
            let product_transform = dihedral_fft(
                &product_rotations,
                &product_reflections,
                DEFAULT_MODULUS,
                omega,
            )
            .unwrap();
            let multiplied_transform =
                multiply_fourier_transforms(&lhs_transform, &rhs_transform).unwrap();

            assert_eq!(
                flatten_transform(&multiplied_transform),
                flatten_transform(&product_transform)
            );
        }
    }

    #[test]
    fn dihedral_multiply_fft_matches_naive_group_algebra_product() {
        for n in [4, 8, 16, 32, 64] {
            let omega = root(n);
            for seed in 0..3 {
                let (lhs_rotations, lhs_reflections) =
                    deterministic_dihedral_function(n, 60_000 * n as u64 + seed, DEFAULT_MODULUS);
                let (rhs_rotations, rhs_reflections) =
                    deterministic_dihedral_function(n, 70_000 * n as u64 + seed, DEFAULT_MODULUS);

                assert_eq!(
                    dihedral_multiply_fft(
                        &lhs_rotations,
                        &lhs_reflections,
                        &rhs_rotations,
                        &rhs_reflections,
                        DEFAULT_MODULUS,
                        omega,
                    )
                    .unwrap(),
                    dihedral_multiply_naive(
                        &lhs_rotations,
                        &lhs_reflections,
                        &rhs_rotations,
                        &rhs_reflections,
                        DEFAULT_MODULUS,
                    )
                    .unwrap()
                );
            }
        }
    }

    #[test]
    fn dihedral_multiply_fft_respects_identity() {
        for n in [4, 8, 16, 32] {
            let omega = root(n);
            let (rotations, reflections) =
                deterministic_dihedral_function(n, 80_000 * n as u64, DEFAULT_MODULUS);
            let mut identity_rotations = vec![0; n];
            identity_rotations[0] = 1;
            let identity_reflections = vec![0; n];

            assert_eq!(
                dihedral_multiply_fft(
                    &rotations,
                    &reflections,
                    &identity_rotations,
                    &identity_reflections,
                    DEFAULT_MODULUS,
                    omega,
                )
                .unwrap(),
                (rotations.clone(), reflections.clone())
            );
            assert_eq!(
                dihedral_multiply_fft(
                    &identity_rotations,
                    &identity_reflections,
                    &rotations,
                    &reflections,
                    DEFAULT_MODULUS,
                    omega,
                )
                .unwrap(),
                (rotations, reflections)
            );
        }
    }

    #[test]
    fn two_dimensional_coefficients_are_assembled_from_ntts() {
        for n in [8, 16, 32, 64] {
            let omega = root(n);
            let inv_omega = inv_mod(omega, DEFAULT_MODULUS);
            let inv_group_order = inv_mod(2 * n as u64, DEFAULT_MODULUS);
            let (rotations, reflections) =
                deterministic_dihedral_function(n, 20_000 * n as u64, DEFAULT_MODULUS);

            let positive_rotations = ntt(&rotations, omega, DEFAULT_MODULUS).unwrap();
            let negative_rotations = ntt(&rotations, inv_omega, DEFAULT_MODULUS).unwrap();
            let positive_reflections = ntt(&reflections, omega, DEFAULT_MODULUS).unwrap();
            let negative_reflections = ntt(&reflections, inv_omega, DEFAULT_MODULUS).unwrap();
            let transform = dihedral_fft(&rotations, &reflections, DEFAULT_MODULUS, omega).unwrap();

            for (j, matrix) in transform.two_dimensional {
                assert_eq!(
                    matrix,
                    Matrix2 {
                        a00: mul_mod(positive_rotations[j], inv_group_order, DEFAULT_MODULUS),
                        a01: mul_mod(negative_reflections[j], inv_group_order, DEFAULT_MODULUS),
                        a10: mul_mod(positive_reflections[j], inv_group_order, DEFAULT_MODULUS),
                        a11: mul_mod(negative_rotations[j], inv_group_order, DEFAULT_MODULUS),
                    }
                );
            }
        }
    }

    #[test]
    fn dihedral_invert_fft_inverts_rotation_unit() {
        let n = 16;
        let modulus = DEFAULT_MODULUS;
        let omega = root(n);
        let mut rotations = vec![0; n];
        let reflections = vec![0; n];
        rotations[3] = 7;

        let (inverse_rotations, inverse_reflections) =
            dihedral_invert_fft(&rotations, &reflections, modulus, omega).unwrap();
        let mut identity_rotations = vec![0; n];
        identity_rotations[0] = 1;
        let identity_reflections = vec![0; n];

        assert_eq!(
            dihedral_multiply_fft(
                &rotations,
                &reflections,
                &inverse_rotations,
                &inverse_reflections,
                modulus,
                omega,
            )
            .unwrap(),
            (identity_rotations.clone(), identity_reflections.clone())
        );
        assert_eq!(
            dihedral_multiply_fft(
                &inverse_rotations,
                &inverse_reflections,
                &rotations,
                &reflections,
                modulus,
                omega,
            )
            .unwrap(),
            (identity_rotations, identity_reflections)
        );
    }

    #[test]
    fn dihedral_invert_fft_inverts_reflection_unit() {
        let n = 16;
        let modulus = DEFAULT_MODULUS;
        let omega = root(n);
        let rotations = vec![0; n];
        let mut reflections = vec![0; n];
        reflections[5] = 11;

        let (inverse_rotations, inverse_reflections) =
            dihedral_invert_fft(&rotations, &reflections, modulus, omega).unwrap();
        let mut identity_rotations = vec![0; n];
        identity_rotations[0] = 1;
        let identity_reflections = vec![0; n];

        assert_eq!(
            dihedral_multiply_fft(
                &rotations,
                &reflections,
                &inverse_rotations,
                &inverse_reflections,
                modulus,
                omega,
            )
            .unwrap(),
            (identity_rotations.clone(), identity_reflections.clone())
        );
        assert_eq!(
            dihedral_multiply_fft(
                &inverse_rotations,
                &inverse_reflections,
                &rotations,
                &reflections,
                modulus,
                omega,
            )
            .unwrap(),
            (identity_rotations, identity_reflections)
        );
    }

    #[test]
    fn dihedral_invert_fft_inverts_dense_units() {
        for n in [8, 16, 32] {
            let omega = root(n);
            let mut checked = 0;
            for seed in 0..5 {
                let (rotations, reflections) =
                    deterministic_dihedral_function(n, 90_000 * n as u64 + seed, DEFAULT_MODULUS);
                let Ok((inverse_rotations, inverse_reflections)) =
                    dihedral_invert_fft(&rotations, &reflections, DEFAULT_MODULUS, omega)
                else {
                    continue;
                };
                let mut identity_rotations = vec![0; n];
                identity_rotations[0] = 1;
                let identity_reflections = vec![0; n];

                assert_eq!(
                    dihedral_multiply_fft(
                        &rotations,
                        &reflections,
                        &inverse_rotations,
                        &inverse_reflections,
                        DEFAULT_MODULUS,
                        omega,
                    )
                    .unwrap(),
                    (identity_rotations.clone(), identity_reflections.clone())
                );
                assert_eq!(
                    dihedral_multiply_fft(
                        &inverse_rotations,
                        &inverse_reflections,
                        &rotations,
                        &reflections,
                        DEFAULT_MODULUS,
                        omega,
                    )
                    .unwrap(),
                    (identity_rotations, identity_reflections)
                );
                checked += 1;
            }
            assert!(checked > 0, "expected at least one dense unit for n={n}");
        }
    }

    #[test]
    fn dihedral_invert_fft_rejects_zero_element() {
        let n = 16;
        let omega = root(n);
        let zero = vec![0; n];

        assert_eq!(
            dihedral_invert_fft(&zero, &zero, DEFAULT_MODULUS, omega),
            Err(Error::NonInvertibleElement)
        );
    }

    #[test]
    fn flattened_transform_has_group_order_length() {
        for n in [4, 8, 16, 32, 64] {
            let omega = root(n);
            let (rotations, reflections) = deterministic_dihedral_function(n, 0, DEFAULT_MODULUS);
            let transform = dihedral_fft(&rotations, &reflections, DEFAULT_MODULUS, omega).unwrap();

            assert_eq!(flatten_transform(&transform).len(), 2 * n);
        }
    }
}
