# fft-dihedral

[![Crates.io](https://img.shields.io/crates/v/fft-dihedral.svg)](https://crates.io/crates/fft-dihedral)
[![Docs.rs](https://docs.rs/fft-dihedral/badge.svg)](https://docs.rs/fft-dihedral)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust 2024](https://img.shields.io/badge/rust-2024-orange.svg)](https://www.rust-lang.org/)

Fast Fourier transforms for the dihedral group over NTT-friendly coefficient
rings.

This crate computes the Fourier transform of functions on

```text
D_{2n} = <r, s | r^n = s^2 = e, srs^{-1} = r^{-1}>.
```

The fast path reduces the nonabelian dihedral transform to four cyclic
number-theoretic transforms (NTTs).

## Quick Start

```rust
use fft_dihedral::{
    DEFAULT_MODULUS, dihedral_fft, flatten_transform, root_of_unity,
};

let n = 16;
let omega = root_of_unity(n, DEFAULT_MODULUS)?;
let rotations: Vec<u64> = (0..n).map(|k| k as u64).collect();
let reflections: Vec<u64> = (0..n).map(|k| (2 * k) as u64).collect();

let transform = dihedral_fft(&rotations, &reflections, DEFAULT_MODULUS, omega)?;
assert_eq!(flatten_transform(&transform).len(), 2 * n);
# Ok::<(), fft_dihedral::Error>(())
```

## Conventions

A function `f: D_{2n} -> R` is represented by two length-`n` arrays:

```text
rotations[k]   = f(r^k)
reflections[k] = f(s r^k)
```

For a primitive `n`th root of unity `omega`, the two-dimensional irreducible
representations satisfy

```text
rho_j(r^k)   = [[omega^(j k), 0], [0, omega^(-j k)]]
rho_j(s r^k) = [[0, omega^(-j k)], [omega^(j k), 0]]
```

With the normalized finite-group DFT convention

```text
fhat(rho) = (1 / |D_{2n}|) sum_g f(g) rho(g),
```

each two-dimensional Fourier coefficient is assembled from four cyclic
transforms:

```text
fhat(rho_j) = 1/(2n) * [
  [DFT_omega(rotations)[j],    DFT_omega^{-1}(reflections)[j]],
  [DFT_omega(reflections)[j],  DFT_omega^{-1}(rotations)[j]]
]
```

The one-dimensional representations are computed by direct character sums.

## Supported Coefficient Rings

The fast path uses the [`ntt`](https://crates.io/crates/ntt) crate for radix-2
number-theoretic transforms.

| Coefficients | Status | Notes |
| --- | --- | --- |
| Prime field `GF(p)` | Supported | Requires a primitive `n`th root of unity. |
| Integer quotient ring `Z/mZ` | Supported when `ntt` supports the root | Includes some prime-power/composite moduli. |
| Extension field `GF(p^e)` | Not yet | `Z/p^eZ` is not `GF(p^e)` when `e > 1`. |

The default prime field is `GF(2013265921)`, where
`2013265921 = 15 * 2^27 + 1`.

## Constraints

For the fast transform:

- `n` must be a power of two.
- `omega` must have exact order `n`.
- `gcd(2n, modulus) = 1`, so the normalization by `1/(2n)` is defined.
- `modulus <= 3037000499` with the current `ntt` backend, because that backend
  uses `i64` multiplication internally.

Use `root_of_unity(n, modulus)` to ask the backend for a compatible root, or
pass your own root to `dihedral_fft`.

## Rust API

```rust
use fft_dihedral::{
    dihedral_dft_naive, dihedral_fft, flatten_transform, root_of_unity,
};

let n = 16;
let modulus = 97;
let omega = root_of_unity(n, modulus)?;
let rotations = vec![1; n];
let reflections = vec![2; n];

let fast = flatten_transform(&dihedral_fft(&rotations, &reflections, modulus, omega)?);
let slow = flatten_transform(&dihedral_dft_naive(&rotations, &reflections, modulus, omega)?);
assert_eq!(fast, slow);
# Ok::<(), fft_dihedral::Error>(())
```

## Verify

```bash
cargo test
cargo run --release -- verify --n 128
cargo run --release -- verify --n 16 --modulus 97
```

## Time It

```bash
cargo run --release -- bench --min-exp 4 --max-exp 20 --repetitions 5 --naive-limit 2048
cargo run --release -- bench --min-exp 4 --max-exp 8 --modulus 97 --repetitions 5
```

Example release timings on the default modulus:

```text
n       |D_{2n}|   FFT median   ns/(N log2 N)   naive median   speedup
2048    4096       266.75 us    5.43            127.71 ms      478.8x
8192    16384      1.22 ms      5.33            -              -
262144  524288     55.99 ms     5.62            -              -
```

Here `N = |D_{2n}| = 2n`. The nearly flat `ns/(N log2 N)` column is the
practical signature of the expected `O(N log N)` scaling.

## Limitations

- No inverse dihedral transform yet.
- No true extension-field `GF(p^e)` implementation yet.
- The current implementation stores residues as `u64` and delegates the NTT to
  the `ntt` crate.
