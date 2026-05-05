use std::env;
use std::time::{Duration, Instant};

use fft_dihedral::{
    DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT, deterministic_dihedral_function, dihedral_dft_naive,
    dihedral_fft, dihedral_inverse_fft, dihedral_multiply_fft, dihedral_multiply_naive,
    flatten_transform, primitive_nth_root, root_of_unity,
};

#[derive(Clone, Copy, Debug)]
struct BenchConfig {
    min_exp: u32,
    max_exp: u32,
    repetitions: usize,
    naive_limit: usize,
    seed: u64,
    modulus: u64,
    omega: Option<u64>,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            min_exp: 4,
            max_exp: 18,
            repetitions: 3,
            naive_limit: 1024,
            seed: 0,
            modulus: DEFAULT_MODULUS,
            omega: None,
        }
    }
}

fn parse_usize_flag(args: &[String], name: &str, default: usize) -> usize {
    args.windows(2)
        .find(|window| window[0] == name)
        .and_then(|window| window[1].parse().ok())
        .unwrap_or(default)
}

fn parse_u32_flag(args: &[String], name: &str, default: u32) -> u32 {
    args.windows(2)
        .find(|window| window[0] == name)
        .and_then(|window| window[1].parse().ok())
        .unwrap_or(default)
}

fn parse_u64_flag(args: &[String], name: &str, default: u64) -> u64 {
    args.windows(2)
        .find(|window| window[0] == name)
        .and_then(|window| window[1].parse().ok())
        .unwrap_or(default)
}

fn parse_optional_u64_flag(args: &[String], name: &str) -> Option<u64> {
    args.windows(2)
        .find(|window| window[0] == name)
        .and_then(|window| window[1].parse().ok())
}

fn omega_for(n: usize, modulus: u64, explicit_omega: Option<u64>) -> u64 {
    if let Some(omega) = explicit_omega {
        return omega;
    }

    if modulus == DEFAULT_MODULUS {
        return root_of_unity(n, modulus)
            .unwrap_or_else(|_| primitive_nth_root(n, modulus, DEFAULT_PRIMITIVE_ROOT).unwrap());
    }

    root_of_unity(n, modulus).unwrap_or_else(|error| {
        panic!(
            "could not compute a {n}th root of unity modulo {modulus}: {error:?}. \
             Try a prime with n | p - 1, or pass --omega explicitly."
        )
    })
}

fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs_f64();
    if seconds < 1e-3 {
        format!("{:.2} us", seconds * 1e6)
    } else if seconds < 1.0 {
        format!("{:.2} ms", seconds * 1e3)
    } else {
        format!("{seconds:.2} s")
    }
}

fn median_duration(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn time_repeated(repetitions: usize, mut callback: impl FnMut()) -> Duration {
    let repetitions = repetitions.max(1);
    let mut samples = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let start = Instant::now();
        callback();
        samples.push(start.elapsed());
    }
    median_duration(samples)
}

fn run_verify(args: &[String]) {
    let n = parse_usize_flag(args, "--n", 64);
    let seed = parse_u64_flag(args, "--seed", 0);
    let modulus = parse_u64_flag(args, "--modulus", DEFAULT_MODULUS);
    let explicit_omega = parse_optional_u64_flag(args, "--omega");
    let omega = omega_for(n, modulus, explicit_omega);
    let (rotations, reflections) = deterministic_dihedral_function(n, seed, modulus);
    let fast = flatten_transform(&dihedral_fft(&rotations, &reflections, modulus, omega).unwrap());
    let naive =
        flatten_transform(&dihedral_dft_naive(&rotations, &reflections, modulus, omega).unwrap());

    assert_eq!(fast, naive);
    let transform = dihedral_fft(&rotations, &reflections, modulus, omega).unwrap();
    assert_eq!(
        dihedral_inverse_fft(&transform, omega).unwrap(),
        (rotations.clone(), reflections.clone())
    );

    let (rhs_rotations, rhs_reflections) =
        deterministic_dihedral_function(n, seed.wrapping_add(1_000_000), modulus);
    assert_eq!(
        dihedral_multiply_fft(
            &rotations,
            &reflections,
            &rhs_rotations,
            &rhs_reflections,
            modulus,
            omega
        )
        .unwrap(),
        dihedral_multiply_naive(
            &rotations,
            &reflections,
            &rhs_rotations,
            &rhs_reflections,
            modulus
        )
        .unwrap()
    );
    println!(
        "OK: DFT, inverse FFT, and multiplication agree for D_{} with n={n} modulo {modulus}.",
        2 * n
    );
}

fn run_bench(args: &[String]) {
    let defaults = BenchConfig::default();
    let config = BenchConfig {
        min_exp: parse_u32_flag(args, "--min-exp", defaults.min_exp),
        max_exp: parse_u32_flag(args, "--max-exp", defaults.max_exp),
        repetitions: parse_usize_flag(args, "--repetitions", defaults.repetitions),
        naive_limit: parse_usize_flag(args, "--naive-limit", defaults.naive_limit),
        seed: parse_u64_flag(args, "--seed", defaults.seed),
        modulus: parse_u64_flag(args, "--modulus", defaults.modulus),
        omega: parse_optional_u64_flag(args, "--omega"),
    };

    println!(
        "{:>8} {:>10} {:>12} {:>20} {:>14} {:>10}",
        "n", "|D_{2n}|", "FFT median", "ns/(N log2 N)", "naive median", "speedup"
    );

    for exponent in config.min_exp..=config.max_exp {
        let n = 1usize << exponent;
        let group_order = 2 * n;
        let omega = omega_for(n, config.modulus, config.omega);
        let (rotations, reflections) =
            deterministic_dihedral_function(n, config.seed + n as u64, config.modulus);

        let fast_time = time_repeated(config.repetitions, || {
            let _ = dihedral_fft(&rotations, &reflections, config.modulus, omega).unwrap();
        });
        let fast_seconds = fast_time.as_secs_f64();
        let normalized = fast_seconds * 1e9 / (group_order as f64 * (group_order as f64).log2());

        let (naive_display, speedup_display) = if n <= config.naive_limit {
            let naive_time = time_repeated(config.repetitions.min(3), || {
                let _ =
                    dihedral_dft_naive(&rotations, &reflections, config.modulus, omega).unwrap();
            });
            (
                format_duration(naive_time),
                format!("{:.1}x", naive_time.as_secs_f64() / fast_seconds),
            )
        } else {
            ("-".to_string(), "-".to_string())
        };

        println!(
            "{n:8} {group_order:10} {:>12} {normalized:20.2} {:>14} {:>10}",
            format_duration(fast_time),
            naive_display,
            speedup_display
        );
    }
}

fn print_usage() {
    println!("Usage:");
    println!("  fft-dihedral verify --n 64 --modulus 2013265921 --seed 0");
    println!("  fft-dihedral verify --n 16 --modulus 97");
    println!(
        "  fft-dihedral bench --min-exp 4 --max-exp 18 --modulus 2013265921 --repetitions 3 --naive-limit 1024"
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("verify") => run_verify(&args[2..]),
        Some("bench") => run_bench(&args[2..]),
        _ => {
            print_usage();
            run_bench(&[]);
        }
    }
}
