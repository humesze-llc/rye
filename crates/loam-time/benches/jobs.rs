//! What one [`JobPool`] stage costs, and the per-stage work a partition has to
//! carry before splitting it wins. `cargo bench -p loam-time`.
//!
//! 256 units throughout, roughly a demo's body count. `unit_ns` is the serial
//! cost of one unit, swept by a per-unit spin so the same kernel spans a
//! trivial stage and an expensive one; `serial_us` is the whole stage at one
//! worker and `wN_us` the same stage at N.
//!
//! Measured on a 13th Gen Intel Core i9-13980HX (24 cores), Windows 11 Pro
//! 10.0.26200, rustc 1.95.0, `cargo bench` (opt-level 3, no debug assertions).
//! Each cell is the median of nine batches of 200 stages. Run-to-run spread is
//! within 1.4x on the barrier row and within 1.15x everywhere else.
//!
//! ```text
//! unit_ns serial_us   w2_us  w2x   w4_us  w4x   w8_us  w8x
//!       0       0.0   103.6 0.00x   180.9 0.00x   393.7 0.00x
//!       2       0.4    80.1 0.00x   161.0 0.00x   296.5 0.00x
//!      38       9.7    98.4 0.10x   183.5 0.05x   395.7 0.02x
//!     708     181.3   195.1 0.93x   259.7 0.70x   397.8 0.46x
//!    6262    1603.2  1012.4 1.58x   708.8 2.26x   667.9 2.40x
//! ```
//!
//! The top row is the barrier alone, since the kernel does nothing: about
//! 100us at two workers and 400us at eight, near enough 45us per extra
//! partition, which is what a Windows thread creation costs. It does not fall
//! as the kernel gets cheaper, because it is not the kernel.
//!
//! Splitting does not break even until the stage costs roughly 1.6ms serially.
//! At 181us serial, eight workers is 2.2x SLOWER than not splitting, so
//! "raise the worker count" is a regression at every stage size this workspace
//! currently runs.
//!
//! Wall-clock, so unlike the sim itself these numbers are not reproducible bit
//! for bit; the ratios are what the table is for.

use std::hint::black_box;
use std::time::Instant;

use loam_time::jobs::JobPool;

const UNITS: usize = 256;
const WORKER_COUNTS: [usize; 3] = [2, 4, 8];
/// Spin iterations per unit, chosen to walk the stage from below the barrier
/// cost to well above it.
const SPINS: [u32; 5] = [0, 8, 64, 512, 4096];
const REPS: u32 = 200;
/// Odd, so the reported median is a batch that was actually observed.
const BATCHES: usize = 9;

/// Per-unit work with a serial dependency chain, so the compiler cannot
/// vectorise the spin away and the cost scales with the iteration count.
fn unit_kernel(seed: f32, spin: u32) -> f32 {
    let mut x = seed;
    for _ in 0..spin {
        x = x * 1.000_000_1 + 1e-7;
    }
    x
}

fn run(pool: &JobPool, units: &mut [f32], partials: &mut Vec<()>, spin: u32) {
    pool.run_stage(units, partials, |_, slice| {
        for slot in slice.iter_mut() {
            *slot = unit_kernel(*slot, spin);
        }
    });
}

/// Median batch mean. Thread creation is a syscall whose tail is fat and
/// machine-load dependent, so a single batch is not reportable.
fn median_nanos(mut body: impl FnMut()) -> f64 {
    let mut batches = [0.0_f64; BATCHES];
    for batch in &mut batches {
        let start = Instant::now();
        for _ in 0..REPS {
            body();
        }
        *batch = start.elapsed().as_nanos() as f64 / f64::from(REPS);
    }
    batches.sort_unstable_by(f64::total_cmp);
    batches[BATCHES / 2]
}

fn main() {
    println!("unit_ns serial_us   w2_us  w2x   w4_us  w4x   w8_us  w8x");

    let serial = JobPool::new(1);
    let mut partials: Vec<()> = Vec::new();

    for spin in SPINS {
        let mut units = vec![1.0_f32; UNITS];
        // One untimed pass so the timed loop measures a warm cache.
        run(&serial, &mut units, &mut partials, spin);
        black_box(&units);

        let serial_ns = median_nanos(|| {
            run(&serial, &mut units, &mut partials, spin);
            black_box(&units);
        });

        let mut row = String::new();
        for workers in WORKER_COUNTS {
            let pool = JobPool::new(workers);
            let parallel_ns = median_nanos(|| {
                run(&pool, &mut units, &mut partials, spin);
                black_box(&units);
            });
            row.push_str(&format!(
                " {:7.1} {:4.2}x",
                parallel_ns / 1000.0,
                serial_ns / parallel_ns
            ));
        }

        println!(
            "{:7.0} {:9.1}{row}",
            serial_ns / UNITS as f64,
            serial_ns / 1000.0
        );
    }
}
