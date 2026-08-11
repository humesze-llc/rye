//! Fixed-partition worker pool: the executor a runner owns and lends to
//! each tick.
//!
//! Lives here rather than in the app crate because the sim crates must be
//! able to take it without depending on wgpu, winit and egui, and because
//! [`crate::frame_trace`] is the other half of the same seam: a cross-worker
//! trace merge has to happen at this pool's join barrier.

use std::num::NonZeroUsize;

/// Units per partition: `ceil(units / workers)`, floored at 1 so an empty
/// buffer still names a legal chunk width.
///
/// The single definition of the split. Contiguous ascending chunks of this
/// width tile the buffer exactly once and produce at most `workers` of them,
/// so which units a partition owns is a pure function of `(units, workers)`
/// and of nothing else: not of arrival order, not of how fast a worker ran.
/// Work stealing is excluded by that sentence, deliberately. Under stealing
/// the owner of a unit stops being a function of the schedule, and the
/// moment any stage carries a per-worker partial its reduction order stops
/// being fixed.
///
/// Panics if `workers` is zero.
pub fn partition_len(units: usize, workers: usize) -> usize {
    units.div_ceil(workers).max(1)
}

/// Worker budget plus the fixed partition policy, borrowed by a tick for the
/// duration of one stage.
///
/// # Concurrency is inside a stage, never across stages
///
/// [`JobPool::run_stage`] joins every partition before it returns, so two
/// stages never overlap and two app-level systems can therefore never run
/// concurrently with each other. That is the right trade while the workspace
/// has exactly one sim system whose phases are strictly sequentially
/// dependent: pipelining phases that feed each other buys nothing, and it
/// would make a tick's result depend on which overlap the pool happened to
/// choose. It becomes the wrong trade the day a second app-level system
/// appears that is independent of the first (audio, or particles with no
/// contact coupling), because real cross-stage parallelism would then exist
/// and this type could not express it. A system that only reads another's
/// output is an edge, not a peer, and does not invalidate the trade.
///
/// # What is bit-identical across worker counts, and what is not
///
/// A stage whose per-unit writes are disjoint is bit-identical between one
/// worker and many: the value written to a unit is a function of the unit
/// index, and the partition only decides which thread evaluates it.
///
/// A partial that reduces across the units of a chunk is not. Chunk
/// boundaries move with the worker count and f32 addition is not
/// associative, so such a stage is bit-reproducible at a fixed worker count
/// (partials come back in ascending partition index, never completion order)
/// and must not be read as reproducible across worker counts. A stage that
/// needs a reduction invariant under the worker count reduces per unit in
/// canonical order instead, and stays serial.
///
/// # Cost, and why more workers is currently a regression
///
/// Partition 0 runs on the calling thread and the rest are spawned per call
/// through `std::thread::scope`, so a single-partition stage spawns nothing
/// and a multi-partition stage pays one thread creation per extra
/// partition. `benches/jobs.rs` measures that at about 45us per extra
/// partition, so a stage has to cost on the order of 1.6ms serially before
/// splitting it breaks even, and an eight-phase tick would spend 3.2ms of a
/// 16.7ms budget on barriers alone at eight workers. No stage in this
/// workspace is near that, so raising the budget today buys nothing and a
/// stage should be split only after a measurement puts it in that regime.
///
/// Parking the workers instead of spawning them is what removes the per-stage
/// cost, and it requires lending a stage's borrows to threads that outlive
/// them, which needs a lifetime transmute this crate does not do. It is a
/// replacement for this method's body, not for the seam: nothing above
/// changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JobPool {
    workers: NonZeroUsize,
}

impl JobPool {
    /// Build a pool for `threads` workers. `0` is read as 1.
    ///
    /// Permanently 1 worker on wasm32. That is the platform, not a policy:
    /// the browser gives the worker one thread, and `frame_trace`'s
    /// thread-local state is sound only because of it.
    pub fn new(threads: usize) -> Self {
        let workers = if cfg!(target_arch = "wasm32") {
            1
        } else {
            threads
        };
        Self {
            workers: NonZeroUsize::new(workers).unwrap_or(NonZeroUsize::MIN),
        }
    }

    /// Worker budget this pool partitions against.
    pub fn threads(&self) -> usize {
        self.workers.get()
    }

    /// Run one stage: split `units` by [`partition_len`], evaluate `kernel`
    /// on every partition, and join before returning.
    ///
    /// `kernel` receives a partition index `k` and the units it owns, which
    /// are `[k * partition_len(units.len(), self.threads()) ..]` capped at
    /// the buffer end, so a kernel needing a unit's global index derives it
    /// from the same function the pool split by.
    ///
    /// `partials` is cleared and refilled with one kernel result per
    /// partition in ascending partition index. Taking the buffer by
    /// reference rather than returning a fresh `Vec` keeps a per-tick stage
    /// off the allocator; a stage with nothing to reduce instantiates `R`
    /// at `()`, whose `Vec` never allocates.
    ///
    /// Panics from a kernel are resumed on the calling thread after the
    /// barrier, so a partition panicking cannot leave the caller believing
    /// the stage ran.
    pub fn run_stage<T, R, F>(&self, units: &mut [T], partials: &mut Vec<R>, kernel: F)
    where
        T: Send,
        R: Send,
        F: Fn(usize, &mut [T]) -> R + Sync,
    {
        partials.clear();
        let chunk = partition_len(units.len(), self.workers.get());

        // Spawning is compiled out on wasm32 rather than guarded at runtime:
        // `threads()` is 1 there, so the branch is unreachable, and a
        // platform whose thread spawn panics should not carry the call.
        #[cfg(not(target_arch = "wasm32"))]
        if self.workers.get() > 1 && units.len() > chunk {
            let kernel = &kernel;
            std::thread::scope(|scope| {
                let mut chunks = units.chunks_mut(chunk).enumerate();
                // Held across the spawn loop: `ChunksMut` yields items that
                // borrow the buffer, not the iterator, so partition 0 can
                // run here while the rest are already in flight.
                let head = chunks.next();
                let mut handles = Vec::with_capacity(self.workers.get() - 1);
                for (index, slice) in chunks {
                    handles.push(scope.spawn(move || kernel(index, slice)));
                }
                if let Some((index, slice)) = head {
                    partials.push(kernel(index, slice));
                }
                for handle in handles {
                    match handle.join() {
                        Ok(partial) => partials.push(partial),
                        Err(payload) => std::panic::resume_unwind(payload),
                    }
                }
            });
            return;
        }

        for (index, slice) in units.chunks_mut(chunk).enumerate() {
            partials.push(kernel(index, slice));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-unit value with no structure a partition boundary could hide:
    /// xorshift64 (Marsaglia 2003, "Xorshift RNGs", J. Stat. Soft. 8(14),
    /// the 13/7/17 triple) off the index, mapped into [-1, 1) by an exact
    /// power-of-two scale so the f32 carries no rounding of its own.
    fn unit_value(index: usize) -> f32 {
        let mut x = (index as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 40) as f32 / 8_388_608.0 - 1.0
    }

    /// A stage meeting the disjoint-write contract: every slot is a function
    /// of its own global index alone.
    fn fill_disjoint(buffer: &mut [f32], workers: usize) {
        let pool = JobPool::new(workers);
        let chunk = partition_len(buffer.len(), pool.threads());
        let mut partials: Vec<()> = Vec::new();
        pool.run_stage(buffer, &mut partials, |partition, slice| {
            let base = partition * chunk;
            for (offset, slot) in slice.iter_mut().enumerate() {
                let v = unit_value(base + offset);
                *slot = v * v - 0.5 * v;
            }
        });
    }

    fn bits(buffer: &[f32]) -> Vec<u32> {
        buffer.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn the_partition_tiles_the_buffer_exactly_once_at_every_worker_count() {
        for units in 0..=17usize {
            for workers in 1..=8usize {
                let chunk = partition_len(units, workers);
                let mut buffer: Vec<usize> = (0..units).collect();
                let widths: Vec<usize> = buffer.chunks_mut(chunk).map(|c| c.len()).collect();

                assert!(
                    widths.len() <= workers,
                    "{units} units over {workers} workers produced {} partitions",
                    widths.len()
                );
                assert!(widths.iter().all(|&w| w > 0 && w <= chunk));
                assert_eq!(
                    widths.iter().sum::<usize>(),
                    units,
                    "{units} units over {workers} workers left a unit uncovered or doubled"
                );
            }
        }
    }

    #[test]
    fn disjoint_writes_are_bit_identical_between_one_worker_and_many() {
        // Deliberately not a multiple of any worker count under test, so
        // every count has a short tail partition.
        const UNITS: usize = 1_009;

        let mut serial = vec![f32::NAN; UNITS];
        fill_disjoint(&mut serial, 1);
        assert!(
            serial.iter().all(|v| v.is_finite()),
            "a unit was never written, so the comparison below would be vacuous"
        );

        for workers in 2..=8usize {
            let mut parallel = vec![f32::NAN; UNITS];
            fill_disjoint(&mut parallel, workers);
            assert_eq!(
                bits(&serial),
                bits(&parallel),
                "{workers} workers changed the bits a one-worker run produced"
            );
        }
    }

    #[test]
    fn partials_come_back_in_ascending_partition_index_not_completion_order() {
        const UNITS: usize = 64;
        const WORKERS: usize = 4;
        // Lower partitions do strictly more work, so a merge in completion
        // order comes back roughly reversed rather than by luck.
        const SKEW: u32 = 200_000;

        let pool = JobPool::new(WORKERS);
        let mut units = vec![0u32; UNITS];
        let expected = UNITS.div_ceil(partition_len(UNITS, pool.threads()));
        let mut partials = vec![usize::MAX; 3];

        pool.run_stage(&mut units, &mut partials, |partition, _slice| {
            let spin = (expected - partition) as u32 * SKEW;
            let mut acc = 0u32;
            for i in 0..spin {
                acc = acc.wrapping_add(std::hint::black_box(i));
            }
            std::hint::black_box(acc);
            partition
        });

        assert_eq!(
            partials,
            (0..expected).collect::<Vec<_>>(),
            "the join must merge by partition index and must clear the buffer first"
        );
    }

    #[test]
    fn a_spawned_partitions_panic_reaches_the_caller_carrying_its_own_payload() {
        const UNITS: usize = 8;
        const WORKERS: usize = 4;
        // Not 0, so the panic crosses the join rather than unwinding straight
        // out of the caller's own kernel call.
        const PANICKING: usize = 2;
        const MESSAGE: &str = "kernel panic under test";

        let pool = JobPool::new(WORKERS);
        let mut units = vec![0u32; UNITS];
        let mut partials: Vec<usize> = Vec::new();

        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.run_stage(&mut units, &mut partials, |partition, _slice| {
                assert_ne!(partition, PANICKING, "{MESSAGE}");
                partition
            });
        }));

        let payload = unwound.expect_err("a partition's panic must reach the caller");
        // The payload, not merely the fact of a panic: `thread::scope` would
        // panic at its own exit regardless, with a message naming the scope
        // instead of what the kernel was asserting.
        let text = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .expect("the kernel's own payload, not the scope's");
        assert!(text.contains(MESSAGE), "resumed some other panic: {text}");
    }

    #[test]
    fn a_budget_of_zero_still_yields_one_worker() {
        assert_eq!(JobPool::new(0).threads(), 1);
        assert_eq!(JobPool::new(1).threads(), 1);
        #[cfg(not(target_arch = "wasm32"))]
        assert_eq!(JobPool::new(7).threads(), 7);
    }

    #[test]
    fn an_empty_buffer_runs_no_partitions_at_any_worker_count() {
        for workers in 1..=4usize {
            let mut units: Vec<f32> = Vec::new();
            let mut partials: Vec<()> = vec![(); 2];
            JobPool::new(workers).run_stage(&mut units, &mut partials, |_, _| unreachable!());
            assert!(partials.is_empty());
        }
    }
}
