//! [`crate::frame_trace`] is the other half of this seam: the cross-worker
//! trace merge happens at this pool's join barrier.

use std::num::NonZeroUsize;

/// `ceil(units / workers)`, floored at 1 so an empty buffer still names a
/// legal chunk width. Contiguous ascending chunks of this width tile the
/// buffer exactly once and produce at most `workers` of them, so which units
/// a partition owns is a pure function of `(units, workers)` and of nothing
/// else: not of arrival order, not of how fast a worker ran.
/// Work stealing is excluded by that sentence, deliberately. Under stealing
/// the owner of a unit stops being a function of the schedule, and the
/// moment any stage carries a per-worker partial its reduction order stops
/// being fixed.
///
/// Panics if `workers` is zero.
pub fn partition_len(units: usize, workers: usize) -> usize {
    units.div_ceil(workers).max(1)
}

/// [`JobPool::run_stage`] joins every partition before it returns, so two
/// stages never overlap and two app-level systems can therefore never run
/// concurrently with each other.
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JobPool {
    workers: NonZeroUsize,
}

impl JobPool {
    /// `0` is read as 1.
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

    pub fn threads(&self) -> usize {
        self.workers.get()
    }

    /// Splits `units` by [`partition_len`] and joins before returning.
    ///
    /// `kernel` receives a partition index `k` and the units it owns, which
    /// are `[k * partition_len(units.len(), self.threads()) ..]` capped at
    /// the buffer end, so a kernel needing a unit's global index derives it
    /// from the same function the pool split by.
    ///
    /// `partials` is cleared and refilled with one kernel result per
    /// partition in ascending partition index.
    ///
    /// Panics from a kernel are resumed on the calling thread after the
    /// barrier, so a partition panicking cannot leave the caller believing
    /// the stage ran.
    ///
    /// A kernel may open [`crate::frame_trace::scope`] freely: recording is
    /// thread-local, so a spawned partition's sections would die with its
    /// thread, and the barrier merges them into the caller's in-flight frame
    /// in ascending partition index instead.
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
                    handles.push(scope.spawn(move || {
                        let partial = kernel(index, slice);
                        (partial, crate::frame_trace::take_worker_trace())
                    }));
                }
                // Partition 0 records straight into the caller's frame, and it
                // runs before any join, so the merge below extends an ascending
                // sequence rather than interleaving with one.
                if let Some((index, slice)) = head {
                    partials.push(kernel(index, slice));
                }
                for handle in handles {
                    match handle.join() {
                        Ok((partial, trace)) => {
                            partials.push(partial);
                            crate::frame_trace::merge_worker_trace(trace);
                        }
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

    // Per-unit value with no structure a partition boundary could hide:
    // xorshift64 (Marsaglia 2003, "Xorshift RNGs", J. Stat. Soft. 8(14),
    // the 13/7/17 triple) off the index, mapped into [-1, 1) by an exact
    // power-of-two scale so the f32 carries no rounding of its own.
    fn unit_value(index: usize) -> f32 {
        let mut x = (index as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 40) as f32 / 8_388_608.0 - 1.0
    }

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
        let text = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .expect("the kernel's own payload, not the scope's");
        assert!(text.contains(MESSAGE), "resumed some other panic: {text}");
    }

    #[cfg(feature = "frame-trace")]
    const PARTITION_SCOPES: [&str; 4] =
        ["partition-0", "partition-1", "partition-2", "partition-3"];

    // Lower partitions spin longer, so the spawned partitions complete in
    // descending index order and a merge that followed completion would come
    // back reversed.
    #[cfg(feature = "frame-trace")]
    fn stage_section_names(workers: usize) -> Vec<&'static str> {
        use crate::frame_trace;
        const UNITS: usize = 64;
        const SKEW: u32 = 200_000;

        // Drain what this test thread had in flight so the frame rolled below
        // holds this stage and nothing else.
        frame_trace::end_frame();

        let pool = JobPool::new(workers);
        let mut units = vec![0u32; UNITS];
        let partitions = UNITS.div_ceil(partition_len(UNITS, pool.threads()));
        let mut partials: Vec<()> = Vec::new();
        pool.run_stage(&mut units, &mut partials, |partition, _slice| {
            let _section = frame_trace::scope(PARTITION_SCOPES[partition]);
            let mut acc = 0u32;
            for i in 0..(partitions - partition) as u32 * SKEW {
                acc = acc.wrapping_add(std::hint::black_box(i));
            }
            std::hint::black_box(acc);
        });
        frame_trace::end_frame();

        frame_trace::last_frame()
            .expect("end_frame rolls a frame")
            .sections
            .iter()
            .map(|section| section.name)
            .filter(|name| PARTITION_SCOPES.contains(name))
            .collect()
    }

    #[cfg(feature = "frame-trace")]
    #[test]
    fn a_spawned_partitions_sections_reach_the_callers_rolling_aggregate() {
        use crate::frame_trace;
        use std::time::Duration;

        frame_trace::set_capacity(4);
        let names = stage_section_names(4);
        assert_eq!(names.len(), 4, "every partition should have recorded once");

        let stats = frame_trace::aggregate();
        for name in PARTITION_SCOPES {
            let row = stats
                .iter()
                .find(|row| row.name == name)
                .unwrap_or_else(|| panic!("'{name}' never reached the aggregate: {stats:?}"));
            assert!(
                row.max > Duration::ZERO,
                "'{name}' aggregated a zero max, so nothing was actually timed"
            );
        }
    }

    #[cfg(feature = "frame-trace")]
    #[test]
    fn merged_sections_follow_partition_index_not_completion_order() {
        assert_eq!(
            stage_section_names(4),
            PARTITION_SCOPES.to_vec(),
            "the join must merge traces by partition index"
        );
    }

    #[cfg(feature = "frame-trace")]
    #[test]
    fn one_worker_records_exactly_the_sections_an_unpooled_call_would() {
        use crate::frame_trace;

        let pooled = stage_section_names(1);

        frame_trace::end_frame();
        {
            let _section = frame_trace::scope(PARTITION_SCOPES[0]);
        }
        frame_trace::end_frame();
        let unpooled: Vec<&'static str> = frame_trace::last_frame()
            .expect("end_frame rolls a frame")
            .sections
            .iter()
            .map(|section| section.name)
            .filter(|name| PARTITION_SCOPES.contains(name))
            .collect();

        assert_eq!(pooled, unpooled);
        assert_eq!(pooled, vec![PARTITION_SCOPES[0]]);
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
