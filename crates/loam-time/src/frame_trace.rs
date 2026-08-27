//! Recording is thread-local: a section lands in the buffer of whichever thread
//! opened it, and a worker thread's buffer dies with the thread. That would
//! silently drop exactly the measurements a parallel stage has to justify itself
//! with, so [`take_worker_trace`] and [`merge_worker_trace`] move a worker's
//! sections onto the thread that owns the frame.
//! [`crate::jobs::JobPool::run_stage`] calls the pair at its join barrier, in
//! ascending partition index, so a frame's section list is a function of the
//! partition and not of which worker finished first.

#[cfg(feature = "frame-trace")]
use std::cell::RefCell;
#[cfg(feature = "frame-trace")]
use std::collections::VecDeque;
use std::time::Duration;
#[cfg(feature = "frame-trace")]
use web_time::Instant;

/// One CPU section timing inside a single frame's trace.
#[derive(Clone, Debug)]
pub struct Section {
    pub name: &'static str,
    pub elapsed: Duration,
}

/// All sections recorded inside one redraw cycle. `Default` is an empty trace.
///
/// `heap_delta_bytes` is signed JS heap growth this frame; only populated when a
/// host registers a [`HeapSampler`] via [`set_heap_sampler`] AND the runtime
/// exposes the API (Chrome/Edge yes, Firefox/native `None`). Negative means a GC
/// reclaimed heap mid-frame.
///
/// `allocs` is the per-frame [`crate::alloc::CountingAllocator`] delta; only
/// populated when a demo installs that allocator as its `#[global_allocator]`.
#[derive(Clone, Debug, Default)]
pub struct FrameTrace {
    pub sections: Vec<Section>,
    pub heap_delta_bytes: Option<i64>,
    pub allocs: Option<crate::alloc::AllocDelta>,
}

impl FrameTrace {
    /// Total time across every section.
    pub fn total(&self) -> Duration {
        self.sections.iter().map(|s| s.elapsed).sum()
    }
}

/// Default rolling-buffer capacity. 120 frames is two seconds at 60fps; footprint
/// stays under ~10 KB.
pub const DEFAULT_CAPACITY: usize = 120;

#[cfg(feature = "frame-trace")]
struct Tracer {
    history: VecDeque<FrameTrace>,
    capacity: usize,
}

#[cfg(feature = "frame-trace")]
impl Tracer {
    fn new(capacity: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(capacity),
            capacity,
        }
    }
}

/// Host-registered JS heap sampler (bytes). On wasm32 + Chromium the host wires
/// this to `performance.memory.usedJSHeapSize`; elsewhere no sampler is set and
/// `heap_delta_bytes` stays `None`.
pub type HeapSampler = fn() -> Option<u64>;

#[cfg(feature = "frame-trace")]
thread_local! {
    static TRACER: RefCell<Tracer> = RefCell::new(Tracer::new(DEFAULT_CAPACITY));
    static CURRENT_SECTIONS: RefCell<Vec<Section>> = const { RefCell::new(Vec::new()) };
    static LAST_FRAME_END: std::cell::Cell<Option<Instant>> = const { std::cell::Cell::new(None) };
    static CURRENT_FRAME_START: std::cell::Cell<Option<Instant>> = const { std::cell::Cell::new(None) };
    static CURRENT_FRAME_HEAP_START: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
    static CURRENT_FRAME_ALLOC_START: std::cell::Cell<Option<crate::alloc::AllocSnapshot>> = const { std::cell::Cell::new(None) };
    static HEAP_SAMPLER: std::cell::Cell<Option<HeapSampler>> = const { std::cell::Cell::new(None) };
    static MAX_EVER: RefCell<std::collections::HashMap<&'static str, Duration>> =
        RefCell::new(std::collections::HashMap::new());
    /// 250ms is "user-perceptible freeze"; below it, routine GC stalls and
    /// wireframe rebuilds (50-150ms) drown the log.
    static SPIKE_THRESHOLD: std::cell::Cell<Duration> =
        const { std::cell::Cell::new(Duration::from_millis(250)) };
    static FRAME_COUNTER: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// RAII guard from [`scope`]; records elapsed time on drop. Not safe to hold
/// across `await` (the tracer is thread-local, borrowed mutably during drop).
#[cfg(feature = "frame-trace")]
#[must_use = "Scope records on drop; binding it to `_` would record immediately"]
pub struct Scope {
    name: &'static str,
    start: Instant,
}

#[cfg(feature = "frame-trace")]
impl Drop for Scope {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        let name = self.name;
        // try_borrow_mut: a scope dropping while `end_frame` rotates the buffer
        // loses its sample rather than panicking inside a `Drop`.
        CURRENT_SECTIONS.with(|s| {
            if let Ok(mut s) = s.try_borrow_mut() {
                s.push(Section { name, elapsed });
            }
        });
    }
}

/// Open a CPU-timing scope; the guard records on drop. Bind it with a real name
/// (`let _s = scope("foo")`); binding to `_` drops immediately and records zero.
#[cfg(feature = "frame-trace")]
#[inline]
pub fn scope(name: &'static str) -> Scope {
    Scope {
        name,
        start: Instant::now(),
    }
}

/// Register the host's JS heap sampler (last write wins).
#[cfg(feature = "frame-trace")]
pub fn set_heap_sampler(sampler: HeapSampler) {
    HEAP_SAMPLER.with(|s| s.set(Some(sampler)));
}

/// Mark the start of a frame's work. Pairs with [`end_frame`] to compute the
/// `idle` section.
#[cfg(feature = "frame-trace")]
pub fn begin_frame() {
    CURRENT_FRAME_START.with(|c| c.set(Some(Instant::now())));
    let sampler = HEAP_SAMPLER.with(|s| s.get());
    CURRENT_FRAME_HEAP_START.with(|c| c.set(sampler.and_then(|f| f())));
    CURRENT_FRAME_ALLOC_START.with(|c| c.set(crate::alloc::current_snapshot()));
}

/// Push the in-flight frame into history and start a new one.
///
/// Records two synthetic sections beyond the explicit scopes:
///
/// - **`between-frames`**: wall-clock between successive `end_frame` calls (= 1/fps).
/// - **`idle`**: last `end_frame` to this frame's `begin_frame`, the gap when our
///   code wasn't running (RAF scheduling, vsync, GC, tab throttling). Dominates
///   `between-frames` on wasm.
///
/// The first frame after startup records neither.
#[cfg(feature = "frame-trace")]
pub fn end_frame() {
    let now = Instant::now();
    let last_end = LAST_FRAME_END.with(|cell| {
        let prev = cell.get();
        cell.set(Some(now));
        prev
    });
    let frame_start = CURRENT_FRAME_START.with(|c| c.take());

    let heap_start = CURRENT_FRAME_HEAP_START.with(|c| c.take());
    let sampler = HEAP_SAMPLER.with(|s| s.get());
    let heap_end = sampler.and_then(|f| f());
    let heap_delta_bytes: Option<i64> = match (heap_start, heap_end) {
        (Some(a), Some(b)) => Some((b as i64).saturating_sub(a as i64)),
        _ => None,
    };

    let alloc_start = CURRENT_FRAME_ALLOC_START.with(|c| c.take());
    let alloc_end = crate::alloc::current_snapshot();
    let alloc_delta: Option<crate::alloc::AllocDelta> = match (alloc_start, alloc_end) {
        (Some(a), Some(b)) => Some(crate::alloc::delta(a, b)),
        _ => None,
    };

    let frame_index = FRAME_COUNTER.with(|c| {
        let n = c.get();
        c.set(n.wrapping_add(1));
        n
    });

    let threshold = SPIKE_THRESHOLD.with(|c| c.get());
    let mut sections = CURRENT_SECTIONS.with(|s| std::mem::take(&mut *s.borrow_mut()));

    if let Some(last_end) = last_end {
        let between_frames = now.saturating_duration_since(last_end);
        sections.push(Section {
            name: "between-frames",
            elapsed: between_frames,
        });
        if let Some(frame_start) = frame_start {
            let idle = frame_start.saturating_duration_since(last_end);
            sections.push(Section {
                name: "idle",
                elapsed: idle,
            });
        }
    }

    // Owning the sections here keeps MAX_EVER, TRACER and the warn loop from
    // ever nesting their borrows, which is what a re-entrant tracing subscriber
    // would deadlock on.
    let mut over_threshold: Vec<(&'static str, Duration)> = Vec::new();
    MAX_EVER.with(|m| {
        let mut m = m.borrow_mut();
        for section in &sections {
            let entry = m.entry(section.name).or_insert(Duration::ZERO);
            if section.elapsed > *entry {
                *entry = section.elapsed;
            }
            if section.elapsed > threshold {
                over_threshold.push((section.name, section.elapsed));
            }
        }
    });

    TRACER.with(|t| {
        let mut t = t.borrow_mut();
        let cap = t.capacity;
        if t.history.len() >= cap {
            t.history.pop_front();
        }
        t.history.push_back(FrameTrace {
            sections,
            heap_delta_bytes,
            allocs: alloc_delta,
        });
    });

    for (name, elapsed) in over_threshold {
        let heap_suffix = heap_delta_bytes
            .map(|d| format!(" heap_delta={:+.2}MB", d as f64 / (1024.0 * 1024.0)))
            .unwrap_or_default();
        let alloc_suffix = alloc_delta
            .map(|d| {
                format!(
                    " allocs={} ({:+.2}MB net)",
                    d.alloc_count,
                    d.net_bytes as f64 / (1024.0 * 1024.0),
                )
            })
            .unwrap_or_default();
        tracing::warn!(
            "frame_trace spike: section='{name}' elapsed={:.1}ms frame={frame_index}{heap_suffix}{alloc_suffix}",
            elapsed.as_secs_f32() * 1000.0,
        );
    }
}

/// Set the rolling window size. Truncates older frames if shrinking.
#[cfg(feature = "frame-trace")]
pub fn set_capacity(capacity: usize) {
    TRACER.with(|t| {
        let mut t = t.borrow_mut();
        t.capacity = capacity.max(1);
        while t.history.len() > t.capacity {
            t.history.pop_front();
        }
    });
}

/// Snapshot the rolling history (oldest-to-newest). Allocates; for the display
/// path, not the hot path.
#[cfg(feature = "frame-trace")]
pub fn history() -> Vec<FrameTrace> {
    TRACER.with(|t| t.borrow().history.iter().cloned().collect())
}

/// Run `f` with a borrow of the rolling history. Zero-allocation read path.
///
/// `f` must NOT call [`end_frame`] or [`set_capacity`] while the borrow is held;
/// that deadlocks the `RefCell`.
#[cfg(feature = "frame-trace")]
pub fn with_history<R>(f: impl FnOnce(&std::collections::VecDeque<FrameTrace>) -> R) -> R {
    TRACER.with(|t| f(&t.borrow().history))
}

/// Snapshot only the last completed frame.
#[cfg(feature = "frame-trace")]
pub fn last_frame() -> Option<FrameTrace> {
    TRACER.with(|t| t.borrow().history.back().cloned())
}

/// Session-lifetime max elapsed for `name` since startup (or last
/// [`clear_max_ever`]). `Duration::ZERO` for never-seen sections.
#[cfg(feature = "frame-trace")]
pub fn max_ever(name: &'static str) -> Duration {
    MAX_EVER.with(|m| m.borrow().get(name).copied().unwrap_or(Duration::ZERO))
}

/// All session-lifetime maxima, name -> duration, sorted descending.
#[cfg(feature = "frame-trace")]
pub fn all_max_ever() -> Vec<(&'static str, Duration)> {
    MAX_EVER.with(|m| {
        let mut out: Vec<(&'static str, Duration)> =
            m.borrow().iter().map(|(k, v)| (*k, *v)).collect();
        out.sort_by_key(|entry| std::cmp::Reverse(entry.1));
        out
    })
}

/// Reset session-lifetime maxima. Doesn't touch the rolling window.
#[cfg(feature = "frame-trace")]
pub fn clear_max_ever() {
    MAX_EVER.with(|m| m.borrow_mut().clear());
}

/// Set the threshold above which `end_frame` logs a spike `tracing::warn!`. Pass
/// `Duration::MAX` to disable.
#[cfg(feature = "frame-trace")]
pub fn set_spike_threshold(threshold: Duration) {
    SPIKE_THRESHOLD.with(|c| c.set(threshold));
}

/// Push a section produced outside the `scope` lifecycle (GPU timer path: a
/// timestamp delta arriving via `map_async`). Lands in the in-flight frame; late
/// timestamps (1-2 frames) attribute to whatever frame is current, which the
/// rolling-window aggregate absorbs.
#[cfg(feature = "frame-trace")]
pub fn record_external(name: &'static str, elapsed: Duration) {
    CURRENT_SECTIONS.with(|s| {
        if let Ok(mut s) = s.try_borrow_mut() {
            s.push(Section { name, elapsed });
        }
    });
}

#[cfg(not(feature = "frame-trace"))]
pub fn record_external(_name: &'static str, _elapsed: Duration) {}

/// Sections one worker thread recorded, in transit to the thread that joins it.
#[cfg(feature = "frame-trace")]
#[derive(Debug, Default)]
pub struct WorkerTrace(Vec<Section>);

/// Detach the calling thread's in-flight sections so a joining thread can merge
/// them.
///
/// Calling this on the thread that owns the frame steals that frame's own
/// sections instead.
#[cfg(feature = "frame-trace")]
pub fn take_worker_trace() -> WorkerTrace {
    CURRENT_SECTIONS.with(|s| WorkerTrace(std::mem::take(&mut *s.borrow_mut())))
}

/// Append a joined worker's sections to the caller's in-flight frame.
///
/// Ordering is the caller's responsibility and is what makes the merge
/// deterministic: [`crate::jobs::JobPool::run_stage`] calls this in ascending
/// partition index, never in completion order.
#[cfg(feature = "frame-trace")]
pub fn merge_worker_trace(trace: WorkerTrace) {
    CURRENT_SECTIONS.with(|s| s.borrow_mut().extend(trace.0));
}

#[cfg(not(feature = "frame-trace"))]
#[derive(Debug, Default)]
pub struct WorkerTrace;

#[cfg(not(feature = "frame-trace"))]
pub fn take_worker_trace() -> WorkerTrace {
    WorkerTrace
}

#[cfg(not(feature = "frame-trace"))]
pub fn merge_worker_trace(_trace: WorkerTrace) {}

#[cfg(not(feature = "frame-trace"))]
pub fn max_ever(_name: &'static str) -> Duration {
    Duration::ZERO
}

#[cfg(not(feature = "frame-trace"))]
pub fn all_max_ever() -> Vec<(&'static str, Duration)> {
    Vec::new()
}

#[cfg(not(feature = "frame-trace"))]
pub fn clear_max_ever() {}

#[cfg(not(feature = "frame-trace"))]
pub fn set_spike_threshold(_threshold: Duration) {}

#[cfg(not(feature = "frame-trace"))]
pub fn set_heap_sampler(_sampler: HeapSampler) {}

/// Aggregate stats across the rolling window for one section name.
#[derive(Clone, Debug)]
pub struct SectionStats {
    pub name: &'static str,
    pub samples: usize,
    pub mean: Duration,
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub max: Duration,
}

/// Aggregate every section across the rolling window, keyed by name, returned in
/// descending p95 order (slowest first).
#[cfg(feature = "frame-trace")]
pub fn aggregate() -> Vec<SectionStats> {
    use std::collections::HashMap;
    let frames = history();
    let mut buckets: HashMap<&'static str, Vec<Duration>> = HashMap::new();
    for frame in &frames {
        for section in &frame.sections {
            buckets
                .entry(section.name)
                .or_default()
                .push(section.elapsed);
        }
    }

    let mut stats: Vec<SectionStats> = buckets
        .into_iter()
        .map(|(name, mut samples)| {
            samples.sort();
            let n = samples.len();
            let mean = samples.iter().sum::<Duration>() / (n as u32).max(1);
            let pick = |q: f32| samples[((n as f32 * q) as usize).min(n - 1)];
            SectionStats {
                name,
                samples: n,
                mean,
                p50: pick(0.50),
                p95: pick(0.95),
                p99: pick(0.99),
                max: *samples.last().unwrap_or(&Duration::ZERO),
            }
        })
        .collect();

    stats.sort_by_key(|s| std::cmp::Reverse(s.p95));
    stats
}

#[cfg(not(feature = "frame-trace"))]
#[must_use]
pub struct Scope;

#[cfg(not(feature = "frame-trace"))]
#[inline]
pub fn scope(_name: &'static str) -> Scope {
    Scope
}

#[cfg(not(feature = "frame-trace"))]
pub fn end_frame() {}

#[cfg(not(feature = "frame-trace"))]
pub fn begin_frame() {}

#[cfg(not(feature = "frame-trace"))]
pub fn set_capacity(_capacity: usize) {}

#[cfg(not(feature = "frame-trace"))]
pub fn history() -> Vec<FrameTrace> {
    Vec::new()
}

#[cfg(not(feature = "frame-trace"))]
pub fn with_history<R>(f: impl FnOnce(&std::collections::VecDeque<FrameTrace>) -> R) -> R {
    let empty = std::collections::VecDeque::new();
    f(&empty)
}

#[cfg(not(feature = "frame-trace"))]
pub fn last_frame() -> Option<FrameTrace> {
    None
}

#[cfg(not(feature = "frame-trace"))]
pub fn aggregate() -> Vec<SectionStats> {
    Vec::new()
}

#[cfg(all(test, feature = "frame-trace"))]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn scope_records_elapsed_on_drop() {
        end_frame(); // discard any pre-existing in-flight frame
        let _ = history();

        {
            let _s = scope("test-a");
            sleep(Duration::from_millis(1));
        }
        let pre_frame = last_frame();
        end_frame();
        let post_frame = last_frame().expect("end_frame should produce a frame");

        let _ = pre_frame;
        let sections = &post_frame.sections;
        assert!(
            sections.iter().any(|s| s.name == "test-a"),
            "expected 'test-a' in {sections:?}"
        );
        let test_a = sections.iter().find(|s| s.name == "test-a").unwrap();
        assert!(
            test_a.elapsed >= Duration::from_millis(1),
            "scope elapsed should be >= sleep duration, got {:?}",
            test_a.elapsed
        );
    }

    #[test]
    fn end_frame_caps_history_to_capacity() {
        set_capacity(3);
        for _ in 0..10 {
            {
                let _s = scope("cap-test");
            }
            end_frame();
        }
        assert!(history().len() <= 3, "history should be capped");
    }

    #[test]
    fn heap_sampler_populates_delta_on_completed_frame() {
        use std::sync::atomic::{AtomicU64, Ordering};
        // Synthetic strictly-increasing sampler; begin + end each call it once,
        // so the delta equals the per-call increment.
        static FAKE_HEAP: AtomicU64 = AtomicU64::new(1_000_000);
        fn fake_sampler() -> Option<u64> {
            Some(FAKE_HEAP.fetch_add(4096, Ordering::SeqCst) + 4096)
        }
        FAKE_HEAP.store(1_000_000, Ordering::SeqCst);
        set_heap_sampler(fake_sampler);
        end_frame(); // discard any pre-existing in-flight frame
        begin_frame();
        end_frame();
        let frame = last_frame().expect("end_frame should produce a frame");
        let delta = frame
            .heap_delta_bytes
            .expect("sampler is registered; delta should be Some");
        assert_eq!(delta, 4096, "expected one-increment delta, got {delta}");
    }

    #[test]
    fn merged_worker_traces_land_in_merge_order_and_the_take_empties_the_source() {
        end_frame(); // discard any pre-existing in-flight frame

        {
            let _s = scope("recorded-first");
        }
        let first = take_worker_trace();
        assert_eq!(first.0.len(), 1, "the take should carry the one section");
        assert!(
            take_worker_trace().0.is_empty(),
            "the take must leave the recording thread's buffer empty"
        );

        {
            let _s = scope("recorded-second");
        }
        let second = take_worker_trace();

        {
            let _s = scope("local");
        }
        // Merged against the order they were recorded in: the frame has to
        // follow the merge calls, which is the whole basis of the pool's
        // ascending-partition join.
        merge_worker_trace(second);
        merge_worker_trace(first);
        end_frame();

        let names: Vec<&str> = last_frame()
            .expect("end_frame should produce a frame")
            .sections
            .iter()
            .map(|s| s.name)
            .filter(|name| !matches!(*name, "between-frames" | "idle"))
            .collect();
        assert_eq!(names, ["local", "recorded-second", "recorded-first"]);
    }

    #[test]
    fn aggregate_sorts_by_p95_descending() {
        set_capacity(20);
        for _ in 0..10 {
            {
                let _s = scope("slow");
                sleep(Duration::from_millis(2));
            }
            {
                let _f = scope("fast");
            }
            end_frame();
        }
        let stats = aggregate();
        let slow_idx = stats.iter().position(|s| s.name == "slow");
        let fast_idx = stats.iter().position(|s| s.name == "fast");
        assert!(
            slow_idx.is_some() && fast_idx.is_some(),
            "both sections present"
        );
        assert!(
            slow_idx.unwrap() < fast_idx.unwrap(),
            "'slow' should sort before 'fast' by descending p95: {stats:?}"
        );
    }
}
