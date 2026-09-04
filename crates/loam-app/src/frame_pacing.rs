//! Native: the runner sleeps to each redraw's deadline; `target_fps = 0` leaves
//! the surface `PresentMode` to pace. Wasm: `requestAnimationFrame` is the
//! upper bound, so the runner can only cap lower by skipping callbacks.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::time::Duration;
use web_time::Instant;

// `0` = uncapped; the runner neither sleeps nor skips.
const DEFAULT_PERIOD_NS: u64 = 0;

static TARGET_PERIOD_NS: AtomicU64 = AtomicU64::new(DEFAULT_PERIOD_NS);

/// `fps <= 0.0` removes the cap.
pub fn set_target_fps(fps: f32) {
    if fps <= 0.0 {
        TARGET_PERIOD_NS.store(0, Ordering::Release);
        return;
    }
    let period_ns = (1_000_000_000.0 / fps as f64) as u64;
    TARGET_PERIOD_NS.store(period_ns.max(1), Ordering::Release);
}

pub fn target_period() -> Option<Duration> {
    let ns = TARGET_PERIOD_NS.load(Ordering::Acquire);
    if ns == 0 {
        None
    } else {
        Some(Duration::from_nanos(ns))
    }
}

pub fn target_fps() -> f32 {
    let ns = TARGET_PERIOD_NS.load(Ordering::Acquire);
    if ns == 0 {
        0.0
    } else {
        1_000_000_000.0 / ns as f32
    }
}

const VSYNC_NONE: u8 = 0;
const VSYNC_REQ_ON: u8 = 1;
const VSYNC_REQ_OFF: u8 = 2;
static PENDING_VSYNC: AtomicU8 = AtomicU8::new(VSYNC_NONE);

pub fn request_vsync_on() {
    PENDING_VSYNC.store(VSYNC_REQ_ON, Ordering::Release);
}

/// The runner picks `Mailbox`, else `Immediate`, else leaves `Fifo`.
pub fn request_vsync_off() {
    PENDING_VSYNC.store(VSYNC_REQ_OFF, Ordering::Release);
}

pub fn take_pending_vsync() -> Option<bool> {
    match PENDING_VSYNC.swap(VSYNC_NONE, Ordering::AcqRel) {
        VSYNC_REQ_ON => Some(true),
        VSYNC_REQ_OFF => Some(false),
        _ => None,
    }
}

// Worst-case `std::thread::sleep` overshoot seen on Win11's 15.625 ms timer tick.
#[cfg(not(target_arch = "wasm32"))]
const SPIN_TAIL: Duration = Duration::from_millis(2);

/// Sleeps, then spins the last `SPIN_TAIL`; plain sleep rounds to the timer tick.
#[cfg(not(target_arch = "wasm32"))]
pub fn precise_sleep_until(deadline: Instant) {
    let now = Instant::now();
    if deadline <= now {
        return;
    }
    let total = deadline - now;
    if total > SPIN_TAIL {
        std::thread::sleep(total - SPIN_TAIL);
    }
    while Instant::now() < deadline {
        std::hint::spin_loop();
    }
}

#[cfg(target_arch = "wasm32")]
pub fn precise_sleep_until(_deadline: Instant) {}

#[cfg(test)]
pub(crate) static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unlimited_round_trip() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_target_fps(0.0);
        assert_eq!(target_fps(), 0.0);
        assert_eq!(target_period(), None);
        set_target_fps(60.0);
    }

    #[test]
    fn vsync_request_round_trip() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = take_pending_vsync();
        request_vsync_on();
        assert_eq!(take_pending_vsync(), Some(true));
        assert_eq!(take_pending_vsync(), None, "should clear after read");
        request_vsync_off();
        assert_eq!(take_pending_vsync(), Some(false));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn precise_sleep_lands_close_to_deadline() {
        let start = Instant::now();
        let deadline = start + Duration::from_millis(50);
        precise_sleep_until(deadline);
        let actual = Instant::now() - start;
        assert!(
            actual >= Duration::from_millis(50),
            "woke up early: {actual:?} (deadline was 50 ms)"
        );
        assert!(
            actual < Duration::from_millis(55),
            "overshot too much: {actual:?} (deadline was 50 ms)"
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn precise_sleep_steady_cadence() {
        let period = Duration::from_millis(20);
        let start = Instant::now();
        let mut deadline = start;
        for _ in 0..5 {
            deadline += period;
            precise_sleep_until(deadline);
        }
        let actual = Instant::now() - start;
        let expected = period * 5;
        let diff = actual.abs_diff(expected);
        assert!(
            diff < Duration::from_millis(5),
            "cadence drifted: actual={actual:?} expected={expected:?}",
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn precise_sleep_past_deadline_returns_immediately() {
        let start = Instant::now();
        let deadline = start - Duration::from_millis(10);
        precise_sleep_until(deadline);
        let actual = Instant::now() - start;
        assert!(
            actual < Duration::from_millis(2),
            "took too long: {actual:?}"
        );
    }
}
