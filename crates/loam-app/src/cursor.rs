//! Pending requests are atomics the runner applies once per redraw, since
//! `Window::set_cursor_grab` must run on the main thread. On wasm every grab
//! maps to Pointer Lock and visibility follows the lock.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

/// Mirrors `winit::window::CursorGrabMode` so winit stays out of demo code.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum GrabMode {
    #[default]
    None,
    Confined,
    /// Motion arrives as raw device delta (`FrameInput::mouse_raw_delta`).
    Locked,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CursorState {
    pub grab: GrabMode,
    pub visible: bool,
}

impl CursorState {
    pub const RELEASED: Self = Self {
        grab: GrabMode::None,
        visible: true,
    };
}

const NONE: u8 = 0;
const GRAB_NONE: u8 = 1;
const GRAB_CONFINED: u8 = 2;
const GRAB_LOCKED: u8 = 3;
static PENDING_GRAB: AtomicU8 = AtomicU8::new(NONE);

const VIS_NONE: u8 = 0;
const VIS_SHOW: u8 = 1;
const VIS_HIDE: u8 = 2;
static PENDING_VISIBLE: AtomicU8 = AtomicU8::new(VIS_NONE);

static PENDING_WARP_CENTER: AtomicBool = AtomicBool::new(false);

static APPLIED_GRAB: AtomicU8 = AtomicU8::new(GRAB_NONE);
static APPLIED_VISIBLE: AtomicBool = AtomicBool::new(true);

fn grab_mode_to_code(mode: GrabMode) -> u8 {
    match mode {
        GrabMode::None => GRAB_NONE,
        GrabMode::Confined => GRAB_CONFINED,
        GrabMode::Locked => GRAB_LOCKED,
    }
}

fn code_to_grab_mode(code: u8) -> GrabMode {
    match code {
        GRAB_CONFINED => GrabMode::Confined,
        GRAB_LOCKED => GrabMode::Locked,
        _ => GrabMode::None,
    }
}

pub fn request_grab_mode(mode: GrabMode) {
    PENDING_GRAB.store(grab_mode_to_code(mode), Ordering::Release);
}

pub fn request_cursor_visible(visible: bool) {
    PENDING_VISIBLE.store(if visible { VIS_SHOW } else { VIS_HIDE }, Ordering::Release);
}

/// Pair with a grab release; no-op on wasm, where the browser owns positioning.
pub fn request_warp_to_center() {
    PENDING_WARP_CENTER.store(true, Ordering::Release);
}

pub fn request_grab() {
    request_grab_mode(GrabMode::Locked);
    request_cursor_visible(false);
}

pub fn request_release() {
    request_grab_mode(GrabMode::None);
    request_cursor_visible(true);
}

#[doc(hidden)]
pub fn take_pending() -> (Option<GrabMode>, Option<bool>) {
    let grab_code = PENDING_GRAB.swap(NONE, Ordering::AcqRel);
    let grab = (grab_code != NONE).then(|| code_to_grab_mode(grab_code));
    let vis_code = PENDING_VISIBLE.swap(VIS_NONE, Ordering::AcqRel);
    let visible = match vis_code {
        VIS_SHOW => Some(true),
        VIS_HIDE => Some(false),
        _ => None,
    };
    (grab, visible)
}

#[doc(hidden)]
pub fn take_pending_warp_center() -> bool {
    PENDING_WARP_CENTER.swap(false, Ordering::AcqRel)
}

#[doc(hidden)]
pub fn mark_applied(grab: GrabMode, visible: bool) {
    APPLIED_GRAB.store(grab_mode_to_code(grab), Ordering::Release);
    APPLIED_VISIBLE.store(visible, Ordering::Release);
}

pub fn current_state() -> CursorState {
    CursorState {
        grab: code_to_grab_mode(APPLIED_GRAB.load(Ordering::Acquire)),
        visible: APPLIED_VISIBLE.load(Ordering::Acquire),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, PoisonError};

    static CURSOR_GLOBALS: Mutex<()> = Mutex::new(());

    fn serialized() -> std::sync::MutexGuard<'static, ()> {
        CURSOR_GLOBALS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    #[test]
    fn round_trip_grab_mode() {
        let _guard = serialized();
        let _ = take_pending();
        request_grab_mode(GrabMode::Locked);
        let (grab, _) = take_pending();
        assert_eq!(grab, Some(GrabMode::Locked));
        let (grab, _) = take_pending();
        assert_eq!(grab, None, "second read clears");
        request_grab_mode(GrabMode::Confined);
        let (grab, _) = take_pending();
        assert_eq!(grab, Some(GrabMode::Confined));
    }

    #[test]
    fn convenience_pairs_set_both() {
        let _guard = serialized();
        let _ = take_pending();
        request_grab();
        let (g, v) = take_pending();
        assert_eq!(g, Some(GrabMode::Locked));
        assert_eq!(v, Some(false));

        request_release();
        let (g, v) = take_pending();
        assert_eq!(g, Some(GrabMode::None));
        assert_eq!(v, Some(true));
    }

    #[test]
    fn warp_to_center_is_one_shot() {
        let _guard = serialized();
        let _ = take_pending_warp_center();

        assert!(!take_pending_warp_center());

        request_warp_to_center();
        assert!(take_pending_warp_center(), "first read after request");
        assert!(
            !take_pending_warp_center(),
            "second read should clear the flag"
        );

        request_warp_to_center();
        request_warp_to_center();
        assert!(take_pending_warp_center());
        assert!(!take_pending_warp_center());
    }
}
