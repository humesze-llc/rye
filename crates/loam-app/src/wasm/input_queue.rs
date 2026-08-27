//! Free of `web-sys` and `js-sys` so the queue policy and the pixel units
//! the messages carry are exercisable off target.

use std::cell::RefCell;
use std::collections::VecDeque;

/// Per-frame inputs the main thread forwards to the worker.
#[derive(Debug)]
pub enum InputMessage {
    /// New canvas pixel dimensions (DPR-multiplied to physical pixels).
    /// `dpr` rides
    /// along because the worker has no `window.devicePixelRatio` and a
    /// monitor change alters the ratio and the size together.
    Resize {
        width: u32,
        height: u32,
        dpr: f32,
    },

    /// Pointer moved to (x, y) in canvas-local CSS pixels. `buttons` is
    /// the `MouseEvent.buttons` bitmask. `dx`/`dy` are raw
    /// `movementX/Y` deltas for FPS mouse-look, valid both before and
    /// after Pointer Lock engages; coalesced moves sum the dropped
    /// intermediate deltas.
    MouseMove {
        x: f32,
        y: f32,
        buttons: u8,
        dx: f32,
        dy: f32,
    },

    /// `button` is `MouseEvent.button`
    /// (0=primary, 1=middle, 2=secondary).
    MouseButton {
        x: f32,
        y: f32,
        button: u8,
        pressed: bool,
    },

    /// Wheel delta in lines (normalized on main thread). DOM convention:
    /// positive = right/down.
    MouseWheel {
        dx: f32,
        dy: f32,
    },

    /// `code` is the physical-key code (for
    /// hotkey routing via `keymap::keycode_*`); `key` is the logical key
    /// (for text-input fan-out to egui).
    Key {
        code: String,
        key: String,
        pressed: bool,
        repeat: bool,
        ctrl: bool,
        shift: bool,
        alt: bool,
        meta: bool,
    },

    Focus(bool),

    Visibility(bool),

    /// Sent after the user clicks the
    /// launch overlay; before it arrives the worker has rendered one
    /// preview frame for the overlay to blur.
    Start,

    /// Pointer Lock state mirror from the main thread's
    /// `pointerlockchange` event. The worker calls
    /// [`crate::cursor::mark_applied`] so `current_state()` tracks what
    /// the browser actually has.
    PointerLockChanged(bool),
}

/// Hard cap on queued messages; reachable only when
/// nothing is draining (paused embed, halted RAF chain), which is
/// unbounded in wall-clock time.
pub const MESSAGE_QUEUE_CAPACITY: usize = 256;

thread_local! {
    /// `thread_local` because the message handler
    /// closure has no handle to the asynchronously-constructed runner.
    static MESSAGE_QUEUE: RefCell<VecDeque<InputMessage>> = const { RefCell::new(VecDeque::new()) };
}

/// Oldest-first eviction because a full queue means no frame has drained
/// for a long time, so the newest messages carry the state that will
/// still be true on resume. Dropping the newest instead can strand a key
/// release behind its press and leave the key stuck down.
pub fn enqueue(msg: InputMessage) {
    MESSAGE_QUEUE.with(|q| {
        let mut q = q.borrow_mut();
        if q.len() >= MESSAGE_QUEUE_CAPACITY {
            let dropped = q.pop_front();
            tracing::warn!("loam_app::wasm::worker: input queue full, dropped {dropped:?}");
        }
        q.push_back(msg);
    });
}

pub fn drain_messages() -> Vec<InputMessage> {
    MESSAGE_QUEUE.with(|q| q.borrow_mut().drain(..).collect())
}

/// The CSS pixels [`InputMessage::MouseMove`] / [`InputMessage::MouseButton`]
/// carry, scaled to the physical pixels `FrameInput::cursor_pos` is
/// specified in. `device_pixel_ratio` is the ratio the canvas backing
/// store was sized with, so a pick ray built from the result indexes the
/// same pixel grid the frame was rendered on.
pub fn physical_cursor(x: f32, y: f32, device_pixel_ratio: f32) -> (f64, f64) {
    (
        (x * device_pixel_ratio) as f64,
        (y * device_pixel_ratio) as f64,
    )
}
