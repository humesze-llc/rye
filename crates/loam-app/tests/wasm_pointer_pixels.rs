//! `crate::wasm` is wasm32-gated, so the CSS-to-physical pixel conversion is
//! included by path and driven against a real `InputState`.

#[path = "../src/wasm/input_queue.rs"]
// Message variants this test has no reason to construct.
#[allow(dead_code)]
mod input_queue;

use input_queue::physical_cursor;
use loam_input::InputState;
use winit::event::{ElementState, MouseButton};

// Literal expectations, not the formula restated.
const CSS_X: f32 = 37.0;
const CSS_Y: f32 = 91.0;
const CASES: [(f32, f32, f32); 3] = [(1.0, 37.0, 91.0), (1.5, 55.5, 136.5), (2.0, 74.0, 182.0)];

fn mouse_move(input: &mut InputState, x: f32, y: f32, dpr: f32) {
    let (x, y) = physical_cursor(x, y, dpr);
    input.cursor_moved(x, y);
}

// Mirrors the worker's `MouseButton` arm: the click's own position lands before
// the transition is recorded.
fn mouse_press(input: &mut InputState, x: f32, y: f32, dpr: f32) {
    let (x, y) = physical_cursor(x, y, dpr);
    input.cursor_moved(x, y);
    input.mouse_input(MouseButton::Left, ElementState::Pressed);
}

#[test]
fn cursor_position_is_reported_in_physical_pixels() {
    for (dpr, want_x, want_y) in CASES {
        let mut input = InputState::default();
        mouse_move(&mut input, CSS_X, CSS_Y, dpr);
        let pos = input
            .take_frame()
            .cursor_pos
            .expect("a move sets the position");
        assert_eq!((pos.x, pos.y), (want_x, want_y), "at DPR {dpr}");
    }
}

#[test]
fn a_press_anchors_at_its_own_position_not_the_last_coalesced_move() {
    for (dpr, want_x, want_y) in CASES {
        let mut input = InputState::default();
        mouse_move(&mut input, CSS_X - 20.0, CSS_Y - 30.0, dpr);
        mouse_press(&mut input, CSS_X, CSS_Y, dpr);
        let frame = input.take_frame();
        let anchor = frame
            .buttons
            .left
            .press_pos
            .expect("a press with a known cursor position anchors");
        assert_eq!((anchor.x, anchor.y), (want_x, want_y), "at DPR {dpr}");
        assert!(frame.left_mouse_down);
    }
}
