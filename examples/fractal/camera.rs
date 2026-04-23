use std::f32::consts::FRAC_PI_2;

use glam::{Quat, Vec2, Vec3};
use rye_math::Iso3;
use winit::event::{ElementState, MouseButton, MouseScrollDelta};

const ORBIT_RADIANS_PER_PIXEL: f32 = 0.006;
const ZOOM_LOG_STEP: f32 = 0.12;
const MIN_DISTANCE: f32 = 1.5;
const MAX_DISTANCE: f32 = 8.0;
const INITIAL_HEIGHT: f32 = 0.6;
const INITIAL_RADIUS: f32 = 3.5;
const MIN_PITCH: f32 = -1.45;
const MAX_PITCH: f32 = 1.45;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameInput {
    pub mouse_delta: Vec2,
    pub scroll_lines: f32,
    pub left_mouse_down: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraView {
    pub position: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraState {
    target: Vec3,
    yaw: f32,
    pitch: f32,
    distance: f32,
    orientation: Iso3,
}

impl Default for CameraState {
    fn default() -> Self {
        let distance = (INITIAL_RADIUS * INITIAL_RADIUS + INITIAL_HEIGHT * INITIAL_HEIGHT).sqrt();
        let pitch = -(INITIAL_HEIGHT / distance).asin();
        let mut camera = Self {
            target: Vec3::ZERO,
            yaw: FRAC_PI_2,
            pitch,
            distance,
            orientation: Iso3::IDENTITY,
        };
        camera.rebuild_orientation();
        camera
    }
}

impl CameraState {
    pub fn advance(&mut self, input: FrameInput) {
        if input.left_mouse_down {
            self.yaw -= input.mouse_delta.x * ORBIT_RADIANS_PER_PIXEL;
            self.pitch = (self.pitch - input.mouse_delta.y * ORBIT_RADIANS_PER_PIXEL)
                .clamp(MIN_PITCH, MAX_PITCH);
        }

        if input.scroll_lines != 0.0 {
            self.distance = (self.distance * (-input.scroll_lines * ZOOM_LOG_STEP).exp())
                .clamp(MIN_DISTANCE, MAX_DISTANCE);
        }

        self.rebuild_orientation();
    }

    pub fn view(&self) -> CameraView {
        let right = self.orientation.rotation * Vec3::X;
        let up = self.orientation.rotation * Vec3::Y;
        let back = self.orientation.rotation * Vec3::Z;
        CameraView {
            position: self.target + back * self.distance,
            forward: -back,
            right,
            up,
        }
    }

    /// Advance the yaw by `delta` radians. Used by auto-rotate mode.
    pub fn rotate_yaw(&mut self, delta: f32) {
        self.yaw += delta;
        self.rebuild_orientation();
    }

    /// Snap to a fixed orbit position (for capture / movie mode).
    pub fn set_orbit(&mut self, distance: f32, pitch: f32) {
        self.distance = distance.clamp(MIN_DISTANCE, MAX_DISTANCE);
        self.pitch = pitch.clamp(MIN_PITCH, MAX_PITCH);
        self.yaw = 0.0;
        self.rebuild_orientation();
    }

    fn rebuild_orientation(&mut self) {
        let yaw = Quat::from_rotation_y(self.yaw);
        let pitch = Quat::from_rotation_x(self.pitch);
        self.orientation = Iso3::from_rotation(yaw * pitch);
    }
}

#[derive(Debug, Default)]
pub struct InputState {
    frame: FrameInput,
    last_cursor: Option<Vec2>,
}

/// Approximate pixels-per-notch used to normalize trackpad `PixelDelta`
/// scroll events into the same `scroll_lines` units as `LineDelta`. winit
/// emits `PixelDelta` in logical pixels on high-precision devices (macOS
/// trackpads, some Wayland compositors); most platforms report one
/// traditional wheel notch as ~50 px.
const SCROLL_PIXELS_PER_LINE: f32 = 50.0;

impl InputState {
    pub fn cursor_moved(&mut self, x: f64, y: f64) {
        let pos = Vec2::new(x as f32, y as f32);
        if let Some(last) = self.last_cursor {
            self.frame.mouse_delta += pos - last;
        }
        self.last_cursor = Some(pos);
    }

    /// Clear the last-known cursor position so the next `cursor_moved`
    /// call re-anchors instead of producing a jump-delta. Call on
    /// `CursorLeft` and on focus loss.
    pub fn cursor_invalidated(&mut self) {
        self.last_cursor = None;
    }

    /// Release any held mouse buttons. Call on focus loss so a drag that
    /// was interrupted by an alt-tab doesn't keep orbiting the camera.
    pub fn release_buttons(&mut self) {
        self.frame.left_mouse_down = false;
    }

    pub fn mouse_input(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Left {
            self.frame.left_mouse_down = state == ElementState::Pressed;
        }
    }

    pub fn mouse_wheel(&mut self, delta: MouseScrollDelta) {
        self.frame.scroll_lines += match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / SCROLL_PIXELS_PER_LINE,
        };
    }

    pub fn take_frame(&mut self) -> FrameInput {
        let frame = self.frame;
        self.frame.mouse_delta = Vec2::ZERO;
        self.frame.scroll_lines = 0.0;
        frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32) {
        assert!(
            (a - b).abs() <= 1e-5,
            "expected {a} to be within 1e-5 of {b}"
        );
    }

    #[test]
    fn starts_at_previous_static_orbit_view() {
        let view = CameraState::default().view();
        assert_close(view.position.x, 3.5);
        assert_close(view.position.y, 0.6);
        assert_close(view.position.z, 0.0);
        assert_close(view.right.x, 0.0);
        assert_close(view.right.y, 0.0);
        assert_close(view.right.z, -1.0);
    }

    #[test]
    fn left_drag_orbits_camera() {
        let mut camera = CameraState::default();
        let before = camera.view();
        camera.advance(FrameInput {
            mouse_delta: Vec2::new(50.0, -20.0),
            left_mouse_down: true,
            ..FrameInput::default()
        });
        let after = camera.view();
        assert_ne!(before.position, after.position);
        assert_close(after.forward.length(), 1.0);
        assert_close(after.right.length(), 1.0);
        assert_close(after.up.length(), 1.0);
    }

    #[test]
    fn wheel_zoom_clamps_distance() {
        let mut camera = CameraState::default();
        camera.advance(FrameInput {
            scroll_lines: 100.0,
            ..FrameInput::default()
        });
        assert_close(camera.distance, MIN_DISTANCE);
        camera.advance(FrameInput {
            scroll_lines: -100.0,
            ..FrameInput::default()
        });
        assert_close(camera.distance, MAX_DISTANCE);
    }

    #[test]
    fn cursor_invalidated_prevents_jump_delta() {
        let mut input = InputState::default();
        input.cursor_moved(100.0, 100.0);
        input.cursor_invalidated();
        input.cursor_moved(500.0, 500.0);
        let frame = input.take_frame();
        assert_eq!(frame.mouse_delta, Vec2::ZERO);
    }

    #[test]
    fn release_buttons_clears_left_drag() {
        let mut input = InputState::default();
        input.mouse_input(MouseButton::Left, ElementState::Pressed);
        input.release_buttons();
        let frame = input.take_frame();
        assert!(!frame.left_mouse_down);
    }

    #[test]
    fn pixel_delta_scroll_uses_pixels_per_line() {
        let mut input = InputState::default();
        input.mouse_wheel(MouseScrollDelta::PixelDelta(
            winit::dpi::PhysicalPosition::new(0.0, SCROLL_PIXELS_PER_LINE as f64),
        ));
        let frame = input.take_frame();
        assert_close(frame.scroll_lines, 1.0);
    }

    #[test]
    fn take_frame_keeps_button_state_and_clears_deltas() {
        let mut input = InputState::default();
        input.mouse_input(MouseButton::Left, ElementState::Pressed);
        input.cursor_moved(10.0, 20.0);
        input.cursor_moved(15.0, 18.0);
        input.mouse_wheel(MouseScrollDelta::LineDelta(0.0, 2.0));

        let frame = input.take_frame();
        assert!(frame.left_mouse_down);
        assert_eq!(frame.mouse_delta, Vec2::new(5.0, -2.0));
        assert_eq!(frame.scroll_lines, 2.0);

        let frame = input.take_frame();
        assert!(frame.left_mouse_down);
        assert_eq!(frame.mouse_delta, Vec2::ZERO);
        assert_eq!(frame.scroll_lines, 0.0);
    }
}
