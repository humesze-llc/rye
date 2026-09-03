//! On wasm `cursor::request_grab` is a no-op (Pointer Lock needs a user
//! gesture), so browser freecam needs a click-to-engage layer on the main thread.

use glam::Vec3;
use loam_camera::{CameraController, FirstPersonController};
use loam_input::FrameInput;
use loam_math::EuclideanR3;

use crate::Camera;

const DEFAULT_SPEED: f32 = 4.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CursorMode {
    /// MMO-style: cursor released while Alt is held, re-grabbed on release.
    Hold,
    /// FPS sticky-modifier: Alt press flips the grab, release is ignored.
    #[default]
    Toggle,
}

impl CursorMode {
    pub fn from_token(s: &str) -> Option<Self> {
        match s {
            "hold" => Some(Self::Hold),
            "toggle" => Some(Self::Toggle),
            _ => None,
        }
    }

    pub fn token(self) -> &'static str {
        match self {
            Self::Hold => "hold",
            Self::Toggle => "toggle",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Freecam {
    pub controller: FirstPersonController<EuclideanR3>,
    pub position: Vec3,
    /// Units per second; look sensitivity lives on `controller`.
    pub speed: f32,
    active: bool,
    cursor_grabbed: bool,
    cursor_mode: CursorMode,
}

impl Default for Freecam {
    fn default() -> Self {
        Self::new()
    }
}

impl Freecam {
    pub fn new() -> Self {
        Self {
            controller: FirstPersonController::<EuclideanR3>::new(0.0, 0.0),
            position: Vec3::ZERO,
            speed: DEFAULT_SPEED,
            active: false,
            cursor_grabbed: false,
            cursor_mode: CursorMode::default(),
        }
    }

    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    pub fn with_cursor_mode(mut self, mode: CursorMode) -> Self {
        self.cursor_mode = mode;
        self
    }

    pub fn cursor_mode(&self) -> CursorMode {
        self.cursor_mode
    }

    /// Does not touch the current grab, only how future Alt events are read.
    pub fn set_cursor_mode(&mut self, mode: CursorMode) {
        self.cursor_mode = mode;
    }

    pub fn active(&self) -> bool {
        self.active
    }

    /// Always false when inactive.
    pub fn cursor_grabbed(&self) -> bool {
        self.cursor_grabbed
    }

    pub fn set_active(&mut self, active: bool, current_camera_pos: Vec3) {
        if active == self.active {
            return;
        }
        self.active = active;
        if active {
            self.position = current_camera_pos;
            self.cursor_grabbed = true;
            self.controller.use_raw_delta = true;
            crate::cursor::request_grab();
        } else {
            self.cursor_grabbed = false;
            self.controller.use_raw_delta = false;
            crate::cursor::request_release();
        }
    }

    pub fn on_alt(&mut self, pressed: bool) {
        if !self.active {
            return;
        }
        let target_grabbed = match self.cursor_mode {
            CursorMode::Hold => !pressed,
            CursorMode::Toggle => {
                if !pressed {
                    return;
                }
                !self.cursor_grabbed
            }
        };
        if target_grabbed == self.cursor_grabbed {
            return;
        }
        self.cursor_grabbed = target_grabbed;
        self.controller.use_raw_delta = target_grabbed;
        if target_grabbed {
            crate::cursor::request_grab();
        } else {
            crate::cursor::request_release();
            crate::cursor::request_warp_to_center();
        }
    }

    /// Ignores [`cursor_mode`](Self::cursor_mode); prefer [`on_alt`](Self::on_alt).
    pub fn toggle_cursor_grab(&mut self) {
        if !self.active {
            return;
        }
        self.cursor_grabbed = !self.cursor_grabbed;
        self.controller.use_raw_delta = self.cursor_grabbed;
        if self.cursor_grabbed {
            crate::cursor::request_grab();
        } else {
            crate::cursor::request_release();
            crate::cursor::request_warp_to_center();
        }
    }

    /// No-op when inactive; look freezes when the cursor is released.
    pub fn advance(&mut self, input: FrameInput, camera: &mut Camera<EuclideanR3>, dt: f32) {
        if !self.active {
            return;
        }
        if self.cursor_grabbed {
            self.controller.advance(input, camera, &EuclideanR3, dt);
        }
        // On wasm `cursor_grabbed` is always false, so translation cannot gate on it.
        let mut delta = camera.forward * input.move_forward
            + camera.right * input.move_right
            + Vec3::Y * input.move_up;
        if delta.length_squared() > 1e-6 {
            delta = delta.normalize();
            self.position += delta * self.speed * dt;
            camera.position = self.position;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cursor;

    fn make_camera() -> Camera<EuclideanR3> {
        let mut c = Camera::<EuclideanR3>::at_origin();
        c.position = Vec3::new(0.0, 1.0, 5.0);
        c.forward = -Vec3::Z;
        c.right = Vec3::X;
        c.up = Vec3::Y;
        c
    }

    #[test]
    fn set_active_true_seeds_position_and_grabs() {
        let _ = cursor::take_pending();
        let mut f = Freecam::new();
        let cam_pos = Vec3::new(3.0, 2.0, 7.0);
        f.set_active(true, cam_pos);
        assert!(f.active());
        assert!(f.cursor_grabbed());
        assert!(f.controller.use_raw_delta);
        assert_eq!(f.position, cam_pos);
        let (grab, vis) = cursor::take_pending();
        assert_eq!(grab, Some(cursor::GrabMode::Locked));
        assert_eq!(vis, Some(false));
    }

    #[test]
    fn set_active_false_releases() {
        let _ = cursor::take_pending();
        let mut f = Freecam::new();
        f.set_active(true, Vec3::ZERO);
        let _ = cursor::take_pending();
        f.set_active(false, Vec3::ZERO);
        assert!(!f.active());
        assert!(!f.cursor_grabbed());
        assert!(!f.controller.use_raw_delta);
        let (grab, vis) = cursor::take_pending();
        assert_eq!(grab, Some(cursor::GrabMode::None));
        assert_eq!(vis, Some(true));
    }

    #[test]
    fn set_active_idempotent() {
        let _ = cursor::take_pending();
        let mut f = Freecam::new();
        f.set_active(true, Vec3::ZERO);
        let _ = cursor::take_pending();
        f.set_active(true, Vec3::new(99.0, 99.0, 99.0));
        let (grab, vis) = cursor::take_pending();
        assert_eq!(grab, None, "idempotent set_active shouldn't re-queue");
        assert_eq!(vis, None);
        assert_eq!(f.position, Vec3::ZERO);
    }

    #[test]
    fn toggle_cursor_grab_flips_within_active() {
        let _ = cursor::take_pending();
        let mut f = Freecam::new();
        f.set_active(true, Vec3::ZERO);
        let _ = cursor::take_pending();

        f.toggle_cursor_grab();
        assert!(f.active(), "toggle doesn't change active");
        assert!(!f.cursor_grabbed());
        assert!(!f.controller.use_raw_delta);
        let (grab, vis) = cursor::take_pending();
        assert_eq!(grab, Some(cursor::GrabMode::None));
        assert_eq!(vis, Some(true));

        f.toggle_cursor_grab();
        assert!(f.cursor_grabbed());
        assert!(f.controller.use_raw_delta);
    }

    #[test]
    fn toggle_cursor_grab_noop_when_inactive() {
        let _ = cursor::take_pending();
        let mut f = Freecam::new();
        f.toggle_cursor_grab();
        assert!(!f.active());
        assert!(!f.cursor_grabbed());
        let (grab, vis) = cursor::take_pending();
        assert_eq!(grab, None);
        assert_eq!(vis, None);
    }

    #[test]
    fn advance_noop_when_inactive() {
        let mut f = Freecam::new();
        let mut cam = make_camera();
        let cam_before = cam;
        let pos_before = f.position;
        let input = FrameInput {
            mouse_raw_delta: glam::Vec2::new(100.0, 50.0),
            move_forward: 1.0,
            ..Default::default()
        };
        f.advance(input, &mut cam, 0.016);
        assert_eq!(cam.position, cam_before.position);
        assert_eq!(cam.forward, cam_before.forward);
        assert_eq!(f.position, pos_before);
    }

    #[test]
    fn advance_with_cursor_released_freezes_look_but_keeps_wasd() {
        let mut f = Freecam::new();
        f.set_active(true, Vec3::ZERO);
        f.toggle_cursor_grab();
        let mut cam = make_camera();
        let cam_before = cam;
        let input = FrameInput {
            mouse_raw_delta: glam::Vec2::new(100.0, 50.0),
            move_forward: 1.0,
            ..Default::default()
        };
        f.advance(input, &mut cam, 0.016);
        assert_eq!(
            cam.forward, cam_before.forward,
            "look frozen when cursor released"
        );
        assert_ne!(
            cam.position, cam_before.position,
            "WASD continues to translate when cursor released (matches wasm where Pointer Lock never engages)"
        );
    }

    #[test]
    fn on_alt_hold_releases_and_regrabs() {
        let _ = cursor::take_pending();
        let mut f = Freecam::new().with_cursor_mode(CursorMode::Hold);
        f.set_active(true, Vec3::ZERO);
        let _ = cursor::take_pending();

        f.on_alt(true);
        assert!(f.active());
        assert!(!f.cursor_grabbed());
        let (grab, _vis) = cursor::take_pending();
        assert_eq!(grab, Some(cursor::GrabMode::None));

        f.on_alt(false);
        assert!(f.cursor_grabbed());
        let (grab, _vis) = cursor::take_pending();
        assert_eq!(grab, Some(cursor::GrabMode::Locked));
    }

    #[test]
    fn on_alt_toggle_ignores_release() {
        let _ = cursor::take_pending();
        let mut f = Freecam::new().with_cursor_mode(CursorMode::Toggle);
        f.set_active(true, Vec3::ZERO);
        let _ = cursor::take_pending();

        f.on_alt(true);
        assert!(!f.cursor_grabbed());
        let _ = cursor::take_pending();

        f.on_alt(false);
        assert!(!f.cursor_grabbed());
        let (grab, vis) = cursor::take_pending();
        assert_eq!(grab, None);
        assert_eq!(vis, None);

        f.on_alt(true);
        assert!(f.cursor_grabbed());
    }

    #[test]
    fn on_alt_noop_when_inactive() {
        let _ = cursor::take_pending();
        let mut f = Freecam::new();
        f.on_alt(true);
        assert!(!f.active());
        let (grab, vis) = cursor::take_pending();
        assert_eq!(grab, None);
        assert_eq!(vis, None);
    }

    #[test]
    fn set_cursor_mode_preserves_grab_state() {
        let _ = cursor::take_pending();
        let mut f = Freecam::new();
        f.set_active(true, Vec3::ZERO);
        let _ = cursor::take_pending();
        let grabbed_before = f.cursor_grabbed();

        f.set_cursor_mode(CursorMode::Toggle);
        assert_eq!(f.cursor_mode(), CursorMode::Toggle);
        assert_eq!(
            f.cursor_grabbed(),
            grabbed_before,
            "mode flip leaves grab alone"
        );
        let (grab, vis) = cursor::take_pending();
        assert_eq!(grab, None);
        assert_eq!(vis, None);
    }

    #[test]
    fn advance_integrates_forward_motion() {
        let _ = cursor::take_pending();
        let mut f = Freecam::new().with_speed(2.0);
        f.set_active(true, Vec3::ZERO);
        let mut cam = make_camera();
        cam.position = Vec3::ZERO;
        cam.forward = -Vec3::Z;
        let input = FrameInput {
            move_forward: 1.0,
            ..Default::default()
        };
        f.advance(input, &mut cam, 0.5);
        // speed * dt = 2.0 * 0.5 = 1.0 unit, in -Z direction.
        let expected = Vec3::new(0.0, 0.0, -1.0);
        assert!(
            (f.position - expected).length() < 1e-5,
            "expected {expected:?}, got {:?}",
            f.position,
        );
        assert!((cam.position - expected).length() < 1e-5);
    }
}
