use std::f32::consts::FRAC_PI_2;
use std::marker::PhantomData;

use glam::{Quat, Vec3};
use loam_input::FrameInput;
use loam_math::Space;

use crate::Camera;

const ORBIT_RADIANS_PER_PIXEL: f32 = 0.006;
const ZOOM_LOG_STEP: f32 = 0.12;
const MIN_DISTANCE: f32 = 1.5;
// Fog washes out geometry past about 20.
const MAX_DISTANCE: f32 = 20.0;
const INITIAL_HEIGHT: f32 = 0.6;
const INITIAL_RADIUS: f32 = 3.5;
const MIN_ORBIT_PITCH: f32 = -1.45;
const MAX_ORBIT_PITCH: f32 = 1.45;

const FIRST_PERSON_MOUSE_SENSITIVITY: f32 = 0.002;
const FIRST_PERSON_MIN_PITCH: f32 = -FRAC_PI_2 + 0.02;
const FIRST_PERSON_MAX_PITCH: f32 = FRAC_PI_2 - 0.02;

/// `dt` is wall-clock seconds since the previous `advance`; orbit ignores it.
pub trait CameraController<S: Space> {
    fn advance(&mut self, input: FrameInput, camera: &mut Camera<S>, space: &S, dt: f32);
}

/// Remaps the right button onto the left; the left belongs to the scene.
pub fn orbit_on_right(mut input: FrameInput) -> FrameInput {
    input.left_mouse_down = input.buttons.right.down;
    input.buttons.left = input.buttons.right;
    input
}

#[derive(Clone, Copy, Debug)]
pub struct OrbitController<S: Space> {
    pub target: S::Point,
    pub yaw: f32,
    pub pitch: f32,
    /// Geodesic distance from `target` to the camera position.
    pub distance: f32,
    _marker: PhantomData<S>,
}

impl<S: Space<Point = Vec3, Vector = Vec3>> Default for OrbitController<S> {
    fn default() -> Self {
        let distance = (INITIAL_RADIUS * INITIAL_RADIUS + INITIAL_HEIGHT * INITIAL_HEIGHT).sqrt();
        let pitch = -(INITIAL_HEIGHT / distance).asin();
        Self {
            target: Vec3::ZERO,
            yaw: FRAC_PI_2,
            pitch,
            distance,
            _marker: PhantomData,
        }
    }
}

impl<S: Space<Point = Vec3, Vector = Vec3>> OrbitController<S> {
    pub fn around(target: Vec3) -> Self {
        Self {
            target,
            ..Default::default()
        }
    }

    pub fn set_orbit(&mut self, distance: f32, pitch: f32) {
        self.distance = distance.clamp(MIN_DISTANCE, MAX_DISTANCE);
        self.pitch = pitch.clamp(MIN_ORBIT_PITCH, MAX_ORBIT_PITCH);
        self.yaw = 0.0;
    }

    pub fn rotate_yaw(&mut self, delta: f32) {
        self.yaw += delta;
    }

    fn back_at_target(&self) -> Vec3 {
        let yaw_q = Quat::from_rotation_y(self.yaw);
        let pitch_q = Quat::from_rotation_x(self.pitch);
        (yaw_q * pitch_q) * Vec3::Z
    }

    fn right_at_target(&self) -> Vec3 {
        let yaw_q = Quat::from_rotation_y(self.yaw);
        let pitch_q = Quat::from_rotation_x(self.pitch);
        (yaw_q * pitch_q) * Vec3::X
    }

    fn up_at_target(&self) -> Vec3 {
        let yaw_q = Quat::from_rotation_y(self.yaw);
        let pitch_q = Quat::from_rotation_x(self.pitch);
        (yaw_q * pitch_q) * Vec3::Y
    }
}

impl<S: Space<Point = Vec3, Vector = Vec3>> CameraController<S> for OrbitController<S> {
    fn advance(&mut self, input: FrameInput, camera: &mut Camera<S>, space: &S, _dt: f32) {
        if input.left_mouse_down {
            self.yaw -= input.mouse_delta.x * ORBIT_RADIANS_PER_PIXEL;
            self.pitch = (self.pitch - input.mouse_delta.y * ORBIT_RADIANS_PER_PIXEL)
                .clamp(MIN_ORBIT_PITCH, MAX_ORBIT_PITCH);
        }
        if input.scroll_lines != 0.0 {
            self.distance = (self.distance * (-input.scroll_lines * ZOOM_LOG_STEP).exp())
                .clamp(MIN_DISTANCE, MAX_DISTANCE);
        }

        let back_at_target = self.back_at_target();
        let right_at_target = self.right_at_target();
        let up_at_target = self.up_at_target();

        let cam_pos = space.exp(self.target, back_at_target * self.distance);

        let path = [self.target, cam_pos];
        let cam_right = space
            .parallel_transport_along(&path, right_at_target)
            .try_normalize()
            .unwrap_or(Vec3::X);
        let cam_up = space
            .parallel_transport_along(&path, up_at_target)
            .try_normalize()
            .unwrap_or(Vec3::Y);
        let cam_back = space
            .parallel_transport_along(&path, back_at_target)
            .try_normalize()
            .unwrap_or(Vec3::Z);

        camera.position = cam_pos;
        camera.right = cam_right;
        camera.up = cam_up;
        camera.forward = -cam_back;
    }
}

/// Writes only the look direction; the caller owns `position`.
#[derive(Clone, Copy, Debug, Default)]
pub struct FirstPersonController<S: Space> {
    pub yaw: f32,
    pub pitch: f32,
    /// Reads `mouse_raw_delta`, which is uncapped at the screen edge.
    pub use_raw_delta: bool,
    _marker: PhantomData<S>,
}

impl<S: Space<Point = Vec3, Vector = Vec3>> FirstPersonController<S> {
    pub fn new(yaw: f32, pitch: f32) -> Self {
        Self {
            yaw,
            pitch: pitch.clamp(FIRST_PERSON_MIN_PITCH, FIRST_PERSON_MAX_PITCH),
            use_raw_delta: false,
            _marker: PhantomData,
        }
    }
}

impl<S: Space<Point = Vec3, Vector = Vec3>> CameraController<S> for FirstPersonController<S> {
    fn advance(&mut self, input: FrameInput, camera: &mut Camera<S>, _space: &S, _dt: f32) {
        let delta = if self.use_raw_delta {
            input.mouse_raw_delta
        } else {
            input.mouse_delta
        };
        self.yaw -= delta.x * FIRST_PERSON_MOUSE_SENSITIVITY;
        self.pitch = (self.pitch - delta.y * FIRST_PERSON_MOUSE_SENSITIVITY)
            .clamp(FIRST_PERSON_MIN_PITCH, FIRST_PERSON_MAX_PITCH);

        let yaw_q = Quat::from_rotation_y(self.yaw);
        let pitch_q = Quat::from_rotation_x(self.pitch);
        let rot = yaw_q * pitch_q;
        camera.right = rot * Vec3::X;
        camera.up = rot * Vec3::Y;
        camera.forward = rot * -Vec3::Z;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;
    use loam_math::EuclideanR3;

    fn close(a: Vec3, b: Vec3, tol: f32) {
        assert!((a - b).length() < tol, "expected {a:?} ≈ {b:?}");
    }

    #[test]
    fn orbit_default_in_e3_frames_the_origin_from_the_x_axis() {
        let mut camera = Camera::<EuclideanR3>::at_origin();
        let mut ctrl: OrbitController<EuclideanR3> = OrbitController::default();
        ctrl.advance(FrameInput::default(), &mut camera, &EuclideanR3, 0.0);
        close(camera.position, Vec3::new(3.5, 0.6, 0.0), 1e-5);
        // Yaw π/2 sends local +X to -Z; the position hides a swapped axis.
        close(camera.right, Vec3::new(0.0, 0.0, -1.0), 1e-5);
        assert!((camera.right.length() - 1.0).abs() < 1e-5);
        assert!((camera.up.length() - 1.0).abs() < 1e-5);
        assert!((camera.forward.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn orbit_left_drag_moves_camera() {
        let mut camera = Camera::<EuclideanR3>::at_origin();
        let mut ctrl: OrbitController<EuclideanR3> = OrbitController::default();
        ctrl.advance(FrameInput::default(), &mut camera, &EuclideanR3, 0.0);
        let before = camera.position;
        ctrl.advance(
            FrameInput {
                mouse_delta: Vec2::new(50.0, -20.0),
                left_mouse_down: true,
                ..FrameInput::default()
            },
            &mut camera,
            &EuclideanR3,
            0.0,
        );
        assert_ne!(before, camera.position);
        assert!((camera.forward.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn orbit_scroll_clamps_distance() {
        let mut ctrl: OrbitController<EuclideanR3> = OrbitController::default();
        let mut camera = Camera::<EuclideanR3>::at_origin();
        ctrl.advance(
            FrameInput {
                scroll_lines: 100.0,
                ..FrameInput::default()
            },
            &mut camera,
            &EuclideanR3,
            0.0,
        );
        assert!((ctrl.distance - MIN_DISTANCE).abs() < 1e-5);
        ctrl.advance(
            FrameInput {
                scroll_lines: -100.0,
                ..FrameInput::default()
            },
            &mut camera,
            &EuclideanR3,
            0.0,
        );
        assert!((ctrl.distance - MAX_DISTANCE).abs() < 1e-5);
    }

    #[test]
    fn orbit_in_hyperbolic_h3_produces_valid_frame() {
        use loam_math::HyperbolicH3;
        let mut camera = Camera::<HyperbolicH3>::at_origin();
        let mut ctrl: OrbitController<HyperbolicH3> = OrbitController::around(Vec3::ZERO);
        // The Poincaré ball is |p| < 1; the default 3.5 leaves the model.
        ctrl.distance = 0.4;
        ctrl.advance(FrameInput::default(), &mut camera, &HyperbolicH3, 0.0);
        assert!(
            camera.position.length() < 1.0,
            "camera escaped the Poincaré ball: {:?}",
            camera.position
        );
        assert!(camera.right.is_finite() && camera.up.is_finite() && camera.forward.is_finite());
        assert!((camera.right.length() - 1.0).abs() < 1e-3);
        assert!((camera.up.length() - 1.0).abs() < 1e-3);
        assert!((camera.forward.length() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn orbit_in_spherical_s3_produces_valid_frame() {
        use loam_math::SphericalS3;
        let mut camera = Camera::<SphericalS3>::at_origin();
        let mut ctrl: OrbitController<SphericalS3> = OrbitController::around(Vec3::ZERO);
        ctrl.distance = 0.5;
        ctrl.advance(FrameInput::default(), &mut camera, &SphericalS3, 0.0);
        // S³ embeds the upper hemisphere with `|p| < 1`.
        assert!(camera.position.length() < 1.0);
        assert!(camera.right.is_finite() && camera.up.is_finite() && camera.forward.is_finite());
        assert!((camera.right.length() - 1.0).abs() < 1e-3);
        assert!((camera.up.length() - 1.0).abs() < 1e-3);
        assert!((camera.forward.length() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn orbit_in_blended_e3_h3_produces_valid_frame() {
        use loam_math::{BlendedSpace, EuclideanR3, HyperbolicH3, LinearBlendX};
        let space = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-2.0, 2.0));
        let mut camera =
            Camera::<BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX>>::at_origin();
        camera.position = Vec3::new(2.5, 0.0, 0.0);
        let mut ctrl = OrbitController::around(Vec3::new(2.5, 0.0, 0.0));
        ctrl.distance = 0.4;
        ctrl.advance(FrameInput::default(), &mut camera, &space, 0.0);
        assert!(camera.position.is_finite());
        assert!(camera.right.is_finite() && camera.up.is_finite() && camera.forward.is_finite());
        // Looser: the variable-metric integrator has finite step error.
        assert!((camera.right.length() - 1.0).abs() < 1e-2);
        assert!((camera.up.length() - 1.0).abs() < 1e-2);
        assert!((camera.forward.length() - 1.0).abs() < 1e-2);
    }

    #[test]
    fn first_person_pitch_clamps() {
        let mut camera = Camera::<EuclideanR3>::at_origin();
        let mut ctrl: FirstPersonController<EuclideanR3> = FirstPersonController::default();
        ctrl.advance(
            FrameInput {
                mouse_delta: Vec2::new(0.0, 1e9),
                ..FrameInput::default()
            },
            &mut camera,
            &EuclideanR3,
            0.0,
        );
        assert!(ctrl.pitch >= FIRST_PERSON_MIN_PITCH);
        assert!(ctrl.pitch <= FIRST_PERSON_MAX_PITCH);
    }
}
