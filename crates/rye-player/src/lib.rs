use glam::Vec3;
use rye_input::FrameInput;
use rye_math::Space;

/// Space-generic player controller.
///
/// Stores a position in the Space's coordinate system and a yaw angle. Call
/// [`PlayerState::advance`] every tick to move along geodesics driven by WASD,
/// and [`PlayerState::advance_look`] to update yaw from mouse.
///
/// The space type `S` must map `Vec3 → Vec3` (point and tangent vector both live
/// in R³ from the Space's ambient embedding, e.g. Poincaré ball for H³).
pub struct PlayerState<S: Space<Point = Vec3, Vector = Vec3>> {
    pub position: Vec3,
    /// Camera/facing yaw in radians; 0 = −Z (into screen), positive = left.
    pub yaw: f32,
    _space: std::marker::PhantomData<fn() -> S>,
}

impl<S: Space<Point = Vec3, Vector = Vec3>> PlayerState<S> {
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            yaw: 0.0,
            _space: std::marker::PhantomData,
        }
    }

    pub fn with_yaw(mut self, yaw: f32) -> Self {
        self.yaw = yaw;
        self
    }

    /// Move the player along a geodesic for one tick.
    ///
    /// `speed` is in Space-distance units per tick. The tangent direction is
    /// built from `input.move_forward / move_right / move_up` rotated by
    /// `self.yaw`. A zero-length tangent produces no movement.
    pub fn advance(&mut self, input: &FrameInput, space: &S, speed: f32) {
        let sin_y = self.yaw.sin();
        let cos_y = self.yaw.cos();

        // Local basis from yaw only (pitch-independent movement plane).
        let fwd = Vec3::new(-sin_y, 0.0, -cos_y);
        let right = Vec3::new(cos_y, 0.0, -sin_y);
        let up = Vec3::Y;

        let tangent = fwd * input.move_forward + right * input.move_right + up * input.move_up;
        let len2 = tangent.length_squared();
        if len2 < 1e-8 {
            return;
        }
        // Normalize so diagonal movement isn't faster, then scale by speed.
        let t = tangent * (speed / len2.sqrt());
        self.position = space.exp(self.position, t);
    }

    /// Update yaw from mouse delta (mouse sensitivity in radians per pixel).
    pub fn advance_look(&mut self, input: &FrameInput, sensitivity: f32) {
        self.yaw -= input.mouse_delta.x * sensitivity;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;
    use rye_math::EuclideanR3;

    fn assert_close(a: f32, b: f32) {
        assert!((a - b).abs() <= 1e-4, "expected {a} close to {b}");
    }

    #[test]
    fn advance_forward_moves_in_minus_z() {
        let mut player: PlayerState<EuclideanR3> = PlayerState::new(Vec3::ZERO);
        // yaw=0 → forward = −Z
        player.advance(
            &FrameInput { move_forward: 1.0, ..FrameInput::default() },
            &EuclideanR3,
            1.0,
        );
        assert_close(player.position.x, 0.0);
        assert_close(player.position.y, 0.0);
        assert_close(player.position.z, -1.0);
    }

    #[test]
    fn advance_with_zero_input_does_not_move() {
        let mut player: PlayerState<EuclideanR3> = PlayerState::new(Vec3::ZERO);
        player.advance(&FrameInput::default(), &EuclideanR3, 1.0);
        assert_eq!(player.position, Vec3::ZERO);
    }

    #[test]
    fn advance_look_updates_yaw() {
        let mut player: PlayerState<EuclideanR3> = PlayerState::new(Vec3::ZERO);
        player.advance_look(
            &FrameInput { mouse_delta: Vec2::new(100.0, 0.0), ..FrameInput::default() },
            0.002,
        );
        assert_close(player.yaw, -0.2);
    }
}
