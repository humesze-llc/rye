//! The projection is Space-generic and lives in `loam-camera` so an editor
//! can anchor to it without taking on egui; this is only the type adapter.

use glam::Vec3;
use loam_camera::Camera;
use loam_math::Space;

/// `viewport` is `(width, height)` in egui points, the units the returned
/// position is in. `None` when the point is outside the frustum, so an
/// anchored widget draws nothing.
pub fn world_to_screen<S: Space<Point = Vec3, Vector = Vec3>>(
    camera: &Camera<S>,
    world: Vec3,
    viewport: (u32, u32),
    space: &S,
) -> Option<egui::Pos2> {
    let pixels = camera.pixels_from_world(world, viewport, space)?;
    Some(egui::Pos2::new(pixels.x, pixels.y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_math::EuclideanR3;

    #[test]
    fn screen_position_carries_the_camera_pixels_unswapped() {
        let viewport = (800_u32, 600_u32);
        let mut camera = Camera::<EuclideanR3>::looking_at(
            Vec3::new(0.0, 0.0, 6.0),
            Vec3::ZERO,
            Vec3::Y,
            &EuclideanR3,
        );
        camera.aspect = viewport.0 as f32 / viewport.1 as f32;

        // Off both axes, and asymmetric in x and y, so a swap is visible.
        let world = Vec3::new(0.9, -0.4, 0.0);
        let pixels = camera
            .pixels_from_world(world, viewport, &EuclideanR3)
            .expect("sample is in view");
        let screen = world_to_screen(&camera, world, viewport, &EuclideanR3)
            .expect("the adapter must agree on visibility");
        assert_eq!((screen.x, screen.y), (pixels.x, pixels.y));
    }

    #[test]
    fn a_point_behind_the_eye_has_no_screen_position() {
        let camera = Camera::<EuclideanR3>::at_origin();
        let behind = Vec3::new(0.0, 0.0, 5.0);
        assert!(world_to_screen(&camera, behind, (800, 600), &EuclideanR3).is_none());
    }
}
