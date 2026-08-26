//! `Camera::ray_from_ndc` and `Camera::pixels_from_world` must invert each
//! other, through the public API rather than through the shared internals
//! the unit tests can reach.

use glam::{Vec2, Vec3};
use loam_camera::Camera;
use loam_math::{EuclideanR3, HyperbolicH3, Space};

/// The screen convention `pixels_from_world` is contracted to produce: NDC x
/// in [-1, 1] runs left to right, NDC y runs up while pixel y runs down.
fn expected_pixel(ndc: Vec2, viewport: Vec2) -> Vec2 {
    Vec2::new(
        (ndc.x * 0.5 + 0.5) * viewport.x,
        (1.0 - (ndc.y * 0.5 + 0.5)) * viewport.y,
    )
}

#[test]
fn ray_from_ndc_round_trips_through_pixels_from_world() {
    for viewport in [(1600_u32, 900_u32), (720, 1280)] {
        let viewport_px = Vec2::new(viewport.0 as f32, viewport.1 as f32);
        let mut camera = Camera::<EuclideanR3>::looking_at(
            Vec3::new(2.0, 1.0, 4.0),
            Vec3::new(-1.0, 0.5, -2.0),
            Vec3::Y,
            &EuclideanR3,
        );
        camera.fov_y = 47.0_f32.to_radians();
        camera.aspect = viewport_px.x / viewport_px.y;

        // Sampling stops short of ±1: a ray through the exact frustum edge
        // round-trips to |ndc| = 1 plus float error, which the projection
        // rejects as off-screen.
        for x_step in -3..=3 {
            for y_step in -3..=3 {
                let ndc = Vec2::new(x_step as f32 * 0.3, y_step as f32 * 0.3);
                let ray = camera.ray_from_ndc(ndc);
                assert!((ray.direction.length() - 1.0).abs() < 1e-6);
                let expected = expected_pixel(ndc, viewport_px);

                for depth in [0.5_f32, 3.0, 25.0] {
                    let world = ray.origin + ray.direction * depth;
                    let pixel = camera
                        .pixels_from_world(world, viewport, &EuclideanR3)
                        .expect("a point on an in-frustum ray must be visible");
                    assert!(
                        (pixel - expected).length() < 1e-2,
                        "{viewport:?}: ndc {ndc:?} at depth {depth} projected to \
                         {pixel:?}, expected {expected:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn round_trip_holds_across_camera_orientations() {
    let viewport = (1280_u32, 720_u32);
    let viewport_px = Vec2::new(viewport.0 as f32, viewport.1 as f32);
    let ndc_samples = [
        Vec2::ZERO,
        Vec2::new(0.7, 0.4),
        Vec2::new(-0.7, 0.4),
        Vec2::new(0.7, -0.4),
        Vec2::new(-0.7, -0.4),
    ];

    for yaw_step in 0..8 {
        let yaw = yaw_step as f32 * std::f32::consts::FRAC_PI_4;
        let position = Vec3::new(3.0 * yaw.cos(), 1.25, 3.0 * yaw.sin());
        let mut camera =
            Camera::<EuclideanR3>::looking_at(position, Vec3::ZERO, Vec3::Y, &EuclideanR3);
        camera.fov_y = 70.0_f32.to_radians();
        camera.aspect = viewport_px.x / viewport_px.y;

        for ndc in ndc_samples {
            let ray = camera.ray_from_ndc(ndc);
            let world = ray.origin + ray.direction * 2.0;
            let pixel = camera
                .pixels_from_world(world, viewport, &EuclideanR3)
                .expect("a point on an in-frustum ray must be visible");
            let expected = expected_pixel(ndc, viewport_px);
            assert!(
                (pixel - expected).length() < 1e-2,
                "yaw {yaw}: ndc {ndc:?} projected to {pixel:?}, expected {expected:?}"
            );
        }
    }
}

#[test]
fn round_trip_holds_in_hyperbolic_h3() {
    let space = HyperbolicH3;
    let viewport = (1280_u32, 720_u32);
    let viewport_px = Vec2::new(viewport.0 as f32, viewport.1 as f32);
    let mut camera = Camera::<HyperbolicH3>::looking_at(
        Vec3::new(0.31, 0.14, 0.42),
        Vec3::ZERO,
        Vec3::Y,
        &space,
    );
    camera.fov_y = 70.0_f32.to_radians();
    camera.aspect = viewport_px.x / viewport_px.y;
    camera.near = 1e-5;

    for x_step in -2..=2 {
        for y_step in -2..=2 {
            let ndc = Vec2::new(x_step as f32 * 0.35, y_step as f32 * 0.35);
            let ray = camera.ray_from_ndc(ndc);
            let expected = expected_pixel(ndc, viewport_px);

            for travel in [0.1_f32, 0.4, 0.9] {
                let world = space.exp(ray.origin, ray.direction * travel);
                let pixel = camera
                    .pixels_from_world(world, viewport, &space)
                    .expect("a point on an in-frustum geodesic must be visible");
                assert!(
                    (pixel - expected).length() < 1e-2,
                    "ndc {ndc:?} at travel {travel} projected to {pixel:?}, \
                     expected {expected:?}"
                );
            }
        }
    }
}
