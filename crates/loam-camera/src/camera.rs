//! Space-generic camera state: a position on the manifold plus an
//! orthonormal frame in its tangent space.
//!
//! [`Camera<S>`] stores a `position: S::Point` and three tangent basis
//! vectors (`right`, `up`, `forward`); the frame is the orientation,
//! there is no separate rotation field. This orthonormal-frame-bundle
//! form avoids the per-Space convention questions of the `Iso` types:
//! [`loam_math::Space::parallel_transport_along`] moves the frame
//! correctly for any Space.
//!
//! [`Camera::view`] yields a [`CameraView`] for direct shader upload,
//! available where `S::Point` and `S::Vector` are both `glam::Vec3`.

use glam::{Vec2, Vec3};
use loam_math::Space;
use std::ops::Mul;

use crate::CameraView;

/// A geodesic ray: a point on the manifold plus the initial velocity of
/// the geodesic leaving it. `direction` is Euclidean-unit in the Space's
/// embedding, matching [`Camera`]'s frame convention, so
/// `Space::exp(origin, direction * t)` walks the ray.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

/// Position + orthonormal tangent frame at that position. Generic over
/// any [`Space`]; `view` and `translate` require `S::Point = S::Vector =
/// Vec3`.
///
/// ## Invariants (caller-maintained, not type-enforced)
///
/// - `right`, `up`, `forward` are pairwise-orthogonal Euclidean-unit
///   vectors in the Space's embedding; the WGSL prelude applies the
///   metric, so storing Riemannian-unit vectors would leak embedding
///   scale factors into the renderer.
/// - Right-handed: `forward` is the look direction, so `right × up =
///   -forward`. Matches the WGSL prelude.
/// - Construct via [`Camera::looking_at`] or a
///   [`crate::CameraController`]; mutating the basis by hand drifts off
///   orthonormal under `translate`.
#[derive(Clone, Copy, Debug)]
pub struct Camera<S: Space> {
    pub position: S::Point,
    pub right: S::Vector,
    pub up: S::Vector,
    pub forward: S::Vector,
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl<S: Space<Point = Vec3, Vector = Vec3>> Camera<S> {
    /// Origin camera looking down −Z, 60° vertical FOV, near/far for
    /// unit-scale scenes. A `Default` stand-in for `S` instances that
    /// can't derive it; controllers usually overwrite the pose at once.
    pub fn at_origin() -> Self {
        Self {
            position: Vec3::ZERO,
            right: Vec3::X,
            up: Vec3::Y,
            forward: -Vec3::Z,
            fov_y: 60.0_f32.to_radians(),
            aspect: 1.0,
            near: 0.05,
            far: 100.0,
        }
    }

    /// Place the camera at `position` looking toward `target`, with
    /// `world_up` as the up hint. Forward is the normalised
    /// `Space::log(position, target)`: the initial geodesic velocity from
    /// `position` to `target`.
    pub fn looking_at(position: Vec3, target: Vec3, world_up: Vec3, space: &S) -> Self {
        let log = space.log(position, target);
        let forward = log.try_normalize().unwrap_or(-Vec3::Z);
        // Right-handed: right = forward × world_up, up = right × forward.
        let right = forward.cross(world_up).try_normalize().unwrap_or(Vec3::X);
        let up = right.cross(forward);
        Self {
            position,
            right,
            up,
            forward,
            fov_y: 60.0_f32.to_radians(),
            aspect: 1.0,
            near: 0.05,
            far: 100.0,
        }
    }

    /// Renderer-facing view basis, in the Space's embedding coordinates
    /// the shader expects (the WGSL prelude applies the metric).
    pub fn view(&self) -> CameraView {
        CameraView {
            position: self.position,
            forward: self.forward,
            right: self.right,
            up: self.up,
        }
    }

    /// Inverse of the perspective projection: the primary ray through the
    /// normalised device coordinate `ndc`, both components in [-1, 1] with
    /// y up. Pixel-space callers must flip y, since window coordinates are
    /// y-down.
    ///
    /// A point at depth `d` along `forward` projects to
    /// `ndc.x = x / (aspect · tan(fov_y/2) · d)` and
    /// `ndc.y = y / (tan(fov_y/2) · d)` under a right-handed perspective
    /// matrix (Akenine-Möller, Haines, Hoffman, *Real-Time Rendering* 4th
    /// ed, 2018, §4.7); solving for the view-space offsets and dropping the
    /// depth scale gives the direction below. The raymarch shaders build
    /// their primary rays from the same three coefficients.
    ///
    /// `ndc` outside [-1, 1] is meaningful and returns the ray through that
    /// off-screen point.
    pub fn ray_from_ndc(&self, ndc: Vec2) -> Ray {
        let tan_half_fov_y = (self.fov_y * 0.5).tan();
        let direction = self.forward
            + self.right * (ndc.x * self.aspect * tan_half_fov_y)
            + self.up * (ndc.y * tan_half_fov_y);
        Ray {
            origin: self.position,
            // The lateral terms are orthogonal to the unit `forward`, so
            // |direction|² = 1 + |lateral|² ≥ 1: normalize cannot underflow
            // for any finite `ndc`, and needs no fallback.
            direction: direction.normalize(),
        }
    }

    /// Where `world` lands in normalised device coordinates, y up: the
    /// inverse of [`Self::ray_from_ndc`]. `None` when the point is outside
    /// the frustum, which is a caller anchoring UI to it drawing nothing.
    ///
    /// An image in a curved Space is formed by the geodesics through the eye
    /// (Gunn, *Discrete Groups and Visualization of Three-Dimensional
    /// Manifolds*, SIGGRAPH 1993, §3), which is the ray `ray_from_ndc`
    /// builds and the raymarch preludes walk with `Space::exp`. The geodesic
    /// reaching `world` leaves along `Space::log(position, world)`, so that
    /// vector stands where the straight offset `world - position` would in
    /// the flat case; the two coincide when the chart is flat.
    ///
    /// Only its direction matters: the frame components enter as a ratio, so
    /// the embedding scale factor a conformal model's `log` carries cancels.
    /// `near` and `far` clip its forward component, the same
    /// embedding-frame depth the primary ray advances in.
    pub fn ndc_from_world(&self, world: Vec3, space: &S) -> Option<Vec2> {
        let to_target = space.log(self.position, world);
        let depth = to_target.dot(self.forward);
        // `near` is a public field a caller may zero to mean "no near clip".
        // The sign test is then what keeps the point mirrored behind the eye,
        // whose frame ratios are identical, from being reported as visible.
        if depth <= 0.0 || depth < self.near || depth > self.far {
            return None;
        }
        let tan_half_fov_y = (self.fov_y * 0.5).tan();
        let ndc = Vec2::new(
            to_target.dot(self.right) / (depth * self.aspect * tan_half_fov_y),
            to_target.dot(self.up) / (depth * tan_half_fov_y),
        );
        // Also the NaN gate: a non-finite log fails both bounds.
        (ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0).then_some(ndc)
    }

    /// [`Self::ndc_from_world`] in window pixels, top-left origin and y down.
    /// `viewport` is `(width, height)` in those same pixels and sets the
    /// scale only: the framing comes from `self.aspect`, so a caller whose
    /// viewport ratio has drifted from it gets the camera's framing, not the
    /// window's.
    pub fn pixels_from_world(&self, world: Vec3, viewport: (u32, u32), space: &S) -> Option<Vec2> {
        if viewport.0 == 0 || viewport.1 == 0 {
            return None;
        }
        let ndc = self.ndc_from_world(world, space)?;
        Some(Vec2::new(
            (ndc.x * 0.5 + 0.5) * viewport.0 as f32,
            (1.0 - (ndc.y * 0.5 + 0.5)) * viewport.1 as f32,
        ))
    }

    /// Move along the geodesic from `v * dt`, parallel-transporting the
    /// frame so it stays orthonormal at the new point. Identity on the
    /// basis in flat space; a holonomy rotation in H³ / S³.
    pub fn translate(&mut self, v: S::Vector, dt: f32, space: &S)
    where
        S::Vector: Mul<f32, Output = S::Vector>,
    {
        let new_pos = space.exp(self.position, v * dt);
        // Re-normalise to Euclidean-unit: transport preserves Riemannian
        // length, but the Poincaré-ball / S³ embedding scales Euclidean
        // length by a position-dependent factor, and the renderer expects
        // Euclidean-unit directions.
        let path = [self.position, new_pos];
        self.right = space
            .parallel_transport_along(&path, self.right)
            .try_normalize()
            .unwrap_or(self.right);
        self.up = space
            .parallel_transport_along(&path, self.up)
            .try_normalize()
            .unwrap_or(self.up);
        self.forward = space
            .parallel_transport_along(&path, self.forward)
            .try_normalize()
            .unwrap_or(self.forward);
        self.position = new_pos;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Vec4, Vec4Swizzles};
    use loam_math::{EuclideanR3, HyperbolicH3, SphericalS3};

    fn close(a: Vec3, b: Vec3, tol: f32) {
        assert!((a - b).length() < tol, "expected {a:?} ≈ {b:?}");
    }

    /// The pipeline the flat callers projected through before
    /// [`Camera::ndc_from_world`] existed. `camera.aspect` stands in for the
    /// ratio the old free function derived from its viewport argument.
    fn matrix_clip(camera: &Camera<EuclideanR3>, world: Vec3) -> Vec4 {
        let view = Mat4::look_to_rh(camera.position, camera.forward, camera.up);
        let proj = Mat4::perspective_rh(camera.fov_y, camera.aspect, camera.near, camera.far);
        proj * view * world.extend(1.0)
    }

    /// Every visible sample is on the geodesic the renderer would march down
    /// its own pixel: unproject the NDC, walk `exp` for the length of the
    /// log, arrive at the point. Pinned against the Space's own exp/log, not
    /// against a second implementation of the projection.
    fn assert_anchors_sit_on_their_own_geodesic<S>(space: &S, eye: Vec3, samples: &[Vec3], tol: f32)
    where
        S: Space<Point = Vec3, Vector = Vec3>,
    {
        let mut camera = Camera::<S>::looking_at(eye, Vec3::ZERO, Vec3::Y, space);
        camera.fov_y = 70_f32.to_radians();
        camera.aspect = 16.0 / 9.0;
        // The frustum depth is in embedding units, which a conformal chart
        // shrinks near the boundary; keep the clip out of this test's way.
        camera.near = 1e-5;
        let mut visible = 0;
        for &world in samples {
            let Some(ndc) = camera.ndc_from_world(world, space) else {
                continue;
            };
            let ray = camera.ray_from_ndc(ndc);
            let travel = space.log(camera.position, world).length();
            let arrival = space.exp(ray.origin, ray.direction * travel);
            let miss = space.distance(arrival, world);
            assert!(
                miss < tol,
                "{world:?} at ndc {ndc:?}: its own pixel's geodesic misses it by {miss}"
            );
            visible += 1;
        }
        assert!(
            visible >= samples.len() / 2,
            "only {visible} of {} samples were in view; the test proves little",
            samples.len()
        );
    }

    #[test]
    fn at_origin_is_orthonormal() {
        let cam = Camera::<EuclideanR3>::at_origin();
        assert!((cam.right.length() - 1.0).abs() < 1e-6);
        assert!((cam.up.length() - 1.0).abs() < 1e-6);
        assert!((cam.forward.length() - 1.0).abs() < 1e-6);
        // Right-handed: right × up = -forward.
        close(cam.right.cross(cam.up), -cam.forward, 1e-6);
        assert!(cam.right.dot(cam.up).abs() < 1e-6);
        assert!(cam.right.dot(cam.forward).abs() < 1e-6);
        assert!(cam.up.dot(cam.forward).abs() < 1e-6);
    }

    #[test]
    fn looking_at_target_points_toward_it() {
        let space = EuclideanR3;
        let cam = Camera::<EuclideanR3>::looking_at(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ZERO,
            Vec3::Y,
            &space,
        );
        close(cam.forward, -Vec3::Z, 1e-6);
        close(cam.up, Vec3::Y, 1e-6);
    }

    #[test]
    fn translate_in_flat_space_preserves_frame() {
        let space = EuclideanR3;
        let mut cam = Camera::<EuclideanR3>::at_origin();
        let original = cam.view();
        cam.translate(Vec3::new(1.0, 2.0, -3.0), 1.0, &space);
        close(cam.position, Vec3::new(1.0, 2.0, -3.0), 1e-6);
        // Transport is the identity in E³.
        close(cam.right, original.right, 1e-6);
        close(cam.up, original.up, 1e-6);
        close(cam.forward, original.forward, 1e-6);
    }

    #[test]
    fn translate_view_position_matches_exp() {
        let space = EuclideanR3;
        let mut cam = Camera::<EuclideanR3>::at_origin();
        cam.translate(Vec3::X, 2.5, &space);
        close(cam.view().position, Vec3::new(2.5, 0.0, 0.0), 1e-6);
    }

    /// `translate` in H³ keeps the position in the Poincaré ball and the
    /// basis finite + ≈orthonormal. Catches NaN drift in `Space::exp` /
    /// `parallel_transport_along` for `HyperbolicH3`.
    #[test]
    fn translate_in_hyperbolic_h3_stays_in_ball_and_orthonormal() {
        use loam_math::HyperbolicH3;
        let space = HyperbolicH3;
        let mut cam = Camera::<HyperbolicH3>::at_origin();
        cam.translate(Vec3::X, 0.5, &space);
        assert!(
            cam.position.length() < 1.0,
            "camera escaped Poincaré ball: {:?}",
            cam.position
        );
        assert!(cam.right.is_finite());
        assert!(cam.up.is_finite());
        assert!(cam.forward.is_finite());
        assert!((cam.right.length() - 1.0).abs() < 1e-3);
        assert!((cam.up.length() - 1.0).abs() < 1e-3);
        assert!((cam.forward.length() - 1.0).abs() < 1e-3);
        assert!(cam.right.dot(cam.up).abs() < 1e-3);
        assert!(cam.right.dot(cam.forward).abs() < 1e-3);
        assert!(cam.up.dot(cam.forward).abs() < 1e-3);
    }

    /// The centre of the screen looks where the camera looks, from where
    /// the camera is, for any fov and any aspect.
    #[test]
    fn centre_ndc_ray_is_camera_position_and_forward() {
        for (fov_y_degrees, aspect) in [(60.0_f32, 1.0_f32), (30.0, 16.0 / 9.0), (95.0, 0.5)] {
            let mut camera = Camera::<EuclideanR3>::looking_at(
                Vec3::new(-3.0, 2.0, 1.5),
                Vec3::new(0.4, -1.0, -2.0),
                Vec3::Y,
                &EuclideanR3,
            );
            camera.fov_y = fov_y_degrees.to_radians();
            camera.aspect = aspect;
            let ray = camera.ray_from_ndc(Vec2::ZERO);
            close(ray.origin, camera.position, 1e-6);
            close(ray.direction, camera.forward, 1e-6);
        }
    }

    /// Edge rays sit on the frustum half-angles: the vertical one is
    /// `fov_y/2` whatever the aspect, the horizontal one satisfies
    /// `tan θ = aspect · tan(fov_y/2)`. Catches a dropped `aspect`, an
    /// `aspect` applied to the vertical axis, and `fov_y` used where
    /// `fov_y/2` belongs.
    #[test]
    fn edge_ndc_half_angles_track_fov_y_and_aspect() {
        for aspect in [16.0_f32 / 9.0, 1.0, 9.0 / 16.0] {
            let mut camera = Camera::<EuclideanR3>::at_origin();
            camera.fov_y = 42.0_f32.to_radians();
            camera.aspect = aspect;
            let tan_half_fov_y = (camera.fov_y * 0.5).tan();

            // Ratio of frame components rather than acos(dot): the tangent
            // is exactly the projection coefficient being pinned, and acos
            // loses precision on the near-parallel rays this samples.
            let top = camera.ray_from_ndc(Vec2::new(0.0, 1.0)).direction;
            let vertical_tan = top.dot(camera.up) / top.dot(camera.forward);
            assert!(
                (vertical_tan - tan_half_fov_y).abs() < 1e-6,
                "aspect {aspect}: vertical tan {vertical_tan} != {tan_half_fov_y}"
            );
            assert!(top.dot(camera.right).abs() < 1e-6, "aspect leaked into y");

            let side = camera.ray_from_ndc(Vec2::new(1.0, 0.0)).direction;
            let horizontal_tan = side.dot(camera.right) / side.dot(camera.forward);
            let expected = aspect * tan_half_fov_y;
            assert!(
                (horizontal_tan - expected).abs() < 1e-6,
                "aspect {aspect}: horizontal tan {horizontal_tan} != {expected}"
            );
            assert!(side.dot(camera.up).abs() < 1e-6, "x NDC tilted the y axis");
        }
    }

    /// `looking_at` with `position == target` has `log = 0`, no defined
    /// forward; the fallback must stay finite and orthonormal.
    #[test]
    fn looking_at_collapsed_target_falls_back_to_finite_frame() {
        let cam = Camera::<EuclideanR3>::looking_at(Vec3::ZERO, Vec3::ZERO, Vec3::Y, &EuclideanR3);
        assert!(cam.forward.is_finite() && cam.right.is_finite() && cam.up.is_finite());
        // Direction is unspecified under fallback; only length is pinned.
        assert!((cam.forward.length() - 1.0).abs() < 1e-6);
        assert!((cam.right.length() - 1.0).abs() < 1e-6);
        assert!((cam.up.length() - 1.0).abs() < 1e-6);
    }

    /// `looking_at` with `forward` parallel to `world_up` makes
    /// `forward × world_up = 0`; the fallback `right` keeps the frame
    /// finite.
    #[test]
    fn looking_at_world_up_parallel_to_forward_falls_back() {
        let cam = Camera::<EuclideanR3>::looking_at(
            Vec3::new(0.0, 0.0, 0.0),
            // Look straight up, parallel to world_up +Y.
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::Y,
            &EuclideanR3,
        );
        assert!(cam.forward.is_finite() && cam.right.is_finite() && cam.up.is_finite());
        assert!((cam.right.length() - 1.0).abs() < 1e-6);
    }

    /// In E³ the geodesic projection is the view-projection pipeline it
    /// replaces: the same pixel wherever both produce one, and the same
    /// rejection wherever either does not. Pins the screen position of every
    /// world-anchored widget across the move off `Mat4::perspective_rh`. The
    /// second frustum is the one the playground's callouts and throw aim ran
    /// through before the move.
    #[test]
    fn flat_projection_agrees_with_the_view_projection_matrix() {
        let viewport = (1600_u32, 900_u32);
        let (vw, vh) = (viewport.0 as f32, viewport.1 as f32);
        let (mut on_screen, mut clipped) = (0_u32, 0_u32);

        for (position, target, fov_y_degrees, near, far) in [
            (
                Vec3::new(0.0, 0.0, 6.0),
                Vec3::ZERO,
                55_f32,
                0.05_f32,
                100.0_f32,
            ),
            (
                Vec3::new(-4.0, 2.5, 3.0),
                Vec3::new(0.5, -0.5, 0.0),
                60.0,
                0.1,
                100.0,
            ),
            (
                Vec3::new(1.0, -3.0, -5.0),
                Vec3::new(-2.0, 1.0, 2.0),
                60.0,
                0.1,
                12.0,
            ),
        ] {
            let mut camera =
                Camera::<EuclideanR3>::looking_at(position, target, Vec3::Y, &EuclideanR3);
            camera.fov_y = fov_y_degrees.to_radians();
            camera.near = near;
            camera.far = far;
            camera.aspect = vw / vh;

            for xi in -4..=4 {
                for yi in -4..=4 {
                    for zi in -4..=4 {
                        let world =
                            Vec3::new(xi as f32 * 1.7, yi as f32 * 1.3, zi as f32 * 2.1) + target;
                        let clip = matrix_clip(&camera, world);
                        let ndc = clip.xyz() / clip.w;
                        // A sample astride a clip plane is decided by float op
                        // order, and the two forms multiply in different
                        // orders by construction. Disagreement there is not
                        // the behaviour under test.
                        let astride_a_clip_plane = clip.w.abs() < 1e-3
                            || (ndc.x.abs() - 1.0).abs() < 1e-3
                            || (ndc.y.abs() - 1.0).abs() < 1e-3
                            || ndc.z.abs() < 1e-3
                            || (ndc.z - 1.0).abs() < 1e-3;
                        if astride_a_clip_plane {
                            continue;
                        }

                        let expected = (clip.w > 0.0
                            && ndc.x.abs() <= 1.0
                            && ndc.y.abs() <= 1.0
                            && (0.0..=1.0).contains(&ndc.z))
                        .then(|| {
                            Vec2::new((ndc.x * 0.5 + 0.5) * vw, (1.0 - (ndc.y * 0.5 + 0.5)) * vh)
                        });
                        let actual = camera.pixels_from_world(world, viewport, &EuclideanR3);

                        match (expected, actual) {
                            (Some(expected), Some(actual)) => {
                                on_screen += 1;
                                assert!(
                                    (actual - expected).length() < 1e-2,
                                    "{world:?}: {actual:?} px, matrix says {expected:?}"
                                );
                            }
                            (None, None) => clipped += 1,
                            (expected, actual) => panic!(
                                "{world:?}: visibility disagrees, matrix {expected:?} vs {actual:?}"
                            ),
                        }
                    }
                }
            }
        }

        assert!(
            on_screen > 100 && clipped > 100,
            "{on_screen} on screen, {clipped} clipped: the grid missed a case"
        );
    }

    /// Both curvature signs anchor on their own geodesics. The eye is off the
    /// chart origin in each, where the geodesic through a point and the chord
    /// to it part company.
    #[test]
    fn curved_projection_lands_on_the_geodesic_through_its_own_pixel() {
        let ball_samples: Vec<Vec3> = (-2..=2)
            .flat_map(|x| {
                (-2..=2).flat_map(move |y| {
                    (-2..=2).map(move |z: i32| {
                        Vec3::new(x as f32 * 0.17, y as f32 * 0.17, z as f32 * 0.17)
                    })
                })
            })
            .collect();

        assert_anchors_sit_on_their_own_geodesic(
            &HyperbolicH3,
            Vec3::new(0.28, 0.12, 0.44),
            &ball_samples,
            1e-3,
        );
        assert_anchors_sit_on_their_own_geodesic(
            &SphericalS3,
            Vec3::new(0.28, 0.12, 0.44),
            &ball_samples,
            1e-3,
        );
    }

    /// The `log` is load-bearing, not decoration: from an off-origin eye in
    /// H³ the geodesic leaves along a direction the chord does not share, so
    /// the anchor moves a visible fraction of the screen. A revert to
    /// `world - position` passes every flat test and fails this one.
    #[test]
    fn curved_projection_departs_from_the_flat_chord_formula() {
        let space = HyperbolicH3;
        let mut camera = Camera::<HyperbolicH3>::looking_at(
            Vec3::new(0.45, 0.15, 0.35),
            Vec3::ZERO,
            Vec3::Y,
            &space,
        );
        camera.fov_y = 70_f32.to_radians();
        camera.aspect = 16.0 / 9.0;
        camera.near = 1e-5;

        let world = Vec3::new(-0.30, 0.26, -0.18);
        let geodesic = camera
            .ndc_from_world(world, &space)
            .expect("sample is in view");

        let chord = world - camera.position;
        let tan_half_fov_y = (camera.fov_y * 0.5).tan();
        let depth = chord.dot(camera.forward);
        let flat = Vec2::new(
            chord.dot(camera.right) / (depth * camera.aspect * tan_half_fov_y),
            chord.dot(camera.up) / (depth * tan_half_fov_y),
        );

        let separation = (geodesic - flat).length();
        assert!(
            separation > 0.05,
            "geodesic {geodesic:?} and chord {flat:?} agree to {separation} NDC"
        );
    }

    /// A zero-area viewport yields no anchor rather than a divide by zero.
    #[test]
    fn degenerate_viewport_has_no_screen_position() {
        let camera = Camera::<EuclideanR3>::at_origin();
        let world = Vec3::new(0.0, 0.0, -5.0);
        assert!(camera
            .pixels_from_world(world, (0, 600), &EuclideanR3)
            .is_none());
        assert!(camera
            .pixels_from_world(world, (800, 0), &EuclideanR3)
            .is_none());
        assert!(camera
            .pixels_from_world(world, (800, 600), &EuclideanR3)
            .is_some());
    }
}
