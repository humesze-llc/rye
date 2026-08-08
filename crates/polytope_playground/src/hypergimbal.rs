//! Hypergimbal overlay: the six rotation planes as grabbable rings.
//!
//! Ring geometry, picking and the drag-to-rotor map are engine machinery in
//! [`loam_render::hypergimbal`], derivation included. This module places the
//! widget in the scene, arbitrates the left button against the flick
//! gesture, and turns a grabbed ring into an edit of the same per-plane
//! angle the Active-set slider writes.

use glam::{Mat4, Vec2, Vec3};
use loam_app::Input;
use loam_camera::Ray;
use loam_math::{EuclideanR3, Rotor4};
use loam_render::device::RenderDevice;
use loam_render::hypergimbal::{Hypergimbal, Ring, RingStyle};
use loam_shape::LineMesh;

use crate::consts::{BASE_ROTATION_RATE, BODY_Y};
use crate::physics::ndc_from_pixels;
use crate::state::{Demo, RotationMode, ViewMode};

/// Widget centre: the row's midpoint at body height, so the rings frame the
/// shapes rather than sitting off to one side.
const CENTER: Vec3 = Vec3::new(0.0, BODY_Y, 0.0);

/// Rings reach `(1 + √2)·scale` from the centre, so this puts the outer edge
/// at 1.33 world units: just under twice `BODY_SIZE`, enough to enclose the
/// leading shape without reaching its neighbour's column.
const SCALE: f32 = 0.55;

/// Grab radius in world units, about seven pixels at the startup framing
/// (720 rows, 60° fov, 8 units out). World-space rather than screen-space,
/// which is what makes grabbing fussier as the camera pulls back.
const PICK_TOLERANCE: f32 = 0.09;

/// Colour for the ring under the cursor, or held.
const HIGHLIGHT: [f32; 4] = [1.0, 0.94, 0.55, 1.0];

/// Widget placement. Constant: the rings are the ambient rotation planes, so
/// nothing about them tracks the subject's pose.
pub(crate) fn widget() -> Hypergimbal {
    Hypergimbal {
        center: CENTER,
        scale: SCALE,
    }
}

/// A held ring, anchored at the press edge so the whole drag is measured
/// against one origin rather than accumulated frame by frame.
#[derive(Copy, Clone, Debug)]
pub(crate) struct GimbalDrag {
    ring: Ring,
    /// Where the press ray met the ring's plane.
    grab: Vec3,
    /// Active-mode displayed angle for the ring's plane at the press edge.
    base_displayed: f32,
    /// Composer-mode pose at the press edge.
    base_rotor: Rotor4,
}

pub(crate) struct GimbalUi {
    pub(crate) enabled: bool,
    pub(crate) drag: Option<GimbalDrag>,
    hover: Option<Ring>,
    /// Highlighted plane the retained mesh was built for. The rings never
    /// move, so this is the only thing that can dirty it.
    built_highlight: Option<usize>,
    mesh: LineMesh<3>,
}

impl Default for GimbalUi {
    /// On at startup: the widget is the demo's answer to "how do I turn this
    /// thing in 4D", and a hidden answer is no answer.
    fn default() -> Self {
        Self {
            enabled: true,
            drag: None,
            hover: None,
            built_highlight: None,
            mesh: LineMesh::<3>::default(),
        }
    }
}

/// Ring the ray grabs, and where it met that ring's plane.
fn grab_ring(gimbal: &Hypergimbal, ray: &Ray) -> Option<(Ring, Vec3)> {
    let ring = gimbal.pick(ray.origin, ray.direction, PICK_TOLERANCE)?;
    ring.ray_hit(ray.origin, ray.direction)
        .map(|hit| (ring, hit))
}

/// Active-mode `base_angles` entry a drag asks for. The displayed angle is
/// `base + spin(t)`, so the drag has to hand back the spin it does not own;
/// this is the same solve the Active slider does on a change.
fn dragged_base_angle(drag: &GimbalDrag, cursor: Vec3, spin_contribution: f32) -> f32 {
    drag.base_displayed + drag.ring.drag_angle(drag.grab, cursor) - spin_contribution
}

impl Demo {
    /// Whether the rings are on screen this frame. Filmstrip composes its own
    /// per-cell viewports around a single subject with no shared world
    /// origin, so the widget has nowhere to stand; drawing and grabbing are
    /// gated together, because a handle that is grabbable while invisible is
    /// worse than no handle.
    fn gimbal_visible(&self) -> bool {
        self.gimbal.enabled && self.view_mode != ViewMode::Filmstrip
    }

    /// Update the hypergimbal against this frame's input. Returns `true`
    /// while a ring is held, which is what keeps the flick gesture and the
    /// orbit off the left button for the rest of the drag.
    ///
    /// Reads `left_was_down` before [`Demo::update_throw`] refreshes it, so
    /// this must stay ahead of that call.
    pub(crate) fn update_gimbal(
        &mut self,
        enabled: bool,
        input: &Input,
        viewport: (u32, u32),
    ) -> bool {
        if !enabled || !self.gimbal_visible() {
            self.gimbal.drag = None;
            self.gimbal.hover = None;
            return false;
        }
        let gimbal = widget();
        let down = input.buttons.left.down;
        let pressed = down && !self.left_was_down;

        if !down {
            self.gimbal.drag = None;
        } else if pressed {
            self.gimbal.drag = input.buttons.left.press_pos.and_then(|press_px| {
                let ray = self
                    .camera
                    .ray_from_ndc(ndc_from_pixels(press_px, viewport));
                grab_ring(&gimbal, &ray).map(|(ring, hit)| GimbalDrag {
                    ring,
                    grab: hit,
                    base_displayed: self.active_displayed_angle(ring.plane as usize),
                    base_rotor: self.rot_state,
                })
            });
        }

        let cursor_ray = input
            .cursor_pos
            .map(|px| self.camera.ray_from_ndc(ndc_from_pixels(px, viewport)));
        self.gimbal.hover = match (self.gimbal.drag, cursor_ray) {
            (Some(drag), _) => Some(drag.ring),
            (None, Some(ray)) => gimbal.pick(ray.origin, ray.direction, PICK_TOLERANCE),
            (None, None) => None,
        };

        let Some(drag) = self.gimbal.drag else {
            return false;
        };
        // A cursor that left the window, or a camera that swung the ring
        // edge-on, leaves the drag held at its last angle rather than
        // snapping the subject somewhere arbitrary.
        if let Some(cursor) =
            cursor_ray.and_then(|ray| drag.ring.ray_hit(ray.origin, ray.direction))
        {
            self.apply_gimbal_drag(&drag, cursor);
        }
        true
    }

    fn apply_gimbal_drag(&mut self, drag: &GimbalDrag, cursor: Vec3) {
        match self.rotation_mode {
            RotationMode::Active => {
                let plane_idx = drag.ring.plane as usize;
                let spin = if self.active[plane_idx] {
                    self.rot_time * BASE_ROTATION_RATE
                } else {
                    0.0
                };
                self.base_angles[plane_idx] = dragged_base_angle(drag, cursor, spin);
                self.rot_state = self.active_rotor();
            }
            RotationMode::Composer => {
                self.rot_state =
                    (drag.ring.drag_rotor(drag.grab, cursor) * drag.base_rotor).normalize();
            }
        }
        self.write_all(self.rot_state);
    }

    /// Draw the six rings. Last pass of the frame and depth-free: a
    /// manipulator the scene can hide is a manipulator that cannot be
    /// grabbed.
    ///
    /// The rings are fixed in world space, so the mesh is rebuilt only when
    /// the highlight moves, not per frame.
    pub(crate) fn record_gimbal(
        &mut self,
        rd: &RenderDevice,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        if !self.gimbal_visible() {
            return;
        }
        let highlight = self.gimbal.hover.map(|ring| ring.plane as usize);
        if self.gimbal.mesh.segments.is_empty() || self.gimbal.built_highlight != highlight {
            let mut style = RingStyle::default();
            if let Some(plane_idx) = highlight {
                style.colors[plane_idx] = HIGHLIGHT;
            }
            let mesh = &mut self.gimbal.mesh;
            mesh.segments.clear();
            mesh.colors.clear();
            mesh.widths.clear();
            widget().append_line_mesh(&style, mesh);
            self.gimbal.built_highlight = highlight;
            self.gimbal_node.upload::<EuclideanR3, 3>(
                &rd.device,
                &rd.queue,
                &self.gimbal.mesh,
                &loam_math::Projection::Identity,
                1,
            );
        }

        let cfg = &rd.surface_bundle.config;
        let view_dir = self.camera.view();
        let aspect = cfg.width as f32 / cfg.height as f32;
        let view_proj = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.1, 100.0)
            * Mat4::look_to_rh(view_dir.position, view_dir.forward, view_dir.up);
        self.gimbal_node.set_camera(
            &rd.queue,
            view_proj,
            Vec2::new(cfg.width as f32, cfg.height as f32),
        );
        self.gimbal_node.record(encoder, view, None, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_app::{Camera, CameraController};
    use loam_camera::OrbitController;
    use loam_math::{Bivector, Plane4, Rotor};

    const VIEWPORT: (u32, u32) = (1280, 720);

    /// Point at great-circle parameter `θ` on the plane's coordinate circle,
    /// matching `loam_render::hypergimbal`'s `p(θ) = a cos θ + b sin θ`.
    fn great_circle_point(plane: Plane4, theta: f32) -> glam::Vec4 {
        let (cos, sin) = (theta.cos(), theta.sin());
        match plane {
            Plane4::Xy => glam::Vec4::new(cos, sin, 0.0, 0.0),
            Plane4::Xz => glam::Vec4::new(cos, 0.0, sin, 0.0),
            Plane4::Xw => glam::Vec4::new(cos, 0.0, 0.0, sin),
            Plane4::Yz => glam::Vec4::new(0.0, cos, sin, 0.0),
            Plane4::Yw => glam::Vec4::new(0.0, cos, 0.0, sin),
            Plane4::Zw => glam::Vec4::new(0.0, 0.0, cos, sin),
        }
    }

    /// The demo's startup framing, reproduced through the same controller
    /// `Demo::new` + `update` drive.
    fn startup_camera() -> Camera<EuclideanR3> {
        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 3.0, 9.0);
        camera.aspect = VIEWPORT.0 as f32 / VIEWPORT.1 as f32;
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        orbit.set_orbit(8.0, -0.25);
        orbit.advance(Input::default(), &mut camera, &EuclideanR3, 0.0);
        camera
    }

    /// Inverse of `Camera::ray_from_ndc`, then the inverse of
    /// `ndc_from_pixels`: where a world point lands on screen. `None` behind
    /// the eye.
    fn pixels_of(camera: &Camera<EuclideanR3>, world: Vec3) -> Option<Vec2> {
        let offset = world - camera.position;
        let depth = offset.dot(camera.forward);
        if depth <= 0.0 {
            return None;
        }
        let tan_half = (camera.fov_y * 0.5).tan();
        let ndc = Vec2::new(
            offset.dot(camera.right) / (depth * camera.aspect * tan_half),
            offset.dot(camera.up) / (depth * tan_half),
        );
        Some(Vec2::new(
            (ndc.x + 1.0) * 0.5 * VIEWPORT.0 as f32,
            (1.0 - ndc.y) * 0.5 * VIEWPORT.1 as f32,
        ))
    }

    /// The pixel seam is round-trip exact: a world point turned into pixels
    /// and back into a ray produces a ray through that point. Catches a
    /// dropped y-flip and a dropped aspect, either of which would leave the
    /// drag tracking the cursor along the wrong screen axis.
    #[test]
    fn the_pixel_to_ray_seam_round_trips_a_world_point() {
        let camera = startup_camera();
        for plane in Plane4::ALL {
            let ring = widget().ring(plane);
            for step in 0..8 {
                let world = ring.point(step as f32 / 8.0 * std::f32::consts::TAU);
                let pixels = pixels_of(&camera, world).expect("ring is in front of the eye");
                let ray = camera.ray_from_ndc(ndc_from_pixels(pixels, VIEWPORT));
                let along = (world - ray.origin).dot(ray.direction);
                let miss = (world - (ray.origin + ray.direction * along)).length();
                assert!(miss < 1e-3, "{plane:?}: ray misses its own pixel by {miss}");
            }
        }
    }

    /// The widget is on screen at the startup framing. This is as close as a
    /// headless test gets to the visual claim: every ring point is in front
    /// of the eye and inside the NDC square, so nothing is clipped away and
    /// no ring is behind the camera.
    #[test]
    fn every_ring_is_inside_the_startup_view() {
        let camera = startup_camera();
        for plane in Plane4::ALL {
            let ring = widget().ring(plane);
            for step in 0..64 {
                let world = ring.point(step as f32 / 64.0 * std::f32::consts::TAU);
                let pixels =
                    pixels_of(&camera, world).unwrap_or_else(|| panic!("{plane:?} behind the eye"));
                assert!(
                    (0.0..=VIEWPORT.0 as f32).contains(&pixels.x)
                        && (0.0..=VIEWPORT.1 as f32).contains(&pixels.y),
                    "{plane:?} leaves the viewport at {pixels:?}"
                );
            }
        }
    }

    /// End to end through the pixel seam: a press aimed at a point on ring
    /// `P` and a release aimed at the point `Δθ` further along `P`'s great
    /// circle asks the Active slider for exactly `Δθ` more. Runs for all six
    /// planes, so a ring wired to the wrong slider index fails here.
    #[test]
    fn a_drag_along_a_ring_asks_its_own_plane_for_the_arc_it_swept() {
        let camera = startup_camera();
        let gimbal = widget();
        let delta = 0.55_f32;
        for plane in Plane4::ALL {
            // Rings cross on screen, so press where this one is the
            // front-most candidate; a user picks such a spot by eye.
            let (start_theta, ring, grab) = (0..48)
                .find_map(|step| {
                    let theta = step as f32 / 48.0 * std::f32::consts::TAU;
                    let world = gimbal.project(great_circle_point(plane, theta))?;
                    let pixels = pixels_of(&camera, world)?;
                    let ray = camera.ray_from_ndc(ndc_from_pixels(pixels, VIEWPORT));
                    let (ring, hit) = grab_ring(&gimbal, &ray)?;
                    (ring.plane == plane).then_some((theta, ring, hit))
                })
                .unwrap_or_else(|| panic!("{plane:?} is never the ring a press grabs"));

            // Where `delta` of rotation in this plane actually sends the
            // grabbed point: the rotated point's projection, not a second
            // copy of the ring's own angle map.
            let rotated = (plane.unit_bivector() * delta)
                .exp()
                .apply(great_circle_point(plane, start_theta));
            let release_px =
                pixels_of(&camera, gimbal.project(rotated).expect("image is finite")).unwrap();
            let release_ray = camera.ray_from_ndc(ndc_from_pixels(release_px, VIEWPORT));
            let cursor = ring
                .ray_hit(release_ray.origin, release_ray.direction)
                .expect("release ray meets the held ring's plane");

            let drag = GimbalDrag {
                ring,
                grab,
                base_displayed: 0.25,
                base_rotor: Rotor4::IDENTITY,
            };
            let asked = dragged_base_angle(&drag, cursor, 0.0) - 0.25;
            assert!(
                (asked - delta).abs() < 5e-3,
                "{plane:?}: drag asked for {asked}, not {delta}"
            );
            // The spin contribution is subtracted verbatim: the slider owns
            // it, the drag does not.
            let with_spin = dragged_base_angle(&drag, cursor, 0.4);
            assert!((with_spin - (asked + 0.25 - 0.4)).abs() < 1e-6);
        }
    }

    /// All six planes are reachable at the startup framing, without orbiting
    /// first: each ring is the front-most grab over most of its own
    /// circumference. The bar is half; the worst ring measures 32 of 48. A
    /// ring the default camera sees edge-on scores zero here, which is what
    /// an image frame with a ring plane square to a view axis produces.
    #[test]
    fn every_plane_is_grabbable_at_the_startup_framing() {
        let camera = startup_camera();
        let gimbal = widget();
        const SAMPLES: usize = 48;
        for plane in Plane4::ALL {
            let own = (0..SAMPLES)
                .filter(|step| {
                    let theta = *step as f32 / SAMPLES as f32 * std::f32::consts::TAU;
                    let Some(world) = gimbal.project(great_circle_point(plane, theta)) else {
                        return false;
                    };
                    let Some(pixels) = pixels_of(&camera, world) else {
                        return false;
                    };
                    let ray = camera.ray_from_ndc(ndc_from_pixels(pixels, VIEWPORT));
                    grab_ring(&gimbal, &ray).is_some_and(|(ring, _)| ring.plane == plane)
                })
                .count();
            assert!(
                own * 2 > SAMPLES,
                "{plane:?} grabbable at only {own}/{SAMPLES} points from the startup camera"
            );
        }
    }

    /// A press that grabbed a ring and never moved asks for no change at
    /// all, so holding the button still cannot drift the subject.
    #[test]
    fn holding_a_ring_still_asks_for_no_rotation() {
        let gimbal = widget();
        for plane in Plane4::ALL {
            let ring = gimbal.ring(plane);
            let grab = ring.point(1.3);
            let drag = GimbalDrag {
                ring,
                grab,
                base_displayed: -0.9,
                base_rotor: Rotor4::IDENTITY,
            };
            assert_eq!(dragged_base_angle(&drag, grab, 0.0), -0.9);
        }
    }
}
