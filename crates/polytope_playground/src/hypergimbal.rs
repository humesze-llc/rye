//! Handle geometry, picking and the drag-to-delta map are engine machinery in
//! [`loam_render::gizmo`], derivation included.

use glam::{Mat4, Vec2, Vec3, Vec4};
use loam_app::Input;
use loam_camera::Ray;
use loam_math::{EuclideanR3, Rotor4};
use loam_render::device::RenderDevice;
use loam_render::gizmo::{
    GizmoStyle, Handle, HandleDrag, HandleId, TransformDelta, TransformGizmo,
};
use loam_shape::LineMesh;

use crate::consts::BASE_ROTATION_RATE;
use crate::physics::{ndc_from_pixels, PlaygroundPhysics};
use crate::state::{Demo, RotationMode, ViewMode};

// Rings reach `(1 + √2)·scale` and arrow tips just under `3·scale`, so this
// puts the widget's outer edge at 1.64 world units: inside `BODY_X_SPACING`,
// so no handle reaches the centre of the neighbouring column.
const SCALE: f32 = 0.55;

// Grab radius in world units, about seven pixels at the startup framing (720
// rows, 60° fov, 8 units out). World-space rather than screen-space, which is
// what makes grabbing fussier as the camera pulls back.
const PICK_TOLERANCE: f32 = 0.09;

const HIGHLIGHT: [f32; 4] = [1.0, 0.94, 0.55, 1.0];

// The handles are the ambient rotation planes and axes, so their shape never
// tracks the subject's orientation; only where they stand does.
pub(crate) fn widget(center: Vec3) -> TransformGizmo {
    TransformGizmo {
        center,
        scale: SCALE,
    }
}

pub(crate) fn gimbal_center(physics: &PlaygroundPhysics, slot: usize, slots: usize) -> Vec3 {
    physics.pose(slot, slots, Rotor4::IDENTITY).position_r3()
}

// Anchored at the press edge so the whole drag is measured against one origin
// rather than accumulated frame by frame.
#[derive(Copy, Clone, Debug)]
pub(crate) struct GimbalDrag {
    held: HandleDrag,
    base_displayed: f32,
    base_rotor: Rotor4,
    base_position: Vec4,
}

#[derive(Default)]
pub(crate) struct GimbalUi {
    /// Off at startup, by the maintainer's call.
    pub(crate) enabled: bool,
    pub(crate) drag: Option<GimbalDrag>,
    hover: Option<HandleId>,
    built_highlight: Option<HandleId>,
    mesh: LineMesh<3>,
}

fn grab_handle(gizmo: &TransformGizmo, ray: &Ray) -> Option<HandleDrag> {
    let handle = gizmo.pick(ray.origin, ray.direction, PICK_TOLERANCE)?;
    HandleDrag::press(handle, ray.origin, ray.direction)
}

// The displayed angle is `base + spin(t)`, so the drag has to hand back the
// spin it does not own; this is the same solve the Active slider does.
fn dragged_base_angle(base_displayed: f32, drag_angle: f32, spin_contribution: f32) -> f32 {
    base_displayed + drag_angle - spin_contribution
}

impl Demo {
    // Filmstrip composes its own per-cell viewports with no shared world
    // origin, so the widget has nowhere to stand; drawing and grabbing are
    // gated together, because a grabbable invisible handle is worse than none.
    fn gimbal_visible(&self) -> bool {
        self.gimbal.enabled && self.view_mode != ViewMode::Filmstrip
    }

    fn gimbal_widget(&self) -> TransformGizmo {
        widget(gimbal_center(
            &self.physics,
            self.selected_slot(),
            self.render_row().len(),
        ))
    }

    fn selected_position(&self) -> Vec4 {
        self.physics
            .pose(
                self.selected_slot(),
                self.render_row().len(),
                Rotor4::IDENTITY,
            )
            .position
    }

    // Returns `true` while a handle is held, which keeps the flick gesture and
    // the orbit off the left button for the rest of the drag. Reads
    // `left_was_down` before [`Demo::update_throw`] refreshes it, so this must
    // stay ahead of that call.
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
        let gizmo = self.gimbal_widget();
        let down = input.buttons.left.down;
        let pressed = down && !self.left_was_down;

        if !down {
            self.gimbal.drag = None;
        } else if pressed {
            self.gimbal.drag = input.buttons.left.press_pos.and_then(|press_px| {
                let ray = self
                    .camera
                    .ray_from_ndc(ndc_from_pixels(press_px, viewport));
                grab_handle(&gizmo, &ray).map(|held| GimbalDrag {
                    held,
                    base_displayed: match held.id() {
                        HandleId::Rotate(plane) => self.active_displayed_angle(plane as usize),
                        HandleId::Translate(_) => 0.0,
                    },
                    base_rotor: self.selected_rotor(),
                    base_position: self.selected_position(),
                })
            });
        }

        let cursor_ray = input
            .cursor_pos
            .map(|px| self.camera.ray_from_ndc(ndc_from_pixels(px, viewport)));
        self.gimbal.hover = match (self.gimbal.drag, cursor_ray) {
            (Some(drag), _) => Some(drag.held.id()),
            (None, Some(ray)) => gizmo
                .pick(ray.origin, ray.direction, PICK_TOLERANCE)
                .map(Handle::id),
            (None, None) => None,
        };

        let Some(drag) = self.gimbal.drag else {
            return false;
        };
        // A cursor that left the window, or a camera that swung the handle
        // edge-on, leaves the drag held at its last delta rather than snapping
        // the subject somewhere arbitrary.
        if let Some(delta) = cursor_ray.and_then(|ray| drag.held.delta(ray.origin, ray.direction)) {
            self.apply_gimbal_drag(&drag, delta);
        }
        true
    }

    fn apply_gimbal_drag(&mut self, drag: &GimbalDrag, delta: TransformDelta) {
        match delta {
            TransformDelta::Rotate { plane, angle } => match self.rotation_mode {
                RotationMode::Active => {
                    let plane_idx = plane as usize;
                    let spin = if self.spins.selected_spin().active[plane_idx] {
                        self.rot_time * BASE_ROTATION_RATE
                    } else {
                        0.0
                    };
                    self.spins.selected_spin_mut().base_angles[plane_idx] =
                        dragged_base_angle(drag.base_displayed, angle, spin);
                    self.apply_selected_active_edit();
                }
                RotationMode::Composer => {
                    self.spins.selected_spin_mut().rotor =
                        (delta.rotor() * drag.base_rotor).normalize();
                    self.rebuild_bodies();
                }
            },
            TransformDelta::Translate { .. } => {
                let slot = self.selected_slot();
                let body = &mut self.physics.world.bodies[slot];
                body.position = drag.base_position + delta.translation();
                // A held shaft owns the subject's position for the whole drag,
                // so it takes the velocity with it: otherwise the frame's
                // physics step integrates the body out from under the cursor.
                body.velocity = Vec4::ZERO;
                // The upload gate keys on a moving world and on changed
                // rotors, and a teleport is neither.
                self.rebuild_bodies();
            }
        }
    }

    // Last pass of the frame and depth-free: a manipulator the scene can hide
    // is a manipulator that cannot be grabbed. The mesh is built about the
    // ORIGIN and carried to the selected body by a translation folded into the
    // view-projection, so only the highlight can dirty the geometry.
    pub(crate) fn record_gimbal(
        &mut self,
        rd: &RenderDevice,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        if !self.gimbal_visible() {
            return;
        }
        let highlight = self.gimbal.hover;
        if self.gimbal.mesh.segments.is_empty() || self.gimbal.built_highlight != highlight {
            let mut style = GizmoStyle::default();
            match highlight {
                Some(HandleId::Rotate(plane)) => style.rings.colors[plane as usize] = HIGHLIGHT,
                Some(HandleId::Translate(axis)) => style.shafts.colors[axis as usize] = HIGHLIGHT,
                None => {}
            }
            let mesh = &mut self.gimbal.mesh;
            mesh.segments.clear();
            mesh.colors.clear();
            mesh.widths.clear();
            widget(Vec3::ZERO).append_line_mesh(&style, mesh);
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
        let center = gimbal_center(&self.physics, self.selected_slot(), self.render_row().len());
        let view_proj = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.1, 100.0)
            * Mat4::look_to_rh(view_dir.position, view_dir.forward, view_dir.up)
            * Mat4::from_translation(center);
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
    use crate::consts::BODY_SIZE;
    use loam_app::{Camera, CameraController};
    use loam_camera::OrbitController;
    use loam_math::{Bivector, Plane4, Rotor};
    use loam_render::gizmo::Axis4;

    const VIEWPORT: (u32, u32) = (1280, 720);

    const WIDEST_ROW: usize = crate::consts::MAX_ROW_LEN;

    fn row_center() -> Vec3 {
        gimbal_center(&PlaygroundPhysics::new(1, BODY_SIZE), 0, 1)
    }

    fn selectable_centers(slots: usize) -> Vec<Vec3> {
        let physics = PlaygroundPhysics::new(slots, BODY_SIZE);
        (0..slots)
            .map(|slot| gimbal_center(&physics, slot, slots))
            .collect()
    }

    fn great_circle_point(plane: Plane4, theta: f32) -> Vec4 {
        let (cos, sin) = (theta.cos(), theta.sin());
        match plane {
            Plane4::Xy => Vec4::new(cos, sin, 0.0, 0.0),
            Plane4::Xz => Vec4::new(cos, 0.0, sin, 0.0),
            Plane4::Xw => Vec4::new(cos, 0.0, 0.0, sin),
            Plane4::Yz => Vec4::new(0.0, cos, sin, 0.0),
            Plane4::Yw => Vec4::new(0.0, cos, 0.0, sin),
            Plane4::Zw => Vec4::new(0.0, 0.0, cos, sin),
        }
    }

    fn handle_points(gizmo: &TransformGizmo, samples: usize) -> Vec<(HandleId, Vec3)> {
        let mut out = Vec::new();
        for ring in gizmo.rings() {
            for step in 0..samples {
                let chi = step as f32 / samples as f32 * std::f32::consts::TAU;
                out.push((HandleId::Rotate(ring.plane), ring.point(chi)));
            }
        }
        for shaft in gizmo.shafts() {
            for step in 0..=samples {
                let along =
                    shaft.inner + (shaft.outer - shaft.inner) * step as f32 / samples as f32;
                out.push((HandleId::Translate(shaft.axis), shaft.point(along)));
            }
        }
        out
    }

    fn startup_camera() -> Camera<EuclideanR3> {
        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 3.0, 9.0);
        camera.aspect = VIEWPORT.0 as f32 / VIEWPORT.1 as f32;
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        orbit.set_orbit(8.0, -0.25);
        orbit.advance(Input::default(), &mut camera, &EuclideanR3, 0.0);
        camera
    }

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

    fn ray_at(camera: &Camera<EuclideanR3>, world: Vec3) -> Option<Ray> {
        let pixels = pixels_of(camera, world)?;
        Some(camera.ray_from_ndc(ndc_from_pixels(pixels, VIEWPORT)))
    }

    #[test]
    fn the_pixel_to_ray_seam_round_trips_a_world_point() {
        let camera = startup_camera();
        for (id, world) in handle_points(&widget(row_center()), 8) {
            let ray = ray_at(&camera, world).expect("handle is in front of the eye");
            let along = (world - ray.origin).dot(ray.direction);
            let miss = (world - (ray.origin + ray.direction * along)).length();
            assert!(miss < 1e-3, "{id:?}: ray misses its own pixel by {miss}");
        }
    }

    #[test]
    fn every_handle_is_inside_the_startup_view_at_every_selectable_slot() {
        let camera = startup_camera();
        for slots in 1..=WIDEST_ROW {
            for (slot, center) in selectable_centers(slots).into_iter().enumerate() {
                for (id, world) in handle_points(&widget(center), 48) {
                    let pixels = pixels_of(&camera, world)
                        .unwrap_or_else(|| panic!("{id:?} behind the eye"));
                    assert!(
                        (0.0..=VIEWPORT.0 as f32).contains(&pixels.x)
                            && (0.0..=VIEWPORT.1 as f32).contains(&pixels.y),
                        "{id:?} on slot {slot} of {slots} leaves the viewport at {pixels:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_widget_stays_inside_its_own_column() {
        let gizmo = widget(Vec3::ZERO);
        for (id, world) in handle_points(&gizmo, 64) {
            let reach = world.length();
            assert!(
                reach < crate::consts::BODY_X_SPACING,
                "{id:?} reaches {reach}, past the neighbouring column"
            );
        }
    }

    #[test]
    fn the_widget_stands_on_the_selected_body() {
        for slots in 2..=WIDEST_ROW {
            let centers = selectable_centers(slots);
            for slot in 1..slots {
                let step = centers[slot] - centers[slot - 1];
                assert!(
                    (step - Vec3::X * crate::consts::BODY_X_SPACING).length() < 1e-6,
                    "slot {slot} of {slots} sits {step} from its neighbour"
                );
            }
            assert!((centers[0].x + centers[slots - 1].x).abs() < 1e-6);
        }
    }

    #[test]
    fn the_widget_follows_a_thrown_subject() {
        let slots = 3;
        let mut physics = PlaygroundPhysics::new(slots, BODY_SIZE);
        let parked = gimbal_center(&physics, 1, slots);
        physics.throw(1, Vec4::new(0.0, 0.6, 0.0, 0.0));
        physics.step(30);
        let moved = gimbal_center(&physics, 1, slots);
        assert!(
            (moved - parked).length() > 0.05,
            "the handles stayed at {parked} while their subject moved to {moved}"
        );
        assert_eq!(
            gimbal_center(&physics, 0, slots),
            selectable_centers(slots)[0],
            "an untouched slot's handles moved"
        );
    }

    #[test]
    fn a_drag_along_a_ring_asks_its_own_plane_for_the_arc_it_swept() {
        let camera = startup_camera();
        let gizmo = widget(row_center());
        let delta = 0.55_f32;
        for plane in Plane4::ALL {
            let (start_theta, held) = (0..48)
                .find_map(|step| {
                    let theta = step as f32 / 48.0 * std::f32::consts::TAU;
                    let world = gizmo
                        .hypergimbal()
                        .project(great_circle_point(plane, theta))?;
                    let ray = ray_at(&camera, world)?;
                    let held = grab_handle(&gizmo, &ray)?;
                    (held.id() == HandleId::Rotate(plane)).then_some((theta, held))
                })
                .unwrap_or_else(|| panic!("{plane:?} is never the handle a press grabs"));

            let rotated = (plane.unit_bivector() * delta)
                .exp()
                .apply(great_circle_point(plane, start_theta));
            let release = ray_at(
                &camera,
                gizmo
                    .hypergimbal()
                    .project(rotated)
                    .expect("image is finite"),
            )
            .expect("release point is in front of the eye");
            let Some(TransformDelta::Rotate { plane: got, angle }) =
                held.delta(release.origin, release.direction)
            else {
                panic!("{plane:?} ring drag produced no rotation");
            };
            assert_eq!(got, plane);

            let asked = dragged_base_angle(0.25, angle, 0.0) - 0.25;
            assert!(
                (asked - delta).abs() < 5e-3,
                "{plane:?}: drag asked for {asked}, not {delta}"
            );
            let with_spin = dragged_base_angle(0.25, angle, 0.4);
            assert!((with_spin - (asked + 0.25 - 0.4)).abs() < 1e-6);
        }
    }

    #[test]
    fn every_handle_is_grabbable_at_the_startup_framing() {
        let camera = startup_camera();
        let gizmo = widget(row_center());
        const SAMPLES: usize = 48;
        for plane in Plane4::ALL {
            let own = (0..SAMPLES)
                .filter(|step| {
                    let theta = *step as f32 / SAMPLES as f32 * std::f32::consts::TAU;
                    let Some(world) = gizmo
                        .hypergimbal()
                        .project(great_circle_point(plane, theta))
                    else {
                        return false;
                    };
                    let Some(ray) = ray_at(&camera, world) else {
                        return false;
                    };
                    gizmo
                        .pick(ray.origin, ray.direction, PICK_TOLERANCE)
                        .is_some_and(|handle| handle.id() == HandleId::Rotate(plane))
                })
                .count();
            assert!(
                own * 2 > SAMPLES,
                "{plane:?} grabbable at only {own}/{SAMPLES} points from the startup camera"
            );
        }
        for axis in Axis4::ALL {
            let shaft = gizmo.shaft(axis);
            for step in 0..=SAMPLES {
                let along = shaft.head_start() + shaft.head * step as f32 / SAMPLES as f32;
                let ray = ray_at(&camera, shaft.point(along)).expect("head is in front");
                let picked = gizmo
                    .pick(ray.origin, ray.direction, PICK_TOLERANCE)
                    .map(Handle::id);
                assert_eq!(
                    picked,
                    Some(HandleId::Translate(axis)),
                    "{axis:?} at {along} out of the centre picked {picked:?}"
                );
            }
        }
    }

    #[test]
    fn holding_a_handle_still_asks_for_no_change() {
        let camera = startup_camera();
        let gizmo = widget(row_center());
        for (id, world) in handle_points(&gizmo, 12) {
            let ray = ray_at(&camera, world).expect("handle is in front of the eye");
            let Some(handle) = gizmo.pick(ray.origin, ray.direction, PICK_TOLERANCE) else {
                continue;
            };
            let held = HandleDrag::press(handle, ray.origin, ray.direction).expect("grab");
            let delta = held.delta(ray.origin, ray.direction).expect("held still");
            assert_eq!(
                delta.translation(),
                Vec4::ZERO,
                "{id:?} drifted while held still"
            );
            match delta {
                TransformDelta::Rotate { angle, .. } => assert_eq!(angle, 0.0),
                TransformDelta::Translate { distance, .. } => assert_eq!(distance, 0.0),
            }
            assert_eq!(
                dragged_base_angle(-0.9, 0.0, 0.0),
                -0.9,
                "the Active solve moved on a still drag"
            );
        }
    }

    #[test]
    fn a_shaft_drag_moves_one_component_of_the_selected_body_and_nothing_else() {
        const SLOTS: usize = 3;
        const SLOT: usize = 1;
        let camera = startup_camera();
        for axis in Axis4::ALL {
            let mut physics = PlaygroundPhysics::new(SLOTS, BODY_SIZE);
            let before: Vec<Vec4> = (0..SLOTS)
                .map(|slot| physics.pose(slot, SLOTS, Rotor4::IDENTITY).position)
                .collect();
            let gizmo = widget(gimbal_center(&physics, SLOT, SLOTS));
            let shaft = gizmo.shaft(axis);

            let grab_at = shaft.outer - 0.1 * SCALE;
            let travel = 0.37_f32;
            let press = ray_at(&camera, shaft.point(grab_at)).expect("head is in front");
            let held = grab_handle(&gizmo, &press).expect("the arrowhead is grabbable");
            assert_eq!(held.id(), HandleId::Translate(axis));
            let release =
                ray_at(&camera, shaft.point(grab_at + travel)).expect("release is in front");
            let delta = held
                .delta(release.origin, release.direction)
                .expect("release ray reaches the shaft");

            physics.world.bodies[SLOT].position = before[SLOT] + delta.translation();

            let after: Vec<Vec4> = (0..SLOTS)
                .map(|slot| physics.pose(slot, SLOTS, Rotor4::IDENTITY).position)
                .collect();
            let moved = after[SLOT] - before[SLOT];
            let index = axis as usize;
            assert!(
                (moved.to_array()[index] - travel).abs() < 5e-3,
                "{axis:?} moved its own component by {}, not {travel}",
                moved.to_array()[index]
            );
            for other in 0..4 {
                if other == index {
                    continue;
                }
                assert_eq!(
                    after[SLOT].to_array()[other],
                    before[SLOT].to_array()[other],
                    "{axis:?} drag moved component {other}"
                );
            }
            for slot in 0..SLOTS {
                if slot == SLOT {
                    continue;
                }
                assert_eq!(after[slot], before[slot], "slot {slot} moved");
            }
        }
    }

    #[test]
    fn a_w_drag_moves_the_slice_and_not_the_r3_position() {
        let mut physics = PlaygroundPhysics::new(1, BODY_SIZE);
        let before = physics.pose(0, 1, Rotor4::IDENTITY);
        let shaft = widget(gimbal_center(&physics, 0, 1)).shaft(Axis4::W);
        let slide = shaft.drag_translation(0.0, 0.42);

        physics.world.bodies[0].position = before.position + slide;
        let after = physics.pose(0, 1, Rotor4::IDENTITY);
        assert_eq!(after.position_r3(), before.position_r3());
        assert_eq!(after.position.w - before.position.w, 0.42);

        let canonical = Vec4::new(0.3, -0.6, 0.2, 0.5);
        assert_eq!(
            after.body_local(canonical, BODY_SIZE) - before.body_local(canonical, BODY_SIZE),
            Vec4::W * 0.42
        );
    }
}
