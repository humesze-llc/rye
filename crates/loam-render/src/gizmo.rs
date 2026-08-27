//! The rotation half is [`crate::hypergimbal`], whose six rings are the
//! stereographic images of the six coordinate great circles of S³; the
//! derivation lives there and is not repeated. This module adds the
//! translation half and the arbitration between the two, so a caller drives
//! all ten handles through one [`TransformGizmo::pick`], one
//! [`HandleDrag`] and one [`TransformDelta`] per frame. Nothing here reads
//! or stores the subject's pose: a drag reports a delta for the caller to
//! compose.
//!
//! # Translating in R⁴ with a pointer that lives in R³
//!
//! A subject's R³ position is drawn at world `x, y, z`, so the `x`, `y` and
//! `z` shafts have to BE those axes; an arrow that moved the subject
//! somewhere other than where it points is not a handle. That leaves `w`,
//! which has no world direction, needing one assigned. The choice is
//! `-(1,1,1)/√3`, the unit vector maximising the smallest angle to `x̂`, `ŷ`
//! and `ẑ` (125.26° to each): `max(d·x̂, d·ŷ, d·ẑ)` is convex and symmetric
//! under permuting the axes, so its minimiser is the fixed point of that
//! symmetry group, and the antipode of the diagonal is the only unit vector
//! there. It is the direction least confusable with a scene axis, and it
//! lies in no coordinate plane.
//!
//! A shaft is grabbed by its arrowhead (see [`Shaft::head_start`]) and read
//! by the closest approach of the cursor ray to the shaft's whole LINE
//! (Ericson, *Real-Time Collision Detection*, 2005, §5.1.9), so the drag
//! runs past both ends of the drawn arrow and the subject travels one world
//! unit per world unit of cursor travel along it. `w` moves at that same
//! rate. There is nothing to scale it against, and any other rate would make
//! the one handle with no visible motion also the one with no predictable
//! size.

use glam::{Vec3, Vec4};
use loam_math::{Bivector, Plane4, Rotor4};
use loam_shape::LineMesh;

use crate::hypergimbal::{Hypergimbal, Ring, RingStyle};

/// Outer reach of the ring shell, in [`TransformGizmo::scale`] units: every
/// ring is centred `1` out with radius `√2`, which is what the pole choice
/// in [`crate::hypergimbal::POLE`] buys.
const RING_REACH: f32 = 1.0 + std::f32::consts::SQRT_2;

/// Inner end of a shaft, in `scale` units. Leaves the hub clear by several
/// grab radii at a usable widget size, so a press aimed at the subject
/// itself still reaches the subject rather than a handle.
const SHAFT_INNER: f32 = 0.45;

/// Arrowhead length, in `scale` units. Drawn and grabbed extent both, so
/// the head cannot be grabbable anywhere it is not drawn.
const SHAFT_HEAD: f32 = 0.28;

/// Open space between the ring shell and a shaft's tip, in `scale` units:
/// two arrowhead lengths, so no head is ever drawn over a ring.
const SHAFT_CLEARANCE: f32 = 2.0 * SHAFT_HEAD;

const SHAFT_OUTER: f32 = RING_REACH + SHAFT_CLEARANCE;

/// `1/√3`, the component size of the `w` shaft's world direction.
const INV_SQRT_3: f32 = 0.577_350_26;

/// Sine of the smallest ray-to-shaft angle a shaft is trusted at. The
/// closest-approach parameter moves as `1/sin`, so below this a one-pixel
/// cursor move slides the subject an arbitrary distance and the drag stops
/// tracking the cursor. Same conditioning class, and the same value, as
/// the ring plane's incidence floor.
const MIN_SHAFT_INCIDENCE: f32 = 1e-2;

/// One of R⁴'s four coordinate axes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Axis4 {
    X = 0,
    Y = 1,
    Z = 2,
    W = 3,
}

impl Axis4 {
    /// All four axes, in `Vec4` component order.
    pub const ALL: [Self; 4] = [Self::X, Self::Y, Self::Z, Self::W];

    /// Single-letter label, matching [`Plane4::label`]'s convention.
    pub fn label(self) -> &'static str {
        match self {
            Self::X => "x",
            Self::Y => "y",
            Self::Z => "z",
            Self::W => "w",
        }
    }

    /// The R⁴ unit vector a drag on this shaft translates along.
    pub fn unit(self) -> Vec4 {
        match self {
            Self::X => Vec4::X,
            Self::Y => Vec4::Y,
            Self::Z => Vec4::Z,
            Self::W => Vec4::W,
        }
    }

    /// World R³ direction the shaft is drawn and dragged along. See the
    /// module docs for why `w` gets the anti-diagonal.
    pub fn shaft_direction(self) -> Vec3 {
        match self {
            Self::X => Vec3::X,
            Self::Y => Vec3::Y,
            Self::Z => Vec3::Z,
            Self::W => Vec3::new(-INV_SQRT_3, -INV_SQRT_3, -INV_SQRT_3),
        }
    }
}

/// One translation axis' handle: the world segment its arrow occupies, and
/// the R⁴ axis a slide along it moves.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Shaft {
    pub axis: Axis4,
    /// Widget centre, the origin the shaft parameter is measured from.
    pub origin: Vec3,
    /// Unit world direction, [`Axis4::shaft_direction`].
    pub direction: Vec3,
    /// Drawn extent as distances from [`Self::origin`] along
    /// [`Self::direction`]. Neither the pick nor the drag is clipped to it;
    /// see [`Self::head_start`].
    pub inner: f32,
    pub outer: f32,
    /// Arrowhead length, measured back from [`Self::outer`].
    pub head: f32,
}

impl Shaft {
    pub fn point(&self, along: f32) -> Vec3 {
        self.origin + self.direction * along
    }

    /// Where the grabbable arrowhead begins.
    ///
    /// The head is the handle; the stem behind it is a rail showing where
    /// the drag will run, not a grab surface.
    pub fn head_start(&self) -> f32 {
        self.outer - self.head
    }

    /// Distance along the shaft of the point on the shaft's LINE closest to
    /// the ray. `None` within 0.6° of parallel, where the answer moves as
    /// `1/sin` and the drag stops tracking the cursor.
    ///
    /// Ericson, *Real-Time Collision Detection* (2005), §5.1.9, specialised
    /// to a unit [`Self::direction`].
    pub fn ray_parameter(&self, ray_origin: Vec3, ray_direction: Vec3) -> Option<f32> {
        let offset = self.origin - ray_origin;
        let alignment = self.direction.dot(ray_direction);
        let ray_length_squared = ray_direction.length_squared();
        // |d|²·sin², so the guard reads as a floor on sin.
        let denominator = ray_length_squared - alignment * alignment;
        if denominator <= MIN_SHAFT_INCIDENCE * MIN_SHAFT_INCIDENCE * ray_length_squared {
            return None;
        }
        Some(
            (alignment * ray_direction.dot(offset)
                - self.direction.dot(offset) * ray_length_squared)
                / denominator,
        )
    }

    /// Depth along the ray at which it passes the arrowhead, when that pass
    /// is within `tolerance` of it. `None` for a miss, a near-parallel ray,
    /// or a pass behind the eye.
    fn hit_depth(&self, ray_origin: Vec3, ray_direction: Vec3, tolerance: f32) -> Option<f32> {
        let along = self.ray_parameter(ray_origin, ray_direction)?;
        let nearest = self.point(along.clamp(self.head_start(), self.outer));
        // `ray_parameter` returning `Some` already rules out a zero ray.
        let unit = ray_direction.normalize();
        let offset = nearest - ray_origin;
        let depth = offset.dot(unit);
        if depth <= 0.0 || (offset - unit * depth).length() > tolerance {
            return None;
        }
        Some(depth)
    }

    /// Translation a slide from shaft parameter `grab` to `cursor` asks for.
    pub fn drag_translation(&self, grab: f32, cursor: f32) -> Vec4 {
        self.axis.unit() * (cursor - grab)
    }

    /// Append the arrow as a stem chord plus [`ShaftStyle::head_barbs`]
    /// barbs. `scale` is the widget's, which is what the barb spread is read
    /// in.
    fn append_line_mesh(&self, style: &ShaftStyle, scale: f32, out: &mut LineMesh<3>) {
        let color = style.colors[self.axis as usize];
        let mut push = |from: Vec3, to: Vec3| {
            out.segments.push((from.to_array(), to.to_array()));
            out.colors.push((color, color));
            out.widths.push(style.width_px);
        };
        let tip = self.point(self.outer);
        push(self.point(self.inner), tip);

        let barb_root = self.point(self.head_start());
        let (across, up) = self.direction.any_orthonormal_pair();
        let radius = style.head_radius * scale;
        let step = std::f32::consts::TAU / style.head_barbs as f32;
        for k in 0..style.head_barbs {
            let angle = k as f32 * step;
            push(
                tip,
                barb_root + (across * angle.cos() + up * angle.sin()) * radius,
            );
        }
    }
}

/// Which handle, stripped of its geometry. The key an overlay dirties its
/// retained mesh against, and the one an editor persists.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum HandleId {
    Rotate(Plane4),
    Translate(Axis4),
}

/// A handle with the geometry a drag is read against.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Handle {
    Rotate(Ring),
    Translate(Shaft),
}

impl Handle {
    pub fn id(self) -> HandleId {
        match self {
            Self::Rotate(ring) => HandleId::Rotate(ring.plane),
            Self::Translate(shaft) => HandleId::Translate(shaft.axis),
        }
    }
}

/// What one drag asks the subject for. A handle drives exactly one of the
/// two, which is why this is a sum and not a pose delta with an identity
/// half.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TransformDelta {
    /// Turn by `angle` in `plane`.
    Rotate { plane: Plane4, angle: f32 },
    /// Slide `distance` world units along `axis`.
    Translate { axis: Axis4, distance: f32 },
}

impl TransformDelta {
    /// The rotation as a rotor, identity for a translation. Compose onto the
    /// subject's pose from the left, matching
    /// [`crate::hypergimbal::Ring::drag_rotor`].
    pub fn rotor(self) -> Rotor4 {
        match self {
            Self::Rotate { plane, angle } => (plane.unit_bivector() * angle).exp(),
            Self::Translate { .. } => Rotor4::IDENTITY,
        }
    }

    /// The translation in R⁴, zero for a rotation.
    pub fn translation(self) -> Vec4 {
        match self {
            Self::Rotate { .. } => Vec4::ZERO,
            Self::Translate { axis, distance } => axis.unit() * distance,
        }
    }
}

/// A held handle, anchored at the press edge so the whole drag is measured
/// against one origin rather than accumulated frame by frame.
///
/// The anchored copy is the handle as it stood at the press. A widget that
/// follows a subject the drag is moving therefore slides under a line that
/// does not, which is what keeps a translation drag from chasing itself.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum HandleDrag {
    Rotate {
        ring: Ring,
        /// Where the press ray met the ring's plane.
        grab: Vec3,
    },
    Translate {
        shaft: Shaft,
        /// Shaft parameter the press ray came closest to.
        grab: f32,
    },
}

impl HandleDrag {
    /// Anchor `handle` where the press ray meets it. `None` when the ray
    /// cannot be read against the handle, which the pick that produced the
    /// handle has already ruled out.
    pub fn press(handle: Handle, ray_origin: Vec3, ray_direction: Vec3) -> Option<Self> {
        match handle {
            Handle::Rotate(ring) => ring
                .ray_hit(ray_origin, ray_direction)
                .map(|grab| Self::Rotate { ring, grab }),
            Handle::Translate(shaft) => shaft
                .ray_parameter(ray_origin, ray_direction)
                .map(|grab| Self::Translate { shaft, grab }),
        }
    }

    pub fn id(&self) -> HandleId {
        match self {
            Self::Rotate { ring, .. } => HandleId::Rotate(ring.plane),
            Self::Translate { shaft, .. } => HandleId::Translate(shaft.axis),
        }
    }

    /// What the drag asks for with the cursor ray here, measured from the
    /// press anchor rather than from the previous frame.
    ///
    /// `None` while the ray cannot be read against the handle: a cursor off
    /// the window, or a camera that swung the handle edge-on. Callers hold
    /// the last delta rather than snapping the subject somewhere arbitrary.
    pub fn delta(&self, ray_origin: Vec3, ray_direction: Vec3) -> Option<TransformDelta> {
        match self {
            Self::Rotate { ring, grab } => {
                let cursor = ring.ray_hit(ray_origin, ray_direction)?;
                Some(TransformDelta::Rotate {
                    plane: ring.plane,
                    angle: ring.drag_angle(*grab, cursor),
                })
            }
            Self::Translate { shaft, grab } => {
                let cursor = shaft.ray_parameter(ray_origin, ray_direction)?;
                Some(TransformDelta::Translate {
                    axis: shaft.axis,
                    distance: cursor - grab,
                })
            }
        }
    }
}

/// Placement of the whole handle set in world R³.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TransformGizmo {
    /// World position of the subject the handles act on.
    pub center: Vec3,
    /// World length of one widget unit. The ring shell reaches `1 + √2` of
    /// it and the arrow tips just under `3`.
    pub scale: f32,
}

impl TransformGizmo {
    /// The rotation half on its own, for callers that want the projection
    /// (`Hypergimbal::project`) rather than the handles.
    pub fn hypergimbal(&self) -> Hypergimbal {
        Hypergimbal {
            center: self.center,
            scale: self.scale,
        }
    }

    /// All six rings, in [`Plane4::ALL`] order.
    pub fn rings(&self) -> [Ring; 6] {
        self.hypergimbal().rings()
    }

    /// All four shafts, in [`Axis4::ALL`] order.
    pub fn shafts(&self) -> [Shaft; 4] {
        Axis4::ALL.map(|axis| self.shaft(axis))
    }

    /// The shaft translating one axis.
    pub fn shaft(&self, axis: Axis4) -> Shaft {
        Shaft {
            axis,
            origin: self.center,
            direction: axis.shaft_direction(),
            inner: SHAFT_INNER * self.scale,
            outer: SHAFT_OUTER * self.scale,
            head: SHAFT_HEAD * self.scale,
        }
    }

    /// Handle the ray reaches within `tolerance`, or `None` when it reaches
    /// none of the ten.
    ///
    /// A shaft beats a ring wherever both are in range, rather than the two
    /// arbitrating by depth. The widget is one depth-free overlay and
    /// [`Self::append_line_mesh`] emits the shafts after the rings, so an
    /// arrow is the line actually on screen everywhere they cross; a pick
    /// that disagreed with what is drawn is a pick nobody can aim. Depth
    /// still separates shaft from shaft, and ring from ring.
    pub fn pick(&self, ray_origin: Vec3, ray_direction: Vec3, tolerance: f32) -> Option<Handle> {
        let mut nearest: Option<(f32, Shaft)> = None;
        for shaft in self.shafts() {
            let Some(depth) = shaft.hit_depth(ray_origin, ray_direction, tolerance) else {
                continue;
            };
            if nearest.is_none_or(|(best, _)| depth < best) {
                nearest = Some((depth, shaft));
            }
        }
        if let Some((_, shaft)) = nearest {
            return Some(Handle::Translate(shaft));
        }
        self.hypergimbal()
            .pick(ray_origin, ray_direction, tolerance)
            .map(Handle::Rotate)
    }

    /// Append every handle as chord runs. Existing contents are kept, so the
    /// widget can share a mesh with other overlay geometry.
    pub fn append_line_mesh(&self, style: &GizmoStyle, out: &mut LineMesh<3>) {
        self.hypergimbal().append_line_mesh(&style.rings, out);
        for shaft in self.shafts() {
            shaft.append_line_mesh(&style.shafts, self.scale, out);
        }
    }
}

/// Tessellation and colour for [`TransformGizmo::append_line_mesh`].
#[derive(Clone, Debug, Default)]
pub struct GizmoStyle {
    pub rings: RingStyle,
    pub shafts: ShaftStyle,
}

/// Arrow geometry and colour for the four translation shafts.
#[derive(Clone, Debug)]
pub struct ShaftStyle {
    /// Screen-space line width.
    pub width_px: f32,
    /// Barbs per arrowhead, spread evenly around the stem.
    pub head_barbs: usize,
    /// Barb spread, in widget scale units. The head's LENGTH is geometry,
    /// not style: it is the grabbable extent, so it cannot be restyled out
    /// of agreement with the pick.
    pub head_radius: f32,
    /// RGBA per axis, in [`Axis4::ALL`] order.
    pub colors: [[f32; 4]; 4],
}

impl ShaftStyle {
    /// `x`, `y`, `z` on the red / green / blue convention every 3D editor
    /// already trained the user on; `w` on a violet no scene axis uses, so
    /// the one shaft with no world direction is also the one with no
    /// borrowed hue.
    pub const AXIS_COLORS: [[f32; 4]; 4] = [
        [0.96, 0.36, 0.34, 1.0],
        [0.42, 0.90, 0.46, 1.0],
        [0.38, 0.58, 0.98, 1.0],
        [0.82, 0.44, 0.98, 1.0],
    ];
}

impl Default for ShaftStyle {
    fn default() -> Self {
        Self {
            width_px: 2.5,
            head_barbs: 4,
            head_radius: 0.09,
            colors: Self::AXIS_COLORS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_math::Rotor;
    use std::f32::consts::{PI, TAU};

    const WIDGET: TransformGizmo = TransformGizmo {
        center: Vec3::new(0.3, -0.7, 1.1),
        scale: 0.8,
    };

    /// Off-axis eye: no ring is seen edge-on, no two handles line up, and no
    /// shaft points at it.
    const EYE: Vec3 = Vec3::new(4.6, 4.4, 7.3);

    fn ray_to(target: Vec3) -> (Vec3, Vec3) {
        let eye = WIDGET.center + EYE;
        (eye, (target - eye).normalize())
    }

    fn tolerance() -> f32 {
        0.02 * WIDGET.scale
    }

    #[test]
    fn every_ring_and_every_shaft_is_reachable_by_pointing_at_it() {
        const SAMPLES: usize = 72;
        for plane in Plane4::ALL {
            let ring = WIDGET.hypergimbal().ring(plane);
            let hits = (0..SAMPLES)
                .filter(|step| {
                    let (origin, direction) =
                        ray_to(ring.point(*step as f32 / SAMPLES as f32 * TAU));
                    WIDGET
                        .pick(origin, direction, tolerance())
                        .is_some_and(|handle| handle.id() == HandleId::Rotate(plane))
                })
                .count();
            assert!(
                hits * 2 > SAMPLES,
                "{plane:?} selectable at only {hits}/{SAMPLES} points on its own ring"
            );
        }
        for axis in Axis4::ALL {
            let shaft = WIDGET.shaft(axis);
            for step in 0..=SAMPLES {
                let along = shaft.head_start() + shaft.head * step as f32 / SAMPLES as f32;
                let (origin, direction) = ray_to(shaft.point(along));
                let picked = WIDGET.pick(origin, direction, tolerance()).map(Handle::id);
                assert_eq!(
                    picked,
                    Some(HandleId::Translate(axis)),
                    "{axis:?} at {along} out picked {picked:?}"
                );
            }
        }
    }

    #[test]
    fn a_w_handle_drag_stays_in_its_own_plane_or_on_its_own_axis() {
        for plane in [Plane4::Xw, Plane4::Yw, Plane4::Zw] {
            let ring = WIDGET.hypergimbal().ring(plane);
            for step in 0..12 {
                let chi = -PI + step as f32 / 12.0 * TAU;
                let held = HandleDrag::Rotate {
                    ring,
                    grab: ring.point(0.0),
                };
                let (origin, direction) = ray_to(ring.point(chi));
                let delta = held.delta(origin, direction).expect("ring faces the eye");
                let log = delta.rotor().log();
                for other in Plane4::ALL {
                    if other == plane {
                        continue;
                    }
                    let component = log.component(other);
                    assert!(
                        component.abs() < 1e-5,
                        "{plane:?} drag leaked {component} into {other:?}"
                    );
                }
                let TransformDelta::Rotate { angle, .. } = delta else {
                    panic!("{plane:?} ring produced a translation");
                };
                // The log's own branch is the half-turn, so only compare
                // where the drag has not wrapped past it.
                if angle.abs() < PI - 0.1 && angle.abs() > 1e-3 {
                    assert!(
                        (log.component(plane) - angle).abs() < 1e-4,
                        "{plane:?} drag of {angle} logged as {}",
                        log.component(plane)
                    );
                }
                assert_eq!(delta.translation(), Vec4::ZERO);
            }
        }

        let shaft = WIDGET.shaft(Axis4::W);
        for grab in [-1.7_f32, 0.0, 0.4, 2.9] {
            for cursor in [-2.2_f32, -0.3, 1.1, 3.6] {
                let translation = shaft.drag_translation(grab, cursor);
                assert_eq!(translation.x, 0.0);
                assert_eq!(translation.y, 0.0);
                assert_eq!(translation.z, 0.0);
                assert_eq!(translation.w, cursor - grab);
                assert_eq!(
                    TransformDelta::Translate {
                        axis: Axis4::W,
                        distance: cursor - grab,
                    }
                    .rotor(),
                    Rotor4::IDENTITY
                );
            }
        }
    }

    #[test]
    fn pure_3d_rings_fix_the_w_axis_and_w_rings_move_it() {
        for (plane, fixes_w) in [
            (Plane4::Xy, true),
            (Plane4::Xz, true),
            (Plane4::Yz, true),
            (Plane4::Xw, false),
            (Plane4::Yw, false),
            (Plane4::Zw, false),
        ] {
            let rotor = TransformDelta::Rotate { plane, angle: 0.7 }.rotor();
            let moved = (rotor.apply(Vec4::W) - Vec4::W).length();
            assert_eq!(
                moved < 1e-6,
                fixes_w,
                "{plane:?} moved ê₄ by {moved}, which is the wrong side of the split"
            );
        }
    }

    #[test]
    fn the_w_shaft_is_the_direction_furthest_from_every_scene_axis() {
        let w = Axis4::W.shaft_direction();
        assert!((w.length() - 1.0).abs() < 1e-6);
        let scene = [Vec3::X, Vec3::Y, Vec3::Z];
        for axis in scene {
            assert!((w.dot(axis) + INV_SQRT_3).abs() < 1e-6, "not equiangular");
            assert!(w.dot(axis).abs() > 1e-3, "w lies in a coordinate plane");
        }
        let worst = |d: Vec3| {
            scene
                .iter()
                .fold(f32::NEG_INFINITY, |m, a| m.max(d.dot(*a)))
        };
        assert!((worst(w) + INV_SQRT_3).abs() < 1e-6);
        for step in 0..2048 {
            // Deterministic low-discrepancy sweep of S²: the golden-angle
            // spiral (Vogel 1979), which needs no RNG and no seed.
            let z = 1.0 - 2.0 * (step as f32 + 0.5) / 2048.0;
            let radius = (1.0 - z * z).max(0.0).sqrt();
            let phi = step as f32 * PI * (3.0 - 5.0_f32.sqrt());
            let probe = Vec3::new(radius * phi.cos(), radius * phi.sin(), z);
            assert!(
                worst(probe) >= worst(w) - 1e-6,
                "{probe} beats the w shaft's separation"
            );
        }
    }

    #[test]
    fn a_shaft_drag_asks_for_the_distance_the_cursor_slid_along_it() {
        for axis in Axis4::ALL {
            let shaft = WIDGET.shaft(axis);
            for grab_step in 0..5 {
                let grab_at = shaft.inner + (shaft.outer - shaft.inner) * grab_step as f32 / 5.0;
                let (origin, direction) = ray_to(shaft.point(grab_at));
                let held = HandleDrag::press(Handle::Translate(shaft), origin, direction)
                    .expect("the press ray reaches the shaft it was aimed at");
                for travel in [-0.9_f32, -0.2, 0.0, 0.35, 1.4] {
                    let (origin, direction) = ray_to(shaft.point(grab_at + travel));
                    let delta = held
                        .delta(origin, direction)
                        .expect("cursor ray reaches it");
                    let TransformDelta::Translate {
                        axis: got,
                        distance,
                    } = delta
                    else {
                        panic!("{axis:?} shaft produced a rotation");
                    };
                    assert_eq!(got, axis);
                    assert!(
                        (distance - travel).abs() < 1e-3,
                        "{axis:?}: slid {travel} and asked for {distance}"
                    );
                    assert_eq!(delta.translation(), axis.unit() * distance);
                }
            }
        }
    }

    #[test]
    fn no_handle_drifts_while_held_still_and_every_drag_is_odd_under_reversal() {
        fn drag(handle: Handle, from: Vec3, to: Vec3) -> TransformDelta {
            let (origin, direction) = ray_to(from);
            let held = HandleDrag::press(handle, origin, direction).expect("press reaches it");
            let (origin, direction) = ray_to(to);
            held.delta(origin, direction).expect("release reaches it")
        }

        for plane in Plane4::ALL {
            let ring = WIDGET.hypergimbal().ring(plane);
            let (grab, cursor) = (ring.point(0.4), ring.point(1.9));
            assert_eq!(
                drag(Handle::Rotate(ring), grab, grab),
                TransformDelta::Rotate { plane, angle: 0.0 },
                "{plane:?} drifted while held still"
            );
            let out = drag(Handle::Rotate(ring), grab, cursor).rotor();
            let back = drag(Handle::Rotate(ring), cursor, grab).rotor();
            let round_trip = (out.log() + back.log()).magnitude_squared();
            assert!(round_trip < 1e-8, "{plane:?} reversal left {round_trip}");
        }
        for axis in Axis4::ALL {
            let shaft = WIDGET.shaft(axis);
            let (grab, cursor) = (
                shaft.point(0.6 * shaft.outer),
                shaft.point(0.95 * shaft.outer),
            );
            assert_eq!(
                drag(Handle::Translate(shaft), grab, grab),
                TransformDelta::Translate {
                    axis,
                    distance: 0.0
                },
                "{axis:?} drifted while held still"
            );
            let out = drag(Handle::Translate(shaft), grab, cursor).translation();
            let back = drag(Handle::Translate(shaft), cursor, grab).translation();
            assert!((out + back).length() < 1e-4, "{axis:?} reversal left {out}");
        }
    }

    #[test]
    fn grazing_and_backward_rays_miss_the_shaft() {
        let shaft = WIDGET.shaft(Axis4::X);
        let eye = shaft.point(shaft.outer * 0.5) + Vec3::Y * 3.0;
        assert!(shaft.ray_parameter(eye, -Vec3::Y).is_some());
        assert!(
            shaft.hit_depth(eye, Vec3::Y, tolerance()).is_none(),
            "a shaft behind the eye must not hit"
        );
        assert!(
            shaft.ray_parameter(eye, shaft.direction).is_none(),
            "a ray along the shaft must miss"
        );
        let grazing = (shaft.direction + Vec3::Y * (MIN_SHAFT_INCIDENCE * 0.5)).normalize();
        assert!(
            shaft.ray_parameter(eye, grazing).is_none(),
            "a grazing ray must miss"
        );
        assert!(
            shaft.ray_parameter(eye, Vec3::ZERO).is_none(),
            "a degenerate ray must miss"
        );
    }

    #[test]
    fn a_ray_reaching_no_handle_picks_nothing() {
        let eye = WIDGET.center + Vec3::Z * 40.0;
        let miss = WIDGET.center + Vec3::Y * 60.0;
        assert!(WIDGET
            .pick(eye, (miss - eye).normalize(), tolerance())
            .is_none());
        // Through the hub, which the shafts' inner ends leave clear and
        // which sits inside every ring.
        assert!(WIDGET
            .pick(eye, (WIDGET.center - eye).normalize(), tolerance())
            .is_none());
    }

    #[test]
    fn arrowheads_clear_the_ring_shell_and_the_hub_stays_empty() {
        let shell = RING_REACH * WIDGET.scale;
        for shaft in WIDGET.shafts() {
            assert!(shaft.inner > 0.0, "a stem starts at the hub");
            assert!(
                shaft.head_start() > shell,
                "{:?}'s head starts at {}, inside the ring shell at {shell}",
                shaft.axis,
                shaft.head_start()
            );
            assert!((shaft.direction.length() - 1.0).abs() < 1e-6);
            assert_eq!(shaft.origin, WIDGET.center);
        }
        for ring in WIDGET.rings() {
            let reach = (ring.center - WIDGET.center).length() + ring.radius;
            assert!(
                reach <= shell + 1e-4,
                "a ring reaches {reach}, past {shell}"
            );
        }
        let shafts = WIDGET.shafts();
        for (i, first) in shafts.iter().enumerate() {
            for second in &shafts[i + 1..] {
                assert!(
                    first.direction.cross(second.direction).length() > 0.5,
                    "{:?} and {:?} point the same way",
                    first.axis,
                    second.axis
                );
            }
        }
    }

    #[test]
    fn line_mesh_is_well_formed_and_on_the_handles() {
        let style = GizmoStyle {
            rings: RingStyle {
                segments: 12,
                ..RingStyle::default()
            },
            shafts: ShaftStyle::default(),
        };
        let mut mesh = LineMesh::<3>::default();
        mesh.segments.push(([0.0; 3], [1.0; 3]));
        mesh.colors.push(([0.0; 4], [0.0; 4]));
        mesh.widths.push(1.0);

        WIDGET.append_line_mesh(&style, &mut mesh);
        let shaft_segments = 4 * (1 + style.shafts.head_barbs);
        assert_eq!(
            mesh.segments.len(),
            1 + 6 * style.rings.segments + shaft_segments
        );
        assert_eq!(mesh.colors.len(), mesh.segments.len());
        assert_eq!(mesh.widths.len(), mesh.segments.len());

        let head_span = style.shafts.head_radius * WIDGET.scale;
        for (index, (start, end)) in mesh.segments[1 + 6 * style.rings.segments..]
            .iter()
            .enumerate()
        {
            let shaft = WIDGET.shafts()[index / (1 + style.shafts.head_barbs)];
            for point in [Vec3::from_array(*start), Vec3::from_array(*end)] {
                let along = (point - shaft.origin).dot(shaft.direction);
                let off_axis = (point - shaft.point(along)).length();
                assert!(
                    (shaft.inner - 1e-4..=shaft.outer + 1e-4).contains(&along)
                        && off_axis <= head_span + 1e-4,
                    "{:?} emitted {point}, off its own arrow",
                    shaft.axis
                );
            }
        }
    }

    #[test]
    fn handle_geometry_is_bit_reproducible() {
        assert_eq!(WIDGET.shafts(), WIDGET.shafts());
        assert_eq!(WIDGET.rings(), WIDGET.rings());
        for axis in Axis4::ALL {
            assert_eq!(WIDGET.shaft(axis), WIDGET.shaft(axis));
        }
    }

    #[test]
    fn axis_labels_and_indices_match_the_vec4_component_order() {
        assert_eq!(
            Axis4::ALL.map(|a| a.label()),
            ["x", "y", "z", "w"],
            "labels drifted from the component order"
        );
        for (index, axis) in Axis4::ALL.into_iter().enumerate() {
            assert_eq!(axis as usize, index);
            assert_eq!(axis.unit().to_array()[index], 1.0);
            assert_eq!(axis.unit().length_squared(), 1.0);
        }
    }
}
