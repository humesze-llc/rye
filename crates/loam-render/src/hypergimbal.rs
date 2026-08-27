//! Hypergimbal: the six SO(4) rotation planes as six grabbable rings.
//!
//! The 16-cell's eight vertices are `±e₁..±e₄` on S³ and its six central
//! squares are inscribed in the six coordinate great circles, one per
//! rotation plane. Stereographic projection carries those great circles to
//! six circles in R³. Grabbing one and dragging along it asks for a rotation
//! in that plane, by the arc the drag swept.
//!
//! # Projection
//!
//! For `p ∈ S³` and a pole `n ∈ S³`, `σ(p) = (p − (p·n)n)/(1 − p·n)`, read
//! in an orthonormal frame of `n^⊥`. Stereographic projection sends circles
//! to circles (Coxeter, *Introduction to Geometry*, 2nd ed. 1969, §6.9), so
//! each great circle has a closed-form image.
//!
//! Take a rotation plane `P = span(a, b)` with `a, b` orthonormal, and set
//!
//! ```text
//!   α = a·n,  β = b·n,  ρ = √(α² + β²),  s = √(1 − ρ²),  φ = atan2(β, α)
//! ```
//!
//! Rotating the plane's basis by `φ` to `a' = (αa + βb)/ρ`,
//! `b' = (αb − βa)/ρ` makes the great circle `p(θ) = a cos θ + b sin θ`
//! read as `p = a' cos ψ + b' sin ψ` with `ψ = θ − φ`, and pins the pole
//! coupling to a single cosine: `p·n = ρ cos ψ`. Substituting into `σ` and
//! eliminating `ψ` gives a circle
//!
//! ```text
//!   centre  (ρ/s)·â,   radius  1/s,   in the plane span(â, b')
//!   with    â = (a' − ρn)/s
//! ```
//!
//! `ρ = 1` (the pole lying inside `P`) sends that circle to a straight line;
//! see [`POLE`] for why the pole here keeps `ρ = 1/√2` on all six planes.
//!
//! # Drag to angle
//!
//! Let `χ` be the angle around the projected circle, measured from `â`
//! toward `b'`. Reading off the substitution above,
//!
//! ```text
//!   cos χ = (cos ψ − ρ)/(1 − ρ cos ψ)
//!   sin χ = s·sin ψ /(1 − ρ cos ψ)
//! ```
//!
//! which is Kepler's true-anomaly / eccentric-anomaly relation with
//! eccentricity `ρ` (Danby, *Fundamentals of Celestial Mechanics*, 2nd ed.
//! 1992, §6.3): the projection runs along the ring at a non-uniform rate,
//! fast near the pole-facing side and slow away from it. Inverting,
//!
//! ```text
//!   ψ = atan2(s·sin χ, cos χ + ρ),    θ = ψ + φ
//! ```
//!
//! Both `1 ∓ ρ cos(·)` are strictly positive for `ρ < 1`, so dropping them
//! as a common factor leaves an `atan2` with no singularity and no branch;
//! the half-angle form `tan(ψ/2) = √((1−ρ)/(1+ρ))·tan(χ/2)` is the same map
//! and loses its second quadrant pair at `χ = ±π`.
//!
//! A drag from world point `g` to world point `c`, both taken on the ring's
//! plane, therefore asks for
//!
//! ```text
//!   Δθ = wrap(θ(c) − θ(g)),   R = exp(Δθ · ê_P)
//! ```
//!
//! `φ` cancels in the difference. `ρ` does not, and that is the whole point:
//! equal screen arcs at different places on the ring are different rotations,
//! and the map accounts for it.
//!
//! # Conventions
//!
//! The rings are the six *ambient* coordinate planes, fixed in world space.
//! They are handles, not a readout of the subject's current orientation, so
//! a drag produces a delta rotor for the caller to compose; nothing here
//! reads or stores the subject's pose.
//!
//! `exp(θ·ê_ij)` under [`loam_math::Rotor::apply`] carries `e_i` to
//! `e_i cos θ + e_j sin θ`, matching the `p(θ)` parametrisation above, so a
//! drag that sweeps the ring forward turns the subject the same way.

use glam::{Vec3, Vec4};
use loam_math::{Bivector, Plane4, Rotor4};
use loam_shape::LineMesh;

/// A cell centre of the 16-cell, `(1,1,1,1)/2`. It lies on none of the six
/// coordinate great circles, so every ring is a finite circle, and
/// `|proj_P(n)| = 1/√2` on all six planes, so the rings are congruent:
/// radius `√2`, centre at distance `1`. Projecting from `±ê₄` instead puts
/// the pole inside the `xw`, `yw` and `zw` planes and sends those three
/// rings to straight lines through the origin.
pub const POLE: Vec4 = Vec4::new(0.5, 0.5, 0.5, 0.5);

// Sine of the smallest ray-to-ring-plane angle the plane hit is trusted at.
// The hit distance grows as `1/sin`, so below this a one-pixel cursor move
// sweeps an arbitrary arc and the drag stops tracking the cursor.
const MIN_PLANE_INCIDENCE: f32 = 1e-2;

// `1 − p·n` below this puts `p` within f32 noise of the pole, where the
// image runs to infinity.
const MIN_POLE_DISTANCE: f32 = 1e-4;

// Orthonormal frame of `POLE`'s orthogonal complement, read as the image
// R³'s x, y, z: the other three 16-cell cell centres orthogonal to `POLE`.
// It lands the six ring centres exactly on `±x̂, ±ŷ, ±ẑ`, one
// orthogonal-plane pair per axis, and `det[u₁,u₂,u₃,POLE] = +1`, so the
// image inherits R⁴'s orientation. A frame with a zero component puts a
// ring plane on an image axis, where the default camera sees it edge-on
// and cannot grab it at all.
const IMAGE_AXES: [Vec4; 3] = [
    Vec4::new(0.5, 0.5, -0.5, -0.5),
    Vec4::new(0.5, -0.5, 0.5, -0.5),
    Vec4::new(0.5, -0.5, -0.5, 0.5),
];

// The pole-parallel part is dropped, which is exactly what stereographic
// projection does with it.
fn to_r3(q: Vec4) -> Vec3 {
    Vec3::new(
        IMAGE_AXES[0].dot(q),
        IMAGE_AXES[1].dot(q),
        IMAGE_AXES[2].dot(q),
    )
}

// In the `p(θ) = a cos θ + b sin θ` order that `Plane4::unit_bivector`
// rotates `a` toward `b`.
fn plane_axes(plane: Plane4) -> (Vec4, Vec4) {
    match plane {
        Plane4::Xy => (Vec4::X, Vec4::Y),
        Plane4::Xz => (Vec4::X, Vec4::Z),
        Plane4::Xw => (Vec4::X, Vec4::W),
        Plane4::Yz => (Vec4::Y, Vec4::Z),
        Plane4::Yw => (Vec4::Y, Vec4::W),
        Plane4::Zw => (Vec4::Z, Vec4::W),
    }
}

fn wrap_pi(angle: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    let wrapped = angle.rem_euclid(TAU);
    if wrapped > PI {
        wrapped - TAU
    } else {
        wrapped
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Hypergimbal {
    pub center: Vec3,
    /// World length of one unit of the stereographic image. Rings come out
    /// at radius `√2·scale`, centred `scale` from [`Self::center`].
    pub scale: f32,
}

impl Hypergimbal {
    /// `None` within f32 noise of [`POLE`], where the image is unbounded.
    pub fn project(&self, p: Vec4) -> Option<Vec3> {
        let denom = 1.0 - p.dot(POLE);
        if denom.abs() < MIN_POLE_DISTANCE {
            return None;
        }
        Some(self.center + to_r3(p) / denom * self.scale)
    }

    /// All six rings, in [`Plane4::ALL`] order.
    pub fn rings(&self) -> [Ring; 6] {
        Plane4::ALL.map(|plane| self.ring(plane))
    }

    pub fn ring(&self, plane: Plane4) -> Ring {
        let (a, b) = plane_axes(plane);
        let alpha = a.dot(POLE);
        let beta = b.dot(POLE);
        let pole_overlap = (alpha * alpha + beta * beta).sqrt();
        // POLE meets every coordinate plane at overlap 1/√2, so `s` is
        // bounded away from zero and the circle never degenerates to a line.
        let s = (1.0 - pole_overlap * pole_overlap).sqrt();
        // Phase-aligned plane basis: `a_phase` carries the pole's whole
        // in-plane component, leaving `b_phase` orthogonal to the pole.
        let (a_phase, b_phase) = (
            (a * alpha + b * beta) / pole_overlap,
            (b * alpha - a * beta) / pole_overlap,
        );
        let u = to_r3((a_phase - POLE * pole_overlap) / s);
        Ring {
            plane,
            center: self.center + u * (self.scale * pole_overlap / s),
            u,
            v: to_r3(b_phase),
            radius: self.scale / s,
            pole_overlap,
            phase: beta.atan2(alpha),
        }
    }

    /// Nearest ring within `tolerance` of the ray; ties break by hit depth.
    ///
    /// The test is against the plane hit, not the true ray-to-circle distance (a
    /// quartic): [`Ring::ray_hit`] rejects an edge-on ring rather than grabbing
    /// it at a distance the drag cannot track.
    pub fn pick(&self, ray_origin: Vec3, ray_direction: Vec3, tolerance: f32) -> Option<Ring> {
        let mut best: Option<(f32, Ring)> = None;
        for ring in self.rings() {
            let Some(hit) = ring.ray_hit(ray_origin, ray_direction) else {
                continue;
            };
            if ((hit - ring.center).length() - ring.radius).abs() > tolerance {
                continue;
            }
            let depth = (hit - ray_origin).length();
            if best.is_none_or(|(nearest, _)| depth < nearest) {
                best = Some((depth, ring));
            }
        }
        best.map(|(_, ring)| ring)
    }

    /// Existing contents are kept, so the widget can share a mesh.
    pub fn append_line_mesh(&self, style: &RingStyle, out: &mut LineMesh<3>) {
        let step = std::f32::consts::TAU / style.segments as f32;
        for ring in self.rings() {
            let color = style.colors[ring.plane as usize];
            let mut prev = ring.point(0.0);
            for k in 1..=style.segments {
                let next = ring.point(k as f32 * step);
                out.segments.push((prev.to_array(), next.to_array()));
                out.colors.push((color, color));
                out.widths.push(style.width_px);
                prev = next;
            }
        }
    }
}

/// The stereographic image of one plane's great circle, plus the two
/// parameters of the map derived in the module docs.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Ring {
    pub plane: Plane4,
    pub center: Vec3,
    /// In-plane unit axis `χ` is measured from.
    pub u: Vec3,
    /// In-plane unit axis `χ` is measured toward.
    pub v: Vec3,
    pub radius: f32,
    /// `ρ`: length of the pole's projection onto this rotation plane, the
    /// eccentricity of the ring-angle-to-arc-angle map.
    pub pole_overlap: f32,
    /// `φ`: great-circle parameter of the ring's `χ = 0` point.
    pub phase: f32,
}

impl Ring {
    /// Normal of the circle's plane, `u × v`, so `χ` runs counter-clockwise
    /// looking down it.
    pub fn normal(&self) -> Vec3 {
        self.u.cross(self.v)
    }

    pub fn point(&self, chi: f32) -> Vec3 {
        self.center + (self.u * chi.cos() + self.v * chi.sin()) * self.radius
    }

    /// Ring angle `χ` of a world point, projected onto the circle's plane.
    pub fn ring_angle(&self, world: Vec3) -> f32 {
        let offset = world - self.center;
        offset.dot(self.v).atan2(offset.dot(self.u))
    }

    /// Great-circle parameter `θ` at ring angle `χ`.
    pub fn arc_angle(&self, chi: f32) -> f32 {
        let s = (1.0 - self.pole_overlap * self.pole_overlap).sqrt();
        (s * chi.sin()).atan2(chi.cos() + self.pole_overlap) + self.phase
    }

    /// Where the ray crosses the circle's plane. `None` behind the eye, or
    /// within 0.6° of parallel, where the hit distance blows up as `1/sin`
    /// and one pixel of cursor motion sweeps an arbitrary arc.
    pub fn ray_hit(&self, ray_origin: Vec3, ray_direction: Vec3) -> Option<Vec3> {
        let normal = self.normal();
        let incidence = ray_direction.dot(normal);
        if incidence.abs() < MIN_PLANE_INCIDENCE * ray_direction.length() {
            return None;
        }
        let t = (self.center - ray_origin).dot(normal) / incidence;
        (t > 0.0).then(|| ray_origin + ray_direction * t)
    }

    /// Wrapped to `(−π, π]`. Both points are taken on the circle's plane; only
    /// their bearing from the centre matters, not their distance from it.
    pub fn drag_angle(&self, grab: Vec3, cursor: Vec3) -> f32 {
        wrap_pi(self.arc_angle(self.ring_angle(cursor)) - self.arc_angle(self.ring_angle(grab)))
    }

    /// Compose onto the subject's pose; this type holds no pose of its own.
    pub fn drag_rotor(&self, grab: Vec3, cursor: Vec3) -> Rotor4 {
        (self.plane.unit_bivector() * self.drag_angle(grab, cursor)).exp()
    }
}

#[derive(Clone, Debug)]
pub struct RingStyle {
    pub segments: usize,
    pub width_px: f32,
    /// RGBA per plane, in [`Plane4::ALL`] order.
    pub colors: [[f32; 4]; 6],
}

impl RingStyle {
    /// One hue per orthogonal-plane pair, so the three Hopf links read as three
    /// colours; within a pair the pure-3D ring is bright and the w-coupled dim.
    pub const PAIRED_HUE_COLORS: [[f32; 4]; 6] = [
        [0.95, 0.30, 0.32, 1.0],
        [0.35, 0.85, 0.40, 1.0],
        [0.30, 0.42, 0.75, 1.0],
        [0.38, 0.62, 0.98, 1.0],
        [0.24, 0.55, 0.30, 1.0],
        [0.70, 0.25, 0.28, 1.0],
    ];
}

impl Default for RingStyle {
    fn default() -> Self {
        Self {
            segments: 96,
            width_px: 2.5,
            colors: Self::PAIRED_HUE_COLORS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_math::Rotor;
    use std::f32::consts::{PI, TAU};

    const WIDGET: Hypergimbal = Hypergimbal {
        center: Vec3::new(0.3, -0.7, 1.1),
        scale: 0.8,
    };

    fn great_circle_point(plane: Plane4, theta: f32) -> Vec4 {
        let (a, b) = plane_axes(plane);
        a * theta.cos() + b * theta.sin()
    }

    fn assert_close(actual: f32, expected: f32, tol: f32, what: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{what}: {actual} != {expected}"
        );
    }

    #[test]
    fn all_six_rings_are_finite_and_congruent() {
        let rings = WIDGET.rings();
        for ring in rings {
            assert_close(
                ring.pole_overlap,
                std::f32::consts::FRAC_1_SQRT_2,
                1e-6,
                "pole overlap",
            );
            assert_close(
                ring.radius,
                WIDGET.scale * 2.0_f32.sqrt(),
                1e-5,
                "ring radius",
            );
            assert_close(
                (ring.center - WIDGET.center).length(),
                WIDGET.scale,
                1e-5,
                "centre offset",
            );
            assert_close(ring.u.length(), 1.0, 1e-5, "u unit");
            assert_close(ring.v.length(), 1.0, 1e-5, "v unit");
            assert_close(ring.u.dot(ring.v), 0.0, 1e-5, "u ⟂ v");
        }
        for (i, first) in rings.iter().enumerate() {
            for second in &rings[i + 1..] {
                assert!(
                    first.normal().cross(second.normal()).length() > 1e-3,
                    "{:?} and {:?} share a circle plane",
                    first.plane,
                    second.plane
                );
            }
        }
    }

    #[test]
    fn dual_plane_pairs_sit_at_antipodal_ring_centres() {
        let offset = |plane| (WIDGET.ring(plane).center - WIDGET.center) / WIDGET.scale;
        for (axis, first, second) in [
            (Vec3::X, Plane4::Xy, Plane4::Zw),
            (Vec3::Y, Plane4::Xz, Plane4::Yw),
            (Vec3::Z, Plane4::Xw, Plane4::Yz),
        ] {
            assert!((offset(first) - axis).length() < 1e-5, "{first:?} off axis");
            assert!(
                (offset(second) + axis).length() < 1e-5,
                "{second:?} not antipodal to {first:?}"
            );
        }
        for ring in WIDGET.rings() {
            let normal = ring.normal();
            for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                assert!(
                    normal.dot(axis).abs() < 0.99,
                    "{:?}'s plane is square to an image axis",
                    ring.plane
                );
            }
        }
    }

    #[test]
    fn ring_is_the_projected_great_circle() {
        for plane in Plane4::ALL {
            let ring = WIDGET.ring(plane);
            for step in 0..24 {
                let theta = step as f32 / 24.0 * TAU;
                let projected = WIDGET
                    .project(great_circle_point(plane, theta))
                    .expect("no coordinate great circle passes through the pole");
                assert_close(
                    (projected - ring.center).length(),
                    ring.radius,
                    2e-4,
                    "off the circle",
                );
                let chi = ring.ring_angle(projected);
                assert!(
                    (ring.point(chi) - projected).length() < 2e-4,
                    "{plane:?} θ={theta}: ring.point(χ) != σ(p(θ))"
                );
            }
        }
    }

    #[test]
    fn arc_angle_inverts_the_projection_along_each_ring() {
        for plane in Plane4::ALL {
            let ring = WIDGET.ring(plane);
            for step in 0..37 {
                let theta = -PI + step as f32 / 37.0 * TAU;
                let projected = WIDGET.project(great_circle_point(plane, theta)).unwrap();
                let recovered = ring.arc_angle(ring.ring_angle(projected));
                assert_close(
                    wrap_pi(recovered - theta),
                    0.0,
                    2e-4,
                    &format!("{plane:?} arc angle at θ={theta}"),
                );
            }
        }
    }

    #[test]
    fn drag_yields_the_rotation_its_plane_predicts() {
        for plane in Plane4::ALL {
            let ring = WIDGET.ring(plane);
            for grab_step in 0..8 {
                let theta = -PI + grab_step as f32 / 8.0 * TAU;
                for delta in [-1.1_f32, -0.4, 0.05, 0.9, 2.2] {
                    let start = great_circle_point(plane, theta);
                    let end = great_circle_point(plane, theta + delta);
                    let grab = WIDGET.project(start).unwrap();
                    let cursor = WIDGET.project(end).unwrap();

                    assert_close(
                        wrap_pi(ring.drag_angle(grab, cursor) - delta),
                        0.0,
                        1e-3,
                        &format!("{plane:?} drag angle from θ={theta} by {delta}"),
                    );

                    let rotor = ring.drag_rotor(grab, cursor);
                    let expected = (plane.unit_bivector() * delta).exp();
                    for (got, want) in <[f32; 8]>::from(rotor)
                        .into_iter()
                        .zip(<[f32; 8]>::from(expected))
                    {
                        assert_close(got, want, 1e-3, &format!("{plane:?} rotor component"));
                    }
                    assert!(
                        (rotor.apply(start) - end).length() < 2e-3,
                        "{plane:?}: rotor does not carry the grabbed point to the cursor"
                    );
                }
            }
        }
    }

    #[test]
    fn every_plane_is_pickable_over_most_of_its_arc() {
        // Off-axis eye so no ring is seen edge-on and no two rings line up.
        let eye = WIDGET.center + Vec3::new(4.3, 5.1, 6.7);
        const SAMPLES: usize = 72;
        for plane in Plane4::ALL {
            let ring = WIDGET.ring(plane);
            let hits = (0..SAMPLES)
                .filter(|step| {
                    let target = ring.point(*step as f32 / SAMPLES as f32 * TAU);
                    WIDGET
                        .pick(eye, (target - eye).normalize(), 0.02 * WIDGET.scale)
                        .is_some_and(|picked| picked.plane == plane)
                })
                .count();
            assert!(
                hits * 2 > SAMPLES,
                "{plane:?} selectable at only {hits}/{SAMPLES} points on its own ring"
            );
        }
    }

    #[test]
    fn drag_angle_is_zero_at_rest_and_odd_under_reversal() {
        for plane in Plane4::ALL {
            let ring = WIDGET.ring(plane);
            let grab = ring.point(0.4);
            let cursor = ring.point(1.9);
            assert_eq!(ring.drag_angle(grab, grab), 0.0);
            assert_close(
                ring.drag_angle(grab, cursor) + ring.drag_angle(cursor, grab),
                0.0,
                1e-5,
                "reversal",
            );
        }
    }

    #[test]
    fn drag_across_the_seam_takes_the_short_way() {
        for plane in Plane4::ALL {
            let ring = WIDGET.ring(plane);
            let grab = ring.point(PI - 0.05);
            let cursor = ring.point(-PI + 0.05);
            let angle = ring.drag_angle(grab, cursor);
            assert!(
                angle.abs() < 0.6,
                "{plane:?}: seam crossing gave {angle} rad"
            );
        }
    }

    #[test]
    fn drag_rotors_stay_unit_norm() {
        for plane in Plane4::ALL {
            let ring = WIDGET.ring(plane);
            for step in 0..64 {
                let chi = -PI + step as f32 / 64.0 * TAU;
                let rotor = ring.drag_rotor(ring.point(0.0), ring.point(chi));
                assert_close(rotor.norm_squared(), 1.0, 1e-5, "rotor norm");
            }
        }
    }

    #[test]
    fn grazing_and_backward_rays_miss_the_ring_plane() {
        let ring = WIDGET.ring(Plane4::Xy);
        let eye = ring.center + ring.normal() * 3.0;
        assert!(ring.ray_hit(eye, -ring.normal()).is_some());
        assert!(
            ring.ray_hit(eye, ring.normal()).is_none(),
            "plane behind the eye must not hit"
        );
        // Ray in the circle's own plane, then tilted just under the
        // incidence floor.
        assert!(
            ring.ray_hit(eye, ring.u).is_none(),
            "parallel ray must miss"
        );
        let grazing = (ring.u + ring.normal() * (MIN_PLANE_INCIDENCE * 0.5)).normalize();
        assert!(
            ring.ray_hit(eye, grazing).is_none(),
            "grazing ray must miss"
        );
    }

    #[test]
    fn a_ray_missing_every_ring_picks_nothing() {
        let eye = WIDGET.center + Vec3::new(0.0, 0.0, 40.0);
        let miss = WIDGET.center + Vec3::new(0.0, 60.0, 0.0);
        assert!(WIDGET
            .pick(eye, (miss - eye).normalize(), 0.02 * WIDGET.scale)
            .is_none());
    }

    #[test]
    fn line_mesh_is_well_formed_and_on_the_rings() {
        let style = RingStyle {
            segments: 12,
            ..RingStyle::default()
        };
        let mut mesh = LineMesh::<3>::default();
        mesh.segments.push(([0.0; 3], [1.0; 3]));
        mesh.colors.push(([0.0; 4], [0.0; 4]));
        mesh.widths.push(1.0);

        WIDGET.append_line_mesh(&style, &mut mesh);
        assert_eq!(mesh.segments.len(), 1 + 6 * style.segments);
        assert_eq!(mesh.colors.len(), mesh.segments.len());
        assert_eq!(mesh.widths.len(), mesh.segments.len());

        let radii: Vec<f32> = mesh.segments[1..]
            .iter()
            .map(|(start, _)| (Vec3::from_array(*start) - WIDGET.center).length())
            .collect();
        // All six rings share a radius, so a chord endpoint's distance from the
        // widget centre is bounded by the ring's reach either side of its centre.
        for radius in radii {
            assert!(
                (radius - WIDGET.scale).abs() <= WIDGET.scale * 2.0_f32.sqrt() + 1e-4,
                "chord endpoint {radius} outside the ring's reach"
            );
        }
        let first = mesh.segments[1].0;
        let last = mesh.segments[style.segments].1;
        assert!((Vec3::from_array(first) - Vec3::from_array(last)).length() < 1e-4);
    }

    #[test]
    fn the_pole_has_no_image() {
        assert!(WIDGET.project(POLE).is_none());
        assert!(WIDGET.project(POLE * 0.99999).is_none());
        assert!(WIDGET.project(-POLE).is_some());
    }

    #[test]
    fn ring_geometry_is_bit_reproducible() {
        for plane in Plane4::ALL {
            assert_eq!(WIDGET.ring(plane), WIDGET.ring(plane));
        }
        assert_eq!(WIDGET.rings(), WIDGET.rings());
    }
}
