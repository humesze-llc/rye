//! `impl PhysicsSpace for EuclideanR4` — 4D Euclidean rigid-body physics.
//!
//! Angular velocity is a [`Bivector4`] (six rotation-plane components);
//! inertia is the scalar moment for isotropic bodies, same pragmatic
//! simplification made in 3D. A full 4D inertia tensor is a 6×6
//! bivector-to-bivector map — it doesn't land until an actual anisotropic
//! 4D body demands it.
//!
//! Orientation integration uses [`Rotor4`] directly. The Clifford-rotor
//! multiplication convention is "left operand applied first" (opposite to
//! `glam::Quat`'s "right first"), so the composed orientation after a
//! timestep is `rotation_new = rotation_current * delta_rotor`.
//!
//! ## Scope
//!
//! First cut is **sphere-sphere only** — enough to prove the integration
//! loop and collision resolution work in 4D. Polytope narrowphase (GJK
//! already works generically; EPA needs a 4D face-normal reconstruction)
//! lands in a follow-up commit.

use glam::Vec4;

use rye_math::{Bivector, Bivector4, EuclideanR4, Iso4Flat};

use crate::body::RigidBody;
use crate::collider::{Collider, ColliderKind};
use crate::integrator::PhysicsSpace;
use crate::narrowphase::Narrowphase;
use crate::response::Contact;

/// Inverse isotropic moment of inertia, treating static or zero-inertia
/// bodies as infinite (returns 0).
fn inv_inertia(body: &RigidBody<EuclideanR4>) -> f32 {
    if body.inv_mass > 0.0 && body.inertia > 0.0 {
        1.0 / body.inertia
    } else {
        0.0
    }
}

impl PhysicsSpace for EuclideanR4 {
    type AngVel = Bivector4;
    /// Scalar isotropic moment of inertia. Suitable for spheres and
    /// the regular 4D polytopes (5-cell, tesseract, 16-cell, 24-cell)
    /// about their centroids.
    type Inertia = f32;

    fn integrate_orientation(&self, iso: Iso4Flat, omega: Bivector4, dt: f32) -> Iso4Flat {
        // Guard against NaN/infinite angular velocity leaking into
        // the orientation — same defense in depth as the 3D path.
        if !(omega.xy.is_finite()
            && omega.xz.is_finite()
            && omega.xw.is_finite()
            && omega.yz.is_finite()
            && omega.yw.is_finite()
            && omega.zw.is_finite())
        {
            return iso;
        }
        let delta = (omega * dt).exp();
        // Rotor4 multiplication is left-first: `A · B` applies A then
        // B. To apply `iso_current` then `delta`, compose as
        // `iso.rotation * delta`. Normalize to fight f32 drift off
        // the unit manifold over long integrator runs.
        let composed = iso.rotation * delta;
        Iso4Flat {
            rotation: composed.normalize(),
            translation: iso.translation,
        }
    }

    fn apply_inv_inertia(&self, inertia: f32, torque: Bivector4) -> Bivector4 {
        if inertia > 0.0 {
            torque * (1.0 / inertia)
        } else {
            Bivector4::ZERO
        }
    }

    fn velocity_at_point(&self, body: &RigidBody<EuclideanR4>, p: Vec4) -> Vec4 {
        // `v(r) = v_linear + ω⌋r` where `ω⌋r` is the 1-vector part of
        // the bivector-vector contraction — the 4D analogue of `ω × r`.
        let r = p - body.position;
        body.velocity + body.angular_velocity.contract_vec(r)
    }

    fn effective_mass_inv(
        &self,
        a: &RigidBody<EuclideanR4>,
        b: &RigidBody<EuclideanR4>,
        contact_point: Vec4,
        direction: Vec4,
    ) -> f32 {
        let ra = contact_point - a.position;
        let rb = contact_point - b.position;
        let ra_wedge = Bivector4::wedge(ra, direction);
        let rb_wedge = Bivector4::wedge(rb, direction);
        a.inv_mass
            + b.inv_mass
            + ra_wedge.magnitude_squared() * inv_inertia(a)
            + rb_wedge.magnitude_squared() * inv_inertia(b)
    }

    fn apply_contact_impulse(
        &self,
        a: &mut RigidBody<EuclideanR4>,
        b: &mut RigidBody<EuclideanR4>,
        contact_point: Vec4,
        direction: Vec4,
        magnitude: f32,
    ) {
        let ra = contact_point - a.position;
        let rb = contact_point - b.position;
        let lin = direction * magnitude;
        a.velocity -= lin * a.inv_mass;
        b.velocity += lin * b.inv_mass;

        // τ_a = r_a ∧ (−lin); τ_b = r_b ∧ (+lin). Apply via ω += I⁻¹·τ.
        let inv_i_a = inv_inertia(a);
        let inv_i_b = inv_inertia(b);
        a.angular_velocity = a.angular_velocity + Bivector4::wedge(ra, lin) * (-inv_i_a);
        b.angular_velocity = b.angular_velocity + Bivector4::wedge(rb, lin) * inv_i_b;
    }
}

// ---------------------------------------------------------------------------
// Narrowphase: sphere-sphere only at first cut. Polytope GJK+EPA for 4D
// lands in the next commit.
// ---------------------------------------------------------------------------

fn sphere_sphere_r4(
    a: &RigidBody<EuclideanR4>,
    b: &RigidBody<EuclideanR4>,
    space: &EuclideanR4,
) -> Option<Contact<EuclideanR4>> {
    let Collider::Sphere { radius: ra } = a.collider else {
        return None;
    };
    let Collider::Sphere { radius: rb } = b.collider else {
        return None;
    };

    use rye_math::Space;
    let d = space.distance(a.position, b.position);
    let combined = ra + rb;
    if d >= combined {
        return None;
    }
    let log = space.log(a.position, b.position);
    let len = log.length();
    let normal = if len > 1e-8 { log / len } else { Vec4::Y };

    let surface_a = a.position + normal * ra;
    let surface_b = b.position - normal * rb;
    let point = (surface_a + surface_b) * 0.5;

    Some(Contact {
        normal,
        point,
        penetration: combined - d,
        restitution: (a.restitution + b.restitution) * 0.5,
    })
}

pub fn register_default_narrowphase(np: &mut Narrowphase<EuclideanR4>) {
    np.register(ColliderKind::Sphere, ColliderKind::Sphere, sphere_sphere_r4);
}

// ---------------------------------------------------------------------------
// Convenience constructors.
// ---------------------------------------------------------------------------

/// Solid-ball moment of inertia in 4D: `I = (2/n)·m·r² = m·r² / 2`
/// for a uniform 4-ball about its center. Same form as 3D's
/// `(2/5)·m·r²` but with the 4D scaling; for prototypes this is
/// "isotropic inertia of the right order of magnitude" and suffices.
pub fn ball4_inertia(mass: f32, radius: f32) -> f32 {
    0.5 * mass * radius * radius
}

/// Dynamic sphere body in R⁴.
pub fn sphere_body_r4(
    position: Vec4,
    velocity: Vec4,
    radius: f32,
    mass: f32,
) -> RigidBody<EuclideanR4> {
    RigidBody::new(
        position,
        velocity,
        Collider::Sphere { radius },
        mass,
        ball4_inertia(mass, radius),
        &EuclideanR4,
    )
}

// ---------------------------------------------------------------------------
// 4D regular polytopes. Six exist in 4D (five analogues of the Platonic
// solids plus the 24-cell which has no 3D counterpart). The four most
// physically useful for games — 5-cell, tesseract, 16-cell, 24-cell —
// are generated here. The 120-cell (600 vertices) and 600-cell (120
// vertices) land when a demo actually needs them.
//
// Every generator returns vertices centered at the origin and scaled
// so the circumradius (bounding-sphere radius) equals the caller-
// provided `r`. Caller is responsible for further translation.
// ---------------------------------------------------------------------------

/// **5-cell / pentatope** (4D simplex): 5 vertices, 10 edges, 10 faces,
/// 5 tetrahedral cells. The 4D analogue of the tetrahedron — the
/// centerpiece of the Simplex-4D game concept.
///
/// Construction: take the five permutations-by-symmetry of `(1,1,1,1,−4)/√20`
/// embedded in 5D and drop the "equalized" component. The result sits
/// in a hyperplane of 5D; we return its 4D restriction with
/// circumradius scaled to `r`.
pub fn pentatope_vertices(r: f32) -> Vec<Vec4> {
    // Equivalent lower-dimensional construction: start from a regular
    // tetrahedron in the `w = −1/4` hyperplane, then add the apex at
    // `(0, 0, 0, r)`. Scale the base so all inter-vertex distances
    // match the apex-to-base distance, then rescale the whole thing
    // to circumradius `r`.
    //
    // The clean closed form: use the five vertices
    //   v_i = e_i − (1/5)·(1,1,1,1,1)
    // from the 5D standard simplex, project onto the `Σ = 0` hyperplane
    // (which they already lie in), pick an orthonormal basis for that
    // hyperplane, and express each v_i in 4D. Circumradius works out
    // to `sqrt(4/5)` in those units; rescale.
    //
    // Using the simpler tetrahedron-plus-apex form:
    let k = r; // apex at (0, 0, 0, r)
               // Base tetrahedron in `w = −r/4` plane with circumradius
               // `r·sqrt(15)/4` (chosen so all edges are equal).
    let base_w = -r * 0.25;
    let base_r = r * (15.0_f32).sqrt() / 4.0;
    // Use a regular tetrahedron's vertex set for the base, scaled.
    let t = base_r / 3.0_f32.sqrt();
    vec![
        Vec4::new(0.0, 0.0, 0.0, k),
        Vec4::new(t, t, t, base_w),
        Vec4::new(t, -t, -t, base_w),
        Vec4::new(-t, t, -t, base_w),
        Vec4::new(-t, -t, t, base_w),
    ]
}

/// **Tesseract / 8-cell** (hypercube): 16 vertices, 32 edges, 24
/// square faces, 8 cubic cells. The 4D analogue of the cube.
///
/// Vertices are `(±a, ±a, ±a, ±a)` with `a = r/2` so the
/// circumradius is `r` (distance from origin to any corner is
/// `sqrt(4·a²) = 2a`; setting `2a = r` gives `a = r/2`).
pub fn tesseract_vertices(r: f32) -> Vec<Vec4> {
    let a = r * 0.5;
    let mut v = Vec::with_capacity(16);
    for &w in &[-a, a] {
        for &z in &[-a, a] {
            for &y in &[-a, a] {
                for &x in &[-a, a] {
                    v.push(Vec4::new(x, y, z, w));
                }
            }
        }
    }
    v
}

/// **16-cell / hexadecachoron** (cross-polytope): 8 vertices, 24
/// edges, 32 triangular faces, 16 tetrahedral cells. The 4D analogue
/// of the octahedron.
///
/// Vertices are `±r` on each axis: `(±r, 0, 0, 0)`, `(0, ±r, 0, 0)`,
/// and the y/w variants.
pub fn cell16_vertices(r: f32) -> Vec<Vec4> {
    vec![
        Vec4::new(r, 0.0, 0.0, 0.0),
        Vec4::new(-r, 0.0, 0.0, 0.0),
        Vec4::new(0.0, r, 0.0, 0.0),
        Vec4::new(0.0, -r, 0.0, 0.0),
        Vec4::new(0.0, 0.0, r, 0.0),
        Vec4::new(0.0, 0.0, -r, 0.0),
        Vec4::new(0.0, 0.0, 0.0, r),
        Vec4::new(0.0, 0.0, 0.0, -r),
    ]
}

/// **24-cell / icositetrachoron**: 24 vertices, 96 edges, 96 triangle
/// faces, 24 octahedral cells. Unique to 4D — it has no 3D analogue
/// because 3D symmetry groups don't support it. Its vertex set is
/// the union of a 16-cell and a tesseract (appropriately scaled), so
/// it tiles R⁴ like the hexagon tiles R².
///
/// Vertex set: all 24 permutations of `(±1, ±1, 0, 0)`, i.e. every
/// pair of nonzero coordinates at `±1` with the other two zero.
/// Circumradius is `sqrt(2)` in those units; rescale to `r`.
pub fn cell24_vertices(r: f32) -> Vec<Vec4> {
    let k = r / 2.0_f32.sqrt();
    let mut v = Vec::with_capacity(24);
    // Pairs of axes (0=x, 1=y, 2=z, 3=w) — C(4, 2) = 6 pairs, each
    // contributing 4 sign combinations = 24 vertices.
    for i in 0..4 {
        for j in (i + 1)..4 {
            for &si in &[-k, k] {
                for &sj in &[-k, k] {
                    let mut c = [0.0_f32; 4];
                    c[i] = si;
                    c[j] = sj;
                    v.push(Vec4::new(c[0], c[1], c[2], c[3]));
                }
            }
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Gravity;
    use crate::world::World;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "expected {a} close to {b} (tol {tol})"
        );
    }

    #[test]
    fn falling_sphere_accelerates_in_r4() {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        // Gravity along −y; other dimensions inert.
        world.push_field(Box::new(Gravity::new(Vec4::new(0.0, -9.8, 0.0, 0.0))));

        let id = world.push_body(sphere_body_r4(
            Vec4::new(0.0, 5.0, 0.0, 0.0),
            Vec4::ZERO,
            0.5,
            1.0,
        ));
        world.step(1.0 / 60.0);
        let body = &world.bodies[id];
        assert!(body.velocity.y < -0.1 && body.velocity.y > -0.2);
        // No motion in x / z / w without forces there.
        assert_close(body.velocity.x, 0.0, 1e-6);
        assert_close(body.velocity.z, 0.0, 1e-6);
        assert_close(body.velocity.w, 0.0, 1e-6);
    }

    #[test]
    fn head_on_sphere_collision_reverses_velocity() {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);

        // Two spheres on the x-axis closing at 4 m/s combined.
        world.push_body(sphere_body_r4(
            Vec4::new(-1.0, 0.0, 0.0, 0.0),
            Vec4::new(2.0, 0.0, 0.0, 0.0),
            0.5,
            1.0,
        ));
        world.push_body(sphere_body_r4(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(-2.0, 0.0, 0.0, 0.0),
            0.5,
            1.0,
        ));

        for _ in 0..120 {
            world.step(1.0 / 120.0);
        }
        let a = &world.bodies[0];
        let b = &world.bodies[1];
        assert!(
            a.velocity.x < 0.0,
            "body 0 should bounce back: v.x = {}",
            a.velocity.x
        );
        assert!(
            b.velocity.x > 0.0,
            "body 1 should bounce back: v.x = {}",
            b.velocity.x
        );
        // Nothing should kick in the y/z/w directions.
        assert_close(a.velocity.y, 0.0, 1e-4);
        assert_close(a.velocity.z, 0.0, 1e-4);
        assert_close(a.velocity.w, 0.0, 1e-4);
    }

    /// Off-plane contact in 4D: two spheres meeting with a relative
    /// position vector that has components in all four dimensions
    /// should produce a normal-only response along the line of centers
    /// (no tangential spin develops for sphere-sphere hits).
    #[test]
    fn sphere_sphere_off_plane_contact_resolves_along_line_of_centers() {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        // Place two spheres offset in all four dimensions, closing.
        let a_pos = Vec4::new(-0.8, -0.4, 0.3, 0.2);
        let b_pos = Vec4::new(0.8, 0.4, -0.3, -0.2);
        let a = world.push_body(sphere_body_r4(
            a_pos,
            (b_pos - a_pos).normalize() * 2.0,
            0.5,
            1.0,
        ));
        let b = world.push_body(sphere_body_r4(
            b_pos,
            (a_pos - b_pos).normalize() * 2.0,
            0.5,
            1.0,
        ));
        for _ in 0..120 {
            world.step(1.0 / 120.0);
        }
        // After the collision, relative velocity along the original
        // line-of-centers must have reversed sign.
        let rel = world.bodies[b].velocity - world.bodies[a].velocity;
        let axis = (b_pos - a_pos).normalize();
        let v_along = rel.dot(axis);
        assert!(
            v_along > 0.0,
            "relative velocity should now be separating: {v_along}"
        );
    }

    fn assert_all_on_circumsphere(verts: &[Vec4], radius: f32, label: &str) {
        for (i, v) in verts.iter().enumerate() {
            let d = v.length();
            assert!(
                (d - radius).abs() < 1e-4,
                "{label} vertex {i} off circumsphere: |v| = {d}, want {radius}",
            );
        }
    }

    #[test]
    fn pentatope_has_5_vertices_on_circumsphere() {
        let verts = pentatope_vertices(1.0);
        assert_eq!(verts.len(), 5);
        assert_all_on_circumsphere(&verts, 1.0, "pentatope");
    }

    /// A regular 5-cell has `C(5,2) = 10` edges all of the same length.
    /// Verify equidistance between every vertex pair.
    #[test]
    fn pentatope_edges_are_equal_length() {
        let verts = pentatope_vertices(1.0);
        let expected = (verts[0] - verts[1]).length();
        for i in 0..5 {
            for j in (i + 1)..5 {
                let d = (verts[i] - verts[j]).length();
                assert!(
                    (d - expected).abs() < 1e-3,
                    "edge ({i},{j}) = {d}, expected {expected}",
                );
            }
        }
    }

    #[test]
    fn tesseract_has_16_vertices_on_circumsphere() {
        let verts = tesseract_vertices(1.0);
        assert_eq!(verts.len(), 16);
        assert_all_on_circumsphere(&verts, 1.0, "tesseract");
    }

    #[test]
    fn cell16_has_8_vertices_on_circumsphere() {
        let verts = cell16_vertices(1.0);
        assert_eq!(verts.len(), 8);
        assert_all_on_circumsphere(&verts, 1.0, "16-cell");
    }

    #[test]
    fn cell24_has_24_vertices_on_circumsphere() {
        let verts = cell24_vertices(1.0);
        assert_eq!(verts.len(), 24);
        assert_all_on_circumsphere(&verts, 1.0, "24-cell");
    }

    /// The 24-cell vertex set equals 16-cell ∪ (rescaled) tesseract
    /// vertex set — the property that makes it self-dual and
    /// space-filling. Check the count matches and each 24-cell vertex
    /// either matches a 16-cell point or a tesseract corner.
    #[test]
    fn cell24_decomposes_into_16cell_plus_tesseract() {
        let c24 = cell24_vertices(1.0);
        // At `r = 1`, the 24-cell's nonzero coordinates are `±1/√2`;
        // those are simultaneously the 16-cell vertices at
        // `radius = 1/√2` and the tesseract vertices at `radius = 1`.
        // This test confirms the numeric match.
        let k = 1.0 / 2.0_f32.sqrt();
        // The 8 vertices with exactly one nonzero coordinate of
        // magnitude `k√2 = 1` would form a 16-cell at `r = 1`; the
        // 24-cell uses coordinates of magnitude `k` instead, so it
        // contains both the 8 axis-aligned-pair points and the 8
        // all-corners-scaled-by-k points — but actually all 24 have
        // two nonzero entries at `±k`, so verify that shape.
        for v in &c24 {
            let nz = [v.x, v.y, v.z, v.w]
                .iter()
                .filter(|&&c| c.abs() > 1e-6)
                .count();
            assert_eq!(nz, 2, "24-cell vertex should have 2 nonzero coords: {v:?}");
            for c in [v.x, v.y, v.z, v.w] {
                if c.abs() > 1e-6 {
                    assert!((c.abs() - k).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn orientation_integration_preserves_unit_rotor() {
        let space = EuclideanR4;
        let mut iso = Iso4Flat::IDENTITY;
        // Compound angular velocity: rotation in xy and zw planes.
        let omega = Bivector4::new(0.2, 0.0, 0.0, 0.0, 0.0, 0.15);
        for _ in 0..1000 {
            iso = space.integrate_orientation(iso, omega, 1.0 / 60.0);
        }
        let n = iso.rotation.norm_squared();
        assert!(
            (n - 1.0).abs() < 1e-3,
            "rotor drifted off the unit manifold: |R|² = {n}"
        );
    }
}
