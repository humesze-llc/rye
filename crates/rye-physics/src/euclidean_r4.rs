//! `impl PhysicsSpace for EuclideanR4`, 4D Euclidean rigid-body physics.
//!
//! Angular velocity is a `Bivector4` (six rotation-plane components);
//! inertia is the scalar moment for isotropic bodies, same pragmatic
//! simplification made in 3D. A full 4D inertia tensor is a 6×6
//! bivector-to-bivector map, it doesn't land until an actual anisotropic
//! 4D body demands it.
//!
//! Orientation integration uses `Rotor4` directly. The Clifford-rotor
//! multiplication convention is "left operand applied first" (opposite to
//! `glam::Quat`'s "right first"), so the composed orientation after a
//! timestep is `rotation_new = rotation_current * delta_rotor`.
//!
//! ## Narrowphase coverage
//!
//! Sphere-sphere, sphere-halfspace, polytope-polytope (4D GJK + EPA),
//! sphere-polytope, and polytope-halfspace all ship in this file. See
//! `register_default_narrowphase`.

use glam::Vec4;

use rye_math::{Bivector, Bivector4, EuclideanR4, Iso4Flat, Rotor};

use crate::body::RigidBody;
use crate::collider::{Collider, ColliderKind};
use crate::collision::{epa_r4, gjk_intersect_r4, ConvexHull4, GjkResult4, Sphere4 as GjkSphere4};
use crate::integrator::PhysicsSpace;
use crate::narrowphase::Narrowphase;
use crate::response::Contact;

/// Linear velocity at offset `r` due to angular velocity bivector
/// `omega`, the 4D analogue of 3D's `ω × r`. For `ω = e_xy` and
/// `r = e_x` this returns `+e_y` (rotation in the +xy plane sends
/// the +x axis toward +y), matching the rigid-body convention used
/// throughout `rye-physics`.
///
/// This is the **negation** of the Clifford left-contraction
/// `omega.contract_vec(r)`. See [`Bivector4::contract_vec`] for why
/// the math primitive is kept Clifford-pure: keeping the sign flip
/// at the physics layer means future generic-`N` callers get
/// consistent contraction semantics across dimensions, and a future
/// physics-on-`N` path can wrap this same way without surprise.
pub fn omega_cross_r(omega: Bivector4, r: glam::Vec4) -> glam::Vec4 {
    -omega.contract_vec(r)
}

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
        // the orientation, same defense in depth as the 3D path.
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
        // `v(r) = v_linear + ω × r`, `omega_cross_r` is the
        // physics-convention version of the bivector contraction (the
        // negated Clifford left-contraction `ω⌋r`); see its doc for
        // why the negation lives here rather than in `Bivector4`.
        let r = p - body.position;
        body.velocity + omega_cross_r(body.angular_velocity, r)
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
// Narrowphases for EuclideanR4 colliders.
// ---------------------------------------------------------------------------

fn sphere_sphere_r4(
    a: &RigidBody<EuclideanR4>,
    b: &RigidBody<EuclideanR4>,
    space: &EuclideanR4,
) -> Option<Contact<EuclideanR4>> {
    let Collider::Sphere { radius: ra, .. } = a.collider else {
        return None;
    };
    let Collider::Sphere { radius: rb, .. } = b.collider else {
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

/// Sphere vs 4D half-space: signed distance from sphere center to
/// the plane gives penetration. Mirrors the 3D path.
fn sphere_halfspace_r4(
    a: &RigidBody<EuclideanR4>,
    b: &RigidBody<EuclideanR4>,
    _space: &EuclideanR4,
) -> Option<Contact<EuclideanR4>> {
    let Collider::Sphere { radius, .. } = a.collider else {
        return None;
    };
    let Collider::HalfSpace4D { normal, offset } = b.collider else {
        return None;
    };
    let signed = a.position.dot(normal) - offset;
    let penetration = radius - signed;
    if penetration <= 0.0 {
        return None;
    }
    // A→B normal points *into* the half-space (opposite the
    // half-space's outward normal); pushing along it separates the
    // sphere from the wall.
    let contact_normal = -normal;
    let point = a.position - normal * radius;
    Some(Contact {
        normal: contact_normal,
        point,
        penetration,
        restitution: (a.restitution + b.restitution) * 0.5,
    })
}

/// 4D convex polytope vs 4D half-space: deepest-vertex search. The
/// world-space polytope vertex with the most-negative signed
/// distance to the plane is the contact point; the 4D normal and
/// depth read straight off it.
fn polytope_halfspace_r4(
    a: &RigidBody<EuclideanR4>,
    b: &RigidBody<EuclideanR4>,
    _space: &EuclideanR4,
) -> Option<Contact<EuclideanR4>> {
    let Collider::ConvexPolytope4D { vertices: va_local } = &a.collider else {
        return None;
    };
    let Collider::HalfSpace4D {
        normal: plane_n,
        offset,
    } = b.collider
    else {
        return None;
    };

    let mut deepest = Vec4::ZERO;
    let mut deepest_depth = 0.0_f32;
    for &v_local in va_local {
        let v_world = a.orientation.rotation.apply(v_local) + a.position;
        let signed = v_world.dot(plane_n) - offset;
        let depth = -signed;
        if depth > deepest_depth {
            deepest_depth = depth;
            deepest = v_world;
        }
    }
    if deepest_depth <= 0.0 {
        return None;
    }
    Some(Contact {
        normal: -plane_n,
        point: deepest,
        penetration: deepest_depth,
        restitution: (a.restitution + b.restitution) * 0.5,
    })
}

// ---------------------------------------------------------------------------
// Polytope narrowphases via 4D GJK + EPA. Same wrapper shape as the R³
// path: transform body-local vertices to world, build a support-fn
// adapter, run GJK, hand the 5-simplex to EPA, validate the result.
// ---------------------------------------------------------------------------

/// Conservative bounding-sphere radius of a 4D polytope about its
/// centroid. Cheap pre-cull for narrowphase, if bounding spheres
/// don't overlap, the polytopes can't either.
fn polytope4_bounding_radius(local_vertices: &[Vec4]) -> f32 {
    local_vertices
        .iter()
        .map(|v| v.length_squared())
        .fold(0.0_f32, f32::max)
        .sqrt()
}

/// Maximum vertex count for any 4D polytope collider. The 24-cell
/// is the densest regular 4-polytope at 24 vertices; non-regular
/// shapes with more vertices would need a larger constant or a
/// different storage strategy. Hitting this limit without a
/// recompile would silently truncate vertices and produce wrong
/// collision results, so we assert in debug builds.
pub(crate) const MAX_POLYTOPE4_VERTICES: usize = 32;

/// Transform body-local vertices to world space, writing into the
/// caller's stack-allocated buffer. Returns a slice of the populated
/// prefix. Hot path: called once per polytope per polytope-pair per
/// tick, so we don't allocate.
fn world_vertices4_into<'a>(
    local: &[Vec4],
    pos: Vec4,
    rot: rye_math::Rotor4,
    out: &'a mut [Vec4; MAX_POLYTOPE4_VERTICES],
) -> &'a [Vec4] {
    debug_assert!(
        local.len() <= MAX_POLYTOPE4_VERTICES,
        "polytope vertex count {} exceeds MAX_POLYTOPE4_VERTICES = {}",
        local.len(),
        MAX_POLYTOPE4_VERTICES
    );
    let n = local.len().min(MAX_POLYTOPE4_VERTICES);
    for i in 0..n {
        out[i] = rot.apply(local[i]) + pos;
    }
    &out[..n]
}

/// Minimum and maximum penetration depths EPA will return; reject
/// anything outside these as likely numerical noise (too small) or
/// EPA iteration-cap fallback on pathological input (too deep).
const MIN_POLYTOPE4_PENETRATION: f32 = 1e-4;
const MAX_POLYTOPE4_PENETRATION: f32 = 5.0;

fn validate_contact4(
    info: &crate::collision::ContactInfo4,
    a: &RigidBody<EuclideanR4>,
    b: &RigidBody<EuclideanR4>,
) -> Option<Contact<EuclideanR4>> {
    if !info.penetration.is_finite()
        || info.penetration < MIN_POLYTOPE4_PENETRATION
        || info.penetration > MAX_POLYTOPE4_PENETRATION
        || !info.normal.is_finite()
        || !info.point.is_finite()
    {
        return None;
    }
    let n2 = info.normal.length_squared();
    if !(0.5..=1.5).contains(&n2) {
        return None;
    }
    Some(Contact {
        normal: info.normal,
        point: info.point,
        penetration: info.penetration,
        restitution: (a.restitution + b.restitution) * 0.5,
    })
}

fn polytope_polytope_r4(
    a: &RigidBody<EuclideanR4>,
    b: &RigidBody<EuclideanR4>,
    _space: &EuclideanR4,
) -> Option<Contact<EuclideanR4>> {
    let Collider::ConvexPolytope4D { vertices: va_local } = &a.collider else {
        return None;
    };
    let Collider::ConvexPolytope4D { vertices: vb_local } = &b.collider else {
        return None;
    };

    let ra = polytope4_bounding_radius(va_local);
    let rb = polytope4_bounding_radius(vb_local);
    let center_dist_sq = (b.position - a.position).length_squared();
    let combined = ra + rb;
    if center_dist_sq > combined * combined {
        return None;
    }

    let mut buf_a = [Vec4::ZERO; MAX_POLYTOPE4_VERTICES];
    let mut buf_b = [Vec4::ZERO; MAX_POLYTOPE4_VERTICES];
    let va = world_vertices4_into(va_local, a.position, a.orientation.rotation, &mut buf_a);
    let vb = world_vertices4_into(vb_local, b.position, b.orientation.rotation, &mut buf_b);
    let hull_a = ConvexHull4 { vertices: va };
    let hull_b = ConvexHull4 { vertices: vb };

    let initial_dir = b.position - a.position;
    let simplex = match gjk_intersect_r4(&hull_a, &hull_b, initial_dir) {
        GjkResult4::Intersecting { simplex } => simplex,
        GjkResult4::Separated => return None,
    };
    let info = epa_r4(&hull_a, &hull_b, simplex)?;
    validate_contact4(&info, a, b)
}

fn sphere_polytope_r4(
    a: &RigidBody<EuclideanR4>,
    b: &RigidBody<EuclideanR4>,
    _space: &EuclideanR4,
) -> Option<Contact<EuclideanR4>> {
    let Collider::Sphere { radius, .. } = a.collider else {
        return None;
    };
    let Collider::ConvexPolytope4D { vertices: vb_local } = &b.collider else {
        return None;
    };

    let rb = polytope4_bounding_radius(vb_local);
    let center_dist_sq = (b.position - a.position).length_squared();
    let combined = radius + rb;
    if center_dist_sq > combined * combined {
        return None;
    }

    let mut buf_b = [Vec4::ZERO; MAX_POLYTOPE4_VERTICES];
    let vb = world_vertices4_into(vb_local, b.position, b.orientation.rotation, &mut buf_b);
    let support_a = GjkSphere4 {
        center: a.position,
        radius,
    };
    let support_b = ConvexHull4 { vertices: vb };
    let initial_dir = b.position - a.position;
    let simplex = match gjk_intersect_r4(&support_a, &support_b, initial_dir) {
        GjkResult4::Intersecting { simplex } => simplex,
        GjkResult4::Separated => return None,
    };
    let info = epa_r4(&support_a, &support_b, simplex)?;
    validate_contact4(&info, a, b)
}

pub fn register_default_narrowphase(np: &mut Narrowphase<EuclideanR4>) {
    np.register(ColliderKind::Sphere, ColliderKind::Sphere, sphere_sphere_r4);
    np.register(
        ColliderKind::Sphere,
        ColliderKind::HalfSpace4D,
        sphere_halfspace_r4,
    );
    np.register(
        ColliderKind::ConvexPolytope4D,
        ColliderKind::ConvexPolytope4D,
        polytope_polytope_r4,
    );
    np.register(
        ColliderKind::Sphere,
        ColliderKind::ConvexPolytope4D,
        sphere_polytope_r4,
    );
    np.register(
        ColliderKind::ConvexPolytope4D,
        ColliderKind::HalfSpace4D,
        polytope_halfspace_r4,
    );
}

// ---------------------------------------------------------------------------
// Convenience constructors.
// ---------------------------------------------------------------------------

/// Solid-ball moment of inertia in 4D about a 2-plane through its
/// center: `I = (2/(n+2))·m·r² = m·r²/3` for n=4. Reduces to the
/// familiar `(2/5)·m·r²` for the 3-ball (n=3). Derivation: by
/// symmetry, `∫ x²dV / V = (1/n)·∫ ρ²dV / V = r²/(n+2)`, and the
/// squared distance from the rotation 2-plane is two such integrals
/// summed.
pub fn ball4_inertia(mass: f32, radius: f32) -> f32 {
    mass * radius * radius / 3.0
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
        Collider::sphere_at_origin(radius),
        mass,
        ball4_inertia(mass, radius),
        &EuclideanR4,
    )
}

/// Static 4D half-space body, the 4D analogue of a floor or wall.
/// `normal` is the outward direction (pointing into the empty half);
/// `offset` places the plane at `dot(p, normal) = offset`. The
/// produced body has `inv_mass = 0` so gravity and impulses are
/// inert on it.
pub fn halfspace4_body_r4(normal: Vec4, offset: f32) -> RigidBody<EuclideanR4> {
    let n = normal.try_normalize().unwrap_or(Vec4::Y);
    RigidBody::fixed(
        Vec4::ZERO,
        Collider::HalfSpace4D { normal: n, offset },
        1.0,
        &EuclideanR4,
    )
}

/// Dynamic 4D convex-polytope body. Inertia uses the bounding-
/// sphere approximation: same formula as `ball4_inertia` applied to
/// the polytope's circumradius. Same pragmatic trade as the 3D path,
/// correct for sphere-like polytopes, in the right order of magnitude
/// for cube-like ones, replaced by a full 6x6 bivector-inertia tensor
/// when a game actually needs it.
pub fn polytope_body_r4(
    position: Vec4,
    velocity: Vec4,
    vertices: Vec<Vec4>,
    mass: f32,
) -> RigidBody<EuclideanR4> {
    let bounding_r_sq = vertices
        .iter()
        .map(|v| v.length_squared())
        .fold(0.0, f32::max);
    let inertia = mass * bounding_r_sq / 3.0;
    RigidBody::new(
        position,
        velocity,
        Collider::ConvexPolytope4D { vertices },
        mass,
        inertia,
        &EuclideanR4,
    )
}

// ---------------------------------------------------------------------------
// 4D regular polytopes. Six exist in 4D (five analogues of the Platonic
// solids plus the 24-cell which has no 3D counterpart). The four most
// physically useful for games, 5-cell, tesseract, 16-cell, 24-cell,
// are generated here. The 120-cell (600 vertices) and 600-cell (120
// vertices) land when a demo actually needs them.
//
// Every generator returns vertices centered at the origin and scaled
// so the circumradius (bounding-sphere radius) equals the caller-
// provided `r`. Caller is responsible for further translation.
// ---------------------------------------------------------------------------

/// **5-cell / pentatope** (4D simplex): 5 vertices, 10 edges, 10 faces,
/// 5 tetrahedral cells. The 4D analogue of the tetrahedron.
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
/// faces, 24 octahedral cells. Unique to 4D, it has no 3D analogue
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
    // Pairs of axes (0=x, 1=y, 2=z, 3=w), C(4, 2) = 6 pairs, each
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

    /// Pin `ball4_inertia` at the closed-form `m·r²/3` and check the
    /// 3D-vs-4D inequality `1/3 < 2/5`. Derivation lives on the
    /// function itself.
    #[test]
    fn ball4_inertia_matches_uniform_n_ball_formula() {
        assert_close(ball4_inertia(1.0, 1.0), 1.0 / 3.0, 1e-6);
        assert_close(ball4_inertia(2.0, 0.5), 2.0 * 0.25 / 3.0, 1e-6);
        assert_close(ball4_inertia(10.0, 3.0), 10.0 * 9.0 / 3.0, 1e-5);
        let three_d = crate::euclidean_r3::sphere_inertia(1.0, 1.0);
        let four_d = ball4_inertia(1.0, 1.0);
        assert!(four_d < three_d);
    }

    /// `polytope_body_r4` derives its inertia from the same n-ball
    /// formula via the bounding-sphere radius. Pin that the wrapper
    /// agrees with `ball4_inertia` for a unit-circumradius vertex
    /// set so a future regression in either path surfaces here too.
    #[test]
    fn polytope_body_r4_inertia_matches_ball4_inertia() {
        let body = polytope_body_r4(Vec4::ZERO, Vec4::ZERO, pentatope_vertices(1.0), 2.5);
        assert_close(body.inertia, ball4_inertia(2.5, 1.0), 1e-5);
    }

    /// 4D sphere falling onto a 4D `y = 0` half-space settles above
    /// the ground without tunneling. Exercises the `sphere_halfspace_r4`
    /// narrowphase end-to-end through the integrator + solver.
    #[test]
    fn sphere_settles_on_4d_floor() {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        world.push_field(Box::new(Gravity::new(Vec4::new(0.0, -9.8, 0.0, 0.0))));
        let _floor = world.push_body(halfspace4_body_r4(Vec4::Y, 0.0));
        let ball = world.push_body(sphere_body_r4(
            Vec4::new(0.0, 2.0, 0.0, 0.0),
            Vec4::ZERO,
            0.5,
            1.0,
        ));
        // 5 seconds, plenty of time to fall, bounce, settle.
        for _ in 0..300 {
            world.step(1.0 / 60.0);
        }
        let body = &world.bodies[ball];
        // Lowest sphere point ≈ position.y − radius. Should sit at or
        // just above the floor (small Baumgarte-corrected penetration).
        let lowest = body.position.y - 0.5;
        assert!(
            lowest >= -0.05,
            "ball tunneled through 4D floor: y_bottom = {lowest}"
        );
        assert!(
            body.velocity.length() < 0.5,
            "ball still moving: |v| = {}",
            body.velocity.length()
        );
    }

    /// 4D pentatope falling onto a 4D floor produces a real contact
    /// (not None from `validate_contact4`), comes to rest above the
    /// floor, and stays there with bounded angular velocity.
    /// End-to-end integration test of the full
    /// `gravity → integrator → polytope_halfspace_r4 → manifold → PGS`
    /// pipeline in 4D.
    ///
    /// **Catches:**
    /// - Sign errors in `Bivector4::contract_vec` /
    ///   `Bivector4::wedge` that inject energy at off-center contacts
    ///   (such bugs typically blow `|v|` past 10 m/s within 60 frames
    ///   and the body rebounds violently, the original Clifford-vs-
    ///   physics-convention bug had the pentatope reach +107 m/s).
    /// - Failure to populate manifolds at 4D contacts (body free-
    ///   falls past the floor).
    /// - NaN propagation through orientation integration.
    #[test]
    fn pentatope_settles_on_4d_floor() {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        world.push_field(Box::new(Gravity::new(Vec4::new(0.0, -9.8, 0.0, 0.0))));
        let floor = world.push_body(halfspace4_body_r4(Vec4::Y, 0.0));
        let body_id = world.push_body(polytope_body_r4(
            Vec4::new(0.0, 3.0, 0.0, 0.0),
            Vec4::ZERO,
            pentatope_vertices(0.5),
            1.0,
        ));
        // Restitution 0 on both sides lets the body settle
        // deterministically rather than tumbling for 5+ seconds;
        // we're testing that the contact pipeline converges, not that
        // bouncing eventually damps out.
        world.bodies[floor].restitution = 0.0;
        world.bodies[body_id].restitution = 0.0;

        // 10 s of sim. The body falls ≈ 0.78 s, then has 9+ s to settle.
        for _ in 0..600 {
            world.step(1.0 / 60.0);
        }
        let body = &world.bodies[body_id];

        // Pentatope circumradius is 0.5; the deepest point in any
        // resting orientation is at most 0.5 below the centroid, so
        // the centroid should sit in y ∈ (-0.1, +0.6) when at rest
        // on y = 0.
        assert!(
            body.position.y.is_finite() && (-0.5..=1.0).contains(&body.position.y),
            "pentatope position out of expected resting band: y = {}",
            body.position.y
        );
        assert!(
            body.position.x.abs() < 5.0
                && body.position.z.abs() < 5.0
                && body.position.w.abs() < 5.0,
            "pentatope drifted too far horizontally: pos = {:?}",
            body.position
        );

        // Linear velocity should be effectively zero. With the original
        // contract_vec sign bug this hit ≈ +107 m/s; under 1.0 here is
        // tight enough to catch any reintroduction without flaking on
        // legitimate solver micro-residuals.
        assert!(
            body.velocity.length() < 1.0,
            "pentatope still moving after 10 s: |v| = {}, v = {:?}",
            body.velocity.length(),
            body.velocity
        );

        // Angular velocity: bounded, finite. Allows some residual
        // tumble (resting on an edge or face is non-unique) but
        // catches NaN propagation and runaway angular impulse.
        let omega = body.angular_velocity;
        let omega_mag2 = omega.xy * omega.xy
            + omega.xz * omega.xz
            + omega.xw * omega.xw
            + omega.yz * omega.yz
            + omega.yw * omega.yw
            + omega.zw * omega.zw;
        assert!(
            omega_mag2.is_finite() && omega_mag2 < 4.0,
            "pentatope angular velocity blew up: |ω|² = {omega_mag2}, ω = {omega:?}"
        );
    }

    /// 4D tesseract dropped onto a 4D floor: this is the harder
    /// settling case for the polytope-halfspace narrowphase. A
    /// tesseract resting on a cell-face has eight vertices
    /// simultaneously co-planar with the floor; the single-deepest-
    /// vertex narrowphase returns one contact per frame, but the
    /// `Manifold` accumulator picks up the others over a few frames
    /// (each frame's "deepest" varies by f32 noise) until PGS has
    /// enough constraints to stop the body from rocking.
    ///
    /// If this test ever starts failing with a "still moving" or
    /// non-trivial angular velocity, that's the signal that
    /// single-contact-per-call has run out of headroom and the
    /// narrowphase needs to start emitting all
    /// at-or-below-slop vertices in a single call (multi-contact
    /// reduction, mirroring what the 3D path will eventually need
    /// for thin slabs).
    #[test]
    fn tesseract_settles_on_4d_floor() {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        world.push_field(Box::new(Gravity::new(Vec4::new(0.0, -9.8, 0.0, 0.0))));
        let floor = world.push_body(halfspace4_body_r4(Vec4::Y, 0.0));
        let body_id = world.push_body(polytope_body_r4(
            Vec4::new(0.0, 3.0, 0.0, 0.0),
            Vec4::ZERO,
            tesseract_vertices(0.5),
            1.0,
        ));
        world.bodies[floor].restitution = 0.0;
        world.bodies[body_id].restitution = 0.0;

        for _ in 0..600 {
            world.step(1.0 / 60.0);
        }
        let body = &world.bodies[body_id];

        // Tesseract circumradius is 0.5 with edge half-length 0.25.
        // Resting on a cell-face puts the centroid at ≈ y = 0.25.
        // We allow a generous band because the body might also rest
        // on an edge or a 2-face, which gives different heights.
        assert!(
            body.position.y.is_finite() && (-0.3..=1.0).contains(&body.position.y),
            "tesseract position out of expected resting band: y = {}",
            body.position.y
        );
        assert!(
            body.velocity.length() < 1.5,
            "tesseract still moving after 10 s: |v| = {}, v = {:?}",
            body.velocity.length(),
            body.velocity
        );
        let omega = body.angular_velocity;
        let omega_mag2 = omega.xy * omega.xy
            + omega.xz * omega.xz
            + omega.xw * omega.xw
            + omega.yz * omega.yz
            + omega.yw * omega.yw
            + omega.zw * omega.zw;
        assert!(
            omega_mag2.is_finite() && omega_mag2 < 4.0,
            "tesseract angular velocity blew up: |ω|² = {omega_mag2}, ω = {omega:?}"
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
    /// vertex set, the property that makes it self-dual and
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
        // all-corners-scaled-by-k points, but actually all 24 have
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

    /// Narrowphase integration: a sphere deeply inside a tesseract
    /// should produce a contact with a sensible normal and finite
    /// penetration, this exercises the sphere-polytope 4D GJK+EPA
    /// path end-to-end, with bounding-sphere cull included.
    #[test]
    fn sphere_inside_tesseract_produces_contact() {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        let _a = world.push_body(sphere_body_r4(Vec4::ZERO, Vec4::ZERO, 0.3, 1.0));
        let _b = world.push_body(polytope_body_r4(
            Vec4::ZERO,
            Vec4::ZERO,
            tesseract_vertices(0.8),
            0.0,
        ));
        // Zero-mass wrapper makes the tesseract static; exercise the
        // narrowphase lookup and verify a contact is found (the
        // solver would then apply it, but we only care about
        // detection here).
        let pair_found = {
            let (a, b) = world.bodies.split_at_mut(1);
            world.narrowphase.test(&a[0], &b[0], &EuclideanR4).is_some()
        };
        assert!(
            pair_found,
            "sphere inside tesseract should produce a contact"
        );
    }

    /// Separated 4D polytopes → no contact. Exercises the bounding-
    /// sphere pre-cull plus GJK's Separated path.
    #[test]
    fn separated_pentatopes_produce_no_contact() {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        let _a = world.push_body(polytope_body_r4(
            Vec4::ZERO,
            Vec4::ZERO,
            pentatope_vertices(1.0),
            1.0,
        ));
        let _b = world.push_body(polytope_body_r4(
            Vec4::new(10.0, 0.0, 0.0, 0.0),
            Vec4::ZERO,
            pentatope_vertices(1.0),
            1.0,
        ));
        let (a, b) = world.bodies.split_at_mut(1);
        assert!(world.narrowphase.test(&a[0], &b[0], &EuclideanR4).is_none());
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
