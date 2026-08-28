//! Rotor4 multiplication is left-first (opposite `glam::Quat`), so the composed
//! orientation after a timestep is `rotation_current * delta_rotor`.

use glam::Vec4;

use loam_math::{Bivector, Bivector4, EuclideanR4, Iso4Flat, Rotor};
use loam_shape::polytope::Polytope4;

use crate::body::RigidBody;
use crate::collider::{Collider, ColliderKind};
use crate::collision::{epa_r4, gjk_intersect_r4, ConvexHull4, GjkResult4, Sphere4 as GjkSphere4};
use crate::integrator::PhysicsSpace;
use crate::narrowphase::Narrowphase;
use crate::response::Contact;

/// The 4D analogue of `ω × r`. Negation of the Clifford left-contraction; the
/// sign flip lives here, not in [`Bivector4::contract_vec`], to keep the math
/// primitive pure.
pub fn omega_cross_r(omega: Bivector4, r: glam::Vec4) -> glam::Vec4 {
    -omega.contract_vec(r)
}

// Static and zero-inertia bodies are treated as infinite, so 0.
fn inv_inertia(body: &RigidBody<EuclideanR4>) -> f32 {
    if body.inv_mass > 0.0 && body.inertia > 0.0 {
        1.0 / body.inertia
    } else {
        0.0
    }
}

impl PhysicsSpace for EuclideanR4 {
    type AngVel = Bivector4;
    type Inertia = f32;

    fn integrate_orientation(&self, iso: Iso4Flat, omega: Bivector4, dt: f32) -> Iso4Flat {
        // Catch a non-finite angular velocity before it reaches the rotor and
        // the GPU buffer. Release trusts internal callers.
        debug_assert!(
            omega.xy.is_finite()
                && omega.xz.is_finite()
                && omega.xw.is_finite()
                && omega.yz.is_finite()
                && omega.yw.is_finite()
                && omega.zw.is_finite(),
            "non-finite Bivector4 angular velocity in integrate_orientation",
        );
        let delta = (omega * dt).exp();
        // Normalize to fight f32 drift off the unit manifold over long runs.
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

    fn wedge(&self, a: Vec4, b: Vec4) -> Bivector4 {
        Bivector4::wedge(a, b)
    }

    fn velocity_at_point(&self, body: &RigidBody<EuclideanR4>, p: Vec4) -> Vec4 {
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

        // τ = r ∧ lin, applied via ω += I⁻¹·τ.
        let inv_i_a = inv_inertia(a);
        let inv_i_b = inv_inertia(b);
        a.angular_velocity = a.angular_velocity + Bivector4::wedge(ra, lin) * (-inv_i_a);
        b.angular_velocity = b.angular_velocity + Bivector4::wedge(rb, lin) * inv_i_b;
    }
}

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

    use loam_math::Space;
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
    // Contact normal points into the half-space (opposite its outward normal).
    let contact_normal = -normal;
    let point = a.position - normal * radius;
    Some(Contact {
        normal: contact_normal,
        point,
        penetration,
        restitution: (a.restitution + b.restitution) * 0.5,
    })
}

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

// Non-overlapping bounding spheres mean non-overlapping polytopes, which is
// the narrowphase pre-cull.
fn polytope4_bounding_radius(local_vertices: &[Vec4]) -> f32 {
    local_vertices
        .iter()
        .map(|v| v.length_squared())
        .fold(0.0_f32, f32::max)
        .sqrt()
}

/// Exceeding it silently truncates vertices and corrupts collisions, so
/// callers debug-assert.
pub const MAX_POLYTOPE4_VERTICES: usize = 32;

// Hot path; allocation-free by contract.
fn world_vertices4_into<'a>(
    local: &[Vec4],
    pos: Vec4,
    rot: loam_math::Rotor4,
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

// Accepted EPA penetration band: below is numerical noise, above is an EPA
// iteration-cap fallback on pathological input.
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
    let info = epa_r4(&hull_a, &hull_b, simplex, combined)?;
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
    let info = epa_r4(&support_a, &support_b, simplex, combined)?;
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

/// `I = (2/(n+2))·m·r² = m·r²/3` at n = 4, against `(2/5)·m·r²` for the
/// 3-ball.
pub fn ball4_inertia(mass: f32, radius: f32) -> f32 {
    mass * radius * radius / 3.0
}

/// Isotropic moment of a uniform-density regular polychoron about any 2-plane
/// through its centroid, at circumradius `circumradius`. `None` for the
/// 120-cell and the 600-cell, whose second moments are not derived here; the
/// bounding ball is within 12% of both.
///
/// Exact, not the bounding-sphere approximation [`polytope_body_r4`] carries.
/// Each of these symmetry groups acts irreducibly on R⁴, so Schur's lemma
/// forces the second-moment matrix `M_ij = <x_i·x_j>` to `μ·I₄` (Serre,
/// *Linear Representations of Finite Groups* (1977), §2.2), and an isotropic
/// `M` is precisely what the scalar [`PhysicsSpace::Inertia`] slot represents
/// without loss: the moment about a coordinate 2-plane is
/// `m·(<x_i²> + <x_j²>) = m·<|x|²>/2` for every plane alike.
///
/// `<|x|²>` at unit circumradius, and where each value comes from:
/// - 5-cell, `1/6`. The Dirichlet second moment of a `d`-simplex is
///   `Σᵢ(vᵢ−c)(vᵢ−c)ᵀ/((d+1)(d+2))` (Lasserre and Avrachenkov 2001, *Amer.
///   Math. Monthly* 108(2), §3), whose trace at `d = 4`, `c = 0`, `|vᵢ| = 1`
///   is `5/30`.
/// - 8-cell, `1/3`. Four independent uniform coordinates of half-width `1/2`,
///   each contributing `1/12`.
/// - 16-cell, `4/15`. The same simplex formula on one orthant simplex
///   (vertices `0`, `eᵢ`) gives `<xᵢ²> = 2/((d+1)(d+2)) = 1/15`.
/// - 24-cell, `13/30`. Cone decomposition over its 24 octahedral facets: for a
///   pyramid with apex at the centroid and base in `{x·n = h}`, the `t·b`
///   parametrisation's `t⁵` and `t³` moments give
///   `<|x|²> = (2/3)·(h² + <|b_⊥|²>)`. At inradius `h = 1` the 24-cell is
///   `{|x|_∞ ≤ 1} ∩ {|x|_1 ≤ 2}` with circumradius `√2`, each facet is the
///   octahedron `{|x|_1 ≤ 1}` with `<|b_⊥|²> = 3·2/((3+1)(3+2)) = 3/10`, so
///   `<|x|²> = (2/3)·(13/10) = 13/15` and `(13/15)/(√2)² = 13/30`.
pub fn regular_polytope4_inertia(shape: Polytope4, mass: f32, circumradius: f32) -> Option<f32> {
    let mean_radius_sq = match shape {
        Polytope4::Pentatope => 1.0 / 6.0,
        Polytope4::Tesseract => 1.0 / 3.0,
        Polytope4::Cell16 => 4.0 / 15.0,
        Polytope4::Cell24 => 13.0 / 30.0,
        Polytope4::Cell120 | Polytope4::Cell600 => return None,
    };
    Some(0.5 * mass * circumradius * circumradius * mean_radius_sq)
}

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

/// `normal` is the outward direction; `offset` places the plane at
/// `dot(p, normal) = offset`.
pub fn halfspace4_body_r4(normal: Vec4, offset: f32) -> RigidBody<EuclideanR4> {
    let n = normal.try_normalize().unwrap_or(Vec4::Y);
    RigidBody::fixed(
        Vec4::ZERO,
        Collider::HalfSpace4D { normal: n, offset },
        1.0,
        &EuclideanR4,
    )
}

/// Inertia is the bounding-sphere approximation ([`ball4_inertia`] at the
/// circumradius): exact for sphere-like shapes, order-of-magnitude for
/// cube-like ones.
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

pub use loam_shape::polytope_geom::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::determinism_fixture::{
        assert_scenario_stays_physical, determinism_scenario_run, determinism_scenario_trajectory,
        fnv1a64, GOLDEN_TRAJECTORY_HASH,
    };
    use crate::field::Gravity;
    use crate::world::{Schedule, World};

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "expected {a} close to {b} (tol {tol})"
        );
    }

    #[test]
    fn ball4_inertia_matches_uniform_n_ball_formula() {
        assert_close(ball4_inertia(1.0, 1.0), 1.0 / 3.0, 1e-6);
        assert_close(ball4_inertia(2.0, 0.5), 2.0 * 0.25 / 3.0, 1e-6);
        assert_close(ball4_inertia(10.0, 3.0), 10.0 * 9.0 / 3.0, 1e-5);
        let three_d = crate::euclidean_r3::sphere_inertia(1.0, 1.0);
        let four_d = ball4_inertia(1.0, 1.0);
        assert!(four_d < three_d);
    }

    #[test]
    fn regular_polytope4_inertia_matches_the_stated_closed_forms() {
        let (m, r) = (2.5_f32, 0.7_f32);
        let mr2 = m * r * r;
        let cases = [
            (Polytope4::Pentatope, mr2 / 12.0),
            (Polytope4::Tesseract, mr2 / 6.0),
            (Polytope4::Cell16, 2.0 * mr2 / 15.0),
            (Polytope4::Cell24, 13.0 * mr2 / 60.0),
        ];
        for (shape, expected) in cases {
            assert_close(
                regular_polytope4_inertia(shape, m, r).expect("closed form"),
                expected,
                1e-6,
            );
        }
        assert_close(
            regular_polytope4_inertia(Polytope4::Cell24, 2.0 * m, 3.0 * r).expect("closed form"),
            2.0 * 9.0 * 13.0 * mr2 / 60.0,
            1e-5,
        );

        let moments: Vec<f32> = cases
            .iter()
            .map(|(shape, _)| regular_polytope4_inertia(*shape, m, r).unwrap())
            .chain(std::iter::once(ball4_inertia(m, r)))
            .collect();
        // Pentatope, tesseract, 16-cell, 24-cell, ball in the declaration
        // order above, sorted by how far the solid pushes its mass out.
        let ordered = [moments[0], moments[2], moments[1], moments[3], moments[4]];
        assert!(
            ordered.windows(2).all(|w| w[0] < w[1]),
            "moments out of order: {ordered:?}"
        );

        for shape in [Polytope4::Cell120, Polytope4::Cell600] {
            assert_eq!(regular_polytope4_inertia(shape, m, r), None);
        }
    }

    // SplitMix64 (Steele, Lea and Flood 2014, *OOPSLA*, §4), so the estimator
    // below is reproducible bit-for-bit from its seed.
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    // `<|x|²>` over the uniform solid at unit circumradius, by rejection
    // sampling the enclosing cube against the shape's facet half-spaces. Cell
    // centroids point along the outward facet normals with length equal to the
    // inradius, so `x·c ≤ |c|²` is membership. f64 and a seeded integer
    // generator, so the estimate is the same number on every machine.
    fn sampled_mean_radius_sq(shape: Polytope4, trials: u32, seed: u64) -> (f64, u32) {
        let planes: Vec<(glam::DVec4, f64)> = shape
            .cell_centers()
            .iter()
            .map(|c| {
                let c = glam::DVec4::new(c.x as f64, c.y as f64, c.z as f64, c.w as f64);
                (c, c.length_squared())
            })
            .collect();
        let mut state = seed;
        let mut hits = 0_u32;
        let mut total = 0.0_f64;
        let coordinate = |state: &mut u64| {
            let bits = splitmix64(state) >> 11;
            (bits as f64) / ((1_u64 << 53) as f64) * 2.0 - 1.0
        };
        for _ in 0..trials {
            let x = glam::DVec4::new(
                coordinate(&mut state),
                coordinate(&mut state),
                coordinate(&mut state),
                coordinate(&mut state),
            );
            if planes.iter().any(|(c, cc)| x.dot(*c) > *cc) {
                continue;
            }
            hits += 1;
            total += x.length_squared();
        }
        (total / hits as f64, hits)
    }

    #[test]
    fn the_polytope4_second_moments_are_the_hulls_own() {
        const TRIALS: u32 = 1 << 19;
        let expected = [
            (Polytope4::Pentatope, 1.0 / 6.0),
            (Polytope4::Tesseract, 1.0 / 3.0),
            (Polytope4::Cell16, 4.0 / 15.0),
            (Polytope4::Cell24, 13.0 / 30.0),
        ];
        for (shape, closed_form) in expected {
            let (measured, hits) = sampled_mean_radius_sq(shape, TRIALS, 0x51ED_5EED);
            assert!(hits > 2000, "{shape:?} accepted only {hits} samples");
            // 6% of the value, against a sampler whose standard error is
            // under 2% at the thinnest shape's acceptance rate, and a nearest
            // rival constant 20% away at the tightest pair.
            let tolerance = 0.06 * closed_form;
            assert!(
                (measured - closed_form).abs() < tolerance,
                "{shape:?} sampled <|x|²> = {measured} over {hits} samples, \
                 not the {closed_form} the closed form claims"
            );
            // The 4-ball's `2/3` is in the rival set because it is the value a
            // shape silently falling back to `ball4_inertia` would produce.
            let rivals = expected
                .iter()
                .map(|(shape, value)| (format!("{shape:?}"), *value))
                .chain(std::iter::once(("4-ball".to_string(), 2.0 / 3.0)));
            for (rival, other) in rivals {
                if rival == format!("{shape:?}") {
                    continue;
                }
                assert!(
                    (measured - closed_form).abs() < (measured - other).abs(),
                    "{shape:?} sampled {measured}, nearer {rival}'s {other} than \
                     its own {closed_form}"
                );
            }
        }
    }

    #[test]
    fn polytope_body_r4_inertia_matches_ball4_inertia() {
        let body = polytope_body_r4(Vec4::ZERO, Vec4::ZERO, pentatope_vertices(1.0), 2.5);
        assert_close(body.inertia, ball4_inertia(2.5, 1.0), 1e-5);
    }

    #[test]
    fn wall_contact_leaves_through_the_near_face_in_either_pair_order() {
        let mut np = Narrowphase::<EuclideanR4>::new();
        register_default_narrowphase(&mut np);

        let wall = RigidBody::fixed(
            Vec4::ZERO,
            Collider::ConvexPolytope4D {
                vertices: tesseract_vertices(0.2)
                    .into_iter()
                    .map(|v| v * Vec4::new(1.0, 20.0, 20.0, 20.0))
                    .collect(),
            },
            1.0,
            &EuclideanR4,
        );
        // Wall half thickness 0.1, ball radius 0.2, ball centre in the near
        // half: the near face is at x = −0.1 and the exit runs along −x̂.
        let ball = sphere_body_r4(Vec4::new(-0.05, 0.0, 0.0, 0.0), Vec4::ZERO, 0.2, 1.0);

        let forward = np.test(&ball, &wall, &EuclideanR4).expect("overlapping");
        assert!(
            (-forward.normal).dot(Vec4::X) < -0.99,
            "ball leaves along {:?}, not back out of the near face",
            -forward.normal
        );
        let reversed = np.test(&wall, &ball, &EuclideanR4).expect("overlapping");
        assert!(
            reversed.normal.dot(Vec4::X) < -0.99,
            "flipped pair leaves the ball along {:?}",
            reversed.normal
        );
        assert_close(forward.penetration, reversed.penetration, 1e-6);
    }

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
        for _ in 0..300 {
            world.step(1.0 / 60.0);
        }
        let body = &world.bodies[ball];
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
        // Restitution 0 so the body settles deterministically: the pin is that
        // the contact pipeline converges, not that bouncing damps out.
        world.bodies[floor].restitution = 0.0;
        world.bodies[body_id].restitution = 0.0;

        for _ in 0..600 {
            world.step(1.0 / 60.0);
        }
        let body = &world.bodies[body_id];

        // Circumradius 0.5, so a resting centroid sits in y ∈ (-0.5, 1.0).
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

        assert!(
            body.velocity.length() < 1.0,
            "pentatope still moving after 10 s: |v| = {}, v = {:?}",
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
            "pentatope angular velocity blew up: |ω|² = {omega_mag2}, ω = {omega:?}"
        );
    }

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

        // Circumradius 0.5. The band is generous: the resting feature varies.
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

    // The playground's toy: a 24-cell of this circumradius, dropped at the
    // rate its sub-steps run, under the angular drag a free tumble gets there.
    const CORNER_DROP_DT: f32 = 1.0 / 240.0;
    const CORNER_DROP_CIRCUMRADIUS: f32 = 0.45;
    const CORNER_DROP_GRAVITY: f32 = -9.8;
    // Per-second exponential decay on the spin. It is what makes the pin
    // sharp: a resting spin under drag can only hold its magnitude if the
    // contact solve is putting back exactly what the drag takes out.
    const CORNER_DROP_DRAG: f32 = 1.2;

    struct CornerDrop {
        world: World<EuclideanR4>,
        body: crate::body::BodyId,
        decay: f32,
    }

    impl CornerDrop {
        // Tipped well off any cell, so the hull must rock over onto one.
        fn new() -> Self {
            let mut world = World::new(EuclideanR4);
            register_default_narrowphase(&mut world.narrowphase);
            world.push_field(Box::new(Gravity::new(Vec4::new(
                0.0,
                CORNER_DROP_GRAVITY,
                0.0,
                0.0,
            ))));
            let floor = world.push_body(halfspace4_body_r4(Vec4::Y, 0.0));
            world.bodies[floor].restitution = 0.05;
            let body = world.push_body(polytope_body_r4(
                Vec4::new(0.0, 1.6, 0.0, 0.0),
                Vec4::ZERO,
                cell24_vertices(CORNER_DROP_CIRCUMRADIUS),
                1.0,
            ));
            world.bodies[body].restitution = 0.05;
            world.bodies[body].orientation.rotation = Bivector4::new(0.0, 0.0, 0.0, 0.6, 0.4, 0.0)
                .exp()
                .normalize();
            world.bodies[body].inertia =
                regular_polytope4_inertia(Polytope4::Cell24, 1.0, CORNER_DROP_CIRCUMRADIUS)
                    .expect("closed form");
            Self {
                world,
                body,
                decay: (-CORNER_DROP_DRAG * CORNER_DROP_DT).exp(),
            }
        }

        fn step(&mut self) {
            self.world.step(CORNER_DROP_DT);
            let decay = self.decay;
            let body = &mut self.world.bodies[self.body];
            body.angular_velocity = body.angular_velocity * decay;
        }

        fn height(&self) -> f32 {
            self.world.bodies[self.body].position.y
        }

        fn angular_speed(&self) -> f32 {
            self.world.bodies[self.body].angular_velocity.magnitude()
        }

        fn energy(&self) -> f32 {
            let body = &self.world.bodies[self.body];
            0.5 * body.velocity.length_squared()
                + 0.5 * body.inertia * body.angular_velocity.magnitude().powi(2)
                - CORNER_DROP_GRAVITY * body.position.y
        }
    }

    #[test]
    fn a_hull_dropped_on_a_corner_settles_without_climbing_its_own_contacts() {
        const LANDING_STEPS: usize = 400;
        const RESTING_STEPS: usize = 800;
        // A tenth of the tolerated penetration: below the resting contact's
        // own Baumgarte limit cycle, far under the 0.068 a stale manifold
        // climbed.
        const CLIMB_TOLERANCE: f32 = 5e-4;

        let mut drop = CornerDrop::new();
        let start_energy = drop.energy();
        let mut landing_spin = 0.0_f32;
        for _ in 0..LANDING_STEPS {
            drop.step();
            landing_spin = landing_spin.max(drop.angular_speed());
        }
        assert!(
            landing_spin > 1.0,
            "the hull never rocked over, so a settle proves nothing: peak |ω| \
             was {landing_spin} rad/s"
        );

        let resting_height = drop.height();
        let mut peak_height = resting_height;
        for _ in 0..RESTING_STEPS {
            drop.step();
            peak_height = peak_height.max(drop.height());
        }

        assert!(
            peak_height - resting_height < CLIMB_TOLERANCE,
            "a resting hull climbed {} against gravity over {RESTING_STEPS} \
             steps",
            peak_height - resting_height,
        );
        // Three seconds of drag takes any spin the contact is not feeding to
        // 3 % of its value.
        assert!(
            drop.angular_speed() < 0.02,
            "a resting hull held {} rad/s against a {CORNER_DROP_DRAG}/s \
             damper, so the contact solve is re-injecting it",
            drop.angular_speed(),
        );
        assert!(
            drop.energy() < start_energy,
            "the drop ended with {} of energy against the {start_energy} it \
             started with",
            drop.energy(),
        );
    }

    #[test]
    fn falling_sphere_accelerates_in_r4() {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
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
        assert_close(body.velocity.x, 0.0, 1e-6);
        assert_close(body.velocity.z, 0.0, 1e-6);
        assert_close(body.velocity.w, 0.0, 1e-6);
    }

    #[test]
    fn head_on_sphere_collision_reverses_velocity() {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);

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
        assert_close(a.velocity.y, 0.0, 1e-4);
        assert_close(a.velocity.z, 0.0, 1e-4);
        assert_close(a.velocity.w, 0.0, 1e-4);
    }

    #[test]
    fn sphere_sphere_off_plane_contact_resolves_along_line_of_centers() {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
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

    #[test]
    fn cell600_has_120_vertices_on_circumsphere() {
        let verts = cell600_vertices(1.0);
        assert_eq!(verts.len(), 120);
        assert_all_on_circumsphere(&verts, 1.0, "600-cell");
    }

    #[test]
    fn cell120_has_600_vertices_on_circumsphere() {
        let verts = cell120_vertices(1.0);
        assert_eq!(verts.len(), 600);
        assert_all_on_circumsphere(&verts, 1.0, "120-cell");
    }

    fn assert_centrally_symmetric(verts: &[Vec4], label: &str) {
        for v in verts {
            let antipode = -*v;
            assert!(
                verts.iter().any(|u| (*u - antipode).length() < 1e-5),
                "{label}: antipode of {v:?} is missing from the vertex set"
            );
        }
    }

    #[test]
    fn cell600_is_centrally_symmetric() {
        assert_centrally_symmetric(&cell600_vertices(1.0), "600-cell");
    }

    #[test]
    fn cell120_is_centrally_symmetric() {
        assert_centrally_symmetric(&cell120_vertices(1.0), "120-cell");
    }

    fn assert_all_unique(verts: &[Vec4], label: &str) {
        for i in 0..verts.len() {
            for j in (i + 1)..verts.len() {
                assert!(
                    (verts[i] - verts[j]).length() > 1e-5,
                    "{label}: duplicate vertices at indices {i} and {j}: {:?}",
                    verts[i]
                );
            }
        }
    }

    #[test]
    fn cell600_vertices_are_unique() {
        assert_all_unique(&cell600_vertices(1.0), "600-cell");
    }

    #[test]
    fn cell120_vertices_are_unique() {
        assert_all_unique(&cell120_vertices(1.0), "120-cell");
    }

    #[test]
    fn icosian_inradius_matches_numerical_max_projection() {
        let r = icosian_inradius_unit();
        let cell600 = cell600_vertices(1.0);
        let cell120 = cell120_vertices(1.0);
        // 120-cell inradius along any 600-cell vertex direction:
        for n in cell600.iter().take(8) {
            let max_proj = cell120
                .iter()
                .map(|v| v.dot(*n))
                .fold(f32::NEG_INFINITY, f32::max);
            assert!(
                (max_proj - r).abs() < 1e-5,
                "120-cell inradius along {n:?}: numerical {max_proj}, constant {r}",
            );
        }
        // 600-cell inradius along any 120-cell vertex direction:
        for n in cell120.iter().take(8) {
            let max_proj = cell600
                .iter()
                .map(|v| v.dot(*n))
                .fold(f32::NEG_INFINITY, f32::max);
            assert!(
                (max_proj - r).abs() < 1e-5,
                "600-cell inradius along {n:?}: numerical {max_proj}, constant {r}",
            );
        }
    }

    #[test]
    fn cell120_face_planes_count_and_unit() {
        let (normals, _r) = cell120_face_planes();
        assert_eq!(normals.len(), 120);
        for (i, n) in normals.iter().enumerate() {
            assert!(
                (n.length() - 1.0).abs() < 1e-5,
                "face normal {i} not unit: {n:?}, |n|={}",
                n.length()
            );
        }
    }

    #[test]
    fn cell600_face_planes_count_and_unit() {
        let (normals, _r) = cell600_face_planes();
        assert_eq!(normals.len(), 600);
        for (i, n) in normals.iter().enumerate() {
            assert!(
                (n.length() - 1.0).abs() < 1e-5,
                "face normal {i} not unit: {n:?}, |n|={}",
                n.length()
            );
        }
    }

    // Tesseract face hyperplanes at unit circumradius: 8 axis planes at ±0.5,
    // inradius 0.5.
    fn tesseract_face_planes() -> (Vec<Vec4>, f32) {
        let normals = vec![
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(-1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, -1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, 0.0, -1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
            Vec4::new(0.0, 0.0, 0.0, -1.0),
        ];
        (normals, 0.5)
    }

    // `outside + inside` decomposition.
    fn tesseract_sdf_truth(p: Vec4, half_extent: f32) -> f32 {
        let q = p.abs() - Vec4::splat(half_extent);
        let outside = q.max(Vec4::ZERO).length();
        let inside = q.x.max(q.y.max(q.z.max(q.w))).min(0.0);
        outside + inside
    }

    #[test]
    fn polytope_sdf_wolfe_matches_tesseract_closed_form() {
        let (normals, r) = tesseract_face_planes();
        let cases = [
            // |S|=0 interior.
            Vec4::ZERO,
            Vec4::new(0.1, 0.2, -0.1, 0.05),
            // |S|=1 face Voronoi.
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.5, 0.0, 0.0),
            // |S|=2 edge Voronoi.
            Vec4::new(1.0, 1.0, 0.0, 0.0),
            // |S|=3 2-face-edge Voronoi.
            Vec4::new(1.0, 1.0, 1.0, 0.0),
            // |S|=4 vertex Voronoi.
            Vec4::new(1.0, 1.0, 1.0, 1.0),
        ];
        for p in cases {
            let truth = tesseract_sdf_truth(p, r);
            let wolfe = polytope_sdf_wolfe(p, &normals, r);
            assert!(
                (truth - wolfe).abs() < 1e-4,
                "p={p:?}: Wolfe={wolfe} != closed-form={truth}",
            );
        }
    }

    #[test]
    fn polytope_sdf_wolfe_120cell_is_lipschitz_1() {
        let (normals, r) = cell120_face_planes();
        let mut state: u32 = 0xACED_F00D;
        let mut nf32 = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 4.0 - 2.0
        };
        for _ in 0..64 {
            let a = Vec4::new(nf32(), nf32(), nf32(), nf32());
            let b = Vec4::new(nf32(), nf32(), nf32(), nf32());
            let dist_ab = (a - b).length();
            if dist_ab < 1e-4 {
                continue;
            }
            let da = polytope_sdf_wolfe(a, &normals, r);
            let db = polytope_sdf_wolfe(b, &normals, r);
            assert!(
                (da - db).abs() <= dist_ab * (1.0 + 1e-4),
                "Lipschitz-1 violated at a={a:?} b={b:?}: |da-db|={} > |a-b|={dist_ab}",
                (da - db).abs()
            );
        }
    }

    #[test]
    fn polytope_sdf_wolfe_120cell_sign_correctness() {
        let (normals, r) = cell120_face_planes();
        // Center: max plane dist = -inradius.
        let d_center = polytope_sdf_wolfe(Vec4::ZERO, &normals, r);
        assert!(
            (d_center + r).abs() < 1e-5,
            "center should give -inradius={}, got {}",
            -r,
            d_center
        );
        // Just inside a face plane: small negative.
        let n = normals[0];
        let just_inside = n * (r - 0.01);
        let d_inside = polytope_sdf_wolfe(just_inside, &normals, r);
        assert!(
            d_inside < 0.0,
            "just inside should be negative, got {d_inside}"
        );
        // Just outside a face plane: small positive, equal to the outward distance.
        let just_outside = n * (r + 0.01);
        let d_outside = polytope_sdf_wolfe(just_outside, &normals, r);
        assert!(
            (d_outside - 0.01).abs() < 1e-4,
            "just outside should give 0.01, got {d_outside}"
        );
    }

    #[test]
    fn cell24_decomposes_into_16cell_plus_tesseract() {
        let c24 = cell24_vertices(1.0);
        let k = 1.0 / 2.0_f32.sqrt();
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
        let pair_found = {
            let (a, b) = world.bodies.dense_mut().split_at_mut(1);
            world.narrowphase.test(&a[0], &b[0], &EuclideanR4).is_some()
        };
        assert!(
            pair_found,
            "sphere inside tesseract should produce a contact"
        );
    }

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
        let (a, b) = world.bodies.dense_mut().split_at_mut(1);
        assert!(world.narrowphase.test(&a[0], &b[0], &EuclideanR4).is_none());
    }

    #[test]
    fn integrated_orientation_advances_a_body_point_along_the_world_frame_omega() {
        let space = EuclideanR4;
        let start = Iso4Flat {
            rotation: Bivector4::new(0.8, 0.0, 0.0, std::f32::consts::FRAC_PI_2, 0.0, 0.9).exp(),
            translation: Vec4::ZERO,
        };
        let omega = Bivector4::new(0.7, 0.0, 0.0, 0.0, 0.5, 0.0);
        let local = Vec4::new(0.5, -0.5, 0.5, 0.5);
        let dt = 1e-3;

        let before = start.rotation.apply(local);
        let after = space
            .integrate_orientation(start, omega, dt)
            .rotation
            .apply(local);
        let residual = (after - (before + omega_cross_r(omega, before) * dt)).length();
        assert!(
            residual < 1e-5,
            "integrated orientation left the world-frame field ω⌋r: residual \
             {residual} over a step of {}",
            (after - before).length()
        );
    }

    #[test]
    fn orientation_integration_preserves_unit_rotor() {
        let space = EuclideanR4;
        let mut iso = Iso4Flat::IDENTITY;
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

    #[test]
    fn fixed_scenario_replay_is_bit_identical_determinism() {
        let first = determinism_scenario_trajectory();
        let second = determinism_scenario_trajectory();
        assert_eq!(first, second, "fixed-scenario replay must be bit-identical");
        for &bits in &first {
            assert!(
                f32::from_bits(bits).is_finite(),
                "non-finite state in replay"
            );
        }
    }

    #[test]
    fn determinism_scenario_stays_above_the_floor_and_never_gains_energy() {
        assert_scenario_stays_physical(&determinism_scenario_run(Schedule::default()));
    }

    #[test]
    fn fixed_scenario_trajectory_matches_golden_determinism_hash() {
        let run = determinism_scenario_run(Schedule::default());
        assert_scenario_stays_physical(&run);
        let hash = fnv1a64(&run.trajectory);
        assert_eq!(
            hash, GOLDEN_TRAJECTORY_HASH,
            "trajectory hash {hash:#018x} does not match the committed golden \
             {GOLDEN_TRAJECTORY_HASH:#018x}; the sanity pin above passed, so \
             this is an intended simulation change and the constant should be \
             re-recorded to {hash:#018x}"
        );
    }
}
