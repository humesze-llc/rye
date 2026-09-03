use glam::{Quat, Vec3};

use loam_math::{Bivector, Bivector3, EuclideanR3, Iso3};

use crate::body::RigidBody;
use crate::collider::{Collider, ColliderKind};
use crate::collision::{epa, gjk_intersect, ConvexHull, GjkResult, Sphere as GjkSphere};
use crate::integrator::PhysicsSpace;
use crate::narrowphase::Narrowphase;
use crate::response::Contact;

// `(s, xy, yz, zx)` maps to `(w, z, x, y)`, matching `Rotor3::apply`.
fn rotor_to_quat(r: loam_math::Rotor3) -> Quat {
    Quat::from_xyzw(r.yz, r.zx, r.xy, r.s)
}

// `v(r) = ω⌋r`, matching `(ω as pseudovector) × r`.
fn omega_cross_r(w: Bivector3, r: Vec3) -> Vec3 {
    Vec3::new(
        w.zx * r.z - w.xy * r.y,
        w.xy * r.x - w.yz * r.z,
        w.yz * r.y - w.zx * r.x,
    )
}

// `r ∧ f`, matching `r × f` under the correspondence `rotor_to_quat` uses.
fn wedge(r: Vec3, f: Vec3) -> Bivector3 {
    Bivector3::new(
        r.x * f.y - r.y * f.x,
        r.y * f.z - r.z * f.y,
        r.z * f.x - r.x * f.z,
    )
}

fn bivec_mag_sq(b: Bivector3) -> f32 {
    b.xy * b.xy + b.yz * b.yz + b.zx * b.zx
}

fn inv_inertia(body: &RigidBody<EuclideanR3>) -> f32 {
    if body.inv_mass > 0.0 && body.inertia > 0.0 {
        1.0 / body.inertia
    } else {
        0.0
    }
}

impl PhysicsSpace for EuclideanR3 {
    type AngVel = Bivector3;
    type Inertia = f32;

    fn integrate_orientation(&self, iso: Iso3, omega: Bivector3, dt: f32) -> Iso3 {
        debug_assert!(
            omega.xy.is_finite() && omega.yz.is_finite() && omega.zx.is_finite(),
            "non-finite Bivector3 angular velocity in integrate_orientation",
        );
        let delta_rotor = (omega * dt).exp();
        let delta_quat = rotor_to_quat(delta_rotor);
        let composed = delta_quat * iso.rotation;
        let rotation = composed.normalize();
        Iso3 {
            rotation,
            translation: iso.translation,
        }
    }

    fn apply_inv_inertia(&self, inertia: f32, torque: Bivector3) -> Bivector3 {
        if inertia > 0.0 {
            torque * (1.0 / inertia)
        } else {
            Bivector3::ZERO
        }
    }

    fn wedge(&self, a: Vec3, b: Vec3) -> Bivector3 {
        wedge(a, b)
    }

    fn velocity_at_point(&self, body: &RigidBody<EuclideanR3>, p: Vec3) -> Vec3 {
        let r = p - body.position;
        body.velocity + omega_cross_r(body.angular_velocity, r)
    }

    fn effective_mass_inv(
        &self,
        a: &RigidBody<EuclideanR3>,
        b: &RigidBody<EuclideanR3>,
        contact_point: Vec3,
        direction: Vec3,
    ) -> f32 {
        let ra = contact_point - a.position;
        let rb = contact_point - b.position;
        let ra_wedge = wedge(ra, direction);
        let rb_wedge = wedge(rb, direction);
        a.inv_mass
            + b.inv_mass
            + bivec_mag_sq(ra_wedge) * inv_inertia(a)
            + bivec_mag_sq(rb_wedge) * inv_inertia(b)
    }

    fn apply_contact_impulse(
        &self,
        a: &mut RigidBody<EuclideanR3>,
        b: &mut RigidBody<EuclideanR3>,
        contact_point: Vec3,
        direction: Vec3,
        magnitude: f32,
    ) {
        let ra = contact_point - a.position;
        let rb = contact_point - b.position;
        let lin = direction * magnitude;
        a.velocity -= lin * a.inv_mass;
        b.velocity += lin * b.inv_mass;
        let inv_i_a = inv_inertia(a);
        let inv_i_b = inv_inertia(b);
        a.angular_velocity = a.angular_velocity + wedge(ra, lin) * (-inv_i_a);
        b.angular_velocity = b.angular_velocity + wedge(rb, lin) * inv_i_b;
    }
}

fn sphere_sphere_r3(
    a: &RigidBody<EuclideanR3>,
    b: &RigidBody<EuclideanR3>,
    space: &EuclideanR3,
) -> Option<Contact<EuclideanR3>> {
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
    let normal = if len > 1e-8 { log / len } else { Vec3::Y };

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

fn sphere_halfspace_r3(
    a: &RigidBody<EuclideanR3>,
    b: &RigidBody<EuclideanR3>,
    _space: &EuclideanR3,
) -> Option<Contact<EuclideanR3>> {
    let Collider::Sphere { radius, .. } = a.collider else {
        return None;
    };
    let Collider::HalfSpace { normal, offset } = b.collider else {
        return None;
    };
    let signed = a.position.dot(normal) - offset;
    let penetration = radius - signed;
    if penetration <= 0.0 {
        return None;
    }
    let contact_normal = -normal;
    let point = a.position - normal * radius;
    Some(Contact {
        normal: contact_normal,
        point,
        penetration,
        restitution: (a.restitution + b.restitution) * 0.5,
    })
}

fn world_vertices(local: &[Vec3], pos: Vec3, rot: Quat) -> Vec<Vec3> {
    local.iter().map(|&v| rot * v + pos).collect()
}

// Below is a degenerate touch; above is an EPA iteration-cap fallback.
const MIN_POLYTOPE_PENETRATION: f32 = 1e-4;
const MAX_POLYTOPE_PENETRATION: f32 = 5.0;

fn validate_contact(
    info: &crate::collision::ContactInfo,
    a: &RigidBody<EuclideanR3>,
    b: &RigidBody<EuclideanR3>,
) -> Option<Contact<EuclideanR3>> {
    if !info.penetration.is_finite()
        || info.penetration < MIN_POLYTOPE_PENETRATION
        || info.penetration > MAX_POLYTOPE_PENETRATION
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

fn polytope_bounding_radius(local_vertices: &[Vec3]) -> f32 {
    local_vertices
        .iter()
        .map(|v| v.length_squared())
        .fold(0.0_f32, f32::max)
        .sqrt()
}

fn polytope_polytope_r3(
    a: &RigidBody<EuclideanR3>,
    b: &RigidBody<EuclideanR3>,
    _space: &EuclideanR3,
) -> Option<Contact<EuclideanR3>> {
    let Collider::ConvexPolytope3D { vertices: va_local } = &a.collider else {
        return None;
    };
    let Collider::ConvexPolytope3D { vertices: vb_local } = &b.collider else {
        return None;
    };

    let ra = polytope_bounding_radius(va_local);
    let rb = polytope_bounding_radius(vb_local);
    let center_dist_sq = (b.position - a.position).length_squared();
    let combined = ra + rb;
    if center_dist_sq > combined * combined {
        return None;
    }

    let va = world_vertices(va_local, a.position, a.orientation.rotation);
    let vb = world_vertices(vb_local, b.position, b.orientation.rotation);
    let hull_a = ConvexHull { vertices: &va };
    let hull_b = ConvexHull { vertices: &vb };

    let initial_dir = b.position - a.position;
    let simplex = match gjk_intersect(&hull_a, &hull_b, initial_dir) {
        GjkResult::Intersecting { simplex } => simplex,
        GjkResult::Separated => return None,
    };
    let info = epa(&hull_a, &hull_b, simplex, combined)?;
    validate_contact(&info, a, b)
}

fn sphere_polytope_r3(
    a: &RigidBody<EuclideanR3>,
    b: &RigidBody<EuclideanR3>,
    _space: &EuclideanR3,
) -> Option<Contact<EuclideanR3>> {
    let Collider::Sphere { radius, .. } = a.collider else {
        return None;
    };
    let Collider::ConvexPolytope3D { vertices: vb_local } = &b.collider else {
        return None;
    };

    let rb = polytope_bounding_radius(vb_local);
    let center_dist_sq = (b.position - a.position).length_squared();
    let combined = radius + rb;
    if center_dist_sq > combined * combined {
        return None;
    }

    let vb = world_vertices(vb_local, b.position, b.orientation.rotation);
    let support_a = GjkSphere {
        center: a.position,
        radius,
    };
    let support_b = ConvexHull { vertices: &vb };

    let initial_dir = b.position - a.position;
    let simplex = match gjk_intersect(&support_a, &support_b, initial_dir) {
        GjkResult::Intersecting { simplex } => simplex,
        GjkResult::Separated => return None,
    };
    let info = epa(&support_a, &support_b, simplex, combined)?;
    validate_contact(&info, a, b)
}

fn polytope_halfspace_r3(
    a: &RigidBody<EuclideanR3>,
    b: &RigidBody<EuclideanR3>,
    _space: &EuclideanR3,
) -> Option<Contact<EuclideanR3>> {
    let Collider::ConvexPolytope3D { vertices: va_local } = &a.collider else {
        return None;
    };
    let Collider::HalfSpace {
        normal: plane_n,
        offset,
    } = b.collider
    else {
        return None;
    };

    let mut deepest = Vec3::ZERO;
    let mut deepest_depth = 0.0_f32;
    for &v_local in va_local {
        let v_world = a.orientation.rotation * v_local + a.position;
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

pub fn register_default_narrowphase(np: &mut Narrowphase<EuclideanR3>) {
    np.register(ColliderKind::Sphere, ColliderKind::Sphere, sphere_sphere_r3);
    np.register(
        ColliderKind::Sphere,
        ColliderKind::HalfSpace,
        sphere_halfspace_r3,
    );
    np.register(
        ColliderKind::ConvexPolytope3D,
        ColliderKind::ConvexPolytope3D,
        polytope_polytope_r3,
    );
    np.register(
        ColliderKind::Sphere,
        ColliderKind::ConvexPolytope3D,
        sphere_polytope_r3,
    );
    np.register(
        ColliderKind::ConvexPolytope3D,
        ColliderKind::HalfSpace,
        polytope_halfspace_r3,
    );
}

pub fn sphere_inertia(mass: f32, radius: f32) -> f32 {
    (2.0 / 5.0) * mass * radius * radius
}

pub fn sphere_body_r3(
    position: Vec3,
    velocity: Vec3,
    radius: f32,
    mass: f32,
) -> RigidBody<EuclideanR3> {
    RigidBody::new(
        position,
        velocity,
        Collider::sphere_at_origin(radius),
        mass,
        sphere_inertia(mass, radius),
        &EuclideanR3,
    )
}

/// `normal` points to the side the world is on; the plane is `dot(p, normal) = offset`.
pub fn halfspace_body_r3(normal: Vec3, offset: f32) -> RigidBody<EuclideanR3> {
    let n = normal.try_normalize().unwrap_or(Vec3::Y);
    RigidBody::fixed(
        Vec3::ZERO,
        Collider::HalfSpace { normal: n, offset },
        1.0,
        &EuclideanR3,
    )
}

/// Mean of the principal-axis tensor's diagonal; exact for a cube.
pub fn box_inertia(mass: f32, half_extents: Vec3) -> f32 {
    let w = half_extents.x * 2.0;
    let h = half_extents.y * 2.0;
    let d = half_extents.z * 2.0;
    mass * (w * w + h * h + d * d) / 18.0
}

/// CCW-wound, centred at the origin.
pub fn box_vertices(half_extents: Vec3) -> Vec<Vec3> {
    let (hx, hy, hz) = (half_extents.x, half_extents.y, half_extents.z);
    vec![
        Vec3::new(-hx, -hy, -hz),
        Vec3::new(hx, -hy, -hz),
        Vec3::new(hx, hy, -hz),
        Vec3::new(-hx, hy, -hz),
        Vec3::new(-hx, -hy, hz),
        Vec3::new(hx, -hy, hz),
        Vec3::new(hx, hy, hz),
        Vec3::new(-hx, hy, hz),
    ]
}

pub fn box_body(
    position: Vec3,
    velocity: Vec3,
    half_extents: Vec3,
    mass: f32,
) -> RigidBody<EuclideanR3> {
    RigidBody::new(
        position,
        velocity,
        Collider::ConvexPolytope3D {
            vertices: box_vertices(half_extents),
        },
        mass,
        box_inertia(mass, half_extents),
        &EuclideanR3,
    )
}

/// Inertia is the bounding sphere's `(2/5)·m·r²`, order-of-magnitude correct.
pub fn polytope_body(
    position: Vec3,
    velocity: Vec3,
    vertices: Vec<Vec3>,
    mass: f32,
) -> RigidBody<EuclideanR3> {
    let bounding_r_sq = vertices
        .iter()
        .map(|v| v.length_squared())
        .fold(0.0, f32::max);
    let inertia = (2.0 / 5.0) * mass * bounding_r_sq;
    RigidBody::new(
        position,
        velocity,
        Collider::ConvexPolytope3D { vertices },
        mass,
        inertia,
        &EuclideanR3,
    )
}

// The Platonic generators below are centred at the origin with circumradius `r`.

pub fn tetrahedron_vertices(r: f32) -> Vec<Vec3> {
    let k = r / 3.0_f32.sqrt();
    vec![
        Vec3::new(k, k, k),
        Vec3::new(k, -k, -k),
        Vec3::new(-k, k, -k),
        Vec3::new(-k, -k, k),
    ]
}

pub fn cube_vertices(r: f32) -> Vec<Vec3> {
    let h = r / 3.0_f32.sqrt();
    box_vertices(Vec3::splat(h))
}

pub fn octahedron_vertices(r: f32) -> Vec<Vec3> {
    vec![
        Vec3::new(r, 0.0, 0.0),
        Vec3::new(-r, 0.0, 0.0),
        Vec3::new(0.0, r, 0.0),
        Vec3::new(0.0, -r, 0.0),
        Vec3::new(0.0, 0.0, r),
        Vec3::new(0.0, 0.0, -r),
    ]
}

pub fn icosahedron_vertices(r: f32) -> Vec<Vec3> {
    let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let norm = (1.0 + phi * phi).sqrt();
    let s = r / norm;
    let p = phi * s;
    vec![
        Vec3::new(0.0, s, p),
        Vec3::new(0.0, s, -p),
        Vec3::new(0.0, -s, p),
        Vec3::new(0.0, -s, -p),
        Vec3::new(s, p, 0.0),
        Vec3::new(s, -p, 0.0),
        Vec3::new(-s, p, 0.0),
        Vec3::new(-s, -p, 0.0),
        Vec3::new(p, 0.0, s),
        Vec3::new(p, 0.0, -s),
        Vec3::new(-p, 0.0, s),
        Vec3::new(-p, 0.0, -s),
    ]
}

pub fn dodecahedron_vertices(r: f32) -> Vec<Vec3> {
    let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let inv_phi = 1.0 / phi;
    let norm = 3.0_f32.sqrt();
    let s = r / norm;
    let a = s * inv_phi;
    let b = s * phi;
    let mut v = Vec::with_capacity(20);
    for &x in &[-s, s] {
        for &y in &[-s, s] {
            for &z in &[-s, s] {
                v.push(Vec3::new(x, y, z));
            }
        }
    }
    for &y in &[-a, a] {
        for &z in &[-b, b] {
            v.push(Vec3::new(0.0, y, z));
        }
    }
    for &x in &[-a, a] {
        for &z in &[-b, b] {
            v.push(Vec3::new(z, 0.0, x));
        }
    }
    for &x in &[-a, a] {
        for &y in &[-b, b] {
            v.push(Vec3::new(y, x, 0.0));
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::World;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "expected {a} close to {b} (tol {tol})"
        );
    }

    #[test]
    fn head_on_sphere_collision_is_elastic_only() {
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);

        world.push_body(sphere_body_r3(
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            0.5,
            1.0,
        ));
        world.push_body(sphere_body_r3(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-2.0, 0.0, 0.0),
            0.5,
            1.0,
        ));

        for _ in 0..120 {
            world.step(1.0 / 120.0);
        }
        // Contact on the line of centres, so r × n = 0.
        let a = &world.bodies[0];
        let b = &world.bodies[1];
        assert_close(a.angular_velocity.magnitude(), 0.0, 1e-3);
        assert_close(b.angular_velocity.magnitude(), 0.0, 1e-3);
        assert!(a.velocity.x < 0.0, "a.velocity.x = {}", a.velocity.x);
        assert!(b.velocity.x > 0.0, "b.velocity.x = {}", b.velocity.x);
    }

    #[test]
    fn off_center_glancing_hit_produces_angular_velocity() {
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);

        let target_id = world.push_body(sphere_body_r3(Vec3::ZERO, Vec3::ZERO, 0.5, 10.0));
        let projectile_id = world.push_body(sphere_body_r3(
            Vec3::new(2.0, 0.4, 0.0),
            Vec3::new(-5.0, 0.0, 0.0),
            0.3,
            1.0,
        ));

        for _ in 0..120 {
            world.step(1.0 / 240.0);
        }

        let target_omega = world.bodies[target_id].angular_velocity.magnitude();
        let proj_omega = world.bodies[projectile_id].angular_velocity.magnitude();
        assert!(
            target_omega > 1e-3,
            "target gained no angular velocity: ω mag = {target_omega}"
        );
        assert!(
            proj_omega > 1e-3,
            "projectile gained no angular velocity: ω mag = {proj_omega}"
        );
    }

    #[test]
    fn integrated_orientation_advances_a_body_point_along_the_world_frame_omega() {
        let space = EuclideanR3;
        let start = Iso3 {
            rotation: rotor_to_quat(Bivector3::new(0.8, 1.2, 0.9).exp()),
            translation: Vec3::ZERO,
        };
        let omega = Bivector3::new(0.7, 0.0, 0.5);
        let local = Vec3::new(0.5, -0.5, 0.5);
        let dt = 1e-3;

        let before = start.rotation * local;
        let after = space.integrate_orientation(start, omega, dt).rotation * local;
        let residual = (after - (before + omega_cross_r(omega, before) * dt)).length();
        assert!(
            residual < 1e-5,
            "integrated orientation left the world-frame field ω⌋r: residual \
             {residual} over a step of {}",
            (after - before).length()
        );
    }

    #[test]
    fn integration_preserves_unit_rotor() {
        let space = EuclideanR3;
        let mut iso = Iso3::IDENTITY;
        let omega = Bivector3::new(0.2, 0.3, -0.1);
        for _ in 0..1000 {
            iso = space.integrate_orientation(iso, omega, 1.0 / 60.0);
        }
        let len = iso.rotation.length();
        assert!(
            (len - 1.0).abs() < 1e-3,
            "orientation drifted off the unit manifold: |q| = {len}"
        );
    }
}
