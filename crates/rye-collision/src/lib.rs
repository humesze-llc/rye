use glam::Vec3;
use rye_math::Space;

/// A sphere collider in Space coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereCollider {
    pub center: Vec3,
    pub radius: f32,
}

impl SphereCollider {
    pub fn new(center: Vec3, radius: f32) -> Self {
        Self { center, radius }
    }
}

/// Contact manifold for a sphere-sphere collision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    /// Unit vector from `a.center` toward `b.center` along the geodesic.
    pub normal: Vec3,
    /// How far the spheres overlap (`a.radius + b.radius − geodesic_distance`).
    pub penetration: f32,
}

/// Test two sphere colliders in a given Space.
///
/// Returns `Some(Contact)` when the geodesic distance between centers is less
/// than the sum of their radii. The contact normal points from `a` toward `b`.
pub fn sphere_sphere<S>(a: &SphereCollider, b: &SphereCollider, space: &S) -> Option<Contact>
where
    S: Space<Point = Vec3, Vector = Vec3>,
{
    let dist = space.distance(a.center, b.center);
    let combined = a.radius + b.radius;
    if dist >= combined {
        return None;
    }

    // log gives the tangent vector from a → b in the tangent space at a.center.
    let log_vec = space.log(a.center, b.center);
    let len = log_vec.length();
    let normal = if len > 1e-8 {
        log_vec / len
    } else {
        Vec3::Y
    };

    Some(Contact {
        normal,
        penetration: combined - dist,
    })
}

/// Resolve an elastic bounce: reflect the velocity of sphere `a` off sphere `b`.
///
/// Returns the new velocity for `a` after the bounce. Simple coefficient-of-
/// restitution model; does not move the centers.
pub fn elastic_response(velocity: Vec3, contact: &Contact, restitution: f32) -> Vec3 {
    let v_along_normal = velocity.dot(contact.normal);
    if v_along_normal >= 0.0 {
        // Moving away already — no impulse.
        return velocity;
    }
    velocity - contact.normal * v_along_normal * (1.0 + restitution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rye_math::EuclideanR3;

    #[test]
    fn overlapping_spheres_produce_contact() {
        let a = SphereCollider::new(Vec3::ZERO, 1.0);
        let b = SphereCollider::new(Vec3::new(1.5, 0.0, 0.0), 1.0);
        let contact = sphere_sphere(&a, &b, &EuclideanR3).expect("should collide");
        assert!((contact.normal - Vec3::X).length() < 1e-5);
        assert!((contact.penetration - 0.5).abs() < 1e-5);
    }

    #[test]
    fn separated_spheres_produce_no_contact() {
        let a = SphereCollider::new(Vec3::ZERO, 0.4);
        let b = SphereCollider::new(Vec3::new(2.0, 0.0, 0.0), 0.4);
        assert!(sphere_sphere(&a, &b, &EuclideanR3).is_none());
    }

    #[test]
    fn elastic_response_reverses_normal_component() {
        let contact = Contact {
            normal: Vec3::X,
            penetration: 0.1,
        };
        let vel = Vec3::new(-2.0, 1.0, 0.0);
        let out = elastic_response(vel, &contact, 1.0);
        // Perfect restitution: normal component flips, tangent unchanged.
        assert!((out.x - 2.0).abs() < 1e-5);
        assert!((out.y - 1.0).abs() < 1e-5);
    }
}
