//! Manifolds key on generational [`BodyId`] handles rather than storage
//! positions, so a despawn elsewhere in the world cannot rebind a key to a
//! different pair of bodies.

use crate::body::BodyId;
use crate::collision::VectorOps;
use crate::integrator::PhysicsSpace;
use crate::response::Contact;

/// At most 4 vertex or edge contacts can be coplanar between two convex
/// polytopes, which is why Box2D and rapier also use 4 in 3D.
pub const MAX_POINTS: usize = 4;

// Squared world units. A new contact within this squared distance of a slot
// refreshes that slot. Tuned for unit-scale demos.
const MERGE_RADIUS_SQ: f32 = 0.02 * 0.02;

#[derive(Clone, Copy)]
pub struct ContactPoint<S: PhysicsSpace> {
    pub world_point: S::Point,
    /// Unit, from A toward B, the separating direction.
    pub normal: S::Vector,
    pub penetration: f32,
    /// Persisted across frames; PGS clamps it to ≥ 0.
    pub normal_impulse: f32,
    /// Along the sliding velocity, and valid only within one step: the slide
    /// direction can flip, which would leave a carried signed accumulator
    /// inconsistent with it.
    pub tangent_dir: S::Vector,
    /// Signed along `tangent_dir` and reset to 0 each step. PGS clamps to
    /// `|jt| ≤ μ·jn` only on the iterations that reach the clamp, which is not
    /// all of them (see `world::solve_normal_then_tangent`).
    pub tangent_impulse: f32,
    /// Snapshot taken before the warm-start, combining restitution
    /// (`−e · v_n_pre` while approaching) and Baumgarte correction
    /// (`−β/dt · max(0, pen − slop)`). Constant across the PGS iterations, so
    /// they converge to a post-impulse v_n instead of chasing a moving target.
    pub velocity_bias: f32,
}

pub struct Manifold<S: PhysicsSpace> {
    /// Always `< body_b`.
    pub body_a: BodyId,
    /// Always `> body_a`.
    pub body_b: BodyId,
    /// Set on first contact and kept: per-pair restitution does not change
    /// between frames.
    pub restitution: f32,
    /// `len() ≤ MAX_POINTS`.
    pub points: Vec<ContactPoint<S>>,
}

impl<S: PhysicsSpace> Manifold<S>
where
    S::Vector: VectorOps,
{
    pub fn new(body_a: BodyId, body_b: BodyId, restitution: f32) -> Self {
        debug_assert!(body_a < body_b);
        Self {
            body_a,
            body_b,
            restitution,
            points: Vec::with_capacity(MAX_POINTS),
        }
    }

    /// A slot whose squared distance to `contact.point` is under
    /// `MERGE_RADIUS_SQ` keeps its accumulated impulses, which is the
    /// warm-start carryover. At `MAX_POINTS` the slot
    /// with the smallest total impulse is evicted: dropping it loses the least
    /// warm-start information.
    pub fn add_or_update(&mut self, contact: Contact<S>)
    where
        S::Point: Copy + std::ops::Sub<Output = S::Vector>,
    {
        let new_point = contact.point;

        for cp in &mut self.points {
            let delta = new_point - cp.world_point;
            if VectorOps::length_squared(delta) < MERGE_RADIUS_SQ {
                cp.world_point = new_point;
                cp.normal = contact.normal;
                cp.penetration = contact.penetration;
                return;
            }
        }

        let fresh = ContactPoint {
            world_point: new_point,
            normal: contact.normal,
            penetration: contact.penetration,
            normal_impulse: 0.0,
            tangent_dir: VectorOps::zero(),
            tangent_impulse: 0.0,
            velocity_bias: 0.0,
        };

        if self.points.len() < MAX_POINTS {
            self.points.push(fresh);
        } else {
            let (worst, _) = self
                .points
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let sa = a.normal_impulse + a.tangent_impulse.abs();
                    let sb = b.normal_impulse + b.tangent_impulse.abs();
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            self.points[worst] = fresh;
        }
    }
}

/// The common figure across 2D and 3D rigid-body engines: enough to settle
/// modest stacks without dominating step cost.
pub const DEFAULT_PGS_ITERS: usize = 8;

/// β ∈ [0.1, 0.3] is the standard range; higher corrects faster and injects
/// more energy. 0.2 is the Bullet and rapier default.
pub const BAUMGARTE_BETA: f32 = 0.2;

/// Penetration tolerated without any bias, which is what stops jitter at rest.
/// World units.
pub const PENETRATION_SLOP: f32 = 0.005;

/// Cap on the `β/dt · (penetration − slop)` velocity correction, so a small
/// `dt` cannot blow it up.
pub const MAX_LINEAR_CORRECTION: f32 = 0.5;

/// Approach speed, m/s, below which restitution is suppressed. Without it every
/// body resting on the floor micro-bounces each frame off the gravity-driven
/// approach velocity. The Box2D figure: 1 m/s reads as a noticeable impact at
/// the demos' unit scale.
pub const RESTITUTION_THRESHOLD: f32 = 1.0;

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;
    use loam_math::EuclideanR2;

    fn contact(point: Vec2, normal: Vec2, penetration: f32) -> Contact<EuclideanR2> {
        Contact {
            normal,
            point,
            penetration,
            restitution: 0.0,
        }
    }

    #[test]
    fn merge_preserves_warm_start_impulses() {
        let mut m: Manifold<EuclideanR2> =
            Manifold::new(BodyId::forge(0, 0), BodyId::forge(1, 0), 0.0);
        m.add_or_update(contact(Vec2::ZERO, Vec2::Y, 0.01));
        m.points[0].normal_impulse = 4.2;
        m.points[0].tangent_impulse = -1.7;
        m.points[0].tangent_dir = Vec2::X;

        // Well within MERGE_RADIUS_SQ.
        let merged_point = Vec2::new(0.01, 0.0);
        let merged_normal = Vec2::new(0.0, -1.0);
        m.add_or_update(contact(merged_point, merged_normal, 0.05));

        assert_eq!(m.points.len(), 1, "merge must not add a new slot");
        let cp = &m.points[0];
        assert_eq!(cp.world_point, merged_point, "geometry refreshed");
        assert_eq!(cp.normal, merged_normal, "normal refreshed");
        assert!(
            (cp.penetration - 0.05).abs() < 1e-6,
            "penetration refreshed",
        );
        assert!(
            (cp.normal_impulse - 4.2).abs() < 1e-6,
            "normal impulse preserved across merge",
        );
        assert!(
            (cp.tangent_impulse - -1.7).abs() < 1e-6,
            "tangent impulse preserved across merge",
        );
    }

    #[test]
    fn add_at_max_points_evicts_weakest_slot() {
        let mut m: Manifold<EuclideanR2> =
            Manifold::new(BodyId::forge(0, 0), BodyId::forge(1, 0), 0.0);
        // Far enough apart that none merge.
        let bases = [
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(3.0, 0.0),
        ];
        for &p in &bases {
            m.add_or_update(contact(p, Vec2::Y, 0.0));
        }
        assert_eq!(m.points.len(), MAX_POINTS);

        // Distinct totals so the weakest is unambiguous.
        m.points[0].normal_impulse = 5.0;
        m.points[1].normal_impulse = 3.0;
        m.points[2].normal_impulse = 0.5;
        m.points[3].normal_impulse = 4.0;
        m.points[1].tangent_impulse = -2.0;
        m.points[2].tangent_impulse = 0.1;
        m.points[3].tangent_impulse = 1.0;

        let intruder = Vec2::new(10.0, 10.0);
        m.add_or_update(contact(intruder, Vec2::Y, 0.0));

        assert_eq!(m.points.len(), MAX_POINTS, "size must stay capped");
        assert!(
            m.points.iter().any(|cp| cp.world_point == intruder),
            "new contact must be present",
        );
        assert!(
            m.points.iter().all(|cp| cp.world_point != bases[2]),
            "the lowest-impulse slot must be evicted",
        );
        assert!(
            m.points.iter().any(|cp| cp.world_point == bases[0]),
            "high-impulse slots must be retained",
        );
    }

    #[test]
    fn new_slot_below_capacity_leaves_others_intact() {
        let mut m: Manifold<EuclideanR2> =
            Manifold::new(BodyId::forge(0, 0), BodyId::forge(1, 0), 0.0);
        m.add_or_update(contact(Vec2::ZERO, Vec2::Y, 0.0));
        m.points[0].normal_impulse = 9.0;
        m.points[0].tangent_impulse = 0.5;
        m.points[0].tangent_dir = Vec2::X;

        // Far enough away to not merge.
        m.add_or_update(contact(Vec2::new(1.0, 0.0), Vec2::Y, 0.0));

        assert_eq!(m.points.len(), 2);
        let original = &m.points[0];
        assert_eq!(original.world_point, Vec2::ZERO, "original geometry intact");
        assert!((original.normal_impulse - 9.0).abs() < 1e-6);
        assert!((original.tangent_impulse - 0.5).abs() < 1e-6);
        assert_eq!(original.tangent_dir, Vec2::X);
        let added = &m.points[1];
        assert_eq!(added.normal_impulse, 0.0, "fresh slot starts at zero");
        assert_eq!(added.tangent_impulse, 0.0);
    }
}
