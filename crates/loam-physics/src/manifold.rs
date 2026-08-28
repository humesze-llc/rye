//! Manifolds key on generational [`BodyId`] handles rather than storage
//! positions, so a despawn elsewhere in the world cannot rebind a key to a
//! different pair of bodies.

use crate::body::{BodyId, RigidBody};
use crate::collision::VectorOps;
use crate::integrator::PhysicsSpace;
use crate::response::Contact;

/// At most 4 vertex or edge contacts can be coplanar between two convex
/// polytopes, which is why Box2D and rapier also use 4 in 3D.
pub const MAX_POINTS: usize = 4;

// Squared world units. A new contact within this squared distance of a slot
// refreshes that slot. Tuned for unit-scale demos.
const MERGE_RADIUS_SQ: f32 = 0.02 * 0.02;

/// Separation, in world units, at which [`Manifold::refresh`] drops a retained
/// point: Bullet's `gContactBreakingThreshold` default, applied the way
/// `btPersistentManifold::refreshContactPoints` applies it (Coumans, Bullet
/// Physics SDK 3.x, `BulletCollision/NarrowPhaseCollision`), to the normal gap
/// and to the tangential drift alike. It is the same 0.02 as
/// `MERGE_RADIUS_SQ`, and deliberately: a point that has moved far enough to
/// stop merging with a fresh contact is a point that no longer describes the
/// geometry it was born on.
pub const CONTACT_BREAK_DISTANCE: f32 = 0.02;

const CONTACT_BREAK_DISTANCE_SQ: f32 = CONTACT_BREAK_DISTANCE * CONTACT_BREAK_DISTANCE;

#[derive(Clone, Copy)]
pub struct ContactPoint<S: PhysicsSpace> {
    pub world_point: S::Point,
    /// Witness point on A in A's own frame: the lever from the body position,
    /// de-rotated by its orientation. Re-projecting it is what lets a retained
    /// point be re-validated after the body has moved.
    pub anchor_a: S::Vector,
    /// Witness point on B in B's frame. The pair is created straddling
    /// `world_point` by half the penetration each way along `normal`, so the
    /// gap between the re-projections is the pair's current separation
    /// whatever surface a narrowphase chose to report its point on.
    pub anchor_b: S::Vector,
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

    /// Re-projects every retained point from the anchors it was created with
    /// and drops the ones that no longer describe a contact: the normal gap or
    /// the tangential drift exceeding [`CONTACT_BREAK_DISTANCE`]. Survivors
    /// keep their accumulated impulses and take their `world_point` and
    /// `penetration` from the re-projection, so a point solved this step
    /// describes where the bodies are now.
    ///
    /// Without this a point kept the geometry of the frame it was created in
    /// and was solved again every step, pushing on a feature the body had
    /// already rocked off: a resting hull climbed against gravity.
    ///
    /// `a` and `b` must be the bodies named by [`Self::body_a`] and
    /// [`Self::body_b`], in that order. Retains in slot order, never in hash
    /// order.
    pub fn refresh(&mut self, space: &S, a: &RigidBody<S>, b: &RigidBody<S>) {
        self.points.retain_mut(|cp| {
            let pa = anchor_world_point(space, a, cp.anchor_a);
            let pb = anchor_world_point(space, b, cp.anchor_b);
            let gap = space.log(pa, pb);
            // The normal runs from A toward B, so a positive component means
            // the two anchors have pulled apart along it.
            let separation = VectorOps::dot(gap, cp.normal);
            if separation > CONTACT_BREAK_DISTANCE {
                return false;
            }
            let tangential = gap - cp.normal * separation;
            if VectorOps::length_squared(tangential) > CONTACT_BREAK_DISTANCE_SQ {
                return false;
            }
            cp.world_point = space.exp(pa, gap * 0.5);
            cp.penetration = -separation;
            true
        });
    }

    /// A slot whose squared distance to `contact.point` is under
    /// `MERGE_RADIUS_SQ` keeps its accumulated impulses, which is the
    /// warm-start carryover. At `MAX_POINTS` the slot
    /// with the smallest total impulse is evicted: dropping it loses the least
    /// warm-start information.
    ///
    /// `a` and `b` must be the bodies named by [`Self::body_a`] and
    /// [`Self::body_b`], in that order: the anchors are read back in that
    /// order by [`Self::refresh`].
    pub fn add_or_update(
        &mut self,
        space: &S,
        a: &RigidBody<S>,
        b: &RigidBody<S>,
        contact: Contact<S>,
    ) where
        S::Point: Copy + std::ops::Sub<Output = S::Vector>,
    {
        let new_point = contact.point;
        // A's witness leads B's by the penetration along the normal, which is
        // what makes `refresh` read back a separation and not a drift.
        let half_gap = contact.normal * (0.5 * contact.penetration);
        let anchor_a = local_anchor(space, a, space.exp(new_point, half_gap));
        let anchor_b = local_anchor(space, b, space.exp(new_point, -half_gap));

        for cp in &mut self.points {
            let delta = new_point - cp.world_point;
            if VectorOps::length_squared(delta) < MERGE_RADIUS_SQ {
                cp.world_point = new_point;
                cp.anchor_a = anchor_a;
                cp.anchor_b = anchor_b;
                cp.normal = contact.normal;
                cp.penetration = contact.penetration;
                return;
            }
        }

        let fresh = ContactPoint {
            world_point: new_point,
            anchor_a,
            anchor_b,
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

// The two are exact inverses because every `PhysicsSpace` is flat, where
// `iso_transport` ignores its base point. In a curved space the inverse
// transport would have to be taken at the rotated point, not at the body.
fn local_anchor<S: PhysicsSpace>(space: &S, body: &RigidBody<S>, p: S::Point) -> S::Vector {
    let lever = space.log(body.position, p);
    space.iso_transport(space.iso_inverse(body.orientation), body.position, lever)
}

fn anchor_world_point<S: PhysicsSpace>(
    space: &S,
    body: &RigidBody<S>,
    anchor: S::Vector,
) -> S::Point {
    let lever = space.iso_transport(body.orientation, body.position, anchor);
    space.exp(body.position, lever)
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
    use crate::collider::Collider;
    use glam::Vec2;
    use loam_math::{Bivector, Bivector2, EuclideanR2};

    const SPACE: EuclideanR2 = EuclideanR2;

    fn contact(point: Vec2, normal: Vec2, penetration: f32) -> Contact<EuclideanR2> {
        Contact {
            normal,
            point,
            penetration,
            restitution: 0.0,
        }
    }

    fn body(position: Vec2) -> RigidBody<EuclideanR2> {
        RigidBody::new(
            position,
            Vec2::ZERO,
            Collider::sphere_at_origin(1.0),
            1.0,
            1.0,
            &SPACE,
        )
    }

    // A hull resting on a floor: A above the contact, B the ground below it.
    // The normal runs A toward B, so it points down.
    fn resting_pair() -> (RigidBody<EuclideanR2>, RigidBody<EuclideanR2>) {
        (body(Vec2::new(0.0, 1.0)), body(Vec2::new(0.0, -1.0)))
    }

    #[test]
    fn merge_preserves_warm_start_impulses() {
        let (a, b) = resting_pair();
        let mut m: Manifold<EuclideanR2> =
            Manifold::new(BodyId::forge(0, 0), BodyId::forge(1, 0), 0.0);
        m.add_or_update(&SPACE, &a, &b, contact(Vec2::ZERO, Vec2::Y, 0.01));
        m.points[0].normal_impulse = 4.2;
        m.points[0].tangent_impulse = -1.7;
        m.points[0].tangent_dir = Vec2::X;

        // Well within MERGE_RADIUS_SQ.
        let merged_point = Vec2::new(0.01, 0.0);
        let merged_normal = Vec2::new(0.0, -1.0);
        m.add_or_update(&SPACE, &a, &b, contact(merged_point, merged_normal, 0.05));

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
        let (a, b) = resting_pair();
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
            m.add_or_update(&SPACE, &a, &b, contact(p, Vec2::Y, 0.0));
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
        m.add_or_update(&SPACE, &a, &b, contact(intruder, Vec2::Y, 0.0));

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
        let (a, b) = resting_pair();
        let mut m: Manifold<EuclideanR2> =
            Manifold::new(BodyId::forge(0, 0), BodyId::forge(1, 0), 0.0);
        m.add_or_update(&SPACE, &a, &b, contact(Vec2::ZERO, Vec2::Y, 0.0));
        m.points[0].normal_impulse = 9.0;
        m.points[0].tangent_impulse = 0.5;
        m.points[0].tangent_dir = Vec2::X;

        // Far enough away to not merge.
        m.add_or_update(&SPACE, &a, &b, contact(Vec2::new(1.0, 0.0), Vec2::Y, 0.0));

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

    // Contact at the origin: A rests above it, B below, normal down.
    fn touching_manifold(
        penetration: f32,
    ) -> (
        Manifold<EuclideanR2>,
        RigidBody<EuclideanR2>,
        RigidBody<EuclideanR2>,
    ) {
        let (a, b) = resting_pair();
        let mut m: Manifold<EuclideanR2> =
            Manifold::new(BodyId::forge(0, 0), BodyId::forge(1, 0), 0.0);
        m.add_or_update(&SPACE, &a, &b, contact(Vec2::ZERO, -Vec2::Y, penetration));
        (m, a, b)
    }

    #[test]
    fn refresh_reads_back_the_penetration_the_narrowphase_reported() {
        let (mut m, a, b) = touching_manifold(0.006);
        m.refresh(&SPACE, &a, &b);
        assert_eq!(m.points.len(), 1, "an unmoved contact must survive");
        assert!(
            (m.points[0].penetration - 0.006).abs() < 1e-6,
            "refresh reported {} for a 0.006 penetration, so it is measuring \
             drift and not separation",
            m.points[0].penetration,
        );
        assert!(
            (m.points[0].world_point - Vec2::ZERO).length() < 1e-6,
            "an unmoved contact moved to {}",
            m.points[0].world_point,
        );
    }

    #[test]
    fn refresh_drops_a_point_the_body_has_lifted_off() {
        let (mut m, mut a, b) = touching_manifold(0.006);
        m.points[0].normal_impulse = 3.0;
        a.position.y += 0.006 + CONTACT_BREAK_DISTANCE + 1e-4;
        m.refresh(&SPACE, &a, &b);
        assert!(
            m.points.is_empty(),
            "a point past the break distance was solved again with stale \
             geometry: separation {}",
            -m.points[0].penetration,
        );
    }

    #[test]
    fn refresh_keeps_a_lifted_point_inside_the_break_distance_with_its_impulse() {
        let (mut m, mut a, b) = touching_manifold(0.006);
        m.points[0].normal_impulse = 3.0;
        m.points[0].tangent_impulse = -0.5;
        let lift = 0.006 + 0.5 * CONTACT_BREAK_DISTANCE;
        a.position.y += lift;
        m.refresh(&SPACE, &a, &b);
        assert_eq!(m.points.len(), 1, "the point is still inside the band");
        assert!(
            (m.points[0].penetration + (lift - 0.006)).abs() < 1e-6,
            "penetration {} does not track the lift",
            m.points[0].penetration,
        );
        assert!(
            (m.points[0].normal_impulse - 3.0).abs() < 1e-6
                && (m.points[0].tangent_impulse + 0.5).abs() < 1e-6,
            "a surviving point lost its warm-start impulses",
        );
    }

    #[test]
    fn refresh_drops_a_point_the_bodies_slid_apart_across_the_normal() {
        let (mut m, mut a, b) = touching_manifold(0.006);
        a.position.x += CONTACT_BREAK_DISTANCE + 1e-4;
        m.refresh(&SPACE, &a, &b);
        assert!(
            m.points.is_empty(),
            "a point the body slid off kept describing the feature it left",
        );
    }

    #[test]
    fn refresh_follows_a_point_through_a_body_rotation() {
        // Half a right angle about A's centre carries the contact along an arc
        // of radius 1, which is far past the break distance in every direction.
        let (mut m, mut a, b) = touching_manifold(0.006);
        a.orientation.rotation = Bivector2(std::f32::consts::FRAC_PI_4).exp();
        m.refresh(&SPACE, &a, &b);
        assert!(
            m.points.is_empty(),
            "a rotation that swept the contact a full radius away left it in \
             the manifold",
        );

        // Small enough that the arc stays inside the break distance. The
        // contact sits a unit below A's centre, so the turn lifts it by the
        // sagitta `1 − cos θ` and the penetration must lose exactly that.
        let (mut m, mut a, b) = touching_manifold(0.006);
        let theta = 0.5 * CONTACT_BREAK_DISTANCE;
        a.orientation.rotation = Bivector2(theta).exp();
        m.refresh(&SPACE, &a, &b);
        assert_eq!(m.points.len(), 1, "a tiny rotation must not break contact");
        let expected = 0.006 - (1.0 - theta.cos());
        assert!(
            (m.points[0].penetration - expected).abs() < 2e-6,
            "penetration {} does not track the turn's sagitta {expected}",
            m.points[0].penetration,
        );
    }

    #[test]
    fn refresh_retains_in_slot_order() {
        let (a, b) = resting_pair();
        let mut m: Manifold<EuclideanR2> =
            Manifold::new(BodyId::forge(0, 0), BodyId::forge(1, 0), 0.0);
        let kept = [Vec2::new(-0.5, 0.0), Vec2::new(0.5, 0.0)];
        m.add_or_update(&SPACE, &a, &b, contact(kept[0], -Vec2::Y, 0.006));
        // Straddled by the two survivors, and created already separated past
        // the break distance, so only the middle slot goes.
        m.add_or_update(&SPACE, &a, &b, contact(Vec2::ZERO, -Vec2::Y, -0.1));
        m.add_or_update(&SPACE, &a, &b, contact(kept[1], -Vec2::Y, 0.006));

        m.refresh(&SPACE, &a, &b);

        let surviving: Vec<Vec2> = m.points.iter().map(|cp| cp.world_point).collect();
        assert_eq!(
            surviving.len(),
            2,
            "the separated slot must be the only one dropped"
        );
        for (slot, expected) in surviving.iter().zip(kept) {
            assert!(
                (*slot - expected).length() < 1e-6,
                "retain reordered the slots: {surviving:?}",
            );
        }
    }
}
