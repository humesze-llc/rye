//! Persistent contact manifolds and the projected Gauss-Seidel
//! constraint solver primitives.
//!
//! ## Why persistence
//!
//! Single-contact, single-pass impulse resolution can't keep a stack of
//! bodies stable: the bottom body has one contact with the floor, one
//! with the body above; each frame's resolution applies an impulse,
//! the next frame finds the bodies still slightly overlapping (due to
//! gravity over `dt`), repeat → jitter forever.
//!
//! The standard fix is two-fold:
//!
//! 1. **Persistent manifolds.** Cache up to 4 contact points per pair
//!    across frames so the solver sees the same constraint repeatedly
//!    rather than rediscovering it.
//! 2. **Warm-starting.** At the start of each step, re-apply each
//!    cached point's previous-frame accumulated impulse. Bodies start
//!    near their settled velocities, so the iterative solver converges
//!    in a handful of iterations instead of dozens.
//!
//! ## Scope of this first cut
//!
//! - Contact-to-slot matching is by **world-space proximity** (cheap,
//!   adequate when bodies barely move per frame, which is the
//!   stacking-demo case). A future revision can switch to local-frame
//!   matching for fast-rotating bodies.
//! - Up to 4 contact slots per manifold; replacement policy when full
//!   evicts the slot with smallest accumulated total impulse.
//! - Manifolds are keyed by `(body_a, body_b)` with `body_a < body_b`.
//!   This breaks if bodies are removed mid-simulation; not an issue
//!   today since neither demo removes bodies.

use crate::collision::VectorOps;
use crate::integrator::PhysicsSpace;
use crate::response::Contact;

/// Maximum contact slots per manifold. Box2D / rapier use 4 in 3D
/// because at most 4 vertex / edge contacts can be coplanar between
/// two convex polytopes. Sphere-sphere needs only 1; sphere-polytope
/// usually 1; polytope-polytope can use up to 4 once we add SAT
/// clipping (today the GJK+EPA path emits one per frame, and
/// persistence accumulates up to 4 over ~4 frames as the body settles).
pub const MAX_POINTS: usize = 4;

/// Threshold for "this new contact is the same slot as the old one."
/// In world units. Tuned for unit-scale demos (1m boxes); should
/// scale with body size in a future revision.
const MERGE_RADIUS_SQ: f32 = 0.02 * 0.02;

/// One persistent contact constraint between two bodies. Cached
/// across frames; carries accumulated impulses for warm-starting and
/// the world-space geometry refreshed each frame from narrowphase.
#[derive(Clone, Copy)]
pub struct ContactPoint<S: PhysicsSpace> {
    /// World position the contact is applied at. Refreshed each
    /// frame from the narrowphase; preserved across frames in the
    /// `(slot identity)` sense.
    pub world_point: S::Point,
    /// Unit vector from A toward B (separating direction).
    pub normal: S::Vector,
    /// Penetration depth at this contact this frame.
    pub penetration: f32,
    /// Accumulated normal impulse magnitude. Persisted across
    /// frames; PGS clamps to be ≥ 0.
    pub normal_impulse: f32,
    /// Last-seen tangent direction (along sliding velocity). Cached
    /// only within a single step; reset between steps because the
    /// slide direction can flip and a stale signed accumulator would
    /// then be inconsistent with the new direction.
    pub tangent_dir: S::Vector,
    /// Tangent impulse accumulated within the current step, signed
    /// along `tangent_dir`. Reset to 0 each step; PGS clamps to
    /// `|jt| ≤ μ·jn`.
    pub tangent_impulse: f32,
    /// Snapshot of the velocity bias for this contact, taken before
    /// the warm-start. Combines restitution
    /// (`−e · v_n_pre` for approaching contacts) and Baumgarte
    /// positional correction (`−β/dt · max(0, pen − slop)`). Used as
    /// a constant target inside every PGS iteration so the iterations
    /// converge to the correct post-impulse v_n instead of chasing a
    /// moving target.
    pub velocity_bias: f32,
}

/// Persistent contact data for one pair of bodies.
pub struct Manifold<S: PhysicsSpace> {
    /// Index of body A in `World::bodies`. Always `< body_b`.
    pub body_a: usize,
    /// Index of body B. Always `> body_a`.
    pub body_b: usize,
    /// Combined restitution for this pair. Set on first contact and
    /// kept; per-pair restitution doesn't change between frames.
    pub restitution: f32,
    /// Active contact points. `len() ≤ MAX_POINTS`.
    pub points: Vec<ContactPoint<S>>,
}

impl<S: PhysicsSpace> Manifold<S>
where
    S::Vector: VectorOps,
{
    pub fn new(body_a: usize, body_b: usize, restitution: f32) -> Self {
        debug_assert!(body_a < body_b);
        Self {
            body_a,
            body_b,
            restitution,
            points: Vec::with_capacity(MAX_POINTS),
        }
    }

    /// Merge a fresh narrowphase contact into the manifold:
    ///
    /// - If a slot already exists within `MERGE_RADIUS` of `contact.point`,
    ///   refresh its geometry but preserve its accumulated impulses
    ///   (this is the warm-start carryover).
    /// - Otherwise, add as a new slot. If we're at `MAX_POINTS`, evict
    ///   the slot with smallest total accumulated impulse — a low-
    ///   contribution contact is the best candidate to drop because
    ///   the loss of warm-start info there costs least.
    pub fn add_or_update(&mut self, contact: Contact<S>)
    where
        S::Point: Copy + std::ops::Sub<Output = S::Vector>,
    {
        let new_point = contact.point;

        // Try to merge into an existing slot first.
        for cp in &mut self.points {
            let delta = new_point - cp.world_point;
            if VectorOps::length_squared(delta) < MERGE_RADIUS_SQ {
                cp.world_point = new_point;
                cp.normal = contact.normal;
                cp.penetration = contact.penetration;
                // Keep accumulated normal_impulse / tangent_impulse;
                // they're the warm-start data.
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
            // Evict the slot with smallest |jn| + |jt|.
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

/// Number of PGS iterations per step. 8 is a common sweet spot in
/// 2D / 3D rigid-body engines: enough to settle modest stacks
/// without dominating step cost. Configurable via `World::pgs_iters`
/// when scenes need more.
pub const DEFAULT_PGS_ITERS: usize = 8;

/// Baumgarte bias coefficient: how aggressively the velocity-level
/// constraint corrects positional error per timestep. β ∈ [0.1, 0.3]
/// is the standard range. Higher → faster correction, more energetic
/// (can introduce small bursts of velocity). 0.2 is the Bullet /
/// rapier default.
pub const BAUMGARTE_BETA: f32 = 0.2;

/// Penetration we tolerate without applying the bias (avoids
/// jitter at rest). 0.5 cm at unit scale is typical.
pub const PENETRATION_SLOP: f32 = 0.005;

/// Velocity bias clamp: positional bias contributes a velocity
/// correction `β/dt · (penetration − slop)`, but capped to avoid
/// blowing up small-dt cases.
pub const MAX_LINEAR_CORRECTION: f32 = 0.5;

/// Approaching speed (m/s) below which restitution doesn't apply.
/// Without this, every body at rest on the floor would micro-bounce
/// each frame from gravity-driven approach velocity. Standard Box2D
/// trick: 1 m/s reads as "noticeable impact" in the demos' unit
/// scale.
pub const RESTITUTION_THRESHOLD: f32 = 1.0;
