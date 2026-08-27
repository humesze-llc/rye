//! Each tick runs a fixed phase order: apply forces, integrate, broadphase,
//! narrowphase, manifold maintenance, warm start, PGS solve. Every phase
//! materialises its work units into a reused buffer and runs the buffer, so
//! [`Schedule`] can reorder a phase without the phase knowing.

use std::collections::BTreeMap;

use loam_math::{EuclideanR2, EuclideanR3, EuclideanR4};
use loam_time::StateHash;

use crate::body::{BodyArena, BodyId, RigidBody};
use crate::collider::Collider;
use crate::collision::VectorOps;
use crate::field::ForceField;
use crate::integrator::{integrate_body, PhysicsSpace};
use crate::manifold::{
    ContactPoint, Manifold, BAUMGARTE_BETA, DEFAULT_PGS_ITERS, MAX_LINEAR_CORRECTION,
    PENETRATION_SLOP, RESTITUTION_THRESHOLD,
};
use crate::narrowphase::Narrowphase;
use crate::response::FRICTION_COEFF;

/// Pair key for the manifold cache. Convention: `(small, large)` so a pair
/// has one canonical key regardless of broadphase iteration order.
pub type PairKey = (BodyId, BodyId);

fn canonical_pair(a: BodyId, b: BodyId) -> PairKey {
    debug_assert_ne!(a, b, "a body cannot pair with itself");
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// One connected component of the contact graph: a set of bodies no other
/// island's solve can reach, and the constraints coupling them. Membership is
/// over dynamic bodies only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Island {
    /// Lowest handle among [`Self::bodies`].
    pub id: BodyId,
    /// The island's dynamic bodies, ascending.
    pub bodies: Vec<BodyId>,
    /// The manifolds coupling them, ascending. A contact against a static body
    /// belongs to the island of its dynamic side.
    pub constraints: Vec<PairKey>,
}

/// How a step's work units are executed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Schedule {
    /// Worker count. Fixed at 1 until an executor lands, and permanently 1 on
    /// wasm32.
    pub threads: usize,
    pub order: OrderPolicy,
}

impl Default for Schedule {
    fn default() -> Self {
        Self {
            threads: 1,
            order: OrderPolicy::Canonical,
        }
    }
}

/// Visit order for one phase's work-unit buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderPolicy {
    Canonical,
    /// The adversarial case for a Gauss-Seidel sweep: every dependency edge
    /// traversed against the canonical direction.
    Reversed {
        phase: SchedulePhase,
    },
    Permuted {
        phase: SchedulePhase,
        seed: u64,
    },
}

/// A phase group sharing one work-unit buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulePhase {
    /// Body visit order, shared by `apply_forces` and `integrate`.
    Body,
    /// Pair visit order in `update_manifolds`.
    BroadphasePair,
    /// Constraint visit order, shared by `prepare_solve`, `warm_start`, and
    /// every `solve` sweep.
    Constraint,
}

impl OrderPolicy {
    fn apply<T>(self, phase: SchedulePhase, units: &mut [T]) {
        match self {
            OrderPolicy::Canonical => {}
            OrderPolicy::Reversed { phase: target } if target == phase => units.reverse(),
            OrderPolicy::Permuted {
                phase: target,
                seed,
            } if target == phase => shuffle(units, seed),
            _ => {}
        }
    }
}

/// Durstenfeld's in-place Fisher-Yates shuffle (Fisher and Yates 1938, table
/// XXXIII; Durstenfeld 1964, CACM 7(7):420) driven by xorshift64 (Marsaglia
/// 2003, "Xorshift RNGs", J. Stat. Soft. 8(14), the 13/7/17 triple). Modulo
/// bias is accepted: the requirement is a reproducible permutation reportable
/// by seed, not a uniform one.
fn shuffle<T>(units: &mut [T], seed: u64) {
    // xorshift64 is absorbing at zero, so a zero seed must not reach it.
    let mut state = seed | 1;
    for i in (1..units.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        units.swap(i, (state % (i as u64 + 1)) as usize);
    }
}

const STALE_CONSTRAINT_KEY: &str = "constraint buffer outlived its manifold";
const STALE_MANIFOLD_BODY: &str = "manifold key names a body that is gone";

/// Relative widening applied to each sweep interval. The cull rests on
/// `|d(anchor, a) − d(anchor, b)| ≤ d(a, b)`, the triangle inequality for the
/// Riemannian distance function (do Carmo 1992, *Riemannian Geometry*, ch. 7,
/// prop. 3.6), which holds exactly in R but not in f32: each of the three
/// distances carries a few ulps of error, so a pair within an ulp of tangency
/// could be culled and the emitted set would stop being a function of the body
/// set alone. Four eps per side covers all three error terms.
const BROADPHASE_TRIANGLE_SLACK: f32 = 4.0 * f32::EPSILON;

#[derive(Clone, Copy)]
struct RadialInterval {
    lo: f32,
    hi: f32,
    radius: f32,
    dense: u32,
    id: BodyId,
    dynamic: bool,
}

/// Every narrowphase poses local geometry as `rotation · v + position` and a
/// rotation preserves norms, so the largest local vertex norm bounds the body
/// at any orientation. `Sphere`'s and `HyperSphere4D`'s `center` is ignored
/// for the same reason the narrowphases ignore it: in physics the body
/// position is the centre.
fn bounding_radius(collider: &Collider) -> f32 {
    match collider {
        Collider::Sphere { radius, .. } | Collider::HyperSphere4D { radius, .. } => *radius,
        Collider::Box3 { half_extents } => half_extents.length(),
        Collider::Polygon2D { vertices } => max_norm(vertices.iter().map(|v| v.length_squared())),
        Collider::ConvexPolytope3D { vertices } => {
            max_norm(vertices.iter().map(|v| v.length_squared()))
        }
        Collider::ConvexPolytope4D { vertices } => {
            max_norm(vertices.iter().map(|v| v.length_squared()))
        }
        Collider::HalfSpace { .. } | Collider::HalfSpace4D { .. } => f32::INFINITY,
    }
}

fn max_norm(norms_squared: impl Iterator<Item = f32>) -> f32 {
    norms_squared.fold(0.0_f32, f32::max).sqrt()
}

#[derive(Clone, Copy)]
struct ConstraintUnit {
    island: BodyId,
    key: PairKey,
    /// Positions of `key.0` and `key.1`, in the key's order.
    dense: (usize, usize),
}

#[cfg(test)]
#[derive(Default)]
struct VisitLog {
    apply_forces: Vec<usize>,
    integrate: Vec<usize>,
    update_manifolds: Vec<PairKey>,
    prepare_solve: Vec<PairKey>,
    warm_start: Vec<PairKey>,
    solve_sweeps: Vec<PairKey>,
}

/// The whole simulation state, and the unit a scheduler would hand across a
/// thread boundary. `Send + Sync` is part of that contract:
///
/// ```
/// # use loam_math::EuclideanR3;
/// # use loam_physics::World;
/// const fn assert_send_sync<T: Send + Sync>() {}
/// const _: () = assert_send_sync::<World<EuclideanR3>>();
/// ```
///
/// ```compile_fail
/// # use loam_math::EuclideanR3;
/// # use loam_physics::World;
/// const fn assert_send_sync<T: Send + Sync>() {}
/// const _: () = assert_send_sync::<(World<EuclideanR3>, std::rc::Rc<u32>)>();
/// ```
pub struct World<S: PhysicsSpace> {
    pub space: S,
    pub bodies: BodyArena<S>,
    pub fields: Vec<Box<dyn ForceField<S>>>,
    pub narrowphase: Narrowphase<S>,
    /// Persistent contact manifolds, keyed `(body_a, body_b)` with `a < b`.
    /// `BTreeMap` for deterministic iteration: PGS convergence depends on
    /// constraint visit order, which must not be hash order (Tier-0
    /// determinism invariant).
    pub manifolds: BTreeMap<PairKey, Manifold<S>>,
    /// PGS iterations per step. Defaults to [`DEFAULT_PGS_ITERS`].
    pub pgs_iters: usize,
    pub time: f32,
    pub schedule: Schedule,
    /// Work-unit buffers, refilled and reordered at the head of their phase
    /// group and retained across steps so the seam allocates once.
    body_order: Vec<usize>,
    pair_order: Vec<PairKey>,
    constraints: Vec<ConstraintUnit>,
    broadphase_intervals: Vec<RadialInterval>,
    broadphase_active: Vec<u32>,
    touched_pairs: Vec<PairKey>,
    island_parent: Vec<u32>,
    island_labels: Vec<BodyId>,
    #[cfg(test)]
    visit_log: VisitLog,
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<World<EuclideanR2>>();
    assert_send_sync::<World<EuclideanR3>>();
    assert_send_sync::<World<EuclideanR4>>();
};

impl<S: PhysicsSpace> World<S> {
    pub fn new(space: S) -> Self {
        Self {
            space,
            bodies: BodyArena::new(),
            fields: Vec::new(),
            narrowphase: Narrowphase::new(),
            manifolds: BTreeMap::new(),
            pgs_iters: DEFAULT_PGS_ITERS,
            time: 0.0,
            schedule: Schedule::default(),
            body_order: Vec::new(),
            pair_order: Vec::new(),
            constraints: Vec::new(),
            broadphase_intervals: Vec::new(),
            broadphase_active: Vec::new(),
            touched_pairs: Vec::new(),
            island_parent: Vec::new(),
            island_labels: Vec::new(),
            #[cfg(test)]
            visit_log: VisitLog::default(),
        }
    }

    pub fn push_body(&mut self, body: RigidBody<S>) -> BodyId {
        self.bodies.spawn(body)
    }

    /// Remove a body and every manifold it takes part in. Returns false if the
    /// handle is stale.
    pub fn despawn_body(&mut self, id: BodyId) -> bool {
        if self.bodies.despawn(id).is_none() {
            return false;
        }
        self.manifolds.retain(|&(a, b), _| a != id && b != id);
        true
    }

    pub fn push_field(&mut self, field: Box<dyn ForceField<S>>) {
        self.fields.push(field);
    }

    /// Advance the simulation by `dt` seconds.
    pub fn step(&mut self, dt: f32)
    where
        S::Vector: VectorOps,
        S::Point: Copy + std::ops::Sub<Output = S::Vector>,
    {
        self.collect_bodies();
        self.apply_forces(dt);
        self.integrate(dt);
        self.update_manifolds();
        self.collect_constraints();
        self.prepare_solve(dt);
        self.warm_start();
        self.solve();

        self.time += dt;
    }

    /// Refilled rather than reused in place so a permutation cannot compound
    /// across steps.
    fn collect_bodies(&mut self) {
        self.body_order.clear();
        self.body_order.extend(0..self.bodies.len());
        self.schedule
            .order
            .apply(SchedulePhase::Body, &mut self.body_order);
    }

    fn apply_forces(&mut self, dt: f32)
    where
        S::Vector: VectorOps,
    {
        #[cfg(test)]
        self.visit_log.apply_forces.clear();
        let order = std::mem::take(&mut self.body_order);
        for &i in &order {
            #[cfg(test)]
            self.visit_log.apply_forces.push(i);
            let body = &mut self.bodies[i];
            if body.inv_mass == 0.0 {
                continue;
            }
            for field in &self.fields {
                let f = field.force_at(body, self.time);
                body.velocity = body.velocity + f * (dt * body.inv_mass);
            }
        }
        self.body_order = order;
    }

    fn integrate(&mut self, dt: f32)
    where
        S::Vector: VectorOps,
    {
        #[cfg(test)]
        self.visit_log.integrate.clear();
        let order = std::mem::take(&mut self.body_order);
        for &i in &order {
            #[cfg(test)]
            self.visit_log.integrate.push(i);
            integrate_body(&self.space, &mut self.bodies[i], dt);
        }
        self.body_order = order;
    }

    /// Untouched pairs are evicted so stale warm-start impulses can't leak into
    /// the next solve.
    fn update_manifolds(&mut self)
    where
        S::Vector: VectorOps,
        S::Point: Copy + std::ops::Sub<Output = S::Vector>,
    {
        let mut pairs = std::mem::take(&mut self.pair_order);
        let mut intervals = std::mem::take(&mut self.broadphase_intervals);
        let mut active = std::mem::take(&mut self.broadphase_active);
        Self::fill_broadphase(
            &self.bodies,
            &self.space,
            &mut intervals,
            &mut active,
            &mut pairs,
        );
        self.broadphase_intervals = intervals;
        self.broadphase_active = active;
        self.schedule
            .order
            .apply(SchedulePhase::BroadphasePair, &mut pairs);
        let mut touched = std::mem::take(&mut self.touched_pairs);
        touched.clear();
        #[cfg(test)]
        self.visit_log.update_manifolds.clear();

        for &key in &pairs {
            #[cfg(test)]
            self.visit_log.update_manifolds.push(key);
            let (i, j) = self.dense_pair(key);
            let (a, b) = split_two_mut(self.bodies.dense_mut(), i, j);
            let Some(contact) = self.narrowphase.test(a, b, &self.space) else {
                continue;
            };
            touched.push(key);
            let restitution = contact.restitution;
            let manifold = self
                .manifolds
                .entry(key)
                .or_insert_with(|| Manifold::new(key.0, key.1, restitution));
            manifold.add_or_update(contact);
        }

        // Sorted rather than hashed so the eviction membership test runs out of
        // a retained buffer; the schedule may have handed the loop its pairs in
        // any order, so insertion order is not already ascending.
        touched.sort_unstable();
        self.manifolds
            .retain(|k, _| touched.binary_search(k).is_ok());
        self.touched_pairs = touched;
        self.pair_order = pairs;
    }

    /// One buffer serves `prepare_solve`, `warm_start`, and `solve`, so those
    /// three always agree on the constraint order. Nothing between here and
    /// the end of the solve inserts or removes a manifold, which is why those
    /// three phases can index by key without a fallback.
    fn collect_constraints(&mut self) {
        let mut parent = std::mem::take(&mut self.island_parent);
        let mut labels = std::mem::take(&mut self.island_labels);
        Self::fill_islands(
            &self.bodies,
            self.manifolds.keys().copied(),
            &mut parent,
            &mut labels,
        );
        let mut units = std::mem::take(&mut self.constraints);
        units.clear();
        units.extend(self.manifolds.keys().map(|&key| {
            let dense = self.dense_pair(key);
            ConstraintUnit {
                island: constraint_island(&self.bodies, &labels, dense),
                key,
                dense,
            }
        }));
        units.sort_unstable_by_key(|unit| (unit.island, unit.key));
        self.island_parent = parent;
        self.island_labels = labels;
        self.schedule
            .order
            .apply(SchedulePhase::Constraint, &mut units);
        self.constraints = units;
    }

    /// Must run before warm-start so the bias reflects the true approach
    /// velocity, not the post-warm-start v_n; otherwise restitution chases a
    /// moving target and converges to zero bounce.
    fn prepare_solve(&mut self, dt: f32)
    where
        S::Vector: VectorOps,
    {
        #[cfg(test)]
        self.visit_log.prepare_solve.clear();
        let units = std::mem::take(&mut self.constraints);
        for unit in &units {
            #[cfg(test)]
            self.visit_log.prepare_solve.push(unit.key);
            debug_assert_eq!(unit.dense, self.dense_pair(unit.key));
            let (i, j) = unit.dense;
            let manifold = self
                .manifolds
                .get_mut(&unit.key)
                .expect(STALE_CONSTRAINT_KEY);
            let (a, b) = split_two_mut(self.bodies.dense_mut(), i, j);
            for cp in &mut manifold.points {
                let v_rel = self.space.velocity_at_point(b, cp.world_point)
                    - self.space.velocity_at_point(a, cp.world_point);
                let v_n = VectorOps::dot(v_rel, cp.normal);

                let restitution_bias = if v_n < -RESTITUTION_THRESHOLD {
                    manifold.restitution * v_n
                } else {
                    0.0
                };

                let baumgarte_bias = if dt > 0.0 {
                    let target = (cp.penetration - PENETRATION_SLOP).max(0.0) * BAUMGARTE_BETA / dt;
                    -target.min(MAX_LINEAR_CORRECTION / dt)
                } else {
                    0.0
                };

                cp.velocity_bias = restitution_bias + baumgarte_bias;

                // Slide direction can flip between frames, so a stale tangent
                // magnitude would brake the wrong way; re-converges in 1-2 iters.
                cp.tangent_impulse = 0.0;
                cp.tangent_dir = VectorOps::zero();
            }
        }
        self.constraints = units;
    }

    fn warm_start(&mut self)
    where
        S::Vector: VectorOps,
    {
        #[cfg(test)]
        self.visit_log.warm_start.clear();
        let units = std::mem::take(&mut self.constraints);
        for unit in &units {
            #[cfg(test)]
            self.visit_log.warm_start.push(unit.key);
            debug_assert_eq!(unit.dense, self.dense_pair(unit.key));
            let (i, j) = unit.dense;
            let manifold = self.manifolds.get(&unit.key).expect(STALE_CONSTRAINT_KEY);
            let (a, b) = split_two_mut(self.bodies.dense_mut(), i, j);
            for cp in &manifold.points {
                if cp.normal_impulse > 0.0 {
                    self.space.apply_contact_impulse(
                        a,
                        b,
                        cp.world_point,
                        cp.normal,
                        cp.normal_impulse,
                    );
                }
            }
        }
        self.constraints = units;
    }

    /// The pre-snapshotted `velocity_bias` is the fixed target; this loop
    /// chases it and never recomputes restitution or correction.
    fn solve(&mut self)
    where
        S::Vector: VectorOps,
    {
        #[cfg(test)]
        self.visit_log.solve_sweeps.clear();
        let units = std::mem::take(&mut self.constraints);
        for _ in 0..self.pgs_iters {
            for unit in &units {
                #[cfg(test)]
                self.visit_log.solve_sweeps.push(unit.key);
                debug_assert_eq!(unit.dense, self.dense_pair(unit.key));
                let (i, j) = unit.dense;
                let manifold = self
                    .manifolds
                    .get_mut(&unit.key)
                    .expect(STALE_CONSTRAINT_KEY);
                let (a, b) = split_two_mut(self.bodies.dense_mut(), i, j);
                for cp in &mut manifold.points {
                    solve_normal_then_tangent(&self.space, a, b, cp);
                }
            }
        }
        self.constraints = units;
    }

    /// Candidate pairs for the current body configuration: one canonical
    /// [`PairKey`] per pair whose bounding balls overlap, in ascending key
    /// order, skipping pairs of two static bodies.
    pub fn broadphase(&self) -> Vec<PairKey> {
        let mut pairs = Vec::new();
        Self::fill_broadphase(
            &self.bodies,
            &self.space,
            &mut Vec::new(),
            &mut Vec::new(),
            &mut pairs,
        );
        pairs
    }

    /// The sweep axis is geodesic distance to the lowest-handle body: one
    /// `distance` call per body, and defined in a curved space where a
    /// coordinate axis is not. Interval overlap along it is necessary for ball
    /// overlap by the triangle inequality, so the one-axis sweep (Cohen, Lin,
    /// Manocha, Ponamgi 1995, "I-COLLIDE", sec. 3) carries over unchanged.
    fn fill_broadphase(
        bodies: &BodyArena<S>,
        space: &S,
        intervals: &mut Vec<RadialInterval>,
        active: &mut Vec<u32>,
        pairs: &mut Vec<PairKey>,
    ) {
        pairs.clear();
        intervals.clear();
        active.clear();
        let n = bodies.len();
        if n < 2 {
            return;
        }

        let anchor = (0..n)
            .min_by_key(|&dense| bodies.id_at(dense))
            .expect("a non-empty arena has a lowest handle");
        let origin = bodies[anchor].position;

        for dense in 0..n {
            let body = &bodies[dense];
            let radius = bounding_radius(&body.collider);
            let d = space.distance(origin, body.position);
            let slack = d * BROADPHASE_TRIANGLE_SLACK;
            intervals.push(RadialInterval {
                lo: d - radius - slack,
                hi: d + radius + slack,
                radius,
                dense: dense as u32,
                id: bodies.id_at(dense),
                dynamic: body.inv_mass != 0.0,
            });
        }
        // Unstable sort: the stable one allocates a scratch buffer, and the
        // handle tie-break already makes the order total.
        intervals.sort_unstable_by(|a, b| a.lo.total_cmp(&b.lo).then(a.id.cmp(&b.id)));

        for i in 0..n {
            let entry = intervals[i];
            // `lo` is non-decreasing, so an interval that ends before this one
            // starts also ends before every later one starts.
            active.retain(|&open| intervals[open as usize].hi >= entry.lo);
            for &open in active.iter() {
                let other = intervals[open as usize];
                if !entry.dynamic && !other.dynamic {
                    continue;
                }
                let gap = space.distance(
                    bodies[other.dense as usize].position,
                    bodies[entry.dense as usize].position,
                );
                if gap <= other.radius + entry.radius {
                    pairs.push(canonical_pair(other.id, entry.id));
                }
            }
            active.push(i as u32);
        }

        pairs.sort_unstable();
    }

    /// Fold the carried contact state into `hash`: each manifold's key, its
    /// point count, and each point's accumulated normal impulse, in `BTreeMap`
    /// key order.
    pub fn hash_contacts(&self, hash: &mut StateHash) {
        for (key, manifold) in &self.manifolds {
            for id in [key.0, key.1] {
                hash.write_u32(id.slot());
                hash.write_u32(id.generation());
            }
            hash.write_u32(manifold.points.len() as u32);
            for point in &manifold.points {
                hash.write_f32(point.normal_impulse);
            }
        }
    }

    /// One value standing for the whole simulation state: every body in
    /// ascending [`BodyId`] order, then [`Self::hash_contacts`].
    ///
    /// The sampler owes a fixed-width, fixed-order layout, because
    /// [`StateHash`] carries no framing of its own.
    pub fn state_hash(&self, sample_body: impl Fn(&RigidBody<S>, &mut Vec<u32>)) -> u64 {
        let mut order: Vec<BodyId> = (0..self.bodies.len())
            .map(|dense| self.bodies.id_at(dense))
            .collect();
        order.sort_unstable();

        let mut words = Vec::new();
        for id in order {
            sample_body(&self.bodies[id], &mut words);
        }
        let mut hash = StateHash::new();
        hash.write_u32s(&words);
        self.hash_contacts(&mut hash);
        hash.finish()
    }

    /// The islands of the current manifold set, ascending by island id.
    ///
    /// Panics if `manifolds` names a body the arena no longer holds, which is
    /// reachable only between a bare [`BodyArena::despawn`] and the next
    /// [`Self::step`]; [`Self::despawn_body`] is the entry point that keeps the
    /// two consistent.
    pub fn islands(&self) -> Vec<Island> {
        let mut parent = Vec::new();
        let mut labels = Vec::new();
        Self::fill_islands(
            &self.bodies,
            self.manifolds.keys().copied(),
            &mut parent,
            &mut labels,
        );

        let mut by_id: BTreeMap<BodyId, Island> = BTreeMap::new();
        for &key in self.manifolds.keys() {
            let id = constraint_island(&self.bodies, &labels, self.dense_pair(key));
            let island = by_id.entry(id).or_insert_with(|| Island {
                id,
                bodies: Vec::new(),
                constraints: Vec::new(),
            });
            island.constraints.push(key);
            for member in [key.0, key.1] {
                if self.bodies[member].inv_mass != 0.0 {
                    island.bodies.push(member);
                }
            }
        }

        let mut islands: Vec<Island> = by_id.into_values().collect();
        for island in &mut islands {
            island.bodies.sort_unstable();
            island.bodies.dedup();
        }
        islands
    }

    /// A pair with a static body merges nothing: that body absorbs no impulse,
    /// so the two sides of it are independent and joining them would hand a
    /// parallel solver one island where it has two. The label is a post-pass
    /// minimum over each component rather than whichever root the unions
    /// happened to leave, which is what makes an island's identity a function
    /// of the handles in it and not of the order the pairs arrived in.
    fn fill_islands(
        bodies: &BodyArena<S>,
        touched: impl Iterator<Item = PairKey>,
        parent: &mut Vec<u32>,
        labels: &mut Vec<BodyId>,
    ) {
        let n = bodies.len();
        parent.clear();
        parent.extend(0..n as u32);
        labels.clear();
        labels.extend((0..n).map(|dense| bodies.id_at(dense)));

        for key in touched {
            let (i, j) = (
                bodies.dense_index(key.0).expect(STALE_MANIFOLD_BODY),
                bodies.dense_index(key.1).expect(STALE_MANIFOLD_BODY),
            );
            if bodies[i].inv_mass == 0.0 || bodies[j].inv_mass == 0.0 {
                continue;
            }
            let (a, b) = (find_root(parent, i), find_root(parent, j));
            if a != b {
                parent[a.max(b)] = a.min(b) as u32;
            }
        }

        for dense in 0..n {
            let root = find_root(parent, dense);
            labels[root] = labels[root].min(bodies.id_at(dense));
        }
        for dense in 0..n {
            let label = labels[find_root(parent, dense)];
            labels[dense] = label;
        }
    }

    fn dense_pair(&self, key: PairKey) -> (usize, usize) {
        (
            self.bodies.dense_index(key.0).expect(STALE_MANIFOLD_BODY),
            self.bodies.dense_index(key.1).expect(STALE_MANIFOLD_BODY),
        )
    }
}

/// Representative of `dense`'s component, halving the path it walks on the way
/// (Tarjan and van Leeuwen 1984, JACM 31(2), sec. 2: path halving carries the
/// same amortized bound as full compression in one pass).
fn find_root(parent: &mut [u32], mut dense: usize) -> usize {
    while parent[dense] as usize != dense {
        parent[dense] = parent[parent[dense] as usize];
        dense = parent[dense] as usize;
    }
    dense
}

/// The island a constraint is solved in: the island of its dynamic body. Every
/// constraint has one, since the broadphase never emits a pair of two statics.
fn constraint_island<S: PhysicsSpace>(
    bodies: &BodyArena<S>,
    labels: &[BodyId],
    dense: (usize, usize),
) -> BodyId {
    let (i, j) = dense;
    debug_assert!(
        bodies[i].inv_mass != 0.0 || bodies[j].inv_mass != 0.0,
        "a contact between two static bodies has no island to solve in",
    );
    if bodies[i].inv_mass != 0.0 {
        labels[i]
    } else {
        labels[j]
    }
}

/// Caller must ensure `i != j`. The two are ordered by [`BodyId`], not by
/// storage position, so either may be the lower index.
fn split_two_mut<T>(slice: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    debug_assert_ne!(i, j, "split_two_mut requires distinct indices");
    if i < j {
        let (left, right) = slice.split_at_mut(j);
        (&mut left[i], &mut right[0])
    } else {
        let (left, right) = slice.split_at_mut(i);
        (&mut right[0], &mut left[j])
    }
}

/// `jn ≥ 0` holds on exit unconditionally; Coulomb's `jt ≤ μ·jn` does not. The
/// tangent solve returns before its clamp when the relative tangential velocity
/// underflows or the tangent effective mass is degenerate, leaving `jt` at the
/// value an earlier iteration accumulated while the normal solve above may have
/// shrunk `jn` beneath it. A later iteration that does slide pulls `jt` back to
/// `μ·jn`, but only in the accumulator: a reduction is not applied as an
/// impulse.
fn solve_normal_then_tangent<S>(
    space: &S,
    a: &mut RigidBody<S>,
    b: &mut RigidBody<S>,
    cp: &mut ContactPoint<S>,
) where
    S: PhysicsSpace,
    S::Vector: VectorOps,
{
    debug_assert!(
        VectorOps::is_finite(cp.normal) && cp.penetration.is_finite(),
        "non-finite contact in solve_normal_then_tangent",
    );

    let v_rel_n_vec =
        space.velocity_at_point(b, cp.world_point) - space.velocity_at_point(a, cp.world_point);
    let v_n = VectorOps::dot(v_rel_n_vec, cp.normal);
    let k_n = space.effective_mass_inv(a, b, cp.world_point, cp.normal);

    if k_n > 0.0 {
        // Target post-impulse v_n is `−velocity_bias`, clamped so accumulated
        // normal impulse stays ≥ 0.
        let dj = -(v_n + cp.velocity_bias) / k_n;
        let new_acc = (cp.normal_impulse + dj).max(0.0);
        let actual = new_acc - cp.normal_impulse;
        cp.normal_impulse = new_acc;
        if actual.abs() > 0.0 {
            space.apply_contact_impulse(a, b, cp.world_point, cp.normal, actual);
        }
    }

    let v_rel_t_vec =
        space.velocity_at_point(b, cp.world_point) - space.velocity_at_point(a, cp.world_point);
    let v_t_vec = v_rel_t_vec - cp.normal * VectorOps::dot(v_rel_t_vec, cp.normal);
    let v_t_mag = VectorOps::length(v_t_vec);

    if v_t_mag < 1e-8 {
        return;
    }

    let tangent = v_t_vec * (1.0 / v_t_mag);
    let k_t = space.effective_mass_inv(a, b, cp.world_point, tangent);
    if k_t <= 0.0 {
        return;
    }

    // Accumulated as a magnitude-only positive scalar within the step (cleared
    // in `prepare_solve`); `tangent_dir` snapshots the direction.
    let dj_t = v_t_mag / k_t;
    let max_friction = cp.normal_impulse * FRICTION_COEFF;
    let new_acc = (cp.tangent_impulse + dj_t).min(max_friction);
    let actual = new_acc - cp.tangent_impulse;
    cp.tangent_impulse = new_acc;
    cp.tangent_dir = tangent;

    if actual > 0.0 {
        // tangent points along the slide; apply along −tangent to brake it.
        space.apply_contact_impulse(a, b, cp.world_point, tangent, -actual);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::determinism_fixture::{
        assert_scenario_stays_physical, determinism_scenario_run, first_divergent_step, fnv1a64,
        multi_island_groups, multi_island_scenario_run, multi_island_world,
        record_flick_chamber_tape, replay_flick_chamber_tape, sample_body_r3, ScenarioRun,
        GOLDEN_MULTI_ISLAND_HASH, GOLDEN_TRAJECTORY_HASH, MULTI_ISLAND_DT, MULTI_ISLAND_STEPS,
        REPLAY_SEED, REPLAY_TICKS,
    };
    use crate::euclidean_r3::{
        box_body, halfspace_body_r3, register_default_narrowphase, sphere_body_r3,
    };
    use crate::field::Gravity;
    use glam::Vec3;
    use loam_math::{Bivector3, EuclideanR3, Space};
    use loam_time::Tape;

    const PERMUTATION_SEEDS: [u64; 4] = [1, 0x9e37_79b9_7f4a_7c15, 0xdead_beef_cafe_f00d, 424_242];

    fn constraint_order<S: PhysicsSpace>(world: &World<S>) -> Vec<PairKey> {
        world.constraints.iter().map(|unit| unit.key).collect()
    }

    /// The test runner gives each test its own thread, so a probe never sees a
    /// concurrent test's allocations; the counter is thread-local rather than
    /// global for exactly that reason. Const-initialised so reading it inside
    /// `alloc` cannot itself allocate.
    mod alloc_probe {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        thread_local! {
            static BYTES: Cell<usize> = const { Cell::new(0) };
        }

        pub struct Counting;

        unsafe impl GlobalAlloc for Counting {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                let _ = BYTES.try_with(|bytes| bytes.set(bytes.get() + layout.size()));
                System.alloc(layout)
            }

            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                System.dealloc(ptr, layout)
            }

            unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
                let _ = BYTES.try_with(|bytes| bytes.set(bytes.get() + new_size));
                System.realloc(ptr, layout, new_size)
            }
        }

        pub fn bytes_allocated_by(body: impl FnOnce()) -> usize {
            let before = BYTES.with(Cell::get);
            body();
            BYTES.with(Cell::get) - before
        }
    }

    #[global_allocator]
    static COUNTING_ALLOCATOR: alloc_probe::Counting = alloc_probe::Counting;

    fn order_variants(phase: SchedulePhase) -> Vec<OrderPolicy> {
        let mut variants = vec![OrderPolicy::Reversed { phase }];
        variants.extend(
            PERMUTATION_SEEDS
                .iter()
                .map(|&seed| OrderPolicy::Permuted { phase, seed }),
        );
        variants
    }

    fn run_with(order: OrderPolicy) -> ScenarioRun {
        determinism_scenario_run(Schedule { threads: 1, order })
    }

    #[test]
    fn global_solve_order_permutation_changes_state_hash_determinism() {
        let canonical = run_with(OrderPolicy::Canonical);
        assert!(
            canonical.step_hashes.len() > 1,
            "fixture produced no steps to compare"
        );
        for order in order_variants(SchedulePhase::Constraint) {
            let permuted = run_with(order);
            assert!(
                first_divergent_step(&canonical, &permuted).is_some(),
                "{order:?} left the state hash identical: the hash cannot see \
                 constraint visit order, so the positive axes below prove nothing"
            );
        }
    }

    fn assert_phase_order_does_not_reach_the_state_hash(phase: SchedulePhase) {
        let canonical = run_with(OrderPolicy::Canonical);
        assert_scenario_stays_physical(&canonical);
        let canonical_hash = fnv1a64(&canonical.trajectory);
        assert_eq!(
            canonical_hash, GOLDEN_TRAJECTORY_HASH,
            "canonical run hashed {canonical_hash:#018x} against the committed \
             {GOLDEN_TRAJECTORY_HASH:#018x}; the sanity pin above passed, so \
             this is an intended simulation change and GOLDEN_TRAJECTORY_HASH \
             should be re-recorded to {canonical_hash:#018x}"
        );

        for order in order_variants(phase) {
            let permuted = run_with(order);
            if let Some(step) = first_divergent_step(&canonical, &permuted) {
                panic!("{order:?} diverged from the canonical schedule at step {step}");
            }
            let word_gap = canonical
                .trajectory
                .iter()
                .zip(&permuted.trajectory)
                .position(|(a, b)| a != b);
            assert!(
                word_gap.is_none() && permuted.trajectory.len() == canonical.trajectory.len(),
                "{order:?} moved trajectory word {word_gap:?}"
            );
            let hash = fnv1a64(&permuted.trajectory);
            assert_eq!(
                hash, GOLDEN_TRAJECTORY_HASH,
                "{order:?} produced {hash:#018x} against the committed golden \
                 {GOLDEN_TRAJECTORY_HASH:#018x}"
            );
        }
    }

    #[test]
    fn body_visit_order_permutation_preserves_state_hash_determinism() {
        assert_phase_order_does_not_reach_the_state_hash(SchedulePhase::Body);
    }

    #[test]
    fn broadphase_pair_order_permutation_preserves_state_hash_determinism() {
        assert_phase_order_does_not_reach_the_state_hash(SchedulePhase::BroadphasePair);
    }

    #[test]
    fn order_policy_permutes_reproducibly_and_never_to_identity_determinism() {
        for len in [7usize, 21] {
            let canonical: Vec<usize> = (0..len).collect();
            for phase in [
                SchedulePhase::Body,
                SchedulePhase::BroadphasePair,
                SchedulePhase::Constraint,
            ] {
                for order in order_variants(phase) {
                    let mut units = canonical.clone();
                    order.apply(phase, &mut units);
                    assert_ne!(units, canonical, "{order:?} on {len} units is the identity");
                    let mut sorted = units.clone();
                    sorted.sort_unstable();
                    assert_eq!(
                        sorted, canonical,
                        "{order:?} on {len} units lost or duplicated a unit"
                    );

                    let mut repeat = canonical.clone();
                    order.apply(phase, &mut repeat);
                    assert_eq!(repeat, units, "{order:?} is not reproducible");

                    for other in [
                        SchedulePhase::Body,
                        SchedulePhase::BroadphasePair,
                        SchedulePhase::Constraint,
                    ] {
                        if other == phase {
                            continue;
                        }
                        let mut untouched = canonical.clone();
                        order.apply(other, &mut untouched);
                        assert_eq!(untouched, canonical, "{order:?} reordered {other:?}");
                    }
                }
            }
        }
    }

    fn assert_buffer_matches_policy<T>(
        order: OrderPolicy,
        owner: SchedulePhase,
        buffer: &[T],
        canonical: &[T],
    ) where
        T: Clone + PartialEq + std::fmt::Debug,
    {
        let mut expected = canonical.to_vec();
        order.apply(owner, &mut expected);
        if matches!(
            order,
            OrderPolicy::Reversed { phase } | OrderPolicy::Permuted { phase, .. } if phase == owner
        ) {
            assert_ne!(
                expected.as_slice(),
                canonical,
                "{order:?} is the identity on this {owner:?} buffer, so the \
                 comparison below would hold with the phase never reordered"
            );
        }
        assert_eq!(
            buffer,
            expected.as_slice(),
            "under {order:?} the retained {owner:?} buffer is wrong: the policy \
             either never reached that phase or reached a different one"
        );
    }

    fn buffer_seam_orders() -> Vec<OrderPolicy> {
        let mut orders = vec![
            OrderPolicy::Canonical,
            OrderPolicy::Reversed {
                phase: SchedulePhase::Body,
            },
            OrderPolicy::Reversed {
                phase: SchedulePhase::BroadphasePair,
            },
        ];
        orders.extend(order_variants(SchedulePhase::Constraint));
        orders
    }

    #[test]
    fn schedule_reordering_reaches_its_named_phase_buffer_determinism() {
        let dt = 1.0 / 240.0;
        let settle_steps = 200;

        for order in buffer_seam_orders() {
            let mut world = settled_sphere_stack(dt, 0);
            world.schedule = Schedule { threads: 1, order };
            for _ in 0..settle_steps {
                world.step(dt);
            }

            let canonical_bodies: Vec<usize> = (0..world.bodies.len()).collect();
            let canonical_pairs = world.broadphase();
            let canonical_constraints: Vec<PairKey> = world.manifolds.keys().copied().collect();
            assert!(
                canonical_bodies.len() >= 2
                    && canonical_pairs.len() >= 2
                    && canonical_constraints.len() >= 2,
                "{order:?} left a buffer too short for a reorder to be visible: \
                 {} bodies, {} pairs, {} constraints",
                canonical_bodies.len(),
                canonical_pairs.len(),
                canonical_constraints.len()
            );

            assert_buffer_matches_policy(
                order,
                SchedulePhase::Body,
                &world.body_order,
                &canonical_bodies,
            );
            assert_buffer_matches_policy(
                order,
                SchedulePhase::BroadphasePair,
                &world.pair_order,
                &canonical_pairs,
            );
            assert_buffer_matches_policy(
                order,
                SchedulePhase::Constraint,
                &constraint_order(&world),
                &canonical_constraints,
            );
        }
    }

    #[test]
    fn phase_loops_visit_the_buffer_the_schedule_ordered_determinism() {
        let dt = 1.0 / 240.0;
        let settle_steps = 200;

        for order in buffer_seam_orders() {
            let mut world = settled_sphere_stack(dt, 0);
            world.schedule = Schedule { threads: 1, order };
            for _ in 0..settle_steps {
                world.step(dt);
            }

            let canonical_bodies: Vec<usize> = (0..world.bodies.len()).collect();
            let canonical_pairs = world.broadphase();
            let canonical_constraints: Vec<PairKey> = world.manifolds.keys().copied().collect();
            assert!(
                canonical_bodies.len() >= 2
                    && canonical_pairs.len() >= 2
                    && canonical_constraints.len() >= 2,
                "{order:?} left a buffer too short for a reorder to be visible: \
                 {} bodies, {} pairs, {} constraints",
                canonical_bodies.len(),
                canonical_pairs.len(),
                canonical_constraints.len()
            );

            for (phase, visited) in [
                ("apply_forces", &world.visit_log.apply_forces),
                ("integrate", &world.visit_log.integrate),
            ] {
                assert_eq!(
                    visited, &world.body_order,
                    "{phase} under {order:?} visited a list other than the \
                     ordered body buffer"
                );
                assert_buffer_matches_policy(
                    order,
                    SchedulePhase::Body,
                    visited,
                    &canonical_bodies,
                );
            }

            assert_eq!(
                world.visit_log.update_manifolds, world.pair_order,
                "update_manifolds under {order:?} visited a list other than the \
                 ordered pair buffer"
            );
            assert_buffer_matches_policy(
                order,
                SchedulePhase::BroadphasePair,
                &world.visit_log.update_manifolds,
                &canonical_pairs,
            );

            let solve_order = constraint_order(&world);
            for (phase, visited) in [
                ("prepare_solve", &world.visit_log.prepare_solve),
                ("warm_start", &world.visit_log.warm_start),
            ] {
                assert_eq!(
                    visited, &solve_order,
                    "{phase} under {order:?} visited a list other than the \
                     ordered constraint buffer"
                );
                assert_buffer_matches_policy(
                    order,
                    SchedulePhase::Constraint,
                    visited,
                    &canonical_constraints,
                );
            }

            let sweep_len = solve_order.len();
            assert_eq!(
                world.visit_log.solve_sweeps.len(),
                sweep_len * world.pgs_iters,
                "solve under {order:?} logged {} visits, not {} sweeps of {sweep_len}",
                world.visit_log.solve_sweeps.len(),
                world.pgs_iters
            );
            for (sweep, visited) in world
                .visit_log
                .solve_sweeps
                .chunks_exact(sweep_len)
                .enumerate()
            {
                assert_eq!(
                    visited,
                    solve_order.as_slice(),
                    "solve sweep {sweep} under {order:?} visited a list other \
                     than the ordered constraint buffer"
                );
                assert_buffer_matches_policy(
                    order,
                    SchedulePhase::Constraint,
                    visited,
                    &canonical_constraints,
                );
            }
        }
    }

    #[test]
    fn multi_island_scenario_matches_golden_determinism_hash() {
        let run = multi_island_scenario_run(Schedule::default());
        assert_scenario_stays_physical(&run);
        let hash = fnv1a64(&run.trajectory);
        assert_eq!(
            hash, GOLDEN_MULTI_ISLAND_HASH,
            "multi-island trajectory hashed {hash:#018x} against the committed \
             {GOLDEN_MULTI_ISLAND_HASH:#018x}; the sanity pin above passed, so \
             this is an intended simulation change and the constant should be \
             re-recorded to {hash:#018x}"
        );
    }

    #[test]
    fn a_recorded_tape_replays_to_the_same_state_hash_determinism() {
        let tape = record_flick_chamber_tape(REPLAY_SEED);
        assert_eq!(
            tape.ticks(),
            REPLAY_TICKS,
            "the tape must cover the run it recorded"
        );
        assert!(
            tape.checkpoints().len() > 1,
            "one checkpoint would let a divergence hide until the last tick"
        );
        assert_eq!(
            1.0 / tape.tick_hz() as f32,
            MULTI_ISLAND_DT,
            "the header's tick rate must be the step the scenario actually runs"
        );
        assert_eq!(
            replay_flick_chamber_tape(&tape),
            tape.checkpoints(),
            "replay diverged from the recording it was made from"
        );
    }

    #[test]
    fn a_tape_replays_the_same_after_a_round_trip_through_its_byte_format_determinism() {
        let recorded = record_flick_chamber_tape(REPLAY_SEED);
        let decoded = Tape::decode(&recorded.encode()).expect("own encoding decodes");
        assert_eq!(decoded, recorded);
        assert_eq!(replay_flick_chamber_tape(&decoded), decoded.checkpoints());
    }

    #[test]
    fn a_flipped_input_word_moves_the_replayed_state_hash_determinism() {
        let recorded = record_flick_chamber_tape(REPLAY_SEED);
        let throw = (0..recorded.ticks())
            .find(|&tick| recorded.input(tick).expect("inside the tape")[0] != u32::MAX)
            .expect("the scripted stream throws at least once");

        let mut tampered = Tape::new(
            recorded.tick_hz(),
            recorded.seed(),
            recorded.words_per_tick(),
        );
        for tick in 0..recorded.ticks() {
            let mut frame: Vec<u32> = recorded.input(tick).expect("inside the tape").to_vec();
            if tick == throw {
                // Low mantissa bit of the impulse's x component: the smallest
                // edit the format can express.
                frame[2] ^= 1;
            }
            tampered.push_tick(&frame);
        }

        assert_ne!(
            replay_flick_chamber_tape(&tampered),
            recorded.checkpoints(),
            "one ulp of input made no difference, so the tape is not driving \
             the simulation"
        );
    }

    #[test]
    fn state_hash_is_invariant_under_arena_compaction_determinism() {
        let radii = [0.3_f32, 0.4, 0.5];
        let body = |i: usize| {
            sphere_body_r3(
                Vec3::new(i as f32, 2.0 * i as f32, 0.0),
                Vec3::new(0.0, -1.0, 0.5 * i as f32),
                radii[i],
                1.0 + i as f32,
            )
        };

        let mut direct = World::new(EuclideanR3);
        for i in 0..3 {
            direct.push_body(body(i));
        }

        let mut compacted = World::new(EuclideanR3);
        compacted.push_body(body(0));
        let doomed = compacted.push_body(sphere_body_r3(Vec3::ZERO, Vec3::ZERO, 9.0, 4.0));
        compacted.push_body(body(1));
        compacted.push_body(body(2));
        assert!(compacted.despawn_body(doomed));

        let dense_order = |world: &World<EuclideanR3>| {
            world
                .bodies
                .iter()
                .map(|body| body.mass.to_bits())
                .collect::<Vec<_>>()
        };
        assert_ne!(
            dense_order(&direct),
            dense_order(&compacted),
            "the two worlds must differ in storage order or the pin is vacuous"
        );
        assert_eq!(
            direct.state_hash(sample_body_r3),
            compacted.state_hash(sample_body_r3),
        );
    }

    #[test]
    fn state_hash_covers_carried_contact_impulses_determinism() {
        let mut world = multi_island_world(Schedule::default());
        for _ in 0..MULTI_ISLAND_STEPS {
            world.step(MULTI_ISLAND_DT);
        }
        let settled = world.state_hash(sample_body_r3);

        let point = world
            .manifolds
            .values_mut()
            .flat_map(|manifold| manifold.points.iter_mut())
            .next()
            .expect("the fixture rests in contact");
        point.normal_impulse = f32::from_bits(point.normal_impulse.to_bits() ^ 1);

        assert_ne!(
            settled,
            world.state_hash(sample_body_r3),
            "one ulp of warm-start impulse left the state hash unmoved"
        );
    }

    #[test]
    fn multi_island_scenario_stays_above_the_floor_and_never_gains_energy_determinism() {
        assert_scenario_stays_physical(&multi_island_scenario_run(Schedule::default()));
    }

    #[test]
    fn multi_island_contact_graph_stays_three_disjoint_islands_determinism() {
        let groups = multi_island_groups();
        let mut world = multi_island_world(Schedule::default());
        let start_x: Vec<f32> = world.bodies.iter().map(|b| b.position.x).collect();
        let mut contacts_per_group = [0usize; 3];
        let mut chain_contacts_peak = 0usize;

        for _ in 0..MULTI_ISLAND_STEPS {
            world.step(MULTI_ISLAND_DT);
            let mut this_step = [0usize; 3];
            for &(id_a, id_b) in world.manifolds.keys() {
                let (i, j) = (id_a.slot() as usize, id_b.slot() as usize);
                let a = groups.iter().position(|g| g.contains(&i));
                let b = groups.iter().position(|g| g.contains(&j));
                let group = match (a, b) {
                    (Some(x), Some(y)) => {
                        assert_eq!(x, y, "contact ({i}, {j}) joined islands {x} and {y}");
                        x
                    }
                    (Some(x), None) | (None, Some(x)) => x,
                    (None, None) => panic!("contact ({i}, {j}) between two static bodies"),
                };
                this_step[group] += 1;
            }
            for (group, count) in this_step.iter().enumerate() {
                contacts_per_group[group] += count;
            }
            chain_contacts_peak = chain_contacts_peak.max(this_step[0]);
        }

        for (group, count) in contacts_per_group.iter().enumerate() {
            assert!(*count > 0, "island {group} never made contact");
        }
        assert_eq!(
            chain_contacts_peak, 4,
            "the four-body chain never rested as floor-A0, A0-A1, A1-A2, A2-A3"
        );
        let end_x: Vec<f32> = world.bodies.iter().map(|b| b.position.x).collect();
        assert_eq!(
            end_x, start_x,
            "a body left its group's vertical axis, so the island partition is \
             not constant by construction"
        );
    }

    const SPHERE_RADIUS: f32 = 0.5;
    const GRAVITY_Y: f32 = -9.8;

    fn settled_sphere_stack(dt: f32, settle_steps: usize) -> World<EuclideanR3> {
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        world.push_field(Box::new(Gravity::new(Vec3::new(0.0, GRAVITY_Y, 0.0))));

        for level in 0..3 {
            let y = SPHERE_RADIUS + level as f32 * 2.0 * SPHERE_RADIUS;
            let id = world.push_body(sphere_body_r3(
                Vec3::new(0.0, y, 0.0),
                Vec3::ZERO,
                SPHERE_RADIUS,
                1.0,
            ));
            world.bodies[id].restitution = 0.0;
        }
        let floor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0));
        world.bodies[floor].restitution = 0.0;

        for _ in 0..settle_steps {
            world.step(dt);
        }
        world
    }

    mod solver_contracts {
        use glam::{Vec2, Vec4};
        use loam_math::{Bivector2, Bivector4, EuclideanR2, EuclideanR4};

        use super::*;
        use crate::euclidean_r2::{sphere_body, static_wall};
        use crate::euclidean_r4::{halfspace4_body_r4, sphere_body_r4};

        trait SolverSpace: PhysicsSpace<Inertia = f32> {
            fn angular_speed(omega: Self::AngVel) -> f32;
        }

        impl SolverSpace for EuclideanR2 {
            fn angular_speed(omega: Bivector2) -> f32 {
                omega.0.abs()
            }
        }

        impl SolverSpace for EuclideanR3 {
            fn angular_speed(omega: Bivector3) -> f32 {
                omega.magnitude()
            }
        }

        impl SolverSpace for EuclideanR4 {
            fn angular_speed(omega: Bivector4) -> f32 {
                omega.magnitude()
            }
        }

        fn kinetic_energy<S: SolverSpace>(body: &RigidBody<S>) -> f32
        where
            S::Vector: VectorOps,
        {
            let omega = S::angular_speed(body.angular_velocity);
            0.5 * body.mass * VectorOps::length_squared(body.velocity)
                + 0.5 * body.inertia * omega * omega
        }

        const FLOOR_HALF: Vec2 = Vec2::new(50.0, 1.0);

        fn floor_r2() -> RigidBody<EuclideanR2> {
            static_wall(Vec2::new(0.0, -FLOOR_HALF.y), FLOOR_HALF)
        }

        const REBOUND_APPROACH: f32 = 2.0;
        /// One step of approach must bury less than [`PENETRATION_SLOP`] so the
        /// positional bias is identically zero and the rebound is pure
        /// restitution.
        const REBOUND_DT: f32 = 1.0 / 1000.0;
        const REBOUND_GAP: f32 = 0.01;

        const _: () = assert!(
            REBOUND_APPROACH > RESTITUTION_THRESHOLD,
            "below the threshold restitution is deliberately suppressed",
        );
        const _: () = assert!(
            REBOUND_APPROACH * REBOUND_DT < PENETRATION_SLOP,
            "a deeper first-frame burial admits a Baumgarte contribution",
        );

        fn assert_elastic_rebound_conserves_kinetic_energy<S>(
            mut world: World<S>,
            faller: BodyId,
            up: S::Vector,
        ) where
            S: SolverSpace,
            S::Vector: VectorOps,
            S::Point: Copy + std::ops::Sub<Output = S::Vector>,
        {
            let energy_before = kinetic_energy(&world.bodies[faller]);
            let steps = (4.0 * REBOUND_GAP / (REBOUND_APPROACH * REBOUND_DT)).ceil() as usize;
            for _ in 0..steps {
                world.step(REBOUND_DT);
            }

            let body = &world.bodies[faller];
            let rebound = VectorOps::dot(body.velocity, up);
            assert!(rebound > 0.0, "body did not rebound: v·up = {rebound}");
            // The contact lies on the line through the centre of mass, so
            // friction and torque have no lever arm and every joule stays
            // translational.
            let spin = S::angular_speed(body.angular_velocity);
            assert!(spin < 1e-6, "central impact spun the body: |ω| = {spin}");

            let energy_after = kinetic_energy(body);
            let ratio = energy_after / energy_before;
            assert!(
                ratio <= 1.0 + 1e-4,
                "e = 1 impact added energy: {energy_before} -> {energy_after}"
            );
            assert!(
                ratio >= 1.0 - 1e-4,
                "e = 1 impact lost energy: {energy_before} -> {energy_after}"
            );
        }

        #[test]
        fn perfectly_elastic_rebound_conserves_kinetic_energy_r2() {
            let mut world = World::new(EuclideanR2);
            crate::euclidean_r2::register_default_narrowphase(&mut world.narrowphase);
            let disk = world.push_body(sphere_body(
                Vec2::new(0.0, SPHERE_RADIUS + REBOUND_GAP),
                Vec2::new(0.0, -REBOUND_APPROACH),
                SPHERE_RADIUS,
                1.0,
            ));
            let floor = world.push_body(floor_r2());
            world.bodies[disk].restitution = 1.0;
            world.bodies[floor].restitution = 1.0;

            assert_elastic_rebound_conserves_kinetic_energy(world, disk, Vec2::Y);
        }

        #[test]
        fn perfectly_elastic_rebound_conserves_kinetic_energy_r3() {
            let mut world = World::new(EuclideanR3);
            register_default_narrowphase(&mut world.narrowphase);
            let sphere = world.push_body(sphere_body_r3(
                Vec3::new(0.0, SPHERE_RADIUS + REBOUND_GAP, 0.0),
                Vec3::new(0.0, -REBOUND_APPROACH, 0.0),
                SPHERE_RADIUS,
                1.0,
            ));
            let floor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0));
            world.bodies[sphere].restitution = 1.0;
            world.bodies[floor].restitution = 1.0;

            assert_elastic_rebound_conserves_kinetic_energy(world, sphere, Vec3::Y);
        }

        #[test]
        fn perfectly_elastic_rebound_conserves_kinetic_energy_r4() {
            let mut world = World::new(EuclideanR4);
            crate::euclidean_r4::register_default_narrowphase(&mut world.narrowphase);
            let sphere = world.push_body(sphere_body_r4(
                Vec4::new(0.0, SPHERE_RADIUS + REBOUND_GAP, 0.0, 0.0),
                Vec4::new(0.0, -REBOUND_APPROACH, 0.0, 0.0),
                SPHERE_RADIUS,
                1.0,
            ));
            let floor = world.push_body(halfspace4_body_r4(Vec4::Y, 0.0));
            world.bodies[sphere].restitution = 1.0;
            world.bodies[floor].restitution = 1.0;

            assert_elastic_rebound_conserves_kinetic_energy(world, sphere, Vec4::Y);
        }

        const SLIDE_SPEED: f32 = 5.0;
        const SLIDE_DT: f32 = 1.0 / 240.0;
        const SLIDE_STEPS: usize = 240;

        fn assert_tangent_impulse_stays_inside_the_coulomb_cone<S>(
            mut world: World<S>,
            slider: BodyId,
            slide: S::Vector,
            up: S::Vector,
        ) where
            S: PhysicsSpace,
            S::Vector: VectorOps,
            S::Point: Copy + std::ops::Sub<Output = S::Vector>,
        {
            let mut cone_ever_binds = false;
            for _ in 0..SLIDE_STEPS {
                world.step(SLIDE_DT);
                for manifold in world.manifolds.values() {
                    for cp in &manifold.points {
                        let cap = cp.normal_impulse * FRICTION_COEFF;
                        // The 1e-6 slack covers f32 accumulation across
                        // `pgs_iters` passes; the impulses here are of order
                        // 1e-2, so it cannot hide a widened cone.
                        assert!(
                            cp.tangent_impulse <= cap + 1e-6,
                            "friction escaped the cone: jt = {}, μ·jn = {cap}",
                            cp.tangent_impulse
                        );
                        assert!(
                            cp.tangent_impulse >= 0.0,
                            "tangent accumulator went negative: {}",
                            cp.tangent_impulse
                        );
                        if cap > 1e-6 && cp.tangent_impulse >= cap - 1e-6 {
                            cone_ever_binds = true;
                        }
                    }
                }
            }

            assert!(
                cone_ever_binds,
                "friction never saturated, so the clamp was never exercised"
            );
            let body = &world.bodies[slider];
            let along = VectorOps::dot(body.velocity, slide);
            assert!(
                along < SLIDE_SPEED,
                "friction did not brake the slide: v·slide = {along}"
            );

            // Sense, not magnitude: `|ω| > 0` is satisfied by a torque of
            // either sign, so it cannot tell contact friction from its
            // negation. Friction acting below the centre of mass has to spin
            // the body toward rolling, which is the angular term carrying the
            // contact point backwards along the slide relative to the centre.
            let contact = world.space.exp(body.position, up * -SPHERE_RADIUS);
            let angular_at_contact = world.space.velocity_at_point(body, contact) - body.velocity;
            let backspin = VectorOps::dot(angular_at_contact, slide);
            assert!(
                backspin < -1e-3,
                "friction spun the body away from rolling: contact-point \
                 angular velocity along the slide is {backspin}"
            );
        }

        #[test]
        fn tangent_impulse_stays_inside_the_coulomb_cone_r2() {
            let mut world = World::new(EuclideanR2);
            crate::euclidean_r2::register_default_narrowphase(&mut world.narrowphase);
            world.push_field(Box::new(Gravity::new(Vec2::new(0.0, GRAVITY_Y))));
            let disk = world.push_body(sphere_body(
                Vec2::new(0.0, SPHERE_RADIUS),
                Vec2::new(SLIDE_SPEED, 0.0),
                SPHERE_RADIUS,
                1.0,
            ));
            let floor = world.push_body(floor_r2());
            world.bodies[disk].restitution = 0.0;
            world.bodies[floor].restitution = 0.0;

            assert_tangent_impulse_stays_inside_the_coulomb_cone(world, disk, Vec2::X, Vec2::Y);
        }

        #[test]
        fn tangent_impulse_stays_inside_the_coulomb_cone_r3() {
            let mut world = World::new(EuclideanR3);
            register_default_narrowphase(&mut world.narrowphase);
            world.push_field(Box::new(Gravity::new(Vec3::new(0.0, GRAVITY_Y, 0.0))));
            let sphere = world.push_body(sphere_body_r3(
                Vec3::new(0.0, SPHERE_RADIUS, 0.0),
                Vec3::new(SLIDE_SPEED, 0.0, 0.0),
                SPHERE_RADIUS,
                1.0,
            ));
            let floor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0));
            world.bodies[sphere].restitution = 0.0;
            world.bodies[floor].restitution = 0.0;

            assert_tangent_impulse_stays_inside_the_coulomb_cone(world, sphere, Vec3::X, Vec3::Y);
        }

        #[test]
        fn tangent_impulse_stays_inside_the_coulomb_cone_r4() {
            let mut world = World::new(EuclideanR4);
            crate::euclidean_r4::register_default_narrowphase(&mut world.narrowphase);
            world.push_field(Box::new(Gravity::new(Vec4::new(0.0, GRAVITY_Y, 0.0, 0.0))));
            let sphere = world.push_body(sphere_body_r4(
                Vec4::new(0.0, SPHERE_RADIUS, 0.0, 0.0),
                Vec4::new(SLIDE_SPEED, 0.0, 0.0, 0.0),
                SPHERE_RADIUS,
                1.0,
            ));
            let floor = world.push_body(halfspace4_body_r4(Vec4::Y, 0.0));
            world.bodies[sphere].restitution = 0.0;
            world.bodies[floor].restitution = 0.0;

            assert_tangent_impulse_stays_inside_the_coulomb_cone(world, sphere, Vec4::X, Vec4::Y);
        }

        const STACK_DT: f32 = 1.0 / 240.0;
        const STACK_SETTLE_STEPS: usize = 400;
        const STACK_LEVELS: usize = 3;

        fn clear_warm_start<S: PhysicsSpace>(world: &mut World<S>) {
            for manifold in world.manifolds.values_mut() {
                for cp in &mut manifold.points {
                    cp.normal_impulse = 0.0;
                }
            }
        }

        fn velocities<S: PhysicsSpace>(world: &World<S>) -> Vec<S::Vector>
        where
            S::Vector: VectorOps,
        {
            world.bodies.iter().map(|b| b.velocity).collect()
        }

        fn normal_impulses<S: PhysicsSpace>(world: &World<S>) -> Vec<f32> {
            world
                .manifolds
                .values()
                .flat_map(|m| m.points.iter().map(|cp| cp.normal_impulse))
                .collect()
        }

        fn max_velocity_gap<V: VectorOps>(a: &[V], b: &[V]) -> f32 {
            assert_eq!(a.len(), b.len(), "body layouts diverged");
            a.iter()
                .zip(b)
                .map(|(x, y)| VectorOps::length(*x - *y))
                .fold(0.0_f32, f32::max)
        }

        fn max_scalar_gap(a: &[f32], b: &[f32]) -> f32 {
            assert_eq!(a.len(), b.len(), "manifold layouts diverged");
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f32, f32::max)
        }

        fn assert_warm_started_step_matches_cold_started_converged_step<S>(
            fixture: impl Fn() -> World<S>,
        ) where
            S: PhysicsSpace,
            S::Vector: VectorOps,
            S::Point: Copy + std::ops::Sub<Output = S::Vector>,
        {
            let mut warm = fixture();
            let mut cold_converged = fixture();
            let mut cold_default = fixture();

            assert!(
                warm.manifolds
                    .values()
                    .flat_map(|m| &m.points)
                    .any(|cp| cp.normal_impulse > 0.0),
                "fixture carries no warm-start payload"
            );

            clear_warm_start(&mut cold_converged);
            clear_warm_start(&mut cold_default);
            // The reference solution, not another partial solve: raising this
            // to 4000 moves neither assert below.
            cold_converged.pgs_iters = 400;

            warm.step(STACK_DT);
            cold_converged.step(STACK_DT);
            cold_default.step(STACK_DT);

            let converged = velocities(&cold_converged);
            let warm_gap = max_velocity_gap(&velocities(&warm), &converged);
            let cold_gap = max_velocity_gap(&velocities(&cold_default), &converged);

            // Tolerance is on the velocity a single 1/240 s step imparts;
            // gravity alone contributes 0.04 m/s per step, so 1e-5 is a tight
            // fraction of the quantity under test and three orders below the
            // cold-start residual it is distinguishing itself from.
            assert!(
                warm_gap < 1e-5,
                "warm-started step diverged from the converged solve by {warm_gap} m/s"
            );
            assert!(
                warm_gap < cold_gap,
                "warm start bought no convergence: warm {warm_gap} vs cold {cold_gap}"
            );

            let impulse_gap =
                max_scalar_gap(&normal_impulses(&warm), &normal_impulses(&cold_converged));
            let reference = normal_impulses(&cold_converged)
                .into_iter()
                .fold(0.0_f32, f32::max);
            assert!(
                impulse_gap < 1e-3 * reference,
                "warm-started accumulator diverged by {impulse_gap} against a peak impulse of {reference}"
            );
        }

        fn settled_disk_stack_r2() -> World<EuclideanR2> {
            let mut world = World::new(EuclideanR2);
            crate::euclidean_r2::register_default_narrowphase(&mut world.narrowphase);
            world.push_field(Box::new(Gravity::new(Vec2::new(0.0, GRAVITY_Y))));

            for level in 0..STACK_LEVELS {
                let y = SPHERE_RADIUS + level as f32 * 2.0 * SPHERE_RADIUS;
                let id = world.push_body(sphere_body(
                    Vec2::new(0.0, y),
                    Vec2::ZERO,
                    SPHERE_RADIUS,
                    1.0,
                ));
                world.bodies[id].restitution = 0.0;
            }
            let floor = world.push_body(floor_r2());
            world.bodies[floor].restitution = 0.0;

            for _ in 0..STACK_SETTLE_STEPS {
                world.step(STACK_DT);
            }
            world
        }

        fn settled_sphere_stack_r4() -> World<EuclideanR4> {
            let mut world = World::new(EuclideanR4);
            crate::euclidean_r4::register_default_narrowphase(&mut world.narrowphase);
            world.push_field(Box::new(Gravity::new(Vec4::new(0.0, GRAVITY_Y, 0.0, 0.0))));

            for level in 0..STACK_LEVELS {
                let y = SPHERE_RADIUS + level as f32 * 2.0 * SPHERE_RADIUS;
                let id = world.push_body(sphere_body_r4(
                    Vec4::new(0.0, y, 0.0, 0.0),
                    Vec4::ZERO,
                    SPHERE_RADIUS,
                    1.0,
                ));
                world.bodies[id].restitution = 0.0;
            }
            let floor = world.push_body(halfspace4_body_r4(Vec4::Y, 0.0));
            world.bodies[floor].restitution = 0.0;

            for _ in 0..STACK_SETTLE_STEPS {
                world.step(STACK_DT);
            }
            world
        }

        #[test]
        fn warm_started_step_matches_cold_started_converged_step_r2() {
            assert_warm_started_step_matches_cold_started_converged_step(settled_disk_stack_r2);
        }

        #[test]
        fn warm_started_step_matches_cold_started_converged_step_r3() {
            assert_warm_started_step_matches_cold_started_converged_step(|| {
                settled_sphere_stack(STACK_DT, STACK_SETTLE_STEPS)
            });
        }

        #[test]
        fn warm_started_step_matches_cold_started_converged_step_r4() {
            assert_warm_started_step_matches_cold_started_converged_step(settled_sphere_stack_r4);
        }
    }

    const ISLAND_X: [f32; 4] = [-4.0, 0.0, 4.0, 8.0];

    fn settled_islands(dt: f32, settle_steps: usize) -> (World<EuclideanR3>, BodyId, Vec<BodyId>) {
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        world.push_field(Box::new(Gravity::new(Vec3::new(0.0, GRAVITY_Y, 0.0))));

        let floor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0));
        world.bodies[floor].restitution = 0.0;
        let mut spheres = Vec::with_capacity(ISLAND_X.len());
        for x in ISLAND_X {
            let id = world.push_body(island_sphere(x));
            world.bodies[id].restitution = 0.0;
            spheres.push(id);
        }

        for _ in 0..settle_steps {
            world.step(dt);
        }
        (world, floor, spheres)
    }

    fn island_sphere(x: f32) -> RigidBody<EuclideanR3> {
        sphere_body_r3(
            Vec3::new(x, SPHERE_RADIUS, 0.0),
            Vec3::ZERO,
            SPHERE_RADIUS,
            1.0,
        )
    }

    fn body_state(world: &World<EuclideanR3>, id: BodyId) -> (Vec3, Vec3, Bivector3) {
        let body = &world.bodies[id];
        (body.position, body.velocity, body.angular_velocity)
    }

    fn normal_impulses(world: &World<EuclideanR3>, key: PairKey) -> Vec<f32> {
        world.manifolds[&key]
            .points
            .iter()
            .map(|cp| cp.normal_impulse)
            .collect()
    }

    #[test]
    fn bare_arena_despawn_strands_a_manifold_key_until_the_next_step() {
        let dt = 1.0 / 240.0;
        let settle_steps = 400;
        let (mut world, floor, spheres) = settled_islands(dt, settle_steps);
        let doomed = spheres[1];
        assert!(
            world.manifolds.contains_key(&(floor, doomed)),
            "fixture has no manifold on the doomed body, so nothing is stranded"
        );

        assert!(world.bodies.despawn(doomed).is_some());
        assert!(
            world.manifolds.contains_key(&(floor, doomed)),
            "the arena despawn pruned manifolds, so despawn_body is no longer \
             the only removal that keeps the world consistent"
        );
        let resolved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| world.islands()));
        assert!(
            resolved.is_err(),
            "islands resolved a key naming a despawned body"
        );

        world.step(dt);
        assert!(
            !world
                .manifolds
                .keys()
                .any(|&(a, b)| a == doomed || b == doomed),
            "the step did not evict the stranded key"
        );
        assert_eq!(
            world.islands().len(),
            ISLAND_X.len() - 1,
            "the eviction did not close the panic window"
        );

        let (mut control, control_floor, control_spheres) = settled_islands(dt, settle_steps);
        assert!(control.despawn_body(control_spheres[1]));
        assert!(!control
            .manifolds
            .contains_key(&(control_floor, control_spheres[1])));
        assert_eq!(control.islands().len(), ISLAND_X.len() - 1);
    }

    #[test]
    fn despawn_preserves_surviving_manifold_keys_and_warm_start_impulses() {
        let dt = 1.0 / 240.0;
        let settle_steps = 400;
        let (mut world, floor, spheres) = settled_islands(dt, settle_steps);
        let (mut control, _, control_spheres) = settled_islands(dt, settle_steps);

        let doomed = spheres[1];
        let keeper = spheres[3];
        let keeper_key = (floor, keeper);
        let control_key = (floor, control_spheres[3]);

        let impulses_before = normal_impulses(&world, keeper_key);
        assert!(
            impulses_before.iter().any(|&jn| jn > 0.0),
            "fixture carries no warm-start payload, so nothing below is being preserved"
        );
        let keeper_position_before = world.bodies.dense_index(keeper).unwrap();

        assert!(world.despawn_body(doomed));

        assert_ne!(
            world.bodies.dense_index(keeper).unwrap(),
            keeper_position_before,
            "despawn moved no surviving body, so this test never reached the compaction case"
        );
        assert!(
            !world
                .manifolds
                .keys()
                .any(|&(a, b)| a == doomed || b == doomed),
            "the removed body's manifolds outlived it"
        );
        assert_eq!(
            normal_impulses(&world, keeper_key),
            impulses_before,
            "compaction disturbed a surviving manifold's warm-start impulses"
        );

        for _ in 0..60 {
            world.step(dt);
            control.step(dt);
        }

        assert_eq!(
            body_state(&world, keeper),
            body_state(&control, control_spheres[3]),
            "an unrelated despawn perturbed a surviving island"
        );
        let impulses_after = normal_impulses(&world, keeper_key);
        assert_eq!(
            impulses_after,
            normal_impulses(&control, control_key),
            "an unrelated despawn moved a surviving manifold's impulses"
        );
        assert!(
            impulses_after.iter().any(|&jn| jn > 0.0),
            "the surviving contact stopped carrying an impulse"
        );
    }

    #[test]
    fn spawn_mid_simulation_leaves_existing_islands_bit_identical() {
        let dt = 1.0 / 240.0;
        let settle_steps = 400;
        let (mut world, floor, spheres) = settled_islands(dt, settle_steps);
        let (mut control, _, control_spheres) = settled_islands(dt, settle_steps);

        let keeper = spheres[0];
        let keeper_key = (floor, keeper);
        let newcomer = world.push_body(island_sphere(12.0));
        world.bodies[newcomer].restitution = 0.0;

        for _ in 0..60 {
            world.step(dt);
            control.step(dt);
        }

        assert!(
            world.manifolds.contains_key(&(floor, newcomer)),
            "the spawned body never made contact, so it exercised no solver state"
        );
        assert_eq!(
            body_state(&world, keeper),
            body_state(&control, control_spheres[0]),
            "a spawn perturbed an existing island"
        );
        assert_eq!(
            normal_impulses(&world, keeper_key),
            normal_impulses(&control, (floor, control_spheres[0])),
            "a spawn moved an existing manifold's warm-start impulses"
        );
    }

    #[test]
    fn a_recycled_slot_inherits_no_manifold_from_the_previous_body() {
        let dt = 1.0 / 240.0;
        let (mut world, floor, spheres) = settled_islands(dt, 400);
        let doomed = spheres[1];
        assert!(world.manifolds.contains_key(&(floor, doomed)));

        assert!(world.despawn_body(doomed));
        assert!(
            !world.despawn_body(doomed),
            "a stale handle despawned a second body"
        );
        assert!(world.bodies.get(doomed).is_none());

        let reborn = world.push_body(island_sphere(ISLAND_X[1]));
        assert_eq!(
            reborn.slot(),
            doomed.slot(),
            "the slot was not recycled, so this test is not exercising aliasing"
        );
        assert_ne!(reborn, doomed);
        assert!(
            world.bodies.get(doomed).is_none(),
            "the stale handle resolved to the body that took its slot"
        );

        world.step(dt);
        assert!(
            world.manifolds.contains_key(&(floor, reborn)),
            "the respawned body made no contact"
        );
        assert!(
            !world.manifolds.contains_key(&(floor, doomed)),
            "a manifold keyed on the despawned body came back with the slot"
        );
    }

    #[test]
    fn split_two_mut_returns_borrows_in_argument_order() {
        let mut slice = [0u32, 1, 2, 3];
        for (i, j) in [(1usize, 3usize), (3, 1)] {
            let (a, b) = split_two_mut(&mut slice, i, j);
            assert_eq!((*a, *b), (i as u32, j as u32), "split_two_mut({i}, {j})");
        }
    }

    /// Boxes rather than spheres because the box pair runs GJK + EPA, whose
    /// result depends on which hull is the Minkowski minuend. The sphere
    /// narrowphases and the impulse response are exactly antisymmetric under an
    /// operand swap, so a sphere pair would settle to the same state either way
    /// and could not tell the two orders apart.
    fn stacked_pair_world(doomed_first: bool) -> (World<EuclideanR3>, BodyId, BodyId, BodyId) {
        const LOWER_HALF_EXTENT: f32 = 0.5;
        const UPPER_HALF_EXTENT: f32 = 0.35;
        const DOOMED_X: f32 = -6.0;

        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        world.push_field(Box::new(Gravity::new(Vec3::new(0.0, GRAVITY_Y, 0.0))));

        let floor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0));
        world.bodies[floor].restitution = 0.0;

        let spawned_first = doomed_first.then(|| world.push_body(island_sphere(DOOMED_X)));
        let lower = world.push_body(box_body(
            Vec3::new(0.0, LOWER_HALF_EXTENT, 0.0),
            Vec3::ZERO,
            Vec3::splat(LOWER_HALF_EXTENT),
            1.0,
        ));
        let upper = world.push_body(box_body(
            Vec3::new(0.0, 2.0 * LOWER_HALF_EXTENT + UPPER_HALF_EXTENT, 0.0),
            Vec3::ZERO,
            Vec3::splat(UPPER_HALF_EXTENT),
            3.0,
        ));
        let doomed = spawned_first.unwrap_or_else(|| world.push_body(island_sphere(DOOMED_X)));

        for id in [lower, upper, doomed] {
            world.bodies[id].restitution = 0.0;
        }
        (world, lower, upper, doomed)
    }

    fn despawned_pair_world(
        doomed_first: bool,
        dt: f32,
    ) -> (World<EuclideanR3>, BodyId, BodyId, PairKey) {
        const SETTLE_STEPS: usize = 40;

        let (mut world, lower, upper, doomed) = stacked_pair_world(doomed_first);
        for _ in 0..SETTLE_STEPS {
            world.step(dt);
        }

        let key = canonical_pair(lower, upper);
        assert!(
            world.manifolds.contains_key(&key),
            "the two dynamic bodies never settled into contact"
        );
        assert!(world.despawn_body(doomed));

        let (i, j) = world.dense_pair(key);
        assert_eq!(
            i > j,
            doomed_first,
            "the pair is stored at {i}, {j}, which is not the order this fixture \
             was built to produce"
        );
        (world, lower, upper, key)
    }

    #[test]
    fn contact_normal_points_from_the_pair_key_low_body_to_the_high_one() {
        let dt = 1.0 / 240.0;
        for doomed_first in [false, true] {
            let (mut world, _, _, key) = despawned_pair_world(doomed_first, dt);
            world.step(dt);

            let manifold = world.manifolds.get(&key).expect("the pair separated");
            let key_axis = world.bodies[key.1].position - world.bodies[key.0].position;
            assert!(!manifold.points.is_empty(), "manifold carries no contact");
            for cp in &manifold.points {
                assert!(
                    cp.normal.dot(key_axis) > 0.0,
                    "normal {} points back toward the key's low body",
                    cp.normal
                );
            }
        }
    }

    #[test]
    fn storage_order_does_not_reach_a_contacting_pairs_trajectory() {
        let dt = 1.0 / 240.0;
        let steps = 120;
        let mut trajectories = Vec::new();
        for doomed_first in [false, true] {
            let (mut world, lower, upper, key) = despawned_pair_world(doomed_first, dt);
            let mut contact_steps = 0;
            let mut trajectory = Vec::with_capacity(steps);
            for _ in 0..steps {
                world.step(dt);
                if world.manifolds.contains_key(&key) {
                    contact_steps += 1;
                }
                trajectory.push((body_state(&world, lower), body_state(&world, upper)));
            }
            assert_eq!(
                contact_steps, steps,
                "the pair held contact for only {contact_steps} of {steps} steps"
            );
            trajectories.push(trajectory);
        }

        let step = trajectories[0]
            .iter()
            .zip(&trajectories[1])
            .position(|(a, b)| a != b);
        assert!(
            step.is_none(),
            "storage order reached the solve: the pair diverged from the control \
             at step {step:?}"
        );
    }

    /// xorshift64 (Marsaglia 2003, "Xorshift RNGs", J. Stat. Soft. 8(14), the
    /// 13/7/17 triple) so a randomized scene replays from the seed in the
    /// failure message.
    struct Xorshift(u64);

    impl Xorshift {
        fn new(seed: u64) -> Self {
            // Absorbing at zero.
            Self(seed | 1)
        }

        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        /// Uniform in `[lo, hi)` off the top 24 bits, which is the whole f32
        /// significand.
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            let unit = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
            lo + (hi - lo) * unit
        }
    }

    fn random_scene(seed: u64, count: usize, spread: f32) -> World<EuclideanR3> {
        let mut rng = Xorshift::new(seed);
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        world.push_field(Box::new(Gravity::new(Vec3::new(0.0, GRAVITY_Y, 0.0))));
        world.push_body(halfspace_body_r3(Vec3::Y, 0.0));

        let mut spawned = Vec::with_capacity(count);
        for _ in 0..count {
            let position = Vec3::new(
                rng.range(-spread, spread),
                rng.range(0.5, spread + 0.5),
                rng.range(-spread, spread),
            );
            let id = if rng.next_u64() & 1 == 0 {
                world.push_body(sphere_body_r3(
                    position,
                    Vec3::ZERO,
                    rng.range(0.2, 0.8),
                    1.0,
                ))
            } else {
                world.push_body(box_body(
                    position,
                    Vec3::ZERO,
                    Vec3::splat(rng.range(0.2, 0.6)),
                    1.0,
                ))
            };
            if rng.next_u64().is_multiple_of(8) {
                world.bodies[id].mass = 0.0;
                world.bodies[id].inv_mass = 0.0;
            }
            world.bodies[id].restitution = 0.0;
            spawned.push(id);
        }
        for doomed in spawned.iter().step_by(5) {
            assert!(world.despawn_body(*doomed));
        }
        world
    }

    fn all_pairs_reference(world: &World<EuclideanR3>) -> Vec<PairKey> {
        let mut pairs = Vec::new();
        let n = world.bodies.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let (a, b) = (&world.bodies[i], &world.bodies[j]);
                if a.inv_mass == 0.0 && b.inv_mass == 0.0 {
                    continue;
                }
                let reach = bounding_radius(&a.collider) + bounding_radius(&b.collider);
                if world.space.distance(a.position, b.position) <= reach {
                    pairs.push(canonical_pair(world.bodies.id_at(i), world.bodies.id_at(j)));
                }
            }
        }
        pairs.sort_unstable();
        pairs
    }

    fn dynamic_body_count(world: &World<EuclideanR3>) -> usize {
        world.bodies.iter().filter(|b| b.inv_mass != 0.0).count()
    }

    const RANDOM_SCENE_SHAPES: [(usize, f32); 3] = [(12, 2.0), (40, 6.0), (80, 3.0)];

    #[test]
    fn sweep_broadphase_emits_exactly_the_all_pairs_candidate_set_determinism() {
        let mut ever_beyond_the_floor = false;
        for seed in PERMUTATION_SEEDS {
            for (count, spread) in RANDOM_SCENE_SHAPES {
                let mut world = random_scene(seed, count, spread);
                for step in 0..8 {
                    let expected = all_pairs_reference(&world);
                    assert_eq!(
                        world.broadphase(),
                        expected,
                        "seed {seed}, {count} bodies, spread {spread}, step {step}"
                    );
                    ever_beyond_the_floor |= expected.len() > dynamic_body_count(&world);
                    world.step(1.0 / 240.0);
                }
            }
        }
        assert!(
            ever_beyond_the_floor,
            "no scene ever produced a candidate pair between two finite colliders"
        );
    }

    #[test]
    fn broadphase_culls_only_pairs_the_narrowphase_would_reject() {
        for seed in PERMUTATION_SEEDS {
            let mut world = random_scene(seed, 40, 3.0);
            for step in 0..8 {
                let emitted = world.broadphase();
                let n = world.bodies.len();
                let mut culled = 0usize;
                for i in 0..n {
                    for j in (i + 1)..n {
                        if world.bodies[i].inv_mass == 0.0 && world.bodies[j].inv_mass == 0.0 {
                            continue;
                        }
                        let key = canonical_pair(world.bodies.id_at(i), world.bodies.id_at(j));
                        if emitted.binary_search(&key).is_ok() {
                            continue;
                        }
                        culled += 1;
                        let contact = world.narrowphase.test(
                            &world.bodies[i],
                            &world.bodies[j],
                            &world.space,
                        );
                        assert!(
                            contact.is_none(),
                            "seed {seed} step {step}: the sweep culled {key:?}, which the \
                             narrowphase reports in contact"
                        );
                    }
                }
                assert!(
                    culled > 0,
                    "seed {seed} step {step}: nothing was culled, so this pass proved nothing"
                );
                world.step(1.0 / 240.0);
            }
        }
    }

    #[test]
    fn broadphase_emits_strictly_ascending_keys_under_disagreeing_storage_order_determinism() {
        for seed in PERMUTATION_SEEDS {
            let world = random_scene(seed, 40, 3.0);
            let disagrees = (1..world.bodies.len())
                .any(|dense| world.bodies.id_at(dense) < world.bodies.id_at(dense - 1));
            assert!(
                disagrees,
                "seed {seed}: storage order still agrees with handle order, so this \
                 scene cannot tell the two apart"
            );

            let pairs = world.broadphase();
            assert!(pairs.len() > 1, "seed {seed}: too few pairs to be ordered");
            assert!(
                pairs.windows(2).all(|w| w[0] < w[1]),
                "seed {seed}: emission order is not strictly ascending in BodyId"
            );
        }
    }

    #[test]
    fn broadphase_prunes_the_quadratic_pair_set_at_scale() {
        let world = random_scene(PERMUTATION_SEEDS[1], 200, 20.0);
        let n = world.bodies.len();
        assert!(
            n >= 100,
            "the scale case needs at least 100 bodies, got {n}"
        );
        let all_pairs = n * (n - 1) / 2;
        let emitted = world.broadphase().len();
        assert!(
            emitted * 10 < all_pairs,
            "the sweep emitted {emitted} of {all_pairs} pairs, which is no better than \
             a constant-factor cull"
        );
    }

    #[test]
    fn broadphase_fill_allocates_nothing_after_the_first_pass() {
        let world = random_scene(PERMUTATION_SEEDS[0], 120, 8.0);
        let mut intervals = Vec::new();
        let mut active = Vec::new();
        let mut pairs = Vec::new();

        for _ in 0..2 {
            World::fill_broadphase(
                &world.bodies,
                &world.space,
                &mut intervals,
                &mut active,
                &mut pairs,
            );
        }
        assert!(!pairs.is_empty(), "the fixture produced no pairs to emit");

        let bytes = alloc_probe::bytes_allocated_by(|| {
            for _ in 0..16 {
                World::fill_broadphase(
                    &world.bodies,
                    &world.space,
                    &mut intervals,
                    &mut active,
                    &mut pairs,
                );
            }
        });
        assert_eq!(
            bytes, 0,
            "16 sweeps over a steady body set asked the allocator for {bytes} bytes"
        );
    }

    #[test]
    fn exactly_tangent_spheres_are_a_candidate_pair() {
        const RADIUS: f32 = 0.5;
        let mut world = World::new(EuclideanR3);
        let anchor = world.push_body(sphere_body_r3(Vec3::ZERO, Vec3::ZERO, RADIUS, 1.0));
        let tangent = world.push_body(sphere_body_r3(Vec3::X, Vec3::ZERO, RADIUS, 1.0));
        // One ulp past tangency, so the assertion below pins the closed side of
        // the boundary rather than a widened one.
        let separated = world.push_body(sphere_body_r3(
            Vec3::new(-(1.0 + f32::EPSILON), 0.0, 0.0),
            Vec3::ZERO,
            RADIUS,
            1.0,
        ));

        let position = |id: BodyId| {
            let dense = world.bodies.dense_index(id).expect("nothing despawned");
            world.bodies[dense].position
        };
        assert_eq!(
            world.space.distance(position(anchor), position(tangent)),
            RADIUS + RADIUS,
            "the fixture is not exactly tangent, so it cannot state the boundary"
        );
        assert!(
            world.space.distance(position(anchor), position(separated)) > RADIUS + RADIUS,
            "the separated sphere is not past the boundary"
        );

        assert_eq!(world.broadphase(), vec![canonical_pair(anchor, tangent)]);
    }

    #[test]
    fn coincident_point_colliders_are_a_candidate_pair() {
        let mut world = World::new(EuclideanR3);
        let a = world.push_body(sphere_body_r3(Vec3::ZERO, Vec3::ZERO, 0.0, 1.0));
        let b = world.push_body(sphere_body_r3(Vec3::ZERO, Vec3::ZERO, 0.0, 1.0));
        assert_eq!(world.broadphase(), vec![canonical_pair(a, b)]);
    }

    #[test]
    fn bounding_radius_contains_every_posed_vertex_of_its_collider() {
        let half_extents = Vec3::new(0.5, 1.25, 0.25);
        let vertices = crate::euclidean_r3::box_vertices(half_extents);
        let radius = bounding_radius(&Collider::ConvexPolytope3D {
            vertices: vertices.clone(),
        });
        assert_eq!(radius, half_extents.length());

        let rotation = glam::Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalize(), 0.7);
        for v in &vertices {
            let posed = rotation * *v;
            assert!(
                posed.length() <= radius + 1e-6,
                "posed vertex {posed} escaped the bounding radius {radius}"
            );
        }

        assert_eq!(bounding_radius(&Collider::sphere_at_origin(0.75)), 0.75);
        assert_eq!(
            bounding_radius(&Collider::HalfSpace {
                normal: Vec3::Y,
                offset: 0.0
            }),
            f32::INFINITY,
            "a half-space is unbounded and must never be culled"
        );
    }

    const ISLAND_SETTLE_STEPS: usize = 400;
    const ISLAND_COLUMN_PITCH: f32 = 4.0;
    const ISLAND_COLUMNS: usize = 6;
    const ISLAND_COLUMN_HEIGHT: usize = 3;
    const ISLAND_PINNED_COLUMN: usize = 2;

    fn settled_columns(seed: u64) -> World<EuclideanR3> {
        const GAP: f32 = 0.05;
        const PINNED_TOUCH: f32 = 0.01;

        let mut rng = Xorshift::new(seed);
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        world.push_field(Box::new(Gravity::new(Vec3::new(0.0, GRAVITY_Y, 0.0))));
        world.push_body(halfspace_body_r3(Vec3::Y, 0.0));

        let mut columns: Vec<Vec<BodyId>> = Vec::with_capacity(ISLAND_COLUMNS);
        for column in 0..ISLAND_COLUMNS {
            let x = column as f32 * ISLAND_COLUMN_PITCH + rng.range(-0.5, 0.5);
            let radius = rng.range(0.3, 0.6);
            let mut ids = Vec::with_capacity(ISLAND_COLUMN_HEIGHT);
            for level in 0..ISLAND_COLUMN_HEIGHT {
                let pinned = column == ISLAND_PINNED_COLUMN && level == 1;
                let y = if pinned {
                    3.0 * radius - PINNED_TOUCH
                } else {
                    radius + GAP + level as f32 * (2.0 * radius + GAP)
                };
                let id = world.push_body(sphere_body_r3(
                    Vec3::new(x, y, 0.0),
                    Vec3::ZERO,
                    radius,
                    1.0,
                ));
                world.bodies[id].restitution = 0.0;
                if pinned {
                    world.bodies[id].mass = 0.0;
                    world.bodies[id].inv_mass = 0.0;
                }
                ids.push(id);
            }
            columns.push(ids);
        }

        for column in columns.iter().take(2) {
            assert!(world.despawn_body(column[0]));
        }
        for _ in 0..ISLAND_SETTLE_STEPS {
            world.step(1.0 / 240.0);
        }
        world
    }

    fn labels_for(world: &World<EuclideanR3>, keys: &[PairKey]) -> Vec<BodyId> {
        let mut parent = Vec::new();
        let mut labels = Vec::new();
        World::fill_islands(
            &world.bodies,
            keys.iter().copied(),
            &mut parent,
            &mut labels,
        );
        labels
    }

    fn synthetic_island_bodies() -> (World<EuclideanR3>, Vec<BodyId>) {
        const SPAWNS: usize = 14;
        const DESPAWNS: usize = 2;

        let mut world = World::new(EuclideanR3);
        let spawned: Vec<BodyId> = (0..SPAWNS)
            .map(|i| world.push_body(island_sphere(i as f32 * ISLAND_COLUMN_PITCH)))
            .collect();
        let survivors = spawned[DESPAWNS..].to_vec();
        for &position in &SYNTHETIC_STATICS {
            let id = survivors[position];
            world.bodies[id].mass = 0.0;
            world.bodies[id].inv_mass = 0.0;
        }
        for &doomed in &spawned[..DESPAWNS] {
            assert!(world.despawn_body(doomed));
        }
        (world, survivors)
    }

    const SYNTHETIC_STATICS: [usize; 2] = [1, 7];

    const SYNTHETIC_EDGES: [(usize, usize); 11] = [
        (0, 2),
        (0, 4),
        (0, 5),
        (2, 5),
        (6, 8),
        (8, 10),
        (10, 11),
        (1, 3),
        (1, 6),
        (7, 9),
        (7, 11),
    ];

    const SYNTHETIC_COMPONENTS: [&[usize]; 2] = [&[0, 2, 4, 5], &[6, 8, 10, 11]];

    #[test]
    fn island_labels_are_the_component_minimum_whatever_order_pairs_arrive_in_determinism() {
        let (world, ids) = synthetic_island_bodies();
        let canonical_keys: Vec<PairKey> = SYNTHETIC_EDGES
            .iter()
            .map(|&(a, b)| canonical_pair(ids[a], ids[b]))
            .collect();
        let canonical = labels_for(&world, &canonical_keys);

        for members in SYNTHETIC_COMPONENTS {
            let expected = members
                .iter()
                .map(|&i| ids[i])
                .min()
                .expect("a component with no members");
            for &member in members {
                let dense = world.bodies.dense_index(ids[member]).unwrap();
                assert_eq!(
                    canonical[dense], expected,
                    "body {member} is labelled {:?}, not its component's lowest handle",
                    canonical[dense]
                );
            }
        }
        let grouped: BTreeSet<usize> = SYNTHETIC_COMPONENTS
            .iter()
            .flat_map(|m| *m)
            .copied()
            .collect();
        for (position, &id) in ids.iter().enumerate() {
            if grouped.contains(&position) {
                continue;
            }
            let dense = world.bodies.dense_index(id).unwrap();
            assert_eq!(
                canonical[dense], id,
                "body {position} joined an island it has no edge into"
            );
        }

        for order in order_variants(SchedulePhase::Constraint) {
            let mut keys = canonical_keys.clone();
            order.apply(SchedulePhase::Constraint, &mut keys);
            assert_ne!(keys, canonical_keys, "{order:?} is the identity");
            assert_eq!(
                labels_for(&world, &keys),
                canonical,
                "{order:?} produced a different island assignment"
            );
        }
    }

    #[test]
    fn island_ids_are_the_lowest_body_id_not_the_lowest_storage_position_determinism() {
        let mut orders_disagreed = 0usize;
        for seed in PERMUTATION_SEEDS {
            let mut world = settled_columns(seed);
            for step in 0..8 {
                world.step(1.0 / 240.0);
                let islands = world.islands();
                for island in &islands {
                    let lowest_handle = island.bodies.iter().copied().min();
                    assert_eq!(
                        Some(island.id),
                        lowest_handle,
                        "seed {seed} step {step}: island {:?} is not named by its \
                         lowest handle",
                        island.id
                    );
                    let lowest_stored =
                        island.bodies.iter().copied().min_by_key(|&id| {
                            world.bodies.dense_index(id).expect(STALE_MANIFOLD_BODY)
                        });
                    if lowest_stored != lowest_handle {
                        orders_disagreed += 1;
                    }
                }
                assert!(
                    islands.windows(2).all(|w| w[0].id < w[1].id),
                    "seed {seed} step {step}: islands are not strictly ascending in id"
                );
            }
        }
        assert!(
            orders_disagreed > 0,
            "no island ever held a body whose handle order disagreed with its \
             storage order, so the two labelling rules were never told apart"
        );
    }

    #[test]
    fn a_static_body_joins_no_island_and_merges_none_determinism() {
        for seed in PERMUTATION_SEEDS {
            let world = settled_columns(seed);
            let pinned = world
                .bodies
                .iter()
                .position(|body| {
                    body.inv_mass == 0.0 && matches!(body.collider, Collider::Sphere { .. })
                })
                .map(|dense| world.bodies.id_at(dense))
                .expect("the fixture lost its static sphere");

            let neighbours: Vec<BodyId> = world
                .manifolds
                .keys()
                .filter_map(|&(a, b)| match (a == pinned, b == pinned) {
                    (true, false) => Some(b),
                    (false, true) => Some(a),
                    _ => None,
                })
                .collect();
            assert_eq!(
                neighbours.len(),
                2,
                "seed {seed}: the static sphere touches {} bodies, so it is \
                 not wedged between two",
                neighbours.len()
            );

            let islands = world.islands();
            let island_of = |id: BodyId| {
                islands
                    .iter()
                    .find(|island| island.bodies.contains(&id))
                    .map(|island| island.id)
            };
            assert!(
                island_of(pinned).is_none(),
                "seed {seed}: a static body joined an island"
            );
            assert_ne!(
                island_of(neighbours[0]),
                island_of(neighbours[1]),
                "seed {seed}: two bodies that meet only through a static one \
                 were merged into one island"
            );
        }
    }

    fn flood_fill_islands(world: &World<EuclideanR3>) -> Vec<Island> {
        let dynamic = |id: BodyId| world.bodies[id].inv_mass != 0.0;
        let mut adjacency: BTreeMap<BodyId, Vec<BodyId>> = BTreeMap::new();
        for &(a, b) in world.manifolds.keys() {
            for id in [a, b].into_iter().filter(|&id| dynamic(id)) {
                adjacency.entry(id).or_default();
            }
            if dynamic(a) && dynamic(b) {
                adjacency.entry(a).or_default().push(b);
                adjacency.entry(b).or_default().push(a);
            }
        }

        let mut seen: BTreeSet<BodyId> = BTreeSet::new();
        let mut islands = Vec::new();
        for &seed in adjacency.keys() {
            if !seen.insert(seed) {
                continue;
            }
            let mut bodies = vec![seed];
            let mut frontier = vec![seed];
            while let Some(body) = frontier.pop() {
                for &next in &adjacency[&body] {
                    if seen.insert(next) {
                        bodies.push(next);
                        frontier.push(next);
                    }
                }
            }
            bodies.sort_unstable();
            let constraints = world
                .manifolds
                .keys()
                .copied()
                .filter(|&(a, b)| {
                    bodies.binary_search(&a).is_ok() || bodies.binary_search(&b).is_ok()
                })
                .collect();
            islands.push(Island {
                id: bodies[0],
                bodies,
                constraints,
            });
        }
        islands.sort_unstable_by_key(|island| island.id);
        islands
    }

    #[test]
    fn islands_match_a_flood_fill_of_the_contact_graph_determinism() {
        let mut ever_multi_body = false;
        for seed in PERMUTATION_SEEDS {
            let mut world = settled_columns(seed);
            for step in 0..8 {
                world.step(1.0 / 240.0);
                let islands = world.islands();
                ever_multi_body |= islands.iter().any(|island| island.bodies.len() > 1);
                assert_eq!(
                    islands,
                    flood_fill_islands(&world),
                    "seed {seed} step {step}: union-find disagreed with the flood fill"
                );
            }
        }
        assert!(
            ever_multi_body,
            "no island ever held two bodies, so the comparison never covered a union"
        );
    }

    #[test]
    fn a_single_island_solves_in_the_global_ascending_key_order_determinism() {
        let world = settled_sphere_stack(1.0 / 240.0, 200);
        let islands = world.islands();
        assert_eq!(
            islands.len(),
            1,
            "the stack is not one island, so this fixture cannot state the \
             single-island case"
        );

        let ascending: Vec<PairKey> = world.manifolds.keys().copied().collect();
        assert!(ascending.len() > 1, "too few constraints to be ordered");
        assert_eq!(
            constraint_order(&world),
            ascending,
            "grouping moved a constraint in a world with a single island"
        );
        assert_eq!(islands[0].constraints, ascending);
        assert_eq!(
            islands[0].bodies.len(),
            3,
            "the island should hold the three spheres and not the static floor"
        );
    }

    #[test]
    fn constraint_buffer_runs_island_by_island_determinism() {
        let mut world = multi_island_world(Schedule::default());
        for _ in 0..MULTI_ISLAND_STEPS {
            world.step(MULTI_ISLAND_DT);
        }

        let islands = world.islands();
        assert_eq!(
            islands.len(),
            3,
            "the groups share only the static floor, so they are three islands"
        );
        let grouped: Vec<PairKey> = islands
            .iter()
            .flat_map(|island| island.constraints.iter().copied())
            .collect();
        assert_eq!(
            constraint_order(&world),
            grouped,
            "the solved buffer is not the islands in order"
        );

        let ascending: Vec<PairKey> = world.manifolds.keys().copied().collect();
        assert_ne!(
            grouped, ascending,
            "the fixture's islands happen to be contiguous in ascending key \
             order, so it cannot show that grouping reorders anything"
        );
    }

    #[test]
    fn island_grouping_allocates_nothing_after_the_first_pass() {
        let mut world = settled_columns(PERMUTATION_SEEDS[0]);
        for _ in 0..2 {
            world.collect_constraints();
        }
        assert!(world.constraints.len() > 1);

        let bytes = alloc_probe::bytes_allocated_by(|| {
            for _ in 0..16 {
                world.collect_constraints();
            }
        });
        assert_eq!(
            bytes, 0,
            "16 island passes over a steady contact set asked the allocator for \
             {bytes} bytes"
        );
    }

    #[test]
    fn manifold_update_allocates_nothing_after_the_first_pass() {
        let mut world = settled_columns(PERMUTATION_SEEDS[0]);
        for _ in 0..2 {
            world.update_manifolds();
        }
        assert!(
            world.manifolds.len() > 1,
            "the fixture holds too few contacts to exercise the eviction pass"
        );

        let bytes = alloc_probe::bytes_allocated_by(|| {
            for _ in 0..16 {
                world.update_manifolds();
            }
        });
        assert_eq!(
            bytes, 0,
            "16 manifold passes over a steady contact set asked the allocator \
             for {bytes} bytes"
        );
    }

    const MEASUREMENT_BODY_COUNTS: [usize; 3] = [100, 200, 400];
    /// Half-width of the spawn box at 100 bodies; scaled as the cube root of
    /// the count to hold density.
    const MEASUREMENT_SPREAD: f32 = 6.0;
    const MEASUREMENT_SETTLE_STEPS: usize = 240;
    const MEASUREMENT_REPS: u32 = 200;

    fn mean_nanos(mut body: impl FnMut()) -> f64 {
        let start = std::time::Instant::now();
        for _ in 0..MEASUREMENT_REPS {
            body();
        }
        start.elapsed().as_nanos() as f64 / f64::from(MEASUREMENT_REPS)
    }

    fn settled_sphere_scene(seed: u64, count: usize) -> World<EuclideanR3> {
        let spread = MEASUREMENT_SPREAD * (count as f32 / 100.0).cbrt();
        let mut rng = Xorshift::new(seed);
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        world.push_field(Box::new(Gravity::new(Vec3::new(0.0, GRAVITY_Y, 0.0))));
        world.push_body(halfspace_body_r3(Vec3::Y, 0.0));

        for _ in 0..count {
            let position = Vec3::new(
                rng.range(-spread, spread),
                rng.range(0.5, spread + 0.5),
                rng.range(-spread, spread),
            );
            let id = world.push_body(sphere_body_r3(
                position,
                Vec3::ZERO,
                rng.range(0.2, 0.8),
                1.0,
            ));
            world.bodies[id].restitution = 0.0;
        }
        for _ in 0..MEASUREMENT_SETTLE_STEPS {
            world.step(1.0 / 240.0);
        }
        world
    }

    #[test]
    #[ignore = "measurement; run with --release -- --ignored --nocapture"]
    fn step_phase_cost_measurement() {
        println!("bodies pairs constraints sweep_ns scan_ns grouped_ns ungrouped_ns");
        for count in MEASUREMENT_BODY_COUNTS {
            let mut world = settled_sphere_scene(PERMUTATION_SEEDS[0], count);
            world.collect_constraints();
            let bodies = world.bodies.len();
            let constraints = world.constraints.len();

            let mut intervals = Vec::new();
            let mut active = Vec::new();
            let mut pairs = Vec::new();
            let mut radii = Vec::new();
            let mut scanned = Vec::new();
            for _ in 0..2 {
                World::fill_broadphase(
                    &world.bodies,
                    &world.space,
                    &mut intervals,
                    &mut active,
                    &mut pairs,
                );
                scan_broadphase(&world, &mut radii, &mut scanned);
            }
            assert_eq!(pairs, scanned, "the sweep and the scan disagree");

            let sweep_ns = mean_nanos(|| {
                World::fill_broadphase(
                    &world.bodies,
                    &world.space,
                    &mut intervals,
                    &mut active,
                    &mut pairs,
                );
            });
            let scan_ns = mean_nanos(|| {
                scan_broadphase(&world, &mut radii, &mut scanned);
            });

            let grouped_ns = mean_nanos(|| {
                world.collect_constraints();
            });
            let mut ungrouped: Vec<PairKey> = Vec::with_capacity(constraints);
            let ungrouped_ns = mean_nanos(|| {
                ungrouped.clear();
                ungrouped.extend(world.manifolds.keys().copied());
            });

            let emitted = pairs.len();
            print!("{bodies:6} {emitted:5} {constraints:11}");
            println!(" {sweep_ns:8.0} {scan_ns:7.0} {grouped_ns:10.0} {ungrouped_ns:12.0}");
        }
    }

    fn scan_broadphase(world: &World<EuclideanR3>, radii: &mut Vec<f32>, pairs: &mut Vec<PairKey>) {
        radii.clear();
        radii.extend(world.bodies.iter().map(|b| bounding_radius(&b.collider)));
        pairs.clear();
        let n = world.bodies.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let (a, b) = (&world.bodies[i], &world.bodies[j]);
                if a.inv_mass == 0.0 && b.inv_mass == 0.0 {
                    continue;
                }
                if world.space.distance(a.position, b.position) <= radii[i] + radii[j] {
                    pairs.push(canonical_pair(world.bodies.id_at(i), world.bodies.id_at(j)));
                }
            }
        }
        pairs.sort_unstable();
    }

    /// The step tests a body only where the integrator left it, so a wall is
    /// seen only when a sample lands inside it, and only a sample on the near
    /// side of the slab's midplane escapes back the way the body came: past
    /// the midplane the minimum-translation vector points onward. The bound is
    /// therefore geometric, `wall_half_thickness + body_radius` for a wall hit
    /// head-on, and no impulse magnitude, iteration count, or Baumgarte term
    /// moves it. Only a swept test or a speculative contact does.
    ///
    /// Options, in ascending cost, for when a caller needs more reach than the
    /// recorded bound:
    ///
    /// - Substep the fast body alone: integrate it in `ceil(|v|·dt / bound)`
    ///   slices and run the narrowphase per slice.
    /// - Speculative contacts: widen the broadphase interval by the sweep
    ///   extent and emit a contact with negative penetration, letting the
    ///   existing PGS normal row stop the body before it arrives (Catto 2013,
    ///   GDC, "Continuous Collision").
    /// - Conservative advancement or a swept narrowphase: exact time of
    ///   impact, per pair, and a step that can end early (Redon, Kheddar,
    ///   Coquillart 2002, Eurographics 21(3), sec. 4).
    mod tunneling {
        use glam::{Vec2, Vec3, Vec4};
        use loam_math::{EuclideanR2, EuclideanR3, EuclideanR4};

        use crate::body::RigidBody;
        use crate::collider::Collider;
        use crate::euclidean_r2::{sphere_body, static_wall};
        use crate::euclidean_r3::{box_vertices, sphere_body_r3};
        use crate::euclidean_r4::sphere_body_r4;
        use crate::world::World;

        const DT: f32 = 1.0 / 240.0;
        const WALL_HALF_THICKNESS: f32 = 0.05;
        /// Wide enough that a projectile fired down the launch axis meets a
        /// face, never an edge.
        const WALL_HALF_SPAN: f32 = 2.0;
        const PROJECTILE_RADIUS: f32 = 0.1;
        const APPROACH: f32 = 0.4;
        const OVERSHOOT: f32 = 0.4;
        const PHASES: u32 = 64;
        const SCAN_MIN: f32 = 0.01;
        const SCAN_STEP: f32 = 0.0025;
        const SCAN_MAX: f32 = 2.0 * (WALL_HALF_THICKNESS + PROJECTILE_RADIUS);

        /// Width of the interval of body positions where the wall is both
        /// overlapped and still escapable backwards.
        const GEOMETRIC_BOUND: f32 = WALL_HALF_THICKNESS + PROJECTILE_RADIUS;

        /// Each constant is a FLOOR, not a two-sided pin. The assertion below
        /// fires when a space resolves LESS than its recorded reach, which is a
        /// regression, and says nothing when it resolves more, which is not: a
        /// scan that finds more reach should raise the constant, not fail.
        const RECORDED_R2: f32 = 0.150;
        const RECORDED_R3: f32 = 0.150;
        const RECORDED_R4: f32 = 0.150;

        /// The R⁴ depth whose EPA normal used to point through the wall, in
        /// launch-axis coordinates with the slab spanning `|x| ≤ 0.05`. Kept
        /// as the fixture for the regression pin: a normal that inverts again
        /// is likeliest to do it at the depth that already caught it once.
        const R4_TRAP_DEPTH: f32 = -0.090;

        fn slab_vertices_r4(half: Vec4) -> Vec<Vec4> {
            let mut vertices = Vec::with_capacity(16);
            for &x in &[-half.x, half.x] {
                for &y in &[-half.y, half.y] {
                    for &z in &[-half.z, half.z] {
                        for &w in &[-half.w, half.w] {
                            vertices.push(Vec4::new(x, y, z, w));
                        }
                    }
                }
            }
            vertices
        }

        /// Steps to fly `APPROACH + phase + OVERSHOOT` at `displacement` per
        /// step, plus one so the last sample is unambiguously past the wall.
        fn flight_steps(displacement: f32, phase: f32) -> usize {
            ((APPROACH + phase + OVERSHOOT) / displacement).ceil() as usize + 1
        }

        fn wall_world_r4() -> World<EuclideanR4> {
            let mut world = World::new(EuclideanR4);
            crate::euclidean_r4::register_default_narrowphase(&mut world.narrowphase);
            world.push_body(RigidBody::fixed(
                Vec4::ZERO,
                Collider::ConvexPolytope4D {
                    vertices: slab_vertices_r4(Vec4::new(
                        WALL_HALF_THICKNESS,
                        WALL_HALF_SPAN,
                        WALL_HALF_SPAN,
                        WALL_HALF_SPAN,
                    )),
                },
                1.0,
                &EuclideanR4,
            ));
            world
        }

        /// Final launch-axis coordinate of a projectile fired along +x at a
        /// static wall spanning `|x| ≤ WALL_HALF_THICKNESS`. Negative means the
        /// wall held; positive means the body is through it.
        fn fire_r2(displacement: f32, phase: f32) -> f32 {
            let mut world = World::new(EuclideanR2);
            crate::euclidean_r2::register_default_narrowphase(&mut world.narrowphase);
            world.push_body(static_wall(
                Vec2::ZERO,
                Vec2::new(WALL_HALF_THICKNESS, WALL_HALF_SPAN),
            ));
            let ball = world.push_body(sphere_body(
                Vec2::new(-(APPROACH + phase), 0.0),
                Vec2::new(displacement / DT, 0.0),
                PROJECTILE_RADIUS,
                1.0,
            ));
            for _ in 0..flight_steps(displacement, phase) {
                world.step(DT);
            }
            world.bodies[ball].position.x
        }

        fn fire_r3(displacement: f32, phase: f32) -> f32 {
            let mut world = World::new(EuclideanR3);
            crate::euclidean_r3::register_default_narrowphase(&mut world.narrowphase);
            world.push_body(RigidBody::fixed(
                Vec3::ZERO,
                Collider::ConvexPolytope3D {
                    vertices: box_vertices(Vec3::new(
                        WALL_HALF_THICKNESS,
                        WALL_HALF_SPAN,
                        WALL_HALF_SPAN,
                    )),
                },
                1.0,
                &EuclideanR3,
            ));
            let ball = world.push_body(sphere_body_r3(
                Vec3::new(-(APPROACH + phase), 0.0, 0.0),
                Vec3::new(displacement / DT, 0.0, 0.0),
                PROJECTILE_RADIUS,
                1.0,
            ));
            for _ in 0..flight_steps(displacement, phase) {
                world.step(DT);
            }
            world.bodies[ball].position.x
        }

        fn fire_r4(displacement: f32, phase: f32) -> f32 {
            let mut world = wall_world_r4();
            let ball = world.push_body(sphere_body_r4(
                Vec4::new(-(APPROACH + phase), 0.0, 0.0, 0.0),
                Vec4::new(displacement / DT, 0.0, 0.0, 0.0),
                PROJECTILE_RADIUS,
                1.0,
            ));
            for _ in 0..flight_steps(displacement, phase) {
                world.step(DT);
            }
            world.bodies[ball].position.x
        }

        /// Launch-axis velocity one step gives a body released at rest at `x`,
        /// already overlapping the slab. Negative is the way it came.
        fn released_at_rest_r4(x: f32) -> f32 {
            let mut world = wall_world_r4();
            let ball = world.push_body(sphere_body_r4(
                Vec4::new(x, 0.0, 0.0, 0.0),
                Vec4::ZERO,
                PROJECTILE_RADIUS,
                1.0,
            ));
            world.step(DT);
            world.bodies[ball].velocity.x
        }

        fn wall_holds(fire: impl Fn(f32, f32) -> f32, displacement: f32) -> bool {
            (0..PHASES).all(|k| {
                let phase = displacement * k as f32 / PHASES as f32;
                fire(displacement, phase) < 0.0
            })
        }

        /// Largest ladder displacement at which every sampled launch alignment
        /// still resolves, scanning upward and stopping at the first failure,
        /// so the answer has nothing tunneling under it at the sampled
        /// alignments rather than being an isolated success above a failure.
        fn max_resolved_displacement(fire: impl Fn(f32, f32) -> f32) -> f32 {
            let mut resolved = 0.0;
            for k in 0.. {
                let displacement = SCAN_MIN + SCAN_STEP * k as f32;
                if displacement > SCAN_MAX || !wall_holds(&fire, displacement) {
                    break;
                }
                resolved = displacement;
            }
            resolved
        }

        fn assert_tunneling_bound(space: &str, recorded: f32, fire: impl Fn(f32, f32) -> f32) {
            let measured = max_resolved_displacement(fire);
            println!(
                "{space}: resolves up to {measured} per step ({} m/s at {} Hz), \
                 scanned at {SCAN_STEP} over {PHASES} launch alignments",
                measured / DT,
                1.0 / DT
            );
            assert!(
                measured > recorded - SCAN_STEP,
                "{space} resolves only {measured} per step against the recorded \
                 {recorded}: the safe throw ceiling dropped"
            );
        }

        #[test]
        fn thin_wall_holds_only_below_a_recorded_per_step_displacement_r2() {
            assert_tunneling_bound("R2", RECORDED_R2, fire_r2);
        }

        #[test]
        fn thin_wall_holds_only_below_a_recorded_per_step_displacement_r3() {
            assert_tunneling_bound("R3", RECORDED_R3, fire_r3);
        }

        #[test]
        fn thin_wall_holds_only_below_a_recorded_per_step_displacement_r4() {
            assert_tunneling_bound("R4", RECORDED_R4, fire_r4);
        }

        #[test]
        fn resolving_interval_is_the_slab_half_thickness_plus_the_body_radius() {
            let gap = (RECORDED_R3 - GEOMETRIC_BOUND).abs();
            assert!(
                gap <= SCAN_STEP,
                "the recorded R3 bound {RECORDED_R3} is {gap} off the geometric \
                 {GEOMETRIC_BOUND}, so it is no longer the sampling gap it is \
                 documented as"
            );
        }

        #[test]
        fn r4_contact_normal_leaves_through_the_near_face_at_every_depth() {
            // Three depths through the hole that used to invert: R4's EPA
            // reported a normal pointing through the wall here and drove the
            // body toward the far face. A negative exit is the near face,
            // the one the ball entered.
            for depth in [R4_TRAP_DEPTH - 0.002, R4_TRAP_DEPTH, R4_TRAP_DEPTH + 0.002] {
                let left = released_at_rest_r4(depth);
                assert!(left < 0.0, "R4 drove a body at {depth} toward the FAR face, leaving at {left}: the contact normal points through the wall again");
            }
        }

        #[test]
        fn thin_wall_is_transparent_to_a_body_that_steps_clear_over_it() {
            let displacement = 4.0 * GEOMETRIC_BOUND;
            for (space, fired) in [
                ("R2", fire_r2(displacement, 0.0)),
                ("R3", fire_r3(displacement, 0.0)),
                ("R4", fire_r4(displacement, 0.0)),
            ] {
                assert!(
                    fired > GEOMETRIC_BOUND,
                    "{space} stopped a body stepping {displacement} clear over a \
                     {GEOMETRIC_BOUND} resolving interval, which position \
                     sampling alone cannot do: it ended at {fired}"
                );
            }
        }
    }
}
