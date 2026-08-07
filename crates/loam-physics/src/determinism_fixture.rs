//! The determinism scenarios and one committed golden constant per scenario,
//! shared by every determinism gate in the crate. Callers drive
//! [`World::step`] through this module rather than re-implementing the phase
//! loop, so a schedule variant is always compared against the simulation the
//! golden hash pins.
//!
//! The replay fixture below is the same idea run against a recorded input
//! stream: it drives the scenario from a [`Tape`] instead of from silence, so
//! the pin covers the inputs as well as the integrator.

use std::ops::Range;

use glam::{Vec3, Vec4};
use loam_math::{EuclideanR3, EuclideanR4};
use loam_time::{Checkpoint, StateHash, Tape};

use crate::body::{BodyId, RigidBody};
use crate::collision::VectorOps;
use crate::euclidean_r3::{
    halfspace_body_r3, register_default_narrowphase as register_narrowphase_r3, sphere_body_r3,
};
use crate::euclidean_r4::{
    halfspace4_body_r4, register_default_narrowphase as register_narrowphase_r4, sphere_body_r4,
};
use crate::field::Gravity;
use crate::integrator::PhysicsSpace;
use crate::world::{Schedule, World};

pub fn fnv1a64(words: &[u32]) -> u64 {
    let mut hash = StateHash::new();
    hash.write_u32s(words);
    hash.finish()
}

/// FNV-1a of [`ScenarioRun::trajectory`] under the default schedule, recorded
/// on x86_64. This pins behavior rather than self-consistency: a
/// deterministic-but-changed integrator, solver order, contact constant, or
/// narrowphase moves it. Scoped to one architecture family, since glam's SIMD
/// dot reduces in a different order than its scalar fallback.
///
/// Re-recording: the scenario is chaotic, so the hash alone cannot say whether
/// a mismatch is an intended simulation change or a solver regression.
/// [`assert_scenario_stays_physical`] is what separates them, and it runs
/// first. If it passed and only the hash moved, the change is intended:
/// replace this constant with the value the hash assertion prints. If it
/// failed, the scenario stopped being physical and no re-recorded hash is
/// correct.
pub const GOLDEN_TRAJECTORY_HASH: u64 = 0xfcfa_9165_cc85_e57b;

pub struct ScenarioRun {
    /// Every step, every body, linear and angular state as raw f32 bits, so
    /// the pins see the path and not just the endpoint.
    pub trajectory: Vec<u32>,
    /// Running hash after each step over a superset of `trajectory`: the same
    /// body words plus the manifold key list, each manifold's point count, and
    /// each point's accumulated normal impulse. Warm-start impulses are
    /// carried state, so a schedule change can leave body state identical for
    /// one step and diverge several steps later; hashing bodies alone would
    /// find that late or not at all. A vector rather than a scalar because the
    /// first differing index is the first divergent step, which is the whole
    /// triage story for a hash mismatch.
    pub step_hashes: Vec<u64>,
    /// The physical readings the golden hash cannot supply on its own.
    pub envelope: PhysicalEnvelope,
}

/// Extremes of the two quantities [`assert_scenario_stays_physical`] pins,
/// with the step each was reached on so a failure names when the scenario
/// stopped being physical rather than only that it did.
pub struct PhysicalEnvelope {
    /// Mechanical energy of the pre-step configuration. Never sampled into
    /// [`ScenarioRun::trajectory`], so recording it cannot move a golden hash.
    pub initial_energy: f32,
    pub peak_energy: f32,
    pub peak_energy_step: usize,
    /// Over dynamic bodies only: the static half-space sits at the floor
    /// height by construction and would pin the minimum there.
    pub lowest_height: f32,
    pub lowest_height_step: usize,
}

const STEPS: usize = 240;

/// Both fixtures fall along -y at this magnitude onto a static half-space at
/// `y = 0`, so heights and potential energies are read off the y component
/// directly. Named here rather than at the two `Gravity::new` calls so the
/// energy budget cannot drift from the field the scenarios install.
const GRAVITY_MAGNITUDE: f32 = 9.8;
const UP_COMPONENT: usize = 1;

/// Sampled words for one body: `dims` position, `dims` velocity, and the
/// `C(dims, 2)` bivector components. Both samplers write this layout, and
/// [`run_scenario`] asserts the sampled width agrees before reading any
/// component out by index.
const fn words_per_body(dims: usize) -> usize {
    2 * dims + dims * (dims - 1) / 2
}

/// Translational plus rotational plus gravitational-potential energy of one
/// sampled body. Every space here uses a scalar isotropic moment, so the
/// rotational term is `½·I·|ω|²` with `|ω|` the Euclidean norm of the
/// bivector components.
fn mechanical_energy(words: &[u32], dims: usize, mass: f32, inertia: f32) -> f32 {
    let read = |i: usize| f32::from_bits(words[i]);
    let sum_squares = |range: Range<usize>| range.map(|i| read(i) * read(i)).sum::<f32>();
    0.5 * mass * sum_squares(dims..2 * dims)
        + 0.5 * inertia * sum_squares(2 * dims..words.len())
        + mass * GRAVITY_MAGNITUDE * read(UP_COMPONENT)
}

/// Total mechanical energy of the dynamic bodies in one sampled configuration.
/// Statics are skipped rather than contributing zero: their sampled height is
/// whatever the constructor gave them, not their plane's.
fn configuration_energy<S>(world: &World<S>, words: &[u32], dims: usize) -> f32
where
    S: PhysicsSpace<Inertia = f32>,
{
    world
        .bodies
        .iter()
        .zip(words.chunks_exact(words_per_body(dims)))
        .filter(|(body, _)| body.inv_mass != 0.0)
        .map(|(body, sampled)| mechanical_energy(sampled, dims, body.mass, body.inertia))
        .sum()
}

/// The physical envelope a golden-hash fixture must stay inside for a hash
/// mismatch to be readable as an intended change. Neither limit is a recorded
/// measurement: both follow from the scenario, so tripping one says the
/// simulation stopped being physical rather than that it moved.
///
/// Floor: no dynamic body's centre ever reaches the plane. A sphere centre at
/// or below `y = 0` is more than half-buried, which is tunnelling and not a
/// settling depth; the depth the solver actually targets is
/// `PENETRATION_SLOP`, a small fraction of either fixture's radius.
///
/// Energy: the scenario has no source. Semi-implicit Euler in a uniform field
/// loses `½·|g|²·dt²` per unit mass per step (velocity is advanced before
/// position, so the position update uses the post-gravity velocity), the
/// default restitution is 0.2 and is suppressed entirely below
/// `RESTITUTION_THRESHOLD`, and Coulomb friction only opposes sliding. The
/// Baumgarte positional bias is the one term that can add energy, and it must
/// not add enough to beat the starting configuration.
pub fn assert_scenario_stays_physical(run: &ScenarioRun) {
    let envelope = &run.envelope;
    assert!(
        envelope.lowest_height > 0.0,
        "a dynamic body reached y = {} at step {}, at or through the floor \
         plane at y = 0: the scenario is no longer physical, so a golden-hash \
         mismatch is a solver regression and not an intended change",
        envelope.lowest_height,
        envelope.lowest_height_step
    );
    assert!(
        envelope.peak_energy <= envelope.initial_energy,
        "mechanical energy reached {} at step {}, above the initial \
         configuration's {}: the solve is creating energy, so a golden-hash \
         mismatch is a solver regression and not an intended change",
        envelope.peak_energy,
        envelope.peak_energy_step,
        envelope.initial_energy
    );
}

/// Drive `world` and sample it. Every fixture routes through here so all of
/// them hash the same quantities in the same order, and so none of them can
/// drift into re-implementing the phase loop instead of driving
/// [`World::step`].
fn run_scenario<S, F>(
    world: &mut World<S>,
    dt: f32,
    steps: usize,
    dims: usize,
    sample: F,
) -> ScenarioRun
where
    S: PhysicsSpace<Inertia = f32>,
    S::Vector: VectorOps,
    S::Point: Copy + std::ops::Sub<Output = S::Vector>,
    F: Fn(&RigidBody<S>, &mut Vec<u32>),
{
    let words = words_per_body(dims);
    let per_step = world.bodies.len() * words;

    // Sampled through the same closure as every hashed step, but into its own
    // buffer, so the energy budget is the configuration the scenario starts
    // from rather than the first step's already-fallen state.
    let mut initial = Vec::with_capacity(per_step);
    for body in world.bodies.iter() {
        sample(body, &mut initial);
    }
    assert_eq!(
        initial.len(),
        per_step,
        "sampler layout disagrees with words_per_body({dims}), so the physical \
         envelope would read the wrong components"
    );
    let initial_energy = configuration_energy(world, &initial, dims);

    let mut trajectory = Vec::with_capacity(steps * per_step);
    let mut step_hashes = Vec::with_capacity(steps);
    let mut peak_energy = f32::NEG_INFINITY;
    let mut peak_energy_step = 0;
    let mut lowest_height = f32::INFINITY;
    let mut lowest_height_step = 0;
    let mut hash = StateHash::new();
    for step in 0..steps {
        world.step(dt);
        let step_start = trajectory.len();
        for body in world.bodies.iter() {
            sample(body, &mut trajectory);
        }
        let energy = configuration_energy(world, &trajectory[step_start..], dims);
        if energy > peak_energy {
            peak_energy = energy;
            peak_energy_step = step;
        }
        for (body, sampled) in world
            .bodies
            .iter()
            .zip(trajectory[step_start..].chunks_exact(words))
        {
            let height = f32::from_bits(sampled[UP_COMPONENT]);
            if body.inv_mass != 0.0 && height < lowest_height {
                lowest_height = height;
                lowest_height_step = step;
            }
        }

        hash.write_u32s(&trajectory[step_start..]);
        // `BTreeMap` key order under every schedule, so a moved hash means the
        // simulation diverged and never that the instrument read the same
        // state in a different order.
        world.hash_contacts(&mut hash);
        step_hashes.push(hash.finish());
    }
    ScenarioRun {
        trajectory,
        step_hashes,
        envelope: PhysicalEnvelope {
            initial_energy,
            peak_energy,
            peak_energy_step,
            lowest_height,
            lowest_height_step,
        },
    }
}

/// Orientation is deliberately not sampled by either sampler. `Bivector::exp`
/// routes through libm `sin`/`cos`, whose last-ULP results differ between
/// platform libms, and for sphere colliders orientation never feeds back into
/// the dynamics. Every sampled quantity comes from +, -, *, / and sqrt, which
/// IEEE-754 rounds exactly, so a trajectory is reproducible wherever glam takes
/// the same reduction path.
fn sample_body_r4(body: &RigidBody<EuclideanR4>, words: &mut Vec<u32>) {
    let p = body.position;
    let v = body.velocity;
    let w = body.angular_velocity;
    words.extend_from_slice(&[
        p.x.to_bits(),
        p.y.to_bits(),
        p.z.to_bits(),
        p.w.to_bits(),
        v.x.to_bits(),
        v.y.to_bits(),
        v.z.to_bits(),
        v.w.to_bits(),
        w.xy.to_bits(),
        w.xz.to_bits(),
        w.xw.to_bits(),
        w.yz.to_bits(),
        w.yw.to_bits(),
        w.zw.to_bits(),
    ]);
}

pub fn sample_body_r3(body: &RigidBody<EuclideanR3>, words: &mut Vec<u32>) {
    let p = body.position;
    let v = body.velocity;
    let w = body.angular_velocity;
    words.extend_from_slice(&[
        p.x.to_bits(),
        p.y.to_bits(),
        p.z.to_bits(),
        v.x.to_bits(),
        v.y.to_bits(),
        v.z.to_bits(),
        w.xy.to_bits(),
        w.yz.to_bits(),
        w.zx.to_bits(),
    ]);
}

/// The determinism fixture: 4D gravity, a static floor, and a six-sphere stack
/// at fixed offsets landing on it. No RNG, so any run-to-run difference is
/// genuine nondeterminism rather than seed noise.
pub fn determinism_scenario_run(schedule: Schedule) -> ScenarioRun {
    let mut world = World::new(EuclideanR4);
    // Contacts are what make the trajectory worth pinning: without a
    // narrowphase the scenario is free fall and exercises no solver, manifold,
    // or iteration-order behavior.
    register_narrowphase_r4(&mut world.narrowphase);
    world.schedule = schedule;
    world.push_field(Box::new(Gravity::new(Vec4::new(
        0.0,
        -GRAVITY_MAGNITUDE,
        0.0,
        0.0,
    ))));
    world.push_body(halfspace4_body_r4(Vec4::Y, 0.0));
    for i in 0..6u32 {
        let y = 1.0 + i as f32 * 0.45;
        let x = ((i % 3) as f32 - 1.0) * 0.05;
        world.push_body(sphere_body_r4(
            Vec4::new(x, y, 0.0, 0.0),
            Vec4::ZERO,
            0.2,
            1.0,
        ));
    }

    run_scenario(&mut world, 1.0 / 60.0, STEPS, 4, sample_body_r4)
}

pub fn determinism_scenario_trajectory() -> Vec<u32> {
    determinism_scenario_run(Schedule::default()).trajectory
}

/// Index of the first step whose hash differs, for an assertion message that
/// names where a schedule diverged instead of only that it did.
pub fn first_divergent_step(a: &ScenarioRun, b: &ScenarioRun) -> Option<usize> {
    a.step_hashes
        .iter()
        .zip(&b.step_hashes)
        .position(|(x, y)| x != y)
}

// ---------------------------------------------------------------------------
// Multi-island R3 fixture.
// ---------------------------------------------------------------------------

const ISLAND_RADIUS: f32 = 0.5;
/// Group centres on the x axis, separated by several diameters so a group
/// would have to travel to reach its neighbour.
const ISLAND_X: [f32; 3] = [-4.0, 0.0, 4.0];
/// Bodies per group. The four-body chain is why this fixture exists: island
/// order and colour order are both vacuous on a single stack, so the R4
/// scenario cannot serve the axes that arrive with union-find islands and a
/// coloured solve.
const ISLAND_SIZES: [usize; 3] = [4, 2, 1];
/// Initial surface-to-surface separation, so the run opens with an impact
/// rather than with bodies already resting in contact.
const ISLAND_GAP: f32 = 0.05;
pub const MULTI_ISLAND_DT: f32 = 1.0 / 60.0;
pub const MULTI_ISLAND_STEPS: usize = 240;

/// Dynamic body slot ranges, one per group. Slot 0 is the static floor and
/// belongs to no group. Slots rather than handles because the fixture never
/// despawns, so slot allocation is dense and contiguous by group.
pub fn multi_island_groups() -> [Range<usize>; 3] {
    std::array::from_fn(|group| {
        let start = 1 + ISLAND_SIZES[..group].iter().sum::<usize>();
        start..start + ISLAND_SIZES[group]
    })
}

/// Three spatially disjoint sphere groups over one static floor: a four-body
/// chain, a pair, and a singleton.
///
/// Every body starts on its group's vertical axis at rest, every contact normal
/// in the scenario is therefore vertical, and the friction solve returns early
/// below its 1e-8 tangential-speed floor, so no group ever moves laterally.
/// That is what makes the island partition constant for the whole run rather
/// than a property that happens to hold for the first few hundred steps.
///
/// The floor is static, so it transmits no impulse between groups and does not
/// merge islands under the usual union-find rule.
pub fn multi_island_world(schedule: Schedule) -> World<EuclideanR3> {
    let mut world = World::new(EuclideanR3);
    register_narrowphase_r3(&mut world.narrowphase);
    world.schedule = schedule;
    world.push_field(Box::new(Gravity::new(Vec3::new(
        0.0,
        -GRAVITY_MAGNITUDE,
        0.0,
    ))));
    world.push_body(halfspace_body_r3(Vec3::Y, 0.0));
    for (group, &size) in ISLAND_SIZES.iter().enumerate() {
        for level in 0..size {
            let y = ISLAND_RADIUS + ISLAND_GAP + level as f32 * (2.0 * ISLAND_RADIUS + ISLAND_GAP);
            world.push_body(sphere_body_r3(
                Vec3::new(ISLAND_X[group], y, 0.0),
                Vec3::ZERO,
                ISLAND_RADIUS,
                1.0,
            ));
        }
    }
    world
}

pub fn multi_island_scenario_run(schedule: Schedule) -> ScenarioRun {
    let mut world = multi_island_world(schedule);
    run_scenario(
        &mut world,
        MULTI_ISLAND_DT,
        MULTI_ISLAND_STEPS,
        3,
        sample_body_r3,
    )
}

/// FNV-1a of [`multi_island_scenario_run`]'s trajectory under the default
/// schedule, recorded on x86_64 on the same terms as
/// [`GOLDEN_TRAJECTORY_HASH`].
///
/// Re-recording: [`assert_scenario_stays_physical`] runs first and is what
/// tells the two cases apart. If it passed and only the hash moved, the
/// change is intended: replace this constant with the value the hash
/// assertion prints. If it failed, the scenario stopped being physical and no
/// re-recorded hash is correct.
pub const GOLDEN_MULTI_ISLAND_HASH: u64 = 0x56fd_21a0_2e4f_76e2;

// ---------------------------------------------------------------------------
// Replay fixture: the multi-island world driven from a recorded input tape.
// ---------------------------------------------------------------------------

/// Words per recorded tick: the target handle as `(slot, generation)`, then the
/// impulse. A handle rather than a storage position, because `despawn` compacts
/// the arena and a tape outlives the run that wrote it.
const THROW_WORDS: u32 = 5;
/// `slot` value meaning the tick carried no throw. A tape must have one frame
/// per tick for its inputs to be addressable by tick, and a quiet tick still
/// has to say so.
const NO_THROW: u32 = u32::MAX;
/// Ticks between throws, so a throw lands on a solver already carrying
/// warm-start impulses rather than on a world still in free fall.
const THROW_PERIOD: u64 = 20;
/// Impulse magnitude bound, kg·m/s. The fixture's spheres are unit mass with
/// radius [`ISLAND_RADIUS`], so this is at most 2 m/s, or 1/30 m of travel per
/// step: a fifteenth of a radius, well inside the tunnelling bound the
/// `thin_wall_holds_only_below_a_recorded_per_step_displacement_r3` fixture
/// records.
const THROW_IMPULSE: f32 = 2.0;
pub const REPLAY_TICKS: u64 = 180;
/// Arbitrary but fixed, so a replay failure is reproducible from its message.
pub const REPLAY_SEED: u64 = 0x5eed_f11c_c0de_0001;
/// Checkpoint stride. Every tick would make the tape a trajectory dump; only
/// the last would let a divergence hide until the end.
const CHECKPOINT_PERIOD: u64 = 30;
/// Tick rate written into the tape header: the reciprocal of
/// [`MULTI_ISLAND_DT`], the step the flick chamber runs at.
const REPLAY_TICK_HZ: u32 = 60;

/// xorshift64* (Vigna 2016, *An experimental exploration of Marsaglia's
/// xorshift generators, scrambled*, §4). The input stream has to come from
/// somewhere reproducible; the tape header's seed is that somewhere.
fn xorshift64star(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

/// Draw in `[-1, 1)`. The top 24 bits convert to `f32` exactly and the scale is
/// a power of two, so the draw is the same value on any host that rounds
/// IEEE-754.
fn signed_unit(draw: u64) -> f32 {
    ((draw >> 40) as u32) as f32 * (1.0 / 8_388_608.0) - 1.0
}

/// The scripted throws for one seed: an impulse at a pseudo-random dynamic body
/// every [`THROW_PERIOD`] ticks, silence between.
fn generate_throws(
    seed: u64,
    ticks: u64,
    dynamic_slots: Range<u32>,
) -> Vec<[u32; THROW_WORDS as usize]> {
    // xorshift64* is a fixed point at zero, so a zero seed would emit one
    // constant forever.
    let mut state = seed | 1;
    (0..ticks)
        .map(|tick| {
            if !tick.is_multiple_of(THROW_PERIOD) {
                return [NO_THROW, 0, 0, 0, 0];
            }
            let span = u64::from(dynamic_slots.end - dynamic_slots.start);
            let slot = dynamic_slots.start + (xorshift64star(&mut state) % span) as u32;
            let mut component =
                || (signed_unit(xorshift64star(&mut state)) * THROW_IMPULSE).to_bits();
            [slot, 0, component(), component(), component()]
        })
        .collect()
}

fn apply_throw(world: &mut World<EuclideanR3>, frame: [u32; THROW_WORDS as usize]) {
    let [slot, generation, x, y, z] = frame;
    if slot == NO_THROW {
        return;
    }
    // A stale handle is a throw at a body that is gone, which is a legal thing
    // for a shared tape to contain and not an error.
    if let Some(body) = world.bodies.get_mut(BodyId::forge(slot, generation)) {
        body.apply_impulse(Vec3::new(
            f32::from_bits(x),
            f32::from_bits(y),
            f32::from_bits(z),
        ));
    }
}

/// Drive the multi-island world for `ticks`, applying each tick's input before
/// stepping, and sample [`World::state_hash`] every [`CHECKPOINT_PERIOD`]
/// ticks.
///
/// Recording and replaying differ only in the `input` they pass, so a replay
/// cannot drift into a second implementation of the scenario it is supposed to
/// reproduce.
fn drive_flick_chamber(
    ticks: u64,
    input: impl Fn(u64) -> [u32; THROW_WORDS as usize],
) -> Vec<Checkpoint> {
    let mut world = multi_island_world(Schedule::default());
    let mut checkpoints = Vec::new();
    for tick in 0..ticks {
        apply_throw(&mut world, input(tick));
        world.step(MULTI_ISLAND_DT);
        if (tick + 1).is_multiple_of(CHECKPOINT_PERIOD) {
            checkpoints.push(Checkpoint {
                tick,
                state_hash: world.state_hash(sample_body_r3),
            });
        }
    }
    checkpoints
}

/// Record a run of the flick chamber: the scripted input stream for `seed`,
/// plus the state hashes that run passed through.
pub fn record_flick_chamber_tape(seed: u64) -> Tape {
    // Slot 0 is the static floor, which absorbs no impulse; throwing at it
    // would silently turn a throw into a quiet tick.
    let dynamic_slots = 1..1 + ISLAND_SIZES.iter().sum::<usize>() as u32;
    let frames = generate_throws(seed, REPLAY_TICKS, dynamic_slots);

    let mut tape = Tape::new(REPLAY_TICK_HZ, seed, THROW_WORDS);
    for frame in &frames {
        tape.push_tick(frame);
    }
    for checkpoint in drive_flick_chamber(REPLAY_TICKS, |tick| frames[tick as usize]) {
        tape.checkpoint(checkpoint.tick, checkpoint.state_hash);
    }
    tape
}

/// Replay `tape` from its recorded input alone and return the state hashes the
/// replay observed, for comparison against [`Tape::checkpoints`].
///
/// Panics if the tape's frame width is not [`THROW_WORDS`]: a tape recorded
/// against another input layout is a tape this scenario cannot drive, and
/// reading it anyway would compare hashes of two different runs.
pub fn replay_flick_chamber_tape(tape: &Tape) -> Vec<Checkpoint> {
    assert_eq!(
        tape.words_per_tick(),
        THROW_WORDS,
        "tape frame width is not the flick chamber's: {tape}",
    );
    drive_flick_chamber(tape.ticks(), |tick| {
        let frame = tape.input(tick).expect("tick is inside the tape");
        <[u32; THROW_WORDS as usize]>::try_from(frame).expect("frame width checked above")
    })
}

/// Both fixtures pass [`assert_scenario_stays_physical`], so nothing else in
/// the suite would notice an assertion that had stopped discriminating. These
/// approach each limit from its failing side.
#[cfg(test)]
mod tests {
    use super::*;

    fn run_with(envelope: PhysicalEnvelope) -> ScenarioRun {
        ScenarioRun {
            trajectory: Vec::new(),
            step_hashes: Vec::new(),
            envelope,
        }
    }

    fn inside_both_limits() -> PhysicalEnvelope {
        PhysicalEnvelope {
            initial_energy: 1.0,
            peak_energy: 1.0,
            peak_energy_step: 0,
            lowest_height: 0.25,
            lowest_height_step: 0,
        }
    }

    #[test]
    #[should_panic(expected = "at or through the floor plane")]
    fn physical_pin_rejects_a_body_reaching_the_floor_plane_determinism() {
        let mut envelope = inside_both_limits();
        envelope.lowest_height = 0.0;
        assert_scenario_stays_physical(&run_with(envelope));
    }

    #[test]
    #[should_panic(expected = "the solve is creating energy")]
    fn physical_pin_rejects_energy_above_the_initial_configuration_determinism() {
        let mut envelope = inside_both_limits();
        // One ULP over: what the pin owes is the sign of the comparison, not a
        // margin around it.
        envelope.peak_energy = f32::from_bits(envelope.initial_energy.to_bits() + 1);
        assert_scenario_stays_physical(&run_with(envelope));
    }

    /// The two pins above cover the comparison but not the extremum scan that
    /// feeds it: reverse either search and both fixtures still pass a pin
    /// that has stopped meaning anything. Recomputed from the trajectory the
    /// run already returned, and [`run_scenario`] is shared, so pinning one
    /// fixture's scan pins it for both.
    #[test]
    fn the_recorded_envelope_is_the_trajectory_extremum_determinism() {
        let run = multi_island_scenario_run(Schedule::default());
        let world = multi_island_world(Schedule::default());
        let words = words_per_body(3);
        let per_step = world.bodies.len() * words;

        let peak = run
            .trajectory
            .chunks_exact(per_step)
            .map(|step| configuration_energy(&world, step, 3))
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(
            run.envelope.peak_energy, peak,
            "recorded peak energy is not the trajectory's maximum"
        );

        let lowest = run
            .trajectory
            .chunks_exact(per_step)
            .flat_map(|step| world.bodies.iter().zip(step.chunks_exact(words)))
            .filter(|(body, _)| body.inv_mass != 0.0)
            .map(|(_, sampled)| f32::from_bits(sampled[UP_COMPONENT]))
            .fold(f32::INFINITY, f32::min);
        assert_eq!(
            run.envelope.lowest_height, lowest,
            "recorded lowest height is not the trajectory's dynamic minimum"
        );
    }
}
