//! Callers drive [`World::step`] through this module rather than
//! re-implementing the phase loop, so a schedule variant is always compared
//! against the simulation the golden hash pins.

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

// Contact anchors carry platform libm rounding from orientation into the solve.
#[cfg(all(target_arch = "x86_64", target_os = "windows", target_env = "msvc"))]
pub const GOLDEN_TRAJECTORY_HASH: u64 = 0xbc70_273d_6c03_b6da;

#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
pub const GOLDEN_TRAJECTORY_HASH: u64 = 0x922b_fa3d_97bf_fc7a;

pub struct ScenarioRun {
    /// Every step, every body, linear and angular state as raw f32 bits.
    pub trajectory: Vec<u32>,
    /// Running hash after each step over a superset of `trajectory`: the same
    /// body words plus the manifold keys, point counts, and normal impulses.
    pub step_hashes: Vec<u64>,
    pub envelope: PhysicalEnvelope,
}

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

// Both fixtures fall along -y onto a static half-space at `y = 0`, so heights
// and potential energies read off the y component directly.
const GRAVITY_MAGNITUDE: f32 = 9.8;
const UP_COMPONENT: usize = 1;

// `dims` position, `dims` velocity, and the `C(dims, 2)` bivector components.
const fn words_per_body(dims: usize) -> usize {
    2 * dims + dims * (dims - 1) / 2
}

// Every space here uses a scalar isotropic moment, so the rotational term is
// `½·I·|ω|²` with `|ω|` the Euclidean norm of the bivector components.
fn mechanical_energy(words: &[u32], dims: usize, mass: f32, inertia: f32) -> f32 {
    let read = |i: usize| f32::from_bits(words[i]);
    let sum_squares = |range: Range<usize>| range.map(|i| read(i) * read(i)).sum::<f32>();
    0.5 * mass * sum_squares(dims..2 * dims)
        + 0.5 * inertia * sum_squares(2 * dims..words.len())
        + mass * GRAVITY_MAGNITUDE * read(UP_COMPONENT)
}

// Statics are skipped rather than contributing zero: their sampled height is
// whatever the constructor gave them, not their plane's.
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

/// Neither limit is a recorded measurement: both follow from the scenario, so
/// tripping one says the simulation stopped being physical.
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

// Every fixture routes through here so all of them hash the same quantities in
// the same order.
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

    // Into its own buffer, so the energy budget is the starting configuration
    // and not the first step's already-fallen state.
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
        // simulation diverged and never that the instrument reordered.
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

/// No RNG, so any run-to-run difference is genuine nondeterminism rather than
/// seed noise.
pub fn determinism_scenario_run(schedule: Schedule) -> ScenarioRun {
    let mut world = World::new(EuclideanR4);
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

pub fn first_divergent_step(a: &ScenarioRun, b: &ScenarioRun) -> Option<usize> {
    a.step_hashes
        .iter()
        .zip(&b.step_hashes)
        .position(|(x, y)| x != y)
}

const ISLAND_RADIUS: f32 = 0.5;
const ISLAND_X: [f32; 3] = [-4.0, 0.0, 4.0];
const ISLAND_SIZES: [usize; 3] = [4, 2, 1];
const ISLAND_GAP: f32 = 0.05;
pub const MULTI_ISLAND_DT: f32 = 1.0 / 60.0;
pub const MULTI_ISLAND_STEPS: usize = 240;

/// Slot 0 is the static floor and belongs to no group. Slots rather than
/// handles because the fixture never despawns, so slot allocation is dense and
/// contiguous by group.
pub fn multi_island_groups() -> [Range<usize>; 3] {
    std::array::from_fn(|group| {
        let start = 1 + ISLAND_SIZES[..group].iter().sum::<usize>();
        start..start + ISLAND_SIZES[group]
    })
}

/// Every body starts on its group's vertical axis at rest, so every contact
/// normal is vertical and the friction solve returns early below its 1e-8
/// tangential-speed floor. No group moves laterally, which is what makes the
/// island partition constant for the whole run.
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

/// Recorded on x86_64 with Windows MSVC and Linux GNU.
pub const GOLDEN_MULTI_ISLAND_HASH: u64 = 0x56fd_21a0_2e4f_76e2;

// The target handle as `(slot, generation)`, then the impulse. A handle rather
// than a storage position, because `despawn` compacts the arena and a tape
// outlives the run that wrote it.
const THROW_WORDS: u32 = 5;
// `slot` value meaning the tick carried no throw.
const NO_THROW: u32 = u32::MAX;
const THROW_PERIOD: u64 = 20;
// Per-component bound, kg·m/s.
const THROW_IMPULSE: f32 = 2.0;
pub const REPLAY_TICKS: u64 = 180;
pub const REPLAY_SEED: u64 = 0x5eed_f11c_c0de_0001;
const CHECKPOINT_PERIOD: u64 = 30;
// The reciprocal of [`MULTI_ISLAND_DT`], the step the flick chamber runs at.
const REPLAY_TICK_HZ: u32 = 60;

// xorshift64* (Vigna 2016, *An experimental exploration of Marsaglia's
// xorshift generators, scrambled*, §4).
fn xorshift64star(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

// Draw in `[-1, 1)`. The top 24 bits convert to `f32` exactly and the scale is
// a power of two, so the draw is the same on any host that rounds IEEE-754.
fn signed_unit(draw: u64) -> f32 {
    ((draw >> 40) as u32) as f32 * (1.0 / 8_388_608.0) - 1.0
}

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

// Recording and replaying differ only in the `input` they pass, so a replay
// cannot drift into a second implementation of the scenario.
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

/// The returned tape carries the scripted input stream for `seed` and the
/// state hashes that run passed through.
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

/// Drives from the tape's recorded input alone, for comparison against
/// [`Tape::checkpoints`]. Panics if the frame width is not `THROW_WORDS`:
/// another input layout would compare hashes of two different runs.
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
