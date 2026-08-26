//! Letters as bodies a solver carries: the dynamic half of the glyph collider
//! path, against the static half `glyph_collider_world.rs` covers.
//!
//! A rigid body in `loam-physics` has one collider and no per-collider offset,
//! so the two halves need different decompositions and this file pins both
//! sides of that split: a letter's convex hull falls, lands and holds still,
//! and the faithful cover provably cannot, which is what makes
//! `GlyphSolid::colliders_4d`'s static-only contract a measurement rather than
//! a caution.
//!
//! Fonts are not vendored, so every test here skips cleanly when the machine
//! has none.

use ab_glyph::FontRef;
use glam::{Vec2, Vec4, Vec4Swizzles};

use loam_math::{EuclideanR4, Rotor};
use loam_physics::euclidean_r4::{
    halfspace4_body_r4, polytope_body_r4, register_default_narrowphase,
};
use loam_physics::manifold::PENETRATION_SLOP;
use loam_physics::{BodyId, Gravity, World};
use loam_shape::{Shape, Visualizable};
use loam_text::glyph::{layout_word, GlyphParams, GlyphSolid};

const WORD: &str = "LOAM";
const GRAVITY: f32 = -9.8;

/// 240 Hz. At 120 Hz the same drops do not come to rest: over ten seconds `O`,
/// `A` and `M` skid 0.45 to 0.61 em off their spawn column and end with their
/// centre 0.14 to 0.25 em lower, deepest vertex still on the floor, i.e. tipped
/// rather than sunk. The narrowphase reports one deepest vertex per step and a
/// manifold holds four, so halving the rate halves the constraints the solver
/// has accumulated by the time the letter has rocked onto a corner.
const DT: f32 = 1.0 / 240.0;

/// Clearance between a letter's lowest hull vertex and the floor at spawn, so
/// the landing is at 1.0 m/s.
///
/// Deliberately gentle, and a sampled point rather than the top of an envelope.
/// `polytope_halfspace_r4` reports one deepest vertex per step and a manifold
/// holds four, so a level 4D landing has more tied deepest corners than the
/// solver can carry constraints for, and whether a given landing recovers is
/// not monotone in drop height. Sampled at 0.01 through 0.17 em over twenty
/// seconds: 0.05 holds every letter within 0.005 em of residual motion and
/// 0.082 em of its spawn column, while 0.02 lets `O` skid 0.48 em, 0.04 lets
/// `M` skid 0.83 em, and everything from 0.06 up skids 0.27 to 2.2 em. Widening
/// this to a range is a multi-contact narrowphase in `loam-physics`, not a
/// change here.
const DROP_CLEARANCE: f32 = 0.05;

/// 8 s. The letter lands inside 0.15 s; the rest is the window in which a
/// jitter that never damps would show itself.
const SETTLE_STEPS: usize = 1920;
/// Final second of that run, over which a settled letter must hold still.
const REST_WINDOW: usize = 240;
/// Position spread allowed across [`REST_WINDOW`], against a measured worst of
/// 0.005 em over twenty seconds. What is left at rest is the Baumgarte limit
/// cycle: the letter sinks by gravity over a step and is pushed back out by the
/// positional bias, forever, at an amplitude set by the slop.
const REST_SPREAD: f32 = 0.02;
/// Horizontal displacement allowed between spawn and rest, against a measured
/// worst of 0.082 em. The landing impulse arrives at one corner, so the letter
/// takes a small kick before the manifold fills.
const LANDING_SLIDE: f32 = 0.15;

fn system_font() -> Option<Vec<u8>> {
    const CANDIDATES: &[&str] = &[
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ];
    CANDIDATES
        .iter()
        .find_map(|path| std::fs::read(path).ok())
        .or_else(|| {
            eprintln!("skip: no system font found in {CANDIDATES:?}");
            None
        })
}

fn word() -> Option<Vec<GlyphSolid>> {
    let bytes = system_font()?;
    let font = FontRef::try_from_slice(&bytes).expect("parse font");
    Some(layout_word(&font, WORD, &GlyphParams::default()).expect("layout"))
}

fn hull_of(letter: &GlyphSolid) -> (Vec4, Vec<Vec4>) {
    let (centre, shape) = letter.rigid_hull_4d().expect("letter has a hull");
    let Shape::ConvexPolytope4D { vertices } = shape else {
        panic!("dynamic letter collider is not 4D convex");
    };
    (centre, vertices)
}

/// A world with gravity along `-y` and a floor at `y = 0`, i.e. letters
/// standing on the baseline they were laid out on.
fn floor_world() -> World<EuclideanR4> {
    let mut world = World::new(EuclideanR4);
    register_default_narrowphase(&mut world.narrowphase);
    world.push_field(Box::new(Gravity::new(Vec4::new(0.0, GRAVITY, 0.0, 0.0))));
    let floor = world.push_body(halfspace4_body_r4(Vec4::Y, 0.0));
    // Restitution 0 on both sides so the tests are about the contact pipeline
    // converging rather than about a bounce damping out, the same choice the
    // in-crate 4D settle tests make.
    world.bodies[floor].restitution = 0.0;
    world
}

/// Spawn `letter` as one dynamic body with its lowest hull vertex
/// [`DROP_CLEARANCE`] above the floor, keeping its laid-out `x`, `z` and `w`.
/// Returns the handle and the spawn position.
fn drop_letter(world: &mut World<EuclideanR4>, letter: &GlyphSolid) -> (BodyId, Vec4) {
    let (centre, vertices) = hull_of(letter);
    let lowest = vertices.iter().fold(f32::INFINITY, |m, v| m.min(v.y));
    let spawn = Vec4::new(centre.x, DROP_CLEARANCE - lowest, centre.z, centre.w);
    let id = world.push_body(polytope_body_r4(spawn, Vec4::ZERO, vertices, 1.0));
    world.bodies[id].restitution = 0.0;
    (id, spawn)
}

/// World-space `y` of the lowest point of a posed hull.
fn deepest_y(world: &World<EuclideanR4>, id: BodyId) -> f32 {
    let body = &world.bodies[id];
    let Shape::ConvexPolytope4D { vertices } = &body.collider else {
        unreachable!("spawned as a 4D polytope")
    };
    vertices
        .iter()
        .map(|v| body.orientation.rotation.apply(*v).y + body.position.y)
        .fold(f32::INFINITY, f32::min)
}

/// Step to rest and return `(worst position spread over the last second,
/// deepest hull point, horizontal displacement from spawn)`.
fn settle(world: &mut World<EuclideanR4>, id: BodyId, spawn: Vec4) -> (f32, f32, f32) {
    let mut rest = Vec::with_capacity(REST_WINDOW);
    for step in 0..SETTLE_STEPS {
        world.step(DT);
        let position = world.bodies[id].position;
        assert!(
            position.is_finite(),
            "diverged to {position} at step {step}"
        );
        if step >= SETTLE_STEPS - REST_WINDOW {
            rest.push(position);
        }
    }
    let spread = rest
        .iter()
        .map(|p| p.distance(rest[0]))
        .fold(0.0f32, f32::max);
    let settled = *rest.last().expect("rest window");
    let slide = ((settled - spawn) * Vec4::new(1.0, 0.0, 1.0, 1.0)).length();
    (spread, deepest_y(world, id), slide)
}

/// A body at rest on the floor is in contact with it: touching within the
/// solver's own resting overlap, neither floating above nor sunk into it.
fn assert_resting_on_the_floor(ch: char, deepest: f32) {
    assert!(
        deepest <= 1.0e-4,
        "{ch:?} floats with its lowest point at {deepest}"
    );
    assert!(
        deepest >= -2.0 * PENETRATION_SLOP,
        "{ch:?} sank to {deepest} through the floor"
    );
}

#[test]
fn a_word_becomes_one_dynamic_body_per_letter_within_the_polytope_vertex_cap() {
    let Some(letters) = word() else { return };
    let mut world = World::new(EuclideanR4);
    register_default_narrowphase(&mut world.narrowphase);

    for letter in &letters {
        let (centre, vertices) = hull_of(letter);
        let sides = letter.rigid_hull_sides();
        assert!(
            (3..=8).contains(&sides),
            "{:?} hulls to {sides} sides",
            letter.ch()
        );
        assert_eq!(vertices.len(), 4 * sides);
        assert!(
            vertices.len() <= 32,
            "{:?} emits {} vertices, past the 4D narrowphase buffer",
            letter.ch(),
            vertices.len()
        );
        world.push_body(polytope_body_r4(centre, Vec4::ZERO, vertices, 1.0));
    }

    assert_eq!(world.bodies.iter().count(), letters.len());
    let boxes: usize = letters.iter().map(GlyphSolid::collider_count).sum();
    assert!(
        boxes > 4 * letters.len(),
        "{WORD} covers in {boxes} boxes, so one body per letter is not a cut"
    );
}

#[test]
fn a_letters_hull_contains_both_its_render_mesh_and_its_cover() {
    let Some(letters) = word() else { return };
    let params = GlyphParams::default();
    for letter in &letters {
        let (centre, vertices) = hull_of(letter);
        let sides = letter.rigid_hull_sides();
        // The prism repeats the ring once per `(z, w)` corner, so its first
        // `sides` vertices are the ring itself, in order.
        let ring: Vec<Vec2> = vertices[..sides]
            .iter()
            .map(|v| (*v + centre).xy())
            .collect();
        let outside = |p: Vec2| {
            (0..sides).any(|k| {
                let a = ring[k];
                let b = ring[(k + 1) % sides];
                (b - a).perp_dot(p - a) < -1.0e-5
            })
        };

        let mesh = Visualizable::<3>::to_triangles(letter).expect("mesh");
        for v in &mesh.vertices {
            assert!(
                !outside(Vec2::new(v[0], v[1])),
                "{:?} renders a vertex at ({}, {}) outside its dynamic hull",
                letter.ch(),
                v[0],
                v[1]
            );
        }

        let cover = letter.collider_cover().expect("cover");
        for index in 0..cover.piece_count() {
            let (lo, hi) = cover.piece_bounds(index);
            for y in [lo[1], hi[1]] {
                for x in [lo[0], hi[0]] {
                    assert!(
                        !outside(Vec2::new(x, y)),
                        "{:?} covers ({x}, {y}) outside its dynamic hull",
                        letter.ch()
                    );
                }
            }
        }

        // z and w are the prism's own axes, so containment there is exact.
        for v in &vertices {
            let world = *v + centre;
            assert!((world.z.abs() - 0.5 * params.depth).abs() < 1.0e-6);
            assert!(world.w >= params.slab.0 - 1.0e-6 && world.w <= params.slab.1 + 1.0e-6);
        }
    }
}

#[test]
fn a_letter_dropped_on_a_halfspace_settles_without_jitter() {
    let Some(letters) = word() else { return };
    for letter in &letters {
        let mut world = floor_world();
        let (id, spawn) = drop_letter(&mut world, letter);
        let (spread, deepest, slide) = settle(&mut world, id, spawn);

        assert!(
            spread < REST_SPREAD,
            "{:?} still moves {spread} in the final second",
            letter.ch()
        );
        assert_resting_on_the_floor(letter.ch(), deepest);
        assert!(
            slide < LANDING_SLIDE,
            "{:?} slid {slide} off its spawn column",
            letter.ch()
        );
        // It fell rather than being spawned in contact.
        assert!(world.bodies[id].position.y < spawn.y - 0.5 * DROP_CLEARANCE);
    }
}

#[test]
fn a_letter_falls_only_because_a_force_field_supplies_gravity() {
    let Some(letters) = word() else { return };
    let letter = &letters[0];

    let mut inert = World::new(EuclideanR4);
    register_default_narrowphase(&mut inert.narrowphase);
    let (centre, vertices) = hull_of(letter);
    let id = inert.push_body(polytope_body_r4(centre, Vec4::ZERO, vertices, 1.0));
    assert!(inert.fields.is_empty());
    for _ in 0..240 {
        inert.step(DT);
    }
    assert_eq!(
        inert.bodies[id].position, centre,
        "a letter moved with no force field registered"
    );
    assert_eq!(inert.bodies[id].velocity, Vec4::ZERO);

    let mut falling = floor_world();
    assert_eq!(falling.fields.len(), 1);
    let (id, spawn) = drop_letter(&mut falling, letter);
    falling.step(DT);
    let velocity = falling.bodies[id].velocity;
    assert!(
        (velocity.y - GRAVITY * DT).abs() < 1.0e-6,
        "one step under gravity gave {velocity}, not {} along y",
        GRAVITY * DT
    );
    assert_eq!(velocity.xz(), Vec2::ZERO);
    assert_eq!(velocity.w, 0.0);
    assert!(falling.bodies[id].position.y < spawn.y);
}

#[test]
fn the_cover_spawned_as_dynamic_bodies_tears_a_letter_apart() {
    let Some(letters) = word() else { return };
    let o = letters.iter().find(|l| l.ch() == 'O').expect("O");

    let mut world = World::new(EuclideanR4);
    register_default_narrowphase(&mut world.narrowphase);
    let colliders = o.colliders_4d();
    assert!(
        colliders.len() > 8,
        "fixture is too small to show the effect"
    );
    let spawned: Vec<(BodyId, Vec4)> = colliders
        .into_iter()
        .map(|(centre, hull)| {
            let Shape::ConvexPolytope4D { vertices } = hull else {
                unreachable!()
            };
            (
                world.push_body(polytope_body_r4(centre, Vec4::ZERO, vertices, 1.0)),
                centre,
            )
        })
        .collect();

    for _ in 0..480 {
        world.step(DT);
    }
    let worst = spawned
        .iter()
        .map(|(id, spawn)| world.bodies[*id].position.distance(*spawn))
        .fold(0.0f32, f32::max);
    assert!(
        worst > GlyphParams::default().em_size,
        "the cover held together to within {worst}; if `loam-physics` grew a \
         compound collider, `colliders_4d`'s static-only contract is stale"
    );
}

#[test]
fn two_drops_of_a_letter_agree_bit_for_bit() {
    let Some(letters) = word() else { return };
    let letter = &letters[2];
    let trajectory = || {
        let mut world = floor_world();
        let (id, _) = drop_letter(&mut world, letter);
        let mut samples = Vec::with_capacity(600);
        for _ in 0..600 {
            world.step(DT);
            samples.push(world.bodies[id].position.to_array());
        }
        samples
    };
    assert_eq!(trajectory(), trajectory());
}

#[test]
fn a_whole_word_dropped_together_settles_in_its_own_line() {
    let Some(letters) = word() else { return };
    let mut world = floor_world();
    let dropped: Vec<(BodyId, Vec4)> = letters
        .iter()
        .map(|letter| drop_letter(&mut world, letter))
        .collect();
    assert_eq!(world.bodies.iter().count(), letters.len() + 1);

    let mut rest: Vec<Vec<Vec4>> = vec![Vec::with_capacity(REST_WINDOW); letters.len()];
    for step in 0..SETTLE_STEPS {
        world.step(DT);
        if step >= SETTLE_STEPS - REST_WINDOW {
            for (samples, (id, _)) in rest.iter_mut().zip(&dropped) {
                samples.push(world.bodies[*id].position);
            }
        }
    }

    for ((samples, (id, spawn)), letter) in rest.iter().zip(&dropped).zip(&letters) {
        let spread = samples
            .iter()
            .map(|p| p.distance(samples[0]))
            .fold(0.0f32, f32::max);
        assert!(
            spread < REST_SPREAD,
            "{:?} still moves {spread} in the final second",
            letter.ch()
        );
        assert_resting_on_the_floor(letter.ch(), deepest_y(&world, *id));
        let settled = world.bodies[*id].position;
        assert!(
            (settled.x - spawn.x).abs() < LANDING_SLIDE,
            "{:?} slid to x = {} from {}",
            letter.ch(),
            settled.x,
            spawn.x
        );
    }

    // Reading order survives the landing.
    for pair in dropped.windows(2) {
        assert!(world.bodies[pair[1].0].position.x > world.bodies[pair[0].0].position.x);
    }
}

#[test]
fn a_blank_has_no_dynamic_body() {
    let Some(bytes) = system_font() else { return };
    let font = FontRef::try_from_slice(&bytes).expect("parse font");
    let letters = layout_word(&font, "A B", &GlyphParams::default()).expect("layout");
    assert!(letters[1].is_blank());
    assert!(letters[1].rigid_hull_4d().is_none());
    assert_eq!(letters[1].rigid_hull_sides(), 0);
    assert!(letters[0].rigid_hull_4d().is_some());
}
