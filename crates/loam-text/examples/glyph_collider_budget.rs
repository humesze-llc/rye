//! `cargo run --release -p loam-text --example glyph_collider_budget`.

use std::time::Instant;

use ab_glyph::FontRef;
use glam::Vec4;
use loam_math::EuclideanR4;
use loam_physics::euclidean_r4::{register_default_narrowphase, sphere_body_r4};
use loam_physics::{Gravity, RigidBody, World};
use loam_shape::Shape;
use loam_text::glyph::{layout_word, GlyphParams, GlyphSolid};

const WORD: &str = "LOAM";
const FIXED_HZ: f32 = 60.0;
const SETTLE_STEPS: usize = 300;
const TIMED_STEPS: usize = 60;
const BALL_RADIUS: f32 = 0.04;
const BALLS: usize = 30;

// Fonts are not vendored.
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
            eprintln!("no system font found in {CANDIDATES:?}");
            None
        })
}

// Xorshift64 (Marsaglia, 2003, "Xorshift RNGs", the 13/7/17 triple), so the
// drop pattern is seeded and the timing is reproducible.
struct Xorshift64(u64);

impl Xorshift64 {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u32 << 24) as f32
    }
}

fn main() {
    let Some(bytes) = system_font() else { return };
    let font = FontRef::try_from_slice(&bytes).expect("parse font");

    let render = GlyphParams::default();
    let render_pieces: usize = layout_word(&font, WORD, &render)
        .expect("layout")
        .iter()
        .map(GlyphSolid::piece_count)
        .sum();

    println!("{WORD}, render pitch {} cells/em", render.resolution);
    println!("pitch    L    O    A    M  boxes  render  margin/em   bake");
    for collider_resolution in [16u32, 24, 32, 48, 96] {
        let params = GlyphParams {
            collider_resolution,
            ..GlyphParams::default()
        };
        let started = Instant::now();
        let letters = layout_word(&font, WORD, &params).expect("layout");
        let elapsed = started.elapsed();
        let per_letter: Vec<usize> = letters.iter().map(GlyphSolid::collider_count).collect();
        println!(
            "{collider_resolution:5} {:4} {:4} {:4} {:4} {:6} {render_pieces:7} {:10.4} {elapsed:>6.1?}",
            per_letter[0],
            per_letter[1],
            per_letter[2],
            per_letter[3],
            per_letter.iter().sum::<usize>(),
            letters[0].collider_margin(),
        );
    }

    let letters = layout_word(&font, WORD, &GlyphParams::default()).expect("layout");

    println!();
    println!("dynamic letters at the default pitch");
    println!("letter  boxes  sides  verts  hull/cover");
    for letter in &letters {
        let Some((_, Shape::ConvexPolytope4D { vertices })) = letter.rigid_hull_4d() else {
            continue;
        };
        let cover = letter.collider_cover().expect("cover");
        println!(
            "{:>6} {:6} {:6} {:6} {:11.2}",
            letter.ch(),
            letter.collider_count(),
            letter.rigid_hull_sides(),
            vertices.len(),
            hull_area(letter) / cover.volume(),
        );
    }
    println!(
        "{WORD}: {} dynamic bodies against {} static boxes",
        letters.iter().filter(|l| !l.is_blank()).count(),
        letters
            .iter()
            .map(GlyphSolid::collider_count)
            .sum::<usize>(),
    );

    let (bodies, micros) = time_word(&letters);
    let frame = 1.0e6 / FIXED_HZ as f64;
    println!();
    println!(
        "{bodies} bodies ({} static boxes + {BALLS} spheres): {micros:.1} us/step, {:.2}% of a {FIXED_HZ:.0} Hz frame",
        bodies - BALLS,
        100.0 * micros / frame,
    );
}

// The first `sides` vertices of the 4D prism are the ring itself, in order, so
// the shoelace formula reads straight off them.
fn hull_area(letter: &GlyphSolid) -> f32 {
    let Some((_, Shape::ConvexPolytope4D { vertices })) = letter.rigid_hull_4d() else {
        return 0.0;
    };
    let n = letter.rigid_hull_sides();
    let doubled: f32 = (0..n)
        .map(|i| {
            let a = vertices[i];
            let b = vertices[(i + 1) % n];
            a.x * b.y - b.x * a.y
        })
        .sum();
    0.5 * doubled
}

// Returns the body count and the mean microseconds per fixed step.
fn time_word(letters: &[GlyphSolid]) -> (usize, f64) {
    let mut world = World::new(EuclideanR4);
    register_default_narrowphase(&mut world.narrowphase);
    // Along -z, so the spheres land on the letters' front faces rather than
    // edge-on.
    world.push_field(Box::new(Gravity::new(Vec4::new(0.0, 0.0, -9.8, 0.0))));

    for letter in letters {
        for (centre, hull) in letter.colliders_4d() {
            world.push_body(RigidBody::fixed(centre, hull, 1.0, &EuclideanR4));
        }
    }

    let span_x = letters
        .iter()
        .fold(0.0f32, |m, l| m.max(l.pen_origin().x + l.advance()));
    let mut rng = Xorshift64(0x2545_F491_4F6C_DD1D);
    for _ in 0..BALLS {
        let position = Vec4::new(
            rng.unit() * span_x,
            rng.unit() * 0.7,
            0.3 + rng.unit() * 0.3,
            0.0,
        );
        world.push_body(sphere_body_r4(position, Vec4::ZERO, BALL_RADIUS, 1.0));
    }

    let dt = 1.0 / FIXED_HZ;
    for _ in 0..SETTLE_STEPS {
        world.step(dt);
    }
    let started = Instant::now();
    for _ in 0..TIMED_STEPS {
        world.step(dt);
    }
    let micros = started.elapsed().as_secs_f64() * 1.0e6 / TIMED_STEPS as f64;
    (world.bodies.iter().count(), micros)
}
