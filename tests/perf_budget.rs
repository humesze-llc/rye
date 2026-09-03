//! CPU sphere-trace tripwire: a fixed grid march timed against a release-build
//! budget. Gates the median of `BATCHES` passes, not a frame percentile, and
//! covers nothing on the GPU, in shader assembly, or in the frame loop.

use std::hint::black_box;
use std::time::Instant;

use loam_math::{EuclideanR3, Space, SphericalS3};
use loam_scene::{Scene, SceneNode};

/// `glam::Vec3` by the one name the facade crate can reach.
type Point = <EuclideanR3 as Space>::Point;

/// 64² rays keeps a pass in the low milliseconds.
const GRID: usize = 64;
/// Odd, so the reported median is a pass that was actually observed.
const BATCHES: usize = 9;
const WARMUP_PASSES: usize = 2;

/// CI budget over the maintainer-machine median.
const RUNNER_MARGIN: f64 = 4.0;
/// Quiet-machine medians: i9-13980HX, Windows 11, rustc 1.95.0, release.
const EUCLIDEAN_MEDIAN_NS: f64 = 300.0;
const SPHERICAL_MEDIAN_NS: f64 = 1390.0;

/// Loose band: rejects an invisible scene and one that swallowed the eye.
const MIN_HIT_FRACTION: f64 = 0.05;
const MAX_HIT_FRACTION: f64 = 0.60;

/// Shaped after `loam_shader::GEODESIC_MARCH_KERNEL`, not pinned to it.
const MAX_STEPS: usize = 128;
const STEP_SAFETY: f32 = 0.85;
const PROBE_EPS: f32 = 1e-4;

struct MarchConfig {
    /// Distance from the origin along +Z at which the grid's rays start.
    eye_distance: f32,
    /// Total geodesic arc a ray may accumulate before it is a miss.
    arc_budget: f32,
    /// Past this a ray has left the chart (S³ saturates at unit radius).
    escape_radius: f32,
    hit_eps: f32,
    min_step: f32,
}

/// No half-spaces: `Primitive::eval` sentinels them in curved Spaces.
fn workload_scene(scale: f32) -> Scene {
    Scene::new(
        SceneNode::sphere(Point::new(-0.35 * scale, 0.0, 0.0), 0.30 * scale)
            .smooth_union(
                SceneNode::sphere(Point::new(0.35 * scale, 0.0, 0.0), 0.22 * scale),
                0.15 * scale,
            )
            .union(SceneNode::box_(Point::new(
                0.18 * scale,
                0.10 * scale,
                0.18 * scale,
            ))),
    )
}

/// The distance sum feeds `black_box`; the count proves the pass marched.
fn march_grid<S: Space<Point = Point, Vector = Point>>(
    space: &S,
    scene: &Scene,
    config: &MarchConfig,
) -> (u32, f32) {
    let eye = Point::new(0.0, 0.0, config.eye_distance);
    let mut hits = 0u32;
    let mut distance_sum = 0.0f32;

    for row in 0..GRID {
        for column in 0..GRID {
            // Half-pixel centres over [-0.5, 0.5]² at unit depth, 53° across.
            let sx = (column as f32 + 0.5) / GRID as f32 - 0.5;
            let sy = (row as f32 + 0.5) / GRID as f32 - 0.5;
            let direction = Point::new(sx, sy, -1.0).normalize();

            if let Some(t) = march(space, scene, config, eye, direction) {
                hits += 1;
                distance_sum += t;
            }
        }
    }
    (hits, distance_sum)
}

fn march<S: Space<Point = Point, Vector = Point>>(
    space: &S,
    scene: &Scene,
    config: &MarchConfig,
    eye: Point,
    direction: Point,
) -> Option<f32> {
    let mut p = eye;

    // Chart direction to unit tangent: rescale by the metric's local stretch.
    let probed = space.exp(p, direction * PROBE_EPS);
    let stretch = space.distance(p, probed) / PROBE_EPS;
    let mut v = direction / stretch.max(1e-7);

    let mut arc = 0.0f32;
    for _ in 0..MAX_STEPS {
        if p.length() > config.escape_radius {
            return None;
        }
        let d = scene.eval(space, p);
        if d < config.hit_eps {
            return Some(arc);
        }
        if arc > config.arc_budget {
            return None;
        }
        let step = (d * STEP_SAFETY).max(config.min_step);
        let next = space.exp(p, v * step);
        let transported = space.parallel_transport(p, next, v);
        p = next;
        if transported.length_squared() > 1e-12 {
            v = transported;
        }
        arc += step;
    }
    None
}

fn median_nanos_per_ray(mut pass: impl FnMut() -> (u32, f32)) -> (f64, u32) {
    for _ in 0..WARMUP_PASSES {
        black_box(pass());
    }

    let rays = (GRID * GRID) as f64;
    let mut batches = [0.0f64; BATCHES];
    let mut hits = 0u32;
    for batch in &mut batches {
        let start = Instant::now();
        let (pass_hits, distance_sum) = pass();
        *batch = start.elapsed().as_nanos() as f64 / rays;
        black_box(distance_sum);
        hits = pass_hits;
    }
    batches.sort_unstable_by(f64::total_cmp);
    (batches[BATCHES / 2], hits)
}

#[test]
#[ignore = "perf budget; needs --release, run in CI's own job"]
fn cpu_march_stays_inside_its_budget() {
    if cfg!(debug_assertions) {
        panic!("the budget is a release-build number; re-run with --release");
    }

    // E³ has no chart boundary; the escape radius bounds an escaping ray.
    let euclidean_scene = workload_scene(1.0);
    let euclidean_config = MarchConfig {
        eye_distance: 2.0,
        arc_budget: 20.0,
        escape_radius: 8.0,
        hit_eps: 1e-3,
        min_step: 1e-4,
    };

    // S³'s chart saturates at unit radius; everything stays inside it.
    let spherical_scene = workload_scene(0.35);
    let spherical_config = MarchConfig {
        eye_distance: 0.80,
        arc_budget: 2.0,
        escape_radius: 0.92,
        hit_eps: 3.5e-4,
        min_step: 3.5e-5,
    };

    let (euclidean_ns, euclidean_hits) =
        median_nanos_per_ray(|| march_grid(&EuclideanR3, &euclidean_scene, &euclidean_config));
    let (spherical_ns, spherical_hits) =
        median_nanos_per_ray(|| march_grid(&SphericalS3, &spherical_scene, &spherical_config));

    let rays = (GRID * GRID) as u32;
    println!(
        "[perf_budget] rays/pass {rays}, batches {BATCHES}\n\
         [perf_budget] EuclideanR3 {euclidean_ns:7.1} ns/ray  budget {:7.1}  hits {euclidean_hits}\n\
         [perf_budget] SphericalS3 {spherical_ns:7.1} ns/ray  budget {:7.1}  hits {spherical_hits}",
        EUCLIDEAN_MEDIAN_NS * RUNNER_MARGIN,
        SPHERICAL_MEDIAN_NS * RUNNER_MARGIN,
    );

    assert_measured_the_work("EuclideanR3", euclidean_hits);
    assert_measured_the_work("SphericalS3", spherical_hits);

    assert!(
        euclidean_ns <= EUCLIDEAN_MEDIAN_NS * RUNNER_MARGIN,
        "EuclideanR3 march {euclidean_ns:.1} ns/ray is past its \
         {:.1} ns/ray budget",
        EUCLIDEAN_MEDIAN_NS * RUNNER_MARGIN,
    );
    assert!(
        spherical_ns <= SPHERICAL_MEDIAN_NS * RUNNER_MARGIN,
        "SphericalS3 march {spherical_ns:.1} ns/ray is past its \
         {:.1} ns/ray budget",
        SPHERICAL_MEDIAN_NS * RUNNER_MARGIN,
    );
}

fn assert_measured_the_work(label: &str, hits: u32) {
    let fraction = f64::from(hits) / (GRID * GRID) as f64;
    assert!(
        (MIN_HIT_FRACTION..=MAX_HIT_FRACTION).contains(&fraction),
        "{label} hit fraction {fraction:.3} is outside \
         [{MIN_HIT_FRACTION}, {MAX_HIT_FRACTION}]: the pass is no longer \
         marching the scene, so its timing is not a march timing",
    );
}
