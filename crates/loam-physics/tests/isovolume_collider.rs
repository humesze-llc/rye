use glam::Vec3;

use loam_math::EuclideanR3;
use loam_physics::euclidean_r3::{register_default_narrowphase, sphere_body_r3};
use loam_physics::{BodyId, Collider, Gravity, RigidBody, World};
use loam_shape::Isovolume;

const MAJOR: f32 = 1.0;
const MINOR: f32 = 0.3;
const BOUND: f32 = 1.6;
/// The resolution the extractor's measured budget names as usable in 3D.
const RESOLUTION: usize = 32;
const DT: f32 = 1.0 / 120.0;
const SPHERE_RADIUS: f32 = 0.1;
const DROP_HEIGHT: f32 = 1.1;

/// Torus of revolution about `y`, exact signed distance (Quilez, "distance
/// functions", 2019, `sdTorus`).
fn torus(p: [f32; 3]) -> f32 {
    let radial = (p[0] * p[0] + p[2] * p[2]).sqrt() - MAJOR;
    (radial * radial + p[1] * p[1]).sqrt() - MINOR
}

fn extract() -> Isovolume<3> {
    let volume = Isovolume::extract([-BOUND; 3], [BOUND; 3], RESOLUTION, torus);
    assert!(!volume.clipped(), "sampling domain clipped the solid");
    volume
}

fn torus_world(volume: &Isovolume<3>, x: f32) -> (World<EuclideanR3>, BodyId) {
    let mut world = World::new(EuclideanR3);
    register_default_narrowphase(&mut world.narrowphase);
    world.push_field(Box::new(Gravity::new(Vec3::new(0.0, -9.8, 0.0))));

    for (centre, shape) in volume.colliders() {
        world.push_body(RigidBody::fixed(centre, shape, 1.0, &EuclideanR3));
    }
    let sphere = world.push_body(sphere_body_r3(
        Vec3::new(x, DROP_HEIGHT, 0.0),
        Vec3::ZERO,
        SPHERE_RADIUS,
        1.0,
    ));
    (world, sphere)
}

#[test]
fn a_body_dropped_on_an_sdf_extracted_torus_rests_on_its_surface() {
    let volume = extract();
    assert!(
        volume.piece_count() < 128,
        "{} pieces",
        volume.piece_count()
    );
    let (mut world, sphere) = torus_world(&volume, MAJOR);

    let mut lowest = f32::INFINITY;
    for _ in 0..600 {
        world.step(DT);
        let y = world.bodies.get(sphere).unwrap().position.y;
        assert!(y.is_finite(), "simulation diverged");
        lowest = lowest.min(y);
    }
    let body = world.bodies.get(sphere).unwrap();

    assert!(
        body.position.y < DROP_HEIGHT - 0.25,
        "sphere never fell: y = {}",
        body.position.y
    );
    let clearance = torus(body.position.to_array());
    assert!(
        clearance >= SPHERE_RADIUS - 1.0e-3,
        "sphere sank into the surface: clearance {clearance}"
    );
    assert!(
        clearance <= SPHERE_RADIUS + volume.enclosure_margin(),
        "sphere is not touching the cover: clearance {clearance}"
    );
    assert!(
        lowest > MINOR,
        "sphere reached y = {lowest}, i.e. passed through the tube"
    );
    assert!(
        body.velocity.length() < 0.5,
        "sphere has not settled: |v| = {}",
        body.velocity.length()
    );
}

#[test]
fn a_body_dropped_down_the_torus_axis_passes_through_the_hole() {
    let volume = extract();
    let (mut world, sphere) = torus_world(&volume, 0.0);
    for _ in 0..600 {
        world.step(DT);
    }
    let y = world.bodies.get(sphere).unwrap().position.y;
    assert!(
        y < -1.0,
        "sphere stopped at y = {y} instead of falling through the hole"
    );
}

#[test]
fn two_runs_over_the_extracted_colliders_agree_bit_for_bit() {
    let trajectory = || {
        let volume = extract();
        let (mut world, sphere) = torus_world(&volume, MAJOR);
        let mut samples = Vec::with_capacity(200);
        for _ in 0..200 {
            world.step(DT);
            samples.push(world.bodies.get(sphere).unwrap().position.to_array());
        }
        samples
    };
    assert_eq!(trajectory(), trajectory());
}

#[test]
fn every_extracted_piece_is_a_non_degenerate_hull() {
    let volume = extract();
    let cell = volume.cell_size();
    for (_, shape) in volume.colliders() {
        let Collider::ConvexPolytope3D { vertices } = shape else {
            panic!("expected ConvexPolytope3D, got {:?}", shape.kind());
        };
        assert_eq!(vertices.len(), 8);
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for v in &vertices {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        assert!(
            (hi - lo).min_element() >= cell - 1.0e-5,
            "piece thinner than a cell: {:?}",
            hi - lo
        );
    }
}
