//! `cargo run --release -p loam-shape --example isovolume_piece_budget`.

use std::time::Instant;

use loam_shape::Isovolume;

const MAJOR: f32 = 1.0;
const MINOR: f32 = 0.3;
const BOUND: f32 = 1.6;

fn torus_3d(p: [f32; 3]) -> f32 {
    let radial = (p[0] * p[0] + p[2] * p[2]).sqrt() - MAJOR;
    (radial * radial + p[1] * p[1]).sqrt() - MINOR
}

fn torus_4d(p: [f32; 4]) -> f32 {
    let radial = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - MAJOR;
    (radial * radial + p[3] * p[3]).sqrt() - MINOR
}

// Pappus: `V = 2π²·R·r²`.
fn torus_volume_3d() -> f32 {
    2.0 * std::f32::consts::PI.powi(2) * MAJOR * MINOR * MINOR
}

// Revolving the meridian disk gives `V = 4π²·r²·(R² + r²/4)`.
fn torus_volume_4d() -> f32 {
    4.0 * std::f32::consts::PI.powi(2) * MINOR * MINOR * (MAJOR * MAJOR + 0.25 * MINOR * MINOR)
}

fn main() {
    println!("dim  res  cells  occupied  pieces  cover/solid  margin  extract");
    for resolution in [16usize, 32, 64] {
        let started = Instant::now();
        let volume = Isovolume::extract([-BOUND; 3], [BOUND; 3], resolution, torus_3d);
        let elapsed = started.elapsed();
        println!(
            "3D  {resolution:4} {:7} {:9} {:7} {:11.2} {:8.4} {elapsed:>10.2?}",
            resolution.pow(3),
            volume.occupied_cells(),
            volume.piece_count(),
            volume.volume() / torus_volume_3d(),
            volume.enclosure_margin(),
        );
    }
    for resolution in [12usize, 16, 24] {
        let started = Instant::now();
        let volume = Isovolume::extract([-BOUND; 4], [BOUND; 4], resolution, torus_4d);
        let elapsed = started.elapsed();
        println!(
            "4D  {resolution:4} {:7} {:9} {:7} {:11.2} {:8.4} {elapsed:>10.2?}",
            resolution.pow(4),
            volume.occupied_cells(),
            volume.piece_count(),
            volume.volume() / torus_volume_4d(),
            volume.enclosure_margin(),
        );
    }
}
