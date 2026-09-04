//! `SphericalS3Embedded` isometry contracts, exercised against real polychora.

use glam::{Mat4, Vec4};
use loam_math::{
    Bivector, Bivector4, Iso4, IsometryGroup, Rotor, Rotor4, Space, SphericalS3Embedded,
};
use loam_shape::polytope::Polytope4;

const TEST_ANGLE: f32 = 0.7;

/// Equal angles in two orthogonal planes: the isoclinic (Clifford) generator.
fn clifford_generator(theta: f32) -> Bivector4 {
    Bivector4::new(theta, 0.0, 0.0, 0.0, 0.0, theta)
}

/// `Rotor4::to_mat4` is column-major to match glam.
fn iso_from_rotor(rotor: Rotor4) -> Iso4 {
    Iso4 {
        matrix: Mat4::from_cols_array_2d(&rotor.to_mat4()),
    }
}

fn pose_vertices(iso: Iso4, source: &[Vec4], out: &mut Vec<Vec4>) {
    out.clear();
    out.extend(
        source
            .iter()
            .map(|&v| SphericalS3Embedded.iso_apply(iso, v)),
    );
}

fn posed_at(polytope: Polytope4, angle: f32) -> Vec<Vec4> {
    let iso = iso_from_rotor(clifford_generator(angle).exp());
    let mut out = Vec::new();
    pose_vertices(iso, polytope.topology().vertices, &mut out);
    out
}

#[test]
fn the_isoclinic_rotor_displaces_every_vertex_equally_and_a_simple_one_does_not() {
    let vertices = Polytope4::Cell600.topology().vertices;
    let posed = posed_at(Polytope4::Cell600, TEST_ANGLE);
    let mut spread = 0.0_f32;
    for (v, p) in vertices.iter().zip(&posed) {
        let d = SphericalS3Embedded.distance(*v, *p);
        assert!(
            (d - TEST_ANGLE).abs() < 1e-4,
            "isoclinic displacement {d} should equal the generator angle {TEST_ANGLE}"
        );
        spread = spread.max((d - TEST_ANGLE).abs());
    }
    assert!(spread < 1e-4, "displacement spread {spread} must vanish");

    let simple = iso_from_rotor(Bivector4::new(TEST_ANGLE, 0.0, 0.0, 0.0, 0.0, 0.0).exp());
    let mut simple_posed = Vec::new();
    pose_vertices(simple, vertices, &mut simple_posed);
    let displacements: Vec<f32> = vertices
        .iter()
        .zip(&simple_posed)
        .map(|(v, p)| SphericalS3Embedded.distance(*v, *p))
        .collect();
    let lo = displacements.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = displacements
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        hi - lo > 0.1,
        "a simple rotation must displace vertices unequally, got spread {}",
        hi - lo
    );
}

#[test]
fn the_space_isometry_and_the_rotor_sandwich_agree_on_every_vertex() {
    let rotor = clifford_generator(TEST_ANGLE).exp();
    let iso = iso_from_rotor(rotor);
    for &v in Polytope4::Cell24.topology().vertices {
        let by_space = SphericalS3Embedded.iso_apply(iso, v);
        let by_rotor = Rotor::apply(&rotor, v);
        assert!(
            (by_space - by_rotor).length() < 1e-5,
            "space isometry {by_space:?} should match rotor sandwich {by_rotor:?}"
        );
    }
}

#[test]
fn posed_vertices_stay_on_the_unit_three_sphere_at_every_angle() {
    for step in 0..257 {
        let angle = step as f32 * 0.0245;
        for p in posed_at(Polytope4::Cell24, angle) {
            assert!(
                (p.length() - 1.0).abs() < 1e-5,
                "posed vertex off S³ at angle {angle}: |p| = {}",
                p.length()
            );
        }
    }
}

#[test]
fn the_pose_preserves_every_pairwise_separation() {
    // Antipodal pairs sit on the cut locus.
    const CUT_LOCUS_MARGIN: f32 = 0.05;
    let vertices = Polytope4::Cell24.topology().vertices;
    let posed = posed_at(Polytope4::Cell24, TEST_ANGLE);
    let mut cut_locus_pairs = 0usize;
    for i in 0..vertices.len() {
        for j in (i + 1)..vertices.len() {
            let dot_before = vertices[i].dot(vertices[j]);
            let dot_after = posed[i].dot(posed[j]);
            assert!(
                (dot_before - dot_after).abs() < 1e-5,
                "inner product {dot_before} -> {dot_after} for pair ({i}, {j})"
            );
            let before = SphericalS3Embedded.distance(vertices[i], vertices[j]);
            if before > std::f32::consts::PI - CUT_LOCUS_MARGIN {
                cut_locus_pairs += 1;
                continue;
            }
            let after = SphericalS3Embedded.distance(posed[i], posed[j]);
            assert!(
                (before - after).abs() < 1e-4,
                "separation {before} -> {after} for pair ({i}, {j})"
            );
        }
    }
    assert!(
        cut_locus_pairs > 0,
        "the 24-cell has antipodal pairs; skipping none means the margin is wrong"
    );
}
