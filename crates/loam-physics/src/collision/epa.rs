//! Every threshold is a coefficient times `scale` to the homogeneity degree of
//! the quantity it guards, so depth and normal are scale-equivariant.

use glam::Vec3;

use super::gjk::{minkowski_support, MinkowskiPoint, SupportFn};

const EPA_MAX_ITERATIONS: u32 = 48;
// A well-formed EPA finishes with under 30 vertices; past this is a stall.
const EPA_MAX_VERTICES: usize = 96;

// Support gap per unit of `scale` at which expansion stops. Degree 1.
const EPA_TOLERANCE: f32 = 1e-4;

// Band per unit of `scale` in which a support point retires a face. Degree 1.
const FACE_COPLANAR_EPS: f32 = 1e-5;

// Floor on `|(b−a) × (c−a)|` per `scale`². Degree 2.
const FACE_DEGENERATE_CROSS: f32 = 1e-8;

// Floor on the seed's scalar triple product per `scale`³. Degree 3.
const SEED_DEGENERATE_VOLUME: f32 = 1e-8;

// Floor on a face's edge Gram determinant per `scale`⁴. Degree 4.
const BARYCENTRIC_SINGULAR_EPS: f32 = 1e-12;

#[derive(Clone, Copy)]
struct Thresholds {
    support_gap: f32,
    coplanar_band: f32,
    cross_norm: f32,
    barycentric_gram: f32,
}

impl Thresholds {
    fn for_scale(scale: f32) -> Self {
        Self {
            support_gap: EPA_TOLERANCE * scale,
            coplanar_band: FACE_COPLANAR_EPS * scale,
            cross_norm: FACE_DEGENERATE_CROSS * scale * scale,
            barycentric_gram: BARYCENTRIC_SINGULAR_EPS * scale.powi(4),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ContactInfo {
    /// Unit, world, from A toward B.
    pub normal: Vec3,
    pub penetration: f32,
    /// Midpoint of the closest features on A and B, in world space.
    pub point: Vec3,
}

#[derive(Clone, Copy, Debug)]
struct Face {
    /// Wound so `normal` points outward.
    v: [usize; 3],
    normal: Vec3,
    /// Origin to face plane, always ≥ 0.
    distance: f32,
}

struct Polytope {
    vertices: Vec<MinkowskiPoint>,
    faces: Vec<Face>,
    /// Seed centroid; stays interior under every convex expansion.
    interior: glam::Vec3,
    thresholds: Thresholds,
}

impl Polytope {
    fn from_tetra(tetra: [MinkowskiPoint; 4], thresholds: Thresholds) -> Self {
        let vertices = tetra.to_vec();
        let interior = (tetra[0].point + tetra[1].point + tetra[2].point + tetra[3].point) * 0.25;
        let mut faces = Vec::with_capacity(4);
        for &(i, j, k, l) in &[(0, 1, 2, 3), (0, 3, 1, 2), (0, 2, 3, 1), (1, 3, 2, 0)] {
            faces.push(build_face_vs_point(
                &vertices,
                i,
                j,
                k,
                vertices[l].point,
                thresholds.cross_norm,
            ));
        }
        Self {
            vertices,
            faces,
            interior,
            thresholds,
        }
    }

    fn closest_face(&self) -> Option<usize> {
        let (idx, _) = self
            .faces
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.distance.total_cmp(&b.1.distance))?;
        Some(idx)
    }

    fn expand(&mut self, support: MinkowskiPoint) {
        let new_idx = self.vertices.len();
        self.vertices.push(support);

        let mut horizon: Vec<(usize, usize)> = Vec::new();
        let mut keep = Vec::with_capacity(self.faces.len());

        let coplanar_band = self.thresholds.coplanar_band;
        for f in self.faces.drain(..) {
            let view = support.point - self.vertices[f.v[0]].point;
            if f.normal.dot(view) > -coplanar_band {
                add_or_remove_edge(&mut horizon, f.v[0], f.v[1]);
                add_or_remove_edge(&mut horizon, f.v[1], f.v[2]);
                add_or_remove_edge(&mut horizon, f.v[2], f.v[0]);
            } else {
                keep.push(f);
            }
        }
        self.faces = keep;

        let interior = self.interior;
        let cross_norm = self.thresholds.cross_norm;
        for &(i, j) in &horizon {
            self.faces.push(build_face_vs_point(
                &self.vertices,
                i,
                j,
                new_idx,
                interior,
                cross_norm,
            ));
        }
    }
}

// The caller guarantees `interior_point` is inside the polytope.
fn build_face_vs_point(
    verts: &[MinkowskiPoint],
    a: usize,
    b: usize,
    c: usize,
    interior_point: Vec3,
    cross_norm_floor: f32,
) -> Face {
    let pa = verts[a].point;
    let pb = verts[b].point;
    let pc = verts[c].point;

    let mut normal = (pb - pa).cross(pc - pa);
    let len = normal.length();
    if len < cross_norm_floor {
        normal = Vec3::Y;
    } else {
        normal /= len;
    }

    let to_interior = interior_point - pa;
    let (v_order, outward_normal) = if normal.dot(to_interior) > 0.0 {
        ([a, c, b], -normal)
    } else {
        ([a, b, c], normal)
    };

    let raw_distance = outward_normal.dot(verts[v_order[0]].point);
    let distance = raw_distance.max(0.0);
    Face {
        v: v_order,
        normal: outward_normal,
        distance,
    }
}

// A shared edge appears as `(a, b)` and `(b, a)`; the reverse cancels both.
fn add_or_remove_edge(horizon: &mut Vec<(usize, usize)>, a: usize, b: usize) {
    if let Some(pos) = horizon.iter().position(|&e| e == (b, a)) {
        horizon.swap_remove(pos);
    } else {
        horizon.push((a, b));
    }
}

fn barycentric(a: Vec3, b: Vec3, c: Vec3, p: Vec3, gram_floor: f32) -> (f32, f32, f32) {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < gram_floor {
        return (1.0, 0.0, 0.0);
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    (u, v, w)
}

/// `scale` is the characteristic length of the Minkowski difference: the sum
/// of the two bounding radii.
pub fn epa<A: SupportFn, B: SupportFn>(
    a: &A,
    b: &B,
    initial_simplex: [MinkowskiPoint; 4],
    scale: f32,
) -> Option<ContactInfo> {
    let (polytope, face) = expand_to_boundary(a, b, initial_simplex, scale)?;
    contact_from_face(&polytope, face)
}

// `None` for a degenerate seed, a non-finite support, or a collapsed polytope.
fn expand_to_boundary<A: SupportFn, B: SupportFn>(
    a: &A,
    b: &B,
    initial_simplex: [MinkowskiPoint; 4],
    scale: f32,
) -> Option<(Polytope, Face)> {
    let thresholds = Thresholds::for_scale(scale);

    let p0 = initial_simplex[0].point;
    let p1 = initial_simplex[1].point;
    let p2 = initial_simplex[2].point;
    let p3 = initial_simplex[3].point;
    let volume6 = (p1 - p0).dot((p2 - p0).cross(p3 - p0)).abs();
    if volume6 < SEED_DEGENERATE_VOLUME * scale.powi(3) {
        return None;
    }

    let mut polytope = Polytope::from_tetra(initial_simplex, thresholds);

    for _ in 0..EPA_MAX_ITERATIONS {
        let face_idx = polytope.closest_face()?;
        let face = polytope.faces[face_idx];

        let support = minkowski_support(a, b, face.normal);
        let new_distance = support.point.dot(face.normal);

        if !new_distance.is_finite() || !support.point.is_finite() {
            return None;
        }

        if (new_distance - face.distance).abs() < thresholds.support_gap {
            return Some((polytope, face));
        }

        polytope.expand(support);

        if polytope.vertices.len() > EPA_MAX_VERTICES {
            break;
        }
    }

    tracing::debug!(
        max_iterations = EPA_MAX_ITERATIONS,
        vertices = polytope.vertices.len(),
        "EPA 3D hit iteration cap; returning best-estimate contact",
    );
    let face_idx = polytope.closest_face()?;
    let face = polytope.faces[face_idx];
    Some((polytope, face))
}

fn contact_from_face(polytope: &Polytope, face: Face) -> Option<ContactInfo> {
    let v0 = polytope.vertices[face.v[0]];
    let v1 = polytope.vertices[face.v[1]];
    let v2 = polytope.vertices[face.v[2]];

    let closest = face.normal * face.distance;

    let (u, v, w) = barycentric(
        v0.point,
        v1.point,
        v2.point,
        closest,
        polytope.thresholds.barycentric_gram,
    );

    let point_a = v0.sa * u + v1.sa * v + v2.sa * w;
    let point_b = v0.sb * u + v1.sb * v + v2.sb * w;

    Some(ContactInfo {
        normal: face.normal,
        penetration: face.distance,
        point: (point_a + point_b) * 0.5,
    })
}

#[cfg(test)]
mod tests {
    use super::super::gjk::{gjk_intersect, ConvexHull, GjkResult, Sphere};
    use super::*;

    fn box_vertices(center: Vec3, half: Vec3) -> Vec<Vec3> {
        vec![
            center + Vec3::new(-half.x, -half.y, -half.z),
            center + Vec3::new(half.x, -half.y, -half.z),
            center + Vec3::new(half.x, half.y, -half.z),
            center + Vec3::new(-half.x, half.y, -half.z),
            center + Vec3::new(-half.x, -half.y, half.z),
            center + Vec3::new(half.x, -half.y, half.z),
            center + Vec3::new(half.x, half.y, half.z),
            center + Vec3::new(-half.x, half.y, half.z),
        ]
    }

    fn run(a: &impl SupportFn, b: &impl SupportFn, d: Vec3, scale: f32) -> ContactInfo {
        match gjk_intersect(a, b, d) {
            GjkResult::Intersecting { simplex } => {
                epa(a, b, simplex, scale).expect("EPA should converge")
            }
            GjkResult::Separated => panic!("GJK says separated, EPA can't run"),
        }
    }

    fn box_radius(half: f32) -> f32 {
        half * 3.0_f32.sqrt()
    }

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "expected {a} close to {b} (tol {tol})"
        );
    }

    #[test]
    fn sphere_sphere_penetration_matches_distance() {
        // Two unit-radius spheres with centres 1.5 apart overlap by 0.5.
        let a = Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        };
        let b = Sphere {
            center: Vec3::new(1.5, 0.0, 0.0),
            radius: 1.0,
        };
        let info = run(&a, &b, Vec3::new(1.5, 0.0, 0.0), 2.0);
        assert_close(info.penetration, 0.5, 1e-3);
        assert!(info.normal.dot(Vec3::X) > 0.99, "normal: {:?}", info.normal);
    }

    #[test]
    fn box_box_axis_aligned_overlap_penetration_matches_axis() {
        // Unit boxes offset 1.5 along x overlap by 0.5.
        let va = box_vertices(Vec3::ZERO, Vec3::ONE);
        let vb = box_vertices(Vec3::new(1.5, 0.0, 0.0), Vec3::ONE);
        let a = ConvexHull { vertices: &va };
        let b = ConvexHull { vertices: &vb };

        let info = run(&a, &b, Vec3::new(1.5, 0.0, 0.0), 2.0 * box_radius(1.0));
        assert!(
            info.normal.dot(Vec3::X) > 0.99,
            "normal must run from A toward B along +x, got {:?}",
            info.normal
        );
        assert_close(info.penetration, 0.5, 1e-3);
    }

    #[test]
    fn sphere_box_corner_penetration_points_outward() {
        // Corner-centre distance √(3·0.04) ≈ 0.346, so penetration ≈ 0.154.
        let vb = box_vertices(Vec3::ZERO, Vec3::ONE);
        let b = ConvexHull { vertices: &vb };
        let s = Sphere {
            center: Vec3::new(1.2, 1.2, 1.2),
            radius: 0.5,
        };

        let info = run(&b, &s, Vec3::new(1.0, 1.0, 1.0), box_radius(1.0) + 0.5);
        let expected = Vec3::new(1.0, 1.0, 1.0).normalize();
        assert!(
            info.normal.dot(expected) > 0.95,
            "normal {:?} not aligned with {:?}",
            info.normal,
            expected
        );
        assert_close(info.penetration, 0.5 - 3.0_f32.sqrt() * 0.2, 1e-2);
    }

    const SPHERE_PAIR_SCALE: f32 = 2.0;

    fn seed(points: [Vec3; 4]) -> [MinkowskiPoint; 4] {
        points.map(|point| MinkowskiPoint {
            point,
            sa: point,
            sb: Vec3::ZERO,
        })
    }

    // `|det| = 4·h` against a floor of `SEED_DEGENERATE_VOLUME·SPHERE_PAIR_SCALE³` = 8e-8.
    fn seed_of_height(h: f32) -> [MinkowskiPoint; 4] {
        seed([
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, h),
        ])
    }

    #[test]
    fn seeds_across_the_volume_floor_resolve_finitely_or_not_at_all() {
        let a = Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        };
        let b = Sphere {
            center: Vec3::new(0.5, 0.0, 0.0),
            radius: 1.0,
        };

        assert!(
            epa(&a, &b, seed_of_height(0.0), SPHERE_PAIR_SCALE).is_none(),
            "a seed with no 3-volume has no interior to orient against"
        );
        for h in [1e-6, 1e-4, 1e-2, 1.0] {
            let contact = epa(&a, &b, seed_of_height(h), SPHERE_PAIR_SCALE)
                .unwrap_or_else(|| panic!("seed of height {h} clears the volume floor"));
            assert!(
                contact.normal.is_finite()
                    && contact.point.is_finite()
                    && contact.penetration.is_finite(),
                "seed of height {h} resolved to {contact:?}"
            );
            assert!(
                contact.penetration >= 0.0,
                "seed of height {h}: negative depth"
            );
        }
    }

    #[test]
    fn the_seed_volume_floor_admits_the_same_seed_shapes_at_every_size() {
        fn resized(seed: [MinkowskiPoint; 4], s: f32) -> [MinkowskiPoint; 4] {
            seed.map(|p| MinkowskiPoint {
                point: p.point * s,
                sa: p.sa * s,
                sb: p.sb * s,
            })
        }

        for s in [1e-3_f32, 1.0, 1e3] {
            let a = Sphere {
                center: Vec3::ZERO,
                radius: s,
            };
            let b = Sphere {
                center: Vec3::new(0.5 * s, 0.0, 0.0),
                radius: s,
            };
            let scale = SPHERE_PAIR_SCALE * s;
            assert!(
                epa(&a, &b, resized(seed_of_height(1e-9), s), scale).is_none(),
                "at size {s} a seed 20x under the volume floor resolved a contact"
            );
            assert!(
                epa(&a, &b, resized(seed_of_height(1e-6), s), scale).is_some(),
                "at size {s} a seed 50x over the volume floor resolved to nothing"
            );
        }
    }

    #[test]
    fn collinear_and_repeated_vertex_seeds_are_rejected_rather_than_resolved() {
        let a = Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        };
        let b = Sphere {
            center: Vec3::new(0.5, 0.0, 0.0),
            radius: 1.0,
        };
        let collinear = seed([
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        ]);
        let repeated = seed([
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(0.0, 1.0, 1.0),
        ]);
        assert!(epa(&a, &b, collinear, SPHERE_PAIR_SCALE).is_none());
        assert!(epa(&a, &b, repeated, SPHERE_PAIR_SCALE).is_none());
    }

    #[test]
    fn deeply_nested_boxes_contact_matches_shallowest_axis() {
        let va = box_vertices(Vec3::ZERO, Vec3::ONE);
        let vb = box_vertices(Vec3::new(0.3, 0.1, 0.2), Vec3::ONE);
        let a = ConvexHull { vertices: &va };
        let b = ConvexHull { vertices: &vb };
        let info = run(&a, &b, Vec3::new(0.3, 0.1, 0.2), 2.0 * box_radius(1.0));
        assert_close(info.penetration, 2.0 - 0.3, EPA_TOLERANCE);
        assert!(
            info.normal.dot(Vec3::X) > 0.999,
            "normal must run from A toward B along +x, got {:?}",
            info.normal
        );
    }

    #[test]
    fn flush_box_contacts_resolve_against_a_difference_body_facet() {
        let cases: [(f32, Vec3, &[Vec3]); 5] = [
            (1.0, Vec3::new(0.1, 0.1, 0.0), &[Vec3::X, Vec3::Y]),
            (1.0, Vec3::new(0.1, 0.05, 0.0), &[Vec3::X]),
            (1.0, Vec3::new(0.0, 0.05, 0.1), &[Vec3::Z]),
            (0.1, Vec3::new(0.002, 0.0, 0.0), &[Vec3::X]),
            (0.1, Vec3::new(0.0, 0.0, -0.002), &[Vec3::NEG_Z]),
        ];

        for (half, offset, admissible) in cases {
            let va = box_vertices(Vec3::ZERO, Vec3::splat(half));
            let vb = box_vertices(offset, Vec3::splat(half));
            let a = ConvexHull { vertices: &va };
            let b = ConvexHull { vertices: &vb };

            let info = run(&a, &b, offset, 2.0 * box_radius(half));
            assert_close(
                info.penetration,
                2.0 * half - offset.abs().max_element(),
                EPA_TOLERANCE,
            );
            let alignment = admissible.iter().fold(f32::NEG_INFINITY, |best, &axis| {
                best.max(info.normal.dot(axis))
            });
            assert!(
                alignment > 0.999,
                "half {half}, offset {offset:?}: normal {:?} on none of {admissible:?}",
                info.normal
            );
        }
    }

    #[test]
    fn two_axis_flush_expansion_terminates_with_a_bounded_face_count() {
        let offset = Vec3::new(0.0, 0.0, -0.01);
        let va = box_vertices(Vec3::ZERO, Vec3::splat(0.1));
        let vb = box_vertices(offset, Vec3::splat(0.1));
        let a = ConvexHull { vertices: &va };
        let b = ConvexHull { vertices: &vb };

        let simplex = match gjk_intersect(&a, &b, offset) {
            GjkResult::Intersecting { simplex } => simplex,
            GjkResult::Separated => panic!("boxes overlap; GJK must report intersecting"),
        };
        let (polytope, face) = expand_to_boundary(&a, &b, simplex, 2.0 * box_radius(0.1))
            .expect("EPA should converge");

        assert!(
            polytope.faces.len() <= 24,
            "expansion left {} faces",
            polytope.faces.len()
        );

        let info = contact_from_face(&polytope, face).expect("a converged face yields a contact");
        assert_close(info.penetration, 0.19, EPA_TOLERANCE);
        assert!(
            info.normal.dot(Vec3::NEG_Z) > 0.999,
            "normal must run from A toward B along -z, got {:?}",
            info.normal
        );
    }

    #[test]
    fn epa_contact_is_equivariant_under_uniform_scaling() {
        fn scaled(simplex: [MinkowskiPoint; 4], s: f32) -> [MinkowskiPoint; 4] {
            simplex.map(|p| MinkowskiPoint {
                point: p.point * s,
                sa: p.sa * s,
                sb: p.sb * s,
            })
        }

        fn seed_at_unit_scale(
            a: &impl SupportFn,
            b: &impl SupportFn,
            d: Vec3,
        ) -> [MinkowskiPoint; 4] {
            match gjk_intersect(a, b, d) {
                GjkResult::Intersecting { simplex } => simplex,
                GjkResult::Separated => panic!("unit-scale fixture must overlap"),
            }
        }

        let box_offsets = [Vec3::new(0.3, 0.1, 0.2), Vec3::new(0.1, 0.05, 0.0)];
        let unit_seeds: Vec<[MinkowskiPoint; 4]> = box_offsets
            .iter()
            .map(|&offset| {
                let va = box_vertices(Vec3::ZERO, Vec3::ONE);
                let vb = box_vertices(offset, Vec3::ONE);
                seed_at_unit_scale(
                    &ConvexHull { vertices: &va },
                    &ConvexHull { vertices: &vb },
                    offset,
                )
            })
            .collect();
        let sphere_seed = seed_at_unit_scale(
            &Sphere {
                center: Vec3::ZERO,
                radius: 1.0,
            },
            &Sphere {
                center: Vec3::new(1.5, 0.0, 0.0),
                radius: 1.0,
            },
            Vec3::X,
        );

        for s in [1e-3_f32, 1e-2, 1.0, 1e2, 1e3] {
            let sphere_a = Sphere {
                center: Vec3::ZERO,
                radius: s,
            };
            let sphere_b = Sphere {
                center: Vec3::new(1.5 * s, 0.0, 0.0),
                radius: s,
            };
            let scale = 2.0 * s;
            let contact = epa(&sphere_a, &sphere_b, scaled(sphere_seed, s), scale)
                .unwrap_or_else(|| panic!("spheres at scale {s} resolved to no contact"));
            assert!(
                (contact.penetration - 0.5 * s).abs() <= EPA_TOLERANCE * scale,
                "spheres at scale {s}: depth {} is not {}",
                contact.penetration,
                0.5 * s
            );
            assert!(
                contact.normal.dot(Vec3::X) > 0.999,
                "spheres at scale {s}: normal {:?}",
                contact.normal
            );

            for (&offset, &unit_seed) in box_offsets.iter().zip(unit_seeds.iter()) {
                let va = box_vertices(Vec3::ZERO, Vec3::splat(s));
                let vb = box_vertices(offset * s, Vec3::splat(s));
                let a = ConvexHull { vertices: &va };
                let b = ConvexHull { vertices: &vb };
                let scale = 2.0 * box_radius(s);
                let contact = epa(&a, &b, scaled(unit_seed, s), scale)
                    .unwrap_or_else(|| panic!("boxes {offset:?} at scale {s}: no contact"));
                let want = (2.0 - offset.abs().max_element()) * s;
                assert!(
                    (contact.penetration - want).abs() <= EPA_TOLERANCE * scale,
                    "boxes {offset:?} at scale {s}: depth {} is not {want}",
                    contact.penetration
                );
                assert!(
                    contact.normal.dot(Vec3::X) > 0.999,
                    "boxes {offset:?} at scale {s}: normal {:?}",
                    contact.normal
                );
            }
        }
    }
}
