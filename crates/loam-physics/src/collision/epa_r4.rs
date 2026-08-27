//! Every geometric threshold here is a dimensionless coefficient times the
//! caller's `scale` raised to the homogeneity degree of the quantity it
//! guards, so depth and normal are equivariant under a uniform scaling of both
//! bodies. The contact point is not, and cannot be: a facet tiled by several
//! coplanar tetra offers one witness per tile, and which tile terminates the
//! expansion is not a similarity invariant.

use glam::Vec4;

use super::gjk_r4::{minkowski_support_r4, MinkowskiPoint4, SupportFn4};
use super::simplex_r4::project_origin_onto_affine_hull;

const EPA_MAX_ITERATIONS: u32 = 96;
const EPA_MAX_VERTICES: usize = 192;

// Support gap at which the expansion has reached the boundary, per unit of
// `scale`. Degree 1: the compared quantity is the distance between two planes
// with the same unit normal. Held absolute it buys a fixed number of world
// units of depth accuracy rather than a fixed number of digits.
const EPA_TOLERANCE: f32 = 1e-3;

#[derive(Clone, Copy, Debug)]
pub struct ContactInfo4 {
    pub normal: Vec4,
    pub penetration: f32,
    pub point: Vec4,
}

#[derive(Clone, Copy, Debug)]
struct Face4 {
    v: [usize; 4],
    normal: Vec4,
    distance: f32,
}

#[derive(Clone, Copy)]
struct Thresholds {
    support_gap: f32,
    coplanar_band: f32,
    wedge_norm: f32,
}

impl Thresholds {
    fn for_scale(scale: f32) -> Self {
        Self {
            support_gap: EPA_TOLERANCE * scale,
            coplanar_band: FACE_COPLANAR_EPS * scale,
            wedge_norm: FACE_DEGENERATE_WEDGE * scale.powi(3),
        }
    }
}

struct Polytope4 {
    vertices: Vec<MinkowskiPoint4>,
    faces: Vec<Face4>,
    /// Centroid of the seed 5-simplex. Expansion only adds vertices, so it
    /// stays interior and every face can orient against it.
    centroid: Vec4,
    thresholds: Thresholds,
}

impl Polytope4 {
    fn from_simplex(simplex: [MinkowskiPoint4; 5], thresholds: Thresholds) -> Self {
        let vertices = simplex.to_vec();
        let centroid = (simplex[0].point
            + simplex[1].point
            + simplex[2].point
            + simplex[3].point
            + simplex[4].point)
            * 0.2;

        let mut faces = Vec::with_capacity(5);
        for l in 0..5 {
            let mut tet = [0usize; 4];
            let mut idx = 0;
            for i in 0..5 {
                if i != l {
                    tet[idx] = i;
                    idx += 1;
                }
            }
            if let Some(face) = build_face(
                &vertices,
                tet[0],
                tet[1],
                tet[2],
                tet[3],
                centroid,
                thresholds.wedge_norm,
            ) {
                faces.push(face);
            }
        }
        Self {
            vertices,
            faces,
            centroid,
            thresholds,
        }
    }

    // Distance-0 faces are the rule in 4D: `gjk_intersect_r4` grows a
    // sub-simplex of fewer than five vertices by adding supports on one side,
    // leaving the origin on a proper face of the seed. They compete for the
    // minimum on equal terms; skipping them converges on a far facet whose
    // normal points through the obstacle. `FACE_COPLANAR_EPS` retires the
    // coplanar tiles, which is what makes expansion across them terminate.
    fn closest_face(&self) -> Option<usize> {
        self.faces
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.distance.total_cmp(&b.1.distance))
            .map(|(idx, _)| idx)
    }

    fn expand(&mut self, support: MinkowskiPoint4) {
        let new_idx = self.vertices.len();
        self.vertices.push(support);

        let mut horizon: Vec<Triangle> = Vec::new();
        let mut keep = Vec::with_capacity(self.faces.len());

        let coplanar_band = self.thresholds.coplanar_band;
        for f in self.faces.drain(..) {
            let view = support.point - self.vertices[f.v[0]].point;
            if f.normal.dot(view) > -coplanar_band {
                for tri in tet_triangles(&f.v) {
                    add_or_remove_triangle(&mut horizon, tri);
                }
            } else {
                keep.push(f);
            }
        }
        self.faces = keep;

        let centroid = self.centroid;
        let wedge_norm = self.thresholds.wedge_norm;
        for tri in &horizon {
            if let Some(face) = build_face(
                &self.vertices,
                tri.0,
                tri.1,
                tri.2,
                new_idx,
                centroid,
                wedge_norm,
            ) {
                self.faces.push(face);
            }
        }
    }
}

// Band around a face plane, per unit of `scale`, in which a support point
// counts as lying on it, so `expand` retires the face. A facet of a Minkowski
// difference is generally not a simplex, so it is tiled by coplanar tetra and
// support points land exactly on their shared plane; keeping one tile while
// its neighbours retire makes the surface non-convex and EPA can return a
// normal pointing from B toward A. The plane residual of a 4-term dot product
// is a few ULPs of the operands (~4e-7 per unit of `scale`; 1e-7 measurably
// misses coplanar tiles), against `EPA_TOLERANCE` two orders up.
//
// Degree 1: a distance along a unit normal. Held absolute, at circumradius 1e3
// the pentatope resolves 216.7 against an exact 746.8, and the 600-cell and
// 120-cell collapse to zero depth on a zero normal.
const FACE_COPLANAR_EPS: f32 = 1e-5;

// Floor on the wedge of a face tetra's three edge vectors, per `scale` cubed,
// below which the Hodge dual of three dependent edges normalizes to NaN and
// the NaN reaches the solver as a contact.
//
// Degree 3: the wedge of three vectors is a 3-volume. Held absolute it is the
// second guard to fire on the small side, behind [`SEED_DEGENERATE_VOLUME`].
const FACE_DEGENERATE_WEDGE: f32 = 1e-8;

// Floor on the seed 5-simplex's 4-volume, per `scale` to the fourth, below
// which the seed lies in a hyperplane and has no interior for `build_face` to
// orient against.
//
// Degree 4: `det4` of four edge vectors is a 4-volume, so it loses four
// decades of headroom per decade of shrinkage. Held absolute it is the first
// guard to fire on the small side, and the reason all six polychora fixtures
// return `None` at circumradius 1e-3 and resolve at 1e-2.
const SEED_DEGENERATE_VOLUME: f32 = 1e-8;

type Triangle = (usize, usize, usize);

// Winding is irrelevant: matching is order-insensitive.
fn tet_triangles(tet: &[usize; 4]) -> [Triangle; 4] {
    let (a, b, c, d) = (tet[0], tet[1], tet[2], tet[3]);
    [(a, b, c), (a, b, d), (a, c, d), (b, c, d)]
}

// A triangle shared by two removed tetra is interior, so a second occurrence
// cancels the first.
fn add_or_remove_triangle(horizon: &mut Vec<Triangle>, tri: Triangle) {
    let key = sort_triangle(tri);
    if let Some(pos) = horizon.iter().position(|t| sort_triangle(*t) == key) {
        horizon.swap_remove(pos);
    } else {
        horizon.push(tri);
    }
}

fn sort_triangle(t: Triangle) -> (usize, usize, usize) {
    let mut a = [t.0, t.1, t.2];
    a.sort_unstable();
    (a[0], a[1], a[2])
}

// Orientation comes from the seed centroid alone. The origin is interior only
// while GJK's containment verdict holds exactly, and GJK accepts a simplex
// whose closest point is within its own tolerance of the origin, so on a
// near-tangency orienting against the origin inverts a face and EPA converges
// on a plane that is not the boundary.
fn build_face(
    verts: &[MinkowskiPoint4],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    centroid: Vec4,
    wedge_norm_floor: f32,
) -> Option<Face4> {
    let pa = verts[a].point;
    let pb = verts[b].point;
    let pc = verts[c].point;
    let pd = verts[d].point;

    let raw_normal = hodge_dual_of_trivector_wedge(pb - pa, pc - pa, pd - pa);
    let len = raw_normal.length();
    if len < wedge_norm_floor {
        return None;
    }
    let normal = raw_normal / len;

    // Interior on the `+normal` side means `+normal` points inward.
    let flip = normal.dot(centroid - pa) > 0.0;

    let (outward, v_order) = if flip {
        (-normal, [a, b, d, c])
    } else {
        (normal, [a, b, c, d])
    };

    // Clamp at 0: on a near-tangency the origin can sit a hair outside the
    // face, and a negative offset is not a penetration depth.
    let distance = outward.dot(pa).max(0.0);

    Some(Face4 {
        v: v_order,
        normal: outward,
        distance,
    })
}

// Hodge dual of `u ∧ v ∧ w`, mapping `e_123 -> −e_4, e_124 -> +e_3,
// e_134 -> −e_2, e_234 -> +e_1`.
fn hodge_dual_of_trivector_wedge(u: Vec4, v: Vec4, w: Vec4) -> Vec4 {
    // t_ijk = det of (u, v, w) on columns (i, j, k).
    let t_234 = det3(u.y, u.z, u.w, v.y, v.z, v.w, w.y, w.z, w.w);
    let t_134 = det3(u.x, u.z, u.w, v.x, v.z, v.w, w.x, w.z, w.w);
    let t_124 = det3(u.x, u.y, u.w, v.x, v.y, v.w, w.x, w.y, w.w);
    let t_123 = det3(u.x, u.y, u.z, v.x, v.y, v.z, w.x, w.y, w.z);

    Vec4::new(t_234, -t_134, t_124, -t_123)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn det3(
    a00: f32,
    a01: f32,
    a02: f32,
    a10: f32,
    a11: f32,
    a12: f32,
    a20: f32,
    a21: f32,
    a22: f32,
) -> f32 {
    a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)
}

/// `initial_simplex` is GJK's terminating 5-simplex. `scale` is the
/// characteristic length of the Minkowski difference: the caller owes the sum
/// of the two bounding radii, and a `scale` that does not track the geometry
/// narrows or widens all four thresholds at once.
pub fn epa_r4<A: SupportFn4, B: SupportFn4>(
    a: &A,
    b: &B,
    initial_simplex: [MinkowskiPoint4; 5],
    scale: f32,
) -> Option<ContactInfo4> {
    let thresholds = Thresholds::for_scale(scale);

    let p0 = initial_simplex[0].point;
    let d1 = initial_simplex[1].point - p0;
    let d2 = initial_simplex[2].point - p0;
    let d3 = initial_simplex[3].point - p0;
    let d4 = initial_simplex[4].point - p0;
    let volume = det4(d1, d2, d3, d4).abs();
    if volume < SEED_DEGENERATE_VOLUME * scale.powi(4) {
        return None;
    }

    let mut polytope = Polytope4::from_simplex(initial_simplex, thresholds);

    for _ in 0..EPA_MAX_ITERATIONS {
        let face_idx = polytope.closest_face()?;
        let face = polytope.faces[face_idx];

        let support = minkowski_support_r4(a, b, face.normal);
        let new_distance = support.point.dot(face.normal);

        if !new_distance.is_finite() || !support.point.is_finite() {
            return None;
        }

        if (new_distance - face.distance).abs() < thresholds.support_gap {
            return contact_from_face(&polytope, face);
        }

        polytope.expand(support);
        if polytope.vertices.len() > EPA_MAX_VERTICES {
            break;
        }
    }

    tracing::debug!(
        max_iterations = EPA_MAX_ITERATIONS,
        vertices = polytope.vertices.len(),
        "EPA 4D hit iteration cap; returning best-estimate contact",
    );
    let face_idx = polytope.closest_face()?;
    contact_from_face(&polytope, polytope.faces[face_idx])
}

// Laplace expansion on the first row.
fn det4(r0: Vec4, r1: Vec4, r2: Vec4, r3: Vec4) -> f32 {
    r0.x * det3(r1.y, r1.z, r1.w, r2.y, r2.z, r2.w, r3.y, r3.z, r3.w)
        - r0.y * det3(r1.x, r1.z, r1.w, r2.x, r2.z, r2.w, r3.x, r3.z, r3.w)
        + r0.z * det3(r1.x, r1.y, r1.w, r2.x, r2.y, r2.w, r3.x, r3.y, r3.w)
        - r0.w * det3(r1.x, r1.y, r1.z, r2.x, r2.y, r2.z, r3.x, r3.y, r3.z)
}

fn contact_from_face(polytope: &Polytope4, face: Face4) -> Option<ContactInfo4> {
    let tetra = face.v.map(|i| polytope.vertices[i]);

    let closest = face.normal * face.distance;
    let weights = face_barycentrics(&tetra.map(|p| p.point), closest);

    let mut point_a = Vec4::ZERO;
    let mut point_b = Vec4::ZERO;
    for (vertex, w) in tetra.iter().zip(weights) {
        point_a += vertex.sa * w;
        point_b += vertex.sb * w;
    }

    Some(ContactInfo4 {
        normal: face.normal,
        penetration: face.distance,
        point: (point_a + point_b) * 0.5,
    })
}

// Affine, not convex: `Σ wᵢ·vᵢ = closest` is what makes the witnesses satisfy
// `point_a − point_b = normal·penetration`, and clamping to this tile as
// `simplex_r4::closest_to_origin` does would drop vertices and break it.
// Weights go negative when the projection lands on a neighbouring coplanar
// tile (see `FACE_COPLANAR_EPS`), leaving both witnesses off their own hulls;
// the contact point feeds a lever arm and a merge radius, which want the
// combination that reproduces `closest`. The centroid fallback covers a
// singular Gram system, reachable only for a face anisotropic enough to lose a
// pivot in f32.
fn face_barycentrics(points: &[Vec4; 4], closest: Vec4) -> [f32; 4] {
    let shifted = points.map(|p| p - closest);
    match project_origin_onto_affine_hull(&[0, 1, 2, 3], &shifted) {
        Some((_, w)) => [w[0], w[1], w[2], w[3]],
        None => [0.25; 4],
    }
}

#[cfg(test)]
mod tests {
    use super::super::gjk_r4::{gjk_intersect_r4, ConvexHull4, GjkResult4, Sphere4};
    use super::*;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "{a} not close to {b} (tol {tol}, diff {})",
            (a - b).abs()
        );
    }

    const UNIT_POLYCHORON_SCALE: f32 = 2.0;

    #[test]
    fn sphere_sphere_penetration_matches_analytical() {
        // Centers 0.8 apart, radius 0.5 each: penetration = 1.0 − 0.8 = 0.2.
        let a = Sphere4 {
            center: Vec4::new(0.0, 0.0, 0.0, 0.0),
            radius: 0.5,
        };
        let b = Sphere4 {
            center: Vec4::new(0.8, 0.0, 0.0, 0.0),
            radius: 0.5,
        };
        let simplex = match gjk_intersect_r4(&a, &b, Vec4::X) {
            GjkResult4::Intersecting { simplex } => simplex,
            _ => panic!("spheres should overlap"),
        };
        // Two radius-0.5 spheres: characteristic length 1.0.
        let contact = epa_r4(&a, &b, simplex, 1.0).expect("EPA should succeed");
        assert_close(contact.penetration, 0.2, 5e-3);
        assert!(
            contact.normal.dot(Vec4::X) > 0.99,
            "normal must run from A toward B along +x, got {:?}",
            contact.normal
        );
    }

    #[test]
    fn sphere_sphere_penetration_moderate() {
        // Analytical depth 0.5.
        let a = Sphere4 {
            center: Vec4::ZERO,
            radius: 0.5,
        };
        let b = Sphere4 {
            center: Vec4::new(0.5, 0.0, 0.0, 0.0),
            radius: 0.5,
        };
        let simplex = match gjk_intersect_r4(&a, &b, Vec4::X) {
            GjkResult4::Intersecting { simplex } => simplex,
            _ => panic!("spheres should overlap"),
        };
        let contact = epa_r4(&a, &b, simplex, 1.0).expect("EPA should succeed");
        assert_close(contact.penetration, 0.5, 5e-2);
        assert!(
            contact.normal.dot(Vec4::X) > 0.99,
            "normal must run from A toward B along +x, got {:?}",
            contact.normal
        );
    }

    #[test]
    fn sphere_sphere_contact_point_between_centers() {
        let a = Sphere4 {
            center: Vec4::ZERO,
            radius: 0.5,
        };
        let b = Sphere4 {
            center: Vec4::new(0.8, 0.0, 0.0, 0.0),
            radius: 0.5,
        };
        let simplex = match gjk_intersect_r4(&a, &b, Vec4::X) {
            GjkResult4::Intersecting { simplex } => simplex,
            _ => panic!("spheres should overlap"),
        };
        let contact = epa_r4(&a, &b, simplex, 1.0).expect("EPA should succeed");
        assert!(contact.point.y.abs() < 0.1);
        assert!(contact.point.z.abs() < 0.1);
        assert!(contact.point.w.abs() < 0.1);
    }

    #[test]
    fn contact_from_face_realizes_the_plane_projection_outside_the_tetra() {
        use super::super::simplex_r4::closest_to_origin;

        // Tetra in the hyperplane x = 1, so the plane projection is x̂. The
        // affine coordinates below are not all non-negative: x̂ is outside it.
        let points = [
            Vec4::new(1.0, 1.0, 0.0, 0.0),
            Vec4::new(1.0, 0.0, 1.0, 0.0),
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            Vec4::new(1.0, 6.0, -3.0, 2.0),
        ];
        let pre_images_b = [
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
            Vec4::new(1.0, 0.0, 0.0, 0.0),
        ];
        let vertices: Vec<MinkowskiPoint4> = points
            .iter()
            .zip(pre_images_b)
            .map(|(&point, sb)| MinkowskiPoint4 {
                point,
                sa: point + sb,
                sb,
            })
            .collect();

        // Fixture premise: the convex solve clamps to the triangle v₀v₁v₂ and
        // drops v₃ from the combination.
        let clamped = closest_to_origin(&points.map(|p| p - Vec4::X));
        assert_eq!(clamped.kept, vec![0, 1, 2]);

        // Affine coordinates of x̂ on the tetra. In the yzw slice the first
        // three vertices are the standard basis and the fourth is
        // q₃ = (6, −3, 2), so Σ wᵢ·qᵢ = 0 forces (w₀, w₁, w₂) = −w₃·q₃, and
        // Σ wᵢ = 1 then gives −4·w₃ = 1.
        let weights = [1.5, -0.75, 0.5, -0.25];
        let realized = points
            .iter()
            .zip(weights)
            .fold(Vec4::ZERO, |acc, (&p, w)| acc + p * w);
        assert!((realized - Vec4::X).length() < 1e-6, "{realized:?}");

        let solved = face_barycentrics(&points, Vec4::X);
        assert!(
            solved
                .iter()
                .zip(weights)
                .all(|(&got, want)| (got - want).abs() < 1e-5),
            "weights {solved:?} should be {weights:?}"
        );

        let centroid = (points[0] + points[1] + points[2] + points[3]) * 0.25;
        let thresholds = Thresholds::for_scale(1.0);
        let face = build_face(&vertices, 0, 1, 2, 3, centroid, thresholds.wedge_norm)
            .expect("tetra is non-degenerate");
        assert_close(face.distance, 1.0, 1e-6);

        let polytope = Polytope4 {
            vertices,
            faces: vec![face],
            centroid,
            thresholds,
        };
        let contact = contact_from_face(&polytope, face).expect("face resolves a contact");

        // Σ wᵢ·saᵢ = (0.75, 1.5, −0.75, 0.5) and Σ wᵢ·sbᵢ = (−0.25, 1.5,
        // −0.75, 0.5): the witnesses differ by x̂ = normal·penetration, and the
        // contact is their midpoint.
        let expected = Vec4::new(0.25, 1.5, -0.75, 0.5);
        assert!(
            (contact.point - expected).length() < 1e-5,
            "contact {:?} should be {expected:?}",
            contact.point
        );
    }

    const SPHERE_PAIR_SCALE: f32 = 2.0;

    fn seed(points: [Vec4; 5]) -> [MinkowskiPoint4; 5] {
        points.map(|point| MinkowskiPoint4 {
            point,
            sa: point,
            sb: Vec4::ZERO,
        })
    }

    // Base tetrahedron around the origin in `w = 0` plus an apex at height
    // `h`, so `|det| = 8·h` against a volume floor of
    // `SEED_DEGENERATE_VOLUME·SPHERE_PAIR_SCALE⁴` = 1.6e-7.
    fn seed_of_height(h: f32) -> [MinkowskiPoint4; 5] {
        seed([
            Vec4::new(-1.0, -1.0, -1.0, 0.0),
            Vec4::new(1.0, -1.0, -1.0, 0.0),
            Vec4::new(0.0, 1.0, -1.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, h),
        ])
    }

    #[test]
    fn seeds_across_the_volume_floor_resolve_finitely_or_not_at_all() {
        let a = Sphere4 {
            center: Vec4::ZERO,
            radius: 1.0,
        };
        let b = Sphere4 {
            center: Vec4::new(0.5, 0.0, 0.0, 0.0),
            radius: 1.0,
        };

        assert!(
            epa_r4(&a, &b, seed_of_height(0.0), SPHERE_PAIR_SCALE).is_none(),
            "a seed inside a hyperplane has no interior to orient against"
        );
        for h in [1e-6, 1e-4, 1e-2, 1.0] {
            let contact = epa_r4(&a, &b, seed_of_height(h), SPHERE_PAIR_SCALE)
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
        fn resized(seed: [MinkowskiPoint4; 5], s: f32) -> [MinkowskiPoint4; 5] {
            seed.map(|p| MinkowskiPoint4 {
                point: p.point * s,
                sa: p.sa * s,
                sb: p.sb * s,
            })
        }

        for s in [1e-3_f32, 1.0, 1e3] {
            let a = Sphere4 {
                center: Vec4::ZERO,
                radius: s,
            };
            let b = Sphere4 {
                center: Vec4::new(0.5 * s, 0.0, 0.0, 0.0),
                radius: s,
            };
            let scale = SPHERE_PAIR_SCALE * s;
            assert!(
                epa_r4(&a, &b, resized(seed_of_height(1e-9), s), scale).is_none(),
                "at size {s} a seed 20x under the volume floor resolved a contact"
            );
            assert!(
                epa_r4(&a, &b, resized(seed_of_height(1e-6), s), scale).is_some(),
                "at size {s} a seed 50x over the volume floor resolved to nothing"
            );
        }
    }

    #[test]
    fn collinear_and_repeated_vertex_seeds_are_rejected_rather_than_resolved() {
        let a = Sphere4 {
            center: Vec4::ZERO,
            radius: 1.0,
        };
        let b = Sphere4 {
            center: Vec4::new(0.5, 0.0, 0.0, 0.0),
            radius: 1.0,
        };
        let collinear = seed([
            Vec4::new(-1.0, 0.0, 0.0, 0.0),
            Vec4::new(-0.5, 0.0, 0.0, 0.0),
            Vec4::ZERO,
            Vec4::new(0.5, 0.0, 0.0, 0.0),
            Vec4::new(1.0, 0.0, 0.0, 0.0),
        ]);
        let mut repeated = seed_of_height(1.0);
        repeated[2] = repeated[1];
        assert!(epa_r4(&a, &b, collinear, SPHERE_PAIR_SCALE).is_none());
        assert!(epa_r4(&a, &b, repeated, SPHERE_PAIR_SCALE).is_none());
    }

    fn box4_vertices(half: Vec4) -> Vec<Vec4> {
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

    const WALL_HALF: f32 = 0.05;
    const WALL_SPAN: f32 = 2.0;
    const BALL_RADIUS: f32 = 0.1;
    // Distance from the wall's midplane at which the ball stops touching it.
    const CAPTURE: f32 = WALL_HALF + BALL_RADIUS;
    // `epa_r4`'s contract is a bounding radius, so the wall contributes its
    // diagonal half-span and not its thickness: 3.46 for a slab 0.1 thick.
    // That overstates the gaps the contact turns on by a factor of 35, and the
    // sweep is the evidence that the widened bands swallow nothing, worst
    // error 2.5e-4.
    fn wall_scale() -> f32 {
        BALL_RADIUS + Vec4::new(WALL_HALF, WALL_SPAN, WALL_SPAN, WALL_SPAN).length()
    }

    fn ball_vs_wall(x: f32) -> ContactInfo4 {
        let vertices = box4_vertices(Vec4::new(WALL_HALF, WALL_SPAN, WALL_SPAN, WALL_SPAN));
        let wall = ConvexHull4 {
            vertices: &vertices,
        };
        let ball = Sphere4 {
            center: Vec4::new(x, 0.0, 0.0, 0.0),
            radius: BALL_RADIUS,
        };
        let simplex = match gjk_intersect_r4(&ball, &wall, -ball.center) {
            GjkResult4::Intersecting { simplex } => simplex,
            GjkResult4::Separated => panic!("ball at {x} overlaps the wall"),
        };
        epa_r4(&ball, &wall, simplex, wall_scale()).expect("EPA should resolve an overlap")
    }

    #[test]
    fn wall_contact_leaves_through_the_face_the_ball_entered() {
        for side in [-1.0_f32, 1.0] {
            for k in 1..(CAPTURE / 5e-4) as u32 {
                let x = side * (CAPTURE - 5e-4 * k as f32);
                let contact = ball_vs_wall(x);
                assert!(
                    contact.normal.x * side < -0.99,
                    "ball at {x} leaves along {:?}, not back out of its own face",
                    -contact.normal
                );
                assert_close(contact.penetration, CAPTURE - x.abs(), EPA_TOLERANCE);
            }
        }
    }

    #[test]
    fn wall_contact_depth_stays_zero_up_to_exact_touching() {
        for gap in [1e-3_f32, 1e-4, 1e-5, 0.0] {
            let contact = ball_vs_wall(-(CAPTURE + gap));
            assert_eq!(
                contact.penetration, 0.0,
                "a ball {gap} clear of the wall is not {} deep in it",
                contact.penetration
            );
        }
        for overlap in [1e-5_f32, 1e-4, 1e-3, 1e-2] {
            let contact = ball_vs_wall(-(CAPTURE - overlap));
            assert_close(contact.penetration, overlap, EPA_TOLERANCE);
            assert!(
                contact.normal.x > 0.99,
                "an overlap of {overlap} leaves along {:?}",
                -contact.normal
            );
        }
    }

    // The fixtures below take `B = A + t`, so the Minkowski difference is
    // `K − t` with `K = A ⊕ (−A)`, and for an interior origin the depth is
    // `min_j (h_K(u_j) − ⟨u_j, t⟩)` over K's facet normals, the minimizer
    // being the contact normal. Support functions add over Minkowski sums
    // (Schneider 2014, *Convex Bodies: The Brunn-Minkowski Theory*, §1.7) and
    // the normal fan of a sum is the common refinement of the summands' fans
    // (Ziegler 1995, *Lectures on Polytopes*, §7.1), which gives each K its
    // facet list. Every fixture but the pentatope is centrally symmetric, so
    // there `K = 2A` and `h_K = 2·h_A`. Drop `FACE_COPLANAR_EPS` to 0 and the
    // pentatope, 24-cell and 600-cell resolve against a retired face.

    #[test]
    fn pentatope_pentatope_contact_matches_difference_body_facet() {
        use crate::collision::gjk_r4::ConvexHull4;
        use crate::euclidean_r4::pentatope_vertices;

        let va: Vec<Vec4> = pentatope_vertices(1.0);
        let vb: Vec<Vec4> = pentatope_vertices(1.0)
            .into_iter()
            .map(|v| v + Vec4::new(0.3, 0.0, 0.0, 0.0))
            .collect();

        let a = ConvexHull4 { vertices: &va };
        let b = ConvexHull4 { vertices: &vb };
        let simplex = match gjk_intersect_r4(&a, &b, Vec4::X) {
            GjkResult4::Intersecting { simplex } => simplex,
            _ => panic!("pentatopes should overlap"),
        };
        let contact = epa_r4(&a, &b, simplex, UNIT_POLYCHORON_SCALE).expect("EPA should succeed");

        let root6 = 6.0_f32.sqrt();
        let normal_x = 5.0_f32.sqrt() / root6;
        assert_close(
            contact.penetration,
            5.0 / (2.0 * root6) - 0.3 * normal_x,
            EPA_TOLERANCE,
        );
        // The tie leaves the facet unpinned but not the direction: both tied
        // facets carry the same x-component, and it is positive (A toward B).
        assert_close(contact.normal.x, normal_x, EPA_TOLERANCE);
        let n2 = contact.normal.length_squared();
        assert!(
            (n2 - 1.0).abs() < 1e-2,
            "normal should be unit-length: |n|² = {n2}"
        );
    }

    #[test]
    fn tesseract_tesseract_contact_matches_deepest_axis() {
        use crate::collision::gjk_r4::ConvexHull4;
        use crate::euclidean_r4::tesseract_vertices;

        let va: Vec<Vec4> = tesseract_vertices(1.0);
        let vb: Vec<Vec4> = tesseract_vertices(1.0)
            .into_iter()
            .map(|v| v + Vec4::new(0.4, 0.2, 0.1, 0.0))
            .collect();

        let a = ConvexHull4 { vertices: &va };
        let b = ConvexHull4 { vertices: &vb };
        let simplex = match gjk_intersect_r4(&a, &b, Vec4::X) {
            GjkResult4::Intersecting { simplex } => simplex,
            _ => panic!("tesseracts should overlap"),
        };
        let contact = epa_r4(&a, &b, simplex, UNIT_POLYCHORON_SCALE).expect("EPA should succeed");

        assert_close(contact.penetration, 1.0 - 0.4, EPA_TOLERANCE);
        assert!(
            contact.normal.dot(Vec4::X) > 0.999,
            "normal must run from A toward B along +x, got {:?}",
            contact.normal
        );
    }

    #[test]
    fn cell16_cell16_contact_matches_l1_facet() {
        use crate::collision::gjk_r4::ConvexHull4;
        use crate::euclidean_r4::cell16_vertices;

        let va: Vec<Vec4> = cell16_vertices(1.0);
        let vb: Vec<Vec4> = cell16_vertices(1.0)
            .into_iter()
            .map(|v| v + Vec4::new(0.5, 0.0, 0.0, 0.0))
            .collect();

        let a = ConvexHull4 { vertices: &va };
        let b = ConvexHull4 { vertices: &vb };
        let simplex = match gjk_intersect_r4(&a, &b, Vec4::X) {
            GjkResult4::Intersecting { simplex } => simplex,
            _ => panic!("16-cells should overlap"),
        };
        let contact = epa_r4(&a, &b, simplex, UNIT_POLYCHORON_SCALE).expect("EPA should succeed");

        assert_close(contact.penetration, 1.0 - 0.25, EPA_TOLERANCE);
        assert_close(contact.normal.x, 0.5, EPA_TOLERANCE);
    }

    #[test]
    fn cell24_cell24_contact_matches_axis_facet() {
        use crate::collision::gjk_r4::ConvexHull4;
        use crate::euclidean_r4::cell24_vertices;

        let va: Vec<Vec4> = cell24_vertices(1.0);
        let vb: Vec<Vec4> = cell24_vertices(1.0)
            .into_iter()
            .map(|v| v + Vec4::new(0.3, 0.0, 0.0, 0.0))
            .collect();

        let a = ConvexHull4 { vertices: &va };
        let b = ConvexHull4 { vertices: &vb };
        let simplex = match gjk_intersect_r4(&a, &b, Vec4::X) {
            GjkResult4::Intersecting { simplex } => simplex,
            _ => panic!("24-cells should overlap"),
        };
        let contact = epa_r4(&a, &b, simplex, UNIT_POLYCHORON_SCALE).expect("EPA should succeed");

        assert_close(contact.penetration, 2.0_f32.sqrt() - 0.3, EPA_TOLERANCE);
        assert!(
            contact.normal.dot(Vec4::X) > 0.999,
            "normal must run from A toward B along +x, got {:?}",
            contact.normal
        );
    }

    #[test]
    fn cell600_cell600_contact_matches_vertex_incident_facet() {
        use crate::collision::gjk_r4::ConvexHull4;
        use crate::euclidean_r4::cell600_vertices;

        let va: Vec<Vec4> = cell600_vertices(1.0);
        let vb: Vec<Vec4> = cell600_vertices(1.0)
            .into_iter()
            .map(|v| v + Vec4::new(0.3, 0.0, 0.0, 0.0))
            .collect();

        let a = ConvexHull4 { vertices: &va };
        let b = ConvexHull4 { vertices: &vb };
        let simplex = match gjk_intersect_r4(&a, &b, Vec4::X) {
            GjkResult4::Intersecting { simplex } => simplex,
            _ => panic!("600-cells should overlap"),
        };
        let contact = epa_r4(&a, &b, simplex, UNIT_POLYCHORON_SCALE).expect("EPA should succeed");

        let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
        let inradius = phi * phi / (2.0 * 2.0_f32.sqrt());
        assert_close(contact.penetration, inradius * (2.0 - 0.3), EPA_TOLERANCE);
        assert_close(contact.normal.x, inradius, EPA_TOLERANCE);
    }

    #[test]
    fn cell120_cell120_contact_matches_dodecahedral_facet() {
        use crate::collision::gjk_r4::ConvexHull4;
        use crate::euclidean_r4::cell120_vertices;

        let va: Vec<Vec4> = cell120_vertices(1.0);
        let vb: Vec<Vec4> = cell120_vertices(1.0)
            .into_iter()
            .map(|v| v + Vec4::new(0.3, 0.0, 0.0, 0.0))
            .collect();

        let a = ConvexHull4 { vertices: &va };
        let b = ConvexHull4 { vertices: &vb };
        let simplex = match gjk_intersect_r4(&a, &b, Vec4::X) {
            GjkResult4::Intersecting { simplex } => simplex,
            _ => panic!("120-cells should overlap"),
        };
        let contact = epa_r4(&a, &b, simplex, UNIT_POLYCHORON_SCALE).expect("EPA should succeed");

        let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
        assert_close(
            contact.penetration,
            phi * phi / 2.0_f32.sqrt() - 0.3,
            EPA_TOLERANCE,
        );
        assert!(
            contact.normal.dot(Vec4::X) > 0.999,
            "normal must run from A toward B along +x, got {:?}",
            contact.normal
        );
    }

    #[test]
    fn epa_r4_contact_is_equivariant_under_uniform_scaling() {
        use crate::euclidean_r4::{
            cell120_vertices, cell16_vertices, cell24_vertices, cell600_vertices,
            pentatope_vertices, tesseract_vertices,
        };

        let root5 = 5.0_f32.sqrt();
        let root6 = 6.0_f32.sqrt();
        let phi = (1.0 + root5) * 0.5;
        let inradius600 = phi * phi / (2.0 * 2.0_f32.sqrt());

        // Closed-form depth and normal x-component at circumradius 1 with
        // `t = 0.3·x̂`, each derived in the single-scale fixture above.
        type ScaleFixture = (&'static str, fn(f32) -> Vec<Vec4>, f32, f32);
        let fixtures: [ScaleFixture; 6] = [
            (
                "pentatope",
                pentatope_vertices,
                5.0 / (2.0 * root6) - 0.3 * root5 / root6,
                root5 / root6,
            ),
            ("tesseract", tesseract_vertices, 1.0 - 0.3, 1.0),
            ("16-cell", cell16_vertices, 1.0 - 0.15, 0.5),
            ("24-cell", cell24_vertices, 2.0_f32.sqrt() - 0.3, 1.0),
            (
                "600-cell",
                cell600_vertices,
                inradius600 * (2.0 - 0.3),
                inradius600,
            ),
            (
                "120-cell",
                cell120_vertices,
                phi * phi / 2.0_f32.sqrt() - 0.3,
                1.0,
            ),
        ];

        for (name, vertices_of, unit_depth, unit_normal_x) in fixtures {
            for s in [1e-3_f32, 1e-2, 1.0, 1e2, 1e3] {
                let va = vertices_of(s);
                let vb: Vec<Vec4> = vertices_of(s)
                    .into_iter()
                    .map(|v| v + Vec4::new(0.3 * s, 0.0, 0.0, 0.0))
                    .collect();
                let a = ConvexHull4 { vertices: &va };
                let b = ConvexHull4 { vertices: &vb };
                let simplex = match gjk_intersect_r4(&a, &b, Vec4::X) {
                    GjkResult4::Intersecting { simplex } => simplex,
                    GjkResult4::Separated => panic!("{name} at scale {s} should overlap"),
                };

                let scale = 2.0 * s;
                let contact = epa_r4(&a, &b, simplex, scale)
                    .unwrap_or_else(|| panic!("{name} at scale {s} resolved to no contact"));

                // The termination test is `EPA_TOLERANCE * scale`, so
                // anything tighter would pin the fixture's luck.
                let want_depth = unit_depth * s;
                assert!(
                    (contact.penetration - want_depth).abs() <= EPA_TOLERANCE * scale,
                    "{name} at scale {s}: depth {} is not {want_depth}",
                    contact.penetration
                );
                assert!(
                    (contact.normal.x - unit_normal_x).abs() <= 1e-3,
                    "{name} at scale {s}: normal {:?} has x != {unit_normal_x}",
                    contact.normal
                );
                assert!(
                    (contact.normal.length_squared() - 1.0).abs() < 1e-3,
                    "{name} at scale {s}: normal {:?} is not unit",
                    contact.normal
                );
            }
        }
    }
}
