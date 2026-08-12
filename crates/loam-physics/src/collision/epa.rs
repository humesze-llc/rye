//! EPA: Expanding Polytope Algorithm (3D).
//!
//! Given GJK's terminating tetrahedron, EPA grows it into a polytope whose surface
//! approaches the boundary of `A ⊖ B`. The closest surface point to the origin is
//! the minimum translation: contact normal and penetration depth. The contact
//! point comes from the closest point's barycentric weights applied to the cached
//! per-shape supports in each [`MinkowskiPoint`].
//!
//! Each iteration: find the closest face, query a support along its outward normal,
//! terminate if that support's distance matches the face's, else add the support
//! (remove faces it sees, stitch new triangles from each horizon edge to it).
//!
//! Every geometric threshold here is a dimensionless coefficient times the
//! caller's `scale` raised to the homogeneity degree of the quantity it guards,
//! so the resolved contact is equivariant under a uniform scaling of both
//! bodies. See [`Thresholds`], and [`super::epa_r4`][mod@super::epa_r4] for the
//! 4D twin of the same treatment.

use glam::Vec3;

use super::gjk::{minkowski_support, MinkowskiPoint, SupportFn};

const EPA_MAX_ITERATIONS: u32 = 48;
/// Sanity cap: a well-formed EPA finishes with < 30 vertices; past this we are in
/// a degenerate stall.
const EPA_MAX_VERTICES: usize = 96;

/// Support gap at which the expansion has reached the boundary, per unit of
/// `scale`. Degree 1: the compared quantity is the distance between two planes
/// with the same unit normal. Held absolute it buys a fixed number of world
/// units of depth accuracy rather than a fixed number of digits.
const EPA_TOLERANCE: f32 = 1e-4;

/// Band around a face plane, per unit of `scale`, in which a support point
/// counts as lying on it, so that `expand` retires the face instead of keeping
/// it.
///
/// A facet of a Minkowski difference of polytopes is generally not a simplex:
/// the difference body of two boxes is a box, whose square facets are each
/// tiled by at least two coplanar triangles, and a support point lands exactly
/// on their shared plane. Retiring one tile while keeping its coplanar
/// neighbour stitches the new vertex into a facet that is already covered, so
/// the surface stops being convex and later iterations resolve against faces
/// interior to the difference body. Sibling of `epa_r4::FACE_COPLANAR_EPS`:
/// same value, same conditioning class.
///
/// Degree 1: the residual is a distance along a unit normal. Measured over
/// 11160 axis-aligned box pairs spanning scales 0.02 to 10, the band that
/// resolves every one against a real facet is `[3e-7, 1e-4]` absolute, and it
/// covered both ends of that grid at once only because the coefficient sits
/// near the band's logarithmic centre. The floor is where f32 stops resolving
/// the plane residual of a dot product at the operands' magnitude, so it tracks
/// the geometry; the ceiling is `EPA_TOLERANCE`, above which the epsilon
/// retires faces whose support gap is the very thing EPA is still closing. Both
/// ends being degree 1, the band scales with `scale` and 1e-5 keeps its 33x of
/// floor clearance and its order of ceiling clearance at every size.
///
/// Below the floor is worse than zero, not merely weaker. `add_or_remove_edge`
/// cancels horizon edges by winding, so retiring part of a coplanar band and
/// keeping the rest leaves edges with no reverse to cancel against, the horizon
/// stops being a topological disc, and each iteration stitches more faces than
/// it removed. `expand` caps iterations and vertices but not faces, so the
/// count climbs geometrically: over that same grid, peak faces at 1e-7 is 5368
/// against 180 at exactly zero, and at 1e-8 the wrong-contact count rises from
/// 21 to 28.
const FACE_COPLANAR_EPS: f32 = 1e-5;

/// Floor on `|(b−a) × (c−a)|`, per `scale` squared, below which a face has no
/// usable normal direction and one is fabricated. Degree 2: the cross product
/// of two edge vectors is an area.
const FACE_DEGENERATE_CROSS: f32 = 1e-8;

/// Floor on the seed tetrahedron's `|(p1−p0)·((p2−p0)×(p3−p0))|`, per `scale`
/// cubed, below which the seed is coplanar and has no interior to orient faces
/// against. Degree 3: a scalar triple product is a volume, so it loses three
/// decades of headroom per decade of shrinkage. Held absolute this is the first
/// guard to fire on the small side, and it is what rejected 1040 of 1208
/// axis-aligned box pairs at half-extent 0.02.
const SEED_DEGENERATE_VOLUME: f32 = 1e-8;

/// Floor on the Gram determinant of a face's two edge vectors, per `scale` to
/// the fourth, below which the barycentric solve has no pivot and falls back to
/// the first vertex. Degree 4: `d00·d11 − d01²` is a product of two inner
/// products.
const BARYCENTRIC_SINGULAR_EPS: f32 = 1e-12;

/// The scale-derived thresholds of one `epa` call: each constant times `scale`
/// raised to that constant's degree, evaluated once at entry because `scale` is
/// fixed for the call while `build_face_vs_point` runs once per horizon edge
/// per iteration.
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

/// Resolved contact info for a [`crate::Contact`].
#[derive(Clone, Copy, Debug)]
pub struct ContactInfo {
    /// Unit vector from A toward B (world), per the `Contact::normal` convention.
    pub normal: Vec3,
    /// Overlap depth along `normal`.
    pub penetration: f32,
    /// World-space contact point: midpoint of the closest features on A and B.
    pub point: Vec3,
}

/// Triangle face of the expanding polytope, stored by vertex index.
#[derive(Clone, Copy, Debug)]
struct Face {
    /// Indices into `Polytope::vertices`, wound so `normal` points outward.
    v: [usize; 3],
    /// Unit outward normal; `normal * distance` is the plane's closest point to
    /// the origin.
    normal: Vec3,
    /// Distance from origin to the face plane, always ≥ 0.
    distance: f32,
}

struct Polytope {
    vertices: Vec<MinkowskiPoint>,
    faces: Vec<Face>,
    /// Seed-tetrahedron centroid; stays interior under all convex expansions, so
    /// it orients new faces outward. An arbitrary old vertex can sit on a new
    /// face's plane and flip the orientation, corrupting the polytope.
    interior: glam::Vec3,
    thresholds: Thresholds,
}

impl Polytope {
    fn from_tetra(tetra: [MinkowskiPoint; 4], thresholds: Thresholds) -> Self {
        let vertices = tetra.to_vec();
        let interior = (tetra[0].point + tetra[1].point + tetra[2].point + tetra[3].point) * 0.25;
        // Four faces, each wound so its normal points away from the opposite vertex.
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

    /// Closest face to the origin, or `None` if a degenerate expansion removed
    /// every face without restitching.
    fn closest_face(&self) -> Option<usize> {
        let (idx, _) = self
            .faces
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.distance.total_cmp(&b.1.distance))?;
        Some(idx)
    }

    /// Add `support`: remove every face whose outward normal faces it, then connect
    /// `support` to each horizon edge.
    fn expand(&mut self, support: MinkowskiPoint) {
        let new_idx = self.vertices.len();
        self.vertices.push(support);

        // Horizon = edges of removed faces not shared with another removed face.
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

        // Stitch each horizon edge to the new vertex, oriented against the seed
        // centroid (a guaranteed interior reference; see the `interior` field).
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

/// Build a face with outward normal, oriented away from `interior_point` (the
/// caller guarantees it is inside the polytope).
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

    // Flip if the normal points toward the interior reference.
    let to_interior = interior_point - pa;
    let (v_order, outward_normal) = if normal.dot(to_interior) > 0.0 {
        ([a, c, b], -normal)
    } else {
        ([a, b, c], normal)
    };

    // Clamp at zero for noise near the origin-on-boundary case.
    let raw_distance = outward_normal.dot(verts[v_order[0]].point);
    let distance = raw_distance.max(0.0);
    Face {
        v: v_order,
        normal: outward_normal,
        distance,
    }
}

/// Track a horizon edge by winding. Two removed faces sharing an edge store it as
/// `(a, b)` and `(b, a)`; finding the reverse cancels both. Survivors keep the
/// winding that gives correct outward normals when stitched to the new vertex.
fn add_or_remove_edge(horizon: &mut Vec<(usize, usize)>, a: usize, b: usize) {
    if let Some(pos) = horizon.iter().position(|&e| e == (b, a)) {
        horizon.swap_remove(pos);
    } else {
        horizon.push((a, b));
    }
}

/// Barycentric coords `(u, v, w)` of `p` projected onto triangle `(a, b, c)`, with
/// `u·a + v·b + w·c` the projection.
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

/// Penetration normal, depth, and contact point for two overlapping shapes, given
/// GJK's terminating tetrahedron.
///
/// `scale` is the characteristic length of the Minkowski difference; the
/// caller's contract is to pass the sum of the two bodies' bounding radii,
/// which every caller already computes for its broadphase pre-cull. Each of the
/// five thresholds in this module is its constant times `scale` raised to that
/// constant's degree, which is what makes the result equivariant under a
/// uniform scaling of both bodies.
pub fn epa<A: SupportFn, B: SupportFn>(
    a: &A,
    b: &B,
    initial_simplex: [MinkowskiPoint; 4],
    scale: f32,
) -> Option<ContactInfo> {
    let (polytope, face) = expand_to_boundary(a, b, initial_simplex, scale)?;
    contact_from_face(&polytope, face)
}

/// Grow the seed tetrahedron until its closest face lies on the Minkowski
/// boundary, returning the polytope with that face. `None` for a zero-volume
/// seed, a non-finite support, or a polytope a degenerate expansion collapsed.
fn expand_to_boundary<A: SupportFn, B: SupportFn>(
    a: &A,
    b: &B,
    initial_simplex: [MinkowskiPoint; 4],
    scale: f32,
) -> Option<(Polytope, Face)> {
    let thresholds = Thresholds::for_scale(scale);

    // Reject a near-coplanar (zero-volume) seed: |det([p1-p0, p2-p0, p3-p0])|.
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
        // A collapsed polytope (no faces) is outside EPA's domain; bail cleanly.
        let face_idx = polytope.closest_face()?;
        let face = polytope.faces[face_idx];

        let support = minkowski_support(a, b, face.normal);
        let new_distance = support.point.dot(face.normal);

        // Bail on non-finite support rather than grow the polytope with NaN.
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

    // Cap hit (near-degenerate inputs only): return the best estimate.
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

    // Closest point on the face to the origin, in Minkowski-diff space.
    let closest = face.normal * face.distance;

    let (u, v, w) = barycentric(
        v0.point,
        v1.point,
        v2.point,
        closest,
        polytope.thresholds.barycentric_gram,
    );

    // Same weights against the cached supports recover the points on A and B.
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

    /// Bounding radius of `box_vertices(_, splat(half))` about its own centre,
    /// which is the term `epa`'s `scale` contract asks a box to contribute.
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
        // Unit boxes offset by 1.5 along X -> 0.5 overlap along +X.
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
        // Sphere at (1.2,1.2,1.2), r=0.5 vs box corner (1,1,1): corner-centre
        // distance √(3·0.04) ≈ 0.346, penetration ≈ 0.154, normal along +(1,1,1)/√3.
        let vb = box_vertices(Vec3::ZERO, Vec3::ONE);
        let b = ConvexHull { vertices: &vb };
        let s = Sphere {
            center: Vec3::new(1.2, 1.2, 1.2),
            radius: 0.5,
        };

        // Box as A, sphere as B so normal A->B points toward (1,1,1)/√3.
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

    /// Both degeneracy fixtures below drive the same pair of radius-1 spheres.
    const SPHERE_PAIR_SCALE: f32 = 2.0;

    /// Pre-images are irrelevant to a seed that never reaches contact
    /// reconstruction, so they are the difference points themselves.
    fn seed(points: [Vec3; 4]) -> [MinkowskiPoint; 4] {
        points.map(|point| MinkowskiPoint {
            point,
            sa: point,
            sb: Vec3::ZERO,
        })
    }

    /// Base triangle around the origin plus an apex at height `h`, giving
    /// `|det| = 4·h` against a volume floor of
    /// `SEED_DEGENERATE_VOLUME·SPHERE_PAIR_SCALE³` = 8e-8: `h = 0` is the flat
    /// seed, and every height sampled below clears the floor by more than an
    /// order.
    fn seed_of_height(h: f32) -> [MinkowskiPoint; 4] {
        seed([
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, h),
        ])
    }

    /// The floor is a threshold on a determinant, so seeds sit arbitrarily
    /// close on both sides of it. Below, there is no interior to orient faces
    /// against and the only honest answer is none; above, however thin, the
    /// answer has to be a number. A cross product of two nearly parallel edges
    /// normalizes to NaN, and NaN reaches the solver as a contact.
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

    /// Which seeds the volume floor rejects is fixed by the seed's shape and
    /// not by its size, which is the whole content of the floor being degree 3:
    /// `seed_of_height` has `|det| = 4·h`, so at size `s` it reads `4·h·s³`
    /// against `SEED_DEGENERATE_VOLUME·(SPHERE_PAIR_SCALE·s)³` and the two
    /// heights below stay 20x under and 50x over the floor at every `s`. The
    /// scaling property test cannot see this guard at all, because its seeds
    /// come from GJK and sit nowhere near the floor.
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

    /// The rest of the degeneracy ladder: a seed collapsed to a segment, and
    /// one with a repeated vertex. Both have zero volume by a different route
    /// than coplanarity, and both must take the same exit.
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

    /// Boxes nested deeply enough that separating along the shallowest axis is a
    /// translation of nearly a full body width. The difference body of two unit
    /// half-extent boxes is the half-extent-2 box, so the depth is
    /// `min_i (2 − |t_i|)` and the normal is that axis: `2 − 0.3` along `+x̂`.
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

    /// The difference body of two axis-aligned boxes is the box of summed
    /// half-extents centred at `-t`, so the depth is `min_i (h_i - |t_i|)` and
    /// the admissible normals are the signed axes attaining it. Every case here
    /// has an axis exactly flush, which is what tiles a facet of that body with
    /// coplanar triangles; without [`FACE_COPLANAR_EPS`] the first and the last
    /// resolve against a face interior to the body. The first returns 0.85 for
    /// an exact 1.9 on a normal 27 degrees off the nearest separating axis, and
    /// clears the caller's contact validation on the way to the solver; the
    /// last collapses to zero depth, which that validation drops, leaving an
    /// overlapping pair with no contact at all.
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

    /// Two axes exactly flush and a 5% depth deficit on the third. Without
    /// [`FACE_COPLANAR_EPS`] this configuration retires part of a coplanar band
    /// and keeps the rest, which leaves horizon edges with no reverse to cancel
    /// against, so every iteration stitches more faces than it removed. Neither
    /// the iteration cap nor the vertex cap bounds face growth, so the count
    /// climbs geometrically to order 3e5 by the 48th iteration, and
    /// `add_or_remove_edge` scans the horizon linearly on every one of them.
    /// Converged, this expansion is 16 faces.
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

    /// The scaling twin of the fixtures above, and the 3D half of the property
    /// `epa_r4_contact_is_equivariant_under_uniform_scaling` pins in R⁴. Each
    /// case is similar to itself at every `s`, so the difference body is `s`
    /// times the unit one, the depth is `s` times the unit depth, and the
    /// normal does not move at all.
    ///
    /// Every guard in this module compares a distance, an area, a volume or a
    /// Gram determinant, so a threshold held absolute loses one, two, three or
    /// four decades of headroom per decade of shrinkage. Passing `scale = 1` at
    /// every `s`, which is what the pre-`scale` thresholds amount to, fails
    /// this at the small end on the curved fixture: the spheres come out 4.5%
    /// shallow at `s = 1e-3` with a normal 17 degrees off the separating axis,
    /// and 0.3% shallow at `s = 1e-2` with 4.3 degrees. `EPA_TOLERANCE` is the
    /// guard that does it, because absolute it buys a fixed number of world
    /// units of support gap while the whole depth at `s = 1e-3` is 5e-4. The
    /// box families return the exact depth at every `s` either way; they are
    /// here to keep the flush-facet path in the property, not because a
    /// threshold moves them.
    ///
    /// The seed is taken once at unit scale and scaled, rather than re-derived
    /// by `gjk_intersect` at each `s`. GJK compares `length_squared` against
    /// absolute constants of its own, so below `s ~ 1e-2` it calls these
    /// overlapping pairs separated and no seed arrives at all. That is a defect
    /// in `gjk`, in a module this change does not touch, and scaling the seed
    /// is what isolates the property being pinned here.
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

        // Generic position and one-axis flush are the two box families the
        // coplanar band separates; the spheres carry the curved support.
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
