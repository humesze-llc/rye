use glam::Vec4;
use loam_math::{Rotor, Rotor4};

use super::simplex_r4::{closest_to_origin, Closest};

pub trait SupportFn4 {
    fn support(&self, direction: Vec4) -> Vec4;
}

pub struct ConvexHull4<'a> {
    pub vertices: &'a [Vec4],
}

impl<'a> SupportFn4 for ConvexHull4<'a> {
    fn support(&self, direction: Vec4) -> Vec4 {
        let mut best = self.vertices[0];
        let mut best_d = best.dot(direction);
        for &v in &self.vertices[1..] {
            let d = v.dot(direction);
            if d > best_d {
                best_d = d;
                best = v;
            }
        }
        best
    }
}

/// Body-local vertices supported in the body frame via `<R v, d> = <v, R⁻¹ d>`,
/// so the world vertices are never materialized.
pub struct PosedHull4<'a> {
    pub local: &'a [Vec4],
    pub position: Vec4,
    pub rotation: Rotor4,
}

impl SupportFn4 for PosedHull4<'_> {
    fn support(&self, direction: Vec4) -> Vec4 {
        let local_dir = self.rotation.inverse().apply(direction);
        let mut best = self.local[0];
        let mut best_d = best.dot(local_dir);
        for &v in &self.local[1..] {
            let d = v.dot(local_dir);
            if d > best_d {
                best_d = d;
                best = v;
            }
        }
        self.rotation.apply(best) + self.position
    }
}

pub struct Sphere4 {
    pub center: Vec4,
    pub radius: f32,
}

impl SupportFn4 for Sphere4 {
    fn support(&self, direction: Vec4) -> Vec4 {
        let d = direction.length_squared();
        let dir = if d > 1e-12 {
            direction / d.sqrt()
        } else {
            Vec4::Y
        };
        self.center + dir * self.radius
    }
}

/// `sa` and `sb` are the contributing support points on A and B.
#[derive(Clone, Copy, Debug)]
pub struct MinkowskiPoint4 {
    pub point: Vec4,
    pub sa: Vec4,
    pub sb: Vec4,
}

pub fn minkowski_support_r4<A: SupportFn4, B: SupportFn4>(
    a: &A,
    b: &B,
    direction: Vec4,
) -> MinkowskiPoint4 {
    let sa = a.support(direction);
    let sb = b.support(-direction);
    MinkowskiPoint4 {
        point: sa - sb,
        sa,
        sb,
    }
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum GjkResult4 {
    Intersecting { simplex: [MinkowskiPoint4; 5] },
    Separated,
}

const GJK_MAX_ITERATIONS: u32 = 48;
const GJK_EPS: f32 = 1e-6;

/// On overlap the returned 4-simplex is the seed EPA expects.
pub fn gjk_intersect_r4<A: SupportFn4, B: SupportFn4>(
    a: &A,
    b: &B,
    initial_direction: Vec4,
) -> GjkResult4 {
    let mut dir = if initial_direction.length_squared() > GJK_EPS {
        initial_direction
    } else {
        Vec4::X
    };
    let mut simplex: Vec<MinkowskiPoint4> = Vec::with_capacity(5);
    simplex.push(minkowski_support_r4(a, b, dir));
    dir = -simplex[0].point;

    for _ in 0..GJK_MAX_ITERATIONS {
        if dir.length_squared() < GJK_EPS {
            break;
        }
        let new_point = minkowski_support_r4(a, b, dir);
        if new_point.point.dot(dir) < 0.0 {
            return GjkResult4::Separated;
        }
        // A duplicate support confirms enclosure.
        if simplex
            .iter()
            .any(|p| (p.point - new_point.point).length_squared() < 1e-10)
        {
            break;
        }
        simplex.push(new_point);

        let points: Vec<Vec4> = simplex.iter().map(|p| p.point).collect();
        let Closest {
            point: closest,
            kept,
            ..
        } = closest_to_origin(&points);
        if closest.length_squared() < GJK_EPS {
            let pruned: Vec<MinkowskiPoint4> = kept.iter().map(|&i| simplex[i]).collect();
            simplex = pruned;
            break;
        }
        let pruned: Vec<MinkowskiPoint4> = kept.iter().map(|&i| simplex[i]).collect();
        simplex = pruned;
        dir = -closest;
    }

    let mut tried: Vec<Vec4> = Vec::new();
    while simplex.len() < 5 {
        let Some(probe) = orthogonal_to_hull(&simplex, &tried) else {
            break;
        };
        tried.push(probe);

        let sup = minkowski_support_r4(a, b, probe);
        if simplex
            .iter()
            .all(|p| (p.point - sup.point).length_squared() >= 1e-10)
        {
            simplex.push(sup);
            continue;
        }
        let sup_neg = minkowski_support_r4(a, b, -probe);
        if simplex
            .iter()
            .all(|p| (p.point - sup_neg.point).length_squared() >= 1e-10)
        {
            simplex.push(sup_neg);
            continue;
        }
    }

    if simplex.len() == 5 {
        finalize_intersecting(simplex)
    } else {
        GjkResult4::Separated
    }
}

fn finalize_intersecting(simplex: Vec<MinkowskiPoint4>) -> GjkResult4 {
    if simplex.len() == 5 {
        let arr: [MinkowskiPoint4; 5] =
            [simplex[0], simplex[1], simplex[2], simplex[3], simplex[4]];
        GjkResult4::Intersecting { simplex: arr }
    } else {
        GjkResult4::Separated
    }
}

// `tried` directions are projected out so a probe is never re-picked.
fn orthogonal_to_hull(simplex: &[MinkowskiPoint4], tried: &[Vec4]) -> Option<Vec4> {
    let points: Vec<Vec4> = simplex.iter().map(|p| p.point).collect();
    let basis: Vec<Vec4> = if points.len() <= 1 {
        Vec::new()
    } else {
        let v0 = points[0];
        points[1..].iter().map(|&p| p - v0).collect()
    };

    let mut onb: Vec<Vec4> = Vec::with_capacity(basis.len());
    for &b in &basis {
        let mut r = b;
        for o in &onb {
            r -= *o * r.dot(*o);
        }
        let m = r.length_squared();
        if m > 1e-10 {
            onb.push(r / m.sqrt());
        }
    }

    let axes = [Vec4::X, Vec4::Y, Vec4::Z, Vec4::W];
    let mut best: Option<(f32, Vec4)> = None;
    for &axis in &axes {
        let mut r = axis;
        for o in &onb {
            r -= *o * r.dot(*o);
        }
        for t in tried {
            let tl = t.length_squared();
            if tl > 1e-12 {
                r -= *t * (r.dot(*t) / tl);
            }
        }
        let mag_sq = r.length_squared();
        if mag_sq > 1e-8 && best.is_none_or(|(m, _)| mag_sq > m) {
            best = Some((mag_sq, r));
        }
    }
    best.map(|(_, v)| v.normalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_math::{Bivector, Bivector4, Plane4};

    #[test]
    fn a_posed_hull_supports_where_the_world_vertices_would() {
        let local = [
            Vec4::new(0.6, 0.1, -0.2, 0.35),
            Vec4::new(-0.4, 0.7, 0.15, -0.5),
            Vec4::new(0.05, -0.8, 0.45, 0.2),
            Vec4::new(-0.3, 0.2, -0.65, -0.1),
            Vec4::new(0.25, 0.4, 0.5, 0.6),
        ];
        let position = Vec4::new(3.0, -1.5, 0.75, -2.25);
        // The double rotation mixes all four axes, which an un-inverted rotor gets wrong.
        let simple = (Plane4::Xw.unit_bivector() * 0.8).exp().normalize();
        let double = (Bivector4::new(0.5, 0.0, 0.0, 0.0, 0.0, -0.9).exp()).normalize();

        for rotation in [Rotor4::identity(), simple, double] {
            let world: Vec<Vec4> = local
                .iter()
                .map(|v| rotation.apply(*v) + position)
                .collect();
            let posed = PosedHull4 {
                local: &local,
                position,
                rotation,
            };
            let materialised = ConvexHull4 { vertices: &world };
            for dir in [
                Vec4::X,
                Vec4::W,
                Vec4::new(1.0, 1.0, 1.0, 1.0),
                Vec4::new(-0.3, 0.9, -0.2, 0.7),
                Vec4::new(0.0, -1.0, 0.4, -0.4),
            ] {
                let (a, b) = (posed.support(dir), materialised.support(dir));
                assert!(
                    (a - b).length() < 1e-5,
                    "posed support {a:?} against materialised {b:?} for {dir:?}"
                );
            }
        }
    }

    #[test]
    fn separated_spheres() {
        let a = Sphere4 {
            center: Vec4::new(-5.0, 0.0, 0.0, 0.0),
            radius: 1.0,
        };
        let b = Sphere4 {
            center: Vec4::new(5.0, 0.0, 0.0, 0.0),
            radius: 1.0,
        };
        match gjk_intersect_r4(&a, &b, Vec4::X) {
            GjkResult4::Separated => {}
            _ => panic!("expected Separated"),
        }
    }

    #[test]
    fn overlapping_spheres() {
        let a = Sphere4 {
            center: Vec4::new(0.0, 0.0, 0.0, 0.0),
            radius: 2.0,
        };
        let b = Sphere4 {
            center: Vec4::new(1.0, 0.0, 0.0, 0.0),
            radius: 2.0,
        };
        assert!(matches!(
            gjk_intersect_r4(&a, &b, Vec4::X),
            GjkResult4::Intersecting { .. }
        ));
    }

    #[test]
    fn tesseracts_overlap_past_touching() {
        use crate::euclidean_r4::tesseract_vertices;
        let va: Vec<Vec4> = tesseract_vertices(1.0);
        let vb: Vec<Vec4> = tesseract_vertices(1.0)
            .into_iter()
            .map(|v| v + Vec4::new(0.6, 0.6, 0.6, 0.6))
            .collect();
        let a = ConvexHull4 { vertices: &va };
        let b = ConvexHull4 { vertices: &vb };
        assert!(matches!(
            gjk_intersect_r4(&a, &b, Vec4::X),
            GjkResult4::Intersecting { .. }
        ));
    }

    #[test]
    fn deeply_overlapping_pentatopes() {
        use crate::euclidean_r4::pentatope_vertices;
        let va: Vec<Vec4> = pentatope_vertices(1.0);
        let vb: Vec<Vec4> = pentatope_vertices(1.0)
            .into_iter()
            .map(|v| v + Vec4::new(0.2, 0.0, 0.0, 0.0))
            .collect();
        let a = ConvexHull4 { vertices: &va };
        let b = ConvexHull4 { vertices: &vb };
        assert!(matches!(
            gjk_intersect_r4(&a, &b, Vec4::X),
            GjkResult4::Intersecting { .. }
        ));
    }

    #[test]
    fn fully_separated_pentatopes() {
        use crate::euclidean_r4::pentatope_vertices;
        let va: Vec<Vec4> = pentatope_vertices(1.0);
        let vb: Vec<Vec4> = pentatope_vertices(1.0)
            .into_iter()
            .map(|v| v + Vec4::new(10.0, 0.0, 0.0, 0.0))
            .collect();
        let a = ConvexHull4 { vertices: &va };
        let b = ConvexHull4 { vertices: &vb };
        assert!(matches!(
            gjk_intersect_r4(&a, &b, Vec4::X),
            GjkResult4::Separated
        ));
    }

    #[test]
    fn tesseract_face_touching_is_bracketed_and_never_resolves_to_a_depth() {
        use crate::collision::epa_r4::epa_r4;
        use crate::euclidean_r4::tesseract_vertices;

        let va: Vec<Vec4> = tesseract_vertices(1.0);
        let a = ConvexHull4 { vertices: &va };
        // `tesseract_vertices(1.0)` has half extent 0.5, so the faces meet at shift 1.
        for (shift, expect_overlap) in [
            (1.0 - 1e-1, true),
            (1.0 - 1e-2, true),
            (1.0, false),
            (1.0 + 1e-1, false),
            (1.0 + 1.0, false),
        ] {
            let vb: Vec<Vec4> = tesseract_vertices(1.0)
                .into_iter()
                .map(|v| v + Vec4::new(shift, 0.0, 0.0, 0.0))
                .collect();
            let b = ConvexHull4 { vertices: &vb };
            let depth = match gjk_intersect_r4(&a, &b, Vec4::X) {
                GjkResult4::Intersecting { simplex } => {
                    // Both circumradius 1, so the difference body's scale is 2.
                    epa_r4(&a, &b, simplex, 2.0).map_or(0.0, |c| c.penetration)
                }
                GjkResult4::Separated => 0.0,
            };
            if expect_overlap {
                assert!(
                    (depth - (1.0 - shift)).abs() < 1e-2,
                    "shift {shift} overlaps by {} but resolved to {depth}",
                    1.0 - shift
                );
            } else {
                assert_eq!(depth, 0.0, "shift {shift} is clear but resolved to {depth}");
            }
        }
    }

    #[test]
    fn sphere_and_tesseract_inside() {
        use crate::euclidean_r4::tesseract_vertices;
        let sphere = Sphere4 {
            center: Vec4::ZERO,
            radius: 0.1,
        };
        let vs: Vec<Vec4> = tesseract_vertices(1.0);
        let tess = ConvexHull4 { vertices: &vs };
        assert!(matches!(
            gjk_intersect_r4(&sphere, &tess, Vec4::X),
            GjkResult4::Intersecting { .. }
        ));
    }
}
