//! The convex ring a dynamic letter collides with: the convex hull of the
//! collider cover, simplified down to a bounded side count.
//! Simplification only ever grows the ring, so a hull of the cover stays a
//! hull of the letter at every side count.

use glam::Vec2;

/// Sides the reduced ring is capped at. The 4D prism repeats the ring once per
/// `(z, w)` corner, so its vertex count is `4 * sides`, and `loam-physics`
/// carries a 4D polytope through a fixed 32-vertex stack buffer.
pub(super) const MAX_HULL_SIDES: usize = 8;

/// Convex hull of `points`, counter-clockwise, with collinear points dropped.
///
/// Andrew's monotone chain (Andrew 1979, "Another efficient algorithm for
/// convex hulls in two dimensions", Inf. Process. Lett. 9(5)). Sorting by
/// `total_cmp` rather than a partial comparator keeps the order total, so the
/// hull is a function of the point set and not of its arrival order.
pub(super) fn convex_hull(mut points: Vec<Vec2>) -> Vec<Vec2> {
    points.sort_unstable_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    points.dedup();
    if points.len() < 3 {
        return points;
    }

    let mut hull: Vec<Vec2> = Vec::with_capacity(points.len() + 1);
    for &p in &points {
        pop_non_left_turns(&mut hull, p, 2);
        hull.push(p);
    }
    let lower = hull.len() + 1;
    for &p in points.iter().rev().skip(1) {
        pop_non_left_turns(&mut hull, p, lower);
        hull.push(p);
    }
    hull.pop();
    hull
}

/// Drop trailing vertices of `hull` until appending `p` turns strictly left,
/// keeping at least `floor` of them. A non-strict test would retain collinear
/// runs, and a collinear vertex has no removable edge for [`reduce_sides`].
fn pop_non_left_turns(hull: &mut Vec<Vec2>, p: Vec2, floor: usize) {
    while hull.len() >= floor {
        let b = hull[hull.len() - 1];
        let a = hull[hull.len() - 2];
        if (b - a).perp_dot(p - a) > 0.0 {
            break;
        }
        hull.pop();
    }
}

/// Shrink a counter-clockwise convex ring to at most `sides` vertices by
/// repeatedly deleting the edge whose deletion adds the least area.
///
/// Deleting edge `(b, c)` extends its two neighbouring edges to their
/// intersection `x` and replaces both endpoints by it, so the ring stays convex
/// and gains exactly the triangle `(b, c, x)`. The result therefore still
/// contains everything the input contained, which is the property the collider
/// needs; it is not a bounded-error approximation and none is claimed.
///
/// A ring of `n >= 5` vertices always has a deletable edge: exterior angles sum
/// to `2*pi`, so some adjacent pair sums to at most `4*pi/n < pi` and its two
/// edges meet ahead of the deleted one. Below five the loop is already done,
/// since `sides >= 4`.
pub(super) fn reduce_sides(ring: &mut Vec<Vec2>, sides: usize) {
    debug_assert!(sides >= 4, "a convex ring cannot be reduced below a quad");
    while ring.len() > sides {
        let n = ring.len();
        let mut best: Option<(f32, usize, Vec2)> = None;
        for i in 0..n {
            let a = ring[(i + n - 1) % n];
            let b = ring[i];
            let c = ring[(i + 1) % n];
            let d = ring[(i + 2) % n];
            let Some(x) = extend_to_meet(a, b, c, d) else {
                continue;
            };
            let added = 0.5 * (c - b).perp_dot(x - b).abs();
            if best.is_none_or(|(least, _, _)| added < least) {
                best = Some((added, i, x));
            }
        }
        let Some((_, i, x)) = best else { break };
        ring[i] = x;
        ring.remove((i + 1) % n);
    }
}

/// Intersection of line `a -> b` extended past `b` with line `c -> d` extended
/// back before `c`, or `None` when the two do not meet on that side.
fn extend_to_meet(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> Option<Vec2> {
    let ab = b - a;
    let cd = d - c;
    let denom = ab.perp_dot(cd);
    // Parallel edges never meet; near-parallel ones meet so far out that the
    // added area disqualifies them anyway, so no separate tolerance is needed.
    if denom == 0.0 {
        return None;
    }
    let t = (c - a).perp_dot(cd) / denom;
    let s = (c - a).perp_dot(ab) / denom;
    if !(t >= 1.0 && s <= 0.0) {
        return None;
    }
    Some(a + ab * t)
}

/// Twice the signed area of a ring, positive when counter-clockwise (the
/// shoelace formula).
pub(super) fn double_area(ring: &[Vec2]) -> f32 {
    let n = ring.len();
    (0..n)
        .map(|i| {
            let a = ring[i];
            let b = ring[(i + 1) % n];
            a.x * b.y - b.x * a.y
        })
        .sum()
}

/// Area centroid of a counter-clockwise convex ring, i.e. the centre of mass of
/// a uniform prism over it. The vertex mean is not: it weights a subdivided
/// edge as if the shape had mass there.
pub(super) fn centroid(ring: &[Vec2]) -> Vec2 {
    let n = ring.len();
    let mut moment = Vec2::ZERO;
    for i in 0..n {
        let a = ring[i];
        let b = ring[(i + 1) % n];
        moment += (a + b) * (a.x * b.y - b.x * a.y);
    }
    moment / (3.0 * double_area(ring))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regular_ring(sides: usize, radius: f32) -> Vec<Vec2> {
        (0..sides)
            .map(|k| {
                let angle = std::f32::consts::TAU * k as f32 / sides as f32;
                Vec2::new(radius * angle.cos(), radius * angle.sin())
            })
            .collect()
    }

    #[test]
    fn hull_is_counter_clockwise_and_free_of_collinear_vertices() {
        let mut points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
        ];
        points.extend([
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 1.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
        ]);
        let hull = convex_hull(points.clone());
        assert_eq!(hull.len(), 4, "hull kept a collinear or interior point");
        assert!(double_area(&hull) > 0.0, "hull is clockwise");
        let n = hull.len();
        for p in &points {
            for k in 0..n {
                let a = hull[k];
                let b = hull[(k + 1) % n];
                assert!(
                    (b - a).perp_dot(*p - a) >= 0.0,
                    "input {p} lies outside hull edge {k}"
                );
            }
        }
    }

    #[test]
    fn reduction_encloses_the_ring_it_started_from() {
        for sides in [5usize, 7, 12, 31] {
            let original = regular_ring(sides, 1.0);
            let mut reduced = original.clone();
            reduce_sides(&mut reduced, 4);
            assert_eq!(reduced.len(), 4, "{sides}-gon stalled at {}", reduced.len());
            assert!(double_area(&reduced) >= double_area(&original));
            for p in &original {
                let n = reduced.len();
                for k in 0..n {
                    let a = reduced[k];
                    let b = reduced[(k + 1) % n];
                    assert!(
                        (b - a).perp_dot(*p - a) >= -1.0e-5,
                        "{sides}-gon: reduction dropped {p} outside edge {k}"
                    );
                }
            }
        }
    }

    #[test]
    fn reduction_leaves_a_convex_counter_clockwise_ring() {
        let mut ring = regular_ring(17, 2.0);
        reduce_sides(&mut ring, MAX_HULL_SIDES);
        assert_eq!(ring.len(), MAX_HULL_SIDES);
        assert!(double_area(&ring) > 0.0);
        let n = ring.len();
        for k in 0..n {
            let a = ring[k];
            let b = ring[(k + 1) % n];
            let c = ring[(k + 2) % n];
            assert!(
                (b - a).perp_dot(c - b) > 0.0,
                "vertex {k} is reflex after reduction"
            );
        }
    }

    #[test]
    fn a_ring_within_the_cap_is_left_alone() {
        let original = regular_ring(5, 1.0);
        let mut ring = original.clone();
        reduce_sides(&mut ring, MAX_HULL_SIDES);
        assert_eq!(ring, original);
    }

    #[test]
    fn reduction_deletes_the_cheapest_edge_first() {
        // A square with one corner shaved by a tiny chamfer. Deleting the
        // chamfer restores the corner and costs almost nothing; deleting any
        // other edge costs a quarter of the square or more.
        let mut ring = vec![
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(1.0, 0.99),
            Vec2::new(0.99, 1.0),
            Vec2::new(-1.0, 1.0),
        ];
        reduce_sides(&mut ring, 4);
        assert_eq!(ring.len(), 4);
        let restored = ring
            .iter()
            .any(|p| p.distance(Vec2::new(1.0, 1.0)) < 1.0e-5);
        assert!(restored, "the chamfer was not the edge deleted: {ring:?}");
    }

    #[test]
    fn centroid_is_the_area_centroid_and_ignores_edge_subdivision() {
        let square = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.5, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        let c = centroid(&square);
        assert!(
            c.distance(Vec2::splat(0.5)) < 1.0e-6,
            "centroid {c} is not the square's centre"
        );
    }
}
