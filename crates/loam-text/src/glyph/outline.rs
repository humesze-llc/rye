//! Font outline to flattened closed contours.
//!
//! `ab_glyph` hands back a flat `Vec<OutlineCurve>` with no contour separators:
//! contours are delimited implicitly by a curve whose start point does not
//! continue the previous curve's end, and each is explicitly closed by a final
//! `Line` back to its move-to point.

use ab_glyph::{OutlineCurve, Point};
use glam::Vec2;

/// Upper bound on uniform subdivisions of a single Bezier segment. A cap is
/// needed because the subdivision count grows with curve size over tolerance,
/// and a pathological font can hand back an arbitrarily large control polygon.
const MAX_SUBDIVISIONS: u32 = 64;

/// Points closer than this (in em) are the same point. Font outlines routinely
/// repeat a vertex where two segments meet and where the closing line lands on
/// the move-to point; both produce zero-length segments, which have no
/// well-defined direction for the winding-number sign test in
/// [`super::field`].
const COINCIDENT_POINT_EM: f32 = 1.0e-6;

/// A closed contour of a glyph outline, y-up with the baseline at `y = 0` and
/// one em spanning 1.0. The closing edge from the last point back to the first
/// is implicit, so `points` never repeats the start.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Contour {
    pub points: Vec<Vec2>,
}

/// Split and flatten `curves` into closed contours, scaling font units by
/// `units_to_em`. `tolerance_em` bounds the chord deviation of each flattened
/// Bezier segment.
pub(super) fn contours_from_curves(
    curves: &[OutlineCurve],
    units_to_em: f32,
    tolerance_em: f32,
) -> Vec<Contour> {
    let tolerance_units = tolerance_em / units_to_em;

    let mut contours = Vec::new();
    let mut current: Vec<Vec2> = Vec::new();
    let mut previous_end: Option<Point> = None;

    for curve in curves {
        let start = curve_start(curve);
        if previous_end.is_none_or(|end| end != start) {
            close_contour(&mut contours, std::mem::take(&mut current), units_to_em);
            current.push(to_vec2(start));
        }
        flatten_into(curve, tolerance_units, &mut current);
        previous_end = Some(curve_end(curve));
    }
    close_contour(&mut contours, current, units_to_em);

    contours
}

/// Scale, drop coincident neighbours (including the explicit closing point),
/// and keep the contour only if a nondegenerate polygon survives.
fn close_contour(contours: &mut Vec<Contour>, raw: Vec<Vec2>, units_to_em: f32) {
    let mut points: Vec<Vec2> = Vec::with_capacity(raw.len());
    for p in raw {
        let p = p * units_to_em;
        if points
            .last()
            .is_some_and(|last| last.distance_squared(p) <= COINCIDENT_POINT_EM.powi(2))
        {
            continue;
        }
        points.push(p);
    }
    while points.len() >= 2
        && points[0].distance_squared(points[points.len() - 1]) <= COINCIDENT_POINT_EM.powi(2)
    {
        points.pop();
    }
    if points.len() >= 3 {
        contours.push(Contour { points });
    }
}

/// Append every point of `curve` after its start, flattened to within
/// `tolerance` of the true curve.
fn flatten_into(curve: &OutlineCurve, tolerance: f32, out: &mut Vec<Vec2>) {
    match *curve {
        OutlineCurve::Line(_, p1) => out.push(to_vec2(p1)),
        OutlineCurve::Quad(p0, p1, p2) => {
            let (p0, p1, p2) = (to_vec2(p0), to_vec2(p1), to_vec2(p2));
            let n = subdivisions(QUAD_CHORD_BOUND * (p0 - 2.0 * p1 + p2).length(), tolerance);
            for k in 1..=n {
                let t = k as f32 / n as f32;
                out.push(eval_quad(p0, p1, p2, t));
            }
        }
        OutlineCurve::Cubic(p0, p1, p2, p3) => {
            let (p0, p1, p2, p3) = (to_vec2(p0), to_vec2(p1), to_vec2(p2), to_vec2(p3));
            let second_differences = (p0 - 2.0 * p1 + p2).length() + (p1 - 2.0 * p2 + p3).length();
            let n = subdivisions(CUBIC_CHORD_BOUND * second_differences, tolerance);
            for k in 1..=n {
                let t = k as f32 / n as f32;
                out.push(eval_cubic(p0, p1, p2, p3, t));
            }
        }
    }
}

/// Bounds on the distance from a Bezier segment to its own chord, as multiples
/// of the control points' second differences. Note this is chord distance, not
/// the control-polygon distance the usual `N_inf(n)` constants bound; a chord
/// is what flattening actually substitutes for the curve.
///
/// Quadratic: `B(t) - chord(t) = 2t(1-t)(P1 - (P0 + P2)/2)`, so the maximum is
/// `|P0 - 2P1 + P2| / 4`, attained at `t = 1/2`. Exact, not an estimate.
///
/// Cubic: the same expansion gives
/// `B(t) - chord(t) = t(1-t)[(t-2)a - (1+t)b]` with `a = P0 - 2P1 + P2` and
/// `b = P1 - 2P2 + P3`. Both `t(1-t)(2-t)` and `t(1-t)(1+t)` peak at
/// `0.38490...` (at `t = 1 -+ 1/sqrt(3)`), so `0.385 (|a| + |b|)` bounds it.
const QUAD_CHORD_BOUND: f32 = 0.25;
const CUBIC_CHORD_BOUND: f32 = 0.385;

/// Uniform subdivision count meeting `tolerance` for a segment whose whole-curve
/// deviation bound is `deviation`. Splitting into `n` uniform pieces scales the
/// control-point second differences by `1/n^2`, so the bound scales the same
/// way and `n = ceil(sqrt(deviation / tolerance))` suffices.
fn subdivisions(deviation: f32, tolerance: f32) -> u32 {
    if tolerance <= 0.0 || tolerance.is_nan() || deviation <= tolerance {
        return 1;
    }
    let n = (deviation / tolerance).sqrt().ceil();
    (n as u32).clamp(1, MAX_SUBDIVISIONS)
}

fn eval_quad(p0: Vec2, p1: Vec2, p2: Vec2, t: f32) -> Vec2 {
    let u = 1.0 - t;
    p0 * (u * u) + p1 * (2.0 * u * t) + p2 * (t * t)
}

fn eval_cubic(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f32) -> Vec2 {
    let u = 1.0 - t;
    p0 * (u * u * u) + p1 * (3.0 * u * u * t) + p2 * (3.0 * u * t * t) + p3 * (t * t * t)
}

fn curve_start(curve: &OutlineCurve) -> Point {
    match *curve {
        OutlineCurve::Line(p0, _) | OutlineCurve::Quad(p0, ..) | OutlineCurve::Cubic(p0, ..) => p0,
    }
}

fn curve_end(curve: &OutlineCurve) -> Point {
    match *curve {
        OutlineCurve::Line(_, p)
        | OutlineCurve::Quad(_, _, p)
        | OutlineCurve::Cubic(_, _, _, p) => p,
    }
}

fn to_vec2(p: Point) -> Vec2 {
    Vec2::new(p.x, p.y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ab_glyph::point;

    fn square(cx: f32, cy: f32, half: f32, clockwise: bool) -> Vec<OutlineCurve> {
        let mut corners = [
            point(cx - half, cy - half),
            point(cx + half, cy - half),
            point(cx + half, cy + half),
            point(cx - half, cy + half),
        ];
        if clockwise {
            corners.reverse();
        }
        let mut curves = Vec::new();
        for i in 0..4 {
            curves.push(OutlineCurve::Line(corners[i], corners[(i + 1) % 4]));
        }
        // ab_glyph's builder always emits the explicit closing line, even when
        // it lands on a point already present.
        curves.push(OutlineCurve::Line(corners[0], corners[0]));
        curves
    }

    /// A curve whose start does not continue the previous curve's end opens a
    /// new contour, which is the only contour delimiter `ab_glyph` provides.
    #[test]
    fn discontinuous_start_opens_a_new_contour() {
        let mut curves = square(0.0, 0.0, 1.0, false);
        curves.extend(square(10.0, 0.0, 0.5, true));
        let contours = contours_from_curves(&curves, 1.0, 0.01);
        assert_eq!(contours.len(), 2);
        assert_eq!(contours[0].points.len(), 4);
        assert_eq!(contours[1].points.len(), 4);
    }

    /// The explicit closing line back to the move-to point must not survive as
    /// a duplicate vertex; a zero-length edge has no winding-test direction.
    #[test]
    fn closing_line_does_not_duplicate_the_first_point() {
        let contours = contours_from_curves(&square(0.0, 0.0, 1.0, false), 1.0, 0.01);
        assert_eq!(contours.len(), 1);
        let points = &contours[0].points;
        assert_eq!(points.len(), 4);
        assert!(points[0].distance(points[points.len() - 1]) > 0.5);
    }

    /// Font units are scaled to em exactly once, on the way out.
    #[test]
    fn font_units_scale_to_em() {
        let contours = contours_from_curves(&square(0.0, 0.0, 1024.0, false), 1.0 / 2048.0, 0.001);
        for p in &contours[0].points {
            assert!((p.x.abs() - 0.5).abs() < 1e-6, "x = {}", p.x);
            assert!((p.y.abs() - 0.5).abs() < 1e-6, "y = {}", p.y);
        }
    }

    /// The polyline a quadratic flattens to stays within the requested chord
    /// tolerance of the true curve, at every parameter and not just at the
    /// subdivision points.
    #[test]
    fn flattened_quadratic_stays_within_tolerance() {
        let (p0, p1, p2) = (
            Vec2::new(0.0, 0.0),
            Vec2::new(0.5, 1.0),
            Vec2::new(1.0, 0.0),
        );
        let tolerance = 1.0e-3;
        let mut out = vec![p0];
        flatten_into(
            &OutlineCurve::Quad(point(0.0, 0.0), point(0.5, 1.0), point(1.0, 0.0)),
            tolerance,
            &mut out,
        );
        assert!(out.len() > 2, "curve was not subdivided: {out:?}");
        assert_max_deviation(&out, tolerance, |t| eval_quad(p0, p1, p2, t));
    }

    /// Same for a cubic, whose chord bound is the looser of the two.
    #[test]
    fn flattened_cubic_stays_within_tolerance() {
        let (p0, p1, p2, p3) = (
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
        );
        let tolerance = 5.0e-4;
        let mut out = vec![p0];
        flatten_into(
            &OutlineCurve::Cubic(
                point(0.0, 0.0),
                point(0.0, 1.0),
                point(1.0, 1.0),
                point(1.0, 0.0),
            ),
            tolerance,
            &mut out,
        );
        assert!(out.len() > 2, "curve was not subdivided: {out:?}");
        assert_max_deviation(&out, tolerance, |t| eval_cubic(p0, p1, p2, p3, t));
    }

    /// Sample the true curve far more densely than it was subdivided and check
    /// every sample against the flattened polyline.
    fn assert_max_deviation(polyline: &[Vec2], tolerance: f32, curve: impl Fn(f32) -> Vec2) {
        const SAMPLES: u32 = 512;
        for k in 0..=SAMPLES {
            let t = k as f32 / SAMPLES as f32;
            let truth = curve(t);
            let nearest = polyline
                .windows(2)
                .map(|w| point_segment_distance(truth, w[0], w[1]))
                .fold(f32::INFINITY, f32::min);
            assert!(
                nearest <= tolerance * 1.02,
                "deviation {nearest} exceeds tolerance {tolerance} at t = {t}"
            );
        }
    }

    /// A straight-line "curve" needs no subdivision: the deviation bound is
    /// zero, so the count must be one segment, not `MAX_SUBDIVISIONS`.
    #[test]
    fn collinear_quadratic_is_not_subdivided() {
        let mut out = vec![Vec2::ZERO];
        flatten_into(
            &OutlineCurve::Quad(point(0.0, 0.0), point(0.5, 0.0), point(1.0, 0.0)),
            1.0e-4,
            &mut out,
        );
        assert_eq!(out.len(), 2);
    }

    /// Subdivision count is bounded even when the tolerance is absurd, so a
    /// hostile font cannot make flattening allocate without limit.
    #[test]
    fn subdivision_count_is_clamped() {
        assert_eq!(subdivisions(1.0e12, 1.0e-12), MAX_SUBDIVISIONS);
        assert_eq!(subdivisions(0.0, 1.0e-6), 1);
        assert_eq!(subdivisions(1.0, 0.0), 1);
    }

    fn point_segment_distance(p: Vec2, a: Vec2, b: Vec2) -> f32 {
        let ab = b - a;
        let len2 = ab.length_squared();
        if len2 <= f32::EPSILON {
            return p.distance(a);
        }
        let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
        p.distance(a + ab * t)
    }
}
