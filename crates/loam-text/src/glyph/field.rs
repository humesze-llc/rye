//! Sign comes from the nonzero winding rule, which is the fill rule both
//! TrueType and CFF outlines are defined by. Winding rather than contour
//! orientation is what makes hole handling format-independent: TrueType writes
//! outer contours clockwise and CFF counter-clockwise, but both fill the
//! nonzero region.

use glam::Vec2;

use super::outline::Contour;

// Ring of all-outside samples around the glyph's bounding box. Two cells
// guarantee the zero isoline is strictly interior to the grid, which is what
// lets `super::solid` treat every clipped-cell edge not on the isoline as
// shared with a neighbour rather than as an open boundary.
const PADDING_CELLS: usize = 2;

// Contour construction already drops coincident neighbours at 1e-6 em, so this
// only catches edges that survive scaling into world units at a tiny `em_size`.
const DEGENERATE_EDGE_LENGTH2: f32 = 1.0e-24;

/// Negative inside the glyph.
#[derive(Clone, Debug)]
pub struct DistanceField2D {
    origin: Vec2,
    cell: f32,
    samples_x: usize,
    samples_y: usize,
    /// Row-major, `samples_x` per row, `y` increasing with row index.
    samples: Vec<f32>,
}

impl DistanceField2D {
    // `None` when the contours enclose no area, which for a glyph means a
    // degenerate outline rather than a blank. The caller fixes `cell` rather
    // than a per-glyph subdivision count so every letter of a word is sampled
    // at the same fidelity regardless of how tall its own outline is.
    pub(super) fn bake(contours: &[Contour], cell: f32) -> Option<Self> {
        let (min, max) = bounds(contours)?;
        let extent = max - min;
        // Either extent collapsing leaves the winding rule nothing to call
        // inside. The vector compare also rejects NaN coordinates.
        if !extent.cmpgt(Vec2::ZERO).all() {
            return None;
        }

        let cells_x = (extent.x / cell).ceil() as usize + 2 * PADDING_CELLS;
        let cells_y = (extent.y / cell).ceil() as usize + 2 * PADDING_CELLS;
        let origin = min - Vec2::splat(PADDING_CELLS as f32 * cell);

        let samples_x = cells_x + 1;
        let samples_y = cells_y + 1;
        let mut samples = Vec::with_capacity(samples_x * samples_y);
        for j in 0..samples_y {
            for i in 0..samples_x {
                let p = origin + Vec2::new(i as f32, j as f32) * cell;
                samples.push(signed_distance(contours, p));
            }
        }

        Some(Self {
            origin,
            cell,
            samples_x,
            samples_y,
            samples,
        })
    }

    /// Bilinear between grid samples, so an approximation of the exact contour
    /// distance: adequate for sphere tracing, not for exact containment.
    ///
    /// The exact contour distance is 1-Lipschitz in L2; this interpolant is
    /// not. Adjacent samples differ by at most one cell, so each partial
    /// derivative of the bilinear form is bounded by 1 and
    /// `|sample(a) - sample(b)| <= |a - b|` holds in the L1 norm, hence only
    /// `sqrt(2) |a - b|` in L2. The L2 constant is attained: at a cell corner
    /// on the medial axis the three near corners carry the same distance and
    /// the fourth is one cell nearer the wall, which drives the interpolant's
    /// gradient to `(1, 1)`. A sphere tracer stepping by this value must scale
    /// steps by `1 / sqrt(2)`, or step per axis, to avoid tunnelling.
    pub fn sample(&self, p: Vec2) -> f32 {
        let grid = (p - self.origin) / self.cell;
        let clamped = grid.clamp(
            Vec2::ZERO,
            Vec2::new((self.samples_x - 1) as f32, (self.samples_y - 1) as f32),
        );
        let inside_grid = self.bilinear(clamped);

        let overshoot = (grid - clamped).length() * self.cell;
        if overshoot == 0.0 {
            return inside_grid;
        }
        // Outside the padded grid two lower bounds hold: every contour lies
        // strictly inside the grid, so the true distance is at least the
        // distance back to the grid; and the exact contour distance is
        // 1-Lipschitz, so it is at least the clamped value less that same
        // offset. Take the larger, so a sphere tracer under-steps. Decaying at
        // one per unit rather than the sqrt(2) the interior interpolation can
        // reach keeps the extrapolation inside the per-axis bound.
        overshoot.max(inside_grid - overshoot)
    }

    pub(super) fn at(&self, i: usize, j: usize) -> f32 {
        self.samples[j * self.samples_x + i]
    }

    pub(super) fn corner(&self, i: usize, j: usize) -> Vec2 {
        self.origin + Vec2::new(i as f32, j as f32) * self.cell
    }

    pub(super) fn cell_counts(&self) -> (usize, usize) {
        (self.samples_x - 1, self.samples_y - 1)
    }

    /// Grid spacing in world units.
    pub fn cell_size(&self) -> f32 {
        self.cell
    }

    fn bilinear(&self, grid: Vec2) -> f32 {
        let i0 = (grid.x.floor() as usize).min(self.samples_x - 2);
        let j0 = (grid.y.floor() as usize).min(self.samples_y - 2);
        let tx = grid.x - i0 as f32;
        let ty = grid.y - j0 as f32;
        let d00 = self.at(i0, j0);
        let d10 = self.at(i0 + 1, j0);
        let d01 = self.at(i0, j0 + 1);
        let d11 = self.at(i0 + 1, j0 + 1);
        let bottom = d00 + (d10 - d00) * tx;
        let top = d01 + (d11 - d01) * tx;
        bottom + (top - bottom) * ty
    }
}

fn bounds(contours: &[Contour]) -> Option<(Vec2, Vec2)> {
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    let mut any = false;
    for contour in contours {
        for p in &contour.points {
            min = min.min(*p);
            max = max.max(*p);
            any = true;
        }
    }
    any.then_some((min, max))
}

// Exact distance to the nearest contour edge, negated inside the nonzero
// winding region.
fn signed_distance(contours: &[Contour], p: Vec2) -> f32 {
    let mut nearest = f32::INFINITY;
    let mut winding = 0i32;

    for contour in contours {
        let points = &contour.points;
        let n = points.len();
        for i in 0..n {
            let a = points[i];
            let b = points[(i + 1) % n];
            nearest = nearest.min(point_edge_distance(p, a, b));
            winding += crossing(p, a, b);
        }
    }

    if winding != 0 {
        -nearest
    } else {
        nearest
    }
}

// Ericson, "Real-Time Collision Detection" (2005), section 5.1.2: closest point
// on a segment is the projection parameter clamped to `[0, 1]`.
fn point_edge_distance(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let length2 = ab.length_squared();
    if length2 <= DEGENERATE_EDGE_LENGTH2 {
        return p.distance(a);
    }
    let t = ((p - a).dot(ab) / length2).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

// Winding-number contribution of edge `a -> b` for the ray from `p` towards
// `+x`. Sunday (2001), "Inclusion of a Point in a Polygon": the half-open `y`
// comparison counts each crossing exactly once, so a vertex lying on the ray
// does not double-count.
fn crossing(p: Vec2, a: Vec2, b: Vec2) -> i32 {
    let side = (b.x - a.x) * (p.y - a.y) - (p.x - a.x) * (b.y - a.y);
    if a.y <= p.y {
        if b.y > p.y && side > 0.0 {
            return 1;
        }
    } else if b.y <= p.y && side < 0.0 {
        return -1;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(min: Vec2, max: Vec2, counter_clockwise: bool) -> Contour {
        let mut points = vec![min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)];
        if !counter_clockwise {
            points.reverse();
        }
        Contour { points }
    }

    #[test]
    fn sign_is_negative_inside_and_magnitude_is_edge_distance() {
        let square = vec![rect(Vec2::splat(-1.0), Vec2::splat(1.0), true)];
        assert!((signed_distance(&square, Vec2::ZERO) + 1.0).abs() < 1e-6);
        assert!((signed_distance(&square, Vec2::new(0.5, 0.0)) + 0.5).abs() < 1e-6);
        assert!((signed_distance(&square, Vec2::new(2.0, 0.0)) - 1.0).abs() < 1e-6);
        assert!((signed_distance(&square, Vec2::new(2.0, 2.0)) - 2.0_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn sign_is_independent_of_contour_orientation() {
        let ccw = vec![rect(Vec2::splat(-1.0), Vec2::splat(1.0), true)];
        let cw = vec![rect(Vec2::splat(-1.0), Vec2::splat(1.0), false)];
        assert_eq!(
            signed_distance(&ccw, Vec2::ZERO),
            signed_distance(&cw, Vec2::ZERO)
        );
    }

    #[test]
    fn counter_wound_inner_contour_is_a_hole() {
        let ring = vec![
            rect(Vec2::splat(-2.0), Vec2::splat(2.0), true),
            rect(Vec2::splat(-1.0), Vec2::splat(1.0), false),
        ];
        assert!((signed_distance(&ring, Vec2::ZERO) - 1.0).abs() < 1e-6);
        assert!((signed_distance(&ring, Vec2::new(1.5, 0.0)) + 0.5).abs() < 1e-6);
        assert!(signed_distance(&ring, Vec2::new(3.0, 0.0)) > 0.0);
    }

    #[test]
    fn same_wound_inner_contour_is_not_a_hole() {
        let both_ccw = vec![
            rect(Vec2::splat(-2.0), Vec2::splat(2.0), true),
            rect(Vec2::splat(-1.0), Vec2::splat(1.0), true),
        ];
        assert!(signed_distance(&both_ccw, Vec2::ZERO) < 0.0);
    }

    #[test]
    fn grid_boundary_samples_are_all_outside() {
        let square = vec![rect(Vec2::splat(-1.0), Vec2::splat(1.0), true)];
        let field = DistanceField2D::bake(&square, 2.0 / 16.0).expect("bake");
        let (cx, cy) = field.cell_counts();
        for i in 0..=cx {
            assert!(field.at(i, 0) > 0.0);
            assert!(field.at(i, cy) > 0.0);
        }
        for j in 0..=cy {
            assert!(field.at(0, j) > 0.0);
            assert!(field.at(cx, j) > 0.0);
        }
    }

    // Probe separations are ~1e-3 and sampled values ~1e-1, so cancellation in
    // the difference costs a few times f32 epsilon of the larger operand.
    const LIPSCHITZ_SLACK: f32 = 1.0e-6;

    #[test]
    fn sampling_is_one_lipschitz_per_axis_including_off_grid() {
        let square = vec![rect(Vec2::splat(-1.0), Vec2::splat(1.0), true)];
        let field = DistanceField2D::bake(&square, 2.0 / 24.0).expect("bake");
        let cell = field.cell_size();

        // Pitches incommensurate with the cell so probes land at many sub-cell
        // phases, and the span reaches past the padded grid into the clamped
        // extrapolation.
        let bases = (-40..=40)
            .flat_map(|i| (-40..=40).map(move |j| Vec2::new(i as f32 * 0.0371, j as f32 * 0.0397)));
        let mut steps = Vec::new();
        for fraction in [1.0 / 64.0, 1.0 / 7.0, 1.0 / 2.3] {
            for k in 0..8 {
                let angle = k as f32 * std::f32::consts::FRAC_PI_4;
                steps.push(Vec2::new(angle.cos(), angle.sin()) * (cell * fraction));
            }
        }

        for a in bases {
            for step in &steps {
                let b = a + *step;
                let delta = (field.sample(a) - field.sample(b)).abs();
                let l1 = step.x.abs() + step.y.abs();
                assert!(
                    delta <= l1 + LIPSCHITZ_SLACK,
                    "|d({a}) - d({b})| = {delta} exceeds |dx| + |dy| = {l1}"
                );
                let l2 = std::f32::consts::SQRT_2 * step.length();
                assert!(
                    delta <= l2 + LIPSCHITZ_SLACK,
                    "|d({a}) - d({b})| = {delta} exceeds sqrt(2)|a - b| = {l2}"
                );
            }
        }
    }

    #[test]
    fn bilinear_sampling_exceeds_one_lipschitz_on_the_medial_axis() {
        let square = vec![rect(Vec2::splat(-1.0), Vec2::splat(1.0), true)];
        let field = DistanceField2D::bake(&square, 2.0 / 24.0).expect("bake");
        let (near, far) = (field.corner(4, 4), field.corner(5, 5));
        assert!(
            near.x == near.y && far.x == far.y,
            "cell is off the diagonal"
        );
        // The two off-diagonal corners are each an equal-distance point of the
        // same pair of walls, so they agree with `near` to within a rounding
        // of the projection that computes them.
        assert!((field.at(5, 4) - field.at(4, 4)).abs() <= f32::EPSILON);
        assert!((field.at(4, 5) - field.at(4, 4)).abs() <= f32::EPSILON);
        assert!(field.at(5, 5) < field.at(4, 4), "far corner is not deeper");

        // One sixty-fourth of the way back from the far corner, where the
        // closed form predicts a ratio of `(2 - 1/64) / sqrt(2) = 1.403`.
        let inner = far - (far - near) / 64.0;
        let delta = (field.sample(far) - field.sample(inner)).abs();
        let ratio = delta / far.distance(inner);
        assert!(ratio > 1.35, "diagonal L2 ratio {ratio} is below 1.35");
        assert!(
            ratio <= std::f32::consts::SQRT_2,
            "diagonal L2 ratio {ratio} exceeds sqrt(2)"
        );
    }

    #[test]
    fn off_grid_sampling_never_overshoots() {
        let square = vec![rect(Vec2::splat(-1.0), Vec2::splat(1.0), true)];
        let field = DistanceField2D::bake(&square, 2.0 / 16.0).expect("bake");
        for k in 1..40 {
            let p = Vec2::new(1.0 + k as f32 * 0.5, 0.0);
            let truth = signed_distance(&square, p);
            assert!(
                field.sample(p) <= truth + 1e-4,
                "sample {} exceeds true distance {truth} at {p}",
                field.sample(p)
            );
        }
    }

    #[test]
    fn baking_is_bit_reproducible() {
        let square = vec![rect(Vec2::splat(-1.0), Vec2::splat(1.0), true)];
        let a = DistanceField2D::bake(&square, 2.0 / 20.0).expect("bake");
        let b = DistanceField2D::bake(&square, 2.0 / 20.0).expect("bake");
        assert_eq!(a.samples, b.samples);
    }

    #[test]
    fn zero_area_contours_have_no_field() {
        let collapsed = vec![Contour {
            points: vec![Vec2::ZERO, Vec2::ZERO, Vec2::ZERO],
        }];
        assert!(DistanceField2D::bake(&collapsed, 0.1).is_none());
        assert!(DistanceField2D::bake(&[], 0.1).is_none());

        let flat = vec![Contour {
            points: vec![Vec2::ZERO, Vec2::new(1.0, 0.0), Vec2::new(2.0, 0.0)],
        }];
        assert!(DistanceField2D::bake(&flat, 0.1).is_none());
    }
}
