//! `BlendedSpace<A, B, F>` — a `Space` whose metric smoothly
//! interpolates between two source Spaces $A$ and $B$ via a
//! blending field $F: \mathbb{R}^3 \to [0, 1]$.
//!
//! THESIS §2.2 design goal: *"Seamless transitions between
//! geometries, not camera tricks."*
//!
//! See [`docs/devlog/BLENDED_SPACE.md`](../../../../docs/devlog/BLENDED_SPACE.md)
//! for the full design — math foundation, numerical scheme,
//! validation strategy, and lock-in audit.
//!
//! ## Module status
//!
//! Phase 4 deliverable, in progress. This file currently ships:
//!
//! 1. The [`BlendingField`] trait — the only mandatory user
//!    extension point.
//! 2. [`LinearBlendX`] — a smooth-step axis-aligned zone, the v0
//!    concrete blending field used by the demo.
//!
//! [`BlendedSpace`] itself (the `Space` impl) lands in subsequent
//! sub-tasks (the conformal-factor / Christoffel / RK4 work).

use glam::Vec3;

/// A scalar blending field over $\mathbb{R}^3$.
///
/// `weight(p)` selects how much of the second Space's metric
/// applies at `p`: `0.0` is "pure $A$", `1.0` is "pure $B$",
/// intermediate values blend continuously.
///
/// **Smoothness matters.** The geodesic integrator differentiates
/// through this field to compute Christoffel symbols. A field
/// that's continuous but not continuously differentiable (e.g. a
/// linear ramp clamped to `[0, 1]` without smoothing) produces
/// integrator artifacts at the breakpoints. Use a smoothstep or
/// equivalent $C^1$ profile.
///
/// **Cost matters.** `weight` and `gradient` are evaluated many
/// times per ray-march step. Closed-form impls override the
/// default finite-difference `gradient` to halve the per-step
/// cost.
pub trait BlendingField: Copy + Send + Sync + 'static {
    /// Blend weight at `p`. Implementations must clamp to
    /// `[0, 1]`; values outside that range produce a metric that
    /// isn't a valid interpolation.
    fn weight(&self, p: Vec3) -> f32;

    /// Spatial gradient of [`Self::weight`]. Default impl uses
    /// central finite differences with `h = 1e-3`. Override for
    /// closed-form blends to (a) reduce evaluation cost and
    /// (b) avoid finite-difference noise near sharp profiles.
    fn gradient(&self, p: Vec3) -> Vec3 {
        const EPS: f32 = 1.0e-3;
        let dx = (self.weight(p + Vec3::X * EPS) - self.weight(p - Vec3::X * EPS)) / (2.0 * EPS);
        let dy = (self.weight(p + Vec3::Y * EPS) - self.weight(p - Vec3::Y * EPS)) / (2.0 * EPS);
        let dz = (self.weight(p + Vec3::Z * EPS) - self.weight(p - Vec3::Z * EPS)) / (2.0 * EPS);
        Vec3::new(dx, dy, dz)
    }
}

// ---------------------------------------------------------------------------
// LinearBlendX — axis-aligned smoothstep zone
// ---------------------------------------------------------------------------

/// Smoothstep blending zone along the X axis: pure $A$ at
/// `x ≤ start`, pure $B$ at `x ≥ end`, smooth $C^1$ transition
/// in between.
///
/// The smoothing profile is the standard cubic Hermite
/// $3t^2 - 2t^3$, which:
///
/// - Is continuous and continuously differentiable everywhere.
/// - Has zero gradient at both endpoints (so the metric reduces
///   exactly to $g_A$ / $g_B$ outside the zone, with no
///   integrator kicks at the boundary).
/// - Maps `t ∈ [0, 1]` to `[0, 1]` monotonically.
///
/// The Phase 4 milestone demo (`examples/blended`) uses this
/// with `start = -2.0`, `end = +2.0` so the player rolls from
/// $\mathbb{E}^3$ into $H^3$ over 4 units of X-axis distance.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LinearBlendX {
    pub start: f32,
    pub end: f32,
}

impl LinearBlendX {
    /// New zone with the given start / end x-coordinates.
    /// Requires `end > start`; reversed inputs are silently
    /// swapped to maintain `weight ∈ [0, 1]` semantics.
    pub fn new(start: f32, end: f32) -> Self {
        if end >= start {
            Self { start, end }
        } else {
            Self {
                start: end,
                end: start,
            }
        }
    }

    /// Width of the blending zone in world units.
    fn width(&self) -> f32 {
        self.end - self.start
    }
}

impl BlendingField for LinearBlendX {
    fn weight(&self, p: Vec3) -> f32 {
        let w = self.width();
        if w <= 0.0 {
            // Degenerate: treat as a step function at `start`.
            return if p.x < self.start { 0.0 } else { 1.0 };
        }
        let t = ((p.x - self.start) / w).clamp(0.0, 1.0);
        // Smoothstep: 3t² - 2t³.
        t * t * (3.0 - 2.0 * t)
    }

    fn gradient(&self, p: Vec3) -> Vec3 {
        let w = self.width();
        if w <= 0.0 {
            // Degenerate step function — zero gradient
            // everywhere except the point of discontinuity, which
            // we treat as zero (the integrator avoids it).
            return Vec3::ZERO;
        }
        let raw_t = (p.x - self.start) / w;
        // Outside `[0, 1]` the field is constant and its
        // gradient vanishes.
        if !(0.0..=1.0).contains(&raw_t) {
            return Vec3::ZERO;
        }
        // d/dx [3t² - 2t³] · dt/dx = (6t - 6t²)·(1/w) = 6t(1-t)/w.
        let t = raw_t;
        let dx = 6.0 * t * (1.0 - t) / w;
        Vec3::new(dx, 0.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() <= tol, "expected {a} ≈ {b} (tol {tol})");
    }

    /// Smoothstep boundary values: 0 at start, 1 at end, 0.5 at
    /// midpoint. Pin the canonical profile.
    #[test]
    fn linear_blend_x_smoothstep_endpoints() {
        let f = LinearBlendX::new(-1.0, 1.0);
        close(f.weight(Vec3::new(-1.0, 0.0, 0.0)), 0.0, 1e-6);
        close(f.weight(Vec3::new(1.0, 0.0, 0.0)), 1.0, 1e-6);
        close(f.weight(Vec3::ZERO), 0.5, 1e-6);
    }

    /// Outside the zone, the field is constant (0 or 1) and the
    /// gradient is exactly zero. This is what makes the metric
    /// reduce to pure $A$ / pure $B$ at the extremes — the
    /// integrator sees no Christoffel kick.
    #[test]
    fn linear_blend_x_is_constant_outside_zone() {
        let f = LinearBlendX::new(-1.0, 1.0);
        // Far on the A side.
        close(f.weight(Vec3::new(-100.0, 0.0, 0.0)), 0.0, 0.0);
        assert_eq!(f.gradient(Vec3::new(-100.0, 0.0, 0.0)), Vec3::ZERO);
        // Far on the B side.
        close(f.weight(Vec3::new(100.0, 0.0, 0.0)), 1.0, 0.0);
        assert_eq!(f.gradient(Vec3::new(100.0, 0.0, 0.0)), Vec3::ZERO);
        // Just at the boundary points (smoothstep gradient is 0
        // there too — that's the whole point of using smoothstep
        // over a linear ramp).
        close(f.gradient(Vec3::new(-1.0, 0.0, 0.0)).x, 0.0, 1e-6);
        close(f.gradient(Vec3::new(1.0, 0.0, 0.0)).x, 0.0, 1e-6);
    }

    /// Inside the zone, the gradient is non-zero along X and zero
    /// along Y / Z (axis-aligned blend). Gradient magnitude peaks
    /// at the midpoint where the smoothstep curve is steepest.
    #[test]
    fn linear_blend_x_gradient_is_axis_aligned() {
        let f = LinearBlendX::new(-1.0, 1.0);
        let g = f.gradient(Vec3::ZERO);
        assert!(g.x > 0.0, "midpoint gradient should be positive along +x");
        close(g.y, 0.0, 0.0);
        close(g.z, 0.0, 0.0);
        // Midpoint: t = 0.5, gradient = 6·0.5·0.5/(2.0) = 0.75.
        close(g.x, 0.75, 1e-6);
    }

    /// Closed-form gradient agrees with the central-finite-
    /// difference default impl. Catches sign or scale errors in
    /// the analytic gradient.
    #[test]
    fn linear_blend_x_closed_form_matches_finite_diff() {
        let f = LinearBlendX::new(-1.0, 1.0);
        // Reference: a wrapper that uses the default `gradient`
        // (finite difference). Since `gradient` has a default
        // impl on the trait, we replicate that path manually
        // here without hitting the override.
        fn finite_diff_gradient<F: BlendingField>(field: &F, p: Vec3) -> Vec3 {
            const EPS: f32 = 1.0e-3;
            let dx =
                (field.weight(p + Vec3::X * EPS) - field.weight(p - Vec3::X * EPS)) / (2.0 * EPS);
            let dy =
                (field.weight(p + Vec3::Y * EPS) - field.weight(p - Vec3::Y * EPS)) / (2.0 * EPS);
            let dz =
                (field.weight(p + Vec3::Z * EPS) - field.weight(p - Vec3::Z * EPS)) / (2.0 * EPS);
            Vec3::new(dx, dy, dz)
        }

        for x in [-0.7_f32, -0.3, 0.0, 0.3, 0.7] {
            let p = Vec3::new(x, 0.0, 0.0);
            let analytic = f.gradient(p);
            let numeric = finite_diff_gradient(&f, p);
            close(analytic.x, numeric.x, 5e-4);
            close(analytic.y, numeric.y, 1e-6);
            close(analytic.z, numeric.z, 1e-6);
        }
    }

    /// Reversed inputs (`end < start`) get auto-swapped so the
    /// resulting field still ramps from 0 to 1 over the zone.
    #[test]
    fn linear_blend_x_handles_reversed_inputs() {
        let f = LinearBlendX::new(1.0, -1.0);
        // After swap, equivalent to `new(-1.0, 1.0)`.
        close(f.weight(Vec3::new(-1.0, 0.0, 0.0)), 0.0, 1e-6);
        close(f.weight(Vec3::new(1.0, 0.0, 0.0)), 1.0, 1e-6);
    }

    /// Smoothstep is monotonic non-decreasing across the zone.
    /// Pin so a future "use a different smoothing profile"
    /// regression doesn't accidentally introduce overshoot.
    #[test]
    fn linear_blend_x_is_monotonic() {
        let f = LinearBlendX::new(-1.0, 1.0);
        let xs: Vec<f32> = (0..=20).map(|i| -1.0 + (i as f32) / 10.0).collect();
        let mut prev = f.weight(Vec3::new(xs[0], 0.0, 0.0));
        for &x in &xs[1..] {
            let curr = f.weight(Vec3::new(x, 0.0, 0.0));
            assert!(
                curr >= prev - 1e-6,
                "non-monotonic: at x={x}, weight={curr} < previous {prev}"
            );
            prev = curr;
        }
    }
}
