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

use std::marker::PhantomData;

use glam::Vec3;

use crate::space::Space;

// ---------------------------------------------------------------------------
// ConformallyFlat — extension trait for Spaces with scalar metric
// ---------------------------------------------------------------------------

/// A [`Space`] whose metric tensor is a scalar multiple of the
/// identity in its standard chart: $g_{ij}(p) = f(p) \, \delta_{ij}$
/// for some positive scalar function $f$.
///
/// Implemented by every constant-curvature 3-Space currently in
/// `rye-math`:
///
/// - [`crate::EuclideanR3`]: $f \equiv 1$.
/// - [`crate::HyperbolicH3`] (Poincaré ball model):
///   $f(p) = 4 / (1 - |p|^2)^2$, valid for $|p| < 1$.
/// - [`crate::SphericalS3`] (stereographic model):
///   $f(p) = 4 / (1 + |p|^2)^2$.
/// - [`HyperbolicH3UpperHalf`] (upper-half-space, *future* —
///   lands when the BlendedSpace demo needs an *unbounded* E³
///   side; see [`docs/devlog/BLENDED_SPACE.md`](../../../../docs/devlog/BLENDED_SPACE.md)):
///   $f(p) = 1 / z^2$, valid for $z > 0$.
///
/// **Why a separate trait, not a method on `Space`:** not every
/// Space is conformally flat. $\text{Sol}^3$ and $\text{Nil}^3$
/// (two of Thurston's eight 3-geometries) have anisotropic
/// metrics that aren't scalar-multiples of identity. Confining
/// the conformal-factor query to its own trait keeps the base
/// `Space` trait honest about what generalises.
///
/// **Why this matters for [`BlendedSpace`]:** when both source
/// Spaces are conformally flat, the blended metric is *also*
/// conformally flat (a blend of scalar multiples is a scalar
/// multiple), and the geodesic ODE collapses to a closed-form
/// expression in $\nabla \log f$. That's the fast path the
/// numerical integrator takes. A future non-conformally-flat
/// blend would need the full Christoffel-symbol machinery.
pub trait ConformallyFlat: Space {
    /// Conformal scale factor at `p`: the scalar $f$ such that
    /// $g_{ij}(p) = f(p) \, \delta_{ij}$.
    ///
    /// Must be positive and finite at `p` for `p` inside the
    /// chart. Boundary points (e.g. Poincaré ideal boundary
    /// $|p| \to 1$) may diverge to `f32::INFINITY`; the integrator
    /// detects and clamps.
    fn conformal_factor(&self, p: Vec3) -> f32;

    /// Logarithm of the conformal factor: $\phi(p) = \frac{1}{2} \ln f(p)$.
    /// Default impl computes from `conformal_factor`; closed-form
    /// overrides save a `ln` per evaluation in the hot path.
    fn conformal_log_half(&self, p: Vec3) -> f32 {
        0.5 * self.conformal_factor(p).ln()
    }

    /// Spatial gradient of [`Self::conformal_log_half`]. Default
    /// impl uses central finite differences; closed-form overrides
    /// halve cost and avoid finite-difference noise.
    fn conformal_log_half_gradient(&self, p: Vec3) -> Vec3 {
        const EPS: f32 = 1.0e-3;
        let dx = (self.conformal_log_half(p + Vec3::X * EPS)
            - self.conformal_log_half(p - Vec3::X * EPS))
            / (2.0 * EPS);
        let dy = (self.conformal_log_half(p + Vec3::Y * EPS)
            - self.conformal_log_half(p - Vec3::Y * EPS))
            / (2.0 * EPS);
        let dz = (self.conformal_log_half(p + Vec3::Z * EPS)
            - self.conformal_log_half(p - Vec3::Z * EPS))
            / (2.0 * EPS);
        Vec3::new(dx, dy, dz)
    }
}

// EuclideanR3: trivial conformally-flat case, $f \equiv 1$.
impl ConformallyFlat for crate::EuclideanR3 {
    fn conformal_factor(&self, _p: Vec3) -> f32 {
        1.0
    }
    fn conformal_log_half(&self, _p: Vec3) -> f32 {
        0.0
    }
    fn conformal_log_half_gradient(&self, _p: Vec3) -> Vec3 {
        Vec3::ZERO
    }
}

// HyperbolicH3 (Poincaré ball model): $f(p) = 4 / (1 - |p|^2)^2$
// for $|p| < 1$. Diverges at the ideal boundary $|p| = 1$;
// returns `INFINITY` there to flag chart-boundary crossings.
impl ConformallyFlat for crate::HyperbolicH3 {
    fn conformal_factor(&self, p: Vec3) -> f32 {
        let r2 = p.length_squared();
        let denom = (1.0 - r2).max(0.0);
        if denom <= 0.0 {
            return f32::INFINITY;
        }
        4.0 / (denom * denom)
    }
    fn conformal_log_half(&self, p: Vec3) -> f32 {
        // (1/2) ln(4 / (1 − |p|²)²) = ln 2 − ln(1 − |p|²).
        let r2 = p.length_squared();
        let denom = (1.0 - r2).max(0.0);
        if denom <= 0.0 {
            return f32::INFINITY;
        }
        std::f32::consts::LN_2 - denom.ln()
    }
    fn conformal_log_half_gradient(&self, p: Vec3) -> Vec3 {
        // ∇φ = ∇[ln 2 − ln(1 − |p|²)] = 2p / (1 − |p|²).
        let r2 = p.length_squared();
        let denom = (1.0 - r2).max(0.0);
        if denom <= 0.0 {
            return Vec3::ZERO;
        }
        p * (2.0 / denom)
    }
}

// SphericalS3 (stereographic projection from north pole):
// $f(p) = 4 / (1 + |p|^2)^2$.
impl ConformallyFlat for crate::SphericalS3 {
    fn conformal_factor(&self, p: Vec3) -> f32 {
        let r2 = p.length_squared();
        4.0 / (1.0 + r2).powi(2)
    }
    fn conformal_log_half(&self, p: Vec3) -> f32 {
        // (1/2) ln(4 / (1 + |p|²)²) = ln 2 − ln(1 + |p|²).
        std::f32::consts::LN_2 - (1.0 + p.length_squared()).ln()
    }
    fn conformal_log_half_gradient(&self, p: Vec3) -> Vec3 {
        // ∇φ = -2p / (1 + |p|²).
        let r2 = p.length_squared();
        p * (-2.0 / (1.0 + r2))
    }
}

// ---------------------------------------------------------------------------
// BlendedSpace — Space whose metric varies smoothly with position
// ---------------------------------------------------------------------------

/// A `Space` whose metric is the smooth blend of two source
/// Spaces' metrics, weighted by a [`BlendingField`]:
///
/// $$g(p) = (1 - \alpha(p)) \, g_A(p) + \alpha(p) \, g_B(p)$$
///
/// At zone extremes the metric reduces to pure $g_A$ or pure
/// $g_B$. In between it's an honest variable-metric Riemannian
/// manifold — geodesics curve continuously, parallel transport
/// is path-dependent, distances vary by region.
///
/// **Skeleton today.** This module currently ships the type
/// itself plus stubbed `Space` trait methods (`exp`, `log`,
/// `distance`, `parallel_transport`) that compile but are
/// incomplete — they fall back to the closed-form $A$ /
/// $B$ behaviour at zone extremes (where the blend is exact)
/// and `unimplemented!` in the variable-metric interior. The
/// numerical integrator (RK4 on the geodesic ODE) lands in
/// sub-task 4 of the Phase 4 implementation plan; see
/// [`docs/devlog/BLENDED_SPACE.md`](../../../../docs/devlog/BLENDED_SPACE.md).
///
/// **Trait bounds:** both source Spaces must use $\mathbb{R}^3$
/// for points and tangent vectors (every closed-form 3-Space in
/// `rye-math` does), and both must be [`ConformallyFlat`] (so the
/// blended metric is also conformally flat — the integrator's
/// fast path).
///
/// **Isometries:** none non-trivial. The variable metric breaks
/// translation and rotation symmetry by construction (the
/// blending field $F$ defines a privileged spatial dependence).
/// `Iso = ()`; `iso_apply` is the identity.
pub struct BlendedSpace<A, B, F>
where
    A: Space<Point = Vec3, Vector = Vec3>,
    B: Space<Point = Vec3, Vector = Vec3>,
    F: BlendingField,
{
    pub a: A,
    pub b: B,
    pub field: F,
    _marker: PhantomData<(A, B, F)>,
}

impl<A, B, F> BlendedSpace<A, B, F>
where
    A: Space<Point = Vec3, Vector = Vec3>,
    B: Space<Point = Vec3, Vector = Vec3>,
    F: BlendingField,
{
    pub fn new(a: A, b: B, field: F) -> Self {
        Self {
            a,
            b,
            field,
            _marker: PhantomData,
        }
    }
}

impl<A, B, F> Space for BlendedSpace<A, B, F>
where
    A: Space<Point = Vec3, Vector = Vec3> + ConformallyFlat,
    B: Space<Point = Vec3, Vector = Vec3> + ConformallyFlat,
    F: BlendingField,
{
    type Point = Vec3;
    type Vector = Vec3;
    /// No non-trivial isometries; see type-level docs.
    type Iso = ();

    fn distance(&self, a: Vec3, b: Vec3) -> f32 {
        // Zone-extreme fast path: when both endpoints are at the
        // same zone extreme, distance is exactly the source
        // Space's distance. Useful for tests + visual continuity
        // at the demo's far ends.
        let alpha_a = self.field.weight(a);
        let alpha_b = self.field.weight(b);
        if alpha_a == 0.0 && alpha_b == 0.0 {
            return self.a.distance(a, b);
        }
        if alpha_a == 1.0 && alpha_b == 1.0 {
            return self.b.distance(a, b);
        }
        // Variable-metric path: lands in sub-task 5 (`log`-based
        // distance, $|\log_a(b)|_a$ in the metric at $a$).
        unimplemented!("variable-metric distance — sub-task 5")
    }

    fn exp(&self, at: Vec3, v: Vec3) -> Vec3 {
        // RK4 on the geodesic ODE for conformally-flat metric.
        // For $g_{ij} = e^{2\phi} \delta_{ij}$ the equation reduces
        // to $\dot{\mathbf{v}} = |\mathbf{v}|^2 \nabla\phi
        // - 2 (\nabla\phi \cdot \mathbf{v}) \mathbf{v}$. Caller's
        // initial velocity `v` is interpreted as Euclidean — we
        // travel for unit parameter time, so the geodesic length
        // covered is $|\mathbf{v}|_g = |\mathbf{v}|_E \sqrt{f(at)}$.
        rk4_geodesic(self, at, v, GEODESIC_DEFAULT_STEPS).0
    }

    fn log(&self, from: Vec3, to: Vec3) -> Vec3 {
        let alpha_from = self.field.weight(from);
        let alpha_to = self.field.weight(to);
        if alpha_from == 0.0 && alpha_to == 0.0 && self.field.gradient(from) == Vec3::ZERO {
            return self.a.log(from, to);
        }
        if alpha_from == 1.0 && alpha_to == 1.0 && self.field.gradient(from) == Vec3::ZERO {
            return self.b.log(from, to);
        }
        unimplemented!("variable-metric log — sub-task 5 (Gauss-Newton shooting)")
    }

    fn parallel_transport(&self, from: Vec3, to: Vec3, v: Vec3) -> Vec3 {
        let alpha_from = self.field.weight(from);
        let alpha_to = self.field.weight(to);
        if alpha_from == 0.0 && alpha_to == 0.0 && self.field.gradient(from) == Vec3::ZERO {
            return self.a.parallel_transport(from, to, v);
        }
        if alpha_from == 1.0 && alpha_to == 1.0 && self.field.gradient(from) == Vec3::ZERO {
            return self.b.parallel_transport(from, to, v);
        }
        unimplemented!("variable-metric parallel transport — sub-task 6")
    }

    fn iso_identity(&self) {}
    fn iso_compose(&self, _a: (), _b: ()) {}
    fn iso_inverse(&self, _a: ()) {}
    fn iso_apply(&self, _iso: (), p: Vec3) -> Vec3 {
        p
    }
    fn iso_transport(&self, _iso: (), _at: Vec3, v: Vec3) -> Vec3 {
        v
    }
}

// ---------------------------------------------------------------------------
// RK4 geodesic integrator
// ---------------------------------------------------------------------------

/// Default number of RK4 steps per unit-parameter integration.
/// Empirically: 32 gives ~6 digits of accuracy on moderately
/// curved metrics; 64 gives ~7 (diminishing returns). Per the
/// design doc this is fixed; adaptive step refinement is
/// deferred until measurement shows we need it.
pub const GEODESIC_DEFAULT_STEPS: u32 = 32;

/// Single step of RK4 on the geodesic ODE for a conformally
/// flat metric. State = `(p, v)`; ODE RHS:
///
///   $\dot{\mathbf{p}} = \mathbf{v}$
///   $\dot{\mathbf{v}} = |\mathbf{v}|^2 \nabla\phi(\mathbf{p})
///                       - 2 (\nabla\phi \cdot \mathbf{v}) \mathbf{v}$
///
/// Returns the new state after stepping by `h` in parameter
/// time. The caller chains this `n_steps` times to reach unit
/// parameter time.
fn rk4_geodesic_step<S: ConformallyFlat>(
    space: &S,
    p: Vec3,
    v: Vec3,
    h: f32,
) -> (Vec3, Vec3) {
    // RHS: returns (dp/dt, dv/dt) given (p, v).
    let rhs = |p: Vec3, v: Vec3| -> (Vec3, Vec3) {
        let grad_phi = space.conformal_log_half_gradient(p);
        let v_sq = v.length_squared();
        let dot = grad_phi.dot(v);
        (v, grad_phi * v_sq - v * (2.0 * dot))
    };

    let (k1_p, k1_v) = rhs(p, v);
    let (k2_p, k2_v) = rhs(p + k1_p * (h * 0.5), v + k1_v * (h * 0.5));
    let (k3_p, k3_v) = rhs(p + k2_p * (h * 0.5), v + k2_v * (h * 0.5));
    let (k4_p, k4_v) = rhs(p + k3_p * h, v + k3_v * h);

    let dp = (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p) * (h / 6.0);
    let dv = (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) * (h / 6.0);
    (p + dp, v + dv)
}

/// Integrate the geodesic ODE starting at `(at, v)` for unit
/// parameter time, using `n_steps` RK4 steps. Returns the final
/// `(point, velocity)` pair.
///
/// Public so other modules (e.g. parallel-transport, log
/// shooting) can reuse the same integrator without duplicating
/// the math.
pub fn rk4_geodesic<S: ConformallyFlat>(
    space: &S,
    at: Vec3,
    v: Vec3,
    n_steps: u32,
) -> (Vec3, Vec3) {
    let h = 1.0 / n_steps as f32;
    let mut p = at;
    let mut vel = v;
    for _ in 0..n_steps {
        let (np, nv) = rk4_geodesic_step(space, p, vel, h);
        // Defensive: clamp non-finite states (chart-boundary
        // crossings, integrator blow-ups) to the previous valid
        // state so downstream callers don't propagate NaN.
        if np.is_finite() && nv.is_finite() {
            p = np;
            vel = nv;
        } else {
            tracing::warn!(
                "rk4_geodesic step produced non-finite state; clamping to previous step"
            );
            break;
        }
    }
    (p, vel)
}

// `BlendedSpace` is itself conformally flat when both sources are
// — the blend of scalar multiples of identity is still a scalar
// multiple of identity. This is the property the integrator
// exploits.
impl<A, B, F> ConformallyFlat for BlendedSpace<A, B, F>
where
    A: Space<Point = Vec3, Vector = Vec3> + ConformallyFlat,
    B: Space<Point = Vec3, Vector = Vec3> + ConformallyFlat,
    F: BlendingField,
{
    fn conformal_factor(&self, p: Vec3) -> f32 {
        let alpha = self.field.weight(p);
        let f_a = self.a.conformal_factor(p);
        let f_b = self.b.conformal_factor(p);
        // Defensive: if either source diverges (`f32::INFINITY`
        // outside its chart), and we're at the corresponding zone
        // extreme, the blend takes the *other* Space's value
        // exactly. Otherwise the divergence propagates — the
        // chart is invalid here.
        if alpha <= 0.0 {
            return f_a;
        }
        if alpha >= 1.0 {
            return f_b;
        }
        (1.0 - alpha) * f_a + alpha * f_b
    }
}

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

    // ------ ConformallyFlat impls ------

    /// EuclideanR3's conformal factor is identically 1; its
    /// log-half is identically 0; its gradient is identically
    /// zero. Pin the trivial case.
    #[test]
    fn euclidean_r3_conformal_factor_is_unity() {
        use crate::EuclideanR3;
        let s = EuclideanR3;
        for p in [
            Vec3::ZERO,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-100.0, 50.0, 7.0),
        ] {
            close(s.conformal_factor(p), 1.0, 0.0);
            close(s.conformal_log_half(p), 0.0, 0.0);
            assert_eq!(s.conformal_log_half_gradient(p), Vec3::ZERO);
        }
    }

    /// Poincaré-ball HyperbolicH3 conformal factor: $4 / (1 -
    /// |p|^2)^2$. Pin standard values (origin, halfway out,
    /// near boundary).
    #[test]
    fn hyperbolic_h3_conformal_factor_pin_values() {
        use crate::HyperbolicH3;
        let s = HyperbolicH3;
        // Origin: f(0) = 4 / 1 = 4.
        close(s.conformal_factor(Vec3::ZERO), 4.0, 1e-6);
        // Halfway out: |p| = 0.5, |p|² = 0.25, denom = 0.75,
        // f = 4 / 0.5625 ≈ 7.111.
        close(
            s.conformal_factor(Vec3::new(0.5, 0.0, 0.0)),
            4.0 / 0.5625,
            1e-5,
        );
        // Near boundary: |p|² = 0.99, denom = 0.01, f = 4 /
        // 0.0001 = 40000.
        close(
            s.conformal_factor(Vec3::new(0.0, 0.0, 0.99499f32.sqrt())),
            4.0 / (1.0 - 0.99499_f32).powi(2),
            10.0, // huge magnitude — generous tolerance
        );
    }

    /// Closed-form `conformal_log_half_gradient` for HyperbolicH3
    /// agrees with central finite differences. Catches sign /
    /// scale errors in the analytic gradient.
    #[test]
    fn hyperbolic_h3_log_half_gradient_matches_finite_diff() {
        use crate::HyperbolicH3;
        let s = HyperbolicH3;
        const EPS: f32 = 1e-3;
        let fd = |p: Vec3| -> Vec3 {
            let dx =
                (s.conformal_log_half(p + Vec3::X * EPS) - s.conformal_log_half(p - Vec3::X * EPS))
                    / (2.0 * EPS);
            let dy =
                (s.conformal_log_half(p + Vec3::Y * EPS) - s.conformal_log_half(p - Vec3::Y * EPS))
                    / (2.0 * EPS);
            let dz =
                (s.conformal_log_half(p + Vec3::Z * EPS) - s.conformal_log_half(p - Vec3::Z * EPS))
                    / (2.0 * EPS);
            Vec3::new(dx, dy, dz)
        };
        for p in [
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.3, -0.2, 0.1),
            Vec3::new(-0.4, 0.5, 0.2),
        ] {
            let analytic = s.conformal_log_half_gradient(p);
            let numeric = fd(p);
            close(analytic.x, numeric.x, 5e-3);
            close(analytic.y, numeric.y, 5e-3);
            close(analytic.z, numeric.z, 5e-3);
        }
    }

    /// SphericalS3 conformal factor: $4 / (1 + |p|^2)^2$. Pin
    /// origin and a generic point.
    #[test]
    fn spherical_s3_conformal_factor_pin_values() {
        use crate::SphericalS3;
        let s = SphericalS3;
        close(s.conformal_factor(Vec3::ZERO), 4.0, 1e-6);
        // |p|² = 1: f = 4 / 4 = 1.
        close(s.conformal_factor(Vec3::new(1.0, 0.0, 0.0)), 1.0, 1e-6);
        // |p|² = 3: f = 4 / 16 = 0.25.
        close(
            s.conformal_factor(Vec3::new(1.0, 1.0, 1.0)),
            0.25,
            1e-6,
        );
    }

    // ------ BlendedSpace skeleton ------

    /// At a zone extreme (pure A), `BlendedSpace::distance`
    /// matches `A::distance`. Pin the fast path that lets the
    /// demo's far ends be visually identical to pure E³ / pure H³.
    #[test]
    fn blended_space_distance_at_alpha_zero_matches_a() {
        use crate::EuclideanR3;
        let bs = BlendedSpace::new(
            EuclideanR3,
            EuclideanR3, // dummy; alpha=0 means we never see B
            LinearBlendX::new(10.0, 20.0),
        );
        // Both points at x < 10: alpha = 0, fast path to A.
        let a = Vec3::new(-1.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 0.0, 0.0);
        let d_blend = bs.distance(a, b);
        let d_a = EuclideanR3.distance(a, b);
        close(d_blend, d_a, 1e-6);
    }

    /// At a zone extreme (pure B), distance matches `B::distance`.
    #[test]
    fn blended_space_distance_at_alpha_one_matches_b() {
        use crate::EuclideanR3;
        let bs = BlendedSpace::new(
            EuclideanR3,
            EuclideanR3,
            LinearBlendX::new(-20.0, -10.0),
        );
        // Both points at x > -10: alpha = 1, fast path to B.
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(3.0, 0.0, 0.0);
        let d_blend = bs.distance(a, b);
        let d_b = EuclideanR3.distance(a, b);
        close(d_blend, d_b, 1e-6);
    }

    /// `BlendedSpace::iso_apply` is the identity (no non-trivial
    /// isometries; `Iso = ()`). Pin the convention.
    #[test]
    fn blended_space_iso_is_trivial() {
        use crate::EuclideanR3;
        use crate::Space;
        let bs = BlendedSpace::new(EuclideanR3, EuclideanR3, LinearBlendX::new(-1.0, 1.0));
        let p = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(bs.iso_apply((), p), p);
        let v = Vec3::new(0.5, -0.5, 0.5);
        assert_eq!(bs.iso_transport((), p, v), v);
    }

    /// `BlendedSpace` is itself `ConformallyFlat`: at a zone
    /// extreme, its conformal factor equals the relevant source
    /// Space's. In between, it's the linear blend of the two
    /// sources' factors weighted by alpha.
    #[test]
    fn blended_space_conformal_factor_blends_linearly() {
        use crate::{EuclideanR3, HyperbolicH3};
        // E³ blends to H³(Poincaré) along x ∈ [-1, 1].
        let bs = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-1.0, 1.0));

        // At x = -1 (alpha=0): factor = 1 (pure E³).
        close(bs.conformal_factor(Vec3::new(-1.0, 0.0, 0.0)), 1.0, 1e-6);
        // At x = 1 (alpha=1): factor = HyperbolicH3 at x=1 — but
        // x=1 is on the Poincaré boundary so f → ∞. Use a point
        // *near* x=1 but slightly inside.
        let p = Vec3::new(0.99, 0.0, 0.0);
        let alpha = LinearBlendX::new(-1.0, 1.0).weight(p);
        let f_e = 1.0;
        let f_h = HyperbolicH3.conformal_factor(p);
        let expected = (1.0 - alpha) * f_e + alpha * f_h;
        close(bs.conformal_factor(p), expected, 1e-2);
    }

    // ------ RK4 geodesic integrator ------

    /// In flat E³ the geodesic ODE has zero curvature term, so
    /// RK4 should reproduce straight-line motion exactly (within
    /// f32 rounding): `exp_p(v) = p + v` for all `p`, `v`.
    #[test]
    fn rk4_in_pure_e3_is_straight_line() {
        use crate::EuclideanR3;
        let bs = BlendedSpace::new(
            EuclideanR3,
            EuclideanR3,
            LinearBlendX::new(100.0, 200.0), // far away — alpha ≡ 0 in our test region
        );
        for (p, v) in [
            (Vec3::ZERO, Vec3::X),
            (Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.5, -0.3, 0.7)),
            (Vec3::new(-5.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)),
        ] {
            let (final_p, final_v) = rk4_geodesic(&bs, p, v, GEODESIC_DEFAULT_STEPS);
            let expected = p + v;
            close((final_p - expected).length(), 0.0, 1e-5);
            close((final_v - v).length(), 0.0, 1e-5);
        }
    }

    /// In pure HyperbolicH3 (Poincaré ball), `exp` from the
    /// origin along an axis-aligned **Euclidean** tangent vector
    /// `v` should land at the closed-form Poincaré geodesic
    /// endpoint. The convention (matching the existing
    /// `HyperbolicH3::exp`): `v` is Euclidean, so the Riemannian
    /// length is $|v|_g = \sqrt{f(p)} \cdot |v|_E$, and at the
    /// origin where $f(0) = 4$:
    ///
    /// $$|\exp_0(v)|_E = \tanh(|v|_g / 2) = \tanh(|v|_E).$$
    ///
    /// Pins the convention *and* validates the integrator
    /// against the engine's existing closed-form ground truth.
    #[test]
    fn rk4_in_pure_h3_matches_closed_form_at_origin() {
        use crate::{HyperbolicH3, Space};

        for &mag in &[0.1_f32, 0.3, 0.5] {
            let v = Vec3::new(mag, 0.0, 0.0);
            let (final_p, _) = rk4_geodesic(&HyperbolicH3, Vec3::ZERO, v, GEODESIC_DEFAULT_STEPS);
            // Closed form at origin (sqrt(f(0)) = 2):
            // |exp_0(v)|_E = tanh(|v|_E).
            let expected_radius = mag.tanh();
            close(final_p.x, expected_radius, 5e-3);
            close(final_p.y, 0.0, 1e-4);
            close(final_p.z, 0.0, 1e-4);
        }

        // Cross-check: numerical integrator agrees with the
        // engine's existing closed-form `HyperbolicH3::exp`.
        let v = Vec3::new(0.4, 0.1, 0.0);
        let (numerical, _) = rk4_geodesic(&HyperbolicH3, Vec3::ZERO, v, GEODESIC_DEFAULT_STEPS);
        let closed_form = HyperbolicH3.exp(Vec3::ZERO, v);
        close((numerical - closed_form).length(), 0.0, 5e-3);

        // Drive through `BlendedSpace::exp` (alpha ≡ 1 in the
        // test region) — the variable-metric path collapses to
        // pure H³ here.
        let bs = BlendedSpace::new(
            HyperbolicH3, // dummy; alpha=1 never reaches A
            HyperbolicH3,
            LinearBlendX::new(-100.0, -50.0),
        );
        let v = Vec3::new(0.5, 0.0, 0.0);
        let final_p_blended = bs.exp(Vec3::ZERO, v);
        close(final_p_blended.x, 0.5_f32.tanh(), 5e-3);
    }

    /// Geodesic round-trip in pure E³: `exp_p(v)` then `exp_q(-v)`
    /// returns to `p` (within RK4 noise). Time-reversibility of
    /// the integrator on a flat metric. Catches accumulated drift.
    #[test]
    fn rk4_in_e3_is_time_reversible() {
        use crate::EuclideanR3;
        let p = Vec3::new(1.0, 2.0, 3.0);
        let v = Vec3::new(0.5, -0.3, 0.7);
        let (q, vq) = rk4_geodesic(&EuclideanR3, p, v, GEODESIC_DEFAULT_STEPS);
        let (back, _) = rk4_geodesic(&EuclideanR3, q, -vq, GEODESIC_DEFAULT_STEPS);
        close((back - p).length(), 0.0, 1e-5);
    }

    /// `BlendedSpace::exp` at a zone extreme matches the source
    /// Space's exp (both being E³ here). End-to-end pin: the
    /// trait method actually goes through the integrator and
    /// recovers the closed-form answer.
    #[test]
    fn blended_space_exp_at_alpha_zero_matches_e3() {
        use crate::EuclideanR3;
        use crate::Space;
        let bs = BlendedSpace::new(EuclideanR3, EuclideanR3, LinearBlendX::new(50.0, 100.0));
        let p = Vec3::new(1.0, 2.0, 3.0);
        let v = Vec3::new(0.5, -0.3, 0.7);
        let result = bs.exp(p, v);
        // Pure E³ ⇒ straight-line motion ⇒ result = p + v.
        let expected = p + v;
        close((result - expected).length(), 0.0, 1e-5);
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
