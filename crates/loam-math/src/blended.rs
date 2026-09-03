use std::borrow::Cow;
use std::marker::PhantomData;

use glam::{Mat3, Vec3};

use crate::space::{Space, WgslSpace};

/// `g_ij(p) = f(p)·δ_ij` in the chart `Space::Point` carries, not in some
/// other chart.
pub trait ConformallyFlat: Space {
    /// May return `f32::INFINITY` at the chart boundary.
    fn conformal_factor(&self, p: Vec3) -> f32;

    /// `R = −(4/f)·[∇²φ + |∇φ|²/2]`; the default is finite-difference.
    fn scalar_curvature(&self, p: Vec3) -> f32 {
        const EPS: f32 = 5.0e-3;
        let phi_at = self.conformal_log_half(p);
        let lap = (self.conformal_log_half(p + Vec3::X * EPS)
            + self.conformal_log_half(p - Vec3::X * EPS)
            + self.conformal_log_half(p + Vec3::Y * EPS)
            + self.conformal_log_half(p - Vec3::Y * EPS)
            + self.conformal_log_half(p + Vec3::Z * EPS)
            + self.conformal_log_half(p - Vec3::Z * EPS)
            - 6.0 * phi_at)
            / (EPS * EPS);
        let grad = self.conformal_log_half_gradient(p);
        let grad_sq = grad.length_squared();
        let f_p = self.conformal_factor(p);
        if !f_p.is_finite() || f_p <= 0.0 {
            return 0.0;
        }
        -(4.0 / f_p) * (lap + 0.5 * grad_sq)
    }

    /// Logarithm of the conformal factor: φ(p) = (1/2) ln f(p).
    fn conformal_log_half(&self, p: Vec3) -> f32 {
        0.5 * self.conformal_factor(p).ln()
    }

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
    fn scalar_curvature(&self, _p: Vec3) -> f32 {
        0.0
    }
}

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
        let r2 = p.length_squared();
        let denom = (1.0 - r2).max(0.0);
        if denom <= 0.0 {
            return f32::INFINITY;
        }
        std::f32::consts::LN_2 - denom.ln()
    }
    fn conformal_log_half_gradient(&self, p: Vec3) -> Vec3 {
        let r2 = p.length_squared();
        let denom = (1.0 - r2).max(0.0);
        if denom <= 0.0 {
            return Vec3::ZERO;
        }
        p * (2.0 / denom)
    }
    fn scalar_curvature(&self, _p: Vec3) -> f32 {
        // `R = n(n − 1)K` at `K = −1`, `n = 3`.
        -6.0
    }
}

/// Metric `g = (1 − α)·g_A + α·g_B` with `α` from a [`BlendingField`]; the
/// field breaks every symmetry, so there is no [`crate::IsometryGroup`] impl.
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

    fn distance(&self, a: Vec3, b: Vec3) -> f32 {
        let alpha_a = self.field.weight(a);
        let alpha_b = self.field.weight(b);
        if alpha_a == 0.0 && alpha_b == 0.0 {
            return self.a.distance(a, b);
        }
        if alpha_a == 1.0 && alpha_b == 1.0 {
            return self.b.distance(a, b);
        }
        let log = self.log(a, b);
        let f_a = self.conformal_factor(a);
        f_a.sqrt() * log.length()
    }

    fn exp(&self, at: Vec3, v: Vec3) -> Vec3 {
        rk4_geodesic(self, at, v, GEODESIC_DEFAULT_STEPS).0
    }

    fn log(&self, from: Vec3, to: Vec3) -> Vec3 {
        gauss_newton_log(self, from, to, GEODESIC_DEFAULT_STEPS, LOG_MAX_ITERS)
    }

    fn parallel_transport(&self, from: Vec3, to: Vec3, v: Vec3) -> Vec3 {
        // Along the chart straight line, not the geodesic.
        parallel_transport_segment_rk4(self, from, to, v, PARALLEL_TRANSPORT_DEFAULT_STEPS)
    }

    fn parallel_transport_along(&self, path: &[Vec3], v: Vec3) -> Vec3 {
        let mut current = v;
        for w in path.windows(2) {
            current = parallel_transport_segment_rk4(
                self,
                w[0],
                w[1],
                current,
                PARALLEL_TRANSPORT_DEFAULT_STEPS,
            );
        }
        current
    }
}

/// About six digits on moderately curved metrics.
pub const GEODESIC_DEFAULT_STEPS: u32 = 32;

// Geodesic ODE `v̇ = |v|²·∇φ − 2·(∇φ·v)·v` for `g = e^(2φ)·δ` (Wald, *General
// Relativity*, 1984, App. D).
fn rk4_geodesic_step<S: ConformallyFlat>(space: &S, p: Vec3, v: Vec3, h: f32) -> (Vec3, Vec3) {
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

/// About five digits on moderately curved metrics.
pub const PARALLEL_TRANSPORT_DEFAULT_STEPS: u32 = 8;

/// Transport ODE `V̇ = −[(∇φ·γ̇)V + (∇φ·V)γ̇ − (γ̇·V)∇φ]` along the chart
/// segment (Wald, *General Relativity*, 1984, App. D).
pub fn parallel_transport_segment_rk4<S: ConformallyFlat>(
    space: &S,
    p_from: Vec3,
    p_to: Vec3,
    v: Vec3,
    n_steps: u32,
) -> Vec3 {
    let dgamma = p_to - p_from;
    if dgamma.length_squared() < 1.0e-14 {
        return v;
    }
    let h = 1.0 / n_steps as f32;

    let rhs = |gamma_pt: Vec3, v_at_t: Vec3| -> Vec3 {
        let grad_phi = space.conformal_log_half_gradient(gamma_pt);
        let term1 = v_at_t * grad_phi.dot(dgamma);
        let term2 = dgamma * grad_phi.dot(v_at_t);
        let term3 = grad_phi * dgamma.dot(v_at_t);
        -(term1 + term2 - term3)
    };

    let mut v_curr = v;
    for step in 0..n_steps {
        let t = step as f32 * h;
        let p_t = p_from + dgamma * t;
        let p_t_half = p_from + dgamma * (t + h * 0.5);
        let p_t_full = p_from + dgamma * (t + h);

        let k1 = rhs(p_t, v_curr);
        let k2 = rhs(p_t_half, v_curr + k1 * (h * 0.5));
        let k3 = rhs(p_t_half, v_curr + k2 * (h * 0.5));
        let k4 = rhs(p_t_full, v_curr + k3 * h);

        let dv = (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (h / 6.0);
        if dv.is_finite() {
            v_curr += dv;
        } else {
            tracing::warn!(
                "parallel_transport_segment_rk4: non-finite Δv at step {step}; \
                 stopping segment"
            );
            break;
        }
    }
    v_curr
}

/// About five suffice off the cut locus.
pub const LOG_MAX_ITERS: u32 = 12;

/// Euclidean threshold on the residual `|to − exp_from(v)|`.
pub const LOG_RESIDUAL_TOL: f32 = 1.0e-5;

// Central-difference step for `∂exp/∂v`.
const LOG_JACOBIAN_EPS: f32 = 1.0e-3;

/// Gauss-Newton shooting for `v` with `exp_from(v) ≈ to` (Press et al.,
/// *Numerical Recipes*, 3rd ed., 2007, §18.1; Nocedal & Wright, *Numerical
/// Optimization*, 2nd ed., 2006, ch. 10). A singular Jacobian returns the
/// current guess.
pub fn gauss_newton_log<S: ConformallyFlat>(
    space: &S,
    from: Vec3,
    to: Vec3,
    n_steps: u32,
    max_iters: u32,
) -> Vec3 {
    if (from - to).length() < LOG_RESIDUAL_TOL {
        return Vec3::ZERO;
    }

    let mut v = to - from;

    for iter in 0..max_iters {
        let endpoint = rk4_geodesic(space, from, v, n_steps).0;
        let residual = to - endpoint;
        if residual.length() < LOG_RESIDUAL_TOL {
            return v;
        }

        let two_eps = 2.0 * LOG_JACOBIAN_EPS;
        let mut jac = Mat3::ZERO;
        for j in 0..3 {
            let mut e = Vec3::ZERO;
            e[j] = LOG_JACOBIAN_EPS;
            let plus = rk4_geodesic(space, from, v + e, n_steps).0;
            let minus = rk4_geodesic(space, from, v - e, n_steps).0;
            let col = (plus - minus) / two_eps;
            jac.col_mut(j).x = col.x;
            jac.col_mut(j).y = col.y;
            jac.col_mut(j).z = col.z;
        }

        let det = jac.determinant();
        if det.abs() < 1.0e-8 {
            tracing::warn!(
                "gauss_newton_log: singular Jacobian at iter {iter} (det = {det:e}); \
                 returning best guess. `to` may be in the cut locus of `from`."
            );
            return v;
        }

        let delta = jac.inverse() * residual;
        if !delta.is_finite() {
            tracing::warn!(
                "gauss_newton_log: non-finite Newton update at iter {iter}; \
                 returning best guess."
            );
            return v;
        }
        v += delta;
    }

    tracing::warn!(
        "gauss_newton_log: did not converge in {max_iters} iters; \
         residual remained > {LOG_RESIDUAL_TOL}. Returning best guess."
    );
    v
}

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
        // Exact at the extremes so an off-chart `INFINITY` cannot poison the blend.
        if alpha <= 0.0 {
            return f_a;
        }
        if alpha >= 1.0 {
            return f_b;
        }
        (1.0 - alpha) * f_a + alpha * f_b
    }

    // Chain rule on `φ = ln(f)/2`; mirrors the WGSL `loam_blended_grad_phi`.
    fn conformal_log_half_gradient(&self, p: Vec3) -> Vec3 {
        let alpha = self.field.weight(p);
        if alpha <= 0.0 {
            return self.a.conformal_log_half_gradient(p);
        }
        if alpha >= 1.0 {
            return self.b.conformal_log_half_gradient(p);
        }
        let f_a = self.a.conformal_factor(p);
        let f_b = self.b.conformal_factor(p);
        let f = (1.0 - alpha) * f_a + alpha * f_b;
        // NaN, not zero, so the integrator's finiteness guard surfaces it.
        debug_assert!(
            f.is_finite() && f > 0.0,
            "BlendedSpace conformal factor invalid: f = {f}, alpha = {alpha}, f_a = {f_a}, f_b = {f_b}, p = {p:?}"
        );
        if !f.is_finite() || f <= 0.0 {
            return Vec3::NAN;
        }
        let grad_alpha = self.field.gradient(p);
        let grad_phi_a = self.a.conformal_log_half_gradient(p);
        let grad_phi_b = self.b.conformal_log_half_gradient(p);
        let grad_f = grad_alpha * (f_b - f_a)
            + grad_phi_a * (2.0 * (1.0 - alpha) * f_a)
            + grad_phi_b * (2.0 * alpha * f_b);
        grad_f / (2.0 * f)
    }
}

/// `weight` is 0 for pure A and 1 for pure B, and must be C¹: the integrator
/// differentiates it.
pub trait BlendingField: Copy + Send + Sync + 'static {
    /// Implementations must clamp to `[0, 1]`.
    fn weight(&self, p: Vec3) -> f32;

    fn gradient(&self, p: Vec3) -> Vec3 {
        const EPS: f32 = 1.0e-3;
        let dx = (self.weight(p + Vec3::X * EPS) - self.weight(p - Vec3::X * EPS)) / (2.0 * EPS);
        let dy = (self.weight(p + Vec3::Y * EPS) - self.weight(p - Vec3::Y * EPS)) / (2.0 * EPS);
        let dz = (self.weight(p + Vec3::Z * EPS) - self.weight(p - Vec3::Z * EPS)) / (2.0 * EPS);
        Vec3::new(dx, dy, dz)
    }
}

/// Quintic smootherstep `6t⁵ − 15t⁴ + 10t³` along x (Perlin 2002); C², so
/// the scalar curvature stays continuous across the seam.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LinearBlendX {
    pub start: f32,
    /// `end <= start` degenerates to a step at `start`; [`LinearBlendX::new`]
    /// swaps reversed inputs, a struct literal does not.
    pub end: f32,
}

impl LinearBlendX {
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

    fn width(&self) -> f32 {
        self.end - self.start
    }
}

impl BlendingField for LinearBlendX {
    fn weight(&self, p: Vec3) -> f32 {
        let w = self.width();
        if w <= 0.0 {
            return if p.x < self.start { 0.0 } else { 1.0 };
        }
        let t = ((p.x - self.start) / w).clamp(0.0, 1.0);
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    fn gradient(&self, p: Vec3) -> Vec3 {
        let w = self.width();
        if w <= 0.0 {
            return Vec3::ZERO;
        }
        let raw_t = (p.x - self.start) / w;
        if !(0.0..=1.0).contains(&raw_t) {
            return Vec3::ZERO;
        }
        let t = raw_t;
        let one_minus_t = 1.0 - t;
        let dx = 30.0 * t * t * one_minus_t * one_minus_t / w;
        Vec3::new(dx, 0.0, 0.0)
    }
}

// `loam_parallel_transport` mirrors `parallel_transport_segment_rk4` step for step.
fn blended_e3_h3_linearx_wgsl(field: &LinearBlendX) -> String {
    format!(
        r#"
// loam-math :: BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX> (v0 Space WGSL ABI)
const LOAM_MAX_ARC: f32 = 1e9;
const LOAM_BLENDED_R2_MAX: f32 = 0.9999999;
const LOAM_BLENDED_X_START: f32 = {start:?};
const LOAM_BLENDED_X_END:   f32 = {end:?};
const LOAM_BLENDED_X_WIDTH: f32 = {width:?};
const LOAM_BLENDED_RK4_SUB: i32 = 16;
const LOAM_BLENDED_TRANSPORT_SUB: i32 = 8;

fn loam_blended_alpha(p: vec3<f32>) -> f32 {{
    let raw_t = (p.x - LOAM_BLENDED_X_START) / LOAM_BLENDED_X_WIDTH;
    let t = clamp(raw_t, 0.0, 1.0);
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}}

fn loam_blended_alpha_dx(p: vec3<f32>) -> f32 {{
    let raw_t = (p.x - LOAM_BLENDED_X_START) / LOAM_BLENDED_X_WIDTH;
    if raw_t <= 0.0 || raw_t >= 1.0 {{ return 0.0; }}
    let t = raw_t;
    let one_minus_t = 1.0 - t;
    return 30.0 * t * t * one_minus_t * one_minus_t / LOAM_BLENDED_X_WIDTH;
}}

fn loam_blended_f_h3(p: vec3<f32>) -> f32 {{
    let r2 = min(dot(p, p), LOAM_BLENDED_R2_MAX);
    let denom = 1.0 - r2;
    return 4.0 / (denom * denom);
}}

fn loam_blended_f(p: vec3<f32>) -> f32 {{
    let alpha = loam_blended_alpha(p);
    return (1.0 - alpha) + alpha * loam_blended_f_h3(p);
}}

fn loam_blended_grad_phi(p: vec3<f32>) -> vec3<f32> {{
    let alpha = loam_blended_alpha(p);
    let alpha_dx = loam_blended_alpha_dx(p);
    let f_e3 = 1.0;
    let f_h3 = loam_blended_f_h3(p);

    let r2 = min(dot(p, p), LOAM_BLENDED_R2_MAX);
    let denom = 1.0 - r2;
    let grad_f_h3 = (16.0 / (denom * denom * denom)) * p;

    let grad_f = vec3<f32>(
        alpha_dx * (f_h3 - f_e3) + alpha * grad_f_h3.x,
        alpha * grad_f_h3.y,
        alpha * grad_f_h3.z,
    );

    let f = (1.0 - alpha) * f_e3 + alpha * f_h3;
    return grad_f / (2.0 * max(f, 1e-12));
}}

struct LoamBlendedRhs {{ dp: vec3<f32>, dv: vec3<f32> }};

fn loam_blended_rhs(p: vec3<f32>, v: vec3<f32>) -> LoamBlendedRhs {{
    let g = loam_blended_grad_phi(p);
    let v_sq = dot(v, v);
    let g_dot_v = dot(g, v);
    return LoamBlendedRhs(v, g * v_sq - v * (2.0 * g_dot_v));
}}

struct LoamBlendedState {{ p: vec3<f32>, v: vec3<f32> }};

fn loam_blended_rk4_step(p0: vec3<f32>, v0: vec3<f32>, h: f32) -> LoamBlendedState {{
    let k1 = loam_blended_rhs(p0, v0);
    let p1 = p0 + 0.5 * h * k1.dp;
    let v1 = v0 + 0.5 * h * k1.dv;
    let k2 = loam_blended_rhs(p1, v1);
    let p2 = p0 + 0.5 * h * k2.dp;
    let v2 = v0 + 0.5 * h * k2.dv;
    let k3 = loam_blended_rhs(p2, v2);
    let p3 = p0 + h * k3.dp;
    let v3 = v0 + h * k3.dv;
    let k4 = loam_blended_rhs(p3, v3);
    let p_out = p0 + (h / 6.0) * (k1.dp + 2.0 * k2.dp + 2.0 * k3.dp + k4.dp);
    let v_out = v0 + (h / 6.0) * (k1.dv + 2.0 * k2.dv + 2.0 * k3.dv + k4.dv);
    // No position clamp: it would collapse the E³ side onto the unit ball.
    return LoamBlendedState(p_out, v_out);
}}

fn loam_exp(at: vec3<f32>, v: vec3<f32>) -> vec3<f32> {{
    let n2 = dot(v, v);
    if n2 < 1e-14 {{ return at; }}
    var p = at;
    var vv = v;
    let h = 1.0 / f32(LOAM_BLENDED_RK4_SUB);
    for (var i: i32 = 0; i < LOAM_BLENDED_RK4_SUB; i = i + 1) {{
        let s = loam_blended_rk4_step(p, vv, h);
        p = s.p;
        vv = s.v;
    }}
    return p;
}}

fn loam_blended_transport_rhs(p: vec3<f32>, gamma_dot: vec3<f32>, v: vec3<f32>) -> vec3<f32> {{
    let g = loam_blended_grad_phi(p);
    let g_dot_gd = dot(g, gamma_dot);
    let g_dot_v  = dot(g, v);
    let gd_dot_v = dot(gamma_dot, v);
    return -(g_dot_gd * v + g_dot_v * gamma_dot - gd_dot_v * g);
}}

fn loam_parallel_transport(p_from: vec3<f32>, p_to: vec3<f32>, v: vec3<f32>) -> vec3<f32> {{
    // Mirrors the CPU parallel_transport_segment_rk4 step for step.
    let dgamma = p_to - p_from;
    if dot(dgamma, dgamma) < 1e-14 {{ return v; }}
    let h = 1.0 / f32(LOAM_BLENDED_TRANSPORT_SUB);
    var v_curr = v;
    for (var step: i32 = 0; step < LOAM_BLENDED_TRANSPORT_SUB; step = step + 1) {{
        let t = f32(step) * h;
        let p_t      = p_from + dgamma * t;
        let p_t_half = p_from + dgamma * (t + h * 0.5);
        let p_t_full = p_from + dgamma * (t + h);
        let k1 = loam_blended_transport_rhs(p_t,      dgamma, v_curr);
        let k2 = loam_blended_transport_rhs(p_t_half, dgamma, v_curr + k1 * (h * 0.5));
        let k3 = loam_blended_transport_rhs(p_t_half, dgamma, v_curr + k2 * (h * 0.5));
        let k4 = loam_blended_transport_rhs(p_t_full, dgamma, v_curr + k3 * h);
        v_curr = v_curr + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (h / 6.0);
    }}
    return v_curr;
}}

fn loam_distance(a: vec3<f32>, b: vec3<f32>) -> f32 {{
    // Midpoint chord metric, first-order for nearby points.
    let mid = 0.5 * (a + b);
    let f_mid = loam_blended_f(mid);
    return sqrt(max(f_mid, 0.0)) * length(b - a);
}}

fn loam_origin_distance(p: vec3<f32>) -> f32 {{
    return loam_distance(vec3<f32>(0.0, 0.0, 0.0), p);
}}

fn loam_log(p_from: vec3<f32>, p_to: vec3<f32>) -> vec3<f32> {{
    // Chart difference; a true log is computed on the CPU and passed in.
    return p_to - p_from;
}}
"#,
        start = field.start,
        end = field.end,
        width = (field.end - field.start).max(1e-12),
    )
}

impl WgslSpace for BlendedSpace<crate::EuclideanR3, crate::HyperbolicH3, LinearBlendX> {
    fn wgsl_impl(&self) -> Cow<'static, str> {
        Cow::Owned(blended_e3_h3_linearx_wgsl(&self.field))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() <= tol, "expected {a} ≈ {b} (tol {tol})");
    }

    #[test]
    fn linear_blend_x_smoothstep_endpoints() {
        let f = LinearBlendX::new(-1.0, 1.0);
        close(f.weight(Vec3::new(-1.0, 0.0, 0.0)), 0.0, 1e-6);
        close(f.weight(Vec3::new(1.0, 0.0, 0.0)), 1.0, 1e-6);
        close(f.weight(Vec3::ZERO), 0.5, 1e-6);
    }

    #[test]
    fn linear_blend_x_is_constant_outside_zone() {
        let f = LinearBlendX::new(-1.0, 1.0);
        close(f.weight(Vec3::new(-100.0, 0.0, 0.0)), 0.0, 0.0);
        assert_eq!(f.gradient(Vec3::new(-100.0, 0.0, 0.0)), Vec3::ZERO);
        close(f.weight(Vec3::new(100.0, 0.0, 0.0)), 1.0, 0.0);
        assert_eq!(f.gradient(Vec3::new(100.0, 0.0, 0.0)), Vec3::ZERO);
        close(f.gradient(Vec3::new(-1.0, 0.0, 0.0)).x, 0.0, 1e-6);
        close(f.gradient(Vec3::new(1.0, 0.0, 0.0)).x, 0.0, 1e-6);
    }

    #[test]
    fn linear_blend_x_gradient_is_axis_aligned() {
        let f = LinearBlendX::new(-1.0, 1.0);
        let g = f.gradient(Vec3::ZERO);
        assert!(g.x > 0.0, "midpoint gradient should be positive along +x");
        close(g.y, 0.0, 0.0);
        close(g.z, 0.0, 0.0);
        // `t = 0.5`: `30 · 0.25 · 0.25 / 2 = 0.9375`.
        close(g.x, 0.9375, 1e-6);
    }

    #[test]
    fn linear_blend_x_closed_form_matches_finite_diff() {
        let f = LinearBlendX::new(-1.0, 1.0);
        // The trait's default FD path, which the override hides.
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

    #[test]
    fn linear_blend_x_handles_reversed_inputs() {
        let f = LinearBlendX::new(1.0, -1.0);
        close(f.weight(Vec3::new(-1.0, 0.0, 0.0)), 0.0, 1e-6);
        close(f.weight(Vec3::new(1.0, 0.0, 0.0)), 1.0, 1e-6);
    }

    #[test]
    fn linear_blend_x_literal_with_non_positive_width_is_a_step_at_start() {
        for f in [
            LinearBlendX {
                start: 2.0,
                end: -1.0,
            },
            LinearBlendX {
                start: 2.0,
                end: 2.0,
            },
        ] {
            close(f.weight(Vec3::new(1.999, 0.0, 0.0)), 0.0, 1e-6);
            close(f.weight(Vec3::new(2.0, 0.0, 0.0)), 1.0, 1e-6);
            close(f.weight(Vec3::new(9.0, 0.0, 0.0)), 1.0, 1e-6);
            for x in [-5.0, 2.0, 5.0] {
                assert_eq!(f.gradient(Vec3::new(x, 0.0, 0.0)), Vec3::ZERO);
            }
        }
    }

    #[test]
    fn hyperbolic_h3_log_half_gradient_matches_finite_diff() {
        use crate::HyperbolicH3;
        let s = HyperbolicH3;
        const EPS: f32 = 1e-3;
        let fd = |p: Vec3| -> Vec3 {
            let dx = (s.conformal_log_half(p + Vec3::X * EPS)
                - s.conformal_log_half(p - Vec3::X * EPS))
                / (2.0 * EPS);
            let dy = (s.conformal_log_half(p + Vec3::Y * EPS)
                - s.conformal_log_half(p - Vec3::Y * EPS))
                / (2.0 * EPS);
            let dz = (s.conformal_log_half(p + Vec3::Z * EPS)
                - s.conformal_log_half(p - Vec3::Z * EPS))
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

    // Forward-difference step against `Space::distance`.
    const METRIC_PROBE_EPS: f32 = 1.0e-3;

    // The O(ε²) remainder plus the ~10³ ulps lost forming `p + ε·u`.
    const METRIC_PROBE_SLACK: f32 = 1.0e-3;

    // Radial, two tangential, one oblique; anchored on X at the origin.
    fn metric_probe_directions(p: Vec3) -> [Vec3; 4] {
        let radial = if p.length_squared() > 0.0 {
            p.normalize()
        } else {
            Vec3::X
        };
        let tangential_a = radial.any_orthonormal_vector();
        let tangential_b = radial.cross(tangential_a).normalize();
        let oblique = (radial + tangential_a + tangential_b).normalize();
        [radial, tangential_a, tangential_b, oblique]
    }

    // `d(p, p + ε·u) = √f(p)·ε + O(ε²)` for every Euclidean-unit `u`.
    fn assert_conformal_factor_matches_metric<S>(space: &S, samples: &[Vec3])
    where
        S: ConformallyFlat<Point = Vec3, Vector = Vec3>,
    {
        for &p in samples {
            let root_f = space.conformal_factor(p).sqrt();
            // The leading gap is `√f·ε²·(∇φ·u)/2`.
            let truncation = 0.5 * space.conformal_log_half_gradient(p).length() * METRIC_PROBE_EPS;
            let tol = root_f * (truncation + METRIC_PROBE_SLACK);
            for u in metric_probe_directions(p) {
                let measured = space.distance(p, p + u * METRIC_PROBE_EPS) / METRIC_PROBE_EPS;
                assert!(
                    (measured - root_f).abs() <= tol,
                    "conformal factor disagrees with the metric at p = {p:?} along \
                     u = {u:?}: measured {measured}, √f = {root_f} (tol {tol})"
                );
            }
        }
    }

    #[test]
    fn conformal_factor_reproduces_the_space_metric_in_every_direction() {
        use crate::{EuclideanR3, HyperbolicH3};

        assert_conformal_factor_matches_metric(
            &EuclideanR3,
            &[
                Vec3::ZERO,
                Vec3::new(0.25, -0.4, 0.1),
                Vec3::new(1.0, 0.5, -0.75),
            ],
        );

        // Off the ideal boundary, where the truncation grows as `2|p|/(1 − |p|²)²`.
        assert_conformal_factor_matches_metric(
            &HyperbolicH3,
            &[
                Vec3::ZERO,
                Vec3::new(0.2, 0.0, 0.0),
                Vec3::new(0.3, -0.2, 0.1),
                Vec3::new(0.4, 0.3, -0.2),
            ],
        );

        // Zone extremes only; inside, `distance` is defined as `√f·|log|`.
        assert_conformal_factor_matches_metric(
            &BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-0.5, 0.5)),
            &[Vec3::new(-0.7, 0.1, 0.0), Vec3::new(0.55, 0.15, -0.1)],
        );
    }

    #[test]
    fn blended_space_log_half_gradient_matches_finite_diff() {
        use crate::{EuclideanR3, HyperbolicH3};
        let bs = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-0.5, 0.5));
        let fd = |p: Vec3| -> Vec3 {
            const EPS: f32 = 1e-3;
            let dx = (bs.conformal_log_half(p + Vec3::X * EPS)
                - bs.conformal_log_half(p - Vec3::X * EPS))
                / (2.0 * EPS);
            let dy = (bs.conformal_log_half(p + Vec3::Y * EPS)
                - bs.conformal_log_half(p - Vec3::Y * EPS))
                / (2.0 * EPS);
            let dz = (bs.conformal_log_half(p + Vec3::Z * EPS)
                - bs.conformal_log_half(p - Vec3::Z * EPS))
                / (2.0 * EPS);
            Vec3::new(dx, dy, dz)
        };
        // Pure A, mid-zone, pure B.
        for p in [
            Vec3::new(-0.7, 0.05, 0.0),
            Vec3::new(0.0, 0.1, -0.1),
            Vec3::new(0.6, 0.05, 0.0),
        ] {
            let analytic = bs.conformal_log_half_gradient(p);
            let numeric = fd(p);
            close(analytic.x, numeric.x, 5e-3);
            close(analytic.y, numeric.y, 5e-3);
            close(analytic.z, numeric.z, 5e-3);
        }
    }

    #[test]
    fn rk4_in_pure_e3_is_straight_line() {
        use crate::EuclideanR3;
        let bs = BlendedSpace::new(EuclideanR3, EuclideanR3, LinearBlendX::new(100.0, 200.0));
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

    #[test]
    fn rk4_in_pure_h3_matches_closed_form_at_origin() {
        use crate::{HyperbolicH3, Space};

        for &mag in &[0.1_f32, 0.3, 0.5] {
            let v = Vec3::new(mag, 0.0, 0.0);
            let (final_p, _) = rk4_geodesic(&HyperbolicH3, Vec3::ZERO, v, GEODESIC_DEFAULT_STEPS);
            // `|exp_0(v)| = tanh(|v|)`.
            let expected_radius = mag.tanh();
            close(final_p.x, expected_radius, 5e-3);
            close(final_p.y, 0.0, 1e-4);
            close(final_p.z, 0.0, 1e-4);
        }

        let v = Vec3::new(0.4, 0.1, 0.0);
        let (numerical, _) = rk4_geodesic(&HyperbolicH3, Vec3::ZERO, v, GEODESIC_DEFAULT_STEPS);
        let closed_form = HyperbolicH3.exp(Vec3::ZERO, v);
        close((numerical - closed_form).length(), 0.0, 5e-3);

        let bs = BlendedSpace::new(HyperbolicH3, HyperbolicH3, LinearBlendX::new(-100.0, -50.0));
        let v = Vec3::new(0.5, 0.0, 0.0);
        let final_p_blended = bs.exp(Vec3::ZERO, v);
        close(final_p_blended.x, 0.5_f32.tanh(), 5e-3);
    }

    #[test]
    fn exp_log_round_trip_in_pure_h3() {
        use crate::HyperbolicH3;
        for (from, to) in [
            (Vec3::ZERO, Vec3::new(0.3, 0.0, 0.0)),
            (Vec3::ZERO, Vec3::new(0.2, 0.1, -0.1)),
            (Vec3::new(0.1, 0.1, 0.0), Vec3::new(-0.1, 0.2, 0.1)),
        ] {
            let v = gauss_newton_log(
                &HyperbolicH3,
                from,
                to,
                GEODESIC_DEFAULT_STEPS,
                LOG_MAX_ITERS,
            );
            let endpoint = rk4_geodesic(&HyperbolicH3, from, v, GEODESIC_DEFAULT_STEPS).0;
            close((endpoint - to).length(), 0.0, 5e-4);
        }
    }

    #[test]
    fn log_in_pure_h3_matches_closed_form() {
        use crate::{HyperbolicH3, Space};
        for (from, to) in [
            (Vec3::ZERO, Vec3::new(0.3, 0.0, 0.0)),
            (Vec3::ZERO, Vec3::new(0.2, 0.1, 0.0)),
            (Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.3, 0.2, 0.0)),
        ] {
            let numerical = gauss_newton_log(
                &HyperbolicH3,
                from,
                to,
                GEODESIC_DEFAULT_STEPS,
                LOG_MAX_ITERS,
            );
            let closed_form = HyperbolicH3.log(from, to);
            close((numerical - closed_form).length(), 0.0, 5e-3);
        }
    }

    #[test]
    fn log_of_self_is_zero() {
        use crate::HyperbolicH3;
        let p = Vec3::new(0.2, 0.1, 0.0);
        let v = gauss_newton_log(&HyperbolicH3, p, p, GEODESIC_DEFAULT_STEPS, LOG_MAX_ITERS);
        close(v.length(), 0.0, 1e-5);
    }

    #[test]
    fn h3_transport_agrees_with_the_closed_form_by_a_vanishing_coefficient() {
        use crate::{HyperbolicH3, Space};
        let bases = [
            Vec3::new(0.05, 0.0, 0.0),
            Vec3::new(0.0, 0.12, -0.04),
            Vec3::new(-0.2, 0.1, 0.15),
        ];
        let directions = [
            Vec3::new(1.0, 0.6, 0.0).normalize(),
            Vec3::new(-0.3, 1.0, 0.5).normalize(),
            Vec3::new(0.2, -0.4, 1.0).normalize(),
        ];
        let tangents = [
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.0, 0.07, 0.05),
            Vec3::new(-0.06, 0.03, 0.08),
        ];

        let mut coefficients = Vec::new();
        for h in [0.04_f32, 0.02, 0.01] {
            let mut worst = 0.0_f32;
            for from in bases {
                for dir in directions {
                    let to = from + dir * h;
                    for v in tangents {
                        let numerical = parallel_transport_segment_rk4(
                            &HyperbolicH3,
                            from,
                            to,
                            v,
                            PARALLEL_TRANSPORT_DEFAULT_STEPS,
                        );
                        let closed_form = HyperbolicH3.parallel_transport(from, to, v);
                        worst = worst.max((numerical - closed_form).length() / h);
                    }
                }
            }
            coefficients.push(worst);
        }

        assert!(
            coefficients[2] <= 3.0e-4,
            "the transport disagrees with the closed form by a coefficient of              {} at h = 0.01, which does not vanish with the step",
            coefficients[2]
        );
        for pair in coefficients.windows(2) {
            assert!(
                pair[1] < pair[0] * 0.6,
                "halving h moved the disagreement coefficient from {} to {},                  not the ~4x fall an O(h³) residual has: a term linear in h,                  i.e. a different connection, is the shape that does this",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn transport_is_invariant_to_how_its_own_path_is_subdivided() {
        use crate::{EuclideanR3, HyperbolicH3, Space};
        let bs = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-0.15, 0.15));
        let v = Vec3::new(0.05, -0.03, 0.04);

        // One segment across the seam, one inside it, one in the pure-H³ half.
        let mut worst = 0.0_f32;
        for (a, b) in [
            (Vec3::new(-0.3, 0.05, 0.0), Vec3::new(0.3, -0.05, 0.02)),
            (Vec3::new(-0.1, 0.0, 0.0), Vec3::new(0.1, 0.05, 0.0)),
            (Vec3::new(0.2, 0.0, 0.0), Vec3::new(0.3, 0.1, 0.05)),
        ] {
            let direct = bs.parallel_transport(a, b, v);
            for k in [2_u32, 4, 8, 16] {
                let path: Vec<Vec3> = (0..=k)
                    .map(|i| a + (b - a) * (i as f32 / k as f32))
                    .collect();
                let refined = bs.parallel_transport_along(&path, v);
                worst = worst.max((refined - direct).length());
            }
        }
        // Measured worst is 3.7e-5, flat in `k`.
        assert!(
            worst <= 1.0e-4,
            "subdividing the transport path moved the result by {worst}"
        );
    }

    #[test]
    fn parallel_transport_in_h3_has_nonzero_holonomy() {
        use crate::{HyperbolicH3, Space};
        let bs = BlendedSpace::new(HyperbolicH3, HyperbolicH3, LinearBlendX::new(-100.0, -50.0));
        let path = [
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.3, 0.0, 0.0),
            Vec3::new(0.2, 0.2, 0.0),
            Vec3::new(0.1, 0.0, 0.0),
        ];
        let v = Vec3::new(0.1, 0.0, 0.0);
        let transported = bs.parallel_transport_along(&path, v);
        assert!(transported.is_finite());
        let drift = (transported - v).length();
        assert!(
            drift > 1e-3,
            "expected non-zero holonomy in H³, got drift {drift}"
        );
        assert!(drift < 0.5, "holonomy unreasonably large: {drift}");
    }

    #[test]
    fn finite_diff_curvature_matches_closed_form_in_h3() {
        use crate::HyperbolicH3;
        // Does not override `scalar_curvature`, so the FD default runs.
        struct H3FdOnly;
        impl crate::space::Space for H3FdOnly {
            type Point = Vec3;
            type Vector = Vec3;
            fn distance(&self, _: Vec3, _: Vec3) -> f32 {
                0.0
            }
            fn exp(&self, _: Vec3, _: Vec3) -> Vec3 {
                Vec3::ZERO
            }
            fn log(&self, _: Vec3, _: Vec3) -> Vec3 {
                Vec3::ZERO
            }
            fn parallel_transport(&self, _: Vec3, _: Vec3, v: Vec3) -> Vec3 {
                v
            }
        }
        impl ConformallyFlat for H3FdOnly {
            fn conformal_factor(&self, p: Vec3) -> f32 {
                HyperbolicH3.conformal_factor(p)
            }
            fn conformal_log_half(&self, p: Vec3) -> f32 {
                HyperbolicH3.conformal_log_half(p)
            }
            fn conformal_log_half_gradient(&self, p: Vec3) -> Vec3 {
                HyperbolicH3.conformal_log_half_gradient(p)
            }
        }
        let fd = H3FdOnly;
        // FD at `EPS = 5e-3` gives about three digits.
        for p in [Vec3::ZERO, Vec3::new(0.2, 0.1, 0.0)] {
            close(fd.scalar_curvature(p), -6.0, 0.5);
        }
    }

    #[test]
    fn blended_space_curvature_varies_continuously_across_zone() {
        use crate::{EuclideanR3, HyperbolicH3};
        let bs = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-0.5, 0.5));

        let r_e = bs.scalar_curvature(Vec3::new(-0.7, 0.0, 0.0));
        close(r_e, 0.0, 1e-1);

        let r_h = bs.scalar_curvature(Vec3::new(0.7, 0.0, 0.0));
        close(r_h, -6.0, 1.0);

        let xs: Vec<f32> = (-30..=30).map(|i| (i as f32) * 0.025).collect();
        let curvatures: Vec<f32> = xs
            .iter()
            .map(|&x| bs.scalar_curvature(Vec3::new(x, 0.0, 0.0)))
            .collect();

        // `|R| > 6` inside the seam is real: `|∇φ|²` and `∇²φ` both spike.
        for &r in &curvatures {
            assert!(
                r.is_finite() && (-50.0..=5.0).contains(&r),
                "curvature out of expected range: {r}"
            );
        }

        // FD aliasing reaches ~14 here; a C² break (cubic smoothstep) jumps past 25.
        let max_jump = curvatures
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_jump < 25.0,
            "curvature has a discontinuity: max adjacent jump = {max_jump}"
        );
    }

    #[test]
    fn blended_space_at_alpha_zero_is_pure_a() {
        use crate::{EuclideanR3, HyperbolicH3, Space};
        let bs = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(50.0, 100.0));
        let p = Vec3::new(1.0, 2.0, 3.0);
        let q = Vec3::new(4.0, 5.0, 6.0);
        let v = Vec3::new(0.5, -0.3, 0.7);

        close((bs.exp(p, v) - EuclideanR3.exp(p, v)).length(), 0.0, 1e-5);
        close((bs.log(p, q) - EuclideanR3.log(p, q)).length(), 0.0, 1e-3);
        close(bs.distance(p, q), EuclideanR3.distance(p, q), 1e-3);
        close(
            (bs.parallel_transport(p, q, v) - EuclideanR3.parallel_transport(p, q, v)).length(),
            0.0,
            1e-5,
        );
        close(
            bs.conformal_factor(p),
            EuclideanR3.conformal_factor(p),
            1e-6,
        );
        close(bs.scalar_curvature(p), 0.0, 1e-3);
    }

    #[test]
    fn blended_space_at_alpha_one_is_pure_b() {
        use crate::{EuclideanR3, HyperbolicH3, Space};
        let bs = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-100.0, -50.0));
        let p = Vec3::new(0.1, 0.0, 0.0);
        let q = Vec3::new(0.2, 0.1, 0.0);
        let v = Vec3::new(0.05, 0.0, 0.0);

        close((bs.exp(p, v) - HyperbolicH3.exp(p, v)).length(), 0.0, 5e-3);
        close((bs.log(p, q) - HyperbolicH3.log(p, q)).length(), 0.0, 5e-3);
        close(bs.distance(p, q), HyperbolicH3.distance(p, q), 5e-3);
        close(
            bs.conformal_factor(p),
            HyperbolicH3.conformal_factor(p),
            1e-3,
        );
        close(bs.scalar_curvature(p), -6.0, 0.05);
    }

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
