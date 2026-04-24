//! Bivectors and rotors — the N-dim rotation primitive.
//!
//! A **bivector** represents an oriented plane of rotation plus a magnitude.
//! Its exponential is a **rotor**, which acts on vectors via the sandwich
//! product and composes via Clifford multiplication.
//!
//! This module exposes two traits ([`Bivector`], [`Rotor`]) so physics and
//! other consumers can be written generically over dimension. Concrete
//! implementations ship per-N — [`Bivector2`] / [`Rotor2`] here,
//! [`Bivector3`] / [`Rotor3`] to follow when Simplex 3D needs them,
//! [`Bivector4`] / [`Rotor4`] for Simplex 4D.
//!
//! See `docs/devlog/PHASE_2_3_PLAN.md` §3.4 for why we use handwritten
//! per-N impls rather than `generic_const_exprs`.
//!
//! ## Convention
//!
//! `R = exp(B/2)`, so a bivector of magnitude θ in plane `e_i ∧ e_j`
//! produces a rotor that rotates by θ in that plane, from `e_i` toward
//! `e_j`. Applying a rotor to a vector uses the sandwich product
//! `v' = R̃ · v · R`, implemented here in closed form per dimension.

use std::ops::{Add, Mul};

use glam::{Vec2, Vec3};

/// A bivector in some geometric algebra G(N, 0). Represents an oriented
/// plane of rotation × magnitude; exponentiates to a rotor.
pub trait Bivector: Copy + Add<Output = Self> + Mul<f32, Output = Self> {
    type Rotor: Rotor<Bivector = Self>;

    /// Zero bivector. `zero().exp()` is the identity rotor.
    fn zero() -> Self;

    /// Exponential map: `exp(B) = sum_k B^k / k!` — closed-form per
    /// dimension. Rotates by the bivector's magnitude in its plane.
    fn exp(self) -> Self::Rotor;
}

/// A rotor in some geometric algebra G(N, 0). Elements of Spin(N), unit-
/// norm by construction.
pub trait Rotor: Copy + Mul<Output = Self> {
    type Bivector: Bivector<Rotor = Self>;
    type Vector: Copy;

    /// Identity rotor. Applying to any vector returns it unchanged.
    fn identity() -> Self;

    /// Reverse / conjugate rotor. `R · R.inverse() == identity()` within
    /// floating-point error.
    fn inverse(self) -> Self;

    /// Rotate a vector via the sandwich product.
    fn apply(&self, v: Self::Vector) -> Self::Vector;

    /// Logarithm: the bivector whose `exp` is this rotor. Inverse of
    /// [`Bivector::exp`] modulo branch selection (we pick the principal
    /// branch, angle in `[−π, π]`).
    fn log(self) -> Self::Bivector;
}

// ---------------------------------------------------------------------------
// 2D: Bivector2 is a single scalar (the e1∧e2 coefficient); Rotor2 is a
// unit complex number (cos(θ/2), sin(θ/2)).
// ---------------------------------------------------------------------------

/// 2D bivector: a scalar coefficient on the single basis plane `e1∧e2`.
/// Represents a rotation angle in radians (from `x` toward `y`).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Bivector2(pub f32);

impl Add for Bivector2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Mul<f32> for Bivector2 {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self(self.0 * k)
    }
}

impl Bivector for Bivector2 {
    type Rotor = Rotor2;

    fn zero() -> Self {
        Self(0.0)
    }

    fn exp(self) -> Rotor2 {
        let half = self.0 * 0.5;
        Rotor2 {
            a: half.cos(),
            b: half.sin(),
        }
    }
}

/// 2D rotor: a unit complex number `a + b·e1e2` with `a² + b² = 1`.
/// `a = cos(θ/2)`, `b = sin(θ/2)` for a rotation of θ.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rotor2 {
    pub a: f32,
    pub b: f32,
}

impl Rotor2 {
    pub const IDENTITY: Self = Self { a: 1.0, b: 0.0 };
}

impl Default for Rotor2 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mul for Rotor2 {
    type Output = Self;
    /// Complex multiplication — equivalent to Clifford product in 2D.
    fn mul(self, rhs: Self) -> Self {
        Self {
            a: self.a * rhs.a - self.b * rhs.b,
            b: self.a * rhs.b + self.b * rhs.a,
        }
    }
}

impl Rotor for Rotor2 {
    type Bivector = Bivector2;
    type Vector = Vec2;

    fn identity() -> Self {
        Self::IDENTITY
    }

    fn inverse(self) -> Self {
        // Reverse = conjugate for unit rotors.
        Self {
            a: self.a,
            b: -self.b,
        }
    }

    fn apply(&self, v: Vec2) -> Vec2 {
        // Sandwich collapses to the standard 2D rotation matrix where
        // cos(θ) = a² − b² and sin(θ) = 2ab.
        let c = self.a * self.a - self.b * self.b;
        let s = 2.0 * self.a * self.b;
        Vec2::new(c * v.x - s * v.y, s * v.x + c * v.y)
    }

    fn log(self) -> Bivector2 {
        // Full angle θ = 2·atan2(b, a); principal branch by construction
        // of atan2.
        Bivector2(2.0 * self.b.atan2(self.a))
    }
}

// ---------------------------------------------------------------------------
// 3D: Bivector3 has three basis planes (e1∧e2, e2∧e3, e3∧e1). Rotor3 is
// the even subalgebra element `s + xy·e12 + yz·e23 + zx·e31`, isomorphic
// to a quaternion but with a sign convention tied to the rotor sandwich
// `v' = R̃·v·R` rather than the quaternion `q·v·q*`.
// ---------------------------------------------------------------------------

/// 3D bivector. The three components are the coefficients on the basis
/// planes `e1∧e2`, `e2∧e3`, `e3∧e1`. Magnitude encodes rotation angle.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Bivector3 {
    /// Coefficient on `e1∧e2` — rotation in the xy plane.
    pub xy: f32,
    /// Coefficient on `e2∧e3` — rotation in the yz plane.
    pub yz: f32,
    /// Coefficient on `e3∧e1` — rotation in the zx plane.
    pub zx: f32,
}

impl Bivector3 {
    pub const ZERO: Self = Self {
        xy: 0.0,
        yz: 0.0,
        zx: 0.0,
    };

    pub fn new(xy: f32, yz: f32, zx: f32) -> Self {
        Self { xy, yz, zx }
    }

    /// Magnitude of the bivector — the rotation angle when used as a
    /// rotor generator.
    pub fn magnitude(self) -> f32 {
        (self.xy * self.xy + self.yz * self.yz + self.zx * self.zx).sqrt()
    }
}

impl Add for Bivector3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            xy: self.xy + rhs.xy,
            yz: self.yz + rhs.yz,
            zx: self.zx + rhs.zx,
        }
    }
}

impl Mul<f32> for Bivector3 {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self {
            xy: self.xy * k,
            yz: self.yz * k,
            zx: self.zx * k,
        }
    }
}

impl Bivector for Bivector3 {
    type Rotor = Rotor3;

    fn zero() -> Self {
        Self::ZERO
    }

    /// Exponential map. For a bivector `B` of magnitude `θ`,
    /// `exp(B/2) = cos(θ/2) + sin(θ/2)·B̂`. In 3D every bivector is
    /// simple (single plane), so the closed form is direct — no
    /// decomposition needed (unlike 4D).
    fn exp(self) -> Rotor3 {
        let mag_sq = self.xy * self.xy + self.yz * self.yz + self.zx * self.zx;
        if mag_sq < 1e-16 {
            // Small-angle: Taylor expand `sin(θ/2)/θ ≈ 1/2`.
            return Rotor3 {
                s: 1.0,
                xy: self.xy * 0.5,
                yz: self.yz * 0.5,
                zx: self.zx * 0.5,
            };
        }
        let mag = mag_sq.sqrt();
        let half = mag * 0.5;
        let c = half.cos();
        let k = half.sin() / mag;
        Rotor3 {
            s: c,
            xy: self.xy * k,
            yz: self.yz * k,
            zx: self.zx * k,
        }
    }
}

/// 3D rotor: scalar part plus a bivector part, unit-norm by
/// construction. `s² + xy² + yz² + zx² = 1` for any rotor produced by
/// `Bivector3::exp` or by composing rotors.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rotor3 {
    pub s: f32,
    pub xy: f32,
    pub yz: f32,
    pub zx: f32,
}

impl Rotor3 {
    pub const IDENTITY: Self = Self {
        s: 1.0,
        xy: 0.0,
        yz: 0.0,
        zx: 0.0,
    };
}

impl Default for Rotor3 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mul for Rotor3 {
    type Output = Self;
    /// Geometric product of two rotors. Derived from the Clifford
    /// multiplication table for bivector basis elements of G(3,0).
    fn mul(self, rhs: Self) -> Self {
        let (s1, a1, b1, c1) = (self.s, self.xy, self.yz, self.zx);
        let (s2, a2, b2, c2) = (rhs.s, rhs.xy, rhs.yz, rhs.zx);
        Self {
            s: s1 * s2 - a1 * a2 - b1 * b2 - c1 * c2,
            xy: s1 * a2 + s2 * a1 - b1 * c2 + c1 * b2,
            yz: s1 * b2 + s2 * b1 + a1 * c2 - c1 * a2,
            zx: s1 * c2 + s2 * c1 - a1 * b2 + b1 * a2,
        }
    }
}

impl Rotor for Rotor3 {
    type Bivector = Bivector3;
    type Vector = Vec3;

    fn identity() -> Self {
        Self::IDENTITY
    }

    fn inverse(self) -> Self {
        // Reverse: flip sign of grade-2 part. For unit rotors this is
        // the geometric inverse.
        Self {
            s: self.s,
            xy: -self.xy,
            yz: -self.yz,
            zx: -self.zx,
        }
    }

    /// Apply the rotor to a vector via the sandwich `R̃ · v · R`.
    /// Computed in two stages: first `R̃ · v` (produces vector + trivector),
    /// then multiply by `R` (the trivector part cancels for unit rotors).
    fn apply(&self, v: Vec3) -> Vec3 {
        let (s, a, b, c) = (self.s, self.xy, self.yz, self.zx);
        let (vx, vy, vz) = (v.x, v.y, v.z);

        // R̃·v: vector part (p1, p2, p3) and trivector-coefficient pt.
        let p1 = s * vx - a * vy + c * vz;
        let p2 = s * vy + a * vx - b * vz;
        let p3 = s * vz + b * vy - c * vx;
        let pt = -(a * vz + b * vx + c * vy);

        // (P + pt·I) · R: vector part (trivector cancels for unit R).
        Vec3::new(
            p1 * s - p2 * a + p3 * c - pt * b,
            p2 * s + p1 * a - p3 * b - pt * c,
            p3 * s + p2 * b - p1 * c - pt * a,
        )
    }

    fn log(self) -> Bivector3 {
        // Inverse of `Bivector3::exp`. The bivector part has magnitude
        // sin(θ/2); the scalar is cos(θ/2). θ = 2·atan2(sin, cos).
        let mag_sq = self.xy * self.xy + self.yz * self.yz + self.zx * self.zx;
        if mag_sq < 1e-16 {
            // Near-identity: use linear term (log ≈ 2·bivector_part).
            return Bivector3 {
                xy: self.xy * 2.0,
                yz: self.yz * 2.0,
                zx: self.zx * 2.0,
            };
        }
        let mag = mag_sq.sqrt();
        let theta = 2.0 * mag.atan2(self.s);
        let k = theta / mag;
        Bivector3 {
            xy: self.xy * k,
            yz: self.yz * k,
            zx: self.zx * k,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI, TAU};

    fn assert_close(a: f32, b: f32) {
        assert!(
            (a - b).abs() <= 1e-5,
            "expected {a} close to {b} (diff {})",
            (a - b).abs()
        );
    }

    fn assert_vec2_close(a: Vec2, b: Vec2) {
        assert!(
            (a - b).length() <= 1e-5,
            "expected {a:?} close to {b:?} (diff {})",
            (a - b).length()
        );
    }

    #[test]
    fn zero_bivector_exp_is_identity() {
        let r = Bivector2::zero().exp();
        assert_eq!(r, Rotor2::identity());
    }

    #[test]
    fn quarter_turn_rotates_x_to_y() {
        let r = Bivector2(FRAC_PI_2).exp();
        assert_vec2_close(r.apply(Vec2::X), Vec2::Y);
        assert_vec2_close(r.apply(Vec2::Y), -Vec2::X);
    }

    #[test]
    fn half_turn_negates() {
        let r = Bivector2(PI).exp();
        assert_vec2_close(r.apply(Vec2::new(1.0, 2.0)), Vec2::new(-1.0, -2.0));
    }

    #[test]
    fn full_turn_is_identity_up_to_sign() {
        // exp(B·τ/2) is the negative identity rotor (−1 + 0·e12) — still
        // acts as identity on vectors (sandwich squares the sign out).
        let r = Bivector2(TAU).exp();
        assert_vec2_close(r.apply(Vec2::X), Vec2::X);
    }

    #[test]
    fn composition_adds_angles() {
        let a = Bivector2(0.3).exp();
        let b = Bivector2(0.5).exp();
        let composed = a * b;
        let direct = Bivector2(0.8).exp();
        assert_close(composed.a, direct.a);
        assert_close(composed.b, direct.b);
    }

    #[test]
    fn inverse_cancels() {
        let r = Bivector2(1.234).exp();
        let id = r * r.inverse();
        assert_close(id.a, 1.0);
        assert_close(id.b, 0.0);
    }

    #[test]
    fn apply_preserves_length() {
        let r = Bivector2(0.7).exp();
        let v = Vec2::new(3.0, 4.0);
        assert_close(r.apply(v).length(), v.length());
    }

    #[test]
    fn log_is_inverse_of_exp() {
        for &theta in &[-1.0_f32, -0.1, 0.0, 0.1, 1.0, 3.0] {
            let back = Bivector2(theta).exp().log();
            assert_close(back.0, theta);
        }
    }

    #[test]
    fn scalar_mul_scales_angle() {
        let b = Bivector2(0.5) * 2.0;
        assert_close(b.0, 1.0);
    }

    #[test]
    fn bivector_add_sums_angles() {
        let b = Bivector2(0.3) + Bivector2(0.5);
        assert_close(b.0, 0.8);
    }

    // ---- 3D tests ----

    fn assert_vec3_close(a: Vec3, b: Vec3) {
        assert!(
            (a - b).length() <= 1e-5,
            "expected {a:?} close to {b:?}"
        );
    }

    #[test]
    fn bivector3_zero_exp_is_identity() {
        let r = Bivector3::ZERO.exp();
        assert_eq!(r, Rotor3::IDENTITY);
    }

    #[test]
    fn rotor3_identity_leaves_vector_unchanged() {
        let v = Vec3::new(1.2, -3.4, 0.5);
        assert_vec3_close(Rotor3::IDENTITY.apply(v), v);
    }

    #[test]
    fn bivector3_xy_quarter_turn_sends_x_to_y() {
        // Rotation in xy-plane by π/2: e1 → e2.
        let r = Bivector3::new(FRAC_PI_2, 0.0, 0.0).exp();
        assert_vec3_close(r.apply(Vec3::X), Vec3::Y);
        assert_vec3_close(r.apply(Vec3::Y), -Vec3::X);
        // z-axis is unaffected by xy-plane rotation.
        assert_vec3_close(r.apply(Vec3::Z), Vec3::Z);
    }

    #[test]
    fn bivector3_yz_rotation() {
        // Rotation in yz-plane (e2 → e3) by π/2.
        let r = Bivector3::new(0.0, FRAC_PI_2, 0.0).exp();
        assert_vec3_close(r.apply(Vec3::Y), Vec3::Z);
        assert_vec3_close(r.apply(Vec3::Z), -Vec3::Y);
        assert_vec3_close(r.apply(Vec3::X), Vec3::X);
    }

    #[test]
    fn bivector3_zx_rotation() {
        // Rotation in zx-plane (e3 → e1) by π/2.
        let r = Bivector3::new(0.0, 0.0, FRAC_PI_2).exp();
        assert_vec3_close(r.apply(Vec3::Z), Vec3::X);
        assert_vec3_close(r.apply(Vec3::X), -Vec3::Z);
        assert_vec3_close(r.apply(Vec3::Y), Vec3::Y);
    }

    #[test]
    fn rotor3_full_turn_is_identity_on_vectors() {
        let r = Bivector3::new(TAU, 0.0, 0.0).exp();
        assert_vec3_close(r.apply(Vec3::new(1.0, 2.0, 3.0)), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn rotor3_inverse_cancels() {
        let r = Bivector3::new(0.7, -0.3, 0.5).exp();
        let id = r * r.inverse();
        assert_close(id.s, 1.0);
        assert_close(id.xy, 0.0);
        assert_close(id.yz, 0.0);
        assert_close(id.zx, 0.0);
    }

    #[test]
    fn rotor3_composition_matches_sequential_apply() {
        // With the `v' = R̃·v·R` sandwich, multiplication order equals
        // application order: `(ra · rb).apply(v) = rb.apply(ra.apply(v))`
        // — i.e. `ra` is applied first, then `rb`.
        let ra = Bivector3::new(0.4, 0.0, 0.0).exp();
        let rb = Bivector3::new(0.0, 0.5, 0.0).exp();
        let composed = ra * rb;
        let v = Vec3::new(0.7, -0.3, 1.1);
        assert_vec3_close(composed.apply(v), rb.apply(ra.apply(v)));
    }

    #[test]
    fn rotor3_preserves_length() {
        let r = Bivector3::new(0.3, 0.4, 0.5).exp();
        for &v in &[Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(1.0, 2.0, -3.0)] {
            assert!(
                (r.apply(v).length() - v.length()).abs() < 1e-5,
                "length not preserved: input {v}, output length {}",
                r.apply(v).length()
            );
        }
    }

    #[test]
    fn rotor3_log_is_inverse_of_exp() {
        for bv in [
            Bivector3::new(0.0, 0.0, 0.0),
            Bivector3::new(0.1, 0.0, 0.0),
            Bivector3::new(0.0, 0.6, 0.0),
            Bivector3::new(-0.3, 0.4, 0.5),
            Bivector3::new(1.0, -0.5, 0.2),
        ] {
            let back = bv.exp().log();
            assert!(
                (back.xy - bv.xy).abs() < 1e-5
                    && (back.yz - bv.yz).abs() < 1e-5
                    && (back.zx - bv.zx).abs() < 1e-5,
                "log∘exp mismatch: {bv:?} → {back:?}"
            );
        }
    }

    #[test]
    fn rotor3_matches_glam_quat_for_axis_rotation() {
        // Same rotation (axis + angle) computed two ways — Rye's rotor
        // and glam's Quat — must agree on the acted-upon vector. This
        // cross-checks the sign convention.
        use glam::Quat;

        let theta = 0.7;
        let v = Vec3::new(1.0, 2.0, 3.0).normalize();

        // xy-plane rotation ↔ rotation about +z axis.
        let rotor = Bivector3::new(theta, 0.0, 0.0).exp();
        let quat = Quat::from_axis_angle(Vec3::Z, theta);
        assert_vec3_close(rotor.apply(v), quat * v);

        // yz-plane rotation ↔ rotation about +x axis.
        let rotor = Bivector3::new(0.0, theta, 0.0).exp();
        let quat = Quat::from_axis_angle(Vec3::X, theta);
        assert_vec3_close(rotor.apply(v), quat * v);

        // zx-plane rotation ↔ rotation about +y axis.
        let rotor = Bivector3::new(0.0, 0.0, theta).exp();
        let quat = Quat::from_axis_angle(Vec3::Y, theta);
        assert_vec3_close(rotor.apply(v), quat * v);
    }
}
