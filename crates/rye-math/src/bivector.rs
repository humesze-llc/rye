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

use glam::Vec2;

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
}
