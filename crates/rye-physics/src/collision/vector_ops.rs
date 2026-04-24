//! `VectorOps` — the abstraction GJK and EPA share across dimensions.
//!
//! GJK walks the Minkowski difference of two shapes using nothing but
//! vector algebra and dot products — no cross products, no bivectors,
//! no dimension-specific machinery. This trait captures exactly that
//! surface area, so the same GJK loop can run on `Vec3` (3D physics,
//! Simplex 3D phase) and `Vec4` (4D physics, Simplex 4D phase) when the
//! time comes.
//!
//! EPA's face-normal reconstruction is dimension-specific (cross
//! product in 3D, generalized cross in 4D+) and lives outside this
//! trait — each dimension has its own EPA helper that uses
//! `VectorOps` for the bulk of the math and a per-dimension function
//! for the normal reconstruction step.

use std::ops::{Add, Mul, Neg, Sub};

use glam::{Vec3, Vec4};

/// Vector algebra that GJK (and most of EPA) needs.
pub trait VectorOps:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + Mul<f32, Output = Self>
    + PartialEq
{
    fn zero() -> Self;
    fn dot(self, rhs: Self) -> f32;

    fn length_squared(self) -> f32 {
        self.dot(self)
    }

    fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Normalize or return `fallback` for near-zero vectors. Chosen
    /// over `Option<Self>` because every GJK caller has a sensible
    /// default direction for the degenerate case.
    fn normalize_or(self, fallback: Self) -> Self {
        let l2 = self.length_squared();
        if l2 > 1e-12 {
            self * (1.0 / l2.sqrt())
        } else {
            fallback
        }
    }
}

impl VectorOps for Vec3 {
    fn zero() -> Self {
        Vec3::ZERO
    }
    fn dot(self, rhs: Self) -> f32 {
        Vec3::dot(self, rhs)
    }
}

impl VectorOps for Vec4 {
    fn zero() -> Self {
        Vec4::ZERO
    }
    fn dot(self, rhs: Self) -> f32 {
        Vec4::dot(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_ops_match_glam() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(-1.0, 0.5, 2.0);
        assert_eq!(VectorOps::dot(a, b), a.dot(b));
        assert_eq!(VectorOps::length(a), a.length());
        assert!((VectorOps::normalize_or(a, Vec3::Y).length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn normalize_or_handles_zero() {
        let z = Vec3::ZERO;
        let got = VectorOps::normalize_or(z, Vec3::Y);
        assert_eq!(got, Vec3::Y);
    }

    #[test]
    fn vec4_ops_match_glam() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(-1.0, 0.5, 2.0, -3.0);
        assert_eq!(VectorOps::dot(a, b), a.dot(b));
        assert_eq!(VectorOps::length(a), a.length());
    }
}
