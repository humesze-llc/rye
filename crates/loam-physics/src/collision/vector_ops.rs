use std::ops::{Add, Mul, Neg, Sub};

use glam::{Vec2, Vec3, Vec4};

/// GJK walks the Minkowski difference on vector algebra and dot products
/// alone. EPA's face-normal reconstruction is dimension-specific and stays
/// outside this trait.
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
    fn is_finite(self) -> bool;

    fn length_squared(self) -> f32 {
        self.dot(self)
    }

    fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// `fallback` for near-zero vectors, chosen over `Option<Self>` because
    /// every GJK caller has a default direction for the degenerate case.
    fn normalize_or(self, fallback: Self) -> Self {
        let l2 = self.length_squared();
        if l2 > 1e-12 {
            self * (1.0 / l2.sqrt())
        } else {
            fallback
        }
    }
}

impl VectorOps for Vec2 {
    fn zero() -> Self {
        Vec2::ZERO
    }
    fn dot(self, rhs: Self) -> f32 {
        Vec2::dot(self, rhs)
    }
    fn is_finite(self) -> bool {
        Vec2::is_finite(self)
    }
}

impl VectorOps for Vec3 {
    fn zero() -> Self {
        Vec3::ZERO
    }
    fn dot(self, rhs: Self) -> f32 {
        Vec3::dot(self, rhs)
    }
    fn is_finite(self) -> bool {
        Vec3::is_finite(self)
    }
}

impl VectorOps for Vec4 {
    fn zero() -> Self {
        Vec4::ZERO
    }
    fn dot(self, rhs: Self) -> f32 {
        Vec4::dot(self, rhs)
    }
    fn is_finite(self) -> bool {
        Vec4::is_finite(self)
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
    fn vec2_ops_match_glam() {
        let a = Vec2::new(3.0, 4.0);
        let b = Vec2::new(1.0, -2.0);
        assert_eq!(VectorOps::dot(a, b), a.dot(b));
        assert_eq!(VectorOps::length(a), 5.0);
        assert!(a.is_finite());
        assert!(!Vec2::new(f32::NAN, 0.0).is_finite());
    }
}
