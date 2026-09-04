use std::ops::{Add, Mul, Neg, Sub};

use glam::{Vec2, Vec3, Vec4};

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
    fn normalize_or_handles_zero() {
        let z = Vec3::ZERO;
        let got = VectorOps::normalize_or(z, Vec3::Y);
        assert_eq!(got, Vec3::Y);
    }
}
