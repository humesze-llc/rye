//! [`Tangent`], a tangent vector bundled with the point it lives at.

use std::ops::Mul;

use crate::space::Space;

/// A tangent vector `v` paired with the point `at` where it lives.
pub struct Tangent<S: Space> {
    /// Base point, assumed on the manifold.
    pub at: S::Point,
    /// Tangent vector at [`Self::at`]. Reassigning `at` alone does not
    /// transport it; use [`Tangent::transport_to`].
    pub v: S::Vector,
}

// Hand-rolled Copy/Clone: `#[derive]` adds an unwanted `S: Clone` bound.
impl<S: Space> Clone for Tangent<S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: Space> Copy for Tangent<S> {}

impl<S: Space> Tangent<S> {
    /// Pairs an existing vector with its base point; `v` is taken as already
    /// expressed in the tangent space at `at`, not projected into it.
    pub fn new(at: S::Point, v: S::Vector) -> Self {
        Self { at, v }
    }

    /// Parallel-transport this tangent vector to a new base point along the unique minimizing
    /// geodesic.
    pub fn transport_to(self, space: &S, dest: S::Point) -> Self {
        Self {
            at: dest,
            v: space.parallel_transport(self.at, dest, self.v),
        }
    }

    /// Exponential map: walk from `at` along `v` for unit time.
    pub fn exp(self, space: &S) -> S::Point {
        space.exp(self.at, self.v)
    }
}

impl<S: Space> Tangent<S>
where
    S::Vector: Mul<f32, Output = S::Vector>,
{
    /// Scale the tangent vector by `t`, keeping the base point fixed.
    pub fn scale(self, t: f32) -> Self {
        Self {
            at: self.at,
            v: self.v * t,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EuclideanR3;
    use glam::Vec3;

    #[test]
    fn exp_in_flat_space_is_at_plus_v() {
        let t = Tangent::<EuclideanR3>::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.5, 0.0, -0.5));
        assert_eq!(t.exp(&EuclideanR3), Vec3::new(1.5, 2.0, 2.5));
    }

    #[test]
    fn scale_preserves_base_and_is_invertible() {
        let original = Tangent::<EuclideanR3>::new(Vec3::Y, Vec3::new(2.0, 3.0, 4.0));
        let scaled = original.scale(2.5);
        assert_eq!(scaled.at, Vec3::Y);
        assert_eq!(scaled.v, Vec3::new(5.0, 7.5, 10.0));
        let back = scaled.scale(1.0 / 2.5);
        assert!((back.v - original.v).length() < 1e-6);
    }

    #[test]
    fn transport_to_in_flat_space_preserves_vector() {
        let t = Tangent::<EuclideanR3>::new(Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0));
        let moved = t.transport_to(&EuclideanR3, Vec3::new(5.0, 0.0, 0.0));
        assert_eq!(moved.at, Vec3::new(5.0, 0.0, 0.0));
        assert_eq!(moved.v, Vec3::new(1.0, 2.0, 3.0));
    }
}
