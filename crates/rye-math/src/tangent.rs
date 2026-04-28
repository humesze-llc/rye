//! [`Tangent`], a tangent vector bundled with the point it lives at.
//!
//! In curved geometry, a tangent vector without its base point is
//! geometrically meaningless. The [`Space`] trait permits bare
//! `Self::Vector` for performance, but mistakes there produce *silently*
//! wrong physics rather than crashes, the worst kind of bug. `Tangent`
//! is the recommended holder outside tight numerical kernels.

use std::ops::Mul;

use crate::space::Space;

/// A tangent vector `v` paired with the point `at` where it lives.
///
/// Construct directly or via [`Tangent::new`]. Use [`Tangent::transport_to`]
/// to move to a different base point along a geodesic, [`Tangent::exp`] to
/// walk along the vector, and [`Tangent::scale`] to rescale without
/// moving.
pub struct Tangent<S: Space> {
    pub at: S::Point,
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
    pub fn new(at: S::Point, v: S::Vector) -> Self {
        Self { at, v }
    }

    /// Parallel-transport this tangent vector to a new base point along
    /// the unique minimizing geodesic.
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
