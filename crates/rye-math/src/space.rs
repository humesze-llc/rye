//! The [`Space`] trait — Rye's interface to geometry.
//!
//! A `Space` is a Riemannian manifold equipped with an isometry group.
//! Every concrete geometry (Euclidean R³, hyperbolic H³, spherical S³,
//! Euclidean R⁴, …) implements this trait, and every other system in the
//! engine that needs to know *what space means* takes `S: Space` as a
//! generic.
//!
//! The GPU-facing half of the contract lives on [`WgslSpace`], a separate
//! subtrait — see its docs for the rationale.
//!
//! ## Why methods take `&self`
//!
//! Stateless geometries like [`crate::EuclideanR3`] carry no data and
//! monomorphize to direct calls. Parametric geometries — hyperbolic with a
//! curvature scalar, spherical with a radius — need somewhere to store
//! that parameter. `&self` gives them that without forcing stateless impls
//! to pay a runtime cost.
//!
//! ## Tangent vectors carry an implicit base point
//!
//! `Self::Vector` is a tangent vector at *some* point on the manifold; the
//! trait does not enforce *which* point. Callers tracking tangent vectors
//! outside tight numerical kernels should prefer [`crate::Tangent`], which
//! bundles the base point with the vector and exposes safe transport /
//! exponential / scale helpers.

use std::borrow::Cow;

/// A Riemannian manifold with a transitive isometry group.
///
/// Implementors describe a geometry by exposing its point set, tangent
/// vectors, and isometries. Shader integration lives on [`WgslSpace`].
///
/// All methods must be deterministic and side-effect-free.
pub trait Space {
    /// A point on the manifold.
    type Point: Copy + Send + Sync + 'static;
    /// A tangent vector at *some* point. The base point is tracked by the
    /// caller, not the type; use [`crate::Tangent`] when that tracking
    /// should be enforced.
    type Vector: Copy + Send + Sync + 'static;
    /// An orientation-preserving isometry of the manifold.
    type Iso: Copy + Send + Sync + 'static;

    // ---- Riemannian primitives ----------------------------------------

    /// Geodesic distance between two points.
    fn distance(&self, a: Self::Point, b: Self::Point) -> f32;

    /// Exponential map: travel from `at` along the geodesic with initial
    /// velocity `v` for unit time. Inverse of [`Self::log`].
    fn exp(&self, at: Self::Point, v: Self::Vector) -> Self::Point;

    /// Logarithm map: the tangent vector at `from` whose [`Self::exp`]
    /// reaches `to`. Inverse of [`Self::exp`]. Undefined if `to` is in the
    /// cut locus of `from` (e.g. antipode on a sphere); impls should
    /// document their handling.
    fn log(&self, from: Self::Point, to: Self::Point) -> Self::Vector;

    /// Parallel-transport `v` (a tangent vector at `from`) along the
    /// unique minimizing geodesic to `to`, returning the corresponding
    /// tangent vector at `to`.
    fn parallel_transport(
        &self,
        from: Self::Point,
        to: Self::Point,
        v: Self::Vector,
    ) -> Self::Vector;

    // ---- Isometry group -----------------------------------------------

    /// The identity isometry.
    fn iso_identity(&self) -> Self::Iso;

    /// `a ∘ b` — apply `b` first, then `a`.
    fn iso_compose(&self, a: Self::Iso, b: Self::Iso) -> Self::Iso;

    /// Inverse isometry: `iso_compose(a, iso_inverse(a)) == iso_identity()`.
    fn iso_inverse(&self, a: Self::Iso) -> Self::Iso;

    /// Apply an isometry to a point.
    fn iso_apply(&self, iso: Self::Iso, p: Self::Point) -> Self::Point;

    /// Apply an isometry's differential to a tangent vector at `at`. The
    /// result is a tangent vector at `iso_apply(iso, at)`.
    fn iso_transport(
        &self,
        iso: Self::Iso,
        at: Self::Point,
        v: Self::Vector,
    ) -> Self::Vector;
}

/// A [`Space`] that additionally exposes its primitives as WGSL, for
/// inlining into shaders by `rye-shader`.
///
/// Split from [`Space`] deliberately. The math trait is the most stable
/// surface in the engine; the shader ABI is the least — it will evolve
/// with every render-graph revision. Coupling them would marry their
/// release cadences and break Rye's stability-discipline goal.
///
/// Splitting also lets CPU-only consumers (determinism validators,
/// dedicated servers, offline tools) depend on `rye-math` without being
/// forced to know anything about WGSL.
pub trait WgslSpace: Space {
    /// WGSL source providing this space's primitives, for inlining into
    /// shaders by `rye-shader`.
    ///
    /// Stateless geometries return `Cow::Borrowed`; parametric ones may
    /// `format!` constants in and return `Cow::Owned`.
    fn wgsl_impl(&self) -> Cow<'static, str>;
}
