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
    ///
    /// **Path uniqueness.** This signature implicitly assumes a
    /// canonical geodesic exists between `from` and `to`. That holds
    /// for the closed-form Spaces ($\mathbb{E}^3$, $H^3$, $S^3$ away
    /// from antipodes, $\mathbb{E}^4$) but breaks down in Spaces with
    /// non-uniform curvature where multiple geodesics may connect two
    /// points (or constructing the geodesic requires solving a
    /// boundary-value problem). Callers that already know the path
    /// they want frame transport along should prefer
    /// [`Self::parallel_transport_along`], which takes the path
    /// explicitly and avoids the BVP entirely.
    fn parallel_transport(
        &self,
        from: Self::Point,
        to: Self::Point,
        v: Self::Vector,
    ) -> Self::Vector;

    /// Parallel-transport `v` along the piecewise-geodesic path
    /// through the listed points, segment by segment. Returns the
    /// transported vector at the final point.
    ///
    /// **Why this is the path-aware primitive.** Parallel transport
    /// is *path-dependent* in any non-flat geometry; "transport from
    /// `a` to `b`" is only well-defined once a path is chosen.
    /// [`Self::parallel_transport`] picks the geodesic between the
    /// endpoints; this method lets the caller pick *any* path it
    /// already knows (e.g. the polyline a camera or player traversed
    /// over the last few frames). Spaces with expensive geodesic
    /// construction — `BlendedSpace`, future numerical-only Spaces —
    /// should override this method to integrate the polyline
    /// directly rather than reconstructing each segment's geodesic.
    ///
    /// Edge cases:
    /// - `path.len() < 2`: returns `v` unchanged.
    /// - Consecutive duplicate points contribute identity transports.
    ///
    /// The default implementation chains
    /// [`Self::parallel_transport`] over consecutive pairs, which is
    /// exact for closed-form Spaces.
    fn parallel_transport_along(&self, path: &[Self::Point], v: Self::Vector) -> Self::Vector {
        let mut current = v;
        for w in path.windows(2) {
            current = self.parallel_transport(w[0], w[1], current);
        }
        current
    }

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
    fn iso_transport(&self, iso: Self::Iso, at: Self::Point, v: Self::Vector) -> Self::Vector;
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
    /// The v0 shader ABI is deliberately tiny and single-space:
    ///
    /// ```wgsl
    /// fn rye_distance(a: vec3<f32>, b: vec3<f32>) -> f32
    /// fn rye_exp(at: vec3<f32>, v: vec3<f32>) -> vec3<f32>
    /// fn rye_log(p_from: vec3<f32>, p_to: vec3<f32>) -> vec3<f32>
    /// fn rye_parallel_transport(p_from: vec3<f32>, p_to: vec3<f32>, v: vec3<f32>) -> vec3<f32>
    /// ```
    ///
    /// This ABI intentionally covers only spaces whose point/vector shader
    /// representation is `vec3<f32>`. `Iso` layout and multi-space name
    /// mangling are left out until a render path actually needs them.
    ///
    /// Stateless geometries return `Cow::Borrowed`; parametric ones may
    /// `format!` constants in and return `Cow::Owned`.
    fn wgsl_impl(&self) -> Cow<'static, str>;
}
