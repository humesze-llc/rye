use std::borrow::Cow;

/// A Riemannian manifold: a metric and the Levi-Civita connection it induces.
/// All methods must be deterministic and side-effect-free.
pub trait Space {
    type Point: Copy + Send + Sync + 'static;
    type Vector: Copy + Send + Sync + 'static;

    fn distance(&self, a: Self::Point, b: Self::Point) -> f32;

    /// Exponential map: travel from `at` along the geodesic with initial velocity `v` for
    /// unit time. Inverse of [`Self::log`].
    fn exp(&self, at: Self::Point, v: Self::Vector) -> Self::Point;

    /// Logarithm map: the tangent vector at `from` whose [`Self::exp`] reaches `to`. Inverse
    /// of [`Self::exp`]. Undefined if `to` is in the cut locus of `from` (e.g. antipode on a
    /// sphere); impls should document their handling.
    fn log(&self, from: Self::Point, to: Self::Point) -> Self::Vector;

    /// The path is implementation-defined: parallel transport is path-dependent
    /// in any non-flat geometry and this signature names no path. Each impl
    /// documents its choice. Callers needing a *specific* path should call
    /// [`Self::parallel_transport_along`] with the polyline explicitly.
    fn parallel_transport(
        &self,
        from: Self::Point,
        to: Self::Point,
        v: Self::Vector,
    ) -> Self::Vector;

    /// Contract: finer subdivision converges to true parallel transport
    /// along the polyline. `path.len() < 2` returns `v` unchanged.
    fn parallel_transport_along(&self, path: &[Self::Point], v: Self::Vector) -> Self::Vector {
        let mut current = v;
        for w in path.windows(2) {
            current = self.parallel_transport(w[0], w[1], current);
        }
        current
    }

    /// Whether the chart is globally flat: chart-coord arithmetic computes the
    /// correct geometry without the Riemannian machinery.
    fn is_chart_flat(&self) -> bool {
        false
    }
}

/// The laws, stated as actions on points because `Iso` carries no equality
/// bound and a rotor representation double-covers its rotation group:
/// `iso_compose` is associative with `iso_identity` neutral and `iso_inverse`
/// two-sided, `iso_apply` preserves [`Space::distance`], and `iso_transport`
/// is the differential of `iso_apply`, so
/// `iso_apply(g, exp(p, v)) == exp(iso_apply(g, p), iso_transport(g, p, v))`.
pub trait IsometryGroup: Space {
    /// An orientation-preserving isometry of the manifold.
    type Iso: Copy + Send + Sync + 'static;

    fn iso_identity(&self) -> Self::Iso;

    /// `a ∘ b`, apply `b` first, then `a`.
    fn iso_compose(&self, a: Self::Iso, b: Self::Iso) -> Self::Iso;

    fn iso_inverse(&self, a: Self::Iso) -> Self::Iso;

    fn iso_apply(&self, iso: Self::Iso, p: Self::Point) -> Self::Point;

    /// Apply an isometry's differential to a tangent vector at `at`. The result is a tangent
    /// vector at `iso_apply(iso, at)`.
    fn iso_transport(&self, iso: Self::Iso, at: Self::Point, v: Self::Vector) -> Self::Vector;
}

pub trait WgslSpace: Space {
    /// WGSL source providing this space's primitives. The v0 ABI is tiny and
    /// single-space (`vec3<f32>` point/vector only):
    ///
    /// ```wgsl
    /// fn loam_distance(a: vec3<f32>, b: vec3<f32>) -> f32
    /// fn loam_exp(at: vec3<f32>, v: vec3<f32>) -> vec3<f32>
    /// fn loam_log(p_from: vec3<f32>, p_to: vec3<f32>) -> vec3<f32>
    /// fn loam_parallel_transport(p_from: vec3<f32>, p_to: vec3<f32>, v: vec3<f32>) -> vec3<f32>
    /// ```
    fn wgsl_impl(&self) -> Cow<'static, str>;
}

#[cfg(test)]
mod tests {
    use super::Space;
    use crate::{
        BlendedSpace, EuclideanR2, EuclideanR3, EuclideanR4, FlatTorus3, HyperbolicH3, LensSpace,
        LinearBlendX, SphericalS3, SphericalS3Embedded,
    };

    #[test]
    fn is_chart_flat_holds_exactly_for_the_flat_geometries() {
        assert!(EuclideanR2.is_chart_flat());
        assert!(EuclideanR3.is_chart_flat());
        assert!(EuclideanR4.is_chart_flat());

        assert!(!HyperbolicH3.is_chart_flat());
        assert!(!SphericalS3.is_chart_flat());
        assert!(!SphericalS3Embedded.is_chart_flat());
        assert!(
            !BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-1.0, 1.0))
                .is_chart_flat()
        );
        // Zero curvature but a periodic chart: chart-coord arithmetic is not
        // lattice-invariant, so it disagrees with the geometry across the
        // gluing. `is_chart_flat` licenses that arithmetic, not flatness.
        assert!(!FlatTorus3::cube(2.0).is_chart_flat());
        assert!(!LensSpace::new(5, 2).is_chart_flat());
    }
}
