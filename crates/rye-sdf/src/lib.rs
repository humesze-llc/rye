//! `rye-sdf`: signed-distance field primitives and scene builders for Rye.
//!
//! [`Primitive`] is the typed abstraction for geometric objects. Every
//! primitive emits a WGSL function `fn {name}(p: vec3<f32>) -> f32` that
//! uses only `rye_*` Space-prelude functions, guaranteeing correctness
//! across E³, H³, and S³.
//!
//! [`combinator`] provides Space-agnostic combinators (union, intersection,
//! smooth-min) that operate on the scalar distances returned by primitive SDFs.
//!
//! Demo-shaped scene wrappers (geodesic spheres, corridor, lattice)
//! live in their respective `examples/<name>/scene.rs` files. The
//! crate proper keeps only the typed primitive + scene layer.

pub mod combinator;
pub mod primitive;
pub mod primitive4;
pub mod scene;
pub mod scene4;

pub use primitive::Primitive;
pub use primitive4::Primitive4;
pub use rye_shape::Shape;
pub use scene::{PrimitiveKind, Scene, SceneNode};
pub use scene4::{Scene4, SceneNode4};

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    // ---- Primitive trait tests -------------------------------------------

    #[test]
    fn sphere_emits_rye_distance_call() {
        use rye_math::EuclideanR3;
        let s = Shape::sphere_at(Vec3::ZERO, 0.25);
        let src = s.to_wgsl(&EuclideanR3, "sdf_0");
        assert!(src.contains("fn sdf_0(p: vec3<f32>) -> f32"));
        assert!(src.contains("rye_distance"));
        assert!(src.contains("0.250000"));
    }

    #[test]
    fn sphere_wgsl_is_space_agnostic() {
        use rye_math::{EuclideanR3, HyperbolicH3, SphericalS3};
        let s = Shape::sphere_at_origin(0.3);
        let e3 = s.to_wgsl(&EuclideanR3, "sdf_0");
        let h3 = s.to_wgsl(&HyperbolicH3, "sdf_0");
        let s3 = s.to_wgsl(&SphericalS3, "sdf_0");
        // The emitted body must be identical across spaces, only
        // rye_distance differs at prelude link time, not in the
        // emitted text.
        assert_eq!(e3, h3);
        assert_eq!(h3, s3);
    }

    /// `HalfSpace`'s emission gates on `WgslSpace::is_chart_flat`.
    /// EuclideanR3 reports flat, so the chart-coord `dot(p, n) - d`
    /// formula is honest and gets emitted. Curved Spaces fall
    /// through to the sentinel arm (covered separately).
    #[test]
    fn halfspace_emits_dot_in_flat_chart() {
        use rye_math::EuclideanR3;
        let p = Shape::HalfSpace {
            normal: Vec3::Y,
            offset: -0.5,
        };
        let src = p.to_wgsl(&EuclideanR3, "sdf_floor");
        assert!(src.contains("fn sdf_floor(p: vec3<f32>) -> f32"));
        assert!(src.contains("dot(p,"));
        // Floor at y = -0.5: normal = (0, 1, 0), offset = -0.5.
        assert!(src.contains("-0.500000"));
    }

    /// `HalfSpace` in a curved Space has no honest closed-form SDF
    /// today, so it sentinels until artanh-of-Möbius (H³) /
    /// chord-distance (S³) implementations land. Pinned here so a
    /// future regression that re-enables raw `dot()` in curved
    /// Spaces fails loud.
    #[test]
    fn halfspace_sentinels_in_curved_chart() {
        use rye_math::HyperbolicH3;
        let p = Shape::HalfSpace {
            normal: Vec3::Y,
            offset: -0.5,
        };
        let src = p.to_wgsl(&HyperbolicH3, "sdf_floor");
        assert!(src.contains("fn sdf_floor(_p: vec3<f32>) -> f32"));
        assert!(src.contains("return 1e9"));
        assert!(
            !src.contains("dot(p,"),
            "HalfSpace must not emit raw chart-coord dot product in curved Spaces",
        );
    }

    #[test]
    fn box_emits_euclidean_box_sdf() {
        use rye_math::EuclideanR3;
        let b = Shape::Box3 {
            half_extents: Vec3::splat(0.4),
        };
        let src = b.to_wgsl(&EuclideanR3, "sdf_box");
        assert!(src.contains("fn sdf_box(p: vec3<f32>) -> f32"));
        assert!(src.contains("abs(p)"));
        assert!(src.contains("0.400000"));
    }

    #[test]
    fn combinator_union_expr() {
        use combinator::union_expr;
        let expr = union_expr("da", "db");
        assert_eq!(expr, "min(da, db)");
    }

    #[test]
    fn combinator_smooth_min_fn_compiles() {
        use combinator::smooth_min_fn;
        let src = smooth_min_fn("smin", 0.08);
        assert!(src.contains("fn smin(a: f32, b: f32) -> f32"));
        assert!(src.contains("0.080000"));
        assert!(src.contains("clamp"));
        assert!(src.contains("mix"));
    }

    // ---- Scene-tree integration tests ------------------------------------
    //
    // These cover behaviours the legacy demo-scene tests used to gate.
    // The demo wrappers themselves now live in their respective examples;
    // this layer pins the behaviour at the underlying typed-scene API.

    /// A Scene with a sphere and a half-space plane in flat E³ must
    /// emit both `rye_distance` (sphere) and `dot(p,` (plane), all
    /// inside a single `rye_scene_sdf` entry point.
    #[test]
    fn scene_with_sphere_and_plane_emits_both_paths_in_e3() {
        use rye_math::EuclideanR3;
        let scene =
            Scene::new(SceneNode::sphere(Vec3::ZERO, 0.22).union(SceneNode::plane(Vec3::Y, -0.5)));
        let src = scene.to_wgsl(&EuclideanR3);
        assert!(src.contains("fn rye_scene_sdf"));
        assert!(src.contains("rye_distance"));
        assert!(src.contains("dot(p,"));
        // Plane offset literal appears in the emitted dot() call.
        assert!(src.contains("-0.500000"));
    }

    /// A sphere-only scene must not emit any `dot()` calls; the
    /// half-space gate stays inert when no plane leaves are present.
    #[test]
    fn sphere_only_scene_emits_no_chart_coord_dot() {
        use rye_math::EuclideanR3;
        let scene = Scene::new(SceneNode::sphere(Vec3::ZERO, 0.3));
        let src = scene.to_wgsl(&EuclideanR3);
        assert!(src.contains("fn rye_scene_sdf"));
        assert!(src.contains("rye_distance"));
        assert!(!src.contains("dot(p,"));
    }

    /// Sphere centres baked into the WGSL must literally match the
    /// input point. Pins the per-sphere literal-emission contract the
    /// lattice / corridor scenes rely on.
    #[test]
    fn sphere_center_appears_as_wgsl_literal() {
        use rye_math::EuclideanR3;
        let scene = Scene::new(SceneNode::sphere(Vec3::new(0.5, 0.0, 0.0), 0.1));
        let src = scene.to_wgsl(&EuclideanR3);
        assert!(src.contains("0.500000, 0.000000, 0.000000"));
    }

    /// Same construction in H³ must NOT emit the E³-style literal,
    /// because the lattice-style usage pre-computes centres via
    /// `space.exp` and tanh-compresses them. This test fakes the
    /// compression by exping a tangent vector through HyperbolicH3
    /// and confirming the emitted literal differs.
    #[test]
    fn lattice_centres_compress_under_hyperbolic_exp() {
        use rye_math::{HyperbolicH3, Space};
        let p = HyperbolicH3.exp(Vec3::ZERO, Vec3::X * 0.5);
        let scene = Scene::new(SceneNode::sphere(p, 0.1));
        let src = scene.to_wgsl(&HyperbolicH3);
        // tanh(0.25) ≈ 0.2449, well under 0.5.
        assert!(p.x < 0.5);
        assert!(!src.contains("0.500000, 0.000000, 0.000000"));
    }
}
