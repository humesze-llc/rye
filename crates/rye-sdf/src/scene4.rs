//! Typed 4D scene tree, the 4D analogue of [`crate::scene::Scene`].
//!
//! Build a [`Scene4`] from [`SceneNode4`] combinators and emit
//! WGSL for either:
//!
//! - **Native 4D**: `fn rye_scene_sdf_4d(p: vec4<f32>) -> f32`,
//!   useful for full 4D ray-march renderers (future).
//! - **Hyperslice**: `fn rye_scene_sdf(p: vec3<f32>) -> f32` that
//!   evaluates the 4D SDF at `vec4(p, w_slice)`, where `w_slice`
//!   is a uniform. This is the production path today,
//!   `Hyperslice4DNode` consumes it as the SDF for a 3D ray march.
//!
//! See [`docs/devlog/4D_RENDERING.md`](../../../../docs/devlog/4D_RENDERING.md)
//! for the design rationale.
//!
//! ## Why a parallel `Scene4`, not `Scene<S, const DIM>`
//!
//! Per the 4D-rendering design doc: the 3D and 4D paths share no
//! shader code (different SDF signatures, different ray equations,
//! different uniforms), so dimensioning [`crate::scene::Scene`]
//! generically saves no implementation work and obscures the
//! difference. Parallel hierarchies keep each clear.
//!
//! ## Example
//!
//! ```rust
//! use glam::Vec4;
//! use rye_sdf::scene4::{Scene4, SceneNode4};
//!
//! let scene = Scene4::new(
//!     SceneNode4::hypersphere(Vec4::ZERO, 0.5)
//!         .union(SceneNode4::halfspace(Vec4::Y, 0.0)),
//! );
//! // Native 4D: SDF takes vec4 directly.
//! let wgsl_4d = scene.to_wgsl_4d();
//! assert!(wgsl_4d.contains("fn rye_scene_sdf_4d(p: vec4<f32>) -> f32"));
//! // Hyperslice mode: SDF takes vec3, internally evaluates at
//! // vec4(p, u.w_slice). The `u.w_slice` uniform is supplied by
//! // the render node.
//! let wgsl_hs = scene.to_hyperslice_wgsl("u.w_slice");
//! assert!(wgsl_hs.contains("fn rye_scene_sdf(p3: vec3<f32>) -> f32"));
//! assert!(wgsl_hs.contains("u.w_slice"));
//! ```

use std::boxed::Box;

use glam::Vec4;
use serde::{Deserialize, Serialize};

use crate::primitive4::Primitive4;
pub use rye_shape::Shape;

/// A node in the 4D scene tree. Mirrors
/// [`crate::scene::SceneNode`] but operates on 4D primitives only.
///
/// Smooth-min isn't included today; the math is identical (use the
/// same `smooth_min_fn` wrapper on `f32`) but no demo currently
/// needs it. Add when one does.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SceneNode4 {
    Leaf(Shape),
    Union(Box<SceneNode4>, Box<SceneNode4>),
    Intersection(Box<SceneNode4>, Box<SceneNode4>),
    /// Carve `right` out of `left`: `max(left, −right)`.
    Difference(Box<SceneNode4>, Box<SceneNode4>),
}

impl SceneNode4 {
    // ---- Leaf constructors ------------------------------------------------

    pub fn hypersphere(center: Vec4, radius: f32) -> Self {
        SceneNode4::Leaf(Shape::HyperSphere4D { center, radius })
    }

    /// Half-space (hyperplane) leaf. ℝ⁴ is the only 4D Space rye
    /// ships and it's flat, so [`crate::Primitive4`] emits an
    /// honest `dot(p, n) - offset` hyperplane SDF here. When a
    /// curved 4D Space lands (`BlendedSpace4`, hyperbolic 4-space)
    /// `Primitive4` will grow a `space: &S` parameter and gate this
    /// emission on `WgslSpace::is_chart_flat` the same way the 3D
    /// path does today. The shape itself is canonical, also used
    /// by `rye-physics` for 4D collision walls.
    pub fn halfspace(normal: Vec4, offset: f32) -> Self {
        SceneNode4::Leaf(Shape::HalfSpace4D { normal, offset })
    }

    /// Convex 4D polytope leaf. Note: the static `Primitive4` emit
    /// returns a sentinel today; the production path is via
    /// `Hyperslice4DNode`'s per-frame uniform buffer (the body's
    /// world-space face hyperplanes are computed CPU-side and
    /// uploaded). Until that path lands, polytope leaves render
    /// invisible.
    pub fn polytope(vertices: Vec<Vec4>) -> Self {
        SceneNode4::Leaf(Shape::ConvexPolytope4D { vertices })
    }

    // ---- Combinators ------------------------------------------------------

    pub fn union(self, other: SceneNode4) -> Self {
        SceneNode4::Union(Box::new(self), Box::new(other))
    }

    pub fn intersect(self, other: SceneNode4) -> Self {
        SceneNode4::Intersection(Box::new(self), Box::new(other))
    }

    pub fn subtract(self, other: SceneNode4) -> Self {
        SceneNode4::Difference(Box::new(self), Box::new(other))
    }
}

/// A complete 4D SDF scene, a single root [`SceneNode4`] that
/// emits either `rye_scene_sdf_4d(p: vec4<f32>) -> f32` (full 4D)
/// or `rye_scene_sdf(p: vec3<f32>) -> f32` (hyperslice at the
/// `w_slice` uniform).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene4 {
    pub root: SceneNode4,
}

impl Scene4 {
    pub fn new(root: SceneNode4) -> Self {
        Self { root }
    }

    /// Emit the native 4D SDF module. The entry point is
    /// `fn rye_scene_sdf_4d(p: vec4<f32>) -> f32`. Used by future
    /// full-4D ray-march renderers; the hyperslice path uses
    /// [`Self::to_hyperslice_wgsl`] instead.
    pub fn to_wgsl_4d(&self) -> String {
        let mut helpers = String::new();
        let mut body = String::new();
        let mut counter = 0u32;
        let result_var = emit_node_4d(&self.root, &mut counter, &mut helpers, &mut body);
        format!(
            "// ---- rye-sdf scene4 (native 4D) ----\n\
             {helpers}\
             fn rye_scene_sdf_4d(p: vec4<f32>) -> f32 {{\n\
             {body}\
             \treturn {result_var};\n\
             }}\n"
        )
    }

    /// Emit the hyperslice SDF: `fn rye_scene_sdf(p: vec3<f32>) -> f32`
    /// that evaluates this scene's 4D SDF at `vec4(p, w_slice_expr)`.
    ///
    /// `w_slice_expr` is the WGSL expression the kernel uses to
    /// reach its `w_slice` uniform, typically `"u.w_slice"` when
    /// the node binds a single uniform struct named `u`. Passing a
    /// literal (e.g. `"0.0"`) is also valid for static-slice tests.
    pub fn to_hyperslice_wgsl(&self, w_slice_expr: &str) -> String {
        let mut helpers = String::new();
        let mut body = String::new();
        let mut counter = 0u32;
        let result_var = emit_node_4d(&self.root, &mut counter, &mut helpers, &mut body);
        // Use a distinct parameter name `p3` and an inner `let p`
        // for the 4D point. WGSL forbids declaring a `let` with the
        // same name as the function parameter (no shadowing); naming
        // the parameter `p3` keeps the helper-emit convention (which
        // calls `sdfN_pK(p)`) intact while sidestepping the
        // collision.
        format!(
            "// ---- rye-sdf scene4 (hyperslice at w = {w_slice_expr}) ----\n\
             {helpers}\
             fn rye_scene_sdf(p3: vec3<f32>) -> f32 {{\n\
             \tlet p = vec4<f32>(p3, {w_slice_expr});\n\
             {body}\
             \treturn {result_var};\n\
             }}\n"
        )
    }

    pub fn from_ron(src: &str) -> Result<Self, ron::error::SpannedError> {
        ron::from_str(src)
    }

    pub fn to_ron(&self) -> Result<String, ron::Error> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
    }
}

/// Walk the 4D scene tree, append helper functions to `helpers`
/// and `let` bindings to `body`. Returns the WGSL variable holding
/// the signed distance for this node.
fn emit_node_4d(
    node: &SceneNode4,
    counter: &mut u32,
    helpers: &mut String,
    body: &mut String,
) -> String {
    let idx = *counter;
    *counter += 1;
    match node {
        SceneNode4::Leaf(prim) => {
            let fn_name = format!("sdf4_p{idx}");
            helpers.push_str(&prim.to_wgsl_4d(&fn_name));
            let var = format!("d{idx}");
            body.push_str(&format!("\tlet {var} = {fn_name}(p);\n"));
            var
        }
        SceneNode4::Union(left, right) => {
            let lv = emit_node_4d(left, counter, helpers, body);
            let rv = emit_node_4d(right, counter, helpers, body);
            let var = format!("d{idx}");
            body.push_str(&format!("\tlet {var} = min({lv}, {rv});\n"));
            var
        }
        SceneNode4::Intersection(left, right) => {
            let lv = emit_node_4d(left, counter, helpers, body);
            let rv = emit_node_4d(right, counter, helpers, body);
            let var = format!("d{idx}");
            body.push_str(&format!("\tlet {var} = max({lv}, {rv});\n"));
            var
        }
        SceneNode4::Difference(left, right) => {
            let lv = emit_node_4d(left, counter, helpers, body);
            let rv = emit_node_4d(right, counter, helpers, body);
            let var = format!("d{idx}");
            body.push_str(&format!("\tlet {var} = max({lv}, -({rv}));\n"));
            var
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_hypersphere_emits_4d_scene() {
        let scene = Scene4::new(SceneNode4::hypersphere(Vec4::ZERO, 0.25));
        let wgsl = scene.to_wgsl_4d();
        assert!(wgsl.contains("fn rye_scene_sdf_4d(p: vec4<f32>) -> f32"));
        assert!(wgsl.contains("length(p"));
        assert!(wgsl.contains("- (0.25)"));
    }

    #[test]
    fn hyperslice_wraps_4d_with_w_slice() {
        let scene = Scene4::new(SceneNode4::hypersphere(Vec4::ZERO, 0.5));
        let wgsl = scene.to_hyperslice_wgsl("u.w_slice");
        // Parameter is `p3` to avoid colliding with the inner `let
        // p` 4D point, WGSL doesn't allow declaring a let with the
        // same name as a function parameter.
        assert!(wgsl.contains("fn rye_scene_sdf(p3: vec3<f32>) -> f32"));
        assert!(wgsl.contains("let p = vec4<f32>(p3, u.w_slice)"));
        // The hyperslice emit reuses the 4D SDF helpers, so the
        // sphere's `length(p - ...)` body is still present.
        assert!(wgsl.contains("length(p"));
    }

    #[test]
    fn union_of_two_hyperspheres() {
        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::ZERO, 0.2)
                .union(SceneNode4::hypersphere(Vec4::X * 0.5, 0.2)),
        );
        let wgsl = scene.to_wgsl_4d();
        assert!(wgsl.contains("min("));
        assert!(wgsl.contains("sdf4_p1"));
        assert!(wgsl.contains("sdf4_p2"));
    }

    #[test]
    fn difference_uses_negation_on_4d() {
        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::ZERO, 0.3).subtract(SceneNode4::halfspace(Vec4::Y, 0.0)),
        );
        let wgsl = scene.to_wgsl_4d();
        assert!(wgsl.contains("max("));
        assert!(wgsl.contains("-("));
    }

    #[test]
    fn intersection_emits_max() {
        let scene = Scene4::new(
            SceneNode4::halfspace(Vec4::Y, 0.0).intersect(SceneNode4::hypersphere(Vec4::ZERO, 0.4)),
        );
        let wgsl = scene.to_wgsl_4d();
        assert!(wgsl.contains("max("));
    }

    #[test]
    fn ron_round_trip_4d() {
        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::ZERO, 0.3).union(SceneNode4::halfspace(Vec4::Y, -0.4)),
        );
        let ron_str = scene.to_ron().expect("serialize");
        let recovered: Scene4 = Scene4::from_ron(&ron_str).expect("deserialize");
        assert_eq!(scene.to_wgsl_4d(), recovered.to_wgsl_4d());
    }

    /// Polytope leaves still emit (their helper returns the
    /// sentinel today). The combinator path doesn't break on
    /// polytope leaves; it just produces a far-away surface.
    #[test]
    fn polytope_leaf_emits_sentinel_helper() {
        let scene = Scene4::new(SceneNode4::polytope(vec![Vec4::ZERO; 5]));
        let wgsl = scene.to_wgsl_4d();
        assert!(wgsl.contains("fn sdf4_p0(_p: vec4<f32>) -> f32"));
        assert!(wgsl.contains("return 1e9"));
    }
}
