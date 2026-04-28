//! WGSL emission for [`rye_shape::Shape`].
//!
//! The shape data model lives in `rye-shape`, that's where `Sphere`,
//! `HalfSpace`, `Box3`, and the polytope variants are defined so that
//! `rye-physics` and `rye-sdf` share one canonical type. This module
//! is the rendering half: it implements the [`Primitive`] extension
//! trait on [`Shape`], dispatching per-variant to the WGSL formula
//! appropriate to each.
//!
//! ## Variants and their SDFs
//!
//! - [`Shape::Sphere`], `rye_distance(p, center) − radius`. The
//!   center is part of the shape (unlike physics, where the body's
//!   position is the center); SDF scenes use it to place spheres
//!   without a transform combinator.
//! - [`Shape::Box3`], standard Euclidean box SDF (`max(abs(p) − b, 0)`
//!   and the negative-interior correction). Axis-aligned, centered at
//!   the local frame's origin. Honest in E³; chart-coordinate in
//!   H³/S³ (the trait rule treats this as accepted because no
//!   geodesic-box SDF exists in closed form).
//!
//! Variants that emit a `+1e9` invisible-far-away sentinel:
//!
//! - [`Shape::HalfSpace`] / [`Shape::HalfSpace4D`]: a chart-coordinate
//!   `dot(p, n) − offset` would emit visibly wrong geometry in H³
//!   and S³ (it draws the chart's straight plane, not the geodesic
//!   plane). The shape stays in `rye-shape` because physics still
//!   uses it for collision walls; only the SDF emission is gated
//!   until closed-form geodesic-plane SDFs land (artanh-of-Möbius
//!   in H³, chord-distance to a great hyperplane in S³).
//! - [`Shape::Polygon2D`], [`Shape::ConvexPolytope3D`],
//!   [`Shape::ConvexPolytope4D`]: vertex-list shapes that need
//!   either a baked mesh-SDF or a runtime convex-hull kernel.
//! - [`Shape::HyperSphere4D`]: 4D variant; lives in [`Primitive4`].

use rye_math::WgslSpace;
use rye_shape::Shape;

/// Extension trait on [`Shape`] that emits its signed-distance
/// function as WGSL.
///
/// The emitted function has signature
/// `fn {name}(p: vec3<f32>) -> f32` and must call only `rye_*`
/// functions from the Space prelude, never raw coordinate
/// arithmetic, so that correctness is preserved across E³, H³, and
/// S³. Variants that can't honour this rule (because no
/// closed-form Space-aware SDF exists yet) emit a `+1e9` sentinel
/// rather than a chart-coordinate approximation that would silently
/// render wrong in curved spaces. See the module-level doc for the
/// per-variant status.
pub trait Primitive {
    /// Emit a WGSL function named `name` that returns the signed
    /// distance from `p` to `self` in the given Space.
    fn to_wgsl<S: WgslSpace>(&self, space: &S, name: &str) -> String;
}

impl Primitive for Shape {
    fn to_wgsl<S: WgslSpace>(&self, _space: &S, name: &str) -> String {
        match self {
            Shape::Sphere { center, radius } => format!(
                "fn {name}(p: vec3<f32>) -> f32 {{\n\
                 \treturn rye_distance(p, vec3<f32>({cx:.6}, {cy:.6}, {cz:.6})) - {r:.6};\n\
                 }}\n",
                name = name,
                cx = center.x,
                cy = center.y,
                cz = center.z,
                r = radius,
            ),
            Shape::Box3 { half_extents } => format!(
                "fn {name}(p: vec3<f32>) -> f32 {{\n\
                 \tlet b = vec3<f32>({hx:.6}, {hy:.6}, {hz:.6});\n\
                 \tlet q = abs(p) - b;\n\
                 \treturn length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);\n\
                 }}\n",
                name = name,
                hx = half_extents.x,
                hy = half_extents.y,
                hz = half_extents.z,
            ),
            Shape::HalfSpace { .. }
            | Shape::HalfSpace4D { .. }
            | Shape::Polygon2D { .. }
            | Shape::ConvexPolytope3D { .. }
            | Shape::ConvexPolytope4D { .. }
            | Shape::HyperSphere4D { .. } => {
                // Sentinel emission. `HalfSpace`/`HalfSpace4D` would
                // need closed-form geodesic-plane SDFs in H³ / S³ to
                // honour the trait rule; vertex-list shapes need a
                // baked mesh-SDF or a runtime convex-hull kernel. A
                // `+1e9` SDF renders as an invisible far-away
                // surface so accidental inclusion fails visibly
                // rather than silently drawing wrong geometry.
                format!("fn {name}(_p: vec3<f32>) -> f32 {{\n\treturn 1e9;\n}}\n",)
            }
        }
    }
}
