//! WGSL emission for [`rye_shape::Shape`].
//!
//! The shape data model lives in `rye-shape` — that's where `Sphere`,
//! `HalfSpace`, `Box3`, and the polytope variants are defined so that
//! `rye-physics` and `rye-sdf` share one canonical type. This module
//! is the rendering half: it implements the [`Primitive`] extension
//! trait on [`Shape`], dispatching per-variant to the WGSL formula
//! appropriate to each.
//!
//! ## Variants and their SDFs
//!
//! - [`Shape::Sphere`] — `rye_distance(p, center) − radius`. The
//!   center is part of the shape (unlike physics, where the body's
//!   position is the center); SDF scenes use it to place spheres
//!   without a transform combinator.
//! - [`Shape::HalfSpace`] — Euclidean `dot(p, n) − offset`. Exact in
//!   E³; in H³/S³ this is the Euclidean-coordinate half-space as
//!   embedded in the Space's chart, matching the corridor/lattice
//!   demos' convention. Geodesic-plane formulas will land in a
//!   follow-up.
//! - [`Shape::Box3`] — standard Euclidean box SDF (`max(abs(p) − b, 0)`
//!   and the negative-interior correction). Axis-aligned, centered at
//!   the local frame's origin.
//!
//! Variants without a WGSL emission today (`Polygon2D`,
//! `ConvexPolytope3D`, `ConvexPolytope4D`) return a stub SDF that
//! evaluates to `+infinity` — they're skipped in scene emission.

use rye_math::WgslSpace;
use rye_shape::Shape;

/// Extension trait on [`Shape`] that emits its signed-distance
/// function as WGSL.
///
/// The emitted function has signature
/// `fn {name}(p: vec3<f32>) -> f32` and must call only `rye_*`
/// functions from the Space prelude — never raw coordinate
/// arithmetic — so that correctness is preserved across E³, H³, and
/// S³.
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
            Shape::HalfSpace { normal, offset } => format!(
                "fn {name}(p: vec3<f32>) -> f32 {{\n\
                 \treturn dot(p, vec3<f32>({nx:.6}, {ny:.6}, {nz:.6})) - ({d:.6});\n\
                 }}\n",
                name = name,
                nx = normal.x,
                ny = normal.y,
                nz = normal.z,
                d = offset,
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
            Shape::Polygon2D { .. }
            | Shape::ConvexPolytope3D { .. }
            | Shape::ConvexPolytope4D { .. }
            | Shape::HalfSpace4D { .. } => {
                // No SDF emission today for vertex-list shapes; emit
                // a far-away-sentinel so they don't break scene
                // assembly if accidentally included. Rendering these
                // properly needs either a compiled mesh-SDF bake or
                // a runtime convex-hull SDF kernel — both follow-ups.
                format!("fn {name}(_p: vec3<f32>) -> f32 {{\n\treturn 1e9;\n}}\n",)
            }
        }
    }
}
