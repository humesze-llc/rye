//! `Primitive4`, 4D-SDF emit for the [`rye_shape::Shape`] variants
//! that live in ℝ⁴.
//!
//! Mirrors [`crate::Primitive`] for the 3D path, but operates on 4D
//! points and emits WGSL functions with the signature
//!
//! ```text
//! fn <name>(p: vec4<f32>) -> f32
//! ```
//!
//! Returns the signed distance to the shape's surface in 4D
//! (positive outside, negative inside, zero on the boundary). The
//! hyperslicing render path consumes these by passing
//! `vec4(xyz, w_slice)` and treating the returned scalar as a 3D
//! SDF for ray-marching at that fixed `w`. The eventual full 4D
//! ray-march path will consume the same emit but march a 4D ray.
//!
//! ## Variant coverage
//!
//! | `Shape` variant | 4D SDF emit | Notes |
//! |---|---|---|
//! | [`Shape::HyperSphere4D`] | closed-form | `length(p − center) − radius` |
//! | [`Shape::HalfSpace4D`] | sentinel `1e9` | a chart-coord `dot(p, n) − offset` would render visibly wrong if a future `BlendedSpace4` lands; gated until a closed-form geodesic-hyperplane SDF replaces it |
//! | [`Shape::ConvexPolytope4D`] | sentinel `1e9` | the production path is `Hyperslice4DNode`'s per-frame uniform buffer (CPU computes face hyperplanes from world-transformed vertices, ships them to GPU); this static emit is intentionally inert |
//! | All 3D-only variants (`Sphere`, `Box3`, `HalfSpace`, ...) | sentinel `1e9` | shouldn't appear in `Scene4`; sentinel renders invisible if accidentally included |
//!
//! The polytope emit currently builds the half-spaces inside the
//! function on every call, fine for static scenes (5 to 24
//! vertices), expensive for arbitrary user polytopes. When perf
//! pressure shows up, the half-space derivation moves to the CPU
//! side and the WGSL emit becomes a max-over-static-array.

use rye_shape::Shape;

/// Emit a WGSL function that evaluates the 4D signed-distance
/// field of this shape. Counterpart of [`crate::Primitive`] for
/// 4D variants.
///
/// Same trait rule as the 3D `Primitive`: emitted SDFs must use
/// Space-aware primitives only, never raw chart-coord arithmetic.
/// The 4D pipeline is single-Space today (Euclidean R⁴), so the
/// rule is forward-looking, but variants that would emit
/// chart-coord formulas (e.g. `HalfSpace4D`'s `dot(p, n) - offset`)
/// emit the `+1e9` sentinel rather than ship a future-broken
/// formula. See the module-level table for the per-variant status.
pub trait Primitive4 {
    /// Generate a WGSL function definition. Caller picks `name`;
    /// the function takes a `vec4<f32>` and returns a scalar `f32`.
    fn to_wgsl_4d(&self, name: &str) -> String;
}

impl Primitive4 for Shape {
    fn to_wgsl_4d(&self, name: &str) -> String {
        match self {
            // Closed-form 4-ball SDF.
            Shape::HyperSphere4D { center, radius } => format!(
                "fn {name}(p: vec4<f32>) -> f32 {{\n\
                \treturn length(p - vec4<f32>({cx}, {cy}, {cz}, {cw})) - ({radius});\n\
                }}\n",
                name = name,
                cx = center.x,
                cy = center.y,
                cz = center.z,
                cw = center.w,
                radius = radius,
            ),

            // Convex polytope: SDF is `max_i (n_i · p − d_i)` over
            // the polytope's outward face hyperplanes. With only the
            // vertex list available we'd have to derive those at
            // runtime, which gets gnarly in WGSL.
            //
            // For the demos that actually render polytopes today
            // (`pentatope_slice` and follow-on 4D scenes), the body's
            // pose changes per frame, so the natural design is:
            //   * CPU computes face hyperplanes from the world-
            //     transformed vertices and ships them to the GPU
            //     via a uniform buffer.
            //   * WGSL takes those as input rather than emitting
            //     constants.
            //
            // That's a job for `Hyperslice4DNode` (step 3 of the 4D
            // rendering plan), not for this static emit. Keep this
            // emit as a sentinel until then so the trait is
            // exhaustive but accidental scene inclusion fails
            // visibly (sentinel return = 1e9 = invisible far-away
            // surface).
            Shape::ConvexPolytope4D { .. } => format!(
                "fn {name}(_p: vec4<f32>) -> f32 {{\n\
                \t// ConvexPolytope4D: half-space emit lives in the\n\
                \t// render node's per-frame uniform path, not here.\n\
                \t// See Hyperslice4DNode.\n\
                \treturn 1e9;\n\
                }}\n",
            ),

            // `HalfSpace4D` joins this arm: a chart-coord `dot(p, n)
            // - offset` would render visibly wrong if a future
            // `BlendedSpace4` is plugged in. Sentinel until a
            // closed-form geodesic-hyperplane SDF replaces it. The
            // 3D / 2D variants don't belong in a 4D scene at all;
            // sentinel keeps the trait exhaustive without producing
            // type-mismatched WGSL.
            Shape::HalfSpace4D { .. }
            | Shape::Sphere { .. }
            | Shape::HalfSpace { .. }
            | Shape::Box3 { .. }
            | Shape::Polygon2D { .. }
            | Shape::ConvexPolytope3D { .. } => format!(
                "fn {name}(_p: vec4<f32>) -> f32 {{\n\
                \treturn 1e9;\n\
                }}\n",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec4;

    /// Each emit produces a syntactically valid WGSL function with
    /// the expected `vec4<f32> → f32` signature. We don't run the
    /// shader here (that's the naga-validation test in
    /// `rye-shader`); we just sanity-check the textual output so
    /// the round-trip with the WGSL builder downstream stays
    /// stable.
    #[test]
    fn hypersphere_4d_emit_has_expected_shape() {
        let s = Shape::HyperSphere4D {
            center: Vec4::new(1.0, 2.0, 3.0, 4.0),
            radius: 0.5,
        };
        let wgsl = s.to_wgsl_4d("ball");
        assert!(wgsl.contains("fn ball(p: vec4<f32>) -> f32"));
        assert!(wgsl.contains("length(p - vec4<f32>(1, 2, 3, 4))"));
        assert!(wgsl.contains("- (0.5)"));
    }

    /// `HalfSpace4D` no longer emits a chart-coord `dot(p, n) - d`:
    /// the trait rule (forward-looking for any future BlendedSpace4)
    /// forbids raw coordinate arithmetic. Until a closed-form
    /// geodesic-hyperplane SDF replaces it, the variant emits the
    /// `+1e9` invisible-far-away sentinel.
    #[test]
    fn halfspace_4d_emits_sentinel_sdf() {
        let s = Shape::HalfSpace4D {
            normal: Vec4::new(0.0, 1.0, 0.0, 0.0),
            offset: 0.0,
        };
        let wgsl = s.to_wgsl_4d("floor4");
        assert!(wgsl.contains("fn floor4(_p: vec4<f32>) -> f32"));
        assert!(wgsl.contains("return 1e9"));
        assert!(
            !wgsl.contains("dot(p,"),
            "HalfSpace4D must not emit raw chart-coord dot product",
        );
    }

    /// `ConvexPolytope4D` emits a sentinel today; the real path
    /// goes through `Hyperslice4DNode`'s per-frame uniforms.
    #[test]
    fn polytope_4d_emit_is_sentinel() {
        let s = Shape::ConvexPolytope4D {
            vertices: vec![Vec4::ZERO; 5],
        };
        let wgsl = s.to_wgsl_4d("pent");
        assert!(wgsl.contains("fn pent(_p: vec4<f32>) -> f32"));
        assert!(wgsl.contains("return 1e9"));
    }

    /// 3D-only variants accidentally included in a 4D scene emit
    /// the sentinel, not a 3D-shaped function, keeps the
    /// `Primitive4` trait exhaustive without silently producing
    /// type-mismatched WGSL.
    #[test]
    fn three_d_variants_emit_sentinel_in_4d() {
        let s = Shape::sphere_at(glam::Vec3::ZERO, 1.0);
        let wgsl = s.to_wgsl_4d("oops");
        assert!(wgsl.contains("vec4<f32>"));
        assert!(wgsl.contains("return 1e9"));
    }
}
