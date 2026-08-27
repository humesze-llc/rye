use glam::Vec4;
use loam_shape::Shape;

use crate::literal::wgsl_f32;
use crate::SENTINEL_DISTANCE;

/// [`Self::eval_4d`] is the CPU twin of [`Self::to_wgsl_4d`]'s emitted body,
/// exact up to `f32` rounding. No `space: &S` parameter: ℝ⁴ is the only 4D
/// Space and it is flat, so chart-coord SDFs are correct.
pub trait Primitive4 {
    fn to_wgsl_4d(&self, name: &str) -> String;

    fn eval_4d(&self, p: Vec4) -> f32;
}

impl Primitive4 for Shape {
    fn to_wgsl_4d(&self, name: &str) -> String {
        match self {
            Shape::HyperSphere4D { center, radius } => format!(
                "fn {name}(p: vec4<f32>) -> f32 {{\n\
                \treturn length(p - vec4<f32>({cx}, {cy}, {cz}, {cw})) - ({radius});\n\
                }}\n",
                name = name,
                cx = wgsl_f32(center.x),
                cy = wgsl_f32(center.y),
                cz = wgsl_f32(center.z),
                cw = wgsl_f32(center.w),
                radius = wgsl_f32(*radius),
            ),

            // Sentinel until the real path lands: face hyperplanes are pose-
            // dependent per frame, so `Hyperslice4DNode` ships them via a uniform
            // buffer rather than baking constants here.
            Shape::ConvexPolytope4D { .. } => format!(
                "fn {name}(_p: vec4<f32>) -> f32 {{\n\
                \t// ConvexPolytope4D: half-space emit lives in the\n\
                \t// render node's per-frame uniform path, not here.\n\
                \t// See Hyperslice4DNode.\n\
                \treturn {SENTINEL_DISTANCE:e};\n\
                }}\n",
            ),

            Shape::HalfSpace4D { normal, offset } => format!(
                "fn {name}(p: vec4<f32>) -> f32 {{\n\
                \treturn dot(p, vec4<f32>({nx}, {ny}, {nz}, {nw})) - ({offset});\n\
                }}\n",
                name = name,
                nx = wgsl_f32(normal.x),
                ny = wgsl_f32(normal.y),
                nz = wgsl_f32(normal.z),
                nw = wgsl_f32(normal.w),
                offset = wgsl_f32(*offset),
            ),

            Shape::Sphere { .. }
            | Shape::HalfSpace { .. }
            | Shape::Box3 { .. }
            | Shape::Polygon2D { .. }
            | Shape::ConvexPolytope3D { .. } => format!(
                "fn {name}(_p: vec4<f32>) -> f32 {{\n\
                \treturn {SENTINEL_DISTANCE:e};\n\
                }}\n",
            ),
        }
    }

    fn eval_4d(&self, p: Vec4) -> f32 {
        match self {
            Shape::HyperSphere4D { center, radius } => (p - *center).length() - *radius,

            Shape::ConvexPolytope4D { .. } => SENTINEL_DISTANCE,

            Shape::HalfSpace4D { normal, offset } => p.dot(*normal) - *offset,

            Shape::Sphere { .. }
            | Shape::HalfSpace { .. }
            | Shape::Box3 { .. }
            | Shape::Polygon2D { .. }
            | Shape::ConvexPolytope3D { .. } => SENTINEL_DISTANCE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec4;

    #[test]
    fn hypersphere_4d_emit_has_expected_shape() {
        let s = Shape::HyperSphere4D {
            center: Vec4::new(1.0, 2.0, 3.0, 4.0),
            radius: 0.5,
        };
        let wgsl = s.to_wgsl_4d("ball");
        assert!(wgsl.contains("fn ball(p: vec4<f32>) -> f32"));
        assert!(wgsl.contains("length(p - vec4<f32>(1.0, 2.0, 3.0, 4.0))"));
        assert!(wgsl.contains("- (0.5)"));
    }

    #[test]
    fn halfspace_4d_emits_dot_in_flat_chart() {
        let s = Shape::HalfSpace4D {
            normal: Vec4::new(0.0, 1.0, 0.0, 0.0),
            offset: 0.0,
        };
        let wgsl = s.to_wgsl_4d("floor4");
        assert!(wgsl.contains("fn floor4(p: vec4<f32>) -> f32"));
        assert!(wgsl.contains("dot(p, vec4<f32>(0.0, 1.0, 0.0, 0.0))"));
        assert!(wgsl.contains("- (0.0)"));
    }

    #[test]
    fn polytope_4d_emit_is_sentinel() {
        let s = Shape::ConvexPolytope4D {
            vertices: vec![Vec4::ZERO; 5],
        };
        let wgsl = s.to_wgsl_4d("pent");
        assert!(wgsl.contains("fn pent(_p: vec4<f32>) -> f32"));
        assert!(wgsl.contains("return 1e9"));
    }

    #[test]
    fn three_d_variants_emit_sentinel_in_4d() {
        let s = Shape::sphere_at(glam::Vec3::ZERO, 1.0);
        let wgsl = s.to_wgsl_4d("oops");
        assert!(wgsl.contains("vec4<f32>"));
        assert!(wgsl.contains("return 1e9"));
    }
}
