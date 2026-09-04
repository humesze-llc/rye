//! - [`Shape::Box3`]: honest in E³; chart-coord in H³/S³.
//! - [`Shape::HalfSpace`]: chart-coord `dot(p, n) − offset` only in flat Spaces
//!   (gated by `Space::is_chart_flat`).

use glam::Vec3;
use loam_math::{Space, WgslSpace};
use loam_shape::Shape;

use crate::literal::wgsl_f32;
use crate::SENTINEL_DISTANCE;

/// [`Self::to_wgsl`] emits `fn {name}(p: vec3<f32>) -> f32`; [`Self::eval`] is
/// the CPU twin of that body.
pub trait Primitive {
    fn to_wgsl<S: WgslSpace>(&self, space: &S, name: &str) -> String;

    fn eval<S: Space<Point = Vec3, Vector = Vec3>>(&self, space: &S, p: Vec3) -> f32;
}

impl Primitive for Shape {
    fn to_wgsl<S: WgslSpace>(&self, space: &S, name: &str) -> String {
        match self {
            Shape::Sphere { center, radius } => format!(
                "fn {name}(p: vec3<f32>) -> f32 {{\n\
                 \treturn loam_distance(p, vec3<f32>({cx}, {cy}, {cz})) - ({r});\n\
                 }}\n",
                name = name,
                cx = wgsl_f32(center.x),
                cy = wgsl_f32(center.y),
                cz = wgsl_f32(center.z),
                r = wgsl_f32(*radius),
            ),
            Shape::Box3 { half_extents } => format!(
                "fn {name}(p: vec3<f32>) -> f32 {{\n\
                 \tlet b = vec3<f32>({hx}, {hy}, {hz});\n\
                 \tlet q = abs(p) - b;\n\
                 \treturn length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);\n\
                 }}\n",
                name = name,
                hx = wgsl_f32(half_extents.x),
                hy = wgsl_f32(half_extents.y),
                hz = wgsl_f32(half_extents.z),
            ),
            Shape::HalfSpace { normal, offset } if space.is_chart_flat() => format!(
                "fn {name}(p: vec3<f32>) -> f32 {{\n\
                 \treturn dot(p, vec3<f32>({nx}, {ny}, {nz})) - ({d});\n\
                 }}\n",
                name = name,
                nx = wgsl_f32(normal.x),
                ny = wgsl_f32(normal.y),
                nz = wgsl_f32(normal.z),
                d = wgsl_f32(*offset),
            ),
            Shape::HalfSpace { .. }
            | Shape::HalfSpace4D { .. }
            | Shape::Polygon2D { .. }
            | Shape::ConvexPolytope3D { .. }
            | Shape::ConvexPolytope4D { .. }
            | Shape::HyperSphere4D { .. } => {
                // `{:e}` is the WGSL float-literal spelling of the constant.
                format!("fn {name}(_p: vec3<f32>) -> f32 {{\n\treturn {SENTINEL_DISTANCE:e};\n}}\n",)
            }
        }
    }

    fn eval<S: Space<Point = Vec3, Vector = Vec3>>(&self, space: &S, p: Vec3) -> f32 {
        match self {
            Shape::Sphere { center, radius } => space.distance(p, *center) - *radius,

            // Quilez 2013, "distance functions", exact box SDF.
            Shape::Box3 { half_extents } => {
                let q = p.abs() - *half_extents;
                q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
            }

            Shape::HalfSpace { normal, offset } if space.is_chart_flat() => {
                p.dot(*normal) - *offset
            }

            Shape::HalfSpace { .. }
            | Shape::HalfSpace4D { .. }
            | Shape::Polygon2D { .. }
            | Shape::ConvexPolytope3D { .. }
            | Shape::ConvexPolytope4D { .. }
            | Shape::HyperSphere4D { .. } => SENTINEL_DISTANCE,
        }
    }
}
