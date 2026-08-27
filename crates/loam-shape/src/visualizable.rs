//! Colors are RGBA linear `[f32; 4]`: linear because the fragment shader
//! interpolates in linear space (sRGB conversion is at the output attachment),
//! and alpha because it carries AA coverage at silhouettes.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotVisualizable {
    Unbounded,

    WrongDimension,

    /// Zero radius, empty or collinear vertices. Nothing to draw, not a bug.
    Degenerate,
}

pub trait Visualizable<const N: usize> {
    fn to_lines(&self) -> Result<LineMesh<N>, NotVisualizable>;

    fn to_triangles(&self) -> Result<TriangleMesh<N>, NotVisualizable>;

    fn to_points(&self) -> Result<PointMesh<N>, NotVisualizable>;
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(
    serialize = "[f32; N]: Serialize",
    deserialize = "[f32; N]: Deserialize<'de>"
))]
pub struct LineMesh<const N: usize> {
    pub segments: Vec<([f32; N], [f32; N])>,
    /// `(start_color, end_color)`; `colors.len() == segments.len()`.
    pub colors: Vec<([f32; 4], [f32; 4])>,
    /// Pixels. `widths.len() == segments.len()`.
    pub widths: Vec<f32>,
}

/// No normals: lighting an R⁴ triangle has no standard convention.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(
    serialize = "[f32; N]: Serialize",
    deserialize = "[f32; N]: Deserialize<'de>"
))]
pub struct TriangleMesh<const N: usize> {
    pub vertices: Vec<[f32; N]>,
    /// Counter-clockwise winding, looking down the normal.
    pub indices: Vec<[u32; 3]>,
    /// `colors.len() == vertices.len()`.
    pub colors: Vec<[f32; 4]>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(
    serialize = "[f32; N]: Serialize",
    deserialize = "[f32; N]: Deserialize<'de>"
))]
pub struct PointMesh<const N: usize> {
    pub positions: Vec<[f32; N]>,
    /// `colors.len() == positions.len()`.
    pub colors: Vec<[f32; 4]>,
    /// Screen-space radius in pixels.
    pub sizes: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_mesh_3d_ron_round_trip() {
        let original: LineMesh<3> = LineMesh {
            segments: vec![
                ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
                ([0.5, 1.0, 0.0], [0.5, 1.0, 1.0]),
            ],
            colors: vec![
                ([1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]),
                ([0.0, 0.0, 1.0, 0.7], [0.0, 0.0, 1.0, 0.7]),
            ],
            widths: vec![1.5, 2.0],
        };
        let s = ron::ser::to_string(&original).expect("serialize");
        let parsed: LineMesh<3> = ron::de::from_str(&s).expect("deserialize");
        assert_eq!(parsed.segments, original.segments);
        assert_eq!(parsed.colors, original.colors);
        assert_eq!(parsed.widths, original.widths);
    }

    #[test]
    fn triangle_mesh_3d_ron_round_trip() {
        let original: TriangleMesh<3> = TriangleMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            indices: vec![[0, 1, 2], [1, 3, 2]],
            colors: vec![
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
            ],
        };
        let s = ron::ser::to_string(&original).expect("serialize");
        let parsed: TriangleMesh<3> = ron::de::from_str(&s).expect("deserialize");
        assert_eq!(parsed.vertices, original.vertices);
        assert_eq!(parsed.indices, original.indices);
        assert_eq!(parsed.colors, original.colors);
    }

    #[test]
    fn point_mesh_4d_ron_round_trip() {
        let original: PointMesh<4> = PointMesh {
            positions: vec![[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            colors: vec![[1.0, 1.0, 1.0, 1.0], [1.0, 0.5, 0.5, 1.0]],
            sizes: vec![3.0, 5.0],
        };
        let s = ron::ser::to_string(&original).expect("serialize");
        let parsed: PointMesh<4> = ron::de::from_str(&s).expect("deserialize");
        assert_eq!(parsed.positions, original.positions);
        assert_eq!(parsed.sizes, original.sizes);
    }
}
