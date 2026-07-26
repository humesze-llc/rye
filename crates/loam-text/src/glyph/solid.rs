//! The baked field turned into a solid: convex cross-section pieces, the
//! extruded 3D surface mesh, the slab-embedded 4D colliders, and the analytic
//! 4D distance.
//!
//! The cross-section is recovered by splitting every grid cell into two
//! triangles and clipping each against the half-space `d <= 0` (Sutherland &
//! Hodgman, 1974). Clipping a triangle by one half-space always yields a
//! convex polygon, which is what makes the same decomposition serve as both
//! render geometry and a set of convex colliders. The triangle split also
//! sidesteps the saddle ambiguity a marching-squares cell would have: the
//! interpolated point on a shared cell edge depends only on that edge's two
//! samples, so neighbouring cells agree and the union is watertight. The two
//! cells traverse that edge in opposite directions, so their interpolants agree
//! algebraically but can land an ulp apart; nothing downstream resolves
//! anywhere near that.

use glam::{Vec2, Vec3, Vec4};
use loam_shape::{LineMesh, NotVisualizable, PointMesh, Shape, TriangleMesh, Visualizable};

use super::field::DistanceField2D;
use super::BLANK_DISTANCE;

/// Ring vertices closer than this fraction of a cell are the same vertex.
/// A grid sample landing exactly on the zero level makes the clip emit the
/// corner and the interpolated point at the same position, and a zero-length
/// ring edge has no outward normal for the side wall.
const RING_MERGE_FRACTION: f32 = 1.0e-4;

/// Pieces thinner than this fraction of a cell's area are dropped. Degenerate
/// convex polytopes have no well-defined support direction, so they are worse
/// than useless as colliders.
const SLIVER_AREA_FRACTION: f32 = 1.0e-6;

/// Default wireframe weight for [`Visualizable::to_lines`]. `LineMesh::widths`
/// is public, so callers wanting another weight overwrite it in place rather
/// than threading a parameter through the bake.
const WIREFRAME_WIDTH_PX: f32 = 1.0;

/// Default marker radius for [`Visualizable::to_points`], same rationale as
/// [`WIREFRAME_WIDTH_PX`].
const POINT_MARKER_PX: f32 = 2.0;

/// One convex piece of a glyph's cross-section, counter-clockwise in world XY.
#[derive(Clone, Debug)]
pub(super) struct Piece {
    /// At least three vertices: [`clip_triangle`] drops anything degenerate,
    /// so cap fans and prism construction can index without a length check.
    ring: Vec<Vec2>,
    /// Index `k` such that `ring[k] -> ring[k + 1]` lies on the zero isoline,
    /// i.e. is part of the glyph's silhouette rather than shared with the
    /// neighbouring piece. At most one such edge exists per piece: clipping a
    /// convex polygon by a single half-space introduces exactly one new edge.
    cut: Option<usize>,
}

/// One letter of a laid-out word: a 2D glyph cross-section extruded along `z`
/// and embedded in a `w` slab, carrying its pen position within the word.
///
/// All geometry is in world units, already offset by the pen origin, so the
/// letters of a word share one frame.
#[derive(Clone, Debug)]
pub struct GlyphSolid {
    ch: char,
    pen_origin: Vec2,
    advance: f32,
    half_depth: f32,
    slab_center: f32,
    slab_half: f32,
    color: [f32; 4],
    field: Option<DistanceField2D>,
    pieces: Vec<Piece>,
}

impl GlyphSolid {
    pub(super) fn new(
        ch: char,
        pen_origin: Vec2,
        advance: f32,
        depth: f32,
        slab: (f32, f32),
        color: [f32; 4],
        field: Option<DistanceField2D>,
    ) -> Self {
        let pieces = field.as_ref().map(extract_pieces).unwrap_or_default();
        Self {
            ch,
            pen_origin,
            advance,
            half_depth: 0.5 * depth,
            slab_center: 0.5 * (slab.0 + slab.1),
            slab_half: 0.5 * (slab.1 - slab.0),
            color,
            field,
            pieces,
        }
    }

    /// The character this solid was baked from.
    pub fn ch(&self) -> char {
        self.ch
    }

    /// Baseline pen position of this letter within the laid-out word.
    pub fn pen_origin(&self) -> Vec2 {
        self.pen_origin
    }

    /// Distance from this letter's pen origin to the next letter's.
    pub fn advance(&self) -> f32 {
        self.advance
    }

    /// True for characters the font defines with no outline, i.e. whitespace.
    /// A blank carries an advance and no geometry.
    pub fn is_blank(&self) -> bool {
        self.field.is_none()
    }

    /// The baked cross-section field, absent for a blank.
    pub fn field(&self) -> Option<&DistanceField2D> {
        self.field.as_ref()
    }

    /// Convex cross-section pieces; one 4D collider is emitted per piece.
    pub fn piece_count(&self) -> usize {
        self.pieces.len()
    }

    /// Signed distance to the 2D cross-section, negative inside.
    pub fn distance_2d(&self, p: Vec2) -> f32 {
        self.field
            .as_ref()
            .map_or(BLANK_DISTANCE, |field| field.sample(p))
    }

    /// Signed distance to the slab-embedded 4D solid, negative inside.
    ///
    /// The solid is `cross_section x [-depth/2, depth/2] x [w_min, w_max]`, so
    /// the distance is the cross-section distance extended twice, once per
    /// interval axis. Fidelity is that of the baked cross-section: exact in
    /// `z` and `w`, grid-interpolated in `xy`.
    pub fn distance_4d(&self, p: Vec4) -> f32 {
        let cross_section = self.distance_2d(Vec2::new(p.x, p.y));
        let extruded = extend(cross_section, p.z, 0.0, self.half_depth);
        extend(extruded, p.w, self.slab_center, self.slab_half)
    }

    /// Convex 4D colliders covering the solid exactly, one per cross-section
    /// piece. Each is a `ring x z-interval x w-interval` prism, which is
    /// convex, so `loam-physics`' 4D GJK/EPA narrowphase consumes them
    /// directly.
    ///
    /// The count scales with the bake resolution squared; a caller wanting one
    /// rigid body per letter bakes the collider copy at a coarser resolution
    /// than the render copy.
    pub fn colliders_4d(&self) -> Vec<Shape> {
        let w_min = self.slab_center - self.slab_half;
        let w_max = self.slab_center + self.slab_half;
        self.pieces
            .iter()
            .map(|piece| {
                let mut vertices = Vec::with_capacity(piece.ring.len() * 4);
                for w in [w_min, w_max] {
                    for z in [-self.half_depth, self.half_depth] {
                        for q in &piece.ring {
                            vertices.push(Vec4::new(q.x, q.y, z, w));
                        }
                    }
                }
                Shape::ConvexPolytope4D { vertices }
            })
            .collect()
    }
}

/// Rendered geometry is the 3D cross-section of the 4D solid. Every `w` in the
/// slab gives the same cross-section (the solid is a product with the slab
/// interval), so one mesh serves the whole slab and the title-screen effect is
/// the slab sliding across the render's `w`.
impl Visualizable<3> for GlyphSolid {
    fn to_triangles(&self) -> Result<TriangleMesh<3>, NotVisualizable> {
        if self.pieces.is_empty() {
            return Err(NotVisualizable::Degenerate);
        }
        let mut mesh = TriangleMesh::default();
        for piece in &self.pieces {
            let n = piece.ring.len();
            let back = mesh.vertices.len() as u32;
            let front = back + n as u32;
            for z in [-self.half_depth, self.half_depth] {
                for q in &piece.ring {
                    mesh.vertices.push([q.x, q.y, z]);
                    mesh.colors.push(self.color);
                }
            }
            // Fans are valid because every piece is convex. The back cap is
            // wound in reverse so both caps face outwards.
            for k in 1..n as u32 - 1 {
                mesh.indices.push([front, front + k, front + k + 1]);
                mesh.indices.push([back, back + k + 1, back + k]);
            }
            if let Some(cut) = piece.cut {
                let a = cut as u32;
                let b = ((cut + 1) % n) as u32;
                mesh.indices.push([back + a, back + b, front + b]);
                mesh.indices.push([back + a, front + b, front + a]);
            }
        }
        Ok(mesh)
    }

    fn to_lines(&self) -> Result<LineMesh<3>, NotVisualizable> {
        if self.pieces.is_empty() {
            return Err(NotVisualizable::Degenerate);
        }
        let mut mesh = LineMesh::default();
        let mut push = |from: Vec3, to: Vec3| {
            mesh.segments.push((from.to_array(), to.to_array()));
            mesh.colors.push((self.color, self.color));
            mesh.widths.push(WIREFRAME_WIDTH_PX);
        };
        for piece in &self.pieces {
            let Some(cut) = piece.cut else { continue };
            let a = piece.ring[cut];
            let b = piece.ring[(cut + 1) % piece.ring.len()];
            let (a_back, a_front) = (at_z(a, -self.half_depth), at_z(a, self.half_depth));
            let (b_back, b_front) = (at_z(b, -self.half_depth), at_z(b, self.half_depth));
            push(a_back, b_back);
            push(a_front, b_front);
            push(a_back, a_front);
            push(b_back, b_front);
        }
        Ok(mesh)
    }

    fn to_points(&self) -> Result<PointMesh<3>, NotVisualizable> {
        if self.pieces.is_empty() {
            return Err(NotVisualizable::Degenerate);
        }
        let mut mesh = PointMesh::default();
        for piece in &self.pieces {
            let Some(cut) = piece.cut else { continue };
            let a = piece.ring[cut];
            for z in [-self.half_depth, self.half_depth] {
                mesh.positions.push(at_z(a, z).to_array());
                mesh.colors.push(self.color);
                mesh.sizes.push(POINT_MARKER_PX);
            }
        }
        Ok(mesh)
    }
}

fn at_z(p: Vec2, z: f32) -> Vec3 {
    Vec3::new(p.x, p.y, z)
}

/// Exact distance to `S x [center - half, center + half]` given the exact
/// distance `d` to `S` in the orthogonal complement. Quilez, "distance
/// functions" (2019), `opExtrusion`: the exterior term is the length of the
/// positive part and the interior term is the larger (least negative)
/// component.
fn extend(d: f32, x: f32, center: f32, half: f32) -> f32 {
    let axial = (x - center).abs() - half;
    Vec2::new(d, axial).max(Vec2::ZERO).length() + d.max(axial).min(0.0)
}

/// Split every cell into two triangles and clip each against `d <= 0`.
fn extract_pieces(field: &DistanceField2D) -> Vec<Piece> {
    let (cells_x, cells_y) = field.cell_counts();
    let cell = field.cell_size();
    let mut pieces = Vec::new();

    for j in 0..cells_y {
        for i in 0..cells_x {
            let indices = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)];
            let corners: [Corner; 4] = std::array::from_fn(|k| {
                let (ci, cj) = indices[k];
                Corner {
                    position: field.corner(ci, cj),
                    distance: field.at(ci, cj),
                }
            });
            // Alternate the split diagonal by cell parity so the clipped edges
            // carry no global directional grain. Both variants stay
            // counter-clockwise.
            let triangles: [[usize; 3]; 2] = if (i + j) % 2 == 0 {
                [[0, 1, 2], [0, 2, 3]]
            } else {
                [[0, 1, 3], [1, 2, 3]]
            };
            for tri in triangles {
                let corners = [
                    corners[tri[0]].clone(),
                    corners[tri[1]].clone(),
                    corners[tri[2]].clone(),
                ];
                if let Some(piece) = clip_triangle(&corners, cell) {
                    pieces.push(piece);
                }
            }
        }
    }
    pieces
}

#[derive(Clone, Debug)]
struct Corner {
    position: Vec2,
    distance: f32,
}

/// Clip `tri` against the half-space `d <= 0`, tracking which output edge lies
/// on the zero isoline.
fn clip_triangle(tri: &[Corner; 3], cell: f32) -> Option<Piece> {
    let mut ring: Vec<Vec2> = Vec::with_capacity(4);
    let mut on_isoline: Vec<bool> = Vec::with_capacity(4);

    for k in 0..3 {
        let a = &tri[k];
        let b = &tri[(k + 1) % 3];
        if a.distance <= 0.0 {
            ring.push(a.position);
            on_isoline.push(a.distance == 0.0);
        }
        if (a.distance <= 0.0) != (b.distance <= 0.0) {
            // Signs differ, so the denominator is nonzero by construction.
            let t = a.distance / (a.distance - b.distance);
            ring.push(a.position + (b.position - a.position) * t);
            on_isoline.push(true);
        }
    }
    if ring.len() < 3 {
        return None;
    }

    merge_coincident(&mut ring, &mut on_isoline, cell);
    if ring.len() < 3 || double_signed_area(&ring) <= SLIVER_AREA_FRACTION * cell * cell {
        return None;
    }

    let n = ring.len();
    let cut = (0..n).find(|&k| on_isoline[k] && on_isoline[(k + 1) % n]);
    Some(Piece { ring, cut })
}

fn merge_coincident(ring: &mut Vec<Vec2>, on_isoline: &mut Vec<bool>, cell: f32) {
    let merge2 = (cell * RING_MERGE_FRACTION).powi(2);
    let mut kept_ring: Vec<Vec2> = Vec::with_capacity(ring.len());
    let mut kept_flags: Vec<bool> = Vec::with_capacity(ring.len());
    for (p, flag) in ring.iter().zip(on_isoline.iter()) {
        match kept_ring.last() {
            Some(last) if last.distance_squared(*p) <= merge2 => {
                let end = kept_flags.len() - 1;
                kept_flags[end] |= *flag;
            }
            _ => {
                kept_ring.push(*p);
                kept_flags.push(*flag);
            }
        }
    }
    while kept_ring.len() >= 2
        && kept_ring[0].distance_squared(kept_ring[kept_ring.len() - 1]) <= merge2
    {
        kept_ring.pop();
        let tail = kept_flags.pop().unwrap_or(false);
        kept_flags[0] |= tail;
    }
    *ring = kept_ring;
    *on_isoline = kept_flags;
}

fn double_signed_area(ring: &[Vec2]) -> f32 {
    let n = ring.len();
    let mut sum = 0.0;
    for i in 0..n {
        let a = ring[i];
        let b = ring[(i + 1) % n];
        sum += a.x * b.y - b.x * a.y;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::super::outline::Contour;
    use super::*;

    /// `cells` counts grid cells across the square, so the fixtures read the
    /// way the old per-glyph resolution did.
    fn square_field(half: f32, cells: u32) -> DistanceField2D {
        let contour = Contour {
            points: vec![
                Vec2::new(-half, -half),
                Vec2::new(half, -half),
                Vec2::new(half, half),
                Vec2::new(-half, half),
            ],
        };
        DistanceField2D::bake(&[contour], 2.0 * half / cells as f32).expect("bake")
    }

    fn square_solid(half: f32, resolution: u32, depth: f32, slab: (f32, f32)) -> GlyphSolid {
        GlyphSolid::new(
            'X',
            Vec2::ZERO,
            2.0 * half,
            depth,
            slab,
            [1.0; 4],
            Some(square_field(half, resolution)),
        )
    }

    /// Every piece is a convex counter-clockwise ring. Convexity is the
    /// contract the 4D narrowphase relies on; a reflex vertex would make GJK
    /// report support points inside the hull.
    #[test]
    fn every_piece_is_convex_and_counter_clockwise() {
        let solid = square_solid(1.0, 17, 0.5, (-0.25, 0.25));
        assert!(solid.piece_count() > 0);
        for piece in &solid.pieces {
            let n = piece.ring.len();
            assert!((3..=4).contains(&n), "ring has {n} vertices");
            assert!(double_signed_area(&piece.ring) > 0.0);
            for k in 0..n {
                let a = piece.ring[k];
                let b = piece.ring[(k + 1) % n];
                let c = piece.ring[(k + 2) % n];
                let cross = (b - a).perp_dot(c - b);
                assert!(cross >= -1.0e-6, "reflex vertex at {k}: cross = {cross}");
            }
        }
    }

    /// The pieces tile the cross-section: their total area recovers the true
    /// area up to the grid's boundary error, which shrinks with resolution.
    #[test]
    fn piece_areas_sum_to_the_cross_section_area() {
        for resolution in [16, 32, 64] {
            let solid = square_solid(1.0, resolution, 0.5, (0.0, 1.0));
            let area: f32 = solid
                .pieces
                .iter()
                .map(|p| 0.5 * double_signed_area(&p.ring))
                .sum();
            let cell = solid.field().unwrap().cell_size();
            // Boundary cells are the only source of error: at most one cell of
            // area along a perimeter of 8.
            assert!(
                (area - 4.0).abs() < 8.0 * cell,
                "resolution {resolution}: area {area} off by more than {} ",
                8.0 * cell
            );
        }
    }

    /// The surface mesh is closed and outward-wound: the divergence-theorem
    /// volume of the triangle soup equals cross-section area times depth. This
    /// pins cap winding, side-wall winding, and that a side wall is emitted for
    /// exactly the silhouette edges.
    #[test]
    fn extruded_mesh_volume_matches_area_times_depth() {
        let depth = 0.5;
        let solid = square_solid(1.0, 32, depth, (-1.0, 1.0));
        let mesh = solid.to_triangles().expect("mesh");
        let mut volume = 0.0;
        for [i0, i1, i2] in &mesh.indices {
            let v0 = Vec3::from_array(mesh.vertices[*i0 as usize]);
            let v1 = Vec3::from_array(mesh.vertices[*i1 as usize]);
            let v2 = Vec3::from_array(mesh.vertices[*i2 as usize]);
            volume += v0.dot(v1.cross(v2)) / 6.0;
        }
        let cell = solid.field().unwrap().cell_size();
        let expected = 4.0 * depth;
        assert!(
            (volume - expected).abs() < 8.0 * cell * depth,
            "volume {volume} vs expected {expected}"
        );
    }

    /// Reversing the depth would flip the winding and negate the volume, so a
    /// positive volume is the assertion that the mesh faces outwards rather
    /// than inwards.
    #[test]
    fn extruded_mesh_volume_is_positive() {
        let solid = square_solid(1.0, 24, 0.3, (0.0, 0.5));
        let mesh = solid.to_triangles().expect("mesh");
        let volume: f32 = mesh
            .indices
            .iter()
            .map(|[i0, i1, i2]| {
                let v0 = Vec3::from_array(mesh.vertices[*i0 as usize]);
                let v1 = Vec3::from_array(mesh.vertices[*i1 as usize]);
                let v2 = Vec3::from_array(mesh.vertices[*i2 as usize]);
                v0.dot(v1.cross(v2)) / 6.0
            })
            .sum();
        assert!(volume > 0.0, "volume {volume} is not outward-wound");
    }

    /// Mesh invariants the rasterizer upload path assumes: one colour per
    /// vertex and every index in range.
    #[test]
    fn mesh_buffers_are_consistent() {
        let solid = square_solid(1.0, 20, 0.4, (0.0, 1.0));
        let mesh = solid.to_triangles().expect("mesh");
        assert_eq!(mesh.colors.len(), mesh.vertices.len());
        for [i0, i1, i2] in &mesh.indices {
            for index in [i0, i1, i2] {
                assert!((*index as usize) < mesh.vertices.len());
            }
        }
        let lines = solid.to_lines().expect("lines");
        assert_eq!(lines.colors.len(), lines.segments.len());
        assert_eq!(lines.widths.len(), lines.segments.len());
        let points = solid.to_points().expect("points");
        assert_eq!(points.colors.len(), points.positions.len());
        assert_eq!(points.sizes.len(), points.positions.len());
    }

    /// One collider per piece, each spanning the full slab and depth, and each
    /// a `ring x z x w` prism, so the vertex count is four per ring vertex.
    #[test]
    fn colliders_are_prisms_spanning_depth_and_slab() {
        let slab = (-0.75, 0.25);
        let depth = 0.6;
        let solid = square_solid(1.0, 12, depth, slab);
        let colliders = solid.colliders_4d();
        assert_eq!(colliders.len(), solid.piece_count());
        for (collider, piece) in colliders.iter().zip(&solid.pieces) {
            let Shape::ConvexPolytope4D { vertices } = collider else {
                panic!("expected ConvexPolytope4D, got {:?}", collider.kind());
            };
            assert_eq!(vertices.len(), piece.ring.len() * 4);
            let z_min = vertices.iter().fold(f32::INFINITY, |m, v| m.min(v.z));
            let z_max = vertices.iter().fold(f32::NEG_INFINITY, |m, v| m.max(v.z));
            let w_min = vertices.iter().fold(f32::INFINITY, |m, v| m.min(v.w));
            let w_max = vertices.iter().fold(f32::NEG_INFINITY, |m, v| m.max(v.w));
            assert!((z_min + 0.5 * depth).abs() < 1e-6);
            assert!((z_max - 0.5 * depth).abs() < 1e-6);
            assert!((w_min - slab.0).abs() < 1e-6);
            assert!((w_max - slab.1).abs() < 1e-6);
        }
    }

    /// Every collider vertex is inside the solid: the convex pieces cover the
    /// cross-section without spilling past the silhouette by more than the
    /// grid's interpolation error.
    #[test]
    fn collider_vertices_lie_on_or_inside_the_solid() {
        let solid = square_solid(1.0, 24, 0.5, (0.0, 1.0));
        let cell = solid.field().unwrap().cell_size();
        for collider in solid.colliders_4d() {
            let Shape::ConvexPolytope4D { vertices } = collider else {
                unreachable!()
            };
            for v in vertices {
                assert!(
                    solid.distance_4d(v) <= cell,
                    "collider vertex {v} sits {} outside the solid",
                    solid.distance_4d(v)
                );
            }
        }
    }

    /// The 4D distance is the cross-section distance extended twice: inside the
    /// slab and depth it is the cross-section distance, and past either face it
    /// grows by the axial offset.
    #[test]
    fn distance_4d_extends_the_cross_section_on_both_interval_axes() {
        let depth = 0.5;
        let slab = (-1.0, 1.0);
        let solid = square_solid(1.0, 32, depth, slab);

        // Deep inside: the nearest face is whichever of the three is closest.
        let centre = solid.distance_4d(Vec4::ZERO);
        assert!((centre + 0.25).abs() < 0.05, "centre distance {centre}");

        // Straight out along +z past the cap: exact, since z is not baked.
        let above = solid.distance_4d(Vec4::new(0.0, 0.0, 0.5 * depth + 2.0, 0.0));
        assert!((above - 2.0).abs() < 1e-5, "above {above}");

        // Straight out along +w past the slab: exact for the same reason.
        let beyond = solid.distance_4d(Vec4::new(0.0, 0.0, 0.0, slab.1 + 3.0));
        assert!((beyond - 3.0).abs() < 1e-5, "beyond {beyond}");

        // Diagonally out of both a cap and the slab: Euclidean, not Chebyshev.
        let corner = solid.distance_4d(Vec4::new(0.0, 0.0, 0.5 * depth + 3.0, slab.1 + 4.0));
        assert!((corner - 5.0).abs() < 1e-5, "corner {corner}");
    }

    /// A blank glyph has no geometry at all, and says so through the trait's
    /// own vocabulary instead of returning empty buffers.
    #[test]
    fn blank_glyph_reports_degenerate_and_emits_no_colliders() {
        let blank = GlyphSolid::new(' ', Vec2::ZERO, 0.25, 0.5, (0.0, 1.0), [1.0; 4], None);
        assert!(blank.is_blank());
        assert_eq!(blank.piece_count(), 0);
        assert!(blank.colliders_4d().is_empty());
        assert_eq!(
            Visualizable::<3>::to_triangles(&blank).unwrap_err(),
            NotVisualizable::Degenerate
        );
        assert_eq!(blank.distance_2d(Vec2::ZERO), BLANK_DISTANCE);
    }

    /// A clip that keeps the whole triangle has no silhouette edge; a clip that
    /// cuts it has exactly one, and its endpoints sit on the zero level.
    #[test]
    fn cut_edge_exists_exactly_when_the_triangle_is_clipped() {
        let inside = [
            Corner {
                position: Vec2::ZERO,
                distance: -1.0,
            },
            Corner {
                position: Vec2::X,
                distance: -1.0,
            },
            Corner {
                position: Vec2::Y,
                distance: -1.0,
            },
        ];
        let whole = clip_triangle(&inside, 1.0).expect("whole triangle");
        assert_eq!(whole.ring.len(), 3);
        assert!(whole.cut.is_none());

        let straddling = [
            Corner {
                position: Vec2::ZERO,
                distance: -1.0,
            },
            Corner {
                position: Vec2::X,
                distance: 1.0,
            },
            Corner {
                position: Vec2::Y,
                distance: 1.0,
            },
        ];
        let clipped = clip_triangle(&straddling, 1.0).expect("clipped triangle");
        assert_eq!(clipped.ring.len(), 3);
        let cut = clipped.cut.expect("cut edge");
        let a = clipped.ring[cut];
        let b = clipped.ring[(cut + 1) % 3];
        // Both endpoints are the midpoints of the two straddling edges.
        assert!(
            a.distance(Vec2::new(0.5, 0.0))
                .min(a.distance(Vec2::new(0.0, 0.5)))
                < 1e-6
        );
        assert!(
            b.distance(Vec2::new(0.5, 0.0))
                .min(b.distance(Vec2::new(0.0, 0.5)))
                < 1e-6
        );
    }

    /// A fully outside triangle contributes nothing, and a triangle touching
    /// the zero level at a single corner is a sliver, not a piece.
    #[test]
    fn outside_and_sliver_triangles_produce_no_piece() {
        let outside = [
            Corner {
                position: Vec2::ZERO,
                distance: 1.0,
            },
            Corner {
                position: Vec2::X,
                distance: 1.0,
            },
            Corner {
                position: Vec2::Y,
                distance: 1.0,
            },
        ];
        assert!(clip_triangle(&outside, 1.0).is_none());

        let grazing = [
            Corner {
                position: Vec2::ZERO,
                distance: 0.0,
            },
            Corner {
                position: Vec2::X,
                distance: 1.0,
            },
            Corner {
                position: Vec2::Y,
                distance: 1.0,
            },
        ];
        assert!(clip_triangle(&grazing, 1.0).is_none());
    }

    /// Piece extraction is a pure function of the field: identical inputs give
    /// identical rings in identical order.
    #[test]
    fn piece_extraction_is_reproducible() {
        let field = square_field(1.0, 21);
        let a = extract_pieces(&field);
        let b = extract_pieces(&field);
        assert_eq!(a.len(), b.len());
        for (pa, pb) in a.iter().zip(&b) {
            assert_eq!(pa.ring, pb.ring);
            assert_eq!(pa.cut, pb.cut);
        }
    }
}
