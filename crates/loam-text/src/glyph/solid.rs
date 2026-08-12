//! The baked field turned into a solid: convex cross-section pieces, the
//! extruded 3D surface mesh, the slab-embedded 4D colliders, and the analytic
//! 4D distance.
//!
//! The two consumers want opposite decompositions, so they get different ones.
//!
//! **Render.** The cross-section is recovered by splitting every grid cell into
//! two triangles and clipping each against the half-space `d <= 0` (Sutherland
//! & Hodgman, 1974). The triangle split sidesteps the saddle ambiguity a
//! marching-squares cell would have: the interpolated point on a shared cell
//! edge depends only on that edge's two samples, so neighbouring cells agree
//! and the union is watertight. The two cells traverse that edge in opposite
//! directions, so their interpolants agree algebraically but can land an ulp
//! apart; nothing downstream resolves anywhere near that. The piece count is
//! `Theta(res^2)`, because the letter's interior contributes as many pieces as
//! its boundary does, and that is the right trade for geometry a rasterizer
//! consumes in one draw.
//!
//! **Collision.** A solver pays per body, so the same count is ruinous there.
//! The colliders come instead from an [`Isovolume`] cover of the cross-section
//! at its own pitch, extruded through the depth and the slab. The cover's count
//! tracks silhouette complexity rather than interior area, and its error is a
//! bounded outward margin rather than a lost stem, which is what lets the
//! collider pitch be coarsened independently of legibility.
//!
//! **Collision, moving.** The cover is a set of bodies, not a body, and
//! `loam-physics` gives a rigid body one collider with no local offset, so a
//! letter that moves gets a third decomposition: [`GlyphSolid::rigid_hull_4d`],
//! one convex prism per letter over the hull of the cover. Convexity is the
//! price of rigidity and it is paid where it shows least, since a falling
//! letter is judged on its silhouette and its rest pose rather than on whether
//! something can pass through its counter.

use glam::{Vec2, Vec3, Vec4};
use loam_shape::{
    Isovolume, LineMesh, NotVisualizable, PointMesh, Shape, TriangleMesh, Visualizable,
};

use super::field::DistanceField2D;
use super::hull::{centroid, convex_hull, reduce_sides, MAX_HULL_SIDES};
use super::{GlyphParams, BLANK_DISTANCE};

/// [`DistanceField2D::sample`] is 1-Lipschitz per axis, hence only
/// `sqrt(2)`-Lipschitz in L2, while [`Isovolume::extract`] requires a true
/// 1-Lipschitz field and silently loses its enclosure guarantee without one.
/// Scaling by `1/sqrt(2)` restores the precondition and leaves the zero level
/// where it was, at the price of an occupancy test that reaches a full cell
/// instead of `cell/sqrt(2)`.
const LIPSCHITZ_SCALE: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Collider cells of empty space kept around the render grid.
/// [`Isovolume::clipped`] flags a marked cell touching the domain boundary,
/// and the field's own padding is counted in render cells, which are the finer
/// of the two whenever the collider pitch is coarsened.
const COVER_PADDING_CELLS: f32 = 2.0;

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
    collider_cell: f32,
    field: Option<DistanceField2D>,
    pieces: Vec<Piece>,
    cover: Option<Isovolume<2>>,
    /// Counter-clockwise convex ring the dynamic body collides with; empty for
    /// a blank.
    hull: Vec<Vec2>,
}

impl GlyphSolid {
    pub(super) fn new(
        ch: char,
        pen_origin: Vec2,
        advance: f32,
        params: &GlyphParams,
        field: Option<DistanceField2D>,
    ) -> Self {
        let collider_cell = params.em_size / params.collider_resolution as f32;
        let pieces = field.as_ref().map(extract_pieces).unwrap_or_default();
        let cover = field
            .as_ref()
            .map(|field| extract_cover(field, collider_cell));
        let hull = cover.as_ref().map(rigid_ring).unwrap_or_default();
        Self {
            ch,
            pen_origin,
            advance,
            half_depth: 0.5 * params.depth,
            slab_center: 0.5 * (params.slab.0 + params.slab.1),
            slab_half: 0.5 * (params.slab.1 - params.slab.0),
            color: params.color,
            collider_cell,
            field,
            pieces,
            cover,
            hull,
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

    /// Convex cross-section pieces backing the render mesh. Render-only: the
    /// colliders come from [`Self::collider_cover`], which is a different
    /// decomposition at a different pitch.
    pub fn piece_count(&self) -> usize {
        self.pieces.len()
    }

    /// The collider cover of the cross-section, absent for a blank.
    ///
    /// Extraction scaled the field by 1/sqrt(2), the Lipschitz correction a
    /// 2D outline field needs before a conservative cover is sound, so
    /// [`Isovolume::enclosure_margin`] under-reports this cover by that same
    /// factor. [`Self::collider_margin`] is the bound that holds, and is the
    /// one to quote.
    pub fn collider_cover(&self) -> Option<&Isovolume<2>> {
        self.cover.as_ref()
    }

    /// Number of 4D colliders this letter emits, i.e. the number of bodies a
    /// solver pays for.
    pub fn collider_count(&self) -> usize {
        self.cover.as_ref().map_or(0, Isovolume::piece_count)
    }

    /// Upper bound on how far the collider cover reaches past the letter's
    /// baked surface: two collider cells.
    ///
    /// A cell is marked when its centre samples `LIPSCHITZ_SCALE * d <= m`
    /// for the half-diagonal `m = cell/sqrt(2)`, i.e. when `d <= cell`. From
    /// that centre to any point of the cell is at most one cell in L1, and
    /// [`DistanceField2D::sample`] is 1-Lipschitz in L1, so the far corner has
    /// `d <= 2 * cell`. Halving the collider resolution doubles this.
    pub fn collider_margin(&self) -> f32 {
        2.0 * self.collider_cell
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

    /// Convex 4D colliders enclosing the letter, as `(centre, hull)` pairs.
    /// Pose is extrinsic per the [`Shape`] contract, so `centre` is the body
    /// position and the hull is a 16-vertex box about the origin.
    ///
    /// Each hull is one box of [`Self::collider_cover`] extruded through the
    /// depth and the slab. The union encloses the letter and overshoots its ink
    /// by at most [`Self::collider_margin`]; counters and notches stay open,
    /// because the occupancy test never marks a cell they cover.
    ///
    /// # Static bodies only
    ///
    /// Spawn these fixed, and reach for [`Self::rigid_hull_4d`] when the letter
    /// has to move. `loam-physics` gives a rigid body exactly one collider and
    /// no per-collider local offset, so a letter made dynamic is as many
    /// independent bodies as it has boxes; the cover's boxes overlap by
    /// construction, so those bodies start interpenetrating and drive each
    /// other apart on the first step rather than merely drifting. Lifting that
    /// is a compound shape carrying per-part offsets, in `loam-physics` and
    /// not here.
    pub fn colliders_4d(&self) -> Vec<(Vec4, Shape)> {
        let Some(cover) = &self.cover else {
            return Vec::new();
        };
        (0..cover.piece_count())
            .map(|index| {
                let (lo, hi) = cover.piece_bounds(index);
                let centre = Vec4::new(
                    0.5 * (lo[0] + hi[0]),
                    0.5 * (lo[1] + hi[1]),
                    0.0,
                    self.slab_center,
                );
                let half = Vec4::new(
                    0.5 * (hi[0] - lo[0]),
                    0.5 * (hi[1] - lo[1]),
                    self.half_depth,
                    self.slab_half,
                );
                let mut vertices = Vec::with_capacity(16);
                for sw in [-1.0f32, 1.0] {
                    for sz in [-1.0f32, 1.0] {
                        for sy in [-1.0f32, 1.0] {
                            for sx in [-1.0f32, 1.0] {
                                vertices.push(half * Vec4::new(sx, sy, sz, sw));
                            }
                        }
                    }
                }
                (centre, Shape::ConvexPolytope4D { vertices })
            })
            .collect()
    }

    /// Sides of the convex ring [`Self::rigid_hull_4d`] is built on; zero for a
    /// blank. The 4D hull carries four vertices per side.
    pub fn rigid_hull_sides(&self) -> usize {
        self.hull.len()
    }

    /// The single convex 4D collider a *dynamic* letter gets, as
    /// `(centre, hull)`, or `None` for a blank.
    ///
    /// One body per letter, against [`Self::collider_count`] for the static
    /// path. The hull is the prism `ring x [-depth/2, depth/2] x slab`, where
    /// `ring` is the convex hull of [`Self::collider_cover`] simplified to at
    /// most eight sides, so the vertex count is at most 32 and fits the 4D
    /// narrowphase's fixed polytope buffer.
    ///
    /// `centre` is the prism's centre of mass, i.e. the ring's area centroid in
    /// `xy` and the mid-planes of the depth and the slab; pose is extrinsic per
    /// the [`Shape`] contract, so the returned vertices are about the origin
    /// and a body built from the pair spins about the right point.
    ///
    /// # What convexity costs
    ///
    /// Counters and notches fill: a body dropped down the middle of a moving
    /// `O` lands on it. That is the whole difference from
    /// [`Self::colliders_4d`], which keeps them open and cannot move. Both
    /// enclose the letter, so neither lets anything through the ink.
    pub fn rigid_hull_4d(&self) -> Option<(Vec4, Shape)> {
        if self.hull.is_empty() {
            return None;
        }
        let centre_2d = centroid(&self.hull);
        let centre = Vec4::new(centre_2d.x, centre_2d.y, 0.0, self.slab_center);
        let mut vertices = Vec::with_capacity(4 * self.hull.len());
        for sw in [-1.0f32, 1.0] {
            for sz in [-1.0f32, 1.0] {
                for p in &self.hull {
                    let offset = *p - centre_2d;
                    vertices.push(Vec4::new(
                        offset.x,
                        offset.y,
                        sz * self.half_depth,
                        sw * self.slab_half,
                    ));
                }
            }
        }
        Some((centre, Shape::ConvexPolytope4D { vertices }))
    }
}

/// Convex ring enclosing a cover, at most [`MAX_HULL_SIDES`] long. Hulling the
/// box corners rather than the marked cells costs four points per box and needs
/// no access to the occupancy grid.
fn rigid_ring(cover: &Isovolume<2>) -> Vec<Vec2> {
    let mut corners = Vec::with_capacity(4 * cover.piece_count());
    for index in 0..cover.piece_count() {
        let (lo, hi) = cover.piece_bounds(index);
        for y in [lo[1], hi[1]] {
            for x in [lo[0], hi[0]] {
                corners.push(Vec2::new(x, y));
            }
        }
    }
    let mut ring = convex_hull(corners);
    reduce_sides(&mut ring, MAX_HULL_SIDES);
    ring
}

/// Cover the letter's cross-section with axis-aligned boxes on a grid of the
/// given pitch.
fn extract_cover(field: &DistanceField2D, cell: f32) -> Isovolume<2> {
    let (cells_x, cells_y) = field.cell_counts();
    let padding = Vec2::splat(COVER_PADDING_CELLS * cell);
    let lo = field.corner(0, 0) - padding;
    let span = field.corner(cells_x, cells_y) + padding - lo;
    let counts = (span / cell).ceil().max(Vec2::ONE);
    // Handing `Isovolume` the longer axis' cell count as its resolution makes
    // the pitch it derives the one asked for, so every letter of a word covers
    // on one pitch however wide its own ink is, matching the render bake's
    // fixed-cell rule.
    let resolution = counts.max_element() as usize;
    Isovolume::extract(
        lo.to_array(),
        (lo + counts * cell).to_array(),
        resolution,
        |p| LIPSCHITZ_SCALE * field.sample(Vec2::from_array(p)),
    )
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
    use glam::Vec4Swizzles;

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
        collider_solid(half, resolution, resolution, depth, slab)
    }

    /// Fixture params on a synthetic em: the fixture's own extent is one em, so
    /// `collider_cells` reads as cells across the shape exactly the way the
    /// render `cells` does.
    fn fixture_params(
        em: f32,
        collider_cells: u32,
        depth: f32,
        slab: (f32, f32),
    ) -> super::GlyphParams {
        super::GlyphParams {
            em_size: em,
            depth,
            slab,
            collider_resolution: collider_cells,
            ..super::GlyphParams::default()
        }
    }

    /// The two pitches are independent, so fixtures name both. `cells` counts
    /// cells across the square in each case, so a `collider_cells` below
    /// `cells` is the decoupled-collider-resolution case.
    fn collider_solid(
        half: f32,
        cells: u32,
        collider_cells: u32,
        depth: f32,
        slab: (f32, f32),
    ) -> GlyphSolid {
        GlyphSolid::new(
            'X',
            Vec2::ZERO,
            2.0 * half,
            &fixture_params(2.0 * half, collider_cells, depth, slab),
            Some(square_field(half, cells)),
        )
    }

    /// A square rotated 45 degrees. Convex, so the render clip still tiles it
    /// at `Theta(res^2)` pieces, while the axis-aligned cover of its diagonal
    /// boundary is a staircase whose box count tracks the collider pitch. That
    /// separation is what the two-pitch design is for; an axis-aligned fixture
    /// covers in one box at every resolution and shows nothing.
    fn diamond_solid(half: f32, cells: u32, collider_cells: u32) -> GlyphSolid {
        diamond_hull_solid(half, cells, collider_cells, 0.4, (-0.2, 0.2))
    }

    /// The diamond again, with the depth and slab the prism tests need to tell
    /// the two interval axes apart.
    fn diamond_hull_solid(
        half: f32,
        cells: u32,
        collider_cells: u32,
        depth: f32,
        slab: (f32, f32),
    ) -> GlyphSolid {
        let contour = Contour {
            points: vec![
                Vec2::new(-half, 0.0),
                Vec2::new(0.0, -half),
                Vec2::new(half, 0.0),
                Vec2::new(0.0, half),
            ],
        };
        let field = DistanceField2D::bake(&[contour], 2.0 * half / cells as f32).expect("bake");
        GlyphSolid::new(
            'X',
            Vec2::ZERO,
            2.0 * half,
            &fixture_params(2.0 * half, collider_cells, depth, slab),
            Some(field),
        )
    }

    /// A square annulus: an outer ring with a counter-wound hole, i.e. the
    /// topology of `O` without a font. The hole is what a bounding-volume
    /// shortcut would fill in.
    fn annulus_solid(outer: f32, inner: f32, cells: u32, collider_cells: u32) -> GlyphSolid {
        let ring = |half: f32, counter_clockwise: bool| {
            let mut points = vec![
                Vec2::new(-half, -half),
                Vec2::new(half, -half),
                Vec2::new(half, half),
                Vec2::new(-half, half),
            ];
            if !counter_clockwise {
                points.reverse();
            }
            Contour { points }
        };
        let field = DistanceField2D::bake(
            &[ring(outer, true), ring(inner, false)],
            2.0 * outer / cells as f32,
        )
        .expect("bake");
        GlyphSolid::new(
            'O',
            Vec2::ZERO,
            2.0 * outer,
            &fixture_params(2.0 * outer, collider_cells, 0.4, (-0.2, 0.2)),
            Some(field),
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

    /// One 16-vertex box per cover piece, origin-centred with the extrinsic
    /// centre carrying the placement, spanning the full depth and slab. A box
    /// that did not span both would leave the letter open on a face.
    #[test]
    fn colliders_are_origin_centred_boxes_spanning_depth_and_slab() {
        let slab = (-0.75, 0.25);
        let depth = 0.6;
        let solid = collider_solid(1.0, 24, 12, depth, slab);
        let cover = solid.collider_cover().expect("cover");
        let colliders = solid.colliders_4d();
        assert_eq!(colliders.len(), solid.collider_count());
        assert_eq!(colliders.len(), cover.piece_count());
        assert!(!colliders.is_empty());

        for (index, (centre, collider)) in colliders.iter().enumerate() {
            let Shape::ConvexPolytope4D { vertices } = collider else {
                panic!("expected ConvexPolytope4D, got {:?}", collider.kind());
            };
            assert_eq!(vertices.len(), 16);
            let sum: Vec4 = vertices.iter().copied().sum();
            assert!(sum.length() < 1e-5, "hull is not origin-centred");

            let (lo, hi) = cover.piece_bounds(index);
            let expect = Vec4::new(
                0.5 * (lo[0] + hi[0]),
                0.5 * (lo[1] + hi[1]),
                0.0,
                0.5 * (slab.0 + slab.1),
            );
            assert!(
                centre.distance(expect) < 1e-6,
                "centre {centre} vs {expect}"
            );
            let half = Vec4::new(
                0.5 * (hi[0] - lo[0]),
                0.5 * (hi[1] - lo[1]),
                0.5 * depth,
                0.5 * (slab.1 - slab.0),
            );
            for v in vertices {
                assert!((v.abs() - half).abs().max_element() < 1e-6);
            }
        }
    }

    /// The criterion the cover exists for, checked against the field rather
    /// than against the extractor's own occupancy: every point the baked field
    /// calls solid lies inside some collider box. The probe lattice is offset
    /// off both grids so it cannot sit on cell centres by construction.
    #[test]
    fn every_point_the_field_calls_solid_lies_inside_a_collider_box() {
        let solid = collider_solid(1.0, 32, 12, 0.5, (-0.25, 0.25));
        let cover = solid.collider_cover().expect("cover");
        const PROBES: usize = 201;
        let coord = |i: usize| -1.3 + 2.6 * (i as f32 + 0.317) / PROBES as f32;
        let mut interior = 0;
        for j in 0..PROBES {
            for i in 0..PROBES {
                let p = Vec2::new(coord(i), coord(j));
                if solid.distance_2d(p) > 0.0 {
                    continue;
                }
                interior += 1;
                assert!(
                    cover.contains(p.to_array()),
                    "interior probe {p} is outside every collider box"
                );
            }
        }
        assert!(interior > 10_000, "only {interior} interior probes");
        assert!(!cover.clipped(), "the cover ran off its sampling domain");
    }

    /// The cover does not spill: no box corner is further outside the letter
    /// than [`GlyphSolid::collider_margin`], which is the Lipschitz bound the
    /// occupancy test earns. Without this an enclosure test would pass on a
    /// bounding box.
    #[test]
    fn no_collider_corner_exceeds_the_stated_margin() {
        let solid = collider_solid(1.0, 32, 12, 0.5, (-0.25, 0.25));
        let cover = solid.collider_cover().expect("cover");
        let margin = solid.collider_margin();
        let mut worst = f32::NEG_INFINITY;
        for index in 0..cover.piece_count() {
            let (lo, hi) = cover.piece_bounds(index);
            for y in [lo[1], hi[1]] {
                for x in [lo[0], hi[0]] {
                    worst = worst.max(solid.distance_2d(Vec2::new(x, y)));
                }
            }
        }
        assert!(worst <= margin, "corner reaches {worst}, past {margin}");
    }

    /// A collider pitch coarser than the render pitch is the whole point of
    /// the two being separate: the render mesh keeps its detail while the box
    /// count collapses. The clip decomposition at the same coarse pitch is
    /// what this replaces, so the comparison is against the fine render count.
    #[test]
    fn a_coarser_collider_pitch_cuts_boxes_without_touching_the_render_mesh() {
        let fine = diamond_solid(1.0, 48, 48);
        let coarse = diamond_solid(1.0, 48, 12);
        assert_eq!(fine.piece_count(), coarse.piece_count());
        assert!(
            coarse.collider_count() < fine.collider_count(),
            "coarse {} is not below fine {}",
            coarse.collider_count(),
            fine.collider_count()
        );
        assert!(
            coarse.collider_count() * 20 < fine.piece_count(),
            "{} boxes against {} render pieces is not a cut",
            coarse.collider_count(),
            fine.piece_count()
        );
        assert!(coarse.collider_margin() > fine.collider_margin());
    }

    /// A hole in the letter stays a hole, at a collider pitch four times
    /// coarser than the render pitch. A convex hull and a bounding volume both
    /// lose this, and both would still pass every enclosure check.
    ///
    /// The bound is the same Lipschitz derivation as
    /// [`GlyphSolid::collider_margin`], read the other way: a covered point
    /// lies in a marked cell, whose centre has `d <= cell` and is within one
    /// cell of it in L1, so no point with `d > 2 * cell` can be covered.
    #[test]
    fn a_counter_stays_open_past_the_margin() {
        let solid = annulus_solid(1.0, 0.6, 48, 16);
        let cover = solid.collider_cover().expect("cover");
        let margin = solid.collider_margin();
        assert!(
            solid.distance_2d(Vec2::ZERO) > margin,
            "fixture hole is smaller than the margin"
        );

        const PROBES: usize = 121;
        let coord = |i: usize| -0.6 + 1.2 * (i as f32 + 0.317) / PROBES as f32;
        let mut clear = 0;
        for j in 0..PROBES {
            for i in 0..PROBES {
                let p = Vec2::new(coord(i), coord(j));
                if solid.distance_2d(p) <= margin {
                    continue;
                }
                clear += 1;
                assert!(!cover.contains(p.to_array()), "counter filled at {p}");
            }
        }
        assert!(clear > 5_000, "only {clear} probes deep inside the counter");
    }

    /// The occupancy threshold [`LIPSCHITZ_SCALE`] buys, stated exactly: a
    /// cover cell is marked when its centre samples `d <= cell`, not
    /// `d <= cell/sqrt(2)`. That factor is the whole enclosure argument, and
    /// dropping it fails no containment probe on a fixture whose worst
    /// Lipschitz case the probes happen to miss, so the threshold is pinned
    /// directly rather than through its consequences.
    ///
    /// A cell centre is interior to exactly one cell and every box is a union
    /// of marked cells, so `contains` at a centre reads the occupancy bit.
    #[test]
    fn a_cover_cell_is_marked_out_to_a_full_cell_of_clearance() {
        // A single pitch quantises the sampled distances into an arithmetic
        // progression that can step straight over the band between the two
        // thresholds, so the fixture is swept rather than fixed.
        let (mut band, mut clear) = (0, 0);
        for collider_cells in [11u32, 12, 13, 16, 17] {
            let solid = diamond_solid(1.0, 48, collider_cells);
            let cover = solid.collider_cover().expect("cover");
            let cell = cover.cell_size();
            let margin = solid.collider_margin();
            assert!(
                (2.0 * cell - margin).abs() <= 4.0 * f32::EPSILON * margin,
                "stated margin {margin} is not two extractor cells of {cell}"
            );

            // Box bounds sit on grid nodes, so one of them anchors the lattice.
            let anchor = Vec2::from_array(cover.piece_bounds(0).0);
            let tolerance = 1.0e-4 * cell;
            let half_threshold = std::f32::consts::FRAC_1_SQRT_2 * cell;

            for j in -40..=40 {
                for i in -40..=40 {
                    let centre = anchor + (Vec2::new(i as f32, j as f32) + Vec2::splat(0.5)) * cell;
                    let d = solid.distance_2d(centre);
                    if d <= cell - tolerance {
                        assert!(
                            cover.contains(centre.to_array()),
                            "cell centre {centre} at d = {d} is unmarked inside one cell"
                        );
                        if d > half_threshold + tolerance {
                            band += 1;
                        }
                    } else if d >= cell + tolerance {
                        clear += 1;
                        assert!(
                            !cover.contains(centre.to_array()),
                            "cell centre {centre} at d = {d} is marked past one cell"
                        );
                    }
                }
            }
        }
        assert!(
            band > 0,
            "no cell centre lands between the scaled and unscaled thresholds, \
             so the sqrt(2) correction is untested by these fixtures"
        );
        assert!(
            clear > 0,
            "only {clear} cell centres clear of the threshold"
        );
    }

    /// The dynamic collider's shape contract: one origin-centred prism whose
    /// extrinsic centre is its centre of mass, spanning the depth and the slab
    /// exactly. A hull centred anywhere else spins about the wrong point under
    /// an off-centre contact.
    #[test]
    fn the_rigid_hull_is_one_origin_centred_prism_about_its_centre_of_mass() {
        let slab = (-0.75, 0.25);
        let depth = 0.6;
        let solid = diamond_hull_solid(1.0, 48, 16, depth, slab);
        let sides = solid.rigid_hull_sides();
        let (centre, shape) = solid.rigid_hull_4d().expect("hull");
        let Shape::ConvexPolytope4D { vertices } = shape else {
            panic!("hull is not 4D convex");
        };
        assert_eq!(vertices.len(), 4 * sides);
        assert!(vertices.len() <= 32);

        // Origin-centred on the centre of mass, which for a ring is its area
        // centroid and not its vertex mean: greedy reduction leaves an uneven
        // vertex spacing that a mean would follow and a centroid must not.
        let local: Vec<Vec2> = vertices[..sides].iter().map(|v| v.xy()).collect();
        let local_centroid = centroid(&local);
        assert!(
            local_centroid.length() < 1e-5,
            "hull is not centred on its centre of mass: {local_centroid}"
        );
        assert_eq!(centre.z, 0.0);
        assert!((centre.w - 0.5 * (slab.0 + slab.1)).abs() < 1e-6);
        // A diamond about the origin has its centroid there, up to the cover's
        // own outward margin.
        assert!(centre.xy().length() < solid.collider_margin());

        for v in &vertices {
            assert!((v.z.abs() - 0.5 * depth).abs() < 1e-6);
            assert!((v.w.abs() - 0.5 * (slab.1 - slab.0)).abs() < 1e-6);
        }
        // The ring is repeated once per `(z, w)` corner, in one order.
        for k in 0..sides {
            for copy in 1..4 {
                assert_eq!(vertices[copy * sides + k].xy(), vertices[k].xy());
            }
        }
    }

    /// The soundness property: the hull contains the cover it replaces, so a
    /// dynamic letter never lets a body reach ink the static one would stop.
    #[test]
    fn the_rigid_hull_contains_the_whole_cover() {
        let solid = diamond_hull_solid(1.0, 48, 16, 0.4, (-0.2, 0.2));
        let cover = solid.collider_cover().expect("cover");
        let ring = &solid.hull;
        let n = ring.len();
        for index in 0..cover.piece_count() {
            let (lo, hi) = cover.piece_bounds(index);
            for y in [lo[1], hi[1]] {
                for x in [lo[0], hi[0]] {
                    let p = Vec2::new(x, y);
                    for k in 0..n {
                        let a = ring[k];
                        let b = ring[(k + 1) % n];
                        assert!(
                            (b - a).perp_dot(p - a) >= -1e-5,
                            "box corner {p} lies outside hull edge {k}"
                        );
                    }
                }
            }
        }
    }

    /// What the dynamic path gives up, stated rather than implied: a counter
    /// the cover keeps open is filled by the hull. Convexity is the price of
    /// one collider per body, and a reader comparing the two emitters has to be
    /// able to see it fail here if it ever stops being true.
    #[test]
    fn the_rigid_hull_fills_a_counter_the_cover_keeps_open() {
        let solid = annulus_solid(1.0, 0.6, 48, 16);
        let cover = solid.collider_cover().expect("cover");
        assert!(!cover.contains([0.0, 0.0]), "fixture counter is not open");

        let ring = &solid.hull;
        let n = ring.len();
        let covered = (0..n).all(|k| {
            let a = ring[k];
            let b = ring[(k + 1) % n];
            (b - a).perp_dot(-a) >= 0.0
        });
        assert!(covered, "the hull left the annulus centre outside");
    }

    /// A blank has no ring and therefore no dynamic body, matching the static
    /// emitter rather than returning a degenerate prism.
    #[test]
    fn a_blank_emits_no_rigid_hull() {
        let blank = GlyphSolid::new(
            ' ',
            Vec2::ZERO,
            0.25,
            &fixture_params(1.0, 20, 0.5, (0.0, 1.0)),
            None,
        );
        assert_eq!(blank.rigid_hull_sides(), 0);
        assert!(blank.rigid_hull_4d().is_none());
    }

    /// The hull is a fixed-order construction over a fixed-order cover, so two
    /// bakes of one letter emit the same vertices in the same order. A dynamic
    /// letter whose collider reordered would break replay.
    #[test]
    fn rigid_hull_emission_is_reproducible() {
        let build = || diamond_hull_solid(1.0, 48, 16, 0.4, (-0.2, 0.2)).rigid_hull_4d();
        let (Some((ca, sa)), Some((cb, sb))) = (build(), build()) else {
            panic!("hull is absent")
        };
        assert_eq!(ca, cb);
        let (Shape::ConvexPolytope4D { vertices: va }, Shape::ConvexPolytope4D { vertices: vb }) =
            (sa, sb)
        else {
            unreachable!()
        };
        assert!(va.len() > 4);
        assert_eq!(va, vb);
    }

    /// The cover is a pure fixed-order scan over the baked field, so two bakes
    /// of the same letter emit the same boxes in the same order. A collider set
    /// that reordered would break the solver's deterministic pair ordering.
    #[test]
    fn collider_emission_is_reproducible() {
        let build = || collider_solid(1.0, 24, 16, 0.5, (-0.25, 0.25)).colliders_4d();
        let (a, b) = (build(), build());
        assert_eq!(a.len(), b.len());
        assert!(a.len() > 1);
        for ((ca, sa), (cb, sb)) in a.iter().zip(&b) {
            assert_eq!(ca, cb);
            let (
                Shape::ConvexPolytope4D { vertices: va },
                Shape::ConvexPolytope4D { vertices: vb },
            ) = (sa, sb)
            else {
                unreachable!()
            };
            assert_eq!(va, vb);
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
        let blank = GlyphSolid::new(
            ' ',
            Vec2::ZERO,
            0.25,
            &fixture_params(1.0, 20, 0.5, (0.0, 1.0)),
            None,
        );
        assert!(blank.is_blank());
        assert_eq!(blank.piece_count(), 0);
        assert_eq!(blank.collider_count(), 0);
        assert!(blank.collider_cover().is_none());
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
