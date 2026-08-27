//! Conservative isovolume extraction: a signed-distance field turned into a
//! set of convex boxes whose union encloses the field's solid region, so a
//! shape that exists only as an SDF can be handed to a narrowphase.
//!
//! This is the collider-side analogue of marching cubes (Lorensen & Cline,
//! 1987), and it deliberately solves a different problem. A mesher wants the
//! isosurface interpolated; a collider wants the solid *enclosed*, because a
//! piece of shape that no convex hull covers is a hole a body tunnels
//! through. So the primitive here is a cell-occupancy test with a proof
//! attached rather than an edge-crossing table:
//!
//! An SDF is 1-Lipschitz (Hart, "Sphere Tracing", 1996, §2), so for a cell of
//! centre `c` and half-diagonal `m` and any point `p` in that cell,
//! `f(p) >= f(c) - m`. Marking every cell with `f(c) <= m` therefore marks a
//! superset of the cells that meet `{f <= 0}`: the union of marked cells
//! encloses the whole solid, and overshoots it by at most `2m` (a corner of a
//! marked cell satisfies `f <= 2m`). No interpolation, no saddle cases, no
//! per-cell topology table, and the guarantee holds in any dimension, which
//! is what makes the 4D instantiation the same code as the 3D one.
//!
//! Piece count is the whole problem. One convex piece per occupied cell is
//! correct and useless: `res^D` shell cells is thousands of colliders before
//! the resolution is interesting. The marked set is instead covered by
//! first-fit maximal boxes: seed at the first uncovered cell in row-major
//! order, grow the box one layer at a time in each of the `2D` axis
//! directions while the new layer stays inside the marked set, emit it,
//! repeat. Boxes may overlap (growth reads occupancy, only seeding reads
//! coverage), which is what lets a later box swallow a whole interior slab
//! instead of being fenced in by its predecessors. Seeding by scan order
//! rather than by largest coverage is not the greedy set-cover rule, so
//! Chvátal's `O(log n)` ratio (1979) does not transfer and no approximation
//! bound is claimed here; the measured table below is the only evidence for
//! the piece count.
//!
//! Measured on a torus and its 4D analogue (minor radius 0.3, major 1.0) by
//! `examples/isovolume_piece_budget.rs`, `cover/solid` being the extracted
//! volume over the analytic one:
//!
//! | dim | resolution | occupied cells | pieces | cover/solid |
//! |-----|-----------:|---------------:|-------:|------------:|
//! | 3D  |         16 |            512 |     24 |        2.31 |
//! | 3D  |         32 |          3 064 |     93 |        1.72 |
//! | 3D  |         64 |         18 800 |    309 |        1.32 |
//! | 4D  |         12 |          2 752 |     84 |        3.83 |
//! | 4D  |         16 |          6 032 |    114 |        2.66 |
//! | 4D  |         24 |         23 920 |    382 |        2.08 |
//!
//! Resolution 32 in 3D and 16 in 4D are the usable settings: roughly a
//! hundred pieces, two to three percent of the occupied cells, and a cover
//! under 3x the solid. The finer row under each buys fidelity at three to
//! four times the piece count, which suits one hero shape rather than a
//! scene of them. Piece count tracks the cover's shape complexity, not its
//! cell count: refining the grid 4x in 3D multiplied cells by 64 and pieces
//! by 13.
//!
//! Everything is a fixed-order scan over `f32` with no hashing, so a given
//! field and grid always produce the same boxes in the same order.

use glam::{Vec3, Vec4};

use crate::Shape;

/// A cell range, both corners inclusive, in grid index space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CellBox<const D: usize> {
    lo: [usize; D],
    hi: [usize; D],
}

/// A signed-distance field covered by axis-aligned convex boxes.
///
/// Constructed by [`Isovolume::extract`]. The union of the boxes contains
/// `{p : f(p) <= 0}` restricted to the sampled domain; [`Isovolume::clipped`]
/// reports whether that domain was large enough for the containment to be
/// unconditional.
#[derive(Clone, Debug)]
pub struct Isovolume<const D: usize> {
    origin: [f32; D],
    cell: f32,
    boxes: Vec<CellBox<D>>,
    occupied_cells: usize,
    clipped: bool,
}

impl<const D: usize> Isovolume<D> {
    /// Cover `{p : sdf(p) <= 0}` with convex boxes over the domain
    /// `[min, max]`, sampling `resolution` cells along the longest axis.
    ///
    /// `sdf` must be a true signed distance (1-Lipschitz); a field that
    /// merely has the right sign, such as an unnormalised implicit surface,
    /// breaks the enclosure guarantee. `min`/`max` must contain the solid
    /// with room for one cell of margin, which [`Isovolume::clipped`] checks.
    pub fn extract(
        min: [f32; D],
        max: [f32; D],
        resolution: usize,
        sdf: impl Fn([f32; D]) -> f32,
    ) -> Self {
        assert!(resolution > 0, "resolution must be positive");
        let mut extent = [0.0f32; D];
        for (e, (lo, hi)) in extent.iter_mut().zip(min.iter().zip(max.iter())) {
            *e = hi - lo;
            assert!(*e > 0.0, "domain must be non-degenerate on every axis");
        }
        // One cell size for every axis: the Lipschitz margin is the cell's
        // half-diagonal, and a per-axis size would make that margin
        // direction-dependent for no gain.
        let cell = extent.iter().copied().fold(0.0f32, f32::max) / resolution as f32;
        let mut counts = [0usize; D];
        for (n, e) in counts.iter_mut().zip(extent.iter()) {
            *n = ((e / cell).ceil() as usize).max(1);
        }

        let margin = 0.5 * cell * (D as f32).sqrt();
        let total: usize = counts.iter().product();
        let mut occupied = vec![false; total];
        let mut occupied_cells = 0usize;
        let mut clipped = false;
        for (index, marked) in occupied.iter_mut().enumerate() {
            let coords = unflatten(index, &counts);
            if sdf(cell_centre(&min, cell, &coords)) > margin {
                continue;
            }
            *marked = true;
            occupied_cells += 1;
            clipped |= coords
                .iter()
                .zip(counts.iter())
                .any(|(&c, &n)| c == 0 || c + 1 == n);
        }

        let mut covered = vec![false; total];
        let mut boxes = Vec::new();
        let mut seed = 0usize;
        loop {
            while seed < total && (covered[seed] || !occupied[seed]) {
                seed += 1;
            }
            if seed == total {
                break;
            }
            let grown = grow(unflatten(seed, &counts), &occupied, &counts);
            visit_cells(&grown, |coords| {
                covered[flatten(coords, &counts)] = true;
                true
            });
            boxes.push(grown);
        }

        Self {
            origin: min,
            cell,
            boxes,
            occupied_cells,
            clipped,
        }
    }

    /// Number of convex pieces.
    pub fn piece_count(&self) -> usize {
        self.boxes.len()
    }

    /// Cells the occupancy test marked, i.e. the exact cell count the pieces
    /// cover between them.
    pub fn occupied_cells(&self) -> usize {
        self.occupied_cells
    }

    /// Grid cell edge length.
    pub fn cell_size(&self) -> f32 {
        self.cell
    }

    /// Upper bound on how far the covered region extends past the true
    /// surface: a corner of a marked cell has `f <= cell·sqrt(D)`.
    pub fn enclosure_margin(&self) -> f32 {
        self.cell * (D as f32).sqrt()
    }

    /// Exact volume of the union of the pieces: that union is exactly the
    /// marked cell set, so the overlaps need no inclusion-exclusion.
    pub fn volume(&self) -> f32 {
        self.occupied_cells as f32 * self.cell.powi(D as i32)
    }

    /// True when a marked cell touched the sampled domain's boundary, meaning
    /// the solid ran off the grid and the enclosure guarantee does not hold.
    pub fn clipped(&self) -> bool {
        self.clipped
    }

    /// Corner-to-corner bounds of piece `index` in the grid frame.
    pub fn piece_bounds(&self, index: usize) -> ([f32; D], [f32; D]) {
        let b = self.boxes[index];
        let mut lo = [0.0f32; D];
        let mut hi = [0.0f32; D];
        for k in 0..D {
            lo[k] = self.origin[k] + b.lo[k] as f32 * self.cell;
            hi[k] = self.origin[k] + (b.hi[k] + 1) as f32 * self.cell;
        }
        (lo, hi)
    }

    /// Whether `p` lies in the covered region. Linear in piece count: a
    /// verification query, not a broadphase.
    pub fn contains(&self, p: [f32; D]) -> bool {
        (0..self.boxes.len()).any(|i| {
            let (lo, hi) = self.piece_bounds(i);
            (0..D).all(|k| p[k] >= lo[k] && p[k] <= hi[k])
        })
    }
}

impl Isovolume<3> {
    /// One collider per piece: the piece centre in the grid frame, paired
    /// with an origin-centred 8-vertex hull. Pose is extrinsic per the
    /// [`Shape`] contract, so the centre is the body position.
    pub fn colliders(&self) -> Vec<(Vec3, Shape)> {
        (0..self.boxes.len())
            .map(|i| {
                let (lo, hi) = self.piece_bounds(i);
                let centre = Vec3::from_array(lo).lerp(Vec3::from_array(hi), 0.5);
                let h = 0.5 * (Vec3::from_array(hi) - Vec3::from_array(lo));
                let mut vertices = Vec::with_capacity(8);
                for sz in [-1.0f32, 1.0] {
                    for sy in [-1.0f32, 1.0] {
                        for sx in [-1.0f32, 1.0] {
                            vertices.push(Vec3::new(sx * h.x, sy * h.y, sz * h.z));
                        }
                    }
                }
                (centre, Shape::ConvexPolytope3D { vertices })
            })
            .collect()
    }
}

impl Isovolume<4> {
    /// One collider per piece: the piece centre in the grid frame, paired
    /// with an origin-centred 16-vertex hull.
    pub fn colliders(&self) -> Vec<(Vec4, Shape)> {
        (0..self.boxes.len())
            .map(|i| {
                let (lo, hi) = self.piece_bounds(i);
                let centre = Vec4::from_array(lo).lerp(Vec4::from_array(hi), 0.5);
                let h = 0.5 * (Vec4::from_array(hi) - Vec4::from_array(lo));
                let mut vertices = Vec::with_capacity(16);
                for sw in [-1.0f32, 1.0] {
                    for sz in [-1.0f32, 1.0] {
                        for sy in [-1.0f32, 1.0] {
                            for sx in [-1.0f32, 1.0] {
                                vertices.push(Vec4::new(sx * h.x, sy * h.y, sz * h.z, sw * h.w));
                            }
                        }
                    }
                }
                (centre, Shape::ConvexPolytope4D { vertices })
            })
            .collect()
    }
}

fn cell_centre<const D: usize>(origin: &[f32; D], cell: f32, coords: &[usize; D]) -> [f32; D] {
    let mut p = [0.0f32; D];
    for k in 0..D {
        p[k] = origin[k] + (coords[k] as f32 + 0.5) * cell;
    }
    p
}

/// Axis 0 varies fastest.
fn flatten<const D: usize>(coords: &[usize; D], counts: &[usize; D]) -> usize {
    let mut index = 0;
    for k in (0..D).rev() {
        index = index * counts[k] + coords[k];
    }
    index
}

fn unflatten<const D: usize>(mut index: usize, counts: &[usize; D]) -> [usize; D] {
    let mut coords = [0usize; D];
    for k in 0..D {
        coords[k] = index % counts[k];
        index /= counts[k];
    }
    coords
}

/// Call `f` on every cell of `b` in row-major order until it returns false.
fn visit_cells<const D: usize>(b: &CellBox<D>, mut f: impl FnMut(&[usize; D]) -> bool) {
    let mut coords = b.lo;
    loop {
        if !f(&coords) {
            return;
        }
        let mut axis = 0;
        loop {
            if axis == D {
                return;
            }
            if coords[axis] < b.hi[axis] {
                coords[axis] += 1;
                break;
            }
            coords[axis] = b.lo[axis];
            axis += 1;
        }
    }
}

fn all_occupied<const D: usize>(b: &CellBox<D>, occupied: &[bool], counts: &[usize; D]) -> bool {
    let mut ok = true;
    visit_cells(b, |coords| {
        ok = occupied[flatten(coords, counts)];
        ok
    });
    ok
}

/// Grow a single-cell box outwards one layer at a time, cycling the `2D`
/// directions in a fixed order until a full round adds nothing. Round-robin
/// rather than axis-at-a-time keeps the box from committing to one long
/// direction and then being unable to thicken.
fn grow<const D: usize>(seed: [usize; D], occupied: &[bool], counts: &[usize; D]) -> CellBox<D> {
    let mut b = CellBox { lo: seed, hi: seed };
    loop {
        let mut grew = false;
        for axis in 0..D {
            if b.lo[axis] > 0 {
                let mut layer = b;
                layer.lo[axis] = b.lo[axis] - 1;
                layer.hi[axis] = b.lo[axis] - 1;
                if all_occupied(&layer, occupied, counts) {
                    b.lo[axis] -= 1;
                    grew = true;
                }
            }
            if b.hi[axis] + 1 < counts[axis] {
                let mut layer = b;
                layer.lo[axis] = b.hi[axis] + 1;
                layer.hi[axis] = b.hi[axis] + 1;
                if all_occupied(&layer, occupied, counts) {
                    b.hi[axis] += 1;
                    grew = true;
                }
            }
        }
        if !grew {
            return b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact SDF of a torus of revolution about the y axis (Quilez,
    /// "distance functions", 2019, `sdTorus`). Non-convex and smooth: the
    /// case a vertex list cannot author and a primitive cannot approximate.
    fn torus_3d(major: f32, minor: f32) -> impl Fn([f32; 3]) -> f32 {
        move |p| {
            let radial = (p[0] * p[0] + p[2] * p[2]).sqrt() - major;
            (radial * radial + p[1] * p[1]).sqrt() - minor
        }
    }

    /// The 4D analogue: revolve a 2-sphere of radius `minor` about the `w`
    /// axis at distance `major`. Exact for the same reason the 3D form is.
    fn torus_4d(major: f32, minor: f32) -> impl Fn([f32; 4]) -> f32 {
        move |p| {
            let radial = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - major;
            (radial * radial + p[3] * p[3]).sqrt() - minor
        }
    }

    fn sphere_3d(radius: f32) -> impl Fn([f32; 3]) -> f32 {
        move |p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - radius
    }

    const TORUS_MAJOR: f32 = 1.0;
    const TORUS_MINOR: f32 = 0.3;
    const TORUS_BOUND: f32 = 1.6;

    fn torus_volume_3d() -> f32 {
        2.0 * std::f32::consts::PI.powi(2) * TORUS_MAJOR * TORUS_MINOR * TORUS_MINOR
    }

    fn extract_torus_3d(resolution: usize) -> Isovolume<3> {
        Isovolume::extract(
            [-TORUS_BOUND; 3],
            [TORUS_BOUND; 3],
            resolution,
            torus_3d(TORUS_MAJOR, TORUS_MINOR),
        )
    }

    /// A deterministic low-discrepancy direction sequence: the additive
    /// recurrence with the plastic-number constants (Roberts, "The unreasonable
    /// effectiveness of quasirandom sequences", 2018) mapped to a sphere by
    /// the cylindrical-equal-area construction. Seeded arithmetic only, so the
    /// enclosure tests are reproducible bit for bit.
    fn direction_3d(i: usize) -> [f32; 3] {
        const ALPHA_1: f32 = 0.754_877_7;
        const ALPHA_2: f32 = 0.569_840_3;
        let u = (0.5 + ALPHA_1 * i as f32).fract();
        let v = (0.5 + ALPHA_2 * i as f32).fract();
        let z = 2.0 * u - 1.0;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let phi = 2.0 * std::f32::consts::PI * v;
        [r * phi.cos(), r * phi.sin(), z]
    }

    /// March out from an interior point along `dir` to the first sign change,
    /// then bisect onto the zero level. Marching rather than assuming a
    /// bracket matters for near-tangential rays, which leave the tube far
    /// later than the minor radius would suggest.
    fn surface_point_3d(
        sdf: &impl Fn([f32; 3]) -> f32,
        from: [f32; 3],
        dir: [f32; 3],
    ) -> Option<[f32; 3]> {
        const STEP: f32 = 0.01;
        const STEPS: usize = 800;
        let at = |t: f32| {
            [
                from[0] + dir[0] * t,
                from[1] + dir[1] * t,
                from[2] + dir[2] * t,
            ]
        };
        if sdf(at(0.0)) > 0.0 {
            return None;
        }
        let mut lo = 0.0f32;
        let hi = (1..=STEPS).map(|i| i as f32 * STEP).find(|&t| {
            if sdf(at(t)) <= 0.0 {
                lo = t;
                false
            } else {
                true
            }
        })?;
        let mut hi = hi;
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if sdf(at(mid)) <= 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(at(0.5 * (lo + hi)))
    }

    #[test]
    fn every_isosurface_sample_lies_inside_a_piece() {
        let sdf = torus_3d(TORUS_MAJOR, TORUS_MINOR);
        for resolution in [24, 32, 48] {
            let volume = extract_torus_3d(resolution);
            assert!(!volume.clipped());
            let mut checked = 0;
            for i in 0..512 {
                // Ride the tube's core circle so the rays leave from inside.
                let angle = i as f32 * std::f32::consts::TAU / 512.0;
                let core = [TORUS_MAJOR * angle.cos(), 0.0, TORUS_MAJOR * angle.sin()];
                let dir = direction_3d(i);
                let Some(p) = surface_point_3d(&sdf, core, dir) else {
                    continue;
                };
                assert!(
                    sdf(p).abs() < 1.0e-4,
                    "bisection did not land on the surface: f = {}",
                    sdf(p)
                );
                assert!(
                    volume.contains(p),
                    "resolution {resolution}: surface point {p:?} is outside every piece"
                );
                checked += 1;
            }
            assert!(checked > 400, "only {checked} usable surface samples");
        }
    }

    #[test]
    fn every_interior_probe_lies_inside_a_piece() {
        let sdf = torus_3d(TORUS_MAJOR, TORUS_MINOR);
        let volume = extract_torus_3d(32);
        const PROBES: usize = 61;
        let mut interior = 0;
        for iz in 0..PROBES {
            for iy in 0..PROBES {
                for ix in 0..PROBES {
                    let coord = |i: usize| {
                        -TORUS_BOUND + 2.0 * TORUS_BOUND * (i as f32 + 0.31) / PROBES as f32
                    };
                    let p = [coord(ix), coord(iy), coord(iz)];
                    if sdf(p) > 0.0 {
                        continue;
                    }
                    interior += 1;
                    assert!(volume.contains(p), "interior probe {p:?} is not covered");
                }
            }
        }
        assert!(interior > 1000, "only {interior} interior probes");
    }

    #[test]
    fn no_piece_corner_exceeds_the_enclosure_margin() {
        let sdf = torus_3d(TORUS_MAJOR, TORUS_MINOR);
        let volume = extract_torus_3d(32);
        let margin = volume.enclosure_margin();
        for i in 0..volume.piece_count() {
            let (lo, hi) = volume.piece_bounds(i);
            for cz in [lo[2], hi[2]] {
                for cy in [lo[1], hi[1]] {
                    for cx in [lo[0], hi[0]] {
                        let d = sdf([cx, cy, cz]);
                        assert!(d <= margin + 1.0e-5, "corner distance {d} exceeds {margin}");
                    }
                }
            }
        }
    }

    #[test]
    fn cover_volume_tightens_towards_the_true_volume() {
        let truth = torus_volume_3d();
        let mut previous = f32::INFINITY;
        for resolution in [16, 32, 64] {
            let ratio = extract_torus_3d(resolution).volume() / truth;
            assert!(
                ratio > 1.0,
                "resolution {resolution}: cover {ratio} misses volume"
            );
            assert!(
                ratio < previous,
                "resolution {resolution}: ratio {ratio} did not improve on {previous}"
            );
            previous = ratio;
        }
        assert!(
            previous < 1.6,
            "finest cover is still {previous}x the solid"
        );
    }

    #[test]
    fn piece_count_stays_within_the_measured_budget_at_three_resolutions() {
        let counts_3d: Vec<usize> = [16, 32, 64]
            .iter()
            .map(|&r| extract_torus_3d(r).piece_count())
            .collect();
        assert!(
            counts_3d[0] <= 28 && counts_3d[1] <= 107 && counts_3d[2] <= 355,
            "3D piece counts {counts_3d:?} regressed"
        );

        let counts_4d: Vec<usize> = [12, 16, 24]
            .iter()
            .map(|&r| {
                Isovolume::extract(
                    [-TORUS_BOUND; 4],
                    [TORUS_BOUND; 4],
                    r,
                    torus_4d(TORUS_MAJOR, TORUS_MINOR),
                )
                .piece_count()
            })
            .collect();
        assert!(
            counts_4d[0] <= 97 && counts_4d[1] <= 131 && counts_4d[2] <= 440,
            "4D piece counts {counts_4d:?} regressed"
        );
    }

    #[test]
    fn piece_count_is_far_below_the_occupied_cell_count() {
        for resolution in [16, 32, 64] {
            let volume = Isovolume::extract([-1.5; 3], [1.5; 3], resolution, sphere_3d(1.0));
            assert!(
                volume.piece_count() * 20 < volume.occupied_cells(),
                "resolution {resolution}: {} pieces for {} cells",
                volume.piece_count(),
                volume.occupied_cells()
            );
        }
    }

    #[test]
    fn extraction_is_reproducible() {
        let a = extract_torus_3d(24);
        let b = extract_torus_3d(24);
        assert_eq!(a.piece_count(), b.piece_count());
        assert_eq!(a.occupied_cells(), b.occupied_cells());
        for i in 0..a.piece_count() {
            assert_eq!(a.piece_bounds(i), b.piece_bounds(i));
        }
    }

    #[test]
    fn bounds_that_cut_the_solid_are_reported_as_clipped() {
        let tight = Isovolume::extract([-0.8; 3], [0.8; 3], 24, torus_3d(TORUS_MAJOR, TORUS_MINOR));
        assert!(tight.clipped());
        assert!(!extract_torus_3d(24).clipped());
    }

    #[test]
    fn a_field_with_no_solid_yields_no_pieces() {
        let empty = Isovolume::extract([-1.0; 3], [1.0; 3], 8, |_| 10.0);
        assert_eq!(empty.piece_count(), 0);
        assert_eq!(empty.occupied_cells(), 0);
        assert_eq!(empty.volume(), 0.0);
        assert!(empty.colliders().is_empty());
        assert!(!empty.contains([0.0; 3]));
    }

    #[test]
    fn colliders_are_origin_centred_hulls_matching_their_piece_bounds() {
        let volume = extract_torus_3d(16);
        let colliders = volume.colliders();
        assert_eq!(colliders.len(), volume.piece_count());
        for (i, (centre, shape)) in colliders.iter().enumerate() {
            let Shape::ConvexPolytope3D { vertices } = shape else {
                panic!("expected ConvexPolytope3D, got {:?}", shape.kind());
            };
            assert_eq!(vertices.len(), 8);
            let (lo, hi) = volume.piece_bounds(i);
            let expect = 0.5 * (Vec3::from_array(lo) + Vec3::from_array(hi));
            assert!(centre.distance(expect) < 1.0e-5);
            let sum: Vec3 = vertices.iter().copied().sum();
            assert!(sum.length() < 1.0e-5, "hull is not origin-centred");
            let half = 0.5 * (Vec3::from_array(hi) - Vec3::from_array(lo));
            for v in vertices {
                assert!((v.abs() - half).abs().max_element() < 1.0e-5);
            }
        }
    }

    #[test]
    fn colliders_4d_are_origin_centred_sixteen_vertex_hulls() {
        let volume = Isovolume::extract(
            [-TORUS_BOUND; 4],
            [TORUS_BOUND; 4],
            10,
            torus_4d(TORUS_MAJOR, TORUS_MINOR),
        );
        let colliders = volume.colliders();
        assert_eq!(colliders.len(), volume.piece_count());
        assert!(volume.piece_count() > 0);
        for (i, (centre, shape)) in colliders.iter().enumerate() {
            let Shape::ConvexPolytope4D { vertices } = shape else {
                panic!("expected ConvexPolytope4D, got {:?}", shape.kind());
            };
            assert_eq!(vertices.len(), 16);
            let (lo, hi) = volume.piece_bounds(i);
            let expect = 0.5 * (Vec4::from_array(lo) + Vec4::from_array(hi));
            assert!(centre.distance(expect) < 1.0e-5);
            let half = 0.5 * (Vec4::from_array(hi) - Vec4::from_array(lo));
            for v in vertices {
                assert!((v.abs() - half).abs().max_element() < 1.0e-5);
            }
        }
    }

    #[test]
    fn every_interior_probe_lies_inside_a_piece_in_4d() {
        let sdf = torus_4d(TORUS_MAJOR, TORUS_MINOR);
        let volume = Isovolume::extract([-TORUS_BOUND; 4], [TORUS_BOUND; 4], 16, &sdf);
        assert!(!volume.clipped());
        const PROBES: usize = 21;
        let coord = |i: usize| -TORUS_BOUND + 2.0 * TORUS_BOUND * (i as f32 + 0.37) / PROBES as f32;
        let mut interior = 0;
        for iw in 0..PROBES {
            for iz in 0..PROBES {
                for iy in 0..PROBES {
                    for ix in 0..PROBES {
                        let p = [coord(ix), coord(iy), coord(iz), coord(iw)];
                        if sdf(p) > 0.0 {
                            continue;
                        }
                        interior += 1;
                        assert!(volume.contains(p), "interior probe {p:?} is not covered");
                    }
                }
            }
        }
        assert!(interior > 1000, "only {interior} interior probes");
    }
}
