use std::sync::LazyLock;

use glam::{Vec3, Vec4};

use crate::polytope_geom::{
    cell120_vertices, cell16_vertices, cell24_vertices, cell600_vertices, pentatope_vertices,
    tesseract_vertices,
};

/// Discriminants match `loam_render::raymarch::SHAPE_*`, so one `u32` drives
/// both the kernel and the topology lookup.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Polytope4 {
    Pentatope = 0,
    Tesseract = 1,
    Cell16 = 2,
    Cell24 = 3,
    Cell120 = 4,
    Cell600 = 5,
}

#[derive(Debug)]
pub struct Polytope4Topology {
    /// Canonical (unit-circumradius) coordinates.
    pub vertices: &'static [Vec4],
    /// `[lo, hi]` pairs, lexicographically sorted for deterministic iteration.
    pub edges: &'static [[u32; 2]],
    /// Ascending index lists, lexicographically sorted for the same reason.
    pub cells: &'static [&'static [u32]],
}

impl Polytope4 {
    /// In `repr(u32)` discriminant order.
    pub const ALL: [Polytope4; 6] = [
        Polytope4::Pentatope,
        Polytope4::Tesseract,
        Polytope4::Cell16,
        Polytope4::Cell24,
        Polytope4::Cell120,
        Polytope4::Cell600,
    ];

    pub fn topology(self) -> &'static Polytope4Topology {
        match self {
            Polytope4::Pentatope => &PENTATOPE_TOPOLOGY,
            Polytope4::Tesseract => &TESSERACT_TOPOLOGY,
            Polytope4::Cell16 => &CELL16_TOPOLOGY,
            Polytope4::Cell24 => &CELL24_TOPOLOGY,
            Polytope4::Cell120 => &CELL120_TOPOLOGY,
            Polytope4::Cell600 => &CELL600_TOPOLOGY,
        }
    }

    pub fn vertex_count(self) -> usize {
        self.topology().vertices.len()
    }

    /// Undirected edges, each counted once.
    pub fn edge_count(self) -> usize {
        self.topology().edges.len()
    }

    /// Bounding 3-cells (the facets), not the full face lattice.
    pub fn cell_count(self) -> usize {
        self.topology().cells.len()
    }

    /// Each centroid has length equal to the inradius and points along the
    /// cell's outward face normal.
    pub fn cell_centers(self) -> Vec<Vec4> {
        let topo = self.topology();
        topo.cells
            .iter()
            .map(|cell| {
                cell.iter()
                    .map(|&i| topo.vertices[i as usize])
                    .sum::<Vec4>()
                    / cell.len() as f32
            })
            .collect()
    }

    /// `(normals, inradius)`, shaped for
    /// [`crate::polytope_geom::polytope_sdf_wolfe`]. Unlike the dual-vertex
    /// helpers `cell120_face_planes` / `cell600_face_planes` (approximate on
    /// the 96 golden-ratio orbits; the documented BUG), normals derived from
    /// cell centroids are exact for every regular convex 4-polytope.
    pub fn face_planes(self) -> (Vec<Vec4>, f32) {
        let topo = self.topology();
        let mut normals = Vec::with_capacity(topo.cells.len());
        let mut inradius_sum = 0.0;
        for cell in topo.cells {
            let centroid: Vec4 = cell
                .iter()
                .map(|&i| topo.vertices[i as usize])
                .sum::<Vec4>()
                / cell.len() as f32;
            let r = centroid.length();
            // Average the inradius across all cells to absorb f32 noise rather
            // than read it off the first cell.
            normals.push(centroid / r);
            inradius_sum += r;
        }
        let inradius = inradius_sum / normals.len() as f32;
        (normals, inradius)
    }
}

const DEFAULT_LINE_COLOR: [f32; 4] = [0.9, 0.9, 0.9, 1.0];
const DEFAULT_LINE_WIDTH: f32 = 1.5;

impl crate::Visualizable<4> for Polytope4 {
    fn to_lines(&self) -> Result<crate::LineMesh<4>, crate::NotVisualizable> {
        let topo = self.topology();
        let mut mesh = crate::LineMesh::<4>::default();
        mesh.segments.reserve(topo.edges.len());
        mesh.colors.reserve(topo.edges.len());
        mesh.widths.reserve(topo.edges.len());
        for &[i, j] in topo.edges {
            let a = topo.vertices[i as usize].to_array();
            let b = topo.vertices[j as usize].to_array();
            mesh.segments.push((a, b));
            mesh.colors.push((DEFAULT_LINE_COLOR, DEFAULT_LINE_COLOR));
            mesh.widths.push(DEFAULT_LINE_WIDTH);
        }
        Ok(mesh)
    }

    fn to_triangles(&self) -> Result<crate::TriangleMesh<4>, crate::NotVisualizable> {
        // No 2-face incidence data is shipped, so there is nothing to fill.
        Err(crate::NotVisualizable::Degenerate)
    }

    fn to_points(&self) -> Result<crate::PointMesh<4>, crate::NotVisualizable> {
        let topo = self.topology();
        let mut mesh = crate::PointMesh::<4>::default();
        mesh.positions.reserve(topo.vertices.len());
        mesh.colors.reserve(topo.vertices.len());
        mesh.sizes.reserve(topo.vertices.len());
        for v in topo.vertices {
            mesh.positions.push(v.to_array());
            mesh.colors.push(DEFAULT_LINE_COLOR);
            mesh.sizes.push(4.0);
        }
        Ok(mesh)
    }
}

impl Polytope4 {
    /// Indexes `palette` modulo its length.
    pub fn lines_colored_by_cell(self, palette: &[[f32; 4]]) -> crate::LineMesh<4> {
        let topo = self.topology();
        let mut mesh = crate::LineMesh::<4>::default();
        let n_palette = palette.len().max(1);
        for &[i, j] in topo.edges {
            let cell_idx = topo
                .cells
                .iter()
                .position(|cell| cell.contains(&i) && cell.contains(&j))
                .unwrap_or(0);
            let color = palette[cell_idx % n_palette];
            mesh.segments.push((
                topo.vertices[i as usize].to_array(),
                topo.vertices[j as usize].to_array(),
            ));
            mesh.colors.push((color, color));
            mesh.widths.push(DEFAULT_LINE_WIDTH);
        }
        mesh
    }

    pub fn lines_colored_by_position(self) -> crate::LineMesh<4> {
        let topo = self.topology();
        let mut mesh = crate::LineMesh::<4>::default();
        mesh.segments.reserve(topo.edges.len());
        mesh.colors.reserve(topo.edges.len());
        mesh.widths.reserve(topo.edges.len());
        for &[i, j] in topo.edges {
            let va = topo.vertices[i as usize];
            let vb = topo.vertices[j as usize];
            mesh.segments.push((va.to_array(), vb.to_array()));
            mesh.colors
                .push((vertex_color_by_position(va), vertex_color_by_position(vb)));
            mesh.widths.push(DEFAULT_LINE_WIDTH);
        }
        mesh
    }
}

/// Normalized `xyz` drive R/G/B biased into `[0.25, 1.0]` so no vertex is
/// fully black; `w` modulates brightness in `[0.7, 1.0]` as a depth cue.
pub fn vertex_color_by_position(v: Vec4) -> [f32; 4] {
    let n = v.try_normalize().unwrap_or(Vec4::ZERO);
    let bias = |c: f32| 0.25 + 0.75 * (0.5 + 0.5 * c);
    let w_mod = 0.7 + 0.3 * (0.5 + 0.5 * n.w);
    [bias(n.x) * w_mod, bias(n.y) * w_mod, bias(n.z) * w_mod, 1.0]
}

// Alpha 0.55 keeps the surface behind the cross-section fill visible.
#[cfg(test)]
const SECTION_FILL_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 0.55];
const SECTION_EDGE_COLOR: [f32; 4] = [0.30, 0.85, 0.95, 1.0];
const SECTION_EDGE_WIDTH: f32 = 2.0;

#[cfg(test)]
fn polytope4_section_overlay(
    polytope: Polytope4,
    slice: loam_math::WPlane,
) -> (crate::TriangleMesh<3>, crate::LineMesh<3>) {
    let topo = polytope.topology();
    polytope_section_overlay_with_vertices(topo.edges, topo.cells, topo.vertices, slice)
}

#[cfg(test)]
fn polytope_section_overlay_with_vertices(
    edges: &[[u32; 2]],
    cells: &[&[u32]],
    vertices: &[Vec4],
    slice: loam_math::WPlane,
) -> (crate::TriangleMesh<3>, crate::LineMesh<3>) {
    let mut tri_mesh = crate::TriangleMesh::<3>::default();
    let mut edge_mesh = crate::LineMesh::<3>::default();
    let mut scratch = SectionScratch::default();

    for_each_section_cap(
        edges,
        cells,
        vertices,
        slice,
        &mut scratch,
        |ordered, centroid| {
            push_cap_fan(ordered, centroid, SECTION_FILL_COLOR, &mut tri_mesh);
            push_cap_perimeter(ordered, &mut edge_mesh);
        },
    );

    (tri_mesh, edge_mesh)
}

/// Caller-owned mesh and working set, so a per-frame overlay stops reaching the
/// allocator once both have grown.
pub fn polytope_section_perimeter_append(
    edges: &[[u32; 2]],
    cells: &[&[u32]],
    vertices: &[Vec4],
    slice: loam_math::WPlane,
    scratch: &mut SectionScratch,
    out: &mut crate::LineMesh<3>,
) {
    for_each_section_cap(edges, cells, vertices, slice, scratch, |ordered, _| {
        push_cap_perimeter(ordered, out);
    });
}

fn push_cap_fan(
    ordered: &[Vec3],
    centroid: Vec3,
    color: [f32; 4],
    out: &mut crate::TriangleMesh<3>,
) {
    let cv_base = out.vertices.len() as u32;
    out.vertices.push(centroid.to_array());
    out.colors.push(color);
    for cap_v in ordered {
        out.vertices.push(cap_v.to_array());
        out.colors.push(color);
    }
    let n = ordered.len() as u32;
    for k in 0..n {
        let k_next = (k + 1) % n;
        out.indices
            .push([cv_base, cv_base + 1 + k, cv_base + 1 + k_next]);
    }
}

// Adjacent caps share face-on-slice edges, drawn twice rather than paying for a
// global dedup pass.
fn push_cap_perimeter(ordered: &[Vec3], out: &mut crate::LineMesh<3>) {
    for k in 0..ordered.len() {
        let a = ordered[k];
        let b = ordered[(k + 1) % ordered.len()];
        out.segments.push((a.to_array(), b.to_array()));
        out.colors.push((SECTION_EDGE_COLOR, SECTION_EDGE_COLOR));
        out.widths.push(SECTION_EDGE_WIDTH);
    }
}

#[cfg(test)]
fn polytope_section_faces_with_vertices(
    edges: &[[u32; 2]],
    cells: &[&[u32]],
    vertices: &[Vec4],
    slice: loam_math::WPlane,
    color: [f32; 4],
) -> crate::TriangleMesh<3> {
    let mut tri_mesh = crate::TriangleMesh::<3>::default();
    let mut scratch = SectionScratch::default();
    polytope_section_faces_append(
        edges,
        cells,
        vertices,
        slice,
        color,
        &mut scratch,
        &mut tri_mesh,
    );
    tri_mesh
}

/// Indices are offset by the existing vertex count, so many bodies merge into
/// one reused scratch mesh.
pub fn polytope_section_faces_append(
    edges: &[[u32; 2]],
    cells: &[&[u32]],
    vertices: &[Vec4],
    slice: loam_math::WPlane,
    color: [f32; 4],
    scratch: &mut SectionScratch,
    out: &mut crate::TriangleMesh<3>,
) {
    for_each_section_cap(
        edges,
        cells,
        vertices,
        slice,
        scratch,
        |ordered, centroid| push_cap_fan(ordered, centroid, color, out),
    );
}

#[cfg(test)]
fn polytope4_section_faces(
    polytope: Polytope4,
    slice: loam_math::WPlane,
    color: [f32; 4],
) -> crate::TriangleMesh<3> {
    let topo = polytope.topology();
    polytope_section_faces_with_vertices(topo.edges, topo.cells, topo.vertices, slice, color)
}

/// Caller-held, so a per-frame section does not allocate three buffers per
/// crossed cell.
#[derive(Debug, Default)]
pub struct SectionScratch {
    cap: Vec<Vec3>,
    keys: Vec<(usize, f32)>,
    ordered: Vec<Vec3>,
}

// Degenerate caps (< 3 points, collinear) are skipped. `emit` runs in topology
// cell order, so mesh output is deterministic across runs.
fn for_each_section_cap(
    edges: &[[u32; 2]],
    cells: &[&[u32]],
    vertices: &[Vec4],
    slice: loam_math::WPlane,
    scratch: &mut SectionScratch,
    mut emit: impl FnMut(&[Vec3], Vec3),
) {
    let slice = perturb_slice_if_needed(slice, vertices);

    // Reference "inside" point so each cap's fan-triangle winding can be
    // oriented outward.
    let polytope_center_r3: Vec3 = if vertices.is_empty() {
        Vec3::ZERO
    } else {
        let mean: Vec4 = vertices.iter().copied().sum::<Vec4>() / vertices.len() as f32;
        Vec3::new(mean.x, mean.y, mean.z)
    };

    for cell in cells {
        // Per-cell w-range pruning: the load-bearing optimization for the
        // 600-cell (~430K naive ops -> ~3K typical).
        let (w_min, w_max) = cell_w_range(cell, vertices);
        if w_max < slice.w_slice - loam_math::SLICE_PERTURBATION_EPSILON
            || w_min > slice.w_slice + loam_math::SLICE_PERTURBATION_EPSILON
        {
            continue;
        }

        // Cell edges are parent edges restricted to the cell's vertex set,
        // which avoids needing per-cell 2-face incidence data.
        let cap = &mut scratch.cap;
        cap.clear();
        for &[i, j] in edges {
            if !cell.contains(&i) || !cell.contains(&j) {
                continue;
            }
            if let Some((_, p3)) =
                <loam_math::EuclideanR4 as loam_math::SectionableSpace<4>>::edge_section(
                    &slice,
                    vertices[i as usize],
                    vertices[j as usize],
                )
            {
                cap.push(p3);
            }
        }
        if cap.len() < 3 {
            continue;
        }

        let centroid: Vec3 = cap.iter().copied().sum::<Vec3>() / cap.len() as f32;
        let Some((basis_u, mut basis_v)) = fit_plane_basis(centroid, cap) else {
            continue;
        };

        // Flip `basis_v` so the face normal `u × v` points outward; without it
        // `fit_plane_basis`'s sign varies per cap, giving inconsistent winding.
        // A centroid at the center has no reference direction, so skip it.
        let outward = centroid - polytope_center_r3;
        let face_normal = basis_u.cross(basis_v);
        if outward.length_squared() > 1e-12 && face_normal.dot(outward) < 0.0 {
            basis_v = -basis_v;
        }

        order_around_centroid(
            &scratch.cap,
            centroid,
            basis_u,
            basis_v,
            &mut scratch.keys,
            &mut scratch.ordered,
        );

        emit(&scratch.ordered, centroid);
    }
}

// Shifting by SLICE_PERTURBATION_EPSILON clears vertex-on-slice,
// edge-in-plane, and face-graze degeneracies in one step.
fn perturb_slice_if_needed(slice: loam_math::WPlane, vertices: &[Vec4]) -> loam_math::WPlane {
    let eps = loam_math::SLICE_PERTURBATION_EPSILON;
    let near = vertices.iter().any(|v| (v.w - slice.w_slice).abs() < eps);
    if near {
        loam_math::WPlane::new(slice.w_slice + eps)
    } else {
        slice
    }
}

fn cell_w_range(cell: &[u32], vertices: &[Vec4]) -> (f32, f32) {
    let mut w_min = f32::INFINITY;
    let mut w_max = f32::NEG_INFINITY;
    for &i in cell {
        let w = vertices[i as usize].w;
        if w < w_min {
            w_min = w;
        }
        if w > w_max {
            w_max = w;
        }
    }
    (w_min, w_max)
}

// `None` means a collinear or point-coincident cap.
fn fit_plane_basis(centroid: Vec3, points: &[Vec3]) -> Option<(Vec3, Vec3)> {
    let eps = loam_math::EDGE_PARALLEL_EPSILON;
    let mut basis_u = Vec3::ZERO;
    for p in points {
        let off = *p - centroid;
        if off.length_squared() > eps * eps {
            basis_u = off.normalize();
            break;
        }
    }
    if basis_u == Vec3::ZERO {
        return None;
    }
    for p in points {
        let off = *p - centroid;
        let cross = basis_u.cross(off);
        if cross.length_squared() > eps * eps {
            let normal = cross.normalize();
            let basis_v = normal.cross(basis_u);
            return Some((basis_u, basis_v));
        }
    }
    None
}

// `keys` carries the angle per point so `atan2` runs once each rather than
// once per comparison.
fn order_around_centroid(
    points: &[Vec3],
    centroid: Vec3,
    basis_u: Vec3,
    basis_v: Vec3,
    keys: &mut Vec<(usize, f32)>,
    ordered: &mut Vec<Vec3>,
) {
    keys.clear();
    keys.extend(points.iter().enumerate().map(|(i, p)| {
        let off = *p - centroid;
        let angle = off.dot(basis_v).atan2(off.dot(basis_u));
        (i, angle)
    }));
    // Unstable so the sort cannot allocate: a stable sort falls back to a
    // scratch buffer above its insertion-sort threshold. Ties would be
    // coincident cap points, which the slice perturbation rules out.
    keys.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    ordered.clear();
    ordered.extend(keys.iter().map(|&(i, _)| points[i]));
}

// Leaked to `'static`, never freed, matching the LazyLock's process lifetime.

static PENTATOPE_VERTICES: LazyLock<&'static [Vec4]> =
    LazyLock::new(|| Box::leak(pentatope_vertices(1.0).into_boxed_slice()));
static TESSERACT_VERTICES: LazyLock<&'static [Vec4]> =
    LazyLock::new(|| Box::leak(tesseract_vertices(1.0).into_boxed_slice()));
static CELL16_VERTICES: LazyLock<&'static [Vec4]> =
    LazyLock::new(|| Box::leak(cell16_vertices(1.0).into_boxed_slice()));
static CELL24_VERTICES: LazyLock<&'static [Vec4]> =
    LazyLock::new(|| Box::leak(cell24_vertices(1.0).into_boxed_slice()));
static CELL120_VERTICES: LazyLock<&'static [Vec4]> =
    LazyLock::new(|| Box::leak(cell120_vertices(1.0).into_boxed_slice()));
static CELL600_VERTICES: LazyLock<&'static [Vec4]> =
    LazyLock::new(|| Box::leak(cell600_vertices(1.0).into_boxed_slice()));

// Coxeter, Regular Polytopes, Table I, cross-checked against Wikipedia and the
// empirical min pairwise distance.
//
// The 120-cell value is non-obvious: Wikipedia's `3 − √5` is at circumradius
// `2√2` (the convention of `polytope_geom::cell120_vertices`), so unit
// circumradius gives `(3 − √5)/(2√2) = 1/(φ²·√2)`. Dropping the `√2` is the
// 600-cell dual's identity, not the 120-cell's.
fn canonical_edge_length(p: Polytope4) -> f32 {
    let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let sqrt2 = 2.0_f32.sqrt();
    match p {
        Polytope4::Pentatope => (5.0_f32 / 2.0).sqrt(),
        Polytope4::Tesseract => 1.0,
        Polytope4::Cell16 => sqrt2,
        Polytope4::Cell24 => 1.0,
        Polytope4::Cell120 => 1.0 / (phi * phi * sqrt2),
        Polytope4::Cell600 => 1.0 / phi,
    }
}

// Absorbs f32 accumulation (~1e-5 at unit scale) while staying far below the
// gap to the next-shortest chord; tightest on the 120-cell (edge 0.382, next
// chord ~0.627), where 1e-4 leaves a 4x margin either side.
const EDGE_TOLERANCE: f32 = 1e-4;

fn derive_edges(vertices: &[Vec4], edge_length: f32) -> Vec<[u32; 2]> {
    let mut edges = Vec::new();
    for i in 0..vertices.len() {
        for j in (i + 1)..vertices.len() {
            let d = (vertices[i] - vertices[j]).length();
            if (d - edge_length).abs() < EDGE_TOLERANCE {
                edges.push([i as u32, j as u32]);
            }
        }
    }
    edges
}

// Takes the vertex slice directly, not a `Polytope4`, so the `EDGES` LazyLock
// cannot recursively re-enter `TOPOLOGY` during init.
fn cache_edges(vertices: &'static [Vec4], edge_length: f32) -> &'static [[u32; 2]] {
    Box::leak(derive_edges(vertices, edge_length).into_boxed_slice())
}

static PENTATOPE_EDGES: LazyLock<&'static [[u32; 2]]> = LazyLock::new(|| {
    cache_edges(
        *PENTATOPE_VERTICES,
        canonical_edge_length(Polytope4::Pentatope),
    )
});
static TESSERACT_EDGES: LazyLock<&'static [[u32; 2]]> = LazyLock::new(|| {
    cache_edges(
        *TESSERACT_VERTICES,
        canonical_edge_length(Polytope4::Tesseract),
    )
});
static CELL16_EDGES: LazyLock<&'static [[u32; 2]]> =
    LazyLock::new(|| cache_edges(*CELL16_VERTICES, canonical_edge_length(Polytope4::Cell16)));
static CELL24_EDGES: LazyLock<&'static [[u32; 2]]> =
    LazyLock::new(|| cache_edges(*CELL24_VERTICES, canonical_edge_length(Polytope4::Cell24)));
static CELL120_EDGES: LazyLock<&'static [[u32; 2]]> =
    LazyLock::new(|| cache_edges(*CELL120_VERTICES, canonical_edge_length(Polytope4::Cell120)));
static CELL600_EDGES: LazyLock<&'static [[u32; 2]]> =
    LazyLock::new(|| cache_edges(*CELL600_VERTICES, canonical_edge_length(Polytope4::Cell600)));

// Cells are fit against the polytope's own edge graph, not an external dual:
// the 120-cell and 600-cell generators in `polytope_geom` are not in
// mutually-dual orientation (their 96 golden-ratio vertices differ), so a
// dual's vertices are not the other's face normals.

// Vertex on a cell's 3-flat. f32 noise on `Vec4::dot` is ~5e-7, so 1e-4 leaves
// ~100x margin while still rejecting adjacent-cell vertices (the `n · v` spread
// is order 0.1, tightest on the 120-cell).
const CELL_TOLERANCE: f32 = 1e-4;

// The acceptance filter on a trial 3-flat fit: a non-cell 3-flat through 4
// points generically holds only those 4, a true cell's holds the full cell.
const fn cell_vertex_count(p: Polytope4) -> usize {
    match p {
        Polytope4::Pentatope => 4,
        Polytope4::Tesseract => 8,
        Polytope4::Cell16 => 4,
        Polytope4::Cell24 => 6,
        Polytope4::Cell120 => 20,
        Polytope4::Cell600 => 4,
    }
}

// Component `i` is the signed 3x3 minor of `[a; b; c]` with column `i` dropped
// (cofactor expansion), each minor computed as a triple product.
fn cross4(a: Vec4, b: Vec4, c: Vec4) -> Vec4 {
    let drop_x = |v: Vec4| Vec3::new(v.y, v.z, v.w);
    let drop_y = |v: Vec4| Vec3::new(v.x, v.z, v.w);
    let drop_z = |v: Vec4| Vec3::new(v.x, v.y, v.w);
    let drop_w = |v: Vec4| Vec3::new(v.x, v.y, v.z);
    Vec4::new(
        drop_x(a).dot(drop_x(b).cross(drop_x(c))),
        -drop_y(a).dot(drop_y(b).cross(drop_y(c))),
        drop_z(a).dot(drop_z(b).cross(drop_z(c))),
        -drop_w(a).dot(drop_w(b).cross(drop_w(c))),
    )
}

// `adj[i]` is in deterministic (lexicographic edge) order.
fn adjacency(num_vertices: usize, edges: &[[u32; 2]]) -> Vec<Vec<u32>> {
    let mut adj = vec![Vec::new(); num_vertices];
    for &[i, j] in edges {
        adj[i as usize].push(j);
        adj[j as usize].push(i);
    }
    adj
}

// Below this the three difference vectors are near-dependent. Distinct from
// CELL_TOLERANCE: that bounds on-plane membership in dot units, this bounds
// normal magnitude in vector-length units.
const MIN_CROSS4_LENGTH: f32 = 1e-4;

// Cost `O(V · D³ · V)` (vertices `V`, vertex-figure valence `D`), worst case
// ~3.2M plane scans on the 600-cell. Output is lexicographically sorted.
fn derive_cells(vertices: &[Vec4], edges: &[[u32; 2]], cell_size: usize) -> Vec<Vec<u32>> {
    use std::collections::BTreeSet;

    let adj = adjacency(vertices.len(), edges);
    let mut cells_set: BTreeSet<Vec<u32>> = BTreeSet::new();
    for v_idx in 0..vertices.len() {
        let v_0 = vertices[v_idx];
        let neighbors = &adj[v_idx];
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                for k in (j + 1)..neighbors.len() {
                    let n_a = vertices[neighbors[i] as usize];
                    let n_b = vertices[neighbors[j] as usize];
                    let n_c = vertices[neighbors[k] as usize];
                    let normal = cross4(n_a - v_0, n_b - v_0, n_c - v_0);
                    let mag = normal.length();
                    if mag < MIN_CROSS4_LENGTH {
                        continue;
                    }
                    let n = normal / mag;
                    let offset = v_0.dot(n);
                    let on_plane: Vec<u32> = (0..vertices.len() as u32)
                        .filter(|&p| (vertices[p as usize].dot(n) - offset).abs() < CELL_TOLERANCE)
                        .collect();
                    if on_plane.len() == cell_size {
                        cells_set.insert(on_plane);
                    }
                }
            }
        }
    }
    cells_set.into_iter().collect()
}

// Like `cache_edges`, takes data slices directly so it cannot re-enter the
// LazyLock during init.
fn cache_cells(
    vertices: &'static [Vec4],
    edges: &'static [[u32; 2]],
    cell_size: usize,
) -> &'static [&'static [u32]] {
    let cells: Vec<&'static [u32]> = derive_cells(vertices, edges, cell_size)
        .into_iter()
        .map(|c| &*Box::leak(c.into_boxed_slice()))
        .collect();
    Box::leak(cells.into_boxed_slice())
}

static PENTATOPE_CELLS: LazyLock<&'static [&'static [u32]]> = LazyLock::new(|| {
    cache_cells(
        *PENTATOPE_VERTICES,
        *PENTATOPE_EDGES,
        cell_vertex_count(Polytope4::Pentatope),
    )
});
static TESSERACT_CELLS: LazyLock<&'static [&'static [u32]]> = LazyLock::new(|| {
    cache_cells(
        *TESSERACT_VERTICES,
        *TESSERACT_EDGES,
        cell_vertex_count(Polytope4::Tesseract),
    )
});
static CELL16_CELLS: LazyLock<&'static [&'static [u32]]> = LazyLock::new(|| {
    cache_cells(
        *CELL16_VERTICES,
        *CELL16_EDGES,
        cell_vertex_count(Polytope4::Cell16),
    )
});
static CELL24_CELLS: LazyLock<&'static [&'static [u32]]> = LazyLock::new(|| {
    cache_cells(
        *CELL24_VERTICES,
        *CELL24_EDGES,
        cell_vertex_count(Polytope4::Cell24),
    )
});
static CELL120_CELLS: LazyLock<&'static [&'static [u32]]> = LazyLock::new(|| {
    cache_cells(
        *CELL120_VERTICES,
        *CELL120_EDGES,
        cell_vertex_count(Polytope4::Cell120),
    )
});
static CELL600_CELLS: LazyLock<&'static [&'static [u32]]> = LazyLock::new(|| {
    cache_cells(
        *CELL600_VERTICES,
        *CELL600_EDGES,
        cell_vertex_count(Polytope4::Cell600),
    )
});

static PENTATOPE_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *PENTATOPE_VERTICES,
    edges: *PENTATOPE_EDGES,
    cells: *PENTATOPE_CELLS,
});
static TESSERACT_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *TESSERACT_VERTICES,
    edges: *TESSERACT_EDGES,
    cells: *TESSERACT_CELLS,
});
static CELL16_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *CELL16_VERTICES,
    edges: *CELL16_EDGES,
    cells: *CELL16_CELLS,
});
static CELL24_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *CELL24_VERTICES,
    edges: *CELL24_EDGES,
    cells: *CELL24_CELLS,
});
static CELL120_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *CELL120_VERTICES,
    edges: *CELL120_EDGES,
    cells: *CELL120_CELLS,
});
static CELL600_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *CELL600_VERTICES,
    edges: *CELL600_EDGES,
    cells: *CELL600_CELLS,
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_counts_match_f_vector() {
        assert_eq!(Polytope4::Pentatope.vertex_count(), 5);
        assert_eq!(Polytope4::Tesseract.vertex_count(), 16);
        assert_eq!(Polytope4::Cell16.vertex_count(), 8);
        assert_eq!(Polytope4::Cell24.vertex_count(), 24);
        assert_eq!(Polytope4::Cell120.vertex_count(), 600);
        assert_eq!(Polytope4::Cell600.vertex_count(), 120);
    }

    #[test]
    fn vertices_on_unit_circumradius() {
        for p in Polytope4::ALL {
            for v in p.topology().vertices {
                let r = v.length();
                assert!(
                    (r - 1.0).abs() < 1e-5,
                    "{p:?} vertex {v:?} has |v| = {r}, expected 1.0"
                );
            }
        }
    }

    #[test]
    fn edge_counts_match_f_vector() {
        assert_eq!(Polytope4::Pentatope.edge_count(), 10);
        assert_eq!(Polytope4::Tesseract.edge_count(), 32);
        assert_eq!(Polytope4::Cell16.edge_count(), 24);
        assert_eq!(Polytope4::Cell24.edge_count(), 96);
        assert_eq!(Polytope4::Cell120.edge_count(), 1200);
        assert_eq!(Polytope4::Cell600.edge_count(), 720);
    }

    #[test]
    fn edge_lengths_match_canonical() {
        for p in Polytope4::ALL {
            let expected = canonical_edge_length(p);
            let t = p.topology();
            for &[i, j] in t.edges {
                let d = (t.vertices[i as usize] - t.vertices[j as usize]).length();
                assert!(
                    (d - expected).abs() < EDGE_TOLERANCE,
                    "{p:?} edge ({i}, {j}) length = {d}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn edge_pairs_in_min_max_order() {
        for p in Polytope4::ALL {
            for &[i, j] in p.topology().edges {
                assert!(i < j, "{p:?} edge ({i}, {j}) not in (min, max) order");
            }
        }
    }

    #[test]
    fn edges_are_unique() {
        for p in Polytope4::ALL {
            let edges = p.topology().edges;
            let mut seen = std::collections::HashSet::new();
            for &[i, j] in edges {
                let key = (i.min(j), i.max(j));
                assert!(seen.insert(key), "{p:?} edge ({i}, {j}) duplicated");
            }
        }
    }

    #[test]
    fn canonical_edge_length_matches_empirical_min() {
        for p in Polytope4::ALL {
            let vs = p.topology().vertices;
            let mut min_d = f32::INFINITY;
            for i in 0..vs.len() {
                for j in (i + 1)..vs.len() {
                    let d = (vs[i] - vs[j]).length();
                    if d < min_d {
                        min_d = d;
                    }
                }
            }
            let expected = canonical_edge_length(p);
            assert!(
                (min_d - expected).abs() < EDGE_TOLERANCE,
                "{p:?}: empirical min pairwise distance {min_d} != canonical {expected}"
            );
        }
    }

    #[test]
    fn cell_counts_match_f_vector() {
        assert_eq!(Polytope4::Pentatope.cell_count(), 5);
        assert_eq!(Polytope4::Tesseract.cell_count(), 8);
        assert_eq!(Polytope4::Cell16.cell_count(), 16);
        assert_eq!(Polytope4::Cell24.cell_count(), 24);
        assert_eq!(Polytope4::Cell120.cell_count(), 120);
        assert_eq!(Polytope4::Cell600.cell_count(), 600);
    }

    #[test]
    fn cell_vertex_counts_match_shape() {
        let cases: &[(Polytope4, usize)] = &[
            (Polytope4::Pentatope, 4),
            (Polytope4::Tesseract, 8),
            (Polytope4::Cell16, 4),
            (Polytope4::Cell24, 6),
            (Polytope4::Cell120, 20),
            (Polytope4::Cell600, 4),
        ];
        for &(p, expected) in cases {
            for (i, cell) in p.topology().cells.iter().enumerate() {
                assert_eq!(
                    cell.len(),
                    expected,
                    "{p:?} cell {i} has {} vertices, expected {expected}",
                    cell.len()
                );
            }
        }
    }

    #[test]
    fn cells_lie_on_common_hyperplane() {
        for p in Polytope4::ALL {
            let topo = p.topology();
            for (idx, cell) in topo.cells.iter().enumerate() {
                let centroid: Vec4 = cell
                    .iter()
                    .map(|&i| topo.vertices[i as usize])
                    .fold(Vec4::ZERO, |acc, v| acc + v)
                    / cell.len() as f32;
                let dots: Vec<f32> = cell
                    .iter()
                    .map(|&i| topo.vertices[i as usize].dot(centroid))
                    .collect();
                let lo = dots.iter().copied().fold(f32::INFINITY, f32::min);
                let hi = dots.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                assert!(
                    hi - lo < CELL_TOLERANCE,
                    "{p:?} cell {idx} centroid-projection spread {} > {CELL_TOLERANCE}",
                    hi - lo
                );
            }
        }
    }

    #[test]
    fn cell_internal_edge_counts_match_shape() {
        let cases: &[(Polytope4, usize)] = &[
            (Polytope4::Pentatope, 6),
            (Polytope4::Tesseract, 12),
            (Polytope4::Cell16, 6),
            (Polytope4::Cell24, 12),
            (Polytope4::Cell120, 30),
            (Polytope4::Cell600, 6),
        ];
        for &(p, expected) in cases {
            let topo = p.topology();
            let edge_len = canonical_edge_length(p);
            for (idx, cell) in topo.cells.iter().enumerate() {
                let mut count = 0;
                for i in 0..cell.len() {
                    for j in (i + 1)..cell.len() {
                        let a = topo.vertices[cell[i] as usize];
                        let b = topo.vertices[cell[j] as usize];
                        if ((a - b).length() - edge_len).abs() < EDGE_TOLERANCE {
                            count += 1;
                        }
                    }
                }
                assert_eq!(
                    count, expected,
                    "{p:?} cell {idx} has {count} internal edges, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn every_edge_in_at_least_two_cells() {
        for p in Polytope4::ALL {
            let topo = p.topology();
            for &[i, j] in topo.edges {
                let count = topo
                    .cells
                    .iter()
                    .filter(|cell| cell.contains(&i) && cell.contains(&j))
                    .count();
                assert!(
                    count >= 2,
                    "{p:?} edge ({i}, {j}) is in only {count} cell(s), expected >= 2"
                );
            }
        }
    }

    #[test]
    fn cross4_is_orthogonal_to_inputs() {
        let a = Vec4::new(1.0, 0.5, -0.3, 0.7);
        let b = Vec4::new(-0.2, 1.0, 0.4, -0.1);
        let c = Vec4::new(0.6, -0.8, 1.0, 0.3);
        let n = cross4(a, b, c);
        assert!(
            n.length() > 0.1,
            "cross4 of linearly-independent inputs is near-zero (|n| = {})",
            n.length()
        );
        for (label, v) in [("a", a), ("b", b), ("c", c)] {
            assert!(
                n.dot(v).abs() < 1e-5,
                "cross4 result not orthogonal to {label}: n·{label} = {}",
                n.dot(v)
            );
        }
    }

    #[test]
    fn cross4_zero_for_linearly_dependent_inputs() {
        let a = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let b = Vec4::new(0.0, 1.0, 0.0, 0.0);
        let c = a * 2.0 + b * 3.0;
        let n = cross4(a, b, c);
        assert!(
            n.length() < 1e-5,
            "cross4 of linearly-dependent inputs is not zero: {n:?} (|n| = {})",
            n.length()
        );
    }

    #[test]
    fn euler_poincare_relation_holds() {
        // (Polytope, F = number of 2-faces). V/E/C come from `.topology()`.
        let face_counts: &[(Polytope4, i64)] = &[
            (Polytope4::Pentatope, 10),
            (Polytope4::Tesseract, 24),
            (Polytope4::Cell16, 32),
            (Polytope4::Cell24, 96),
            (Polytope4::Cell120, 720),
            (Polytope4::Cell600, 1200),
        ];
        for &(p, f) in face_counts {
            let v = p.vertex_count() as i64;
            let e = p.edge_count() as i64;
            let c = p.cell_count() as i64;
            assert_eq!(
                v - e + f - c,
                0,
                "{p:?} Euler-Poincaré: V({v}) - E({e}) + F({f}) - C({c}) != 0"
            );
        }
    }

    #[test]
    fn visualizable_line_count_matches_edge_count() {
        use crate::Visualizable;
        for p in Polytope4::ALL {
            let mesh = <Polytope4 as Visualizable<4>>::to_lines(&p)
                .expect("polytopes always produce line meshes");
            assert_eq!(
                mesh.segments.len(),
                p.edge_count(),
                "{p:?} line mesh has {} segments, expected {}",
                mesh.segments.len(),
                p.edge_count()
            );
            assert_eq!(mesh.colors.len(), mesh.segments.len());
            assert_eq!(mesh.widths.len(), mesh.segments.len());
        }
    }

    #[test]
    fn visualizable_line_endpoints_are_polytope_vertices() {
        use crate::Visualizable;
        let topo = Polytope4::Tesseract.topology();
        let mesh = <Polytope4 as Visualizable<4>>::to_lines(&Polytope4::Tesseract).unwrap();
        for (i, &[vi, vj]) in topo.edges.iter().enumerate() {
            let (a, b) = mesh.segments[i];
            assert_eq!(a, topo.vertices[vi as usize].to_array());
            assert_eq!(b, topo.vertices[vj as usize].to_array());
        }
    }

    #[test]
    fn visualizable_point_count_matches_vertex_count() {
        use crate::Visualizable;
        for p in Polytope4::ALL {
            let mesh = <Polytope4 as Visualizable<4>>::to_points(&p)
                .expect("polytopes always produce point meshes");
            assert_eq!(mesh.positions.len(), p.vertex_count());
            assert_eq!(mesh.colors.len(), mesh.positions.len());
            assert_eq!(mesh.sizes.len(), mesh.positions.len());
        }
    }

    #[test]
    fn visualizable_triangles_currently_not_visualizable() {
        use crate::Visualizable;
        for p in Polytope4::ALL {
            let result = <Polytope4 as Visualizable<4>>::to_triangles(&p);
            assert!(matches!(result, Err(crate::NotVisualizable::Degenerate)));
        }
    }

    #[test]
    fn lines_colored_by_cell_uses_palette() {
        let palette: &[[f32; 4]] = &[
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ];
        let mesh = Polytope4::Tesseract.lines_colored_by_cell(palette);
        assert_eq!(mesh.segments.len(), Polytope4::Tesseract.edge_count());
        for (start_color, end_color) in &mesh.colors {
            assert!(palette.contains(start_color));
            assert!(palette.contains(end_color));
        }
    }

    #[test]
    fn pentatope_section_at_midpoint() {
        let (tri, edges) =
            polytope4_section_overlay(Polytope4::Pentatope, loam_math::WPlane::new(0.0));
        assert_eq!(tri.indices.len(), 12, "expected 12 fan triangles");
        assert_eq!(edges.segments.len(), 12, "expected 12 perimeter segments");
        // 4 caps * (centroid + 3 cap points).
        assert_eq!(tri.vertices.len(), 16);
    }

    #[test]
    fn tesseract_section_at_midpoint_has_six_square_caps() {
        let (tri, edges) =
            polytope4_section_overlay(Polytope4::Tesseract, loam_math::WPlane::new(0.0));
        assert_eq!(tri.indices.len(), 24, "6 cubical cells * 4 fan-triangles");
        assert_eq!(edges.segments.len(), 24, "6 caps * 4 perimeter edges");
        // 6 caps * (centroid + 4 cap points).
        assert_eq!(tri.vertices.len(), 30);
    }

    #[test]
    fn section_outside_polytope_is_empty() {
        for polytope in Polytope4::ALL {
            let (tri, edges) = polytope4_section_overlay(polytope, loam_math::WPlane::new(2.0));
            assert!(
                tri.indices.is_empty(),
                "{polytope:?} above-vertex slice should yield no triangles"
            );
            assert!(
                edges.segments.is_empty(),
                "{polytope:?} above-vertex slice should yield no perimeter edges"
            );
        }
    }

    #[test]
    fn vertex_on_slice_is_perturbed_not_nan() {
        let (tri, edges) =
            polytope4_section_overlay(Polytope4::Pentatope, loam_math::WPlane::new(-0.25));
        for v in &tri.vertices {
            for component in v {
                assert!(component.is_finite(), "triangle vertex must be finite");
            }
        }
        for (a, b) in &edges.segments {
            for component in a.iter().chain(b.iter()) {
                assert!(component.is_finite(), "edge vertex must be finite");
            }
        }
    }

    #[test]
    fn midpoint_slice_is_non_empty_for_every_polytope() {
        for polytope in Polytope4::ALL {
            let (tri, edges) = polytope4_section_overlay(polytope, loam_math::WPlane::new(0.0));
            assert!(
                !tri.indices.is_empty(),
                "{polytope:?} midpoint slice should yield triangles"
            );
            assert!(
                !edges.segments.is_empty(),
                "{polytope:?} midpoint slice should yield perimeter edges"
            );
        }
    }

    #[test]
    fn section_faces_triangle_count_matches_section_triangles() {
        let probe_color = [0.5, 0.5, 0.5, 1.0];
        for polytope in Polytope4::ALL {
            let slice = loam_math::WPlane::new(0.1);
            let (overlay_tri, _) = polytope4_section_overlay(polytope, slice);
            let faces_tri = polytope4_section_faces(polytope, slice, probe_color);
            assert_eq!(
                faces_tri.indices.len(),
                overlay_tri.indices.len(),
                "{polytope:?}: section_faces triangle count must match polytope4_section_overlay"
            );
            assert_eq!(
                faces_tri.vertices.len(),
                overlay_tri.vertices.len(),
                "{polytope:?}: section_faces vertex count must match polytope4_section_overlay"
            );
        }
    }

    #[test]
    fn section_faces_use_supplied_color_uniformly() {
        let color = [0.95, 0.55, 0.30, 1.0];
        let mesh =
            polytope4_section_faces(Polytope4::Pentatope, loam_math::WPlane::new(0.0), color);
        assert!(!mesh.colors.is_empty(), "section faces must produce colors");
        for (i, c) in mesh.colors.iter().enumerate() {
            assert_eq!(
                *c, color,
                "section face vertex {i} has color {c:?}, expected {color:?}"
            );
        }
    }

    #[test]
    fn perimeter_append_matches_the_by_value_overlay_perimeter() {
        let mut scratch = SectionScratch::default();
        for polytope in Polytope4::ALL {
            for w in [-0.3_f32, 0.0, 0.42] {
                let slice = loam_math::WPlane::new(w);
                let topo = polytope.topology();
                let (_, expected) = polytope4_section_overlay(polytope, slice);

                let mut appended = crate::LineMesh::<3>::default();
                polytope_section_perimeter_append(
                    topo.edges,
                    topo.cells,
                    topo.vertices,
                    slice,
                    &mut scratch,
                    &mut appended,
                );

                assert_eq!(
                    appended.segments, expected.segments,
                    "{polytope:?} at w = {w}: appended perimeter geometry diverged"
                );
                assert_eq!(appended.colors, expected.colors);
                assert_eq!(appended.widths, expected.widths);
            }
        }
    }

    #[test]
    fn perimeter_append_concatenates_instead_of_replacing() {
        let slice = loam_math::WPlane::new(0.1);
        let topo = Polytope4::Cell24.topology();
        let mut scratch = SectionScratch::default();
        let mut mesh = crate::LineMesh::<3>::default();
        let append = |scratch: &mut SectionScratch, out: &mut crate::LineMesh<3>| {
            polytope_section_perimeter_append(
                topo.edges,
                topo.cells,
                topo.vertices,
                slice,
                scratch,
                out,
            );
        };

        append(&mut scratch, &mut mesh);
        let single = mesh.segments.len();
        assert!(single > 0, "the fixture emitted no perimeter");
        append(&mut scratch, &mut mesh);

        assert_eq!(mesh.segments.len(), 2 * single);
        assert_eq!(mesh.segments[..single], mesh.segments[single..]);
        assert_eq!(mesh.widths.len(), mesh.segments.len());
        assert_eq!(mesh.colors.len(), mesh.segments.len());
    }

    // Every section-perimeter vertex is on the parent's true surface, so a
    // correct SDF returns ~0 there. The 120/600-cell SDFs carry the documented
    // BUG on the 96 golden-ratio orbits; the tests below pin that divergence so
    // a future fix fires here.

    // Every perimeter vertex sits on the slice hyperplane, so its w is exactly
    // `slice.w_slice`.
    fn perimeter_vertices_4d(perim: &crate::LineMesh<3>, w: f32) -> Vec<Vec4> {
        let mut out = Vec::with_capacity(perim.segments.len() * 2);
        for (a, b) in &perim.segments {
            out.push(Vec4::new(a[0], a[1], a[2], w));
            out.push(Vec4::new(b[0], b[1], b[2], w));
        }
        out
    }

    #[test]
    fn cell120_section_perimeter_diverges_from_sdf_documenting_bug() {
        use crate::polytope_geom::{cell120_face_planes, polytope_sdf_wolfe};
        let slice = loam_math::WPlane::new(0.0);
        let (_, perim) = polytope4_section_overlay(Polytope4::Cell120, slice);
        let (normals, inradius) = cell120_face_planes();

        let mut max_dev: f32 = 0.0;
        for p4 in perimeter_vertices_4d(&perim, slice.w_slice) {
            let d = polytope_sdf_wolfe(p4, &normals, inradius).abs();
            if d > max_dev {
                max_dev = d;
            }
        }
        assert!(
            max_dev > 1e-3,
            "Cell120 perimeter agrees with SDF surface within {max_dev}; expected \
             measurable divergence from the documented BUG. Did `cell120_face_planes` \
             get fixed? If so, delete this test and the BUG comment."
        );
        assert!(
            max_dev < 0.1,
            "Cell120 SDF divergence {max_dev} exceeds the documented BUG window; \
             a face-normal regression may have widened the error."
        );
    }

    #[test]
    fn five_eight_sixteen_twentyfour_cell_section_perimeter_on_sdf_surface() {
        use crate::polytope_geom::polytope_sdf_wolfe;
        let cases = [
            Polytope4::Pentatope,
            Polytope4::Tesseract,
            Polytope4::Cell16,
            Polytope4::Cell24,
        ];
        // 1e-3 sits well above f32 noise from the SDF's Wolfe-greedy projection
        // (~1e-5 in practice) but tight enough to fire on any face-plane
        // approximation approaching the 120/600 BUG magnitudes (~1e-2).
        const TOL: f32 = 1e-3;
        let slice = loam_math::WPlane::new(0.0);
        for polytope in cases {
            let (_, perim) = polytope4_section_overlay(polytope, slice);
            let (normals, inradius) = polytope.face_planes();
            for p4 in perimeter_vertices_4d(&perim, slice.w_slice) {
                let d = polytope_sdf_wolfe(p4, &normals, inradius).abs();
                assert!(
                    d < TOL,
                    "{polytope:?}: perimeter vertex {p4:?} has |SDF| = {d}, expected < {TOL}; \
                     section and SDF disagree"
                );
            }
        }
    }

    #[test]
    fn cell600_section_perimeter_diverges_from_sdf_documenting_bug() {
        use crate::polytope_geom::{cell600_face_planes, polytope_sdf_wolfe};
        let slice = loam_math::WPlane::new(0.0);
        let (_, perim) = polytope4_section_overlay(Polytope4::Cell600, slice);
        let (normals, inradius) = cell600_face_planes();

        let mut max_dev: f32 = 0.0;
        for p4 in perimeter_vertices_4d(&perim, slice.w_slice) {
            let d = polytope_sdf_wolfe(p4, &normals, inradius).abs();
            if d > max_dev {
                max_dev = d;
            }
        }
        assert!(
            max_dev > 1e-3,
            "Cell600 perimeter agrees with SDF surface within {max_dev}; expected \
             measurable divergence from the documented BUG. Did `cell600_face_planes` \
             get fixed? If so, delete this test and the BUG comment."
        );
        assert!(
            max_dev < 0.1,
            "Cell600 SDF divergence {max_dev} exceeds the documented BUG window; \
             a face-normal regression may have widened the error."
        );
    }

    // Independent copy of the algorithm's internal `cell_w_range`; if the two
    // ever drift, the test that uses it fires.
    fn test_cell_w_range(cell: &[u32], vertices: &[Vec4]) -> (f32, f32) {
        cell.iter()
            .map(|&i| vertices[i as usize].w)
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), w| {
                (lo.min(w), hi.max(w))
            })
    }

    #[test]
    fn cell_pruning_matches_straddle_count() {
        // Grazing values (within `SLICE_PERTURBATION_EPSILON` of any vertex's w)
        // are avoided so the perturbation path cannot shift the slice between
        // the independent count and the algorithm's.
        let slices = [-0.7, -0.3, -0.1, 0.0, 0.1, 0.3, 0.7];
        let eps = loam_math::SLICE_PERTURBATION_EPSILON;

        for polytope in Polytope4::ALL {
            let topo = polytope.topology();
            for &w in &slices {
                let effective_w = if topo.vertices.iter().any(|v| (v.w - w).abs() < eps) {
                    w + eps
                } else {
                    w
                };
                let expected_caps: usize = topo
                    .cells
                    .iter()
                    .filter(|cell| {
                        let (lo, hi) = test_cell_w_range(cell, topo.vertices);
                        // Strict `<` matches the algorithm: a cell whose
                        // w_max == effective_w + eps produces no crossing edge,
                        // so its edge-section step yields nothing.
                        lo < effective_w && effective_w < hi
                    })
                    .count();

                let (tri, _) = polytope4_section_overlay(polytope, loam_math::WPlane::new(w));
                let actual_caps = tri.vertices.len().saturating_sub(tri.indices.len());

                assert_eq!(
                    actual_caps, expected_caps,
                    "{polytope:?} at slice w={w}: algorithm produced {actual_caps} caps, \
                     topology-derived straddle count expected {expected_caps}"
                );
            }
        }
    }

    #[test]
    fn section_face_normals_point_outward_from_polytope_center() {
        for polytope in Polytope4::ALL {
            // Polytope is centered at origin in canonical coordinates, so the
            // outward direction at any cap is the cap centroid itself.
            let center = Vec3::ZERO;
            for &slice_w in &[-0.5_f32, -0.2, 0.0, 0.2, 0.5] {
                let (mesh, _) =
                    polytope4_section_overlay(polytope, loam_math::WPlane::new(slice_w));
                for &[a, b, c] in &mesh.indices {
                    let va = Vec3::from(mesh.vertices[a as usize]);
                    let vb = Vec3::from(mesh.vertices[b as usize]);
                    let vc = Vec3::from(mesh.vertices[c as usize]);
                    let n = (vb - va).cross(vc - va);
                    if n.length_squared() < 1e-10 {
                        continue; // degenerate triangle; skip
                    }
                    let tri_centroid = (va + vb + vc) / 3.0;
                    let outward = tri_centroid - center;
                    if outward.length_squared() < 1e-10 {
                        continue; // triangle straddles polytope center; orientation ambiguous
                    }
                    assert!(
                        n.dot(outward) > 0.0,
                        "{polytope:?} at w={slice_w}: triangle ({va:?}, {vb:?}, {vc:?}) \
                         has inward-facing normal {n:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn section_under_random_rotors_stays_well_formed() {
        let mut state: u32 = 0x517_C0DE;
        let mut rand = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for polytope in Polytope4::ALL {
            let topo = polytope.topology();
            for _ in 0..16 {
                // `xyzw` is included for full Spin(4) coverage even though it is
                // zero for SO(4) rotations; normalisation absorbs it into the
                // unit-norm constraint.
                let rotor = loam_math::Rotor4 {
                    s: rand(),
                    xy: rand(),
                    xz: rand(),
                    xw: rand(),
                    yz: rand(),
                    yw: rand(),
                    zw: rand(),
                    xyzw: rand(),
                }
                .normalize();
                let rotated: Vec<Vec4> = {
                    use loam_math::Rotor as _;
                    topo.vertices.iter().map(|v| rotor.apply(*v)).collect()
                };
                // Rotors preserve circumradius, so a slice in `(-1, 1)` stays
                // inside any unit-circumradius polytope.
                let slice_w = rand() * 0.8;
                let slice = loam_math::WPlane::new(slice_w);
                let (tri, perim) =
                    polytope_section_overlay_with_vertices(topo.edges, topo.cells, &rotated, slice);

                for v in &tri.vertices {
                    for c in v {
                        assert!(c.is_finite(), "{polytope:?} tri vertex non-finite: {v:?}");
                    }
                }
                for (a, b) in &perim.segments {
                    for c in a.iter().chain(b.iter()) {
                        assert!(c.is_finite(), "{polytope:?} perim endpoint non-finite");
                    }
                }
                for &[i0, i1, i2] in &tri.indices {
                    let n = tri.vertices.len() as u32;
                    assert!(
                        i0 < n && i1 < n && i2 < n,
                        "{polytope:?} index out of bounds"
                    );
                }
                let (w_min, w_max) = rotated
                    .iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| {
                        (lo.min(v.w), hi.max(v.w))
                    });
                if slice_w > w_min + 0.05 && slice_w < w_max - 0.05 {
                    assert!(
                        !perim.segments.is_empty(),
                        "{polytope:?} slice w={slice_w} inside [{w_min}, {w_max}] but produced empty section"
                    );
                }
            }
        }
    }

    #[test]
    fn section_recomputes_when_w_slice_changes() {
        let (a, _) = polytope4_section_overlay(Polytope4::Pentatope, loam_math::WPlane::new(0.0));
        let (b, _) = polytope4_section_overlay(Polytope4::Pentatope, loam_math::WPlane::new(0.4));
        assert_ne!(
            a.vertices, b.vertices,
            "section at w=0.0 and w=0.4 must differ; result was identical, \
             suggesting a stale cache or incorrect slice parameter use"
        );
    }
}
