// Pentatope w-slice viewer.
//
// The pentatope is the 4D analogue of the tetrahedron: 5 vertices, 10
// edges, 10 triangular faces, 5 tetrahedral cells. Slicing it by a
// 3-hyperplane `w = w₀` produces a 3D convex polyhedron that morphs
// as `w₀` sweeps from −¼ (the "base" cell) up to +1 (the apex).
//
// At critical w₀ values the cross-section degenerates to one of the
// pentatope's tetrahedral cells; between those values it interpolates
// through triangular bipyramids and other shapes.
//
// Edit while the example runs; ShaderDb hot-reloads on save.

struct Uniforms {
    camera_pos: vec3<f32>,
    camera_forward: vec3<f32>,
    camera_right: vec3<f32>,
    camera_up: vec3<f32>,
    fov_y_tan: f32,
    resolution: vec2<f32>,
    time: f32,
    tick: f32,
    // params[0] = w₀ slice value
    // params[1] = highlight cell index (0..4) or 5 for "no highlight"
    w_slice: f32,
    highlight: f32,
    pad0: f32,
    pad1: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    // Big-triangle fullscreen quad.
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// ---- Pentatope vertices (regular 4-simplex, circumradius = 1) ----
// Same closed form as `pentatope_vertices(1.0)` in `rye-physics`:
// apex at +w, four base vertices at w = −¼ on a tetrahedron of
// circumradius √15/4 (so all 10 edge lengths match the apex-to-base
// edge length).
const T_BASE: f32 = 0.5590169943749475;  // √15/4 ÷ √3

fn pentatope_v(i: u32) -> vec4<f32> {
    switch (i) {
        case 0u: { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
        case 1u: { return vec4<f32>( T_BASE,  T_BASE,  T_BASE, -0.25); }
        case 2u: { return vec4<f32>( T_BASE, -T_BASE, -T_BASE, -0.25); }
        case 3u: { return vec4<f32>(-T_BASE,  T_BASE, -T_BASE, -0.25); }
        default: { return vec4<f32>(-T_BASE, -T_BASE,  T_BASE, -0.25); }
    }
}

// SDF of the cross-section polyhedron formed by slicing the pentatope
// at `w = w0`. The polyhedron is bounded by up to 5 planes — one per
// tetrahedral cell of the pentatope that the hyperplane crosses.
//
// For each cell (a tetrahedron, the convex hull of 4 of the 5
// pentatope vertices), we find which of its 6 edges straddle `w₀`,
// project those intersection points to 3D (drop the `w` component),
// and use any 3 of the resulting points to define the plane that this
// cell contributes to the cross-section's surface. The signed
// distance to the polyhedron is then `max` of the per-face signed
// distances — the standard "max of half-spaces" SDF for a convex
// solid. Only an upper bound for points well outside, but Lipschitz
// and exact at the surface, which is all the raymarch needs.
//
// Also returns the index of the cell that contributes the "closest"
// face (largest signed distance) for highlighting in the shading.
struct SliceHit {
    dist: f32,
    cell: u32,
};

fn slice_sdf(p: vec3<f32>, w0: f32) -> SliceHit {
    var dist: f32 = -1.0e9;
    var any_face: bool = false;
    var dominant_cell: u32 = 5u;

    // The 5 pentatope cells, each excluding one vertex.
    for (var ci: u32 = 0u; ci < 5u; ci = ci + 1u) {
        // Collect the 4 vertices of this cell into a local array.
        var cell_v: array<vec4<f32>, 4>;
        var n: i32 = 0;
        for (var vi: u32 = 0u; vi < 5u; vi = vi + 1u) {
            if (vi != ci) {
                cell_v[n] = pentatope_v(vi);
                n = n + 1;
            }
        }

        // 6 edges of the tetrahedron: (i, j) for 0 ≤ i < j < 4.
        var verts: array<vec3<f32>, 4>;
        var nv: i32 = 0;
        for (var i: i32 = 0; i < 4; i = i + 1) {
            for (var j: i32 = i + 1; j < 4; j = j + 1) {
                let va = cell_v[i];
                let vb = cell_v[j];
                let dwa = va.w - w0;
                let dwb = vb.w - w0;
                if (dwa * dwb < 0.0) {
                    // Edge crosses the slice hyperplane.
                    let t = dwa / (dwa - dwb);
                    let pmix = mix(va, vb, t);
                    if (nv < 4) {
                        verts[nv] = pmix.xyz;
                        nv = nv + 1;
                    }
                }
            }
        }

        if (nv >= 3) {
            // Plane normal from the first 3 cross-section vertices.
            let n_raw = cross(verts[1] - verts[0], verts[2] - verts[0]);
            let len_sq = dot(n_raw, n_raw);
            if (len_sq > 1.0e-12) {
                let n_unit = n_raw / sqrt(len_sq);
                // The pentatope's centroid is at origin; whichever
                // half-space `+n` ends up in for `verts[0]` is
                // outward (origin on opposite side = interior). When
                // origin is on the plane, the cell is degenerate and
                // we skip it.
                let plane_offset = dot(n_unit, verts[0]);
                if (abs(plane_offset) > 1.0e-6) {
                    var outward_normal: vec3<f32>;
                    var face_offset: f32;
                    if (plane_offset > 0.0) {
                        outward_normal = n_unit;
                        face_offset = plane_offset;
                    } else {
                        outward_normal = -n_unit;
                        face_offset = -plane_offset;
                    }
                    let face_dist = dot(p, outward_normal) - face_offset;
                    if (!any_face || face_dist > dist) {
                        dist = face_dist;
                        dominant_cell = ci;
                    }
                    any_face = true;
                }
            }
        }
    }

    if (!any_face) {
        // The hyperplane doesn't cross the pentatope at this `w0`;
        // ray sees nothing.
        return SliceHit(1.0e9, 5u);
    }
    return SliceHit(dist, dominant_cell);
}

// ---- Lighting helpers ----------------------------------------------

fn cell_color(cell: u32) -> vec3<f32> {
    // Five distinct hues so the user can see which tetrahedral cell
    // is contributing the dominant face.
    switch (cell) {
        case 0u: { return vec3<f32>(0.95, 0.45, 0.25); }   // warm orange
        case 1u: { return vec3<f32>(0.30, 0.75, 0.95); }   // sky blue
        case 2u: { return vec3<f32>(0.95, 0.85, 0.30); }   // amber yellow
        case 3u: { return vec3<f32>(0.55, 0.95, 0.50); }   // green
        case 4u: { return vec3<f32>(0.85, 0.45, 0.95); }   // magenta
        default: { return vec3<f32>(0.7,  0.7,  0.7);  }
    }
}

fn estimate_normal(p: vec3<f32>, w0: f32) -> vec3<f32> {
    let h = 0.0008;
    let dx = slice_sdf(p + vec3<f32>(h, 0.0, 0.0), w0).dist
           - slice_sdf(p - vec3<f32>(h, 0.0, 0.0), w0).dist;
    let dy = slice_sdf(p + vec3<f32>(0.0, h, 0.0), w0).dist
           - slice_sdf(p - vec3<f32>(0.0, h, 0.0), w0).dist;
    let dz = slice_sdf(p + vec3<f32>(0.0, 0.0, h), w0).dist
           - slice_sdf(p - vec3<f32>(0.0, 0.0, h), w0).dist;
    return normalize(vec3<f32>(dx, dy, dz));
}

fn sky(rd: vec3<f32>) -> vec3<f32> {
    let t = (rd.y + 1.0) * 0.5;
    return mix(vec3<f32>(0.04, 0.05, 0.10), vec3<f32>(0.10, 0.13, 0.22), t);
}

// ---- Fragment shader -----------------------------------------------

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    // Build view ray from screen UV.
    let uv = (frag_pos.xy / u.resolution) * 2.0 - vec2<f32>(1.0, 1.0);
    let aspect = u.resolution.x / u.resolution.y;
    let ndc = vec2<f32>(uv.x * aspect, -uv.y);  // flip y for screen coords
    let rd = normalize(
        u.camera_forward
        + u.camera_right * (ndc.x * u.fov_y_tan)
        + u.camera_up    * (ndc.y * u.fov_y_tan)
    );
    let ro = u.camera_pos;

    let w0 = u.w_slice;

    // Sphere-trace the cross-section's max-of-half-spaces SDF.
    var t: f32 = 0.0;
    let max_t = 25.0;
    var hit_cell: u32 = 5u;
    var hit = false;
    for (var i: i32 = 0; i < 128; i = i + 1) {
        let p = ro + rd * t;
        let s = slice_sdf(p, w0);
        if (s.dist < 0.001) {
            hit = true;
            hit_cell = s.cell;
            break;
        }
        t = t + max(s.dist, 0.005);
        if (t > max_t) { break; }
    }

    if (!hit) {
        return vec4<f32>(sky(rd), 1.0);
    }

    let p_hit = ro + rd * t;
    let n = estimate_normal(p_hit, w0);
    let light_dir = normalize(vec3<f32>(0.5, 0.85, 0.3));
    let lambert = max(dot(n, light_dir), 0.0);
    let ambient = 0.20;

    let base = cell_color(hit_cell);
    let highlighted = u.highlight >= 0.0 && u.highlight < 5.0
        && u32(u.highlight) == hit_cell;
    let tint = select(base, base * 1.4 + vec3<f32>(0.1, 0.1, 0.1), highlighted);
    let lit = tint * (ambient + lambert * 0.85);

    // Soft fog with distance for depth cue.
    let fog = 1.0 - exp(-t * 0.06);
    let final_color = mix(lit, sky(rd), fog * 0.45);
    return vec4<f32>(final_color, 1.0);
}
