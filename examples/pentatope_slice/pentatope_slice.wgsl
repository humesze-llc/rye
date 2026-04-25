// Pentatope w-slice viewer — live physics edition.
//
// Differences from the static viewer that preceded this:
//
// - The pentatope's 4D vertices are *not* hardcoded constants; they
//   come in via the uniform buffer each frame, transformed by the
//   physics body's position + Rotor4 orientation on the CPU.
// - There's a 4D ground at `y = 0` (a half-space whose normal is
//   `+y` in 4D, with `w` component zero). Its cross-section at any
//   `w = w₀` is the 3D half-space `y ≥ 0` — i.e. a horizontal floor
//   plane that doesn't move when you change `w₀`.
//
// Rendering: ray march the union of the pentatope cross-section and
// the floor plane. Pentatope faces are tinted by the cell that
// contributes them; floor is a neutral checkerboard so you can see
// the pentatope's shadow-like resting position.

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
    // params[2..4] reserved
    w_slice: f32,
    highlight: f32,
    pad0: f32,
    pad1: f32,
    // Pentatope world-space vertices, transformed each frame on the
    // CPU from local body coords by the body's Rotor4 orientation
    // and translated by its position. Five `vec4<f32>` = 80 bytes.
    pentatope_v: array<vec4<f32>, 5>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// SDF of the cross-section polyhedron formed by slicing the
// pentatope at `w = w₀`. See the pre-edit version of this file for
// the algorithm prose; the only delta here is reading `pentatope_v`
// from the uniform buffer instead of using literal constants.
struct SliceHit {
    dist: f32,
    cell: u32,
};

fn slice_sdf(p: vec3<f32>, w0: f32) -> SliceHit {
    var dist: f32 = -1.0e9;
    var any_face: bool = false;
    var dominant_cell: u32 = 5u;

    // First pass: collect every edge-crossing point and average them
    // for an interior reference. The 4D body-centroid projected to
    // xyz is *not* reliable here — once the body has rotated, that
    // point can sit outside the cross-section, flipping face
    // normals and making the SDF read huge regions as "inside."
    // Cross-section vertices are always inside the cross-section's
    // convex hull, so their mean is always interior.
    var interior_sum: vec3<f32> = vec3<f32>(0.0);
    var interior_count: i32 = 0;
    for (var ii: u32 = 0u; ii < 5u; ii = ii + 1u) {
        for (var jj: u32 = ii + 1u; jj < 5u; jj = jj + 1u) {
            let va = u.pentatope_v[ii];
            let vb = u.pentatope_v[jj];
            let dwa = va.w - w0;
            let dwb = vb.w - w0;
            if (dwa * dwb < 0.0) {
                let t = dwa / (dwa - dwb);
                interior_sum = interior_sum + mix(va, vb, t).xyz;
                interior_count = interior_count + 1;
            }
        }
    }
    if (interior_count < 3) {
        return SliceHit(1.0e9, 5u);
    }
    let interior = interior_sum / f32(interior_count);

    for (var ci: u32 = 0u; ci < 5u; ci = ci + 1u) {
        // Cell ci is the tetrahedron of all pentatope vertices except ci.
        var cell_v: array<vec4<f32>, 4>;
        var n: i32 = 0;
        for (var vi: u32 = 0u; vi < 5u; vi = vi + 1u) {
            if (vi != ci) {
                cell_v[n] = u.pentatope_v[vi];
                n = n + 1;
            }
        }

        // 6 cell edges → up to 4 cross-section vertices.
        var verts: array<vec3<f32>, 4>;
        var nv: i32 = 0;
        for (var i: i32 = 0; i < 4; i = i + 1) {
            for (var j: i32 = i + 1; j < 4; j = j + 1) {
                let va = cell_v[i];
                let vb = cell_v[j];
                let dwa = va.w - w0;
                let dwb = vb.w - w0;
                if (dwa * dwb < 0.0) {
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
            let n_raw = cross(verts[1] - verts[0], verts[2] - verts[0]);
            let len_sq = dot(n_raw, n_raw);
            if (len_sq > 1.0e-12) {
                let n_unit = n_raw / sqrt(len_sq);
                // Orient outward via the cross-section's own interior
                // (mean of edge-crossing points), computed in the
                // first pass above. Robust under any body rotation.
                let to_interior = interior - verts[0];
                var outward_normal: vec3<f32>;
                var face_offset: f32;
                if (dot(n_unit, to_interior) > 0.0) {
                    outward_normal = -n_unit;
                } else {
                    outward_normal = n_unit;
                }
                face_offset = dot(outward_normal, verts[0]);
                let face_dist = dot(p, outward_normal) - face_offset;
                if (!any_face || face_dist > dist) {
                    dist = face_dist;
                    dominant_cell = ci;
                }
                any_face = true;
            }
        }
    }

    if (!any_face) {
        return SliceHit(1.0e9, 5u);
    }
    return SliceHit(dist, dominant_cell);
}

// Combined scene: pentatope cross-section + 4D floor at y = 0. The
// floor's 4D normal is `+y` with w-component zero, so its cross-
// section at any w₀ is the 3D half-space `y ≥ 0`. SDF = `p.y`.
struct SceneHit {
    dist: f32,
    // 0..4: pentatope cell, 5: ground, 6: empty
    material: u32,
};

fn scene_sdf(p: vec3<f32>, w0: f32) -> SceneHit {
    let pent = slice_sdf(p, w0);
    let ground = p.y;
    if (pent.dist < ground) {
        return SceneHit(pent.dist, pent.cell);
    } else {
        return SceneHit(ground, 5u);
    }
}

// ---- Shading helpers ------------------------------------------------

fn cell_color(cell: u32) -> vec3<f32> {
    switch (cell) {
        case 0u: { return vec3<f32>(0.95, 0.45, 0.25); }
        case 1u: { return vec3<f32>(0.30, 0.75, 0.95); }
        case 2u: { return vec3<f32>(0.95, 0.85, 0.30); }
        case 3u: { return vec3<f32>(0.55, 0.95, 0.50); }
        case 4u: { return vec3<f32>(0.85, 0.45, 0.95); }
        default: { return vec3<f32>(0.7,  0.7,  0.7);  }
    }
}

fn ground_color(p: vec3<f32>) -> vec3<f32> {
    // Soft checkerboard, 1m squares. Helps depth perception.
    let g = floor(p.x) + floor(p.z);
    let alt = abs(g - 2.0 * floor(g * 0.5));
    let dark = vec3<f32>(0.18, 0.20, 0.24);
    let light = vec3<f32>(0.30, 0.32, 0.36);
    return mix(dark, light, alt);
}

fn estimate_normal(p: vec3<f32>, w0: f32) -> vec3<f32> {
    let h = 0.001;
    let dx = scene_sdf(p + vec3<f32>(h, 0.0, 0.0), w0).dist
           - scene_sdf(p - vec3<f32>(h, 0.0, 0.0), w0).dist;
    let dy = scene_sdf(p + vec3<f32>(0.0, h, 0.0), w0).dist
           - scene_sdf(p - vec3<f32>(0.0, h, 0.0), w0).dist;
    let dz = scene_sdf(p + vec3<f32>(0.0, 0.0, h), w0).dist
           - scene_sdf(p - vec3<f32>(0.0, 0.0, h), w0).dist;
    return normalize(vec3<f32>(dx, dy, dz));
}

fn sky(rd: vec3<f32>) -> vec3<f32> {
    let t = (rd.y + 1.0) * 0.5;
    return mix(vec3<f32>(0.04, 0.05, 0.10), vec3<f32>(0.10, 0.13, 0.22), t);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (frag_pos.xy / u.resolution) * 2.0 - vec2<f32>(1.0, 1.0);
    let aspect = u.resolution.x / u.resolution.y;
    let ndc = vec2<f32>(uv.x * aspect, -uv.y);
    let rd = normalize(
        u.camera_forward
        + u.camera_right * (ndc.x * u.fov_y_tan)
        + u.camera_up    * (ndc.y * u.fov_y_tan)
    );
    let ro = u.camera_pos;
    let w0 = u.w_slice;

    var t: f32 = 0.0;
    let max_t = 60.0;
    var hit_material: u32 = 6u;
    var hit = false;
    for (var i: i32 = 0; i < 192; i = i + 1) {
        let p = ro + rd * t;
        let s = scene_sdf(p, w0);
        if (s.dist < 0.001) {
            hit = true;
            hit_material = s.material;
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

    var base: vec3<f32>;
    if (hit_material == 5u) {
        base = ground_color(p_hit);
    } else {
        let c = cell_color(hit_material);
        let highlighted = u.highlight >= 0.0 && u.highlight < 5.0
            && u32(u.highlight) == hit_material;
        base = select(c, c * 1.4 + vec3<f32>(0.1, 0.1, 0.1), highlighted);
    }

    let lit = base * (ambient + lambert * 0.85);
    let fog = 1.0 - exp(-t * 0.05);
    let final_color = mix(lit, sky(rd), fog * 0.5);
    return vec4<f32>(final_color, 1.0);
}
