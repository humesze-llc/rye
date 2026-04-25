// Hypersphere `w`-slice viewer.
//
// The 4D ball `B = { x ∈ R⁴ : |x − c| ≤ r }` cross-sectioned by the
// hyperplane `w = w₀` is, for `|w₀ − c.w| < r`, a 3D ball of radius
//   `sqrt(r² − (w₀ − c.w)²)`
// centered at `(c.x, c.y, c.z)`. Outside that w-band the cross-section
// is empty. As the user slides `w₀` the rendered sphere grows from a
// point to its maximum radius and back — the visible signature of
// slicing a 4-ball.
//
// Rendering: ray march the union of every active body's cross-section
// ball plus a floor at y = 0 (the slice of a 4D `y ≥ 0` half-space,
// whose normal has zero w-component, so its 3D slice is the same
// regardless of `w₀`).
//
// All N bodies share radius (params.radius4) and live in the bodies
// array indexed [0, body_count). Bodies whose w-coordinate is outside
// `|w₀ − c.w| < radius4` contribute nothing at the current slice.

const MAX_BODIES: u32 = 32u;

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
    // params[1] = body radius in 4D
    // params[2] = body_count (cast from f32 to u32 in the shader)
    // params[3] reserved
    w_slice: f32,
    radius4: f32,
    body_count_f: f32,
    pad1: f32,
    // Up to MAX_BODIES body 4-positions (xyz, w). Slots >=
    // body_count are ignored.
    bodies: array<vec4<f32>, MAX_BODIES>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

struct SceneHit {
    dist: f32,
    // 0..MAX_BODIES-1 = hypersphere `i`'s cross-section
    // MAX_BODIES     = ground
    // MAX_BODIES + 1 = empty
    material: u32,
    // For body hits, `dw` from this body's w to the slice plane —
    // used by the fragment shader for the warm/cold tint.
    dw: f32,
};

fn slice_radius(w0: f32, body_w: f32, r4: f32) -> f32 {
    let dw = w0 - body_w;
    let r2 = r4 * r4 - dw * dw;
    if (r2 <= 0.0) { return -1.0; }
    return sqrt(r2);
}

fn scene_sdf(p: vec3<f32>) -> SceneHit {
    let body_count = min(u32(u.body_count_f + 0.5), MAX_BODIES);
    let r4 = u.radius4;
    let w0 = u.w_slice;

    var best_dist: f32 = 1.0e9;
    var best_idx: u32 = MAX_BODIES; // sentinel: no body hit yet
    var best_dw: f32 = 0.0;
    for (var i: u32 = 0u; i < body_count; i = i + 1u) {
        let b = u.bodies[i];
        let dw = w0 - b.w;
        let r2 = r4 * r4 - dw * dw;
        if (r2 <= 0.0) { continue; }
        let r3 = sqrt(r2);
        let d = length(p - b.xyz) - r3;
        if (d < best_dist) {
            best_dist = d;
            best_idx = i;
            best_dw = dw;
        }
    }

    let ground = p.y;
    if (best_dist < ground) {
        return SceneHit(best_dist, best_idx, best_dw);
    } else {
        return SceneHit(ground, MAX_BODIES, 0.0);
    }
}

fn ground_color(p: vec3<f32>) -> vec3<f32> {
    let g = floor(p.x) + floor(p.z);
    let alt = abs(g - 2.0 * floor(g * 0.5));
    let dark = vec3<f32>(0.18, 0.20, 0.24);
    let light = vec3<f32>(0.30, 0.32, 0.36);
    return mix(dark, light, alt);
}

fn estimate_normal(p: vec3<f32>) -> vec3<f32> {
    let h = 0.001;
    let dx = scene_sdf(p + vec3<f32>(h, 0.0, 0.0)).dist
           - scene_sdf(p - vec3<f32>(h, 0.0, 0.0)).dist;
    let dy = scene_sdf(p + vec3<f32>(0.0, h, 0.0)).dist
           - scene_sdf(p - vec3<f32>(0.0, h, 0.0)).dist;
    let dz = scene_sdf(p + vec3<f32>(0.0, 0.0, h)).dist
           - scene_sdf(p - vec3<f32>(0.0, 0.0, h)).dist;
    return normalize(vec3<f32>(dx, dy, dz));
}

fn sky(rd: vec3<f32>) -> vec3<f32> {
    let t = (rd.y + 1.0) * 0.5;
    return mix(vec3<f32>(0.04, 0.05, 0.10), vec3<f32>(0.10, 0.13, 0.22), t);
}

/// Per-body base hue. Cycles through 8 distinct hues so adjacent
/// bodies in the spawn order render in distinguishable colors —
/// useful when many cross-sections overlap in the slice plane.
fn body_base_color(idx: u32) -> vec3<f32> {
    let i = idx % 8u;
    switch (i) {
        case 0u:  { return vec3<f32>(0.95, 0.55, 0.30); } // warm orange
        case 1u:  { return vec3<f32>(0.30, 0.55, 0.95); } // cool blue
        case 2u:  { return vec3<f32>(0.55, 0.95, 0.40); } // green
        case 3u:  { return vec3<f32>(0.95, 0.85, 0.30); } // yellow
        case 4u:  { return vec3<f32>(0.85, 0.45, 0.95); } // magenta
        case 5u:  { return vec3<f32>(0.30, 0.95, 0.85); } // teal
        case 6u:  { return vec3<f32>(0.95, 0.40, 0.55); } // pink
        default:  { return vec3<f32>(0.70, 0.70, 0.78); } // neutral
    }
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

    var t: f32 = 0.0;
    let max_t = 60.0;
    var hit_material: u32 = MAX_BODIES + 1u;
    var hit_dw: f32 = 0.0;
    var hit = false;
    for (var i: i32 = 0; i < 192; i = i + 1) {
        let p = ro + rd * t;
        let s = scene_sdf(p);
        if (s.dist < 0.001) {
            hit = true;
            hit_material = s.material;
            hit_dw = s.dw;
            break;
        }
        t = t + max(s.dist, 0.005);
        if (t > max_t) { break; }
    }

    if (!hit) {
        return vec4<f32>(sky(rd), 1.0);
    }

    let p_hit = ro + rd * t;
    let n = estimate_normal(p_hit);
    let light_dir = normalize(vec3<f32>(0.5, 0.85, 0.3));
    let lambert = max(dot(n, light_dir), 0.0);
    let ambient = 0.20;

    var base: vec3<f32>;
    if (hit_material == MAX_BODIES) {
        base = ground_color(p_hit);
    } else {
        // Per-body base color, modulated by how close the slice is to
        // the body's equator. Centered slices read at full saturation;
        // edge slices fade toward neutral grey. Visual cue for
        // "we're seeing the equator" vs "we're near a pole."
        let band = clamp(1.0 - abs(hit_dw) / max(u.radius4, 1e-3), 0.0, 1.0);
        let neutral = vec3<f32>(0.55, 0.55, 0.60);
        base = mix(neutral, body_base_color(hit_material), band);
    }

    let lit = base * (ambient + lambert * 0.85);
    let fog = 1.0 - exp(-t * 0.05);
    let final_color = mix(lit, sky(rd), fog * 0.5);
    return vec4<f32>(final_color, 1.0);
}
