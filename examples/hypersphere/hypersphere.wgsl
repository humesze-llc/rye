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
    // params[3] = ghost_mode (0 = slice mode, 1 = volumetric ghost)
    w_slice: f32,
    radius4: f32,
    body_count_f: f32,
    ghost_mode_f: f32,
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

/// Volumetric "ghost" rendering: integrate each body's `w`-extent
/// (the 4D thickness of the body at each xyz column) along the ray.
/// At a sample point `p` inside body `i`, density contribution is the
/// chord length of body `i` along the `w`-axis through `p`, namely
/// `2·√(r² − |p − c_i|²)`. Outside the xyz-projection it's zero.
///
/// Returns `(rgb, alpha)` in pre-multiplied form, ready to composite
/// over a background.
///
/// Cost: `MAX_STEPS × body_count` per pixel — roughly 250 × 32 ≈ 8K
/// per pixel for the worst case. GPU absorbs this fine at 1080p; the
/// inner loop early-exits as soon as accumulated alpha saturates.
fn ghost_volume(ro: vec3<f32>, rd: vec3<f32>, body_count: u32, t_max: f32) -> vec4<f32> {
    let r4 = u.radius4;
    let r4_sq = r4 * r4;
    // Step size and extinction tuned for visually pleasing density:
    //   * Step too small → fps tank; too large → banding artifacts.
    //   * Sigma too low → ghosts read as faint smudges; too high → fully
    //     opaque blobs that hide structure.
    // Empirically dt = 0.06 with sigma = 0.35 gives smooth shading
    // without banding for unit-radius balls.
    let dt = 0.06;
    let sigma = 0.35;
    let max_steps = 250;

    var t: f32 = 0.0;
    var accum_rgb: vec3<f32> = vec3<f32>(0.0);
    var accum_a: f32 = 0.0;
    for (var step: i32 = 0; step < max_steps; step = step + 1) {
        if (t >= t_max) { break; }
        let p = ro + rd * t;

        var density: f32 = 0.0;
        var weighted_color: vec3<f32> = vec3<f32>(0.0);
        for (var i: u32 = 0u; i < body_count; i = i + 1u) {
            let b = u.bodies[i];
            let d = p - b.xyz;
            let d_sq = dot(d, d);
            let r_sq_at = r4_sq - d_sq;
            if (r_sq_at <= 0.0) { continue; }
            // `w`-extent at this xyz inside body i.
            let w_thick = 2.0 * sqrt(r_sq_at);
            density = density + w_thick;
            weighted_color = weighted_color + body_base_color(i) * w_thick;
        }

        if (density > 1.0e-4) {
            let sample_alpha = 1.0 - exp(-sigma * density * dt);
            let one_minus_a = 1.0 - accum_a;
            // Front-to-back alpha compositing.
            let sample_color = weighted_color / density;
            accum_rgb = accum_rgb + sample_color * sample_alpha * one_minus_a;
            accum_a = accum_a + sample_alpha * one_minus_a;
            if (accum_a > 0.99) { break; }
        }

        t = t + dt;
    }
    return vec4<f32>(accum_rgb, accum_a);
}

/// Background shading for ghost mode: floor if the ray hits the
/// `y = 0` plane in front of the camera, otherwise sky. Returns
/// `(rgb, t_hit)`; if `t_hit < 0`, no floor hit (use sky for the
/// whole ray).
fn ghost_background(ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
    if (rd.y < -1.0e-6) {
        let t_floor = -ro.y / rd.y;
        if (t_floor > 0.0) {
            let p_hit = ro + rd * t_floor;
            // Slightly desaturate floor under fog so ghosts read
            // clearly against it.
            let n = vec3<f32>(0.0, 1.0, 0.0);
            let light_dir = normalize(vec3<f32>(0.5, 0.85, 0.3));
            let lambert = max(dot(n, light_dir), 0.0);
            let base = ground_color(p_hit);
            let lit = base * (0.25 + lambert * 0.75);
            let fog = 1.0 - exp(-t_floor * 0.05);
            return vec4<f32>(mix(lit, sky(rd), fog * 0.5), t_floor);
        }
    }
    return vec4<f32>(sky(rd), -1.0);
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
    let body_count = min(u32(u.body_count_f + 0.5), MAX_BODIES);

    // ---- Ghost (volumetric) mode -----------------------------------
    if (u.ghost_mode_f > 0.5) {
        let bg = ghost_background(ro, rd);
        // Volumetric march up to the floor (or 60 units if no floor
        // hit). We don't penetrate the floor — bodies below `y = 0`
        // are unphysical anyway.
        let t_max = select(60.0, bg.w, bg.w > 0.0);
        let ghost = ghost_volume(ro, rd, body_count, t_max);
        // Front-to-back composite over background.
        let composed = ghost.rgb + bg.rgb * (1.0 - ghost.a);
        return vec4<f32>(composed, 1.0);
    }

    // ---- Slice mode (default) --------------------------------------
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
