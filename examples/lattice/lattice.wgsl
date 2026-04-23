// Geodesic lattice demo — side-by-side E³ / H³ / S³ comparison.
//
// This file is compiled THREE times against three different Space preludes;
// the scene module (rye_scene_sdf) is also different per Space because
// LatticeSphereScene::to_wgsl computes geodesic lattice centers in Rust.
// The visual difference between panels is entirely geometric: the same
// camera, the same WGSL march loop, different center positions and metric.
//
// Uniforms.params layout (see main.rs):
//   params[0] = panel x pixel offset  (e.g. 0, W/3, 2W/3)
//   params[1] = panel index            (0 = E³, 1 = H³, 2 = S³)
//   params[2] = panel pixel width      (e.g. W/3)
//   params[3] = fog scale              (space-tuned: 3.0 / 4.0 / 2.5)

struct Uniforms {
    camera_pos:     vec3<f32>,
    camera_forward: vec3<f32>,
    camera_right:   vec3<f32>,
    camera_up:      vec3<f32>,
    fov_y_tan:      f32,
    resolution:     vec2<f32>,
    time:           f32,
    tick:           f32,
    params:         vec4<f32>,
};

@group(0) @binding(0) var<uniform> ub: Uniforms;

// Scale from camera-space to Space coordinates.
// Must match main.rs BALL_SCALE constant.
const BALL_SCALE: f32 = 0.2;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    if l2 < 1e-12 { return fallback; }
    return v * inverseSqrt(l2);
}

// Ray direction for this panel's pixel.
// Accounts for panel x-offset and panel width so each third of the
// window gets its own full [-1,1] UV range.
fn panel_ray_dir(frag_pos: vec4<f32>) -> vec3<f32> {
    let x_offset = ub.params[0];
    let panel_w  = ub.params[2];
    let panel_h  = ub.resolution.y;
    let px = frag_pos.x - x_offset;
    let py = frag_pos.y;
    let uv = vec2<f32>(px / panel_w, py / panel_h) * 2.0 - 1.0;
    let aspect = panel_w / panel_h;
    return safe_normalize(
        ub.camera_forward
            + ub.camera_right * ( uv.x * aspect * ub.fov_y_tan)
            + ub.camera_up    * (-uv.y          * ub.fov_y_tan),
        ub.camera_forward,
    );
}

// Per-space accent color for visual distinction.
fn accent_color(panel_idx: f32) -> vec3<f32> {
    if panel_idx < 0.5 { return vec3<f32>(0.35, 0.60, 1.00); } // E³ — blue
    if panel_idx < 1.5 { return vec3<f32>(1.00, 0.52, 0.18); } // H³ — orange
    return vec3<f32>(0.22, 0.85, 0.58);                          // S³ — teal
}

// Geodesic ray march in the active Space.
// Returns vec4(p_space.xyz, t_scene) on hit, or w = -1 on miss.
fn march_geodesic(ro_scene: vec3<f32>, rd_scene: vec3<f32>) -> vec4<f32> {
    let scale = max(BALL_SCALE, 1e-5);
    var p_space = ro_scene * scale;

    let rd_unit = safe_normalize(rd_scene, vec3<f32>(0.0, 0.0, -1.0));
    let probe_eps = 1e-4;
    let probed = rye_exp(p_space, rd_unit * probe_eps);
    let riem_norm = rye_distance(p_space, probed) / probe_eps;
    var v_space = rd_unit / max(riem_norm, 1e-7);

    var t_scene = 0.0;
    var t_arc   = 0.0;

    let hit_eps  = 0.001 * scale;
    let min_step = 0.00015 * scale;

    for (var i = 0; i < 192; i = i + 1) {
        let d_space = rye_scene_sdf(p_space);
        if d_space < hit_eps {
            return vec4<f32>(p_space, t_scene);
        }
        if t_scene > 20.0 || t_arc > RYE_MAX_ARC {
            return vec4<f32>(0.0, 0.0, 0.0, -1.0);
        }
        let step_space = max(d_space * 0.8, min_step);
        let next_p     = rye_exp(p_space, v_space * step_space);
        let next_v     = rye_parallel_transport(p_space, next_p, v_space);
        p_space = next_p;
        v_space = select(v_space, next_v, dot(next_v, next_v) > 1e-12);
        t_scene = t_scene + step_space / scale;
        t_arc   = t_arc   + step_space;
    }
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
}

fn estimate_normal(p_space: vec3<f32>) -> vec3<f32> {
    let eps = 0.0012 * max(BALL_SCALE, 1e-5);
    let ex = vec3<f32>(eps, 0.0, 0.0);
    let ey = vec3<f32>(0.0, eps, 0.0);
    let ez = vec3<f32>(0.0, 0.0, eps);
    let g = vec3<f32>(
        rye_scene_sdf(p_space + ex) - rye_scene_sdf(p_space - ex),
        rye_scene_sdf(p_space + ey) - rye_scene_sdf(p_space - ey),
        rye_scene_sdf(p_space + ez) - rye_scene_sdf(p_space - ez),
    );
    return safe_normalize(g, vec3<f32>(0.0, 1.0, 0.0));
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let rd     = panel_ray_dir(frag_pos);
    let accent = accent_color(ub.params[1]);
    let fog_sc = ub.params[3];

    // Sky: vertical gradient tinted by space accent
    let sky_t = 0.5 + 0.5 * rd.y;
    let sky   = mix(vec3<f32>(0.05, 0.04, 0.08), accent * 0.55, sky_t);

    let march = march_geodesic(ub.camera_pos, rd);
    var col: vec3<f32>;

    if march.w < 0.0 {
        col = sky;
    } else {
        let hit_p   = march.xyz;
        let n       = estimate_normal(hit_p);
        let sun_dir = safe_normalize(vec3<f32>(0.5, 0.9, 0.3), vec3<f32>(0.0, 1.0, 0.0));
        let diff    = max(dot(n, sun_dir), 0.0);
        let rim     = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
        let base    = accent * 0.82;
        col = base * (0.15 + 0.75 * diff) + rim * 0.28 * accent;

        // Geodesic fog from camera to hit point
        let fog_dist = rye_distance(ub.camera_pos * BALL_SCALE, hit_p);
        let fog      = exp(-fog_dist / max(fog_sc, 1e-4));
        col = mix(sky, col, fog);
    }

    // Gamma correction
    col = pow(max(col, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    return vec4<f32>(col, 1.0);
}
