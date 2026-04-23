// Corridor geodesic raymarch demo.
//
// Same geodesic march as geodesic_spheres/spheres.wgsl. The difference
// between E³, H³, and S³ is carried entirely by the ray bending:
// Euclidean-coordinate walls in Space look flat in E³, bowed outward in
// H³ (parallel geodesics diverge), and converging in S³.
//
// Edit while the example is running; ShaderDb hot-reloads this file.

struct Uniforms {
    camera_pos: vec3<f32>,
    camera_forward: vec3<f32>,
    camera_right: vec3<f32>,
    camera_up: vec3<f32>,
    fov_y_tan: f32,
    resolution: vec2<f32>,
    time: f32,
    tick: f32,
    // params = [reserved, ball_scale, fog_scale, reserved]
    power_offset: f32,
    ball_scale: f32,
    fog_scale: f32,
    params_pad: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    if (l2 < 1e-12) {
        return fallback;
    }
    return v * inverseSqrt(l2);
}

fn march_geodesic(ro_scene: vec3<f32>, rd_scene: vec3<f32>) -> vec4<f32> {
    let scale = max(u.ball_scale, 1e-5);
    var p_space = ro_scene * scale;

    let rd_unit = safe_normalize(rd_scene, vec3<f32>(0.0, 0.0, -1.0));
    let probe_eps = 1e-4;
    let probed = rye_exp(p_space, rd_unit * probe_eps);
    let riem_norm = rye_distance(p_space, probed) / probe_eps;
    var v_space = rd_unit / max(riem_norm, 1e-7);

    var t_scene = 0.0;
    var t_arc = 0.0;

    let hit_eps = 0.0008 * scale;
    let min_step = 0.00012 * scale;

    for (var i = 0; i < 220; i = i + 1) {
        let d_space = rye_scene_sdf(p_space);
        // Guard against hits reported in the saturation shell near the
        // Poincaré ball boundary, where mixed Euclidean/Riemannian SDFs
        // return garbage and produce speckle noise.
        let origin_d = rye_origin_distance(p_space);
        if (origin_d > RYE_MAX_ARC * 0.92) {
            return vec4<f32>(0.0, 0.0, 0.0, -1.0);
        }
        if (d_space < hit_eps) {
            return vec4<f32>(p_space, t_scene);
        }
        if (t_scene > 40.0 || t_arc > RYE_MAX_ARC) {
            return vec4<f32>(0.0, 0.0, 0.0, -1.0);
        }

        let step_space = max(d_space * 0.85, min_step);
        let next_p_space = rye_exp(p_space, v_space * step_space);
        let next_v_space = rye_parallel_transport(p_space, next_p_space, v_space);
        p_space = next_p_space;
        v_space = select(v_space, next_v_space, dot(next_v_space, next_v_space) > 1e-12);
        t_scene = t_scene + step_space / scale;
        t_arc = t_arc + step_space;
    }
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
}

fn estimate_normal(p_space: vec3<f32>) -> vec3<f32> {
    let eps = 0.0009 * max(u.ball_scale, 1e-5);
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

// Color the surface by which part of the corridor was hit, using the
// normal to classify floor / ceiling / wall / pillar.
fn surface_base_color(n: vec3<f32>, hit: vec3<f32>) -> vec3<f32> {
    // Floor: normal points up strongly.
    if (n.y > 0.75) {
        // Checkerboard in Space coords — tile lines are geodesic in E³,
        // tanh-spaced in H³, sin-spaced in S³. The tiling is the grid.
        let cell = floor(hit.x * 10.0) + floor(hit.z * 10.0);
        let light_cell = 0.72;
        let dark_cell = 0.48;
        let t = select(dark_cell, light_cell, (i32(cell) & 1) == 0);
        return vec3<f32>(t, t * 0.97, t * 0.90);
    }
    // Ceiling: normal points down.
    if (n.y < -0.75) {
        return vec3<f32>(0.22, 0.24, 0.30);
    }
    // Side walls: normal dominates in x.
    if (abs(n.x) > 0.75) {
        // Faint horizontal banding so perspective is obvious.
        let band = 0.85 + 0.15 * sin(hit.y * 30.0);
        return vec3<f32>(0.58, 0.55, 0.50) * band;
    }
    // Pillar (rounded surface): warm off-white.
    return vec3<f32>(0.92, 0.86, 0.72);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (pos.xy / u.resolution) * 2.0 - 1.0;
    let aspect = u.resolution.x / u.resolution.y;
    let ndc = vec2<f32>(uv.x * aspect, -uv.y);

    let rd = safe_normalize(
        u.camera_forward
        + u.camera_right * ndc.x * u.fov_y_tan
        + u.camera_up * ndc.y * u.fov_y_tan,
        vec3<f32>(0.0, 0.0, -1.0),
    );

    let march_out = march_geodesic(u.camera_pos, rd);
    let t = march_out.w;
    if (t < 0.0) {
        // No hit: fall back to a dim sky/void.
        let horizon = clamp((rd.y + 1.0) * 0.5, 0.0, 1.0);
        let sky = mix(vec3<f32>(0.02, 0.03, 0.06), vec3<f32>(0.09, 0.11, 0.18), horizon);
        return vec4<f32>(sky, 1.0);
    }

    let hit_space = march_out.xyz;
    let n = estimate_normal(hit_space);
    let base = surface_base_color(n, hit_space);

    let sun_dir = safe_normalize(vec3<f32>(0.35, 0.75, -0.25), vec3<f32>(0.0, 1.0, 0.0));
    let diffuse = max(dot(n, sun_dir), 0.0);
    let ambient = 0.22;

    // Geodesic fog in the active Space. This is what makes the three
    // spaces read differently beyond just the wall geometry: H³ distances
    // grow faster than Euclidean (deeper-feeling hallway), S³ distances
    // saturate (closed room at the horizon).
    let cam_dist = rye_distance(u.camera_pos * u.ball_scale, hit_space);
    let fog = exp(-cam_dist / max(u.fog_scale, 1e-4));

    let shaded = base * (diffuse + ambient) * fog;
    return vec4<f32>(pow(shaded, vec3<f32>(1.0 / 2.2)), 1.0);
}
