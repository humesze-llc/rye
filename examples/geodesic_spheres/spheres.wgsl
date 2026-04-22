// Geodesic spheres raymarch demo.
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
    // params layout maps to SceneParams in main.rs
    power_offset: f32, // reserved
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

fn scene_de_space(p_space: vec3<f32>) -> f32 {
    return rye_scene_sdf(p_space);
}

fn march_geodesic(ro_scene: vec3<f32>, rd_scene: vec3<f32>) -> vec4<f32> {
    // Scene SDF is authored in Space coordinates. `ball_scale` maps
    // camera-space points into that coordinate chart.
    let scale = max(u.ball_scale, 1e-5);
    var p_space = ro_scene * scale;
    var v_space = safe_normalize(rd_scene, vec3<f32>(0.0, 0.0, -1.0));
    var t_scene = 0.0;

    let hit_eps = 0.001 * scale;
    let min_step = 0.00015 * scale;

    for (var i = 0; i < 192; i = i + 1) {
        let d_space = scene_de_space(p_space);
        if (d_space < hit_eps) {
            return vec4<f32>(p_space, t_scene);
        }
        if (t_scene > 20.0) {
            return vec4<f32>(0.0, 0.0, 0.0, -1.0);
        }

        let step_space = max(d_space * 0.8, min_step);
        let next_p_space = rye_exp(p_space, v_space * step_space);
        let next_v_space = rye_parallel_transport(p_space, next_p_space, v_space);
        p_space = next_p_space;
        v_space = safe_normalize(next_v_space, v_space);
        t_scene = t_scene + step_space / scale;
    }
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
}

fn estimate_normal(p_space: vec3<f32>) -> vec3<f32> {
    let eps = 0.0012 * max(u.ball_scale, 1e-5);
    let ex = vec3<f32>(eps, 0.0, 0.0);
    let ey = vec3<f32>(0.0, eps, 0.0);
    let ez = vec3<f32>(0.0, 0.0, eps);
    let g = vec3<f32>(
        scene_de_space(p_space + ex) - scene_de_space(p_space - ex),
        scene_de_space(p_space + ey) - scene_de_space(p_space - ey),
        scene_de_space(p_space + ez) - scene_de_space(p_space - ez),
    );
    return safe_normalize(g, vec3<f32>(0.0, 1.0, 0.0));
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
        let horizon = (rd.y + 1.0) * 0.5;
        let sky = mix(vec3<f32>(0.03, 0.04, 0.10), vec3<f32>(0.08, 0.10, 0.18), horizon);
        return vec4<f32>(sky, 1.0);
    }

    let hit_space = march_out.xyz;
    let n = estimate_normal(hit_space);
    let sun_dir = safe_normalize(vec3<f32>(0.4, 0.7, 0.3), vec3<f32>(0.0, 1.0, 0.0));
    let diffuse = max(dot(n, sun_dir), 0.0);
    let ambient = 0.18;

    // Geodesic fog in the active Space.
    let cam_dist = rye_distance(u.camera_pos * u.ball_scale, hit_space);
    let fog = exp(-cam_dist / max(u.fog_scale, 1e-4));

    let base = vec3<f32>(0.35 + 0.3 * n.x, 0.55 + 0.2 * n.y, 0.9);
    let shaded = base * (diffuse + ambient) * fog;
    return vec4<f32>(pow(shaded, vec3<f32>(1.0 / 2.2)), 1.0);
}
