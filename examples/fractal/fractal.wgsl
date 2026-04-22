// Mandelbulb raymarcher — the fractal demo for Rye.
//
// Edit this file while the example is running; the ShaderDb watcher
// recompiles and RayMarchNode rebuilds on the next frame.

struct Uniforms {
    camera_pos: vec3<f32>,
    camera_forward: vec3<f32>,
    camera_right: vec3<f32>,
    camera_up: vec3<f32>,
    fov_y_tan: f32,
    resolution: vec2<f32>,
    time: f32,
    tick: f32,
    // params layout maps to FractalParams in main.rs
    power_offset: f32,
    ball_scale: f32,
    fog_scale: f32,
    params_pad: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    // Big-triangle trick: one triangle covers the whole viewport.
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

fn sky_color(rd: vec3<f32>) -> vec3<f32> {
    let horizon = (rd.y + 1.0) * 0.5;
    return mix(vec3<f32>(0.03, 0.04, 0.10), vec3<f32>(0.08, 0.10, 0.18), horizon);
}

fn mandelbulb_de(p: vec3<f32>) -> f32 {
    var z = p;
    var dr = 1.0;
    var r = 0.0;
    let power = 8.0 + u.power_offset;
    let iterations = 8;
    for (var i = 0; i < iterations; i = i + 1) {
        r = length(z);
        if (r > 2.0) { break; }
        let theta = acos(clamp(z.z / r, -1.0, 1.0));
        let phi = atan2(z.y, z.x);
        dr = pow(r, power - 1.0) * power * dr + 1.0;
        let zr = pow(r, power);
        let nt = theta * power;
        let np = phi * power;
        z = zr * vec3<f32>(sin(nt) * cos(np), sin(nt) * sin(np), cos(nt)) + p;
    }
    return 0.5 * log(max(r, 1e-4)) * r / dr;
}

fn march_geodesic(ro_scene: vec3<f32>, rd_scene: vec3<f32>) -> vec4<f32> {
    // Scene geometry (Mandelbulb SDF) is authored in Euclidean scene units.
    // The active Space prelude operates in space coordinates, so we map:
    //   p_space = p_scene * ball_scale
    // and scale tangent-step magnitudes accordingly.
    let scale = max(u.ball_scale, 1e-5);
    var p_space = ro_scene * scale;
    var v_space = safe_normalize(rd_scene, vec3<f32>(0.0, 0.0, -1.0)) * scale;
    var t_scene = 0.0;
    let curved_mode = scale < 0.999;

    for (var i = 0; i < 192; i = i + 1) {
        // H3/S3 saturate near the unit-ball boundary; leaving before the
        // clamp shell avoids ring-like artifacts from repeated saturation.
        if (curved_mode && dot(p_space, p_space) > 0.995) {
            return vec4<f32>(0.0, 0.0, 0.0, -1.0);
        }

        let p_scene = p_space / scale;
        let d_scene = mandelbulb_de(p_scene);
        if (d_scene < 0.001) {
            return vec4<f32>(p_scene, t_scene);
        }
        if (t_scene > 32.0) {
            return vec4<f32>(0.0, 0.0, 0.0, -1.0);
        }

        let step_scene = min(d_scene * 0.75, 0.2);
        let step_space = step_scene * scale;
        let next_p_space = rye_exp(p_space, v_space * step_space);
        let next_v_space = rye_parallel_transport(p_space, next_p_space, v_space);
        p_space = next_p_space;
        v_space = safe_normalize(next_v_space, v_space) * scale;
        t_scene = t_scene + step_scene;
    }
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
}

fn estimate_normal(p: vec3<f32>) -> vec3<f32> {
    let eps = 0.0015;
    let ex = vec3<f32>(eps, 0.0, 0.0);
    let ey = vec3<f32>(0.0, eps, 0.0);
    let ez = vec3<f32>(0.0, 0.0, eps);
    let g = vec3<f32>(
        mandelbulb_de(p + ex) - mandelbulb_de(p - ex),
        mandelbulb_de(p + ey) - mandelbulb_de(p - ey),
        mandelbulb_de(p + ez) - mandelbulb_de(p - ez),
    );
    return safe_normalize(g, vec3<f32>(0.0, 1.0, 0.0));
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (pos.xy / u.resolution) * 2.0 - 1.0;
    let aspect = u.resolution.x / u.resolution.y;
    // Flip Y: pos.y increases downward, we want view Y up.
    let ndc = vec2<f32>(uv.x * aspect, -uv.y);

    let rd = safe_normalize(
        u.camera_forward
        + u.camera_right * ndc.x * u.fov_y_tan
        + u.camera_up * ndc.y * u.fov_y_tan,
        vec3<f32>(0.0, 0.0, -1.0),
    );

    let sky = sky_color(rd);
    let march_out = march_geodesic(u.camera_pos, rd);
    let t = march_out.w;
    if (t < 0.0) {
        return vec4<f32>(sky, 1.0);
    }

    let hit = march_out.xyz;
    let n = estimate_normal(hit);
    let sun_dir = safe_normalize(vec3<f32>(0.4, 0.7, 0.3), vec3<f32>(0.0, 1.0, 0.0));
    let diffuse = max(dot(n, sun_dir), 0.0);
    let ambient = 0.15;

    // Exercise the rye-math Space prelude: geodesic stepping is handled
    // by rye_exp / rye_parallel_transport, and camera-to-hit fog uses
    // rye_distance. Swapping the host Space changes both trajectory and
    // attenuation without rewriting the Mandelbulb SDF.
    //
    // `ball_scale` maps Euclidean scene coords into the unit Poincaré
    // ball when the host Space is hyperbolic. In Euclidean mode it is
    // 1.0, so this is a no-op and the output is byte-identical to the
    // pre-flag version. `fog_scale` is the distance at which fog goes
    // fully opaque.
    let scaled_pos = u.camera_pos * u.ball_scale;
    let scaled_hit = hit * u.ball_scale;
    let cam_dist = rye_distance(scaled_pos, scaled_hit);
    var fog = clamp(cam_dist / u.fog_scale, 0.0, 1.0);

    // Additional fade near the H3/S3 boundary shell where curvature-space
    // clamps dominate and marching error accumulates.
    if (u.ball_scale < 0.999) {
        fog = max(fog, smoothstep(0.90, 0.995, dot(scaled_hit, scaled_hit)));
    }

    let base = vec3<f32>(0.35 + 0.3 * n.x, 0.55 + 0.2 * n.y, 0.9);
    let lit = base * (diffuse + ambient);
    let shaded = mix(lit, sky, fog);
    return vec4<f32>(pow(max(shaded, vec3<f32>(0.0, 0.0, 0.0)), vec3<f32>(1.0 / 2.2)), 1.0);
}
