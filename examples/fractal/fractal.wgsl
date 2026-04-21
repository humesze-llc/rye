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
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    // Big-triangle trick: one triangle covers the whole viewport.
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn mandelbulb_de(p: vec3<f32>) -> f32 {
    var z = p;
    var dr = 1.0;
    var r = 0.0;
    let power = 8.0 + u.params.x; // live-tunable via params.x
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

fn march(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    var t = 0.0;
    for (var i = 0; i < 128; i = i + 1) {
        let p = ro + rd * t;
        let d = mandelbulb_de(p);
        if (d < 0.001) { return t; }
        if (t > 20.0) { return -1.0; }
        t = t + d * 0.9;
    }
    return -1.0;
}

fn estimate_normal(p: vec3<f32>) -> vec3<f32> {
    let eps = 0.0015;
    let ex = vec3<f32>(eps, 0.0, 0.0);
    let ey = vec3<f32>(0.0, eps, 0.0);
    let ez = vec3<f32>(0.0, 0.0, eps);
    return normalize(vec3<f32>(
        mandelbulb_de(p + ex) - mandelbulb_de(p - ex),
        mandelbulb_de(p + ey) - mandelbulb_de(p - ey),
        mandelbulb_de(p + ez) - mandelbulb_de(p - ez),
    ));
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (pos.xy / u.resolution) * 2.0 - 1.0;
    let aspect = u.resolution.x / u.resolution.y;
    // Flip Y: pos.y increases downward, we want view Y up.
    let ndc = vec2<f32>(uv.x * aspect, -uv.y);

    let rd = normalize(
        u.camera_forward
        + u.camera_right * ndc.x * u.fov_y_tan
        + u.camera_up * ndc.y * u.fov_y_tan
    );

    let t = march(u.camera_pos, rd);
    if (t < 0.0) {
        let horizon = (rd.y + 1.0) * 0.5;
        let sky = mix(vec3<f32>(0.03, 0.04, 0.10), vec3<f32>(0.08, 0.10, 0.18), horizon);
        return vec4<f32>(sky, 1.0);
    }

    let hit = u.camera_pos + rd * t;
    let n = estimate_normal(hit);
    let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
    let diffuse = max(dot(n, sun_dir), 0.0);
    let ambient = 0.15;

    // Exercise the rye-math Space prelude: camera-to-hit distance comes
    // from rye_distance, so swapping the host Space for HyperbolicH3
    // changes shading without touching the SDF or ray march.
    //
    // `ball_scale` (params.y) maps Euclidean scene coords into the
    // unit Poincaré ball when the host Space is hyperbolic. In
    // Euclidean mode the host sets it to 1.0, so this is a no-op and
    // the output is byte-identical to the pre-flag version.
    // `fog_scale` (params.z) is the distance at which fog reaches 1.0.
    let scaled_pos = u.camera_pos * u.params.y;
    let scaled_hit = hit * u.params.y;
    let cam_dist = rye_distance(scaled_pos, scaled_hit);
    let fog = 1.0 - clamp(cam_dist / u.params.z, 0.0, 1.0);

    let base = vec3<f32>(0.35 + 0.3 * n.x, 0.55 + 0.2 * n.y, 0.9);
    let shaded = base * (diffuse + ambient) * fog;
    return vec4<f32>(pow(shaded, vec3<f32>(1.0 / 2.2)), 1.0);
}
