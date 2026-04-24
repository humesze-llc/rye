// 3D physics demo shader — raymarches a small scene of sphere bodies
// plus an infinite floor. Designed for clarity over efficiency; at ~32
// bodies the per-pixel loop is cheap.
//
// Body data is packed as (position, radius, kind) in a uniform buffer.

const PI: f32 = 3.14159265359;
const MAX_BODIES: u32 = 32u;
const MAX_STEPS: u32 = 96u;
const MAX_DIST: f32 = 60.0;
const HIT_EPS: f32 = 0.001;

struct Scene {
    camera_pos: vec3<f32>,
    _pad0: f32,
    camera_forward: vec3<f32>,
    fov_y_tan: f32,
    camera_right: vec3<f32>,
    _pad1: f32,
    camera_up: vec3<f32>,
    _pad2: f32,
    resolution: vec2<f32>,
    body_count: u32,
    _pad3: u32,
    floor_normal: vec3<f32>,
    floor_offset: f32,
}

struct Body {
    position: vec3<f32>,
    radius: f32,
    /// Packed as f32 for WGSL std140 alignment convenience.
    /// 0 = sphere.
    kind: f32,
    // Struct-align is 16 (vec3), size is 20, so WGSL implicitly pads
    // the struct to 32 bytes. The Rust-side `GpuBody` carries explicit
    // trailing padding to match that 32-byte stride.
}

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<uniform> bodies: array<Body, 32>;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(pos[vid], 0.0, 1.0);
}

fn sdf_sphere(p: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
    return length(p - c) - r;
}

fn sdf_halfspace(p: vec3<f32>, n: vec3<f32>, offset: f32) -> f32 {
    // Distance to the plane. Negative inside the half-space.
    return dot(p, n) - offset;
}

struct Hit {
    d: f32,
    /// -1 = floor, 0..N = body index.
    id: i32,
}

fn scene_sdf(p: vec3<f32>) -> Hit {
    var best: Hit;
    best.d = sdf_halfspace(p, scene.floor_normal, scene.floor_offset);
    best.id = -1;

    for (var i: u32 = 0u; i < scene.body_count; i = i + 1u) {
        let b = bodies[i];
        let d = sdf_sphere(p, b.position, b.radius);
        if d < best.d {
            best.d = d;
            best.id = i32(i);
        }
    }
    return best;
}

/// Analytical normal for the surface we actually hit. Using
/// finite-differences on `scene_sdf` produces "lensing" artefacts
/// because the min-over-shapes leaks each shape's gradient into its
/// neighbors' surfaces — e.g. floor pixels near a sphere pick up the
/// sphere's radial gradient and bend toward it.
fn normal_for_hit(p: vec3<f32>, id: i32) -> vec3<f32> {
    if id < 0 {
        return scene.floor_normal;
    }
    let b = bodies[u32(id)];
    return normalize(p - b.position);
}

fn raymarch(ro: vec3<f32>, rd: vec3<f32>) -> Hit {
    var t: f32 = 0.0;
    var result: Hit;
    result.id = -2; // miss sentinel
    result.d = MAX_DIST;
    for (var i: u32 = 0u; i < MAX_STEPS; i = i + 1u) {
        let p = ro + rd * t;
        let h = scene_sdf(p);
        if h.d < HIT_EPS {
            result.d = t;
            result.id = h.id;
            return result;
        }
        t = t + max(h.d * 0.9, HIT_EPS);
        if t > MAX_DIST {
            return result;
        }
    }
    return result;
}

fn color_for_body(id: i32) -> vec3<f32> {
    if id < 0 {
        // Floor: subtle grid.
        return vec3<f32>(0.22, 0.23, 0.28);
    }
    // Rotate through a small palette by index.
    let k = id % 5;
    if k == 0 {
        return vec3<f32>(0.92, 0.55, 0.35);
    }
    if k == 1 {
        return vec3<f32>(0.40, 0.75, 0.92);
    }
    if k == 2 {
        return vec3<f32>(0.58, 0.88, 0.48);
    }
    if k == 3 {
        return vec3<f32>(0.88, 0.80, 0.36);
    }
    return vec3<f32>(0.82, 0.55, 0.92);
}

fn floor_grid(p: vec3<f32>) -> f32 {
    // Dark line every 1 unit on x/z.
    let grid = abs(fract(p.xz + vec2<f32>(0.5)) - 0.5);
    let line = min(grid.x, grid.y);
    return smoothstep(0.0, 0.02, line);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_pos.xy / scene.resolution;
    let ndc = vec2<f32>((uv.x - 0.5) * 2.0, (0.5 - uv.y) * 2.0);
    let aspect = scene.resolution.x / scene.resolution.y;

    let rd = normalize(
        scene.camera_forward
            + scene.camera_right * ndc.x * scene.fov_y_tan * aspect
            + scene.camera_up * ndc.y * scene.fov_y_tan,
    );

    let hit = raymarch(scene.camera_pos, rd);

    if hit.id == -2 {
        // Sky: soft vertical gradient.
        let t = smoothstep(-0.2, 0.6, rd.y);
        return vec4<f32>(mix(vec3<f32>(0.08, 0.08, 0.12), vec3<f32>(0.28, 0.34, 0.52), t), 1.0);
    }

    let p = scene.camera_pos + rd * hit.d;
    let n = normal_for_hit(p, hit.id);

    var base = color_for_body(hit.id);
    if hit.id < 0 {
        // Floor gets a grid overlay.
        let g = floor_grid(p);
        base = mix(base * 0.6, base, g);
    }

    let sun = normalize(vec3<f32>(0.3, 0.9, 0.4));
    let diffuse = max(dot(n, sun), 0.0);
    let ambient = 0.35;
    let col = base * (ambient + 0.75 * diffuse);

    return vec4<f32>(col, 1.0);
}
