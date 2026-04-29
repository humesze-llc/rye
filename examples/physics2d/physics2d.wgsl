// Physics 2D demo shader. Renders bodies as SDFs on a fullscreen triangle.
// Each body is (position, rotation, scale, kind); the fragment shader
// evaluates all bodies' SDFs and picks the nearest.

const PI: f32 = 3.14159265359;
const MAX_BODIES: u32 = 64u;

struct Scene {
    /// (world_x_min, world_y_min, world_width, world_height)
    view: vec4<f32>,
    resolution: vec2<f32>,
    body_count: u32,
    _pad: u32,
}

struct Body {
    /// World-space center.
    position: vec2<f32>,
    /// Full-angle rotation as (cos θ, sin θ). The app precomputes this
    /// from the Rotor2 so the shader doesn't redo half-angle math.
    rotation: vec2<f32>,
    /// Sphere radius or polygon circumradius.
    scale: f32,
    /// 0 = circle, 3..6 = regular n-gon with that many sides.
    /// 100 = static wall (axis-aligned rectangle with half_extents encoded in pad).
    kind: u32,
    /// For kind == 100 (static wall), this holds half_extents.
    /// Otherwise padding.
    extent: vec2<f32>,
}

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<uniform> bodies: array<Body, 64>;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    // Oversize triangle: (-1,-1), (3,-1), (-1,3). Covers the viewport
    // and lets the GPU cull the off-screen area automatically.
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(pos[vid], 0.0, 1.0);
}

fn sdf_circle(p: vec2<f32>, r: f32) -> f32 {
    return length(p) - r;
}

/// Regular n-gon SDF (n ≥ 3, circumradius r). Intersection of n
/// half-planes; `apothem` is the distance from center to each edge.
fn sdf_n_gon(p: vec2<f32>, r: f32, n: u32) -> f32 {
    let apothem = r * cos(PI / f32(n));
    var d = -1e30;
    for (var k: u32 = 0u; k < n; k = k + 1u) {
        let theta = (2.0 * f32(k) + 1.0) * PI / f32(n);
        let nrm = vec2<f32>(cos(theta), sin(theta));
        d = max(d, dot(p, nrm) - apothem);
    }
    return d;
}

/// Axis-aligned rectangle SDF; `h` is half-extents.
fn sdf_box(p: vec2<f32>, h: vec2<f32>) -> f32 {
    let q = abs(p) - h;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0);
}

fn body_sdf(p_world: vec2<f32>, body: Body) -> f32 {
    let delta = p_world - body.position;
    // Inverse rotation (rotation stores full-angle cos/sin).
    let c = body.rotation.x;
    let s = body.rotation.y;
    let local = vec2<f32>(c * delta.x + s * delta.y, -s * delta.x + c * delta.y);

    if body.kind == 0u {
        return sdf_circle(local, body.scale);
    }
    if body.kind == 100u {
        return sdf_box(local, body.extent);
    }
    return sdf_n_gon(local, body.scale, body.kind);
}

fn color_for_kind(kind: u32) -> vec3<f32> {
    if kind == 0u {
        return vec3<f32>(0.95, 0.80, 0.35); // circle: amber
    }
    if kind == 3u {
        return vec3<f32>(0.92, 0.40, 0.38); // triangle: coral
    }
    if kind == 4u {
        return vec3<f32>(0.42, 0.68, 0.92); // square: sky
    }
    if kind == 5u {
        return vec3<f32>(0.55, 0.85, 0.50); // pentagon: leaf
    }
    if kind == 6u {
        return vec3<f32>(0.82, 0.58, 0.92); // hexagon: lavender
    }
    if kind == 100u {
        return vec3<f32>(0.30, 0.30, 0.35); // wall: slate
    }
    return vec3<f32>(0.6);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    // Pixel -> UV -> world (y flipped so +Y is up).
    let uv = frag_pos.xy / scene.resolution;
    let world = vec2<f32>(
        scene.view.x + uv.x * scene.view.z,
        scene.view.y + (1.0 - uv.y) * scene.view.w,
    );

    let bg = vec3<f32>(0.08, 0.08, 0.12);

    var closest_d = 1e30;
    var closest_kind = 0u;
    for (var i: u32 = 0u; i < scene.body_count; i = i + 1u) {
        let d = body_sdf(world, bodies[i]);
        if d < closest_d {
            closest_d = d;
            closest_kind = bodies[i].kind;
        }
    }

    // Smooth edge about 1.5 pixels in world units.
    let pixel_size = scene.view.z / scene.resolution.x;
    let edge = 1.0 - smoothstep(-1.5 * pixel_size, 1.5 * pixel_size, closest_d);
    let fill = color_for_kind(closest_kind);
    // Darken the fill slightly near the edge so shapes read.
    let interior_shade = mix(fill * 0.78, fill, clamp(-closest_d * 6.0, 0.0, 1.0));
    let col = mix(bg, interior_shade, edge);

    return vec4<f32>(col, 1.0);
}
