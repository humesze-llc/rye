// 3D physics demo shader — renders a scene of mixed convex bodies
// (spheres, oriented boxes) plus an infinite floor via exact ray-shape
// intersection. Spheres use analytical quadratic; boxes use the
// slab method in the body's local frame.
//
// Body data is packed as (position, kind, half_extents, rotation) per
// body. Rotation is a unit quaternion (x, y, z, w).

const MAX_BODIES: u32 = 32u;
const MAX_DIST: f32 = 60.0;

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
    /// 0 = sphere (extents.x = radius), 1 = box (extents = half_extents)
    kind: f32,
    extents: vec3<f32>,
    _pad: f32,
    rotation: vec4<f32>, // quaternion (x, y, z, w)
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

/// Apply a unit-quaternion rotation to a vector. Standard formula:
/// v' = v + 2·q.xyz × (q.xyz × v + q.w · v).
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.xyz;
    let t = 2.0 * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}

fn quat_conjugate(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

struct Hit {
    d: f32,
    /// -2 = miss (sky), -1 = floor, 0..N = body index.
    id: i32,
}

fn intersect_sphere(ro: vec3<f32>, rd: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
    let oc = ro - c;
    let b = dot(oc, rd);
    let cc = dot(oc, oc) - r * r;
    let disc = b * b - cc;
    if disc < 0.0 {
        return 1e30;
    }
    let t = -b - sqrt(disc);
    if t < 0.0 {
        return 1e30;
    }
    return t;
}

fn intersect_plane(ro: vec3<f32>, rd: vec3<f32>, n: vec3<f32>, offset: f32) -> f32 {
    let denom = dot(rd, n);
    if denom >= -1e-6 {
        return 1e30;
    }
    let t = (offset - dot(ro, n)) / denom;
    if t < 0.0 {
        return 1e30;
    }
    return t;
}

/// Ray vs oriented box via the slab method. Ray is transformed to the
/// box's local frame (position at origin, axis-aligned) before testing.
fn intersect_box(ro: vec3<f32>, rd: vec3<f32>, center: vec3<f32>, half: vec3<f32>, rot: vec4<f32>) -> f32 {
    // World → body-local: rotate by the inverse (conjugate) of rot.
    let inv_rot = quat_conjugate(rot);
    let ro_local = quat_rotate(inv_rot, ro - center);
    let rd_local = quat_rotate(inv_rot, rd);

    // Slab test per axis. Avoid division by zero with a large sentinel.
    let inv_d = select(vec3<f32>(1e30), 1.0 / rd_local, abs(rd_local) > vec3<f32>(1e-6));
    let t0s = (-half - ro_local) * inv_d;
    let t1s = (half - ro_local) * inv_d;
    let t_min = min(t0s, t1s);
    let t_max = max(t0s, t1s);

    let t_near = max(max(t_min.x, t_min.y), t_min.z);
    let t_far = min(min(t_max.x, t_max.y), t_max.z);

    if t_near > t_far || t_far < 0.0 {
        return 1e30;
    }
    return max(t_near, 0.0);
}

/// Analytical box normal at a hit point (world-space). Finds which
/// axis the hit point is on in body-local frame, returns that face's
/// normal rotated back to world.
fn box_normal(world_hit: vec3<f32>, center: vec3<f32>, half: vec3<f32>, rot: vec4<f32>) -> vec3<f32> {
    let inv_rot = quat_conjugate(rot);
    let p_local = quat_rotate(inv_rot, world_hit - center);
    // Compare the hit to each face plane; the closest one wins.
    let d = abs(p_local) - half;
    var local_normal = vec3<f32>(0.0);
    if d.x > d.y && d.x > d.z {
        local_normal = vec3<f32>(sign(p_local.x), 0.0, 0.0);
    } else if d.y > d.z {
        local_normal = vec3<f32>(0.0, sign(p_local.y), 0.0);
    } else {
        local_normal = vec3<f32>(0.0, 0.0, sign(p_local.z));
    }
    return quat_rotate(rot, local_normal);
}

fn cast_ray(ro: vec3<f32>, rd: vec3<f32>) -> Hit {
    var best: Hit;
    best.d = MAX_DIST;
    best.id = -2;

    let floor_t = intersect_plane(ro, rd, scene.floor_normal, scene.floor_offset);
    if floor_t < best.d {
        best.d = floor_t;
        best.id = -1;
    }
    for (var i: u32 = 0u; i < scene.body_count; i = i + 1u) {
        let b = bodies[i];
        var t: f32 = 1e30;
        if b.kind < 0.5 {
            t = intersect_sphere(ro, rd, b.position, b.extents.x);
        } else {
            t = intersect_box(ro, rd, b.position, b.extents, b.rotation);
        }
        if t < best.d {
            best.d = t;
            best.id = i32(i);
        }
    }
    return best;
}

fn normal_for_hit(p: vec3<f32>, id: i32) -> vec3<f32> {
    if id < 0 {
        return scene.floor_normal;
    }
    let b = bodies[u32(id)];
    if b.kind < 0.5 {
        return normalize(p - b.position);
    }
    return box_normal(p, b.position, b.extents, b.rotation);
}

fn color_for_body(id: i32) -> vec3<f32> {
    if id < 0 {
        return vec3<f32>(0.22, 0.23, 0.28);
    }
    let b = bodies[u32(id)];
    // Sphere vs box get different palettes so they're visually distinct.
    if b.kind < 0.5 {
        let k = id % 5;
        if k == 0 { return vec3<f32>(0.92, 0.55, 0.35); }
        if k == 1 { return vec3<f32>(0.40, 0.75, 0.92); }
        if k == 2 { return vec3<f32>(0.58, 0.88, 0.48); }
        if k == 3 { return vec3<f32>(0.88, 0.80, 0.36); }
        return vec3<f32>(0.82, 0.55, 0.92);
    }
    // Box palette: cooler, flatter.
    let k = id % 3;
    if k == 0 { return vec3<f32>(0.68, 0.72, 0.85); }
    if k == 1 { return vec3<f32>(0.78, 0.66, 0.60); }
    return vec3<f32>(0.58, 0.74, 0.76);
}

fn floor_grid(p: vec3<f32>) -> f32 {
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

    let hit = cast_ray(scene.camera_pos, rd);

    if hit.id == -2 {
        let t = smoothstep(-0.2, 0.6, rd.y);
        return vec4<f32>(mix(vec3<f32>(0.08, 0.08, 0.12), vec3<f32>(0.28, 0.34, 0.52), t), 1.0);
    }

    let p = scene.camera_pos + rd * hit.d;
    let n = normal_for_hit(p, hit.id);

    var base = color_for_body(hit.id);
    if hit.id < 0 {
        let g = floor_grid(p);
        base = mix(base * 0.6, base, g);
    }

    let sun = normalize(vec3<f32>(0.3, 0.9, 0.4));
    let diffuse = max(dot(n, sun), 0.0);
    let ambient = 0.35;
    let col = base * (ambient + 0.75 * diffuse);

    return vec4<f32>(col, 1.0);
}
