//! The user assembles the WGSL by concatenating [`HYPERSLICE_KERNEL_WGSL`]
//! with `Scene4::to_hyperslice_wgsl("u.w_slice")` (which defines
//! `loam_scene_sdf`).

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use loam_math::Rotor4;
use wgpu::*;

use crate::device::RenderDevice;
use crate::graph::RenderNode;

/// The uniform layout is fixed-size, so raising this is a recompile.
pub const MAX_BODIES: usize = 32;

/// Stored in [`BodyUniform::radius_or_shape`]. Mirrored as `SHAPE_*`
/// constants in [`HYPERSLICE_KERNEL_WGSL`]; keep in sync.
pub const SHAPE_PENTATOPE: u32 = 0;
pub const SHAPE_TESSERACT: u32 = 1;
pub const SHAPE_16CELL: u32 = 2;
pub const SHAPE_24CELL: u32 = 3;
pub const SHAPE_120CELL: u32 = 4;
pub const SHAPE_600CELL: u32 = 5;
pub const SHAPE_3SPHERE: u32 = 6;
pub const SHAPE_DUOCYLINDER: u32 = 7;
pub const SHAPE_CLIFFORD_TORUS: u32 = 8;
pub const SHAPE_SPHERINDER: u32 = 9;

/// One dynamic-body slot; `kind` selects which fields the shader reads.
/// `std140`-aligned, 80 bytes, matching the kernel's `BodyUniform`. The rotor
/// packs into two `vec4<f32>` so std140 matches Rust's `[f32; 8]`; an
/// `array<f32, 8>` would pad each element to 16 bytes (128-byte slot).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BodyUniform {
    pub position: [f32; 4],
    pub kind: f32,
    pub radius_or_shape: f32,
    pub polytope_size: f32,
    pub _pad0: f32,
    pub color: [f32; 3],
    pub _pad1: f32,
    pub rotor: [f32; 8],
}

impl Default for BodyUniform {
    fn default() -> Self {
        Self {
            position: [0.0; 4],
            kind: BodyKind::Invalid as i32 as f32,
            radius_or_shape: 0.0,
            polytope_size: 0.0,
            _pad0: 0.0,
            color: [0.7, 0.7, 0.7],
            _pad1: 0.0,
            rotor: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    }
}

/// `BodyUniform::kind` discriminator. Cast to `f32` when writing the uniform; the shader reads
/// it back as `u32`-via-`f32`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BodyKind {
    /// `HyperSphere4D`. Reads `position` and `radius_or_shape`; ignores `rotor`.
    Sphere = 0,
    /// `ConvexPolytope4D`. Reads `position`, `rotor`, and `radius_or_shape`.
    Polytope = 1,
    /// Sentinel for slots the kernel skips: no dispatch branch matches, so the
    /// slot stays inert. `255` is outside the live range and exact in `f32`.
    Invalid = 255,
}

impl BodyUniform {
    pub fn sphere(position: [f32; 4], radius: f32, color: [f32; 3]) -> Self {
        Self {
            position,
            kind: BodyKind::Sphere as i32 as f32,
            radius_or_shape: radius,
            color,
            ..Self::default()
        }
    }

    /// `shape_index` indexes the kernel's shape table, `size` is the
    /// circumradius, `rotor` is the Rotor4 packed via `<[f32; 8]>::from(Rotor4)`.
    pub fn polytope(
        position: [f32; 4],
        shape_index: u32,
        size: f32,
        rotor: [f32; 8],
        color: [f32; 3],
    ) -> Self {
        Self {
            position,
            kind: BodyKind::Polytope as i32 as f32,
            radius_or_shape: shape_index as f32,
            polytope_size: size,
            color,
            rotor,
            ..Self::default()
        }
    }

    pub fn polytope_with_rotor(
        position: [f32; 4],
        shape_index: u32,
        size: f32,
        rotor: Rotor4,
        color: [f32; 3],
    ) -> Self {
        Self::polytope(position, shape_index, size, rotor.into(), color)
    }
}

/// Uniform buffer for [`Hyperslice4DNode`]. Bind group 0, binding 0. `std140`-compatible
/// layout matching the kernel's `Uniforms` struct.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Hyperslice4DUniforms {
    pub camera_pos: [f32; 3],
    pub _pad0: f32,
    pub camera_forward: [f32; 3],
    pub _pad1: f32,
    pub camera_right: [f32; 3],
    pub _pad2: f32,
    pub camera_up: [f32; 3],
    pub fov_y_tan: f32,
    pub resolution: [f32; 2],
    pub time: f32,
    pub tick: f32,
    /// Slicing hyperplane `w` coordinate; read by `Scene4`'s hyperslice emit.
    pub w_slice: f32,
    /// Active body-slot count. `f32` for std140 alignment; kernel rounds to int.
    pub body_count: f32,
    /// Pixel offset of the viewport's top-left in the framebuffer:
    /// `uv = (frag_pos.xy - viewport_origin) / resolution`. Zero unless a side
    /// panel carves out a sub-region.
    pub viewport_origin: [f32; 2],
    /// User-shader scalar knobs; mirrors `RayMarchUniforms::params`.
    pub params: [f32; 4],
    /// Dynamic body slots; slots `>= body_count` are unread. See [`BodyUniform`].
    pub bodies: [BodyUniform; MAX_BODIES],
}

impl Default for Hyperslice4DUniforms {
    fn default() -> Self {
        Self {
            camera_pos: [0.0, 0.0, 5.0],
            _pad0: 0.0,
            camera_forward: [0.0, 0.0, -1.0],
            _pad1: 0.0,
            camera_right: [1.0, 0.0, 0.0],
            _pad2: 0.0,
            camera_up: [0.0, 1.0, 0.0],
            fov_y_tan: (60.0_f32.to_radians() * 0.5).tan(),
            resolution: [1.0, 1.0],
            time: 0.0,
            tick: 0.0,
            w_slice: 0.0,
            body_count: 0.0,
            viewport_origin: [0.0, 0.0],
            params: [0.0; 4],
            bodies: [BodyUniform::default(); MAX_BODIES],
        }
    }
}

/// The user's `Scene4` emit supplies `loam_scene_sdf`. Prefixed with
/// [`crate::sky_ground::SKY_GROUND_WGSL`], which owns `sky`; the kernel paints
/// no background of its own and reads `sky` only for the body fog term.
pub const HYPERSLICE_KERNEL_WGSL: &str = concat!(
    include_str!("../sky_ground.wgsl"),
    r#"
const MAX_BODIES: u32 = 32u;

const BODY_KIND_SPHERE: u32 = 0u;
const BODY_KIND_POLYTOPE: u32 = 1u;

// Polytope shape table indices. Mirrored from loam-render Rust-side
// `SHAPE_*` constants; keep in sync.
const SHAPE_PENTATOPE: u32 = 0u;
const SHAPE_TESSERACT: u32 = 1u;
const SHAPE_16CELL: u32 = 2u;
const SHAPE_24CELL: u32 = 3u;
const SHAPE_120CELL: u32 = 4u;
const SHAPE_600CELL: u32 = 5u;
const SHAPE_3SPHERE: u32 = 6u;
const SHAPE_DUOCYLINDER: u32 = 7u;
const SHAPE_CLIFFORD_TORUS: u32 = 8u;
const SHAPE_SPHERINDER: u32 = 9u;
// Mirrors `BodyKind::Invalid` (CPU). Intentionally absent from the
// dispatch chain below: neither the sphere nor the polytope branch
// matches, so the SDF accumulator keeps its 1e9 initial value for that
// slot. `BodyUniform::default()` produces this kind, so uninitialised
// slots are inert; the CPU/GPU protocol breaks if it is removed.
const BODY_KIND_INVALID: u32 = 255u;

struct BodyUniform {
    position: vec4<f32>,
    kind: f32,
    radius_or_shape: f32,
    polytope_size: f32,
    _pad0: f32,
    color: vec3<f32>,
    _pad1: f32,
    // Rotor4 packed as 2 × vec4<f32> so the std140 stride matches
    // Rust's tightly-packed `[f32; 8]` (an `array<f32, 8>` here
    // would pad each element to 16 bytes -> 128-byte slot, breaking
    // the 80-byte total). Order: [s, xy, xz, xw, yz, yw, zw, xyzw].
    rotor_lo: vec4<f32>,
    rotor_hi: vec4<f32>,
};

struct Uniforms {
    camera_pos: vec3<f32>,
    camera_forward: vec3<f32>,
    camera_right: vec3<f32>,
    camera_up: vec3<f32>,
    fov_y_tan: f32,
    resolution: vec2<f32>,
    time: f32,
    tick: f32,
    w_slice: f32,
    body_count: f32,
    viewport_origin: vec2<f32>,
    params: vec4<f32>,
    bodies: array<BodyUniform, MAX_BODIES>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

fn body_sphere_sdf_4d(p4: vec4<f32>, b: BodyUniform) -> f32 {
    return length(p4 - b.position) - b.radius_or_shape;
}

// `Rotor4::apply` (CPU) computes the forward sandwich `R̃ · v · R`. To go
// world -> body local, flip the bivector signs of `R` to get its reverse
// `R̃` and run the same formula with R̃ as the rotor: that equals
// `R · v · R̃`, the inverse rotation.
fn rotor4_inverse_apply(rotor_lo: vec4<f32>, rotor_hi: vec4<f32>, v: vec4<f32>) -> vec4<f32> {
    let rs  = rotor_lo.x;
    // Bivector signs flipped (reverse of R).
    let rxy = -rotor_lo.y;
    let rxz = -rotor_lo.z;
    let rxw = -rotor_lo.w;
    let ryz = -rotor_hi.x;
    let ryw = -rotor_hi.y;
    let rzw = -rotor_hi.z;
    // Pseudoscalar unchanged on reverse.
    let r_i = rotor_hi.w;

    let vx = v.x; let vy = v.y; let vz = v.z; let vw = v.w;

    // Stage 1: R̃ · v. The signs were inverted above, so this stays a direct
    // port of `Rotor4::apply` Stage 1, whose bivector terms are positive.
    let p1 = rs * vx - rxy * vy - rxz * vz - rxw * vw;
    let p2 = rs * vy + rxy * vx - ryz * vz - ryw * vw;
    let p3 = rs * vz + rxz * vx + ryz * vy - rzw * vw;
    let p4 = rs * vw + rxw * vx + ryw * vy + rzw * vz;

    // 3-vector part of R̃ · v in basis (e123, e124, e134, e234).
    let t123 = -rxy * vz + rxz * vy - ryz * vx + r_i * vw;
    let t124 = -rxy * vw + rxw * vy - ryw * vx - r_i * vz;
    let t134 = -rxz * vw + rxw * vz - rzw * vx + r_i * vy;
    let t234 = -ryz * vw + ryw * vz - rzw * vy - r_i * vx;

    // Stage 2: (1-vec + 3-vec) · R, extract the 1-vec output.
    let q1 = rs * p1 - rxy * p2 - rxz * p3 - rxw * p4
           - ryz * t123 - ryw * t124 - rzw * t134 + r_i * t234;
    let q2 = rs * p2 + rxy * p1 - ryz * p3 - ryw * p4
           + rxz * t123 + rxw * t124 - rzw * t234 - r_i * t134;
    let q3 = rs * p3 + rxz * p1 + ryz * p2 - rzw * p4
           - rxy * t123 + rxw * t134 + ryw * t234 + r_i * t124;
    let q4 = rs * p4 + rxw * p1 + ryw * p2 + rzw * p3
           - rxy * t124 - rxz * t134 - ryz * t234 - r_i * t123;

    return vec4<f32>(q1, q2, q3, q4);
}

// Per-shape SDFs take the polytope centred at the origin in its canonical
// frame. The dispatcher translates and inverse-rotates into body-local
// coordinates first, then scales by the body's circumradius.

// Pentatope at unit circumradius: five face hyperplanes, signed distance is
// the max plane distance. Vertices match
// `loam_physics::euclidean_r4::pentatope_vertices(1.0)`; face i is opposite
// vertex i with outward normal -v_i, t = sqrt(5)/4, and the n-simplex
// inradius R/n is 0.25 at n = 4, R = 1.
fn pentatope_sdf_local(p: vec4<f32>) -> f32 {
    let t  = 0.55901699437;  // sqrt(5) / 4
    let n0 = vec4<f32>(0.0, 0.0, 0.0, -1.0);
    let n1 = vec4<f32>(-t, -t, -t, 0.25);
    let n2 = vec4<f32>(-t,  t,  t, 0.25);
    let n3 = vec4<f32>( t, -t,  t, 0.25);
    let n4 = vec4<f32>( t,  t, -t, 0.25);
    let r = 0.25;
    var d: f32 = dot(n0, p) - r;
    d = max(d, dot(n1, p) - r);
    d = max(d, dot(n2, p) - r);
    d = max(d, dot(n3, p) - r);
    d = max(d, dot(n4, p) - r);
    return d;
}

// Tesseract at unit circumradius: faces at ±0.5 along each axis, the
// standard infinity-norm box form.
fn tesseract_sdf_local(p: vec4<f32>) -> f32 {
    let q = abs(p) - vec4<f32>(0.5, 0.5, 0.5, 0.5);
    let outside = length(max(q, vec4<f32>(0.0, 0.0, 0.0, 0.0)));
    let inside = min(max(max(q.x, q.y), max(q.z, q.w)), 0.0);
    return outside + inside;
}

// 16-cell at unit circumradius. The 16 face normals are `(±1, ±1, ±1, ±1)/2`
// at perpendicular distance 0.5, so the max-over-faces plane distance
// reduces in any octant to `(|p.x| + |p.y| + |p.z| + |p.w| - 1) / 2`. The
// `/ 2` is the unit-normal normalisation; without it this returns the L1
// distance, which over-estimates the true SDF and tunnels under sphere
// tracing.
fn cell16_sdf_local(p: vec4<f32>) -> f32 {
    let q = abs(p);
    return (q.x + q.y + q.z + q.w - 1.0) * 0.5;
}

// 24-cell at unit circumradius: the intersection of a tesseract scaled to
// 1/sqrt(2) with a 16-cell scaled to sqrt(2), whose 24 vertices are the
// permutations of (±1/sqrt(2), ±1/sqrt(2), 0, 0). Intersection of convex
// shapes is `max`; the cross-polytope leg carries the same `/ 2` correction
// as `cell16_sdf_local`.
fn cell24_sdf_local(p: vec4<f32>) -> f32 {
    let inv_sqrt2: f32 = 0.70710678;
    let sqrt2:     f32 = 1.41421356;
    let q = abs(p);
    let tess  = max(max(q.x, q.y), max(q.z, q.w)) - inv_sqrt2;
    let cross = (q.x + q.y + q.z + q.w - sqrt2) * 0.5;
    return max(tess, cross);
}

fn sphere3_sdf_local(p: vec4<f32>) -> f32 {
    return length(p) - 1.0;
}

// Duocylinder (D² × D²) at unit circumradius: two 2-discs in the orthogonal
// xy and zw planes, each of radius 1/sqrt(2) so the bounding 4-ball is unit.
// Outside/inside split as in the box SDF; the outside leg is the true 2-D
// Euclidean distance, not an underestimate.
fn duocylinder_sdf_local(p: vec4<f32>) -> f32 {
    let r = 0.7071068;
    let dxy = length(p.xy) - r;
    let dzw = length(p.zw) - r;
    let outside = length(vec2<f32>(max(dxy, 0.0), max(dzw, 0.0)));
    let inside = min(max(dxy, dzw), 0.0);
    return outside + inside;
}

// Clifford torus filled as a 4-D tube of radius `tube` around the surface
// `length(p.xy) = r1, length(p.zw) = r2`. The centre curve is codimension 2,
// so the SDF takes a vec2 length in the (q1, q2) normal plane. Numbers
// chosen so the bounding 4-ball is unit.
fn clifford_torus_sdf_local(p: vec4<f32>) -> f32 {
    let r1 = 0.5;
    let r2 = 0.5;
    let tube = 0.2;
    let q1 = length(p.xy) - r1;
    let q2 = length(p.zw) - r2;
    return length(vec2<f32>(q1, q2)) - tube;
}

// Spherinder (B³ × [-h, h]) at unit circumradius: a 3-ball extruded along w.
// Box-style SDF over the radial distance from the w-axis and the w slab.
fn spherinder_sdf_local(p: vec4<f32>) -> f32 {
    let r = 0.7071068;
    let h = 0.7071068;
    let dxyz = length(p.xyz) - r;
    let dw = abs(p.w) - h;
    let outside = length(vec2<f32>(max(dxyz, 0.0), max(dw, 0.0)));
    let inside = min(max(dxyz, dw), 0.0);
    return outside + inside;
}

// Any unit-circumradius polytope sits inside the unit 4-ball, so well
// outside it the ball SDF `|world_v| - size` is a Lipschitz-1 lower bound on
// the true polytope SDF and the marcher can step on it without paying for
// the rotor inverse. The 1.5 factor leaves margin before silhouette.
fn body_polytope_sdf_4d(p4: vec4<f32>, b: BodyUniform) -> f32 {
    let size = max(b.polytope_size, 1.0e-6);
    let world_v = p4 - b.position;
    let world_dist2 = dot(world_v, world_v);
    let bound = size * 1.5;
    if (world_dist2 > bound * bound) {
        return sqrt(world_dist2) - size;
    }
    let local_v = rotor4_inverse_apply(b.rotor_lo, b.rotor_hi, world_v);
    let unit_p = local_v / size;
    let shape = u32(b.radius_or_shape + 0.5);
    var d: f32 = 1.0e9;
    if (shape == SHAPE_PENTATOPE) {
        d = pentatope_sdf_local(unit_p);
    } else if (shape == SHAPE_TESSERACT) {
        d = tesseract_sdf_local(unit_p);
    } else if (shape == SHAPE_16CELL) {
        d = cell16_sdf_local(unit_p);
    } else if (shape == SHAPE_24CELL) {
        d = cell24_sdf_local(unit_p);
    } else if (shape == SHAPE_120CELL) {
        d = cell120_sdf_local(unit_p);
    } else if (shape == SHAPE_600CELL) {
        d = cell600_sdf_local(unit_p);
    } else if (shape == SHAPE_3SPHERE) {
        d = sphere3_sdf_local(unit_p);
    } else if (shape == SHAPE_DUOCYLINDER) {
        d = duocylinder_sdf_local(unit_p);
    } else if (shape == SHAPE_CLIFFORD_TORUS) {
        d = clifford_torus_sdf_local(unit_p);
    } else if (shape == SHAPE_SPHERINDER) {
        d = spherinder_sdf_local(unit_p);
    }
    return d * size;
}

fn loam_dynamic_bodies_sdf(p3: vec3<f32>) -> f32 {
    let p4 = vec4<f32>(p3, u.w_slice);
    let body_count = u32(u.body_count + 0.5);
    var sdf: f32 = 1.0e9;
    for (var i: u32 = 0u; i < body_count; i = i + 1u) {
        let b = u.bodies[i];
        let kind = u32(b.kind + 0.5);
        if (kind == BODY_KIND_SPHERE) {
            sdf = min(sdf, body_sphere_sdf_4d(p4, b));
        } else if (kind == BODY_KIND_POLYTOPE) {
            sdf = min(sdf, body_polytope_sdf_4d(p4, b));
        }
    }
    return sdf;
}

fn loam_body_sdf_at(p3: vec3<f32>, body_idx: u32) -> f32 {
    if (body_idx >= MAX_BODIES) { return 1.0e9; }
    let p4 = vec4<f32>(p3, u.w_slice);
    let b = u.bodies[body_idx];
    let kind = u32(b.kind + 0.5);
    if (kind == BODY_KIND_SPHERE) {
        return body_sphere_sdf_4d(p4, b);
    } else if (kind == BODY_KIND_POLYTOPE) {
        return body_polytope_sdf_4d(p4, b);
    }
    return 1.0e9;
}

// Per-pixel hit information: which body was hit (or MAX_BODIES
// for the static scene; MAX_BODIES + 1 for nothing).
struct HitInfo {
    dist: f32,
    body_idx: u32,
};

fn loam_total_sdf(p3: vec3<f32>) -> HitInfo {
    let scene_d = loam_scene_sdf(p3);

    let p4 = vec4<f32>(p3, u.w_slice);
    let body_count = u32(u.body_count + 0.5);
    var dyn_d: f32 = 1.0e9;
    var dyn_idx: u32 = MAX_BODIES;
    for (var i: u32 = 0u; i < body_count; i = i + 1u) {
        let b = u.bodies[i];
        let kind = u32(b.kind + 0.5);
        var d: f32 = 1.0e9;
        if (kind == BODY_KIND_SPHERE) {
            d = body_sphere_sdf_4d(p4, b);
        } else if (kind == BODY_KIND_POLYTOPE) {
            d = body_polytope_sdf_4d(p4, b);
        }
        if (d < dyn_d) {
            dyn_d = d;
            dyn_idx = i;
        }
    }

    if (scene_d <= dyn_d) {
        return HitInfo(scene_d, MAX_BODIES);
    }
    return HitInfo(dyn_d, dyn_idx);
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// Central differences on the dominating SDF only; sampling the combined SDF
// blends gradients at silhouettes.
fn estimate_normal(p: vec3<f32>, body_idx: u32) -> vec3<f32> {
    let h = 0.001;
    if (body_idx >= MAX_BODIES) {
        let dx = loam_scene_sdf(p + vec3<f32>(h, 0.0, 0.0))
               - loam_scene_sdf(p - vec3<f32>(h, 0.0, 0.0));
        let dy = loam_scene_sdf(p + vec3<f32>(0.0, h, 0.0))
               - loam_scene_sdf(p - vec3<f32>(0.0, h, 0.0));
        let dz = loam_scene_sdf(p + vec3<f32>(0.0, 0.0, h))
               - loam_scene_sdf(p - vec3<f32>(0.0, 0.0, h));
        return normalize(vec3<f32>(dx, dy, dz));
    }
    let dx = loam_body_sdf_at(p + vec3<f32>(h, 0.0, 0.0), body_idx)
           - loam_body_sdf_at(p - vec3<f32>(h, 0.0, 0.0), body_idx);
    let dy = loam_body_sdf_at(p + vec3<f32>(0.0, h, 0.0), body_idx)
           - loam_body_sdf_at(p - vec3<f32>(0.0, h, 0.0), body_idx);
    let dz = loam_body_sdf_at(p + vec3<f32>(0.0, 0.0, h), body_idx)
           - loam_body_sdf_at(p - vec3<f32>(0.0, 0.0, h), body_idx);
    return normalize(vec3<f32>(dx, dy, dz));
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    // `frag_pos.xy` is in framebuffer coordinates regardless of any
    // `set_viewport` carve-out, so subtract the viewport origin and normalise
    // by `u.resolution` to keep the camera centred.
    let uv = ((frag_pos.xy - u.viewport_origin) / u.resolution) * 2.0 - vec2<f32>(1.0, 1.0);
    let aspect = u.resolution.x / u.resolution.y;
    let ndc = vec2<f32>(uv.x * aspect, -uv.y);
    let rd = normalize(
        u.camera_forward
        + u.camera_right * (ndc.x * u.fov_y_tan)
        + u.camera_up    * (ndc.y * u.fov_y_tan)
    );
    let ro = u.camera_pos;

    var t: f32 = 0.0;
    // Analytical far-clip from any HalfSpace4D leaves, +1.0 so the marcher lands
    // its hit on the floor itself; 60.0 where there is no analytical
    // contribution. Without it, near-horizon rays exhaust the iter budget.
    let scene_max_t = loam_scene_max_t(ro, rd);
    let max_t = min(60.0, scene_max_t + 1.0);
    var hit = false;
    var hit_idx: u32 = MAX_BODIES + 1u;
    // The whole reported distance, floored at min_step < hit_eps so a march
    // grazing a surface still advances.
    //
    // A full step is the largest sphere tracing can prove safe, and everything
    // this kernel composes earns it: each body-local SDF above is 1-Lipschitz
    // and vanishes on its own surface S, so d(p) = d(p) - d(b) <= |p - b| for
    // any b in S, hence d(p) <= dist(p, S), a lower bound that cannot tunnel.
    // Looseness costs iterations, never safety. The bounding-sphere and Wolfe
    // fast paths are lower bounds by their own arguments rather than by
    // continuity. `min` and `max` preserve a lower bound, so the union over
    // bodies and Scene4's CSG tree inherit it, on the one precondition Scene4
    // does not enforce: a HalfSpace4D leaf's normal must be unit, or its SDF
    // scales by |n| and overshoots. 384 iters covers tangent-grazing
    // convergence.
    let hit_eps = 0.001;
    let min_step = 0.0001;
    for (var i: i32 = 0; i < 384; i = i + 1) {
        let p = ro + rd * t;
        let h = loam_total_sdf(p);
        if (h.dist < hit_eps) {
            hit = true;
            hit_idx = h.body_idx;
            break;
        }
        t = t + max(h.dist, min_step);
        if (t > max_t) { break; }
    }

    // The background pass under this one owns every pixel the march does not
    // win: the sky on a miss, and its own analytic ground where the scene's
    // HalfSpace4D leaf wins, which leaves that leaf in the marched union as a
    // pure occluder of the bodies.
    if (!hit) {
        discard;
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let p_hit = ro + rd * t;
    if (hit_idx >= MAX_BODIES) {
        if (loam_scene_at(p_hit).kind == LOAM_PRIM_HALFSPACE4D) {
            discard;
            return vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }
    }
    let n = estimate_normal(p_hit, hit_idx);
    let light_dir = normalize(vec3<f32>(0.5, 0.85, 0.3));
    let lambert = max(dot(n, light_dir), 0.0);
    let ambient = 0.20;
    var base = vec3<f32>(0.65, 0.65, 0.72);
    if (hit_idx < MAX_BODIES) {
        base = u.bodies[hit_idx].color;
    }
    let lit = base * (ambient + lambert * 0.85);
    let fog = 1.0 - exp(-t * 0.05);
    let final_color = mix(lit, sky(rd), fog * 0.5);
    return vec4<f32>(final_color, 1.0);
}
"#
);

const HYPERSLICE_UNIFORMS_SIZE: u64 = std::mem::size_of::<Hyperslice4DUniforms>() as u64;

// A bind group may only view a uniform buffer at a multiple of the adapter's
// `min_uniform_buffer_offset_alignment` (256 on most desktop backends, 32 at
// the spec floor), so the stride rounds the uniform size up to it.
fn strip_cell_stride(min_uniform_buffer_offset_alignment: u64) -> u64 {
    HYPERSLICE_UNIFORMS_SIZE.div_ceil(min_uniform_buffer_offset_alignment)
        * min_uniform_buffer_offset_alignment
}

fn strip_cell_uniforms(
    base: &Hyperslice4DUniforms,
    viewport: crate::Viewport,
    w_slice: f32,
    body: &BodyUniform,
) -> Hyperslice4DUniforms {
    let mut cell = *base;
    cell.bodies[0] = *body;
    cell.body_count = 1.0;
    cell.w_slice = w_slice;
    cell.resolution = viewport.resolution_f32();
    cell.viewport_origin = [viewport.x as f32, viewport.y as f32];
    cell
}

struct StripCellUniforms {
    buffer: Buffer,
    bind_groups: Vec<BindGroup>,
    stride: u64,
}

impl StripCellUniforms {
    fn new(device: &Device, layout: &BindGroupLayout, cell_count: usize) -> Self {
        let stride = strip_cell_stride(device.limits().min_uniform_buffer_offset_alignment as u64);
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("hyperslice4d strip cell uniforms"),
            size: stride * cell_count as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_groups = (0..cell_count)
            .map(|i| {
                device.create_bind_group(&BindGroupDescriptor {
                    label: Some("hyperslice4d strip cell bg"),
                    layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &buffer,
                            offset: i as u64 * stride,
                            size: BufferSize::new(HYPERSLICE_UNIFORMS_SIZE),
                        }),
                    }],
                })
            })
            .collect();
        Self {
            buffer,
            bind_groups,
            stride,
        }
    }
}

pub struct Hyperslice4DNode {
    pipeline: RenderPipeline,
    uniforms: Hyperslice4DUniforms,
    uniform_buf: Buffer,
    bind_group: BindGroup,
    bind_group_layout: BindGroupLayout,
    /// Read by the two paths that own their attachment. Defaults to
    /// [`crate::sky_ground::SKY_HORIZON`] because the kernel discards on a
    /// miss, so this is what stands in for the sky there.
    clear_color: Color,
    strip_cells: Option<StripCellUniforms>,
}

impl Hyperslice4DNode {
    /// `sample_count` must match the color attachment at draw time
    /// ([`crate::device::RenderDevice::sample_count`]; 1 in tests).
    pub fn new(
        device: &Device,
        surface_format: TextureFormat,
        module: &ShaderModule,
        sample_count: u32,
    ) -> Self {
        let uniform_buf = device.create_buffer(&BufferDescriptor {
            label: Some("hyperslice4d uniforms"),
            size: std::mem::size_of::<Hyperslice4DUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("hyperslice4d bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("hyperslice4d bg"),
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("hyperslice4d pipeline layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("hyperslice4d pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            uniforms: Hyperslice4DUniforms::default(),
            uniform_buf,
            bind_group,
            bind_group_layout: bgl,
            clear_color: crate::sky_ground::SKY_HORIZON,
            strip_cells: None,
        }
    }

    pub fn uniforms(&self) -> &Hyperslice4DUniforms {
        &self.uniforms
    }
    pub fn uniforms_mut(&mut self) -> &mut Hyperslice4DUniforms {
        &mut self.uniforms
    }

    pub fn set_uniforms(&mut self, queue: &Queue, uniforms: Hyperslice4DUniforms) {
        self.uniforms = uniforms;
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&self.uniforms));
    }

    pub fn flush_uniforms(&self, queue: &Queue) {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&self.uniforms));
    }

    pub fn set_clear_color(&mut self, color: Color) {
        self.clear_color = color;
    }

    /// Does not auto-flush; pair with [`Self::flush_uniforms`].
    pub fn set_bodies(&mut self, bodies: &[BodyUniform]) {
        let n = bodies.len().min(MAX_BODIES);
        self.uniforms.bodies[..n].copy_from_slice(&bodies[..n]);
        self.uniforms.body_count = n as f32;
    }

    pub fn set_body(&mut self, index: usize, body: BodyUniform) {
        if index < MAX_BODIES {
            self.uniforms.bodies[index] = body;
        }
    }

    pub fn set_body_count(&mut self, count: usize) {
        self.uniforms.body_count = count.min(MAX_BODIES) as f32;
    }
}

impl Hyperslice4DNode {
    /// Records into the caller's `encoder` and restricts the fragment shader to
    /// a sub-region. Does not submit: the runner owns one encoder per frame.
    ///
    /// `LoadOp::Load`, unlike [`Self::execute_strip`]: the kernel discards
    /// every pixel it does not shade, so a caller on this path must record
    /// [`crate::SkyGroundNode`] (or another background) ahead of it.
    pub fn record_in_viewport(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        viewport: crate::Viewport,
    ) {
        let mut rp = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("hyperslice4d pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        viewport.apply(&mut rp);
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &self.bind_group, &[]);
        rp.draw(0..3, 0..1);
    }

    /// Zero-size cells are skipped; the node's own uniforms are left untouched.
    ///
    /// One encoder, one pass, one submit. Legal only because every cell gets its
    /// own uniform image and its own bind group viewing it: `Queue::write_buffer`
    /// lands ahead of the whole command buffer it precedes, so a shared image
    /// would give every cell the last cell's write.
    pub fn execute_strip(
        &mut self,
        device: &Device,
        queue: &Queue,
        view: &wgpu::TextureView,
        cells: &[(crate::Viewport, f32, BodyUniform)],
    ) -> Result<()> {
        let drawn = cells
            .iter()
            .filter(|(viewport, _, _)| viewport.width != 0 && viewport.height != 0)
            .count();
        if drawn == 0 {
            return Ok(());
        }
        if self
            .strip_cells
            .as_ref()
            .is_none_or(|strip| strip.bind_groups.len() < drawn)
        {
            self.strip_cells = Some(StripCellUniforms::new(
                device,
                &self.bind_group_layout,
                drawn,
            ));
        }
        let strip = self
            .strip_cells
            .as_ref()
            .expect("strip cells allocated above");

        {
            // One staging map for the whole strip: `write_buffer` per cell would put
            // every cell's image through its own belt allocation and copy command.
            let upload_size =
                BufferSize::new(drawn as u64 * strip.stride).expect("at least one cell");
            let mut staging = queue
                .write_buffer_with(&strip.buffer, 0, upload_size)
                .context("mapping the filmstrip's per-cell uniform staging buffer")?;
            let mut slot = 0usize;
            for (viewport, w_slice, body) in cells {
                if viewport.width == 0 || viewport.height == 0 {
                    continue;
                }
                let cell = strip_cell_uniforms(&self.uniforms, *viewport, *w_slice, body);
                let start = slot * strip.stride as usize;
                staging[start..start + HYPERSLICE_UNIFORMS_SIZE as usize]
                    .copy_from_slice(bytemuck::bytes_of(&cell));
                slot += 1;
            }
        }

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("hyperslice4d strip encoder"),
        });
        {
            let mut rp = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("hyperslice4d strip pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(self.clear_color),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rp.set_pipeline(&self.pipeline);
            let mut slot = 0usize;
            for (viewport, _, _) in cells {
                if viewport.width == 0 || viewport.height == 0 {
                    continue;
                }
                viewport.apply(&mut rp);
                rp.set_bind_group(0, &strip.bind_groups[slot], &[]);
                rp.draw(0..3, 0..1);
                slot += 1;
            }
        }
        queue.submit(Some(encoder.finish()));
        Ok(())
    }
}

impl RenderNode for Hyperslice4DNode {
    fn name(&self) -> &'static str {
        "hyperslice4d"
    }

    fn execute(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        let mut encoder = rd.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("hyperslice4d encoder"),
        });
        {
            let mut rp = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("hyperslice4d pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(self.clear_color),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.bind_group, &[]);
            rp.draw(0..3, 0..1);
        }
        rd.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_has_expected_entry_points() {
        assert!(HYPERSLICE_KERNEL_WGSL.contains("@vertex"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("fn vs_fullscreen"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("@fragment"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("fn fs_main"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("struct Uniforms"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("@group(0) @binding(0)"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("loam_scene_sdf("));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("BodyUniform"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("loam_dynamic_bodies_sdf"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("BODY_KIND_SPHERE"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("BODY_KIND_POLYTOPE"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("BODY_KIND_INVALID"));
        assert!(HYPERSLICE_KERNEL_WGSL.starts_with(crate::sky_ground::SKY_GROUND_WGSL));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("body_polytope_sdf_4d"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("pentatope_sdf_local"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("tesseract_sdf_local"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("cell16_sdf_local"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("cell24_sdf_local"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("cell120_sdf_local"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("cell600_sdf_local"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("sphere3_sdf_local"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("duocylinder_sdf_local"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("clifford_torus_sdf_local"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("spherinder_sdf_local"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("rotor4_inverse_apply"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("loam_body_sdf_at"));
        assert!(
            HYPERSLICE_KERNEL_WGSL.contains("loam_scene_at(p_hit).kind == LOAM_PRIM_HALFSPACE4D")
        );
        assert!(!HYPERSLICE_KERNEL_WGSL.contains("abs(p_hit.y) < 0.01"));
        assert_eq!(
            HYPERSLICE_KERNEL_WGSL.matches("discard;").count(),
            2,
            "the kernel discards on a miss and on a halfspace hit, so the \
             background pass draws the ground exactly once"
        );
        assert!(!HYPERSLICE_KERNEL_WGSL.contains("base = ground_color("));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("loam_scene_max_t(ro, rd)"));
    }

    #[test]
    fn body_uniform_is_80_bytes() {
        assert_eq!(std::mem::size_of::<BodyUniform>(), 80);
    }

    #[test]
    fn default_body_is_inert_invalid_kind() {
        let b = BodyUniform::default();
        assert_eq!(b.kind as i32, BodyKind::Invalid as i32);
        assert_eq!(b.rotor, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn kernel_validates_with_minimal_scene() {
        const SCENE_STUB: &str = r#"
const LOAM_PRIM_HYPERSPHERE4D: u32 = 0u;
const LOAM_PRIM_HALFSPACE4D: u32 = 1u;
const LOAM_PRIM_OTHER: u32 = 255u;
struct LoamSceneHit { dist: f32, kind: u32 }
fn loam_scene_at(p: vec3<f32>) -> LoamSceneHit {
    return LoamSceneHit(length(p) - 0.5, LOAM_PRIM_OTHER);
}
fn loam_scene_sdf(p: vec3<f32>) -> f32 {
    return loam_scene_at(p).dist;
}
fn loam_scene_max_t(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    return 1.0e9;
}
"#;
        // body_polytope_sdf_4d references cell120/cell600_sdf_local from the
        // polytope_data emit; append it so naga validation succeeds.
        let polytope = super::super::polytope_data::polytope_extended_sdfs_wgsl();
        let source = format!("{HYPERSLICE_KERNEL_WGSL}\n{polytope}\n{SCENE_STUB}");
        let module = naga::front::wgsl::parse_str(&source)
            .expect("hyperslice4d kernel + scene stub should parse as WGSL");
        let flags = naga::valid::ValidationFlags::all();
        let caps = naga::valid::Capabilities::empty();
        naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .expect("hyperslice4d kernel + scene stub should validate");
    }

    #[test]
    fn kernel_validates_with_real_scene_union() {
        use glam::Vec4;
        use loam_scene::{Scene4, SceneNode4};

        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::new(-0.6, 0.7, -1.5, 0.0), 0.7)
                .union(SceneNode4::hypersphere(Vec4::new(0.6, 0.7, -1.5, 0.0), 0.7))
                .union(SceneNode4::hypersphere(Vec4::new(0.0, 1.0, 1.5, 0.0), 1.0))
                .union(SceneNode4::halfspace(Vec4::Y, 0.0)),
        );
        let scene_wgsl = scene.to_hyperslice_wgsl("u.w_slice");
        let polytope = super::super::polytope_data::polytope_extended_sdfs_wgsl();
        let source = format!("{HYPERSLICE_KERNEL_WGSL}\n{polytope}\n{scene_wgsl}");
        let module = naga::front::wgsl::parse_str(&source)
            .expect("hyperslice4d kernel + Scene4 emit should parse as WGSL");
        let flags = naga::valid::ValidationFlags::all();
        let caps = naga::valid::Capabilities::empty();
        naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .expect("hyperslice4d kernel + Scene4 emit should validate");
    }

    #[test]
    fn kernel_validates_with_gated_scene() {
        use glam::Vec4;
        use loam_scene::{Scene4, SceneNode4};

        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::new(0.0, 1.0, 0.0, 0.0), 0.5)
                .union(SceneNode4::halfspace(Vec4::Y, 0.0)),
        );
        let scene_wgsl = scene.to_hyperslice_wgsl_gated("u.w_slice", "u.params.x");
        let polytope = super::super::polytope_data::polytope_extended_sdfs_wgsl();
        let source = format!("{HYPERSLICE_KERNEL_WGSL}\n{polytope}\n{scene_wgsl}");
        let module = naga::front::wgsl::parse_str(&source)
            .expect("gated Scene4 emit should parse against the kernel");
        let flags = naga::valid::ValidationFlags::all();
        let caps = naga::valid::Capabilities::empty();
        naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .expect("gated Scene4 emit should validate against the kernel");
    }

    #[test]
    fn invalid_kind_has_no_kernel_dispatch_branch() {
        for forbidden in ["kind == BODY_KIND_INVALID", "BODY_KIND_INVALID == kind"] {
            assert!(
                !HYPERSLICE_KERNEL_WGSL.contains(forbidden),
                "BODY_KIND_INVALID must remain unreferenced in dispatch \
                 so default-constructed bodies stay inert (matched: {forbidden:?})",
            );
        }
    }

    // Each `*_sdf_local_cpu` is a 1:1 port of the matching WGSL function.

    use glam::{Vec2, Vec4};

    fn pentatope_sdf_local_cpu(p: Vec4) -> f32 {
        // t = sqrt(5)/4. WGSL stores the truncation 0.55901699437; the CPU side
        // reconstructs from the closed form for f32-clean values.
        let t = 5.0_f32.sqrt() * 0.25;
        let r = 0.25_f32;
        let normals = [
            Vec4::new(0.0, 0.0, 0.0, -1.0),
            Vec4::new(-t, -t, -t, 0.25),
            Vec4::new(-t, t, t, 0.25),
            Vec4::new(t, -t, t, 0.25),
            Vec4::new(t, t, -t, 0.25),
        ];
        normals
            .iter()
            .map(|n| n.dot(p) - r)
            .fold(f32::NEG_INFINITY, f32::max)
    }

    fn tesseract_sdf_local_cpu(p: Vec4) -> f32 {
        let q = p.abs() - Vec4::splat(0.5);
        let outside = q.max(Vec4::ZERO).length();
        let inside = q.x.max(q.y).max(q.z).max(q.w).min(0.0);
        outside + inside
    }

    fn cell16_sdf_local_cpu(p: Vec4) -> f32 {
        let q = p.abs();
        (q.x + q.y + q.z + q.w - 1.0) * 0.5
    }

    fn cell24_sdf_local_cpu(p: Vec4) -> f32 {
        // CPU uses stdlib consts, WGSL 8-digit truncations; the tolerance absorbs it.
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let sqrt2 = std::f32::consts::SQRT_2;
        let q = p.abs();
        let tess = q.x.max(q.y).max(q.z).max(q.w) - inv_sqrt2;
        let cross = (q.x + q.y + q.z + q.w - sqrt2) * 0.5;
        tess.max(cross)
    }

    fn sphere3_sdf_local_cpu(p: Vec4) -> f32 {
        p.length() - 1.0
    }

    fn duocylinder_sdf_local_cpu(p: Vec4) -> f32 {
        let r = std::f32::consts::FRAC_1_SQRT_2;
        let dxy = Vec2::new(p.x, p.y).length() - r;
        let dzw = Vec2::new(p.z, p.w).length() - r;
        let outside = Vec2::new(dxy.max(0.0), dzw.max(0.0)).length();
        outside + dxy.max(dzw).min(0.0)
    }

    fn clifford_torus_sdf_local_cpu(p: Vec4) -> f32 {
        let q1 = Vec2::new(p.x, p.y).length() - 0.5;
        let q2 = Vec2::new(p.z, p.w).length() - 0.5;
        Vec2::new(q1, q2).length() - 0.2
    }

    fn spherinder_sdf_local_cpu(p: Vec4) -> f32 {
        let r = std::f32::consts::FRAC_1_SQRT_2;
        let h = std::f32::consts::FRAC_1_SQRT_2;
        let dxyz = Vec3::new(p.x, p.y, p.z).length() - r;
        let dw = p.w.abs() - h;
        let outside = Vec2::new(dxyz.max(0.0), dw.max(0.0)).length();
        outside + dxyz.max(dw).min(0.0)
    }

    type LocalSdf = fn(Vec4) -> f32;

    // The 120-cell and 600-cell are absent on purpose: `polytope_data` emits
    // their SDFs and they carry their own certification test there.
    fn kernel_body_sdfs() -> [(&'static str, LocalSdf); 8] {
        [
            ("pentatope", pentatope_sdf_local_cpu),
            ("tesseract", tesseract_sdf_local_cpu),
            ("16-cell", cell16_sdf_local_cpu),
            ("24-cell", cell24_sdf_local_cpu),
            ("3-sphere", sphere3_sdf_local_cpu),
            ("duocylinder", duocylinder_sdf_local_cpu),
            ("clifford-torus", clifford_torus_sdf_local_cpu),
            ("spherinder", spherinder_sdf_local_cpu),
        ]
    }

    // LCG multiplier and increment are Knuth's MMIX constants. Seeded so the
    // sweeps below are reproducible.
    fn lcg_signed_unit(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 33) as f32) / ((1u64 << 31) as f32) - 1.0
    }

    fn lcg_vec4(state: &mut u64, scale: f32) -> Vec4 {
        Vec4::new(
            lcg_signed_unit(state),
            lcg_signed_unit(state),
            lcg_signed_unit(state),
            lcg_signed_unit(state),
        ) * scale
    }

    #[test]
    fn every_hand_written_body_sdf_is_one_lipschitz() {
        // f32 slack: the secant divides two rounded evaluations by a rounded
        // separation, so an exactly-1-Lipschitz function reads slightly over 1.
        const SLACK: f32 = 1e-4;
        for (name, sdf) in kernel_body_sdfs() {
            let mut state = 0xC0FFEE_u64;
            let mut worst = 0.0_f32;
            for _ in 0..20_000 {
                // 2.5 covers well inside (all shapes have unit circumradius)
                // out to the region the bounding-sphere fast path takes over.
                let a = lcg_vec4(&mut state, 2.5);
                let b = a + lcg_vec4(&mut state, 0.5);
                let separation = (a - b).length();
                if separation > 1e-3 {
                    worst = worst.max((sdf(a) - sdf(b)).abs() / separation);
                }
            }
            assert!(
                worst <= 1.0 + SLACK,
                "{name} has Lipschitz constant {worst} > 1: its step scale is \
                 1/{worst}, not the full step the kernel takes",
            );
        }
    }

    #[test]
    fn every_hand_written_body_sdf_vanishes_on_its_own_surface() {
        let surface_probes: [(&str, LocalSdf, Vec4); 8] = [
            ("pentatope", pentatope_sdf_local_cpu, Vec4::W),
            (
                "tesseract",
                tesseract_sdf_local_cpu,
                Vec4::new(0.5, 0.5, 0.5, 0.5),
            ),
            ("16-cell", cell16_sdf_local_cpu, Vec4::X),
            (
                "24-cell",
                cell24_sdf_local_cpu,
                Vec4::new(
                    std::f32::consts::FRAC_1_SQRT_2,
                    std::f32::consts::FRAC_1_SQRT_2,
                    0.0,
                    0.0,
                ),
            ),
            ("3-sphere", sphere3_sdf_local_cpu, Vec4::X),
            (
                "duocylinder",
                duocylinder_sdf_local_cpu,
                Vec4::new(std::f32::consts::FRAC_1_SQRT_2, 0.0, 0.0, 0.0),
            ),
            (
                "clifford-torus",
                clifford_torus_sdf_local_cpu,
                Vec4::new(0.7, 0.0, 0.5, 0.0),
            ),
            (
                "spherinder",
                spherinder_sdf_local_cpu,
                Vec4::new(std::f32::consts::FRAC_1_SQRT_2, 0.0, 0.0, 0.0),
            ),
        ];
        for (name, sdf, on_surface) in surface_probes {
            let d = sdf(on_surface);
            assert!(
                d.abs() < 1e-6,
                "{name} reads {d} at the surface point {on_surface:?}; a nonzero \
                 offset there breaks the `d <= dist` bound the full step rests on",
            );
        }
    }

    // Inline vertex generators, mirroring loam_physics::euclidean_r4.
    fn pentatope_vertices() -> Vec<Vec4> {
        let base_w = -0.25_f32;
        let base_r = 15.0_f32.sqrt() / 4.0;
        let t = base_r / 3.0_f32.sqrt();
        vec![
            Vec4::new(0.0, 0.0, 0.0, 1.0),
            Vec4::new(t, t, t, base_w),
            Vec4::new(t, -t, -t, base_w),
            Vec4::new(-t, t, -t, base_w),
            Vec4::new(-t, -t, t, base_w),
        ]
    }

    fn tesseract_vertices() -> Vec<Vec4> {
        let a = 0.5_f32;
        let mut v = Vec::with_capacity(16);
        for &w in &[-a, a] {
            for &z in &[-a, a] {
                for &y in &[-a, a] {
                    for &x in &[-a, a] {
                        v.push(Vec4::new(x, y, z, w));
                    }
                }
            }
        }
        v
    }

    fn cell16_vertices() -> Vec<Vec4> {
        vec![
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(-1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, -1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, 0.0, -1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
            Vec4::new(0.0, 0.0, 0.0, -1.0),
        ]
    }

    fn cell24_vertices() -> Vec<Vec4> {
        let k = std::f32::consts::FRAC_1_SQRT_2;
        let mut v = Vec::with_capacity(24);
        for i in 0..4 {
            for j in (i + 1)..4 {
                for &si in &[-k, k] {
                    for &sj in &[-k, k] {
                        let mut c = [0.0_f32; 4];
                        c[i] = si;
                        c[j] = sj;
                        v.push(Vec4::new(c[0], c[1], c[2], c[3]));
                    }
                }
            }
        }
        v
    }

    fn assert_polytope_geometry(
        name: &str,
        sdf: impl Fn(Vec4) -> f32,
        vertices: &[Vec4],
        inradius: f32,
        vertex_tolerance: f32,
    ) {
        for (i, &v) in vertices.iter().enumerate() {
            let d = sdf(v);
            assert!(
                d.abs() < vertex_tolerance,
                "{name} vertex[{i}] = {v:?} should sit on the surface (sdf={d}, tol={vertex_tolerance})",
            );
        }
        let d_origin = sdf(Vec4::ZERO);
        assert!(
            (d_origin - -inradius).abs() < 5e-4,
            "{name} sdf(origin) = {d_origin} should equal -inradius {}",
            -inradius,
        );
        let outside = vertices[0] * 2.0;
        let d_outside = sdf(outside);
        assert!(
            d_outside > 0.0,
            "{name} sdf at 2x first vertex {outside:?} = {d_outside} should be positive (outside)",
        );
        let inside = vertices[0] * 0.5;
        let d_inside = sdf(inside);
        assert!(
            d_inside < 0.0,
            "{name} sdf at 0.5x first vertex {inside:?} = {d_inside} should be negative (inside)",
        );
    }

    #[test]
    fn pentatope_cpu_port_matches_geometry() {
        // n-simplex inradius at unit circumradius is R/n = 0.25 for n=4. Tolerance
        // covers the WGSL 11-digit truncation of the normal constant.
        assert_polytope_geometry(
            "pentatope",
            pentatope_sdf_local_cpu,
            &pentatope_vertices(),
            0.25,
            5e-6,
        );
    }

    #[test]
    fn tesseract_cpu_port_matches_geometry() {
        // Inradius 0.5; abs/min/max SDF is bit-exact at the half-extent.
        assert_polytope_geometry(
            "tesseract",
            tesseract_sdf_local_cpu,
            &tesseract_vertices(),
            0.5,
            1e-7,
        );
    }

    #[test]
    fn cell16_cpu_port_matches_geometry() {
        assert_polytope_geometry(
            "16-cell",
            cell16_sdf_local_cpu,
            &cell16_vertices(),
            0.5,
            1e-7,
        );
    }

    #[test]
    fn cell24_cpu_port_matches_geometry() {
        // Inradius 1/sqrt(2); tolerance covers the WGSL 8-digit truncations.
        assert_polytope_geometry(
            "24-cell",
            cell24_sdf_local_cpu,
            &cell24_vertices(),
            std::f32::consts::FRAC_1_SQRT_2,
            5e-7,
        );
    }

    #[test]
    fn polytope_sdf_constants_match_kernel_source() {
        assert!(HYPERSLICE_KERNEL_WGSL.contains("0.55901699437"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("0.70710678"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("1.41421356"));
    }

    #[test]
    fn polytope_with_rotor_matches_manual_packing() {
        let rotor = Rotor4 {
            s: 0.5,
            xy: 0.1,
            xz: 0.2,
            xw: 0.3,
            yz: 0.4,
            yw: 0.6,
            zw: 0.7,
            xyzw: 0.8,
        };
        let manual = BodyUniform::polytope(
            [1.0, 2.0, 3.0, 4.0],
            SHAPE_24CELL,
            0.7,
            rotor.into(),
            [0.9, 0.4, 0.1],
        );
        let helper = BodyUniform::polytope_with_rotor(
            [1.0, 2.0, 3.0, 4.0],
            SHAPE_24CELL,
            0.7,
            rotor,
            [0.9, 0.4, 0.1],
        );
        assert_eq!(bytemuck::bytes_of(&manual), bytemuck::bytes_of(&helper));
    }

    #[test]
    fn shape_constants_mirror_kernel_table() {
        for (rust_const, wgsl_decl) in [
            (SHAPE_PENTATOPE, "const SHAPE_PENTATOPE: u32 = 0u;"),
            (SHAPE_TESSERACT, "const SHAPE_TESSERACT: u32 = 1u;"),
            (SHAPE_16CELL, "const SHAPE_16CELL: u32 = 2u;"),
            (SHAPE_24CELL, "const SHAPE_24CELL: u32 = 3u;"),
            (SHAPE_120CELL, "const SHAPE_120CELL: u32 = 4u;"),
            (SHAPE_600CELL, "const SHAPE_600CELL: u32 = 5u;"),
            (SHAPE_3SPHERE, "const SHAPE_3SPHERE: u32 = 6u;"),
            (SHAPE_DUOCYLINDER, "const SHAPE_DUOCYLINDER: u32 = 7u;"),
            (
                SHAPE_CLIFFORD_TORUS,
                "const SHAPE_CLIFFORD_TORUS: u32 = 8u;",
            ),
            (SHAPE_SPHERINDER, "const SHAPE_SPHERINDER: u32 = 9u;"),
        ] {
            assert!(
                HYPERSLICE_KERNEL_WGSL.contains(wgsl_decl),
                "kernel missing `{wgsl_decl}` for Rust value {rust_const}"
            );
        }
    }

    // 1:1 mirror of `fs_main`'s sphere-trace loop; constants must match.

    use glam::Vec3;

    // 1.0 because every SDF the kernel composes is 1-Lipschitz and vanishes on
    // its surface, hence bounds the true distance from below; see `fs_main`.
    // A parameter of the marcher below so a test can drive the superseded step.
    const KERNEL_STEP_SCALE: f32 = 1.0;
    const KERNEL_HIT_EPS: f32 = 0.001;

    struct HyperHit {
        hit_pos: Vec3,
        body_idx: u32,
    }

    struct MarchResult {
        hit: Option<HyperHit>,
        steps: u32,
    }

    fn march_hyperslice_cpu<F>(ro: Vec3, rd: Vec3, step_scale: f32, sdf: F) -> MarchResult
    where
        F: Fn(Vec3) -> (f32, u32),
    {
        let mut t: f32 = 0.0;
        let max_t = 60.0_f32;
        let min_step = 0.0001_f32;
        for i in 0..384u32 {
            let p = ro + rd * t;
            let (d, body_idx) = sdf(p);
            if d < KERNEL_HIT_EPS {
                return MarchResult {
                    hit: Some(HyperHit {
                        hit_pos: p,
                        body_idx,
                    }),
                    steps: i + 1,
                };
            }
            t += (d * step_scale).max(min_step);
            if t > max_t {
                return MarchResult {
                    hit: None,
                    steps: i + 1,
                };
            }
        }
        MarchResult {
            hit: None,
            steps: 384,
        }
    }

    fn overlapping_sdfs_scene(p: Vec3, w_slice: f32) -> (f32, u32) {
        use glam::Vec4;
        let p4 = Vec4::new(p.x, p.y, p.z, w_slice);
        let twin_l = (Vec4::new(-0.6, 0.7, -1.5, 0.0), 0.7_f32);
        let twin_r = (Vec4::new(0.6, 0.7, -1.5, 0.0), 0.7_f32);
        let solo = (Vec4::new(0.0, 1.0, 1.5, 0.0), 1.0_f32);
        let d_twin_l = (p4 - twin_l.0).length() - twin_l.1;
        let d_twin_r = (p4 - twin_r.0).length() - twin_r.1;
        let d_solo = (p4 - solo.0).length() - solo.1;
        let d_floor = p.y;
        let d = d_twin_l.min(d_twin_r).min(d_solo).min(d_floor);
        (d, MAX_BODIES as u32)
    }

    #[test]
    fn cpu_march_hits_solo_sphere_top() {
        let ro = Vec3::new(0.0, 5.0, 1.5);
        let rd = Vec3::new(0.0, -1.0, 0.0);
        let hit = march_hyperslice_cpu(ro, rd, KERNEL_STEP_SCALE, |p| {
            overlapping_sdfs_scene(p, 0.0)
        })
        .hit
        .expect("ray pointing at solo sphere should hit something");
        assert_eq!(hit.body_idx, MAX_BODIES as u32, "static-scene hit");
        // Solo sphere top is y = 2.0; the hit registers within hit_eps of it.
        assert!(
            (hit.hit_pos.y - 2.0).abs() < 5e-3,
            "hit y {} should be near 2.0 (sphere top)",
            hit.hit_pos.y
        );
    }

    #[test]
    fn cpu_march_hits_floor_in_gap_between_spheres() {
        let ro = Vec3::new(0.0, 5.0, -3.5);
        let rd = Vec3::new(0.0, -1.0, 0.0);
        let hit = march_hyperslice_cpu(ro, rd, KERNEL_STEP_SCALE, |p| {
            overlapping_sdfs_scene(p, 0.0)
        })
        .hit
        .expect("ray pointing at empty floor should hit");
        assert_eq!(hit.body_idx, MAX_BODIES as u32, "static-scene hit");
        // Floor at y=0; hit_eps=0.001 so the hit y is in [0, 0.001].
        assert!(
            hit.hit_pos.y.abs() < 5e-3,
            "hit y {} should be near 0 (floor)",
            hit.hit_pos.y
        );
    }

    #[test]
    fn cpu_march_hits_twin_overlap_top_not_floor() {
        let ro = Vec3::new(0.0, 5.0, -1.5);
        let rd = Vec3::new(0.0, -1.0, 0.0);
        let hit = march_hyperslice_cpu(ro, rd, KERNEL_STEP_SCALE, |p| {
            overlapping_sdfs_scene(p, 0.0)
        })
        .hit
        .expect("ray pointing at twin overlap should hit");
        assert_eq!(hit.body_idx, MAX_BODIES as u32);
        // Twin-saddle top sits at y ≈ 1.06, well above the floor.
        assert!(
            hit.hit_pos.y > 0.5,
            "hit y {} should be on a sphere surface (>0.5), not the floor",
            hit.hit_pos.y
        );
    }

    #[test]
    fn cpu_march_converges_on_shallow_ray_to_floor() {
        let ro = Vec3::new(0.0, 2.5, 5.0);
        let rd = Vec3::new(0.0, -0.05, -1.0).normalize();
        let hit = march_hyperslice_cpu(ro, rd, KERNEL_STEP_SCALE, |p| {
            overlapping_sdfs_scene(p, 0.0)
        })
        .hit
        .expect("shallow ray with clear path to floor should converge");
        assert_eq!(hit.body_idx, MAX_BODIES as u32);
        assert!(hit.hit_pos.y.abs() < 5e-3);
    }

    #[test]
    fn cpu_march_misses_into_empty_sky() {
        let ro = Vec3::new(0.0, 5.0, 0.0);
        let rd = Vec3::new(0.0, 1.0, 0.0);
        let hit = march_hyperslice_cpu(ro, rd, KERNEL_STEP_SCALE, |p| {
            overlapping_sdfs_scene(p, 0.0)
        })
        .hit;
        assert!(hit.is_none(), "ray into sky should miss the scene");
    }

    // The step scale the kernel used before the Lipschitz certification, kept
    // only so the tests below can measure what replacing it bought.
    const SUPERSEDED_UNDER_STEP: f32 = 0.85;

    const PROBE_WIDTH: usize = 160;
    const PROBE_HEIGHT: usize = 90;

    struct ProbeBody {
        center: Vec4,
        size: f32,
        sdf: LocalSdf,
    }

    fn probe_body_sdf(p4: Vec4, body: &ProbeBody) -> f32 {
        let world_v = p4 - body.center;
        let world_dist2 = world_v.dot(world_v);
        let bound = body.size * 1.5;
        if world_dist2 > bound * bound {
            return world_dist2.sqrt() - body.size;
        }
        (body.sdf)(world_v / body.size) * body.size
    }

    fn probe_bodies() -> Vec<ProbeBody> {
        let shapes: [LocalSdf; 4] = [
            tesseract_sdf_local_cpu,
            cell16_sdf_local_cpu,
            cell24_sdf_local_cpu,
            sphere3_sdf_local_cpu,
        ];
        shapes
            .into_iter()
            .enumerate()
            .map(|(i, sdf)| ProbeBody {
                center: Vec4::new((i as f32 - 1.5) * 1.8, 0.9, 0.0, 0.0),
                size: 0.7,
                sdf,
            })
            .collect()
    }

    fn probe_scene(p3: Vec3, w_slice: f32, bodies: &[ProbeBody]) -> (f32, u32) {
        let p4 = Vec4::new(p3.x, p3.y, p3.z, w_slice);
        let mut dist = p3.y;
        let mut idx = MAX_BODIES as u32;
        for (i, body) in bodies.iter().enumerate() {
            let d = probe_body_sdf(p4, body);
            if d < dist {
                dist = d;
                idx = i as u32;
            }
        }
        (dist, idx)
    }

    struct ProbeFrame {
        depth: Vec<f32>,
        hit: Vec<bool>,
        primitive: Vec<u32>,
        steps: u64,
        width: usize,
        height: usize,
    }

    const NO_PRIMITIVE: u32 = u32::MAX;

    impl ProbeFrame {
        fn render(width: usize, height: usize, w_slice: f32, step_scale: f32) -> Self {
            let bodies = probe_bodies();
            let eye = Vec3::new(0.0, 1.2, 5.0);
            let forward = Vec3::new(0.0, -0.12, -1.0).normalize();
            let right = Vec3::X;
            let up = right.cross(forward).normalize();
            let fov_y_tan = (60.0_f32.to_radians() * 0.5).tan();
            let aspect = width as f32 / height as f32;

            let mut depth = vec![0.0_f32; width * height];
            let mut hit = vec![false; width * height];
            let mut primitive = vec![NO_PRIMITIVE; width * height];
            let mut steps = 0_u64;
            for y in 0..height {
                for x in 0..width {
                    let ndc_x = ((x as f32 + 0.5) / width as f32 * 2.0 - 1.0) * aspect;
                    let ndc_y = -((y as f32 + 0.5) / height as f32 * 2.0 - 1.0);
                    let rd = (forward + right * (ndc_x * fov_y_tan) + up * (ndc_y * fov_y_tan))
                        .normalize();
                    let marched = march_hyperslice_cpu(eye, rd, step_scale, |p| {
                        probe_scene(p, w_slice, &bodies)
                    });
                    let i = y * width + x;
                    steps += marched.steps as u64;
                    if let Some(h) = marched.hit {
                        depth[i] = (h.hit_pos - eye).length();
                        hit[i] = true;
                        primitive[i] = h.body_idx;
                    }
                }
            }
            Self {
                depth,
                hit,
                primitive,
                steps,
                width,
                height,
            }
        }

        // Bilinear on depth, nearest on the attribution (an interpolated primitive
        // index has no meaning), the cheapest upscale a half-res pass could use and
        // so the most favourable error it could report. `depth` is `Some` only
        // where all four taps attribute to the same primitive: blending across a
        // depth discontinuity measures the discontinuity, which the mask counts.
        fn upscale_to(&self, width: usize, height: usize) -> (Vec<Option<f32>>, Vec<u32>) {
            let mut depth = vec![None; width * height];
            let mut primitive = vec![NO_PRIMITIVE; width * height];
            for y in 0..height {
                for x in 0..width {
                    let fx = (x as f32 + 0.5) * self.width as f32 / width as f32 - 0.5;
                    let fy = (y as f32 + 0.5) * self.height as f32 / height as f32 - 0.5;
                    let x0 = (fx.floor().max(0.0) as usize).min(self.width - 1);
                    let y0 = (fy.floor().max(0.0) as usize).min(self.height - 1);
                    let x1 = (x0 + 1).min(self.width - 1);
                    let y1 = (y0 + 1).min(self.height - 1);
                    let ax = (fx - x0 as f32).clamp(0.0, 1.0);
                    let ay = (fy - y0 as f32).clamp(0.0, 1.0);
                    let prim_at = |xx: usize, yy: usize| self.primitive[yy * self.width + xx];
                    let taps = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)];
                    let uniform = prim_at(x0, y0) != NO_PRIMITIVE
                        && taps
                            .iter()
                            .all(|&(xx, yy)| prim_at(xx, yy) == prim_at(x0, y0));
                    if uniform {
                        let at = |xx: usize, yy: usize| self.depth[yy * self.width + xx];
                        depth[y * width + x] = Some(
                            (at(x0, y0) * (1.0 - ax) + at(x1, y0) * ax) * (1.0 - ay)
                                + (at(x0, y1) * (1.0 - ax) + at(x1, y1) * ax) * ay,
                        );
                    }
                    let nx = if ax < 0.5 { x0 } else { x1 };
                    let ny = if ay < 0.5 { y0 } else { y1 };
                    primitive[y * width + x] = prim_at(nx, ny);
                }
            }
            (depth, primitive)
        }
    }

    #[test]
    fn certified_full_step_matches_the_under_step_within_the_hit_epsilon() {
        for w_slice in [0.0_f32, 0.6] {
            let under =
                ProbeFrame::render(PROBE_WIDTH, PROBE_HEIGHT, w_slice, SUPERSEDED_UNDER_STEP);
            let full = ProbeFrame::render(PROBE_WIDTH, PROBE_HEIGHT, w_slice, KERNEL_STEP_SCALE);
            let mut worst_depth_delta = 0.0_f32;
            for i in 0..PROBE_WIDTH * PROBE_HEIGHT {
                assert!(
                    !under.hit[i] || full.hit[i],
                    "w={w_slice} pixel {i}: the full step lost a hit the under-step \
                     found, which a lower-bound sphere trace cannot legitimately do",
                );
                if under.hit[i] && full.hit[i] {
                    worst_depth_delta =
                        worst_depth_delta.max((under.depth[i] - full.depth[i]).abs());
                }
            }
            assert!(
                worst_depth_delta <= KERNEL_HIT_EPS,
                "w={w_slice}: depth moved by {worst_depth_delta}, more than the \
                 {KERNEL_HIT_EPS} hit epsilon the marcher stops within",
            );
        }
    }

    #[test]
    fn certified_full_step_costs_fewer_iterations_than_the_under_step() {
        for w_slice in [0.0_f32, 0.6] {
            let under =
                ProbeFrame::render(PROBE_WIDTH, PROBE_HEIGHT, w_slice, SUPERSEDED_UNDER_STEP);
            let full = ProbeFrame::render(PROBE_WIDTH, PROBE_HEIGHT, w_slice, KERNEL_STEP_SCALE);
            let saved = 1.0 - full.steps as f64 / under.steps as f64;
            println!(
                "w={w_slice}: march steps {} -> {} ({:+.1}%)",
                under.steps,
                full.steps,
                -100.0 * saved
            );
            assert!(
                saved > 0.08,
                "w={w_slice}: the full step saved only {:.1}% of {} steps; the \
                 under-step threw away 15% of every advance, so anything near \
                 zero means the step scale is not reaching the marcher",
                100.0 * saved,
                under.steps,
            );
        }
    }

    #[test]
    fn half_res_upscale_error_against_the_full_res_field_is_bounded() {
        // Measured on the probe scene and rounded up, so a change that makes
        // reconstruction worse is visible. Not a claim that this error is fine.
        const SAME_PRIMITIVE_DEPTH_LINF_BOUND: f32 = 2.0;
        const SAME_PRIMITIVE_DEPTH_RMS_BOUND: f64 = 0.3;
        const MISATTRIBUTION_FRACTION_BOUND: f64 = 0.02;

        for w_slice in [0.0_f32, 0.6] {
            let full = ProbeFrame::render(PROBE_WIDTH, PROBE_HEIGHT, w_slice, KERNEL_STEP_SCALE);
            let half = ProbeFrame::render(
                PROBE_WIDTH / 2,
                PROBE_HEIGHT / 2,
                w_slice,
                KERNEL_STEP_SCALE,
            );
            let (depth, primitive) = half.upscale_to(PROBE_WIDTH, PROBE_HEIGHT);

            let mut misattributed = 0_u64;
            let mut linf = 0.0_f32;
            let mut sum_sq = 0.0_f64;
            let mut compared = 0_u64;
            for i in 0..PROBE_WIDTH * PROBE_HEIGHT {
                if primitive[i] != full.primitive[i] {
                    misattributed += 1;
                    continue;
                }
                if let (true, Some(d)) = (full.hit[i], depth[i]) {
                    let e = (d - full.depth[i]).abs();
                    linf = linf.max(e);
                    sum_sq += (e as f64) * (e as f64);
                    compared += 1;
                }
            }
            let rms = (sum_sq / compared.max(1) as f64).sqrt();
            let misattribution_fraction =
                misattributed as f64 / (PROBE_WIDTH * PROBE_HEIGHT) as f64;
            println!(
                "w={w_slice}: half-res steps {} vs full {} ({:+.1}%), misattributed \
                 {:.2}% of pixels, same-primitive depth Linf {linf:.4} rms \
                 {rms:.6} over {compared} px",
                half.steps,
                full.steps,
                100.0 * (half.steps as f64 - full.steps as f64) / full.steps as f64,
                100.0 * misattribution_fraction,
            );

            assert!(
                linf <= SAME_PRIMITIVE_DEPTH_LINF_BOUND,
                "w={w_slice}: same-primitive depth Linf {linf}",
            );
            assert!(
                rms <= SAME_PRIMITIVE_DEPTH_RMS_BOUND,
                "w={w_slice}: same-primitive depth rms {rms}",
            );
            assert!(
                misattribution_fraction <= MISATTRIBUTION_FRACTION_BOUND,
                "w={w_slice}: misattributed {misattribution_fraction}",
            );
            // Not a rounding artifact: a half-res pass genuinely misattributes
            // pixels, and the marcher's own tolerance is 1e-3.
            assert!(
                misattributed > 0 && linf > KERNEL_HIT_EPS,
                "w={w_slice}: half-res reconstructed the full-res field exactly, \
                 which would make the no-half-res decision wrong",
            );
        }
    }

    #[test]
    fn kernel_marcher_takes_the_full_reported_distance() {
        assert!(HYPERSLICE_KERNEL_WGSL.contains("t = t + max(h.dist, min_step);"));
        assert!(!HYPERSLICE_KERNEL_WGSL.contains("h.dist * 0.85"));
    }

    fn strip_probe_cells() -> Vec<(crate::Viewport, f32, BodyUniform)> {
        crate::Viewport::full([192, 64])
            .split_horizontal(3)
            .into_iter()
            .zip([0.0_f32, 0.6, 0.9])
            .map(|(viewport, w_slice)| {
                (
                    viewport,
                    w_slice,
                    BodyUniform::sphere([0.0; 4], 1.0, [1.0, 0.0, 0.0]),
                )
            })
            .collect()
    }

    #[test]
    fn each_strip_cell_image_carries_its_own_slice() {
        let base = Hyperslice4DUniforms::default();
        let cells = strip_probe_cells();
        let images: Vec<_> = cells
            .iter()
            .map(|(viewport, w_slice, body)| strip_cell_uniforms(&base, *viewport, *w_slice, body))
            .collect();

        for (image, (viewport, w_slice, body)) in images.iter().zip(&cells) {
            assert_eq!(image.w_slice, *w_slice);
            assert_eq!(image.resolution, viewport.resolution_f32());
            assert_eq!(
                image.viewport_origin,
                [viewport.x as f32, viewport.y as f32]
            );
            assert_eq!(image.body_count, 1.0);
            assert_eq!(
                bytemuck::bytes_of(&image.bodies[0]),
                bytemuck::bytes_of(body)
            );
        }
        let last = images.last().expect("three cells");
        for image in &images[..images.len() - 1] {
            assert_ne!(
                image.w_slice, last.w_slice,
                "a cell holding the last cell's slice is the batching bug"
            );
        }
    }

    #[test]
    fn strip_cell_stride_is_offset_legal_and_covers_one_image() {
        // The alignments a WebGPU adapter may report: 32 is the spec floor,
        // 256 the common desktop value.
        for alignment in [32_u64, 64, 128, 256] {
            let stride = strip_cell_stride(alignment);
            assert_eq!(stride % alignment, 0, "stride must be an aligned offset");
            assert!(
                stride >= HYPERSLICE_UNIFORMS_SIZE,
                "stride {stride} would overlap the next cell's image",
            );
            assert!(
                stride - HYPERSLICE_UNIFORMS_SIZE < alignment,
                "stride {stride} wastes a whole alignment unit",
            );
        }
    }

    // A static surface would be `w`-independent and would dilute the per-cell
    // difference the probe measures.
    const EMPTY_SCENE_STUB: &str = r#"
const LOAM_PRIM_HYPERSPHERE4D: u32 = 0u;
const LOAM_PRIM_HALFSPACE4D: u32 = 1u;
const LOAM_PRIM_OTHER: u32 = 255u;
struct LoamSceneHit { dist: f32, kind: u32 }
fn loam_scene_at(p: vec3<f32>) -> LoamSceneHit {
    return LoamSceneHit(1.0e9, LOAM_PRIM_OTHER);
}
fn loam_scene_sdf(p: vec3<f32>) -> f32 {
    return loam_scene_at(p).dist;
}
fn loam_scene_max_t(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    return 1.0e9;
}
"#;

    async fn request_device() -> Result<(Device, Queue), String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("request_adapter failed: {e}"))?;
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("hyperslice4d-strip"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
                experimental_features: Default::default(),
            })
            .await
            .map_err(|e| format!("request_device failed: {e}"))
    }

    fn read_back_rgba(
        device: &Device,
        queue: &Queue,
        texture: &Texture,
        size: [u32; 2],
    ) -> Vec<u8> {
        let bytes_per_row = size[0] * 4;
        assert_eq!(
            bytes_per_row % COPY_BYTES_PER_ROW_ALIGNMENT,
            0,
            "probe width chosen so no row padding is needed"
        );
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("hyperslice4d strip readback"),
            size: (bytes_per_row * size[1]) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("hyperslice4d strip readback encoder"),
        });
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device
            .poll(PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("readback poll");
        let data = slice.get_mapped_range().to_vec();
        readback.unmap();
        data
    }

    fn strip_probe_module(device: &Device) -> ShaderModule {
        let polytope = super::super::polytope_data::polytope_extended_sdfs_wgsl();
        device.create_shader_module(ShaderModuleDescriptor {
            label: Some("hyperslice4d strip probe"),
            source: ShaderSource::Wgsl(
                format!("{HYPERSLICE_KERNEL_WGSL}\n{polytope}\n{EMPTY_SCENE_STUB}").into(),
            ),
        })
    }

    #[test]
    #[ignore = "requires a working wgpu adapter; run with --include-ignored"]
    fn each_strip_cell_renders_its_own_w_slice_gpu_probe() {
        const SIZE: [u32; 2] = [192, 64];
        let (device, queue) = pollster::block_on(request_device()).expect("wgpu device");

        let module = strip_probe_module(&device);
        let target = device.create_texture(&TextureDescriptor {
            label: Some("hyperslice4d strip probe target"),
            size: Extent3d {
                width: SIZE[0],
                height: SIZE[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&TextureViewDescriptor::default());

        let cells = strip_probe_cells();
        let mut node = Hyperslice4DNode::new(&device, TextureFormat::Rgba8Unorm, &module, 1);
        node.execute_strip(&device, &queue, &view, &cells)
            .expect("filmstrip should render");

        // The body is pure red and the sky gradient blue-dominant, so `r > b`
        // separates body from background without depending on shading constants.
        let pixels = read_back_rgba(&device, &queue, &target, SIZE);
        let footprints: Vec<usize> = cells
            .iter()
            .map(|(viewport, _, _)| {
                let mut covered = 0;
                for y in viewport.y..viewport.y + viewport.height {
                    for x in viewport.x..viewport.x + viewport.width {
                        let px = ((y * SIZE[0] + x) * 4) as usize;
                        if pixels[px] > pixels[px + 2] {
                            covered += 1;
                        }
                    }
                }
                covered
            })
            .collect();

        assert!(
            footprints[2] > 0,
            "the w = 0.9 slice should still show a body: {footprints:?}"
        );
        assert!(
            footprints[0] > footprints[1] && footprints[1] > footprints[2],
            "footprints should shrink with |w|; equal cells mean one uniform image \
             fed every cell: {footprints:?}"
        );
    }

    #[test]
    #[ignore = "requires a working wgpu adapter; run with --include-ignored"]
    fn execute_strip_leaves_the_single_slice_uniform_state_untouched_gpu_probe() {
        const SIZE: [u32; 2] = [192, 64];
        let (device, queue) = pollster::block_on(request_device()).expect("wgpu device");
        let module = strip_probe_module(&device);
        let target = device.create_texture(&TextureDescriptor {
            label: Some("hyperslice4d strip probe target"),
            size: Extent3d {
                width: SIZE[0],
                height: SIZE[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&TextureViewDescriptor::default());

        let mut node = Hyperslice4DNode::new(&device, TextureFormat::Rgba8Unorm, &module, 1);
        node.uniforms_mut().w_slice = -0.25;
        node.uniforms_mut().resolution = [640.0, 480.0];
        let before = *node.uniforms();

        node.execute_strip(&device, &queue, &view, &strip_probe_cells())
            .expect("filmstrip should render");

        let after = node.uniforms();
        assert!(
            bytemuck::bytes_of(after) == bytemuck::bytes_of(&before),
            "the strip must not leave a cell's slice, viewport or body in the \
             node's single-slice uniforms: w_slice was {} now {}, resolution \
             was {:?} now {:?}, body_count was {} now {}",
            before.w_slice,
            after.w_slice,
            before.resolution,
            after.resolution,
            before.body_count,
            after.body_count,
        );
    }
}
