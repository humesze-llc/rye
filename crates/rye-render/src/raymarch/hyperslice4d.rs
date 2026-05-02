//! [`Hyperslice4DNode`], render node for 4D scenes via
//! hyperslicing.
//!
//! Designed to pair with `rye_sdf::Scene4` but takes a
//! pre-compiled [`wgpu::ShaderModule`] rather than depending on
//! `rye-sdf` directly (matches the existing
//! [`crate::raymarch::RayMarchNode`] / [`crate::raymarch::GeodesicRayMarchNode`]
//! pattern; keeps `rye-render`'s deps minimal).
//!
//! The user assembles the WGSL by concatenating:
//!
//! 1. [`HYPERSLICE_KERNEL_WGSL`], uniform layout, fullscreen-
//!    triangle vertex stage, ray-march fragment stage. Calls
//!    `rye_scene_sdf` from the scene module.
//! 2. `Scene4::to_hyperslice_wgsl("u.w_slice")`, defines
//!    `rye_scene_sdf` as `4D_SDF(vec4(p, u.w_slice))`.
//!
//! ```ignore
//! let kernel = rye_render::raymarch::HYPERSLICE_KERNEL_WGSL;
//! let scene_wgsl = scene.to_hyperslice_wgsl("u.w_slice");
//! let source = format!("{kernel}\n{scene_wgsl}");
//! let module = device.create_shader_module(...);
//! let node = Hyperslice4DNode::new(device, format, &module);
//! ```
//!
//! ## What it renders
//!
//! - **Static scene primitives** via `Scene4`: hyperspheres at
//!   fixed centres, half-spaces, etc. The `Scene4` is captured at
//!   construction; its primitive parameters become WGSL constants.
//! - **Dynamic bodies** via the [`BodyUniform`] array on
//!   [`Hyperslice4DUniforms`]. Up to 32 bodies per frame; each
//!   slot is a discriminated record covering both
//!   `HyperSphere4D`-shaped bodies (kind = 0) and
//!   `ConvexPolytope4D`-shaped bodies (kind = 1, lands in the
//!   polytope-rendering chunk after this one). Per-frame updates
//!   come through [`Hyperslice4DNode::set_bodies`] /
//!   [`Hyperslice4DNode::set_body_count`] without recompiling the
//!   shader.
//! - **Hyperslice only.** Native 4D ray-march (full 4D camera) is
//!   a separate node, deferred per `4D_RENDERING.md`.
//!
//! The kernel sums the static-scene SDF (`rye_scene_sdf`, defined
//! by the user's `Scene4` emit) and the dynamic-body SDF
//! (`rye_dynamic_bodies_sdf`, defined in the kernel itself) via
//! `min`. So a typical scene composes a static floor (`Scene4`
//! `HalfSpace4D`) plus N moving hyperspheres uploaded each frame
//! through the body array.

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use wgpu::*;

use crate::device::RenderDevice;
use crate::graph::RenderNode;

/// Maximum number of dynamic bodies a single frame can render.
/// Hard cap because the uniform layout is fixed-size; raising it
/// is a recompile, not a runtime knob. 32 is generous for the
/// demos we know about (`hypersphere`'s `-n N` is already capped
/// at 32; pentatope-pile scenes top out around 20 active
/// polytopes).
pub const MAX_BODIES: usize = 32;

/// Shape table indices for `BodyKind::Polytope` bodies. Stored in
/// [`BodyUniform::radius_or_shape`] (cast through `f32`); read by
/// the kernel's `body_polytope_sdf_4d` dispatch chain. Mirrored as
/// `SHAPE_*` constants in [`HYPERSLICE_KERNEL_WGSL`]; keep in sync.
pub const SHAPE_PENTATOPE: u32 = 0;
pub const SHAPE_TESSERACT: u32 = 1;
pub const SHAPE_16CELL: u32 = 2;
pub const SHAPE_24CELL: u32 = 3;

/// One dynamic-body slot. Discriminated record covering both
/// hypersphere and polytope cases, `kind` selects which fields
/// are read by the shader.
///
/// Layout (std140-aligned, 80 bytes total):
///
/// | offset | bytes | field |
/// |---|---|---|
/// |  0 | 16 | `position` (`vec4<f32>`) |
/// | 16 |  4 | `kind` (`f32`: 0 = sphere, 1 = polytope) |
/// | 20 |  4 | `radius_or_shape` (sphere radius / polytope shape index) |
/// | 24 |  4 | `polytope_size` (polytope circumradius; ignored when kind = sphere) |
/// | 28 |  4 | `_pad0` |
/// | 32 | 12 | `color` (`vec3<f32>`) |
/// | 44 |  4 | `_pad1` |
/// | 48 | 32 | `rotor` (8 × f32packed as 2 × `vec4<f32>`, `rotor_lo` then `rotor_hi`; Rotor4 ordering: scalar, xy, xz, xw, yz, yw, zw, pseudoscalar) |
///
/// The rotor lives in two `vec4<f32>` slots in the WGSL struct so
/// the std140 alignment matches Rust's tightly-packed `[f32; 8]`
/// (an `array<f32, 8>` in a uniform buffer would pad each element
/// to 16 bytes, blowing the slot to 128 bytes). The `rotor` slot is
/// identity-valued for sphere bodies; it's loaded from the body's
/// `RigidBody::orientation.rotation` for polytope bodies.
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
        // `Invalid` has no kernel dispatch branch; the SDF accumulator
        // stays at its 1e9 initial value for this slot rather than
        // collapsing to a zero-radius sphere at the origin.
        Self {
            position: [0.0; 4],
            kind: BodyKind::Invalid as i32 as f32,
            radius_or_shape: 0.0,
            polytope_size: 0.0,
            _pad0: 0.0,
            color: [0.7, 0.7, 0.7],
            _pad1: 0.0,
            // Identity Rotor4: scalar=1, bivector=0, pseudoscalar=0.
            rotor: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    }
}

/// `BodyUniform::kind` discriminator. Cast to `f32` when writing
/// the uniform; the shader reads it back as `u32`-via-`f32`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BodyKind {
    /// `HyperSphere4D`. Reads `position` (4D centre) and
    /// `radius_or_shape` (sphere radius). Ignores `rotor`.
    Sphere = 0,
    /// `ConvexPolytope4D`. Reads `position` (body origin in 4D),
    /// `rotor` (orientation), and `radius_or_shape` (shape table
    /// index, 0 = pentatope, 1 = tesseract, etc.). Lands in the
    /// polytope-rendering chunk.
    Polytope = 1,
    /// Sentinel for slots the kernel must skip. The dispatch chain in
    /// `rye_dynamic_bodies_sdf` matches neither sphere nor polytope
    /// branches, so the SDF accumulator keeps its 1e9 initial value
    /// for this slot. `BodyUniform::default()` produces this kind so
    /// uninitialised slots in `Hyperslice4DUniforms::bodies` are inert.
    ///
    /// `255` is chosen as `u8::MAX`: a value that's both far outside
    /// the live discriminator range (today {0, 1}) and that round-trips
    /// cleanly through the `f32`-typed `BodyUniform::kind` field
    /// (every integer in `[-2^24, 2^24]` is exactly representable).
    Invalid = 255,
}

impl BodyUniform {
    /// Build a sphere body. `position` is the world-space 4D
    /// centre, `radius` is the 4-ball radius, `color` is the
    /// per-body hue.
    pub fn sphere(position: [f32; 4], radius: f32, color: [f32; 3]) -> Self {
        Self {
            position,
            kind: BodyKind::Sphere as i32 as f32,
            radius_or_shape: radius,
            color,
            ..Self::default()
        }
    }

    /// Build a polytope body. `shape_index` references the kernel's
    /// shape table (0 = pentatope, 1 = tesseract, ...). `size` is
    /// the polytope's circumradius in world coords. `rotor` is the
    /// body's Rotor4 orientation as `[s, b_xy, b_xz, b_xw, b_yz,
    /// b_yw, b_zw, ps]`.
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
}

/// Uniform buffer for [`Hyperslice4DNode`]. Bind group 0,
/// binding 0. `std140`-compatible layout matching the kernel's
/// `Uniforms` struct.
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
    /// The slicing hyperplane's `w` coordinate. `Scene4`'s
    /// hyperslice emit reads `u.w_slice` directly.
    pub w_slice: f32,
    /// Number of active body slots. Cast from `u32` to `f32` for
    /// std140 alignment (the kernel rounds back to integer).
    pub body_count: f32,
    pub _pad3: f32,
    pub _pad4: f32,
    /// Four scalar knobs for user-shader-side parameters. Same
    /// shape as `RayMarchUniforms::params` for symmetry.
    pub params: [f32; 4],
    /// Dynamic body slots. Slots `>= body_count` are unread by
    /// the shader. See [`BodyUniform`] for the per-slot layout.
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
            _pad3: 0.0,
            _pad4: 0.0,
            params: [0.0; 4],
            bodies: [BodyUniform::default(); MAX_BODIES],
        }
    }
}

/// Hyperslice ray-march kernel. Defines `Uniforms`, the
/// fullscreen triangle, and the ray-march loop. The user's
/// `Scene4` emit fills in `rye_scene_sdf(p: vec3<f32>) -> f32`.
/// Public so callers can build the full shader source themselves
/// (kernel + scene emit).
pub const HYPERSLICE_KERNEL_WGSL: &str = r#"
// ---- Hyperslice4DNode kernel ----

const MAX_BODIES: u32 = 32u;

const BODY_KIND_SPHERE: u32 = 0u;
const BODY_KIND_POLYTOPE: u32 = 1u;

// Polytope shape table indices. Mirrored from rye-render Rust-side
// `SHAPE_*` constants; keep in sync.
const SHAPE_PENTATOPE: u32 = 0u;
const SHAPE_TESSERACT: u32 = 1u;
const SHAPE_16CELL: u32 = 2u;
const SHAPE_24CELL: u32 = 3u;
// Mirrors `BodyKind::Invalid` (CPU). Intentionally absent from the
// dispatch chain in `rye_dynamic_bodies_sdf` and `rye_total_sdf` below:
// neither the sphere nor the polytope branch matches, so the SDF
// accumulator keeps its 1e9 initial value for that slot. `255` is the
// CPU-side `u8::MAX` sentinel, far outside the live discriminator
// range. Do NOT delete: `BodyUniform::default()` produces this kind so
// uninitialised slots are inert. CPU/GPU protocol breaks if removed.
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
    _pad3: f32,
    _pad4: f32,
    params: vec4<f32>,
    bodies: array<BodyUniform, MAX_BODIES>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

// SDF of a single sphere body in 4D, evaluated at `p4`.
fn body_sphere_sdf_4d(p4: vec4<f32>, b: BodyUniform) -> f32 {
    return length(p4 - b.position) - b.radius_or_shape;
}

// ---- Rotor4 sandwich (inverse rotation: world -> body local) ----
//
// `Rotor4::apply` (CPU) computes the forward sandwich `R̃ · v · R`,
// rotating a body-local vector into world coordinates. To go the
// other way (world -> body local) we flip the bivector signs of `R`
// to get its reverse `R̃`, then run the same formula with R̃ as the
// "rotor". That equals `R · v · R̃`, the inverse rotation.
//
// Component order matches `Rotor4` { s, xy, xz, xw, yz, yw, zw, xyzw }
// packed into rotor_lo (s, xy, xz, xw) and rotor_hi (yz, yw, zw, xyzw).
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

    // Stage 1: R̃ · v. R̃ for the inner rotor here re-flips the
    // bivector signs (back to original R's bivector signs); the
    // formula below is the direct port of `Rotor4::apply` Stage 1
    // with bivector terms using positive r{xy,...}, but since we
    // already inverted the signs above, this works out to using the
    // negated values. Keeping the formula identical to the CPU
    // implementation:
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

// ---- Convex polytope SDFs (body-local, unit circumradius) ----
//
// Per-shape SDFs assume the polytope is centered at the origin and
// oriented in its canonical frame. The dispatcher transforms world
// coordinates into body-local coordinates first (translate +
// inverse-rotor), then scales the result by the body's circumradius.

// Pentatope (5-cell, 4D simplex) at unit circumradius. Five face
// hyperplanes; signed distance is the max plane distance.
//
// Vertex set (matching `rye_physics::euclidean_r4::pentatope_vertices(1.0)`):
//   v0 = (0, 0, 0, 1)
//   v1 = (t, t, t, -0.25), v2 = (t, -t, -t, -0.25),
//   v3 = (-t, t, -t, -0.25), v4 = (-t, -t, t, -0.25)
// where t = sqrt(15) / (4 * sqrt(3)) = sqrt(5) / 4 ≈ 0.55901699.
//
// Face i is opposite vertex i; outward normal is -v_i (since |v_i|
// = 1 for unit-circumradius). Inradius for an n-simplex is R/n; for
// pentatope (n=4) at R=1 the inradius is 0.25.
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

// Tesseract (8-cell / hypercube) at unit circumradius. Vertices at
// (±0.5, ±0.5, ±0.5, ±0.5); faces at ±0.5 along each axis. SDF is
// the standard infinity-norm box form.
fn tesseract_sdf_local(p: vec4<f32>) -> f32 {
    let q = abs(p) - vec4<f32>(0.5, 0.5, 0.5, 0.5);
    let outside = length(max(q, vec4<f32>(0.0, 0.0, 0.0, 0.0)));
    let inside = min(max(max(q.x, q.y), max(q.z, q.w)), 0.0);
    return outside + inside;
}

// 16-cell (cross-polytope / hexadecachoron) at unit circumradius.
// Vertices at ±e_x, ±e_y, ±e_z, ±e_w. Face normals are the 16
// unit vectors `(±1, ±1, ±1, ±1) / 2`; each face is at perpendicular
// distance 0.5 from origin (inradius). The max-over-faces signed
// plane distance reduces in any octant to:
//
//     (|p.x| + |p.y| + |p.z| + |p.w| - 1) / 2
//
// The `/ 2` is the unit-normal normalisation, without it the
// function returns the L1 distance (twice the Euclidean), which
// over-estimates the true SDF and causes sphere-tracing tunneling
// (rays step past the surface, surface appears to "disappear" or
// shift when the camera orbits).
fn cell16_sdf_local(p: vec4<f32>) -> f32 {
    let q = abs(p);
    return (q.x + q.y + q.z + q.w - 1.0) * 0.5;
}

// 24-cell (icositetrachoron) at unit circumradius. The 24-cell is
// the intersection of a tesseract scaled to 1/sqrt(2) (so its
// vertices land at distance 1) with a 16-cell scaled to sqrt(2)
// (so its faces tangent the same sphere). The intersection's
// vertices are the 24 permutations of (±1/sqrt(2), ±1/sqrt(2), 0, 0):
// the canonical 24-cell vertex set.
//
// Intersection of two convex shapes: SDF = max(sdf_a, sdf_b).
// The cross-polytope component carries the same `/ 2` correction
// as `cell16_sdf_local`.
fn cell24_sdf_local(p: vec4<f32>) -> f32 {
    let inv_sqrt2: f32 = 0.70710678;
    let sqrt2:     f32 = 1.41421356;
    let q = abs(p);
    let tess  = max(max(q.x, q.y), max(q.z, q.w)) - inv_sqrt2;
    let cross = (q.x + q.y + q.z + q.w - sqrt2) * 0.5;
    return max(tess, cross);
}

// Dispatcher: world-space `p4` against polytope body `b`. Translates
// to body origin, applies the inverse rotor (world -> body local),
// scales by 1/size to evaluate the unit-circumradius shape, then
// rescales the resulting SDF.
fn body_polytope_sdf_4d(p4: vec4<f32>, b: BodyUniform) -> f32 {
    let size = max(b.polytope_size, 1.0e-6);
    let world_v = p4 - b.position;
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
    }
    return d * size;
}

// SDF of all dynamic bodies at `p3`, evaluated at the slicing
// hyperplane `w = u.w_slice`. Returns +infinity if no body is
// active or none cover the slice.
fn rye_dynamic_bodies_sdf(p3: vec3<f32>) -> f32 {
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

// SDF of a single dynamic body at `p3`, evaluated at the slicing
// hyperplane `w = u.w_slice`. Used by `estimate_normal` to compute
// per-surface gradients without contamination from other SDFs that
// happen to be nearby in chart space (issue #17: a polytope close
// to the floor was contaminating the floor's normal estimate near
// the polytope's silhouette, dropping `n.y` below the floor-checker
// threshold and producing visible discoloration in the polytope's
// xy-footprint).
//
// Returns +infinity for invalid indices or kinds.
fn rye_body_sdf_at(p3: vec3<f32>, body_idx: u32) -> f32 {
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
// for the static scene; MAX_BODIES + 1 for nothing). The kernel
// fills this in during ray march; the user's `fs_main` reads it
// to drive shading.
struct HitInfo {
    dist: f32,
    body_idx: u32,
};

// Combined SDF: min(static scene, dynamic bodies). Returns the
// distance to the closer surface plus the body index it came
// from (MAX_BODIES if the static scene is closer).
fn rye_total_sdf(p3: vec3<f32>) -> HitInfo {
    let scene_d = rye_scene_sdf(p3);

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

// Normal estimate via central differences on the *dominating* SDF
// at the hit point, dispatched by `body_idx`:
//   body_idx >= MAX_BODIES => the static scene was closest; sample
//                             only `rye_scene_sdf`.
//   otherwise              => that body was closest; sample only
//                             that body's SDF in isolation.
//
// Sampling the combined `rye_total_sdf` here (the previous
// behaviour) blends contributions from neighbouring SDFs near
// silhouettes, which produces a wrong normal exactly at the visible
// edge of every dynamic body (issue #17). The dispatch keeps each
// surface's normal honest at the price of one branch per fragment.
fn estimate_normal(p: vec3<f32>, body_idx: u32) -> vec3<f32> {
    let h = 0.001;
    if (body_idx >= MAX_BODIES) {
        let dx = rye_scene_sdf(p + vec3<f32>(h, 0.0, 0.0))
               - rye_scene_sdf(p - vec3<f32>(h, 0.0, 0.0));
        let dy = rye_scene_sdf(p + vec3<f32>(0.0, h, 0.0))
               - rye_scene_sdf(p - vec3<f32>(0.0, h, 0.0));
        let dz = rye_scene_sdf(p + vec3<f32>(0.0, 0.0, h))
               - rye_scene_sdf(p - vec3<f32>(0.0, 0.0, h));
        return normalize(vec3<f32>(dx, dy, dz));
    }
    let dx = rye_body_sdf_at(p + vec3<f32>(h, 0.0, 0.0), body_idx)
           - rye_body_sdf_at(p - vec3<f32>(h, 0.0, 0.0), body_idx);
    let dy = rye_body_sdf_at(p + vec3<f32>(0.0, h, 0.0), body_idx)
           - rye_body_sdf_at(p - vec3<f32>(0.0, h, 0.0), body_idx);
    let dz = rye_body_sdf_at(p + vec3<f32>(0.0, 0.0, h), body_idx)
           - rye_body_sdf_at(p - vec3<f32>(0.0, 0.0, h), body_idx);
    return normalize(vec3<f32>(dx, dy, dz));
}

fn sky(rd: vec3<f32>) -> vec3<f32> {
    let t = (rd.y + 1.0) * 0.5;
    return mix(vec3<f32>(0.04, 0.05, 0.10), vec3<f32>(0.10, 0.13, 0.22), t);
}

// Soft 1m-square checkerboard. Used by `fs_main` to shade static-scene
// hits with a near-vertical normal (i.e. y=0 floors), which is the
// common case for the 4D demos. Helps depth perception against an
// otherwise flat grey plane.
fn ground_color(p: vec3<f32>) -> vec3<f32> {
    let g = floor(p.x) + floor(p.z);
    let alt = abs(g - 2.0 * floor(g * 0.5));
    let dark = vec3<f32>(0.18, 0.20, 0.24);
    let light = vec3<f32>(0.30, 0.32, 0.36);
    return mix(dark, light, alt);
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

    var t: f32 = 0.0;
    // Analytical upper bound from any HalfSpace4D leaves the scene
    // emits via `rye_scene_max_t`. Capped at the demo-scale 60.0 so a
    // scene without analytical contributions falls back to the legacy
    // far-clip. The +1.0 buffer past the analytical bound lets the
    // marcher land its hit on the floor surface itself rather than
    // exiting one iteration short. Without this bound, near-horizon
    // rays exhaust the iteration budget converging on the floor and
    // return sky, producing the horizon-banding artifact at shallow
    // viewing angles.
    let scene_max_t = rye_scene_max_t(ro, rd);
    let max_t = min(60.0, scene_max_t + 1.0);
    var hit = false;
    var hit_idx: u32 = MAX_BODIES + 1u;
    // Sphere-trace step calculus (issue #17):
    //   * `hit_eps = 0.001` is the surface-hit gate.
    //   * `step = max(h.dist * 0.85, min_step)`. The 0.85 under-step
    //     factor matches the geodesic kernel and is the safety net
    //     for SDFs that are bounded but not Lipschitz-1 (e.g. the
    //     inlined polytope SDFs here, which underestimate true
    //     Euclidean distance in corner regions).
    //   * `min_step = 0.0001` is one tenth of `hit_eps`, deliberately.
    //     The previous value (0.005) was *larger* than `hit_eps`, so
    //     when an SDF dipped into the (0.001, 0.005) range -- which
    //     happens on every approach to a polytope silhouette --
    //     the marcher was forced to overshoot its actual clearance.
    //   * 384 iterations: the under-step factor + small min_step
    //     means tangent-grazing rays (those that just barely miss
    //     a polytope silhouette and produce screen-space edges
    //     between shapes) need more iterations to traverse the
    //     near-tangent zone where `d` stays small. The earlier 192
    //     cap was sized for the unsafe `min_step = 0.005` regime;
    //     the safe regime needs roughly 6x more iterations through
    //     a tangent zone, and 384 covers it with headroom while
    //     adding negligible cost on non-grazing rays (which exit
    //     in <30 steps via the under-step factor's geometric
    //     convergence).
    let hit_eps = 0.001;
    let min_step = 0.0001;
    for (var i: i32 = 0; i < 384; i = i + 1) {
        let p = ro + rd * t;
        let h = rye_total_sdf(p);
        if (h.dist < hit_eps) {
            hit = true;
            hit_idx = h.body_idx;
            break;
        }
        t = t + max(h.dist * 0.85, min_step);
        if (t > max_t) { break; }
    }

    if (!hit) {
        return vec4<f32>(sky(rd), 1.0);
    }

    let p_hit = ro + rd * t;
    let n = estimate_normal(p_hit, hit_idx);
    let light_dir = normalize(vec3<f32>(0.5, 0.85, 0.3));
    let lambert = max(dot(n, light_dir), 0.0);
    let ambient = 0.20;
    // Color: per-body color if a body was hit; for the static scene,
    // checker for floor-like surfaces (normal close to +Y) and a
    // neutral grey otherwise. Demos that need richer shading still
    // override by writing their own fragment shader against this
    // kernel's uniform layout.
    var base = vec3<f32>(0.65, 0.65, 0.72);
    if (hit_idx < MAX_BODIES) {
        base = u.bodies[hit_idx].color;
    } else if (rye_scene_at(p_hit).kind == RYE_PRIM_HALFSPACE4D) {
        // Floor classification: route on the closest primitive's
        // identity, not a normal/position heuristic. Scene4's emit
        // attaches a `kind` tag to each leaf and propagates it
        // through union/intersection so `rye_scene_at` returns the
        // active boundary's primitive type. `RYE_PRIM_HALFSPACE4D`
        // is a half-space (the conventional "floor" in rye demos).
        // The previous version gated on the surface normal plus the
        // hit y-position, which mis-classified sphere tops at y=0
        // and only worked for floors anchored at y=0 specifically.
        base = ground_color(p_hit);
    }
    let lit = base * (ambient + lambert * 0.85);
    let fog = 1.0 - exp(-t * 0.05);
    let final_color = mix(lit, sky(rd), fog * 0.5);
    return vec4<f32>(final_color, 1.0);
}
"#;

/// Render node that ray-marches the 3D cross-section of a 4D
/// scene at `u.w_slice`. Pairs with `rye_sdf::Scene4`.
pub struct Hyperslice4DNode {
    pipeline: RenderPipeline,
    uniforms: Hyperslice4DUniforms,
    uniform_buf: Buffer,
    bind_group: BindGroup,
    clear_color: Color,
}

impl Hyperslice4DNode {
    /// Build the node from a pre-compiled [`ShaderModule`]. The
    /// caller is responsible for producing it from the kernel
    /// ([`HYPERSLICE_KERNEL_WGSL`]) + their scene's hyperslice
    /// WGSL emit. See the module-level docs for an example.
    pub fn new(device: &Device, surface_format: TextureFormat, module: &ShaderModule) -> Self {
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
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            uniforms: Hyperslice4DUniforms::default(),
            uniform_buf,
            bind_group,
            clear_color: Color::BLACK,
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

    /// Replace the active body slots with `bodies` and set the
    /// active count. Bodies past `bodies.len()` keep their previous
    /// values but aren't read by the shader. Caller-managed: this
    /// doesn't auto-flush; pair with [`Self::flush_uniforms`].
    pub fn set_bodies(&mut self, bodies: &[BodyUniform]) {
        let n = bodies.len().min(MAX_BODIES);
        self.uniforms.bodies[..n].copy_from_slice(&bodies[..n]);
        self.uniforms.body_count = n as f32;
    }

    /// Update one body slot in-place. Useful for per-frame body
    /// updates where you've already populated the slots in
    /// [`Self::set_bodies`] and only their positions change.
    pub fn set_body(&mut self, index: usize, body: BodyUniform) {
        if index < MAX_BODIES {
            self.uniforms.bodies[index] = body;
        }
    }

    /// Override the active body count without rewriting slot data.
    pub fn set_body_count(&mut self, count: usize) {
        self.uniforms.body_count = count.min(MAX_BODIES) as f32;
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

    /// The kernel exposes the expected entry points and uniform
    /// layout. Full naga validation happens at the call site when
    /// the user assembles `kernel + scene_emit` and constructs a
    /// real `ShaderModule`; here we only sanity-check the textual
    /// kernel.
    #[test]
    fn kernel_has_expected_entry_points() {
        assert!(HYPERSLICE_KERNEL_WGSL.contains("@vertex"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("fn vs_fullscreen"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("@fragment"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("fn fs_main"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("struct Uniforms"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("@group(0) @binding(0)"));
        // The scene's `rye_scene_sdf` is the contract the kernel
        // expects, the scene module must define it.
        assert!(HYPERSLICE_KERNEL_WGSL.contains("rye_scene_sdf("));
        // Dynamic-body machinery.
        assert!(HYPERSLICE_KERNEL_WGSL.contains("BodyUniform"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("rye_dynamic_bodies_sdf"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("BODY_KIND_SPHERE"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("BODY_KIND_POLYTOPE"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("BODY_KIND_INVALID"));
        // Floor-checker helper used for static-scene shading.
        assert!(HYPERSLICE_KERNEL_WGSL.contains("fn ground_color"));
        // Polytope-rendering chunk is now in the kernel.
        assert!(HYPERSLICE_KERNEL_WGSL.contains("body_polytope_sdf_4d"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("pentatope_sdf_local"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("tesseract_sdf_local"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("cell16_sdf_local"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("cell24_sdf_local"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("rotor4_inverse_apply"));
        // Per-body SDF accessor used by `estimate_normal` to avoid
        // sampling the combined SDF at silhouettes (issue #17).
        assert!(HYPERSLICE_KERNEL_WGSL.contains("rye_body_sdf_at"));
        // Floor classification routes on per-primitive identity
        // (kind tag from Scene4's emit), not the legacy
        // n.y/p_hit.y heuristic. Pinning the new contract here so
        // a future revert to the heuristic fails loudly.
        assert!(HYPERSLICE_KERNEL_WGSL.contains("rye_scene_at(p_hit).kind == RYE_PRIM_HALFSPACE4D"));
        assert!(!HYPERSLICE_KERNEL_WGSL.contains("abs(p_hit.y) < 0.01"));
        // Analytical max-t shortcut from `rye_scene_max_t` caps the
        // marcher's far-clip so near-horizon rays don't exhaust the
        // iteration budget. Pin the call so a future refactor that
        // drops it fails loudly.
        assert!(HYPERSLICE_KERNEL_WGSL.contains("rye_scene_max_t(ro, rd)"));
    }

    /// `BodyUniform` is exactly 80 bytes, the std140-aligned
    /// layout the kernel's `BodyUniform` struct expects. Off-by-
    /// padding bugs would surface here as a different size.
    #[test]
    fn body_uniform_is_80_bytes() {
        assert_eq!(std::mem::size_of::<BodyUniform>(), 80);
    }

    /// Constructors set kind discriminator correctly.
    #[test]
    fn body_uniform_constructors_set_kind() {
        let s = BodyUniform::sphere([0.0; 4], 1.0, [0.5, 0.5, 0.5]);
        assert_eq!(s.kind as i32, BodyKind::Sphere as i32);
        let p = BodyUniform::polytope(
            [0.0; 4],
            0,
            1.0,
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5; 3],
        );
        assert_eq!(p.kind as i32, BodyKind::Polytope as i32);
        assert_eq!(p.polytope_size, 1.0);
    }

    /// Default body is inert (`kind = Invalid`) so an unused slot
    /// in `Hyperslice4DUniforms::bodies` can't accidentally render.
    /// Also pins the identity-rotor initialization so a future
    /// refactor that zeros the whole rotor array is caught.
    #[test]
    fn default_body_is_inert_invalid_kind() {
        let b = BodyUniform::default();
        assert_eq!(b.kind as i32, BodyKind::Invalid as i32);
        assert_eq!(b.rotor, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    /// Naga-validate the full kernel concatenated with a minimal
    /// scene stub. Catches WGSL syntax / type / binding mismatches
    /// the string-presence assertions above can't see (e.g. a
    /// rotor-sandwich edit that drops a swizzle, a uniform field
    /// renamed without updating the WGSL struct).
    ///
    /// The stub mirrors the contract `Scene4::to_hyperslice_wgsl`
    /// emits: `rye_scene_at(p3) -> RyeSceneHit` plus the kind
    /// constants the kernel references. `rye_scene_sdf` is provided
    /// as a thin wrapper for completeness even though `rye_total_sdf`
    /// reads `dist` via `rye_scene_at` directly in the production
    /// path through Scene4. `rye_scene_max_t` returns the legacy
    /// no-analytical-bound sentinel so the kernel falls back to its
    /// hard-coded far-clip; Scene4's emit overrides this with a real
    /// ray-plane bound when the scene has half-space leaves.
    #[test]
    fn kernel_validates_with_minimal_scene() {
        const SCENE_STUB: &str = r#"
const RYE_PRIM_HYPERSPHERE4D: u32 = 0u;
const RYE_PRIM_HALFSPACE4D: u32 = 1u;
const RYE_PRIM_OTHER: u32 = 255u;
struct RyeSceneHit { dist: f32, kind: u32 }
fn rye_scene_at(p: vec3<f32>) -> RyeSceneHit {
    return RyeSceneHit(length(p) - 0.5, RYE_PRIM_OTHER);
}
fn rye_scene_sdf(p: vec3<f32>) -> f32 {
    return rye_scene_at(p).dist;
}
fn rye_scene_max_t(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    return 1.0e9;
}
"#;
        let source = format!("{HYPERSLICE_KERNEL_WGSL}\n{SCENE_STUB}");
        let module = naga::front::wgsl::parse_str(&source)
            .expect("hyperslice4d kernel + scene stub should parse as WGSL");
        let flags = naga::valid::ValidationFlags::all();
        let caps = naga::valid::Capabilities::empty();
        naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .expect("hyperslice4d kernel + scene stub should validate");
    }

    /// End-to-end validation: take a real `Scene4` (the
    /// `overlapping_sdfs` example's shape: a union of hyperspheres
    /// and a half-space floor), emit its hyperslice WGSL, splice
    /// against the kernel, parse + validate via naga.
    ///
    /// Catches drift between the `Scene4::to_hyperslice_wgsl` emit
    /// and what the kernel expects at the `rye_scene_sdf` boundary.
    /// The `kernel_validates_with_minimal_scene` test above pins
    /// the kernel's call-site shape against a hand-rolled stub;
    /// this one pins the `Scene4` emit produces a stub the kernel
    /// can actually call.
    #[test]
    fn kernel_validates_with_real_scene_union() {
        use glam::Vec4;
        use rye_sdf::{Scene4, SceneNode4};

        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::new(-0.6, 0.7, -1.5, 0.0), 0.7)
                .union(SceneNode4::hypersphere(Vec4::new(0.6, 0.7, -1.5, 0.0), 0.7))
                .union(SceneNode4::hypersphere(Vec4::new(0.0, 1.0, 1.5, 0.0), 1.0))
                .union(SceneNode4::halfspace(Vec4::Y, 0.0)),
        );
        let scene_wgsl = scene.to_hyperslice_wgsl("u.w_slice");
        let source = format!("{HYPERSLICE_KERNEL_WGSL}\n{scene_wgsl}");
        let module = naga::front::wgsl::parse_str(&source)
            .expect("hyperslice4d kernel + Scene4 emit should parse as WGSL");
        let flags = naga::valid::ValidationFlags::all();
        let caps = naga::valid::Capabilities::empty();
        naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .expect("hyperslice4d kernel + Scene4 emit should validate");
    }

    /// `BODY_KIND_INVALID` must not appear in either dispatch chain.
    /// The whole point of the sentinel is that no branch matches it,
    /// so the SDF accumulator passes through unchanged. If a future
    /// edit adds a comparison against `BODY_KIND_INVALID` (in either
    /// operand order), the "inert default" guarantee is broken.
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

    // ---- Polytope SDF parity (CPU port vs WGSL formula) -----------------
    //
    // Each `*_sdf_local_cpu` here is a 1:1 port of the matching WGSL
    // function in `HYPERSLICE_KERNEL_WGSL`. The parity tests below
    // assert the *geometry* the SDF is meant to represent (vertex
    // positions, inradius, sign at sample points), so a port that
    // silently diverges from the WGSL fails the test even though the
    // CPU<->WGSL formulas are textually parallel.

    use glam::Vec4;

    fn pentatope_sdf_local_cpu(p: Vec4) -> f32 {
        // Mirror of `pentatope_sdf_local` in HYPERSLICE_KERNEL_WGSL.
        // t = sqrt(15) / (4 * sqrt(3)) = sqrt(5) / 4. WGSL stores the
        // 11-digit truncation 0.55901699437; the CPU side reconstructs
        // from the closed form so float precision is f32-clean.
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
        // WGSL stores 8-digit truncations (0.70710678 and 1.41421356);
        // the CPU side uses the standard library constants for
        // f32-clean values. Tolerance in the test absorbs the
        // CPU-vs-WGSL precision delta.
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let sqrt2 = std::f32::consts::SQRT_2;
        let q = p.abs();
        let tess = q.x.max(q.y).max(q.z).max(q.w) - inv_sqrt2;
        let cross = (q.x + q.y + q.z + q.w - sqrt2) * 0.5;
        tess.max(cross)
    }

    // ---- Inline vertex generators (mirror rye_physics::euclidean_r4) ----

    fn pentatope_vertices() -> Vec<Vec4> {
        // Apex plus regular tetrahedron in the w = -1/4 hyperplane;
        // matches `rye_physics::euclidean_r4::pentatope_vertices(1.0)`.
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

    /// Shared parity assertions: every vertex on the surface, origin
    /// at -inradius, scaled-out point clearly outside, scaled-in point
    /// clearly inside.
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
        // Inradius for an n-simplex at unit circumradius is R/n; for
        // the pentatope (n=4) that's 0.25.
        assert_polytope_geometry(
            "pentatope",
            pentatope_sdf_local_cpu,
            &pentatope_vertices(),
            0.25,
            // Pentatope normals carry sqrt(15)/(4 sqrt(3)) ~ 0.559;
            // truncated to 11 digits in the WGSL constant. Per-vertex
            // residual is dominated by that truncation, ~5e-7 in
            // measurement.
            5e-6,
        );
    }

    #[test]
    fn tesseract_cpu_port_matches_geometry() {
        // Tesseract at circumradius 1 has half-edge a = 0.5;
        // inradius (perpendicular distance to a cubic cell) = 0.5.
        assert_polytope_geometry(
            "tesseract",
            tesseract_sdf_local_cpu,
            &tesseract_vertices(),
            0.5,
            // Tesseract SDF is built from `abs` and `min`/`max`; the
            // vertex residual is bit-exact for the half-extent 0.5.
            1e-7,
        );
    }

    #[test]
    fn cell16_cpu_port_matches_geometry() {
        // 16-cell at circumradius 1: faces are (+/-, +/-, +/-, +/-)/2
        // hyperplanes at perpendicular distance 0.5.
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
        // 24-cell vertices land at distance 1 from the origin (set by
        // construction). Inradius is 1/sqrt(2) ~ 0.7071: the
        // intersection of tesseract(1/sqrt(2)) and 16-cell(sqrt(2))
        // yields equidistant faces at that radius.
        assert_polytope_geometry(
            "24-cell",
            cell24_sdf_local_cpu,
            &cell24_vertices(),
            std::f32::consts::FRAC_1_SQRT_2,
            // CPU port uses f32 standard-library consts; WGSL ships
            // 8-digit truncations of the same values. The vertex
            // residual is dominated by that delta.
            5e-7,
        );
    }

    /// The CPU SDF ports above must remain textual mirrors of the
    /// WGSL formulas. This guards against drift by re-checking the
    /// load-bearing literals appear in the kernel source.
    #[test]
    fn polytope_sdf_constants_match_kernel_source() {
        // Pentatope: t = 0.55901699437 and r = 0.25.
        assert!(HYPERSLICE_KERNEL_WGSL.contains("0.55901699437"));
        // 24-cell: 1/sqrt(2) and sqrt(2) literals.
        assert!(HYPERSLICE_KERNEL_WGSL.contains("0.70710678"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("1.41421356"));
    }

    /// `BodyUniform::polytope(shape_index, ..)` writes the same numeric
    /// table the kernel branches on. Catches the failure mode where one
    /// side renumbers without the other.
    #[test]
    fn shape_constants_mirror_kernel_table() {
        for (rust_const, wgsl_decl) in [
            (SHAPE_PENTATOPE, "const SHAPE_PENTATOPE: u32 = 0u;"),
            (SHAPE_TESSERACT, "const SHAPE_TESSERACT: u32 = 1u;"),
            (SHAPE_16CELL, "const SHAPE_16CELL: u32 = 2u;"),
            (SHAPE_24CELL, "const SHAPE_24CELL: u32 = 3u;"),
        ] {
            assert!(
                HYPERSLICE_KERNEL_WGSL.contains(wgsl_decl),
                "kernel missing `{wgsl_decl}` for Rust value {rust_const}"
            );
        }
    }

    // ---- CPU port of the hyperslice marcher -----------------------------
    //
    // 1:1 mirror of `fs_main`'s sphere-trace loop in `HYPERSLICE_KERNEL_WGSL`.
    // Used by the integration tests below to assert hit positions against a
    // known SDF without needing a GPU adapter (the existing GPU probes in
    // `rye-shader::db` are `#[ignore]`d locally and run via lavapipe).
    //
    // The kernel reads its constants (`hit_eps`, `min_step`, iteration cap,
    // `max_t`, under-step factor) inline; the CPU port pins the same values
    // here. If the kernel changes them, this port must be updated too.

    use glam::Vec3;

    struct HyperHit {
        hit_pos: Vec3,
        body_idx: u32,
        #[allow(dead_code)]
        iter_count: u32,
    }

    fn march_hyperslice_cpu<F>(ro: Vec3, rd: Vec3, sdf: F) -> Option<HyperHit>
    where
        F: Fn(Vec3) -> (f32, u32),
    {
        let mut t: f32 = 0.0;
        let max_t = 60.0_f32;
        let hit_eps = 0.001_f32;
        let min_step = 0.0001_f32;
        for i in 0..384u32 {
            let p = ro + rd * t;
            let (d, body_idx) = sdf(p);
            if d < hit_eps {
                return Some(HyperHit {
                    hit_pos: p,
                    body_idx,
                    iter_count: i,
                });
            }
            t += (d * 0.85).max(min_step);
            if t > max_t {
                return None;
            }
        }
        None
    }

    /// Static-scene SDF for the `overlapping_sdfs` example: three
    /// hyperspheres at `w = 0` plus a `y = 0` floor, evaluated via 4D
    /// distance at `(p, w_slice)`. All hits attribute to the static scene
    /// (`body_idx = MAX_BODIES`), matching what the kernel's
    /// `rye_total_sdf` returns when no dynamic bodies are active.
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

    // ---- Integration tests against the CPU port -------------------------
    //
    // These exercise the *whole* marcher pipeline (step calculus + SDF
    // composition + hit attribution) on the `overlapping_sdfs` scene, the
    // same scene the example renders. Each test pins a property the visual
    // verification confirmed; together they form a regression net for the
    // issue #17 fix at the geometry level (not just the WGSL string level).

    /// A ray pointing straight down through the solo sphere's centre
    /// hits the sphere's *top* at y close to 2.0 (centre y + radius).
    /// Without the iteration-cap bump (192 -> 384) and the under-step
    /// factor, near-tangent silhouette rays exhausted the budget; the
    /// solo sphere is large enough that this central ray converges
    /// fast, but it pins the basic "hit the top, body_idx = static"
    /// contract.
    #[test]
    fn cpu_march_hits_solo_sphere_top() {
        let ro = Vec3::new(0.0, 5.0, 1.5);
        let rd = Vec3::new(0.0, -1.0, 0.0);
        let hit = march_hyperslice_cpu(ro, rd, |p| overlapping_sdfs_scene(p, 0.0))
            .expect("ray pointing at solo sphere should hit something");
        assert_eq!(hit.body_idx, MAX_BODIES as u32, "static-scene hit");
        // Solo sphere at (0, 1, 1.5) r=1; top at y=2.0. Hit registers
        // when SDF < hit_eps = 0.001; under-step factor + min_step
        // bound the residual at < 5e-3.
        assert!(
            (hit.hit_pos.y - 2.0).abs() < 5e-3,
            "hit y {} should be near 2.0 (sphere top)",
            hit.hit_pos.y
        );
    }

    /// A ray pointing straight down between the spheres hits the floor
    /// at y close to 0. Pins the no-dimple property: the static-scene
    /// `min(spheres, floor)` correctly returns the floor's distance
    /// when no sphere covers the ray's path.
    #[test]
    fn cpu_march_hits_floor_in_gap_between_spheres() {
        let ro = Vec3::new(0.0, 5.0, -3.5);
        let rd = Vec3::new(0.0, -1.0, 0.0);
        let hit = march_hyperslice_cpu(ro, rd, |p| overlapping_sdfs_scene(p, 0.0))
            .expect("ray pointing at empty floor should hit");
        assert_eq!(hit.body_idx, MAX_BODIES as u32, "static-scene hit");
        // Floor at y=0; hit_eps=0.001 so the hit y is in [0, 0.001].
        assert!(
            hit.hit_pos.y.abs() < 5e-3,
            "hit y {} should be near 0 (floor)",
            hit.hit_pos.y
        );
    }

    /// A ray pointing at the twin-pair overlap region (x = 0,
    /// z = -1.5) hits one of the twin spheres' top, NOT the floor.
    /// Pins the `min(sphere_l, sphere_r, floor)` composition: at the
    /// twin overlap, both sphere SDFs are small (the surfaces meet),
    /// the floor SDF is ~0.7 (the sphere top is at y=1.4). The
    /// marcher must prefer the closer sphere surface.
    #[test]
    fn cpu_march_hits_twin_overlap_top_not_floor() {
        let ro = Vec3::new(0.0, 5.0, -1.5);
        let rd = Vec3::new(0.0, -1.0, 0.0);
        let hit = march_hyperslice_cpu(ro, rd, |p| overlapping_sdfs_scene(p, 0.0))
            .expect("ray pointing at twin overlap should hit");
        assert_eq!(hit.body_idx, MAX_BODIES as u32);
        // Twin spheres at (+/-0.6, 0.7, -1.5) r=0.7. At x=z=-1.5, the
        // intersection of the twin tops along the y-axis is at y where
        // both spheres touch: solving (0.6)^2 + (y-0.7)^2 = 0.7^2 gives
        // (y-0.7)^2 = 0.13, y ≈ 0.7 + 0.36 = 1.06 (the twin-saddle top).
        // Hit must be above the floor by clearly more than hit_eps.
        assert!(
            hit.hit_pos.y > 0.5,
            "hit y {} should be on a sphere surface (>0.5), not the floor",
            hit.hit_pos.y
        );
    }

    /// A ray with shallow downward angle on a clear path to the floor
    /// converges within 384 iterations. Pins the iteration-cap fix:
    /// the original 192 cap exhausted on shallow rays; 384 covers
    /// them with headroom.
    #[test]
    fn cpu_march_converges_on_shallow_ray_to_floor() {
        // Camera at y=2.5, ray angled gently down to the horizon.
        // Hits the floor at t ≈ 50 (y=2.5, descending 0.05/unit).
        let ro = Vec3::new(0.0, 2.5, 5.0);
        let rd = Vec3::new(0.0, -0.05, -1.0).normalize();
        let hit = march_hyperslice_cpu(ro, rd, |p| overlapping_sdfs_scene(p, 0.0))
            .expect("shallow ray with clear path to floor should converge");
        assert_eq!(hit.body_idx, MAX_BODIES as u32);
        assert!(hit.hit_pos.y.abs() < 5e-3);
    }

    /// A ray going straight up from the camera into empty space misses
    /// (returns None). Pins the iteration-cap and `max_t` cap exit
    /// paths; without them the marcher would loop forever.
    #[test]
    fn cpu_march_misses_into_empty_sky() {
        let ro = Vec3::new(0.0, 5.0, 0.0);
        let rd = Vec3::new(0.0, 1.0, 0.0);
        let hit = march_hyperslice_cpu(ro, rd, |p| overlapping_sdfs_scene(p, 0.0));
        assert!(hit.is_none(), "ray into sky should miss the scene");
    }
}
