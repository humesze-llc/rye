//! [`Hyperslice4DNode`] — render node for 4D scenes via
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
//! 1. [`HYPERSLICE_KERNEL_WGSL`] — uniform layout, fullscreen-
//!    triangle vertex stage, ray-march fragment stage. Calls
//!    `rye_scene_sdf` from the scene module.
//! 2. `Scene4::to_hyperslice_wgsl("u.w_slice")` — defines
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
//! - **Static scene primitives** via [`Scene4`]: hyperspheres at
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
/// at 32; pentatope-pile scenes for Simplex 4D top out around 20
/// active polytopes).
pub const MAX_BODIES: usize = 32;

/// One dynamic-body slot. Discriminated record covering both
/// hypersphere and polytope cases — `kind` selects which fields
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
/// | 48 | 32 | `rotor` (8 × f32packed as 2 × `vec4<f32>` — `rotor_lo` then `rotor_hi`; Rotor4 ordering: scalar, xy, xz, xw, yz, yw, zw, pseudoscalar) |
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
        Self {
            position: [0.0; 4],
            kind: BodyKind::Sphere as i32 as f32,
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
    /// index — 0 = pentatope, 1 = tesseract, etc.). Lands in the
    /// polytope-rendering chunk.
    Polytope = 1,
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
    // with bivector terms using positive r{xy,...} — but since we
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
    if (shape == 0u) {
        d = pentatope_sdf_local(unit_p);
    } else if (shape == 1u) {
        d = tesseract_sdf_local(unit_p);
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

// Normal estimate via central differences on the *combined*
// (static + dynamic) SDF, so dynamic-body surfaces shade correctly
// when they're the closest hit.
fn estimate_normal(p: vec3<f32>) -> vec3<f32> {
    let h = 0.001;
    let dx = rye_total_sdf(p + vec3<f32>(h, 0.0, 0.0)).dist
           - rye_total_sdf(p - vec3<f32>(h, 0.0, 0.0)).dist;
    let dy = rye_total_sdf(p + vec3<f32>(0.0, h, 0.0)).dist
           - rye_total_sdf(p - vec3<f32>(0.0, h, 0.0)).dist;
    let dz = rye_total_sdf(p + vec3<f32>(0.0, 0.0, h)).dist
           - rye_total_sdf(p - vec3<f32>(0.0, 0.0, h)).dist;
    return normalize(vec3<f32>(dx, dy, dz));
}

fn sky(rd: vec3<f32>) -> vec3<f32> {
    let t = (rd.y + 1.0) * 0.5;
    return mix(vec3<f32>(0.04, 0.05, 0.10), vec3<f32>(0.10, 0.13, 0.22), t);
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
    let max_t = 60.0;
    var hit = false;
    var hit_idx: u32 = MAX_BODIES + 1u;
    for (var i: i32 = 0; i < 192; i = i + 1) {
        let p = ro + rd * t;
        let h = rye_total_sdf(p);
        if (h.dist < 0.001) {
            hit = true;
            hit_idx = h.body_idx;
            break;
        }
        t = t + max(h.dist, 0.005);
        if (t > max_t) { break; }
    }

    if (!hit) {
        return vec4<f32>(sky(rd), 1.0);
    }

    let p_hit = ro + rd * t;
    let n = estimate_normal(p_hit);
    let light_dir = normalize(vec3<f32>(0.5, 0.85, 0.3));
    let lambert = max(dot(n, light_dir), 0.0);
    let ambient = 0.20;
    // Color: per-body color if a body was hit, neutral grey for
    // the static scene. Demos that need richer shading override
    // by writing their own fragment shader against this kernel's
    // uniform layout.
    var base = vec3<f32>(0.65, 0.65, 0.72);
    if (hit_idx < MAX_BODIES) {
        base = u.bodies[hit_idx].color;
    }
    let lit = base * (ambient + lambert * 0.85);
    let fog = 1.0 - exp(-t * 0.05);
    let final_color = mix(lit, sky(rd), fog * 0.5);
    return vec4<f32>(final_color, 1.0);
}
"#;

/// Render node that ray-marches the 3D cross-section of a 4D
/// scene at `u.w_slice`. Pairs with [`rye_sdf::Scene4`].
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
        // expects — the scene module must define it.
        assert!(HYPERSLICE_KERNEL_WGSL.contains("rye_scene_sdf("));
        // Dynamic-body machinery.
        assert!(HYPERSLICE_KERNEL_WGSL.contains("BodyUniform"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("rye_dynamic_bodies_sdf"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("BODY_KIND_SPHERE"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("BODY_KIND_POLYTOPE"));
        // Polytope-rendering chunk is now in the kernel.
        assert!(HYPERSLICE_KERNEL_WGSL.contains("body_polytope_sdf_4d"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("pentatope_sdf_local"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("tesseract_sdf_local"));
        assert!(HYPERSLICE_KERNEL_WGSL.contains("rotor4_inverse_apply"));
    }

    /// `BodyUniform` is exactly 80 bytes — the std140-aligned
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

    /// Default body is an identity-rotor sphere — safe to leave in
    /// unused slots (`set_body_count` excludes them but they're
    /// still read into the uniform buffer).
    #[test]
    fn default_body_is_identity() {
        let b = BodyUniform::default();
        // Identity rotor: scalar = 1, everything else = 0.
        assert_eq!(b.rotor, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(b.kind as i32, BodyKind::Sphere as i32);
    }
}
