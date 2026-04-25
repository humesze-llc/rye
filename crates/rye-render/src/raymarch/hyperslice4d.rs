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
//! ## Scope today
//!
//! - **Static scenes only.** The `Scene4` is captured at
//!   construction; primitive parameters (sphere centres,
//!   half-space normals) become WGSL constants. To change body
//!   positions per frame, rebuild the node — slow.
//! - **No `ConvexPolytope4D` support.** Polytope SDFs in 4D need
//!   per-frame face-hyperplane uniforms, which would inflate the
//!   uniform layout substantially. The polytope `Primitive4` emit
//!   returns a sentinel today; this node renders polytope leaves
//!   as invisible far-away surfaces. Polytope dynamic-body
//!   support is the headliner of step 4 (the `pentatope_slice`
//!   migration) — when it lands, the node grows a body-uniform
//!   array.
//! - **Hyperslice only.** Native 4D ray-march (full 4D camera) is
//!   a separate node, deferred per `4D_RENDERING.md`.

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use wgpu::*;

use crate::device::RenderDevice;
use crate::graph::RenderNode;

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
    pub _pad3: f32,
    pub _pad4: f32,
    pub _pad5: f32,
    /// Four scalar knobs for user-shader-side parameters. Same
    /// shape as `RayMarchUniforms::params` for symmetry.
    pub params: [f32; 4],
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
            _pad3: 0.0,
            _pad4: 0.0,
            _pad5: 0.0,
            params: [0.0; 4],
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
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn estimate_normal(p: vec3<f32>) -> vec3<f32> {
    let h = 0.001;
    let dx = rye_scene_sdf(p + vec3<f32>(h, 0.0, 0.0)) - rye_scene_sdf(p - vec3<f32>(h, 0.0, 0.0));
    let dy = rye_scene_sdf(p + vec3<f32>(0.0, h, 0.0)) - rye_scene_sdf(p - vec3<f32>(0.0, h, 0.0));
    let dz = rye_scene_sdf(p + vec3<f32>(0.0, 0.0, h)) - rye_scene_sdf(p - vec3<f32>(0.0, 0.0, h));
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
    for (var i: i32 = 0; i < 192; i = i + 1) {
        let p = ro + rd * t;
        let d = rye_scene_sdf(p);
        if (d < 0.001) {
            hit = true;
            break;
        }
        t = t + max(d, 0.005);
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
    let base = vec3<f32>(0.65, 0.65, 0.72);
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
}

impl RenderNode for Hyperslice4DNode {
    fn name(&self) -> &'static str {
        "hyperslice4d"
    }

    fn execute(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        let mut encoder = rd
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
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
        // expects — the scene module must define it. The kernel
        // refers to it inside `fs_main` and `estimate_normal`.
        assert!(HYPERSLICE_KERNEL_WGSL.contains("rye_scene_sdf("));
    }
}
