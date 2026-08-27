//! For R⁴ line meshes that do not change between frames, with the rotation
//! and Perspective4D projection running on the GPU. Dropping the per-frame
//! instance upload also halves wasm JS-interop pressure: each
//! `queue.write_buffer` spawns short-lived JS proxies.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2};
use loam_math::Rotor4;
use loam_shape::LineMesh;
use wgpu::util::DeviceExt;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BlendComponent, BlendFactor, BlendOperation, BlendState,
    Buffer, BufferBindingType, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites,
    CompareFunction, DepthStencilState, Device, FragmentState, LoadOp, MultisampleState,
    Operations, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, Queue,
    RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor,
    RenderPipeline, RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    StencilState, StoreOp, TextureFormat, VertexAttribute, VertexBufferLayout, VertexFormat,
    VertexState, VertexStepMode,
};

const SHADER_WGSL: &str = include_str!("line_raster_static_r4.wgsl");

/// Uniform layout matching the `TransformUniform` struct in the WGSL shader.
/// 144 bytes; padded to 16-byte alignment for `std140`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LineRasterStaticR4Uniforms {
    pub rotor_matrix: [[f32; 4]; 4],
    pub view_projection: [[f32; 4]; 4],
    pub viewport_size: [f32; 2],
    pub focal_distance: f32,
    pub _pad: f32,
}

impl Default for LineRasterStaticR4Uniforms {
    fn default() -> Self {
        Self {
            rotor_matrix: Mat4::IDENTITY.to_cols_array_2d(),
            view_projection: Mat4::IDENTITY.to_cols_array_2d(),
            viewport_size: [1.0, 1.0],
            // Neutral focal distance giving the cube-within-cube tesseract view
            // for unit-circumradius polytopes; keeps a pre-set_transform
            // record() from producing NaNs.
            focal_distance: 2.0,
            _pad: 0.0,
        }
    }
}

// 80 bytes, matching the R³ `LineInstance`, with full Vec4 positions instead
// of Vec3 + pad.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct LineInstance4D {
    start_pos: [f32; 4],
    end_pos: [f32; 4],
    start_color: [f32; 4],
    end_color: [f32; 4],
    width_px: f32,
    _pad: [f32; 3],
}

/// One instance per edge, uploaded once; the per-frame uniform carries the
/// rotor, camera, viewport and focal distance.
pub struct LineRasterStaticR4Node {
    pipeline: RenderPipeline,
    uniform_buf: Buffer,
    bind_group: BindGroup,

    corner_buf: Buffer,
    index_buf: Buffer,
    instance_buf: Buffer,
    instance_count: u32,
    instance_capacity: u32,
    has_depth: bool,
}

impl LineRasterStaticR4Node {
    /// `surface_format`, `depth` and `sample_count` must match the attachments at
    /// draw time, as in [`crate::LineRasterNode::new`].
    pub fn new(
        device: &Device,
        surface_format: TextureFormat,
        depth: crate::DepthMode,
        sample_count: u32,
    ) -> Self {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("line_raster_static_r4 shader"),
            source: ShaderSource::Wgsl(SHADER_WGSL.into()),
        });

        let uniform_buf = device.create_buffer(&BufferDescriptor {
            label: Some("line_raster_static_r4 uniforms"),
            size: std::mem::size_of::<LineRasterStaticR4Uniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // The buffer is zero-cleared at creation (wgpu invariant); the first
        // set_transform before any record() establishes valid data.
        let defaults = LineRasterStaticR4Uniforms::default();
        let _ = defaults;

        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("line_raster_static_r4 bgl"),
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
            label: Some("line_raster_static_r4 bg"),
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("line_raster_static_r4 pipeline layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        // One u32 corner index per vertex; instance attributes fill WGSL
        // @location(1..=5), positions as Float32x4 to carry w.
        let corner_attrs = [VertexAttribute {
            format: VertexFormat::Uint32,
            offset: 0,
            shader_location: 0,
        }];
        let corner_layout = VertexBufferLayout {
            array_stride: std::mem::size_of::<u32>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: &corner_attrs,
        };
        let instance_attrs = [
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 0,
                shader_location: 1,
            },
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 16,
                shader_location: 2,
            },
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 32,
                shader_location: 3,
            },
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 48,
                shader_location: 4,
            },
            VertexAttribute {
                format: VertexFormat::Float32,
                offset: 64,
                shader_location: 5,
            },
        ];
        let instance_layout = VertexBufferLayout {
            array_stride: std::mem::size_of::<LineInstance4D>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: &instance_attrs,
        };

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("line_raster_static_r4 pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[corner_layout, instance_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: surface_format,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::SrcAlpha,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: depth.format().map(|format| DepthStencilState {
                format,
                depth_write_enabled: depth.writes(),
                depth_compare: CompareFunction::LessEqual,
                stencil: StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        let corner_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("line_raster_static_r4 corner buffer"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2, 3]),
            usage: BufferUsages::VERTEX,
        });

        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("line_raster_static_r4 index buffer"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2, 2, 1, 3]),
            usage: BufferUsages::INDEX,
        });

        let instance_buf = device.create_buffer(&BufferDescriptor {
            label: Some("line_raster_static_r4 instance buffer"),
            size: 64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            uniform_buf,
            bind_group,
            corner_buf,
            index_buf,
            instance_buf,
            instance_count: 0,
            instance_capacity: 0,
            has_depth: depth.is_active(),
        }
    }

    /// Intended to be called once, or on topology change; per-frame rotation goes
    /// through [`Self::set_transform`]. Allocates a scratch `Vec` per call, so
    /// per-frame uploads belong in [`crate::LineRasterNode`] instead.
    pub fn upload_mesh(&mut self, device: &Device, queue: &Queue, mesh: &LineMesh<4>) {
        let n = mesh.segments.len();
        debug_assert_eq!(mesh.colors.len(), n);
        debug_assert_eq!(mesh.widths.len(), n);

        let mut instances: Vec<LineInstance4D> = Vec::with_capacity(n);
        for ((seg, (color_a, color_b)), &width) in mesh
            .segments
            .iter()
            .zip(mesh.colors.iter())
            .zip(mesh.widths.iter())
        {
            instances.push(LineInstance4D {
                start_pos: seg.0,
                end_pos: seg.1,
                start_color: *color_a,
                end_color: *color_b,
                width_px: width,
                _pad: [0.0; 3],
            });
        }

        let needed = instances.len() as u32;
        if needed > self.instance_capacity {
            let new_cap = needed.next_power_of_two().max(16);
            self.instance_buf = device.create_buffer(&BufferDescriptor {
                label: Some("line_raster_static_r4 instance buffer"),
                size: (new_cap as u64) * (std::mem::size_of::<LineInstance4D>() as u64),
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_capacity = new_cap;
        }

        if !instances.is_empty() {
            queue.write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(&instances));
        }
        self.instance_count = needed;
    }

    /// One `queue.write_buffer` call per frame on the hot path.
    pub fn set_transform(
        &self,
        queue: &Queue,
        rotor: Rotor4,
        view_projection: Mat4,
        viewport_size: Vec2,
        focal_distance: f32,
    ) {
        let uniforms = LineRasterStaticR4Uniforms {
            rotor_matrix: rotor.to_mat4(),
            view_projection: view_projection.to_cols_array_2d(),
            viewport_size: viewport_size.to_array(),
            focal_distance,
            _pad: 0.0,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Same `LoadOp::Load` discipline and depth-attachment contract as
    /// [`crate::LineRasterNode::record`].
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        viewport: Option<&crate::Viewport>,
    ) {
        match (self.has_depth, depth_view.is_some()) {
            (true, false) => panic!(
                "LineRasterStaticR4Node::record: pipeline was created with a depth format but \
                 no depth view was provided"
            ),
            (false, true) => panic!(
                "LineRasterStaticR4Node::record: pipeline was created without a depth format \
                 but a depth view was provided"
            ),
            _ => {}
        }
        if self.instance_count == 0 {
            return;
        }
        let depth_attachment = depth_view.map(|dv| RenderPassDepthStencilAttachment {
            view: dv,
            depth_ops: Some(Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        });
        let mut rp = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("line_raster_static_r4 pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        if let Some(vp) = viewport {
            rp.set_viewport(
                vp.x as f32,
                vp.y as f32,
                vp.width as f32,
                vp.height as f32,
                0.0,
                1.0,
            );
        }
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &self.bind_group, &[]);
        rp.set_vertex_buffer(0, self.corner_buf.slice(..));
        rp.set_vertex_buffer(1, self.instance_buf.slice(..));
        rp.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        rp.draw_indexed(0..6, 0, 0..self.instance_count);
    }

    pub fn instance_count(&self) -> u32 {
        self.instance_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_uniforms_match_wgsl_size() {
        // Must match WGSL `TransformUniform`: 64 + 64 + 8 + 4 + 4 = 144.
        assert_eq!(std::mem::size_of::<LineRasterStaticR4Uniforms>(), 144);
        assert_eq!(std::mem::align_of::<LineRasterStaticR4Uniforms>(), 4);
    }

    #[test]
    fn instance_size_matches_r3_node() {
        assert_eq!(std::mem::size_of::<LineInstance4D>(), 80);
    }

    #[test]
    fn shader_wgsl_validates() {
        let module = naga::front::wgsl::parse_str(SHADER_WGSL)
            .unwrap_or_else(|e| panic!("line_raster_static_r4 WGSL parse failed:\n{e}"));
        let flags = naga::valid::ValidationFlags::all();
        let caps = naga::valid::Capabilities::empty();
        naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .expect("line_raster_static_r4 WGSL must validate");
    }
}
