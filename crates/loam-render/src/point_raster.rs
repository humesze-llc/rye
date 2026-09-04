use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2};
use loam_math::{Projection, RasterizableSpace};
use loam_shape::PointMesh;
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

const POINT_RASTER_WGSL: &str = include_str!("point_raster.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PointRasterUniforms {
    pub view_projection: [[f32; 4]; 4],
    pub viewport_size: [f32; 2],
    pub _pad: [f32; 2],
}

impl Default for PointRasterUniforms {
    fn default() -> Self {
        Self {
            view_projection: Mat4::IDENTITY.to_cols_array_2d(),
            viewport_size: [1.0, 1.0],
            _pad: [0.0; 2],
        }
    }
}

// Layout matches the `@location(1..=3)` attribute slots in `point_raster.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct PointInstance {
    pos: [f32; 3],
    radius_px: f32,
    color: [f32; 4],
}

/// Construct once per `RenderDevice`.
pub struct PointRasterNode {
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

impl PointRasterNode {
    pub fn new(
        device: &Device,
        surface_format: TextureFormat,
        depth: crate::DepthMode,
        sample_count: u32,
    ) -> Self {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("point_raster shader"),
            source: ShaderSource::Wgsl(POINT_RASTER_WGSL.into()),
        });

        let uniform_buf = device.create_buffer(&BufferDescriptor {
            label: Some("point_raster uniforms"),
            size: std::mem::size_of::<PointRasterUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("point_raster bgl"),
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
            label: Some("point_raster bg"),
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("point_raster pipeline layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

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
                format: VertexFormat::Float32x3,
                offset: 0,
                shader_location: 1,
            },
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 16,
                shader_location: 2,
            },
            VertexAttribute {
                format: VertexFormat::Float32,
                offset: 12,
                shader_location: 3,
            },
        ];
        let instance_layout = VertexBufferLayout {
            array_stride: std::mem::size_of::<PointInstance>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: &instance_attrs,
        };

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("point_raster pipeline"),
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
            label: Some("point_raster corner buffer"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2, 3]),
            usage: BufferUsages::VERTEX,
        });

        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("point_raster index buffer"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2, 2, 1, 3]),
            usage: BufferUsages::INDEX,
        });

        let instance_buf = device.create_buffer(&BufferDescriptor {
            label: Some("point_raster instance buffer"),
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

    /// Call before [`Self::record`] each frame.
    pub fn set_camera(&self, queue: &Queue, view_projection: Mat4, viewport_size: Vec2) {
        let uniforms = PointRasterUniforms {
            view_projection: view_projection.to_cols_array_2d(),
            viewport_size: viewport_size.to_array(),
            _pad: [0.0; 2],
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    }

    pub fn upload<S, const N: usize>(
        &mut self,
        device: &Device,
        queue: &Queue,
        mesh: &PointMesh<N>,
        projection: &Projection<N>,
    ) where
        S: RasterizableSpace<N>,
    {
        let n_points = mesh.positions.len();
        assert_eq!(
            mesh.colors.len(),
            n_points,
            "PointMesh invariant: colors.len() == positions.len()"
        );
        assert_eq!(
            mesh.sizes.len(),
            n_points,
            "PointMesh invariant: sizes.len() == positions.len()"
        );

        let mut instances: Vec<PointInstance> = Vec::with_capacity(n_points);
        for ((p, color), size) in mesh
            .positions
            .iter()
            .zip(mesh.colors.iter())
            .zip(mesh.sizes.iter())
        {
            let p_native = S::array_to_point(*p);
            let p3 = S::project_point(p_native, projection);
            // One non-finite point poisons the GPU divide into a full-screen quad.
            if !p3.is_finite() {
                continue;
            }
            instances.push(PointInstance {
                pos: p3.to_array(),
                radius_px: *size,
                color: *color,
            });
        }

        if instances.len() as u32 > self.instance_capacity {
            let new_cap = (instances.len() as u32).next_power_of_two().max(16);
            self.instance_buf = device.create_buffer(&BufferDescriptor {
                label: Some("point_raster instance buffer"),
                size: (new_cap as u64) * (std::mem::size_of::<PointInstance>() as u64),
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_capacity = new_cap;
        }

        if !instances.is_empty() {
            queue.write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(&instances));
        }
        self.instance_count = instances.len() as u32;
    }

    /// Records into the caller's encoder with `LoadOp::Load` on both attachments;
    /// `depth_view` is `Some` iff the pipeline has depth.
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        viewport: Option<&crate::Viewport>,
    ) {
        match (self.has_depth, depth_view.is_some()) {
            (true, false) => panic!(
                "PointRasterNode::record: pipeline was created with a depth format but no \
                 depth view was provided"
            ),
            (false, true) => panic!(
                "PointRasterNode::record: pipeline was created without a depth format but a \
                 depth view was provided"
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
            label: Some("point_raster pass"),
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
            vp.apply(&mut rp);
        }
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &self.bind_group, &[]);
        rp.set_vertex_buffer(0, self.corner_buf.slice(..));
        rp.set_vertex_buffer(1, self.instance_buf.slice(..));
        rp.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        rp.draw_indexed(0..6, 0, 0..self.instance_count);
    }
}

// Pins that `record` takes the caller's encoder and cannot submit.
const _: fn(
    &PointRasterNode,
    &mut wgpu::CommandEncoder,
    &wgpu::TextureView,
    Option<&wgpu::TextureView>,
    Option<&crate::Viewport>,
) = PointRasterNode::record;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_raster_wgsl_validates() {
        let module = naga::front::wgsl::parse_str(POINT_RASTER_WGSL)
            .unwrap_or_else(|e| panic!("point_raster WGSL parse failed:\n{e}"));
        let flags = naga::valid::ValidationFlags::all();
        let caps = naga::valid::Capabilities::empty();
        naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .expect("point_raster WGSL must validate");
    }
}
