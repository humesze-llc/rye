use ab_glyph::{point, Font as _, FontArc, Glyph, PxScale};
use std::borrow::Cow;

pub struct Hud {
    pub tex_size: (u32, u32),
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub bind_group: wgpu::BindGroup,
    pub pipeline: wgpu::RenderPipeline,
    pub vbuf: wgpu::Buffer,
    pub font: FontArc,
}

impl Hud {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let tex_size = (1024, 512);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HUD Texture"),
            size: wgpu::Extent3d {
                width: tex_size.0,
                height: tex_size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("HUD Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HUD Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("hud.wgsl"))),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HUD BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("HUD BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HUD Pipeline Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HUD VBuf"),
            size: (std::mem::size_of::<[f32; 4]>() * 6) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("HUD Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let font = FontArc::try_from_slice(include_bytes!("DejaVuSansMono.ttf"))
            .expect("place DejaVuSansMono.ttf next to hud.rs");

        Self {
            tex_size,
            texture,
            view,
            sampler,
            bind_group,
            pipeline,
            vbuf,
            font,
        }
    }

    pub fn update_vertices(
        &self,
        queue: &wgpu::Queue,
        surface_w: u32,
        surface_h: u32,
        margin_px: u32,
    ) {
        let w = surface_w.max(1) as f32;
        let h = surface_h.max(1) as f32;
        let hud_w = self.tex_size.0.min(surface_w) as f32;
        let hud_h = self.tex_size.1.min(surface_h) as f32;
        let m = margin_px as f32;

        let x0 = -1.0 + 2.0 * (m / w);
        let y0 = 1.0 - 2.0 * (m / h);
        let x1 = x0 + 2.0 * (hud_w / w);
        let y1 = y0 - 2.0 * (hud_h / h);

        let verts: [[f32; 4]; 6] = [
            [x0, y0, 0.0, 0.0],
            [x0, y1, 0.0, 1.0],
            [x1, y0, 1.0, 0.0],
            [x1, y0, 1.0, 0.0],
            [x0, y1, 0.0, 1.0],
            [x1, y1, 1.0, 1.0],
        ];

        let mut bytes = Vec::with_capacity(verts.len() * 4 * 4);
        for v in verts {
            for f in v {
                bytes.extend_from_slice(&f.to_ne_bytes());
            }
        }
        queue.write_buffer(&self.vbuf, 0, &bytes);
    }

    pub fn upload_text(&self, queue: &wgpu::Queue, text: &str) {
        let (w, h) = self.tex_size;
        let mut rgba = vec![0u8; (w * h * 4) as usize];

        let scale = PxScale::from(18.0);

        let upem = ab_glyph::Font::units_per_em(&self.font).unwrap_or(1000.0);
        let sx = scale.x / upem;
        let sy = scale.y / upem;

        let ascent_px = ab_glyph::Font::ascent_unscaled(&self.font) * sy;
        let height_px = ab_glyph::Font::height_unscaled(&self.font) * sy;
        let line_h = (height_px + 4.0).ceil() as i32;

        let mut y = 4i32 + ascent_px as i32;

        for line in text.lines() {
            let mut x = 6i32;

            for ch in line.chars() {
                let gid = ab_glyph::Font::glyph_id(&self.font, ch);
                let glyph = Glyph {
                    id: gid,
                    scale,
                    position: point(x as f32, y as f32),
                };

                if let Some(outlined) = ab_glyph::Font::outline_glyph(&self.font, glyph) {
                    outlined.draw(|gx, gy, cov| {
                        let px = x + gx as i32;
                        let py = y - ascent_px as i32 + gy as i32;
                        if px < 0 || py < 0 || px >= w as i32 || py >= h as i32 {
                            return;
                        }
                        let idx = ((py as u32 * w + px as u32) * 4) as usize;
                        let a = (cov * 255.0) as u8;
                        rgba[idx + 0] = 255;
                        rgba[idx + 1] = 255;
                        rgba[idx + 2] = 255;
                        rgba[idx + 3] = a.max(rgba[idx + 3]);
                    });
                }

                let adv_px = ab_glyph::Font::h_advance_unscaled(&self.font, gid) * sx;
                x += (adv_px + 1.0) as i32;
            }

            y += line_h;
            if y >= h as i32 {
                break;
            }
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vbuf.slice(..));
        pass.draw(0..6, 0..1);
    }
}
