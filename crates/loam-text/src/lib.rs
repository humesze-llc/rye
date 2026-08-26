//! `loam-text`: screen-space text rendering for game HUDs and overlays.
//!
//! No global state, no `App`-trait coupling: construct a [`TextRenderer`] in
//! `setup`, [`queue`](TextRenderer::queue) strings per frame, and
//! [`record`](TextRenderer::record) into the frame's encoder after the main
//! scene's passes.
//!
//! [`ab_glyph`] rasterization + a hand-rolled wgpu atlas/textured-quad pipeline.
//! Printable ASCII (`0x20..=0x7E`) is pre-baked at a fixed atlas size; per-call
//! sizes scale the quads bilinearly. For HUD readouts only, not typographic or
//! non-Latin text; reach for `loam-egui` when an app needs more.
//!
//! [`glyph`] is the other consumer of the same `ab_glyph` font handle: a
//! build-time pipeline turning letters into extruded, slab-embedded 4D solids
//! that are simultaneously render geometry and physics colliders.
//!
//! # Example
//!
//! ```ignore
//! let mut text = TextRenderer::new(device, queue, format, font_bytes, 48.0, 1)?;
//! text.queue("Score: 1234", [16.0, 16.0], 32.0, [1.0, 1.0, 1.0, 1.0]);
//! text.record(device, queue, &mut encoder, &view, [width, height]);
//! ```
//!
//! # Limitations
//!
//! - **Printable ASCII only.** [`queue`](TextRenderer::queue) drops everything
//!   else silently, with no fallback box glyph. [`is_renderable`] is the
//!   pre-check for strings that must survive intact.
//! - **No shaping.** Layout is a per-char advance sum: no kerning, ligatures,
//!   combining marks, or bidi. Proportional faces therefore cannot align into
//!   columns; a caller wanting alignment must supply a monospace face and pad
//!   its own strings.
//! - **Advance-box measurement only.** [`TextMetrics::measure`] sizes the
//!   block the cursor sweeps, not its ink, so a face whose outlines overhang
//!   their advances overhangs the measured box too.
//! - **No wrapping, clipping, or scissor rect.** A line runs past the viewport
//!   edge and off-screen rather than wrapping or being cut at a panel boundary.
//! - **No scale-factor awareness.** `size_px` and `position` are physical
//!   pixels; nothing tracks the window's DPI, so a caller that wants a
//!   DPI-stable readout scales both itself.
//! - **One face at one bake size.** No bold/italic, no font fallback chain, and
//!   no mip chain, so sizes far from `bake_size_px` soften or blur.

pub mod glyph;

use std::collections::HashMap;

use ab_glyph::{Font, FontRef, Glyph, GlyphId, Point, ScaleFont};
use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use wgpu::*;

const ATLAS_SIZE: u32 = 1024;
const ATLAS_FORMAT: TextureFormat = TextureFormat::R8Unorm;

/// Per-glyph metadata in the atlas.
#[derive(Copy, Clone, Debug)]
struct GlyphEntry {
    /// Atlas UV rectangle (normalized 0..1).
    uv_min: [f32; 2],
    uv_max: [f32; 2],
    /// Pixel size at the atlas's bake size.
    px_width: f32,
    px_height: f32,
    /// Horizontal advance to the next glyph, at bake size.
    h_advance: f32,
    /// Pixel offset from baseline to the glyph's top-left, at bake size.
    bearing_x: f32,
    bearing_y: f32,
}

/// One vertex per quad corner; six vertices per glyph (two-triangle fan).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct TextVertex {
    pos: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct TextUniforms {
    viewport_size: [f32; 2],
    _pad: [f32; 2],
}

/// Advance-only text extents, free of GPU resources.
///
/// Layout is a per-char advance sum, so a block's size is exactly this
/// arithmetic. Split out of [`TextRenderer`] because a caller that must size a
/// block before a device exists, or in a test, has no renderer to ask.
pub struct TextMetrics {
    /// Horizontal advance per baked codepoint, at the bake size.
    advances: HashMap<char, f32>,
    bake_size_px: f32,
    line_height_px: f32,
}

impl TextMetrics {
    /// Read `font_bytes`'s advances at `bake_size_px`. Nothing is rasterized,
    /// so this is cheap enough to call outside setup.
    pub fn new(font_bytes: &[u8], bake_size_px: f32) -> Result<Self> {
        let font = FontRef::try_from_slice(font_bytes)
            .map_err(|e| anyhow!("loam-text: failed to parse font: {e}"))?;
        Ok(Self::from_font(&font, bake_size_px))
    }

    fn from_font(font: &FontRef<'_>, bake_size_px: f32) -> Self {
        let scaled = font.as_scaled(bake_size_px);
        let advances = (0x20u32..=0x7E)
            .map(|code| {
                let c = char::from_u32(code).expect("0x20..=0x7E is valid Unicode");
                (c, scaled.h_advance(font.glyph_id(c)))
            })
            .collect();
        Self {
            advances,
            bake_size_px,
            line_height_px: scaled.ascent() - scaled.descent() + scaled.line_gap(),
        }
    }

    /// Width and height of `text` at `size_px`, as `[w, h]`.
    ///
    /// The advance box, not the ink box: width is the widest line's cursor
    /// sweep and height is `lines * line_height`, measured from the `position`
    /// [`TextRenderer::queue`] would take. That height exceeds the last line's
    /// descender by one line gap, so the box contains the block rather than
    /// hugging it. Characters the atlas does not cover contribute nothing,
    /// which is what layout does with them.
    pub fn measure(&self, text: &str, size_px: f32) -> [f32; 2] {
        let mut widest = 0.0_f32;
        let mut line = 0.0_f32;
        let mut lines = 1_u32;
        for c in text.chars() {
            if c == '\n' {
                widest = widest.max(line);
                line = 0.0;
                lines += 1;
                continue;
            }
            line += self.advances.get(&c).copied().unwrap_or(0.0);
        }
        let scale = size_px / self.bake_size_px;
        [
            widest.max(line) * scale,
            lines as f32 * self.line_height_px * scale,
        ]
    }

    /// Bake size the advances were read at; per-frame sizes near this look best.
    pub fn bake_size_px(&self) -> f32 {
        self.bake_size_px
    }

    /// Vertical advance between lines at the bake size.
    pub fn line_height_px(&self) -> f32 {
        self.line_height_px
    }
}

/// Screen-space text renderer.
///
/// Construct once. Each frame:
/// 1. Call [`TextRenderer::queue`] for each string to draw.
/// 2. Call [`TextRenderer::record`] to flush the queue into the frame encoder.
///
/// The queue is reset every `record` call.
pub struct TextRenderer {
    pipeline: RenderPipeline,
    bind_group: BindGroup,
    uniform_buf: Buffer,
    // Held only to keep the GPU resources alive for the bind group's lifetime;
    // dropping them would free what `bind_group` still references.
    #[allow(dead_code)]
    atlas_tex: Texture,
    #[allow(dead_code)]
    atlas_view: TextureView,
    #[allow(dead_code)]
    atlas_sampler: Sampler,

    font_data: Vec<u8>,
    glyphs: HashMap<char, GlyphEntry>,
    metrics: TextMetrics,
    /// Distance from a line's top edge to its baseline, at the bake size. Kept
    /// alongside the line advance because layout needs both independently.
    ascent_px: f32,

    vertex_buf: Buffer,
    vertex_capacity: u64,
    queued: Vec<TextVertex>,
}

impl TextRenderer {
    /// Construct a renderer. `font_bytes` is raw TTF/OTF data; `bake_size_px` is
    /// the atlas rasterization size (smaller per-frame sizes are clean, larger
    /// blur). 48 is a reasonable default. `sample_count` must match the render
    /// target [`record`](TextRenderer::record) draws into, MSAA included.
    pub fn new(
        device: &Device,
        queue: &Queue,
        surface_format: TextureFormat,
        font_bytes: &[u8],
        bake_size_px: f32,
        sample_count: u32,
    ) -> Result<Self> {
        let font_data = font_bytes.to_vec();
        let font = FontRef::try_from_slice(&font_data)
            .map_err(|e| anyhow!("loam-text: failed to parse font: {e}"))?;

        let atlas_tex = device.create_texture(&TextureDescriptor {
            label: Some("loam-text atlas"),
            size: Extent3d {
                width: ATLAS_SIZE,
                height: ATLAS_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: ATLAS_FORMAT,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let atlas_view = atlas_tex.create_view(&TextureViewDescriptor::default());
        let atlas_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("loam-text atlas sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });

        let baked = bake_ascii_atlas(&font, bake_size_px)?;
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &atlas_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &baked.pixels,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ATLAS_SIZE),
                rows_per_image: Some(ATLAS_SIZE),
            },
            Extent3d {
                width: ATLAS_SIZE,
                height: ATLAS_SIZE,
                depth_or_array_layers: 1,
            },
        );

        let uniform_buf = device.create_buffer(&BufferDescriptor {
            label: Some("loam-text uniforms"),
            size: std::mem::size_of::<TextUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("loam-text bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("loam-text bg"),
            layout: &bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&atlas_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&atlas_sampler),
                },
            ],
        });

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("loam-text shader"),
            source: ShaderSource::Wgsl(WGSL_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("loam-text pipeline layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let vertex_attrs = wgpu::vertex_attr_array![
            0 => Float32x2,
            1 => Float32x2,
            2 => Float32x4,
        ];
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("loam-text pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[VertexBufferLayout {
                    array_stride: std::mem::size_of::<TextVertex>() as u64,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &vertex_attrs,
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: surface_format,
                    blend: Some(BlendState::ALPHA_BLENDING),
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

        let initial_capacity = 1024_u64;
        let zero_verts: Vec<TextVertex> =
            vec![bytemuck::Zeroable::zeroed(); initial_capacity as usize];
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("loam-text vertices"),
            contents: bytemuck::cast_slice(&zero_verts),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });

        Ok(Self {
            pipeline,
            bind_group,
            uniform_buf,
            atlas_tex,
            atlas_view,
            atlas_sampler,
            font_data,
            glyphs: baked.glyphs,
            metrics: baked.metrics,
            ascent_px: baked.ascent_px,
            vertex_buf,
            vertex_capacity: initial_capacity,
            queued: Vec::new(),
        })
    }

    /// Queue a string for this frame at `position` (viewport pixels).
    /// `position.y` is the first line's ascender, so a capital letter's top
    /// edge lands there. `size_px` near the bake size is cleanest; `color` is
    /// RGBA 0..1, straight alpha. `\n` advances a line; other control chars are
    /// skipped.
    pub fn queue(&mut self, text: &str, position: [f32; 2], size_px: f32, color: [f32; 4]) {
        layout_text(
            text,
            position,
            size_px,
            color,
            &self.glyphs,
            self.metrics.bake_size_px,
            self.metrics.line_height_px,
            self.ascent_px,
            &mut self.queued,
        );
    }

    /// Record queued text into `encoder` as one load/store pass on `view` at
    /// `viewport_size` (pixels), and reset the queue.
    ///
    /// The path for a host that owns the frame's encoder: a nested
    /// `queue.submit` would reach the GPU before the passes already recorded
    /// into that encoder, painting the text under the scene rather than over
    /// it. No resolve target is attached, so under MSAA the host's own resolve
    /// must come after this pass.
    pub fn record(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        view: &TextureView,
        viewport_size: [f32; 2],
    ) {
        if self.queued.is_empty() {
            return;
        }
        self.upload(device, queue, viewport_size);
        let mut rp = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("loam-text pass"),
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
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &self.bind_group, &[]);
        rp.set_vertex_buffer(0, self.vertex_buf.slice(..));
        rp.draw(0..self.queued.len() as u32, 0..1);
        drop(rp);

        self.queued.clear();
    }

    fn upload(&mut self, device: &Device, queue: &Queue, viewport_size: [f32; 2]) {
        let uniforms = TextUniforms {
            viewport_size,
            _pad: [0.0; 2],
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let needed = self.queued.len() as u64;
        if needed > self.vertex_capacity {
            let mut new_cap = self.vertex_capacity.max(1);
            while new_cap < needed {
                new_cap *= 2;
            }
            self.vertex_buf = device.create_buffer(&BufferDescriptor {
                label: Some("loam-text vertices (grown)"),
                size: new_cap * std::mem::size_of::<TextVertex>() as u64,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.vertex_capacity = new_cap;
        }
        queue.write_buffer(&self.vertex_buf, 0, bytemuck::cast_slice(&self.queued));
    }

    /// Advances and line height this renderer lays out with, for callers
    /// sizing a block before they queue it.
    pub fn metrics(&self) -> &TextMetrics {
        &self.metrics
    }

    /// Bake size glyphs were rasterized at; per-frame sizes near this look best.
    pub fn bake_size_px(&self) -> f32 {
        self.metrics.bake_size_px
    }

    /// Vertical advance between lines at the bake size.
    pub fn line_height_px(&self) -> f32 {
        self.metrics.line_height_px
    }

    /// Borrow the loaded font data. Useful for measurement helpers built on top.
    pub fn font_bytes(&self) -> &[u8] {
        &self.font_data
    }
}

/// The baseline sits `ascent_px * scale` below `position.y`, which puts the
/// ascender line exactly at `position.y`. Offsetting by `line_height_px`
/// instead would push the block down by the descent plus line gap, so
/// `position` would not be the top edge it is documented as.
#[allow(clippy::too_many_arguments)] // parameters are the layout state.
fn layout_text(
    text: &str,
    position: [f32; 2],
    size_px: f32,
    color: [f32; 4],
    glyphs: &HashMap<char, GlyphEntry>,
    bake_size_px: f32,
    line_height_px: f32,
    ascent_px: f32,
    out: &mut Vec<TextVertex>,
) {
    let scale = size_px / bake_size_px;
    let line_h = line_height_px * scale;
    let mut cursor_x = position[0];
    let mut cursor_y = position[1];

    for c in text.chars() {
        if c == '\n' {
            cursor_x = position[0];
            cursor_y += line_h;
            continue;
        }
        if !is_printable_ascii(c) {
            continue;
        }
        let Some(g) = glyphs.get(&c) else {
            continue;
        };

        let x0 = cursor_x + g.bearing_x * scale;
        let y0 = cursor_y + (ascent_px + g.bearing_y) * scale;
        let x1 = x0 + g.px_width * scale;
        let y1 = y0 + g.px_height * scale;

        let (u0, v0) = (g.uv_min[0], g.uv_min[1]);
        let (u1, v1) = (g.uv_max[0], g.uv_max[1]);

        out.extend_from_slice(&[
            TextVertex {
                pos: [x0, y0],
                uv: [u0, v0],
                color,
            },
            TextVertex {
                pos: [x1, y0],
                uv: [u1, v0],
                color,
            },
            TextVertex {
                pos: [x0, y1],
                uv: [u0, v1],
                color,
            },
            TextVertex {
                pos: [x1, y0],
                uv: [u1, v0],
                color,
            },
            TextVertex {
                pos: [x1, y1],
                uv: [u1, v1],
                color,
            },
            TextVertex {
                pos: [x0, y1],
                uv: [u0, v1],
                color,
            },
        ]);

        cursor_x += g.h_advance * scale;
    }
}

/// Atlas coverage: the codepoints [`bake_ascii_atlas`] rasterizes.
fn is_printable_ascii(c: char) -> bool {
    ('\u{20}'..='\u{7E}').contains(&c)
}

/// True when [`TextRenderer::queue`] can draw every character of `text`.
///
/// The atlas covers printable ASCII only, and layout drops everything else
/// without erroring, so a caller whose string must survive intact (a readout,
/// a label built from user data) checks here first. `\n` counts as renderable:
/// layout consumes it as a line break.
pub fn is_renderable(text: &str) -> bool {
    text.chars().all(|c| c == '\n' || is_printable_ascii(c))
}

struct BakedAtlas {
    pixels: Vec<u8>,
    glyphs: HashMap<char, GlyphEntry>,
    metrics: TextMetrics,
    ascent_px: f32,
}

fn bake_ascii_atlas(font: &FontRef<'_>, bake_size_px: f32) -> Result<BakedAtlas> {
    let scaled = font.as_scaled(bake_size_px);

    let mut atlas = vec![0u8; (ATLAS_SIZE * ATLAS_SIZE) as usize];
    let mut entries: HashMap<char, GlyphEntry> = HashMap::with_capacity(96);

    // Shelf packer: glyphs flow left-to-right, new shelf when the row is full.
    let pad = 1u32;
    let mut shelf_y: u32 = pad;
    let mut shelf_x: u32 = pad;
    let mut shelf_h: u32 = 0;

    for code in 0x20u32..=0x7E {
        let c = char::from_u32(code).unwrap();
        let gid: GlyphId = font.glyph_id(c);
        let h_adv = scaled.h_advance(gid);

        // Glyph at origin so px_bounds is offset-from-origin (we need size +
        // offset, not absolute placement).
        let mut glyph: Glyph = scaled.scaled_glyph(c);
        glyph.position = Point { x: 0.0, y: 0.0 };

        let outlined = scaled.outline_glyph(glyph);
        match outlined {
            Some(o) => {
                let bounds = o.px_bounds();
                let gw = bounds.width().ceil() as u32;
                let gh = bounds.height().ceil() as u32;
                if gw == 0 || gh == 0 {
                    // No rasterizable area; record an empty entry so the cursor
                    // still advances.
                    entries.insert(
                        c,
                        GlyphEntry {
                            uv_min: [0.0; 2],
                            uv_max: [0.0; 2],
                            px_width: 0.0,
                            px_height: 0.0,
                            h_advance: h_adv,
                            bearing_x: 0.0,
                            bearing_y: 0.0,
                        },
                    );
                    continue;
                }
                if shelf_x + gw + pad > ATLAS_SIZE {
                    shelf_y += shelf_h + pad;
                    shelf_x = pad;
                    shelf_h = 0;
                }
                if shelf_y + gh + pad > ATLAS_SIZE {
                    return Err(anyhow!(
                        "loam-text: ASCII atlas exceeded {ATLAS_SIZE}x{ATLAS_SIZE} at glyph {c:?}; \
                         reduce bake_size_px or extend the packer"
                    ));
                }

                let dst_x = shelf_x;
                let dst_y = shelf_y;
                o.draw(|gx, gy, cov| {
                    let px_x = dst_x + gx;
                    let px_y = dst_y + gy;
                    if px_x < ATLAS_SIZE && px_y < ATLAS_SIZE {
                        let idx = (px_y * ATLAS_SIZE + px_x) as usize;
                        let v = (cov * 255.0).round().clamp(0.0, 255.0) as u8;
                        atlas[idx] = atlas[idx].max(v);
                    }
                });

                let uv_min = [
                    dst_x as f32 / ATLAS_SIZE as f32,
                    dst_y as f32 / ATLAS_SIZE as f32,
                ];
                let uv_max = [
                    (dst_x + gw) as f32 / ATLAS_SIZE as f32,
                    (dst_y + gh) as f32 / ATLAS_SIZE as f32,
                ];

                entries.insert(
                    c,
                    GlyphEntry {
                        uv_min,
                        uv_max,
                        px_width: gw as f32,
                        px_height: gh as f32,
                        h_advance: h_adv,
                        bearing_x: bounds.min.x,
                        bearing_y: bounds.min.y,
                    },
                );

                shelf_x += gw + pad;
                shelf_h = shelf_h.max(gh);
            }
            None => {
                // Whitespace / no outline.
                entries.insert(
                    c,
                    GlyphEntry {
                        uv_min: [0.0; 2],
                        uv_max: [0.0; 2],
                        px_width: 0.0,
                        px_height: 0.0,
                        h_advance: h_adv,
                        bearing_x: 0.0,
                        bearing_y: 0.0,
                    },
                );
            }
        }
    }

    Ok(BakedAtlas {
        pixels: atlas,
        glyphs: entries,
        metrics: TextMetrics::from_font(font, bake_size_px),
        ascent_px: scaled.ascent(),
    })
}

const WGSL_SHADER: &str = r#"
struct Uniforms {
    viewport_size: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var atlas_tex: texture_2d<f32>;
@group(0) @binding(2) var atlas_sam: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) in_pos: vec2<f32>,
    @location(1) in_uv: vec2<f32>,
    @location(2) in_color: vec4<f32>,
) -> VsOut {
    let ndc_x = (in_pos.x / u.viewport_size.x) * 2.0 - 1.0;
    // pixel y axis points down; NDC y axis points up; flip.
    let ndc_y = 1.0 - (in_pos.y / u.viewport_size.y) * 2.0;
    var out: VsOut;
    out.clip = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = in_uv;
    out.color = in_color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let alpha = textureSample(atlas_tex, atlas_sam, in.uv).r;
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baking_roundtrip_with_system_arial() {
        // Arial is reliably present on Windows; skip cleanly if not.
        let path = std::path::Path::new(r"C:\Windows\Fonts\arial.ttf");
        if !path.exists() {
            eprintln!("skip: arial.ttf not present at {path:?}");
            return;
        }
        let bytes = std::fs::read(path).expect("read arial.ttf");
        let font = FontRef::try_from_slice(&bytes).expect("parse arial.ttf");
        let baked = bake_ascii_atlas(&font, 48.0).expect("bake");
        assert_eq!(baked.pixels.len() as u32, ATLAS_SIZE * ATLAS_SIZE);
        assert_eq!(baked.glyphs.len(), 0x7F - 0x20);
        let line_h = baked.metrics.line_height_px();
        assert!(line_h > 30.0 && line_h < 80.0, "line_h = {line_h}");
        // Ascent is the part of the line box above the baseline, so it is
        // strictly inside it; a swap of the two would be caught here.
        let ascent = baked.ascent_px;
        assert!(
            ascent > 0.0 && ascent < line_h,
            "ascent = {ascent}, line_h = {line_h}"
        );
        let a = baked.glyphs.get(&'A').expect("A in atlas");
        assert!(a.px_width > 0.0 && a.px_height > 0.0);
        // Both construction paths read the same font at the same size, so the
        // cheap non-rasterizing one must not drift from the baked one.
        let standalone = TextMetrics::new(&bytes, 48.0).expect("metrics");
        assert_eq!(
            standalone.measure("Wg|", 48.0),
            baked.metrics.measure("Wg|", 48.0)
        );
    }

    /// Ascent used by the layout tests. Deliberately unequal to their 16.0 line
    /// height so a test cannot pass by confusing the baseline offset with the
    /// line advance.
    const MOCK_ASCENT: f32 = 12.0;

    /// The pinned math (advance, newline reset, vertex count) depends only on
    /// `h_advance`, not atlas geometry.
    fn mock_glyph_table(h_advance: f32) -> HashMap<char, GlyphEntry> {
        (0x20u8..=0x7Eu8)
            .map(|c| {
                (
                    c as char,
                    GlyphEntry {
                        uv_min: [0.0, 0.0],
                        uv_max: [1.0, 1.0],
                        bearing_x: 0.0,
                        bearing_y: 0.0,
                        px_width: 1.0,
                        px_height: 1.0,
                        h_advance,
                    },
                )
            })
            .collect()
    }

    /// Built by hand so `measure` is exercised against the very table
    /// `layout_text` walks rather than against a second font.
    fn mock_metrics(h_advance: f32) -> TextMetrics {
        TextMetrics {
            advances: (0x20u8..=0x7Eu8).map(|c| (c as char, h_advance)).collect(),
            bake_size_px: 16.0,
            line_height_px: 16.0,
        }
    }

    #[test]
    fn layout_newline_resets_x_and_advances_y() {
        let glyphs = mock_glyph_table(10.0);
        let mut out = Vec::new();
        // size_px = bake_size_px = 16.0, so scale = 1.0 and line_h = 16.0.
        layout_text(
            "a\nb",
            [5.0, 0.0],
            16.0,
            [1.0; 4],
            &glyphs,
            16.0,
            16.0,
            MOCK_ASCENT,
            &mut out,
        );
        assert_eq!(out.len(), 12); // 6 verts × 2 glyphs

        // Glyph 'a' top-left: x0 = 5, y0 = ascent (mock bearings are zero).
        let first = out[0];
        assert_eq!(first.pos[0], 5.0);
        assert!((first.pos[1] - MOCK_ASCENT).abs() < 1e-5);

        // Glyph 'b' (vertex 6), after the newline: x reset to 5, y advanced by
        // the line height, not by the ascent.
        let second = out[6];
        assert_eq!(
            second.pos[0], 5.0,
            "newline must reset cursor_x to position[0]"
        );
        assert!(
            (second.pos[1] - (MOCK_ASCENT + 16.0)).abs() < 1e-5,
            "newline must advance cursor_y by line_h, got {}",
            second.pos[1],
        );
    }

    #[test]
    fn position_y_is_the_ascender_line_at_every_scale() {
        let mut glyphs = mock_glyph_table(10.0);
        // Bake-space glyph reaching from the ascender down to the baseline.
        glyphs.insert(
            'A',
            GlyphEntry {
                uv_min: [0.0, 0.0],
                uv_max: [1.0, 1.0],
                bearing_x: 0.0,
                bearing_y: -MOCK_ASCENT,
                px_width: 8.0,
                px_height: MOCK_ASCENT,
                h_advance: 10.0,
            },
        );
        for size_px in [8.0_f32, 16.0, 40.0] {
            let mut out = Vec::new();
            layout_text(
                "A",
                [3.0, 7.0],
                size_px,
                [1.0; 4],
                &glyphs,
                16.0,
                16.0,
                MOCK_ASCENT,
                &mut out,
            );
            assert!(
                (out[0].pos[1] - 7.0).abs() < 1e-5,
                "at {size_px}px the ascender landed at {} instead of 7.0",
                out[0].pos[1]
            );
        }
    }

    #[test]
    fn layout_cursor_advances_by_h_advance_scaled() {
        let glyphs = mock_glyph_table(10.0);
        let mut out = Vec::new();
        // 32px render at 16px bake => scale 2 => effective advance 20 per glyph.
        layout_text(
            "ab",
            [0.0, 0.0],
            32.0,
            [1.0; 4],
            &glyphs,
            16.0,
            16.0,
            MOCK_ASCENT,
            &mut out,
        );

        assert_eq!(out[0].pos[0], 0.0);
        assert_eq!(out[6].pos[0], 20.0);
    }

    #[test]
    fn layout_skips_unprintable_and_out_of_range_chars() {
        let glyphs = mock_glyph_table(10.0);
        let mut out = Vec::new();
        layout_text(
            "a\tb\u{80}c😀d",
            [0.0, 0.0],
            16.0,
            [1.0; 4],
            &glyphs,
            16.0,
            16.0,
            MOCK_ASCENT,
            &mut out,
        );
        // Only 'a', 'b', 'c', 'd' get glyphs. 4 × 6 = 24 vertices.
        assert_eq!(out.len(), 24);
    }

    #[test]
    fn layout_skips_missing_glyphs() {
        let mut glyphs = mock_glyph_table(10.0);
        glyphs.remove(&'b');
        let mut out = Vec::new();
        layout_text(
            "ab",
            [0.0, 0.0],
            16.0,
            [1.0; 4],
            &glyphs,
            16.0,
            16.0,
            MOCK_ASCENT,
            &mut out,
        );
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn is_renderable_agrees_with_what_layout_emits() {
        let glyphs = mock_glyph_table(10.0);
        let mut out = Vec::new();
        for code in (0u32..=0x2FFF).chain([0xFFFD, 0x1F600]) {
            let Some(c) = char::from_u32(code) else {
                continue;
            };
            let text = c.to_string();
            out.clear();
            layout_text(
                &text,
                [0.0, 0.0],
                16.0,
                [1.0; 4],
                &glyphs,
                16.0,
                16.0,
                MOCK_ASCENT,
                &mut out,
            );
            let emitted = !out.is_empty();
            let expected = is_renderable(&text) && c != '\n';
            assert_eq!(
                emitted,
                expected,
                "U+{code:04X} ({c:?}): is_renderable said {}, layout emitted {emitted}",
                is_renderable(&text)
            );
        }
    }

    #[test]
    fn measured_box_contains_every_vertex_layout_emits() {
        const ADVANCE: f32 = 10.0;
        let metrics = mock_metrics(ADVANCE);
        let glyphs: HashMap<char, GlyphEntry> = (0x21u8..=0x7Eu8)
            .map(|c| {
                (
                    c as char,
                    GlyphEntry {
                        uv_min: [0.0, 0.0],
                        uv_max: [1.0, 1.0],
                        bearing_x: 0.0,
                        bearing_y: -MOCK_ASCENT,
                        px_width: ADVANCE,
                        px_height: MOCK_ASCENT,
                        h_advance: ADVANCE,
                    },
                )
            })
            .collect();
        let position = [7.0_f32, 11.0];
        for text in ["A", "AB", "AB\nCDE", "AB\nCDE\nF"] {
            for size_px in [8.0_f32, 16.0, 37.0] {
                let mut out = Vec::new();
                layout_text(
                    text,
                    position,
                    size_px,
                    [1.0; 4],
                    &glyphs,
                    metrics.bake_size_px,
                    metrics.line_height_px,
                    MOCK_ASCENT,
                    &mut out,
                );
                let [w, h] = metrics.measure(text, size_px);
                for v in &out {
                    assert!(
                        v.pos[0] >= position[0] - 1e-4 && v.pos[0] <= position[0] + w + 1e-4,
                        "{text:?} at {size_px}px: x {} outside [{}, {}]",
                        v.pos[0],
                        position[0],
                        position[0] + w
                    );
                    assert!(
                        v.pos[1] >= position[1] - 1e-4 && v.pos[1] <= position[1] + h + 1e-4,
                        "{text:?} at {size_px}px: y {} outside [{}, {}]",
                        v.pos[1],
                        position[1],
                        position[1] + h
                    );
                }
            }
        }
    }

    #[test]
    fn measured_box_is_linear_in_size() {
        let metrics = mock_metrics(10.0);
        let text = "abc\nde";
        let [w1, h1] = metrics.measure(text, 16.0);
        for factor in [0.5_f32, 1.25, 2.0, 3.0] {
            let [w, h] = metrics.measure(text, 16.0 * factor);
            assert!((w - w1 * factor).abs() < 1e-3, "width at {factor}x: {w}");
            assert!((h - h1 * factor).abs() < 1e-3, "height at {factor}x: {h}");
        }
    }

    #[test]
    fn measured_height_is_one_line_box_per_line() {
        let metrics = mock_metrics(10.0);
        let line_h = metrics.line_height_px();
        for (text, lines) in [("a", 1.0), ("a\nb", 2.0), ("a\nb\nc", 3.0), ("\n\n", 3.0)] {
            let [_, h] = metrics.measure(text, metrics.bake_size_px());
            assert!(
                (h - lines * line_h).abs() < 1e-4,
                "{text:?} measured {h}, expected {lines} x {line_h}"
            );
        }
    }

    #[test]
    fn measured_width_is_the_widest_line() {
        let metrics = mock_metrics(10.0);
        let [w, _] = metrics.measure("ab\nabcd\na", metrics.bake_size_px());
        assert!((w - 40.0).abs() < 1e-4, "measured {w}, expected 4 x 10");
    }

    #[test]
    fn wgsl_shader_validates_via_naga() {
        let module = naga::front::wgsl::parse_str(WGSL_SHADER).expect("WGSL parse");
        let flags = naga::valid::ValidationFlags::all();
        let caps = naga::valid::Capabilities::empty();
        naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .expect("WGSL validate");
    }
}
