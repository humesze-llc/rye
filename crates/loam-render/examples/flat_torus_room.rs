//! Offscreen filmstrip of a walk through the flat 3-torus room.
//!
//! Renders eight frames as the camera translates one full lattice vector along
//! +x, tiles them into a single PNG under `captures/`, and reports how far the
//! last frame drifted from the first. The walk closes: crossing the +x face
//! puts the camera back where it started, so tile 1 and tile 8 are the same
//! image, and any difference between them is a defect in the wrap rather than
//! in the eye of the reader.
//!
//! Reading the picture: surfaces are tinted by how many cells of the cover the
//! ray crossed to reach them, so the flat-shaded regions with straight edges
//! are cells, not tearing. With the camera looking down +x, the faces normal to
//! y and z project to horizontal and vertical screen lines and the faces normal
//! to x change the whole field at a depth. The geometry itself is continuous
//! across all of them: no silhouette breaks at a boundary.
//!
//! Run: `cargo run -p loam-render --example flat_torus_room`
//!
//! Headless on purpose. `RenderDevice` owns a surface and a window; this needs
//! neither, and a capture that requires a human at a keyboard is a capture that
//! does not get taken.

use glam::Vec3;
use loam_math::{FlatTorus3, WgslSpace};
use loam_render::RayMarchUniforms;
use loam_scene::{Scene, SceneNode};
use loam_shader::GEODESIC_MARCH_KERNEL;
use wgpu::*;

const CELL: Vec3 = Vec3::new(2.0, 2.6, 1.7);
const EYE: Vec3 = Vec3::new(-0.9, 0.15, 0.55);

/// 512 * 4 bytes per row is a multiple of `COPY_BYTES_PER_ROW_ALIGNMENT`, so
/// the readback needs no per-row repacking.
const TILE_WIDTH: u32 = 512;
const TILE_HEIGHT: u32 = 288;
const COLUMNS: u32 = 4;
const ROWS: u32 = 2;
const FRAMES: u32 = COLUMNS * ROWS;

/// Spheres only: `Shape::Sphere` is the one primitive whose SDF routes through
/// `loam_distance`, so it is the one that inherits the quotient's periodicity.
/// A box or a half-space emits raw chart arithmetic and tears at the gluing.
fn room() -> Scene {
    Scene::new(
        SceneNode::sphere(Vec3::ZERO, 0.36)
            .union(SceneNode::sphere(Vec3::new(0.72, -0.85, 0.45), 0.2))
            .union(SceneNode::sphere(Vec3::new(-0.6, 0.8, -0.55), 0.26)),
    )
}

const ROOM_SHADER: &str = r#"
struct RayMarchUniforms {
    camera_pos: vec3<f32>,
    camera_forward: vec3<f32>,
    camera_right: vec3<f32>,
    camera_up: vec3<f32>,
    fov_y_tan: f32,
    resolution: vec2<f32>,
    time: f32,
    tick: f32,
    params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: RayMarchUniforms;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vid) / 2) * 4.0 - 1.0;
    let y = f32(i32(vid) & 1) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

const VOID_COLOR: vec3<f32> = vec3<f32>(0.015, 0.017, 0.025);

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let ndc = pos.xy / u.resolution * 2.0 - vec2<f32>(1.0, 1.0);
    let aspect = u.resolution.x / u.resolution.y;
    let dir = loam_safe_normalize(
        u.camera_forward
            + u.camera_right * ndc.x * u.fov_y_tan * aspect
            - u.camera_up * ndc.y * u.fov_y_tan,
        vec3<f32>(0.0, 0.0, -1.0),
    );

    let hit = loam_march_geodesic(u.camera_pos, dir, 1.0);
    if hit.w < 0.0 {
        return vec4<f32>(VOID_COLOR, 1.0);
    }

    let normal = loam_estimate_normal(hit.xyz, 1.0);
    let key = normalize(vec3<f32>(0.42, 0.83, 0.37));
    let lambert = max(dot(normal, key), 0.0) * 0.82 + 0.18;

    // How many cells away from the observer's own cell the surface sits: the
    // deck element the ray travelled through. The marcher only ever holds
    // wrapped positions, so it comes from the covering ray. Geodesics of E3/L
    // lift to straight lines, and dividing the kernel's parameter by the speed
    // it built from its own metric probe turns that parameter into arclength.
    //
    // Taken relative to the camera's cell, not absolute: an absolute index is
    // a statement about the chosen lift, so it would shift when the observer
    // walks a lattice vector even though nothing in the quotient moved.
    let probe_eps = 1e-4;
    let speed = loam_distance(u.camera_pos, loam_exp(u.camera_pos, dir * probe_eps)) / probe_eps;
    let copy = loam_torus_index(u.camera_pos + dir * (hit.w / speed))
             - loam_torus_index(u.camera_pos);

    // Cosine palette (Quilez 2015, "palettes") keyed on the copy offset: a
    // distinct, stable colour per cell crossed, so the tiling reads as a
    // tiling instead of as one ambiguous lattice of spheres.
    let phase = 0.11 * copy.x + 0.27 * copy.y + 0.43 * copy.z;
    let tint = 0.55 + 0.4 * cos(6.2831855 * phase + vec3<f32>(0.0, 2.1, 4.2));

    let fog = exp(-0.10 * hit.w);
    return vec4<f32>(mix(VOID_COLOR, tint * lambert, fog), 1.0);
}
"#;

fn main() {
    let space = FlatTorus3::new(CELL);
    let source = format!(
        "{}\n{}\n{}\n{}",
        space.wgsl_impl(),
        room().to_wgsl(&space),
        GEODESIC_MARCH_KERNEL,
        ROOM_SHADER
    );

    let (device, queue) = pollster::block_on(request_device());
    let renderer = Renderer::new(&device, &source);

    let mut tiles = Vec::with_capacity(FRAMES as usize);
    for frame in 0..FRAMES {
        // The last frame lands exactly one lattice vector along +x, so it must
        // reproduce the first.
        let travel = CELL.x * frame as f32 / (FRAMES - 1) as f32;
        tiles.push(renderer.render(&device, &queue, EYE + Vec3::X * travel));
    }

    let closure = channel_difference(&tiles[0], &tiles[FRAMES as usize - 1]);
    println!(
        "one lattice vector along +x: mean channel delta {:.4}/255, worst {}/255, \
         {:.4}% of channels off by more than a quantisation step",
        closure.mean,
        closure.worst,
        100.0 * closure.disagreeing
    );

    let path = write_filmstrip(&tiles);
    println!("capture: {}", path.display());

    // The two frames are separate marches from different covering coordinates
    // and the sphere-tracing step sequence is not identical between them, so a
    // handful of silhouette pixels land on the other side of a hard edge.
    // Bounds are on the population, not the extremum: a wrap that failed to
    // close moves whole spheres, which lights up percent-scale disagreement,
    // not the 0.0007% measured here.
    assert!(
        closure.mean < 0.05 && closure.disagreeing < 1e-3,
        "the walk did not close: mean {:.4}/255 over {:.4}% of channels",
        closure.mean,
        100.0 * closure.disagreeing
    );
}

async fn request_device() -> (Device, Queue) {
    let instance = Instance::default();
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("wgpu adapter");
    adapter
        .request_device(&DeviceDescriptor {
            label: Some("flat torus room"),
            required_features: Features::empty(),
            required_limits: Limits::default(),
            memory_hints: MemoryHints::default(),
            trace: Trace::Off,
            experimental_features: Default::default(),
        })
        .await
        .expect("wgpu device")
}

struct Renderer {
    pipeline: RenderPipeline,
    bind_group: BindGroup,
    uniform_buffer: Buffer,
    color: Texture,
    view: TextureView,
    staging: Buffer,
}

const TILE_BYTES: u64 = (TILE_WIDTH * TILE_HEIGHT * 4) as u64;

impl Renderer {
    fn new(device: &Device, source: &str) -> Self {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("flat torus room"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("flat torus room uniforms"),
            size: std::mem::size_of::<RayMarchUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("flat torus room"),
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
            label: Some("flat torus room"),
            layout: &layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("flat torus room"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("flat torus room"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &module,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    // sRGB target: the shader writes linear and the hardware
                    // encodes, so the readback bytes are already what the PNG
                    // should carry.
                    format: TextureFormat::Rgba8UnormSrgb,
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

        let color = device.create_texture(&TextureDescriptor {
            label: Some("flat torus room color"),
            size: Extent3d {
                width: TILE_WIDTH,
                height: TILE_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = color.create_view(&TextureViewDescriptor::default());
        let staging = device.create_buffer(&BufferDescriptor {
            label: Some("flat torus room readback"),
            size: TILE_BYTES,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group,
            uniform_buffer,
            color,
            view,
            staging,
        }
    }

    fn render(&self, device: &Device, queue: &Queue, eye: Vec3) -> Vec<u8> {
        let uniforms = RayMarchUniforms {
            camera_pos: eye.into(),
            camera_forward: [1.0, 0.0, 0.0],
            camera_right: [0.0, 0.0, 1.0],
            camera_up: [0.0, 1.0, 0.0],
            resolution: [TILE_WIDTH as f32, TILE_HEIGHT as f32],
            ..Default::default()
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("flat torus room"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &self.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &self.color,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &self.staging,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(TILE_WIDTH * 4),
                    rows_per_image: Some(TILE_HEIGHT),
                },
            },
            Extent3d {
                width: TILE_WIDTH,
                height: TILE_HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        self.staging.slice(..).map_async(MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(PollType::wait_indefinitely()).expect("poll");
        rx.recv().expect("map").expect("map succeeded");
        let pixels = self.staging.slice(..).get_mapped_range().to_vec();
        self.staging.unmap();
        pixels
    }
}

struct FrameDifference {
    worst: u8,
    mean: f64,
    /// Fraction of channels off by more than a quantisation step.
    disagreeing: f64,
}

fn channel_difference(a: &[u8], b: &[u8]) -> FrameDifference {
    let mut worst = 0;
    let mut total = 0u64;
    let mut disagreeing = 0u64;
    for (x, y) in a.iter().zip(b) {
        let delta = x.abs_diff(*y);
        worst = worst.max(delta);
        total += delta as u64;
        if delta > 2 {
            disagreeing += 1;
        }
    }
    FrameDifference {
        worst,
        mean: total as f64 / a.len() as f64,
        disagreeing: disagreeing as f64 / a.len() as f64,
    }
}

fn write_filmstrip(tiles: &[Vec<u8>]) -> std::path::PathBuf {
    let strip_width = TILE_WIDTH * COLUMNS;
    let strip_height = TILE_HEIGHT * ROWS;
    let mut strip = vec![0u8; (strip_width * strip_height * 4) as usize];
    for (index, tile) in tiles.iter().enumerate() {
        let column = index as u32 % COLUMNS;
        let row = index as u32 / COLUMNS;
        for y in 0..TILE_HEIGHT {
            let source = (y * TILE_WIDTH * 4) as usize;
            let destination =
                (((row * TILE_HEIGHT + y) * strip_width + column * TILE_WIDTH) * 4) as usize;
            let span = (TILE_WIDTH * 4) as usize;
            strip[destination..destination + span].copy_from_slice(&tile[source..source + span]);
        }
    }

    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("captures");
    std::fs::create_dir_all(&directory).expect("create captures directory");
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock after the epoch")
        .as_secs();
    let path = directory.join(format!("flat_torus_room_{stamp}.png"));

    let file = std::fs::File::create(&path).expect("create capture file");
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), strip_width, strip_height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    encoder
        .write_header()
        .expect("png header")
        .write_image_data(&strip)
        .expect("png data");
    path
}
