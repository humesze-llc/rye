//! Offscreen filmstrip of a walk down the shortest closed geodesic of the lens
//! space L(p, q).
//!
//! Renders eight frames as the observer travels one deck displacement along the
//! `z1` circle, tiles them into a single PNG under `captures/`, and measures the
//! last frame against a reference render of the *same* point of the quotient
//! reached the other way, by applying one deck element to the observer at the
//! start. The two agree only if the gluing is right, so the number printed is
//! the closure of the loop rather than a mood.
//!
//! Reading the picture: the corridor ahead is the observer's own geodesic, which
//! closes after `2π/p`, so every sphere appears p times down it, each copy
//! dimmer than the last. Between one copy and the next the transverse offset
//! turns by `2πq/p`: that spiral *is* the twist, and it is why the eighth tile
//! is the first tile rolled about the direction of travel rather than the first
//! tile again. Colour is per sphere, not per copy: the copies are one object
//! seen p times and no label distinguishing them is a function of the point of
//! the quotient, so depth is what separates them.
//!
//! Run: `cargo run -p loam-render --example lens_space`
//!
//! Headless on purpose. `RenderDevice` owns a surface and a window; this needs
//! neither, and a capture that requires a human at a keyboard is a capture that
//! does not get taken.

use std::f32::consts::TAU;

use glam::Vec4;
use loam_math::{IsometryGroup, LensSpace};
use wgpu::*;

const P: u32 = 5;
const Q: u32 = 2;

/// Spheres as `(along, off, phase, radius)` in radians: `along` slides down the
/// observer's geodesic, `off` lifts the centre off it, and `phase` turns that
/// offset within the `z2` plane. Off-axis by more than their radius, so the
/// observer starts outside every one of them.
const SPHERES: [(f32, f32, f32, f32); 5] = [
    (0.30, 0.30, 0.0, 0.13),
    (0.62, 0.26, 2.1, 0.11),
    (0.95, 0.34, 4.2, 0.15),
    (0.18, 0.55, 1.0, 0.12),
    (0.80, 0.50, 3.4, 0.10),
];

/// 512 * 4 bytes per row is a multiple of `COPY_BYTES_PER_ROW_ALIGNMENT`, so
/// the readback needs no per-row repacking.
const TILE_WIDTH: u32 = 512;
const TILE_HEIGHT: u32 = 288;
const COLUMNS: u32 = 4;
const ROWS: u32 = 2;
const FRAMES: u32 = COLUMNS * ROWS;

/// An observer: a lift and an orthonormal frame of tangents at it.
#[derive(Clone, Copy)]
struct Observer {
    eye: Vec4,
    forward: Vec4,
    right: Vec4,
    up: Vec4,
}

/// The observer `arc` radians down the `z1` great circle, frame parallel-
/// transported along it.
///
/// Transport along this geodesic turns the plane of motion by `arc` and fixes
/// the `z2` plane outright (the transport formula's correction term vanishes on
/// vectors orthogonal to both endpoints), so the frame is the rotated pair in
/// the `z1` plane and the fixed pair in the other.
fn walker(arc: f32) -> Observer {
    let (sin, cos) = arc.sin_cos();
    Observer {
        eye: Vec4::new(cos, sin, 0.0, 0.0),
        forward: Vec4::new(-sin, cos, 0.0, 0.0),
        right: Vec4::Z,
        up: Vec4::W,
    }
}

/// A sphere centre at `off` radians from the point `along` radians down the
/// observer's geodesic, offset in the `z2` plane at `phase`. Unit by
/// construction, `cos²(off) + sin²(off)`.
fn sphere_centre(along: f32, off: f32, phase: f32) -> Vec4 {
    let (sin_along, cos_along) = along.sin_cos();
    let (sin_off, cos_off) = off.sin_cos();
    let (sin_phase, cos_phase) = phase.sin_cos();
    Vec4::new(
        cos_off * cos_along,
        cos_off * sin_along,
        sin_off * cos_phase,
        sin_off * sin_phase,
    )
}

fn scene_wgsl() -> String {
    let centres = SPHERES
        .iter()
        .map(|&(along, off, phase, _)| {
            let c = sphere_centre(along, off, phase);
            format!("vec4<f32>({:?}, {:?}, {:?}, {:?})", c.x, c.y, c.z, c.w)
        })
        .collect::<Vec<_>>()
        .join(",\n    ");
    let radii = SPHERES
        .iter()
        .map(|&(_, _, _, radius)| format!("{radius:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    let count = SPHERES.len();
    format!(
        r#"
const LENS_SPHERE_COUNT: i32 = {count};
const LENS_CENTRES = array<vec4<f32>, {count}>(
    {centres}
);
const LENS_RADII = array<f32, {count}>({radii});

// Exact: the quotient distance to a sphere is the quotient distance to its
// centre less the radius, and the minimum of exact distances is the exact
// distance to the union. That is what lets the marcher below take a full
// sphere-tracing step instead of undershooting an approximate bound.
fn lens_scene_sdf(x: vec4<f32>) -> f32 {{
    var nearest = 8.0;
    for (var i = 0; i < LENS_SPHERE_COUNT; i = i + 1) {{
        nearest = min(nearest, loam_lens_distance(x, LENS_CENTRES[i]) - LENS_RADII[i]);
    }}
    return nearest;
}}

fn lens_scene_nearest(x: vec4<f32>) -> i32 {{
    var nearest = 8.0;
    var index = 0;
    for (var i = 0; i < LENS_SPHERE_COUNT; i = i + 1) {{
        let d = loam_lens_distance(x, LENS_CENTRES[i]) - LENS_RADII[i];
        if d < nearest {{
            nearest = d;
            index = i;
        }}
    }}
    return index;
}}
"#
    )
}

const LENS_SHADER: &str = r#"
struct LensUniforms {
    eye: vec4<f32>,
    forward: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    resolution: vec2<f32>,
    fov_y_tan: f32,
    _pad: f32,
};
@group(0) @binding(0) var<uniform> u: LensUniforms;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vid) / 2) * 4.0 - 1.0;
    let y = f32(i32(vid) & 1) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

const VOID_COLOR: vec3<f32> = vec3<f32>(0.015, 0.017, 0.025);
// A geodesic of S³ closes at 2π, so a ray that has travelled that far has seen
// the whole of its own path: a miss here is a proof, not a step budget.
const LENS_MAX_ARC: f32 = 6.2831855;
const LENS_HIT_EPSILON: f32 = 1.0e-4;

struct LensHit {
    point: vec4<f32>,
    tangent: vec4<f32>,
    arc: f32,
    hit: bool,
};

// Geodesic flow on S³ is a rotation in the plane spanned by the position and
// the unit tangent, so the marcher steps exactly: no metric probe, no
// integration error, and the transported tangent falls out of the same pair of
// sines. The quotient never enters here; it is inside `loam_lens_distance`,
// which is what makes the same march draw the covering sphere or the lens
// space depending only on which SDF it is handed.
fn lens_march(origin: vec4<f32>, direction: vec4<f32>) -> LensHit {
    var arc = 0.0;
    for (var i = 0; i < 256; i = i + 1) {
        let sin_arc = sin(arc);
        let cos_arc = cos(arc);
        let point = cos_arc * origin + sin_arc * direction;
        let d = lens_scene_sdf(point);
        if d < LENS_HIT_EPSILON {
            return LensHit(point, -sin_arc * origin + cos_arc * direction, arc, true);
        }
        arc = arc + max(d, LENS_HIT_EPSILON);
        if arc > LENS_MAX_ARC {
            break;
        }
    }
    return LensHit(origin, direction, arc, false);
}

// Outward unit normal of the nearest sphere, in closed form: the gradient of a
// distance function is the unit tangent pointing away from its source, and the
// source is whichever lift of the centre the quotient distance selected. A
// central difference would want a second epsilon and eight more quotient
// distances, each of them a loop over the deck group.
fn lens_normal(x: vec4<f32>, sphere: i32) -> vec4<f32> {
    let centre = LENS_CENTRES[sphere];
    let lift = loam_lens_apply(centre, loam_lens_nearest_power(x, centre));
    return normalize(dot(x, lift) * x - lift);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let ndc = pos.xy / u.resolution * 2.0 - vec2<f32>(1.0, 1.0);
    let aspect = u.resolution.x / u.resolution.y;
    let direction = normalize(
        u.forward
            + u.right * ndc.x * u.fov_y_tan * aspect
            - u.up * ndc.y * u.fov_y_tan,
    );

    let hit = lens_march(u.eye, direction);
    if !hit.hit {
        return vec4<f32>(VOID_COLOR, 1.0);
    }

    let sphere = lens_scene_nearest(hit.point);
    let normal = lens_normal(hit.point, sphere);
    let lambert = max(dot(normal, -hit.tangent), 0.0) * 0.82 + 0.18;

    // Cosine palette (Quilez 2015, "palettes") keyed on the sphere, and only on
    // the sphere. A per-copy tint would need a label for "which copy", and no
    // such label is a function of the point of the quotient: the copies are one
    // object seen p times. Depth does the separating instead, which is why the
    // fog below is steep.
    let tint = 0.55 + 0.4 * cos(6.2831855 * 0.19 * f32(sphere) + vec3<f32>(0.0, 2.1, 4.2));

    let fog = exp(-0.55 * hit.arc);
    return vec4<f32>(mix(VOID_COLOR, tint * lambert, fog), 1.0);
}
"#;

fn main() {
    let lens = LensSpace::new(P, Q);
    let source = format!("{}\n{}\n{}", lens.wgsl_prelude(), scene_wgsl(), LENS_SHADER);

    let (device, queue) = pollster::block_on(request_device());
    let renderer = Renderer::new(&device, &source);

    // One deck displacement along the z1 circle: the shortest closed geodesic
    // of L(p, q), of length 2π/p.
    let loop_length = TAU / P as f32;
    let mut tiles = Vec::with_capacity(FRAMES as usize);
    for frame in 0..FRAMES {
        let arc = loop_length * frame as f32 / (FRAMES - 1) as f32;
        tiles.push(renderer.render(&device, &queue, walker(arc)));
    }

    // The same observer reached the other way: one deck element applied to the
    // walker at the start. Its frame is the start frame with the z2 plane
    // turned by -2πq/p, which is the holonomy of the loop, so this is also the
    // measurement that the twist is the one L(p, q) asks for and not zero.
    let start = walker(0.0);
    let inverse = lens.deck(-1);
    let reference = Observer {
        eye: start.eye,
        forward: start.forward,
        right: lens.iso_transport(inverse, start.eye, start.right),
        up: lens.iso_transport(inverse, start.eye, start.up),
    };
    let reference_tile = renderer.render(&device, &queue, reference);

    let closure = channel_difference(&tiles[FRAMES as usize - 1], &reference_tile);
    // The control: the eighth tile against the first, which is the same walk
    // assumed to close with no roll at all. It is here so the closure number
    // above is read against something rather than against a threshold someone
    // chose, and it is also the size of the holonomy the twist carries.
    let untwisted = channel_difference(&tiles[FRAMES as usize - 1], &tiles[0]);
    println!(
        "L({P}, {Q}), closed geodesic of length {loop_length:.4}: deck-related observers \
         differ by mean {:.4}/255 (worst {}/255, {:.4}% of channels past a quantisation \
         step); the same pair without the 2πq/p roll differs by mean {:.4}/255 (worst \
         {}/255, {:.2}%)",
        closure.mean,
        closure.worst,
        100.0 * closure.disagreeing,
        untwisted.mean,
        untwisted.worst,
        100.0 * untwisted.disagreeing
    );

    let path = write_filmstrip(&tiles);
    println!("capture: {}", path.display());

    // Two separate marches from different covering coordinates, so the bound is
    // a quantisation step rather than bit equality. What makes it a measurement
    // and not a threshold is the control: a gluing that failed to close moves
    // whole spheres and the untwisted comparison above is what that costs.
    assert!(
        closure.worst <= 1 && untwisted.mean > 100.0 * closure.mean.max(0.01),
        "the loop did not close: mean {:.4}/255, worst {}/255, against an untwisted \
         control of {:.4}/255",
        closure.mean,
        closure.worst,
        untwisted.mean
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
            label: Some("lens space"),
            required_features: Features::empty(),
            required_limits: Limits::default(),
            memory_hints: MemoryHints::default(),
            trace: Trace::Off,
            experimental_features: Default::default(),
        })
        .await
        .expect("wgpu device")
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LensUniforms {
    eye: [f32; 4],
    forward: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    resolution: [f32; 2],
    fov_y_tan: f32,
    _pad: f32,
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
            label: Some("lens space"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("lens space uniforms"),
            size: std::mem::size_of::<LensUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("lens space"),
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
            label: Some("lens space"),
            layout: &layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("lens space"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("lens space"),
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
            label: Some("lens space color"),
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
            label: Some("lens space readback"),
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

    fn render(&self, device: &Device, queue: &Queue, observer: Observer) -> Vec<u8> {
        let uniforms = LensUniforms {
            eye: observer.eye.to_array(),
            forward: observer.forward.to_array(),
            right: observer.right.to_array(),
            up: observer.up.to_array(),
            resolution: [TILE_WIDTH as f32, TILE_HEIGHT as f32],
            fov_y_tan: (70.0_f32.to_radians() * 0.5).tan(),
            _pad: 0.0,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("lens space"),
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
    let path = directory.join(format!("lens_space_l{P}_{Q}_{stamp}.png"));

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
