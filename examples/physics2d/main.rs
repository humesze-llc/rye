//! 2D physics demo — spawns a bucket and a handful of random shapes
//! (circles, triangles, squares, pentagons, hexagons) and lets them
//! fall under gravity, collide, and settle.
//!
//! Visualization: each body's SDF is evaluated per fragment by
//! `physics2d.wgsl`. No raymarching — this is flat 2D.
//!
//! Press Esc or close the window to exit. Press R to reset.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use rye_math::EuclideanR2;
use rye_physics::{
    euclidean_r2::{polygon_body, register_default_narrowphase, sphere_body, static_wall},
    field::Gravity,
    Collider, World,
};
use rye_render::device::RenderDevice;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, KeyCode, NamedKey, PhysicalKey},
    window::{Window, WindowAttributes},
};

// World viewport: x ∈ [−5, 5], y ∈ [0, 8].
const WORLD_X_MIN: f32 = -5.0;
const WORLD_Y_MIN: f32 = 0.0;
const WORLD_WIDTH: f32 = 10.0;
const WORLD_HEIGHT: f32 = 8.0;

const MAX_BODIES: usize = 64;

const FIXED_DT: f32 = 1.0 / 120.0;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SceneUniforms {
    view: [f32; 4],
    resolution: [f32; 2],
    body_count: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct GpuBody {
    position: [f32; 2],
    rotation: [f32; 2],
    scale: f32,
    kind: u32,
    extent: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BodyBuffer([GpuBody; MAX_BODIES]);

/// Deterministic tiny RNG for reproducible scenes.
struct Rng(u32);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.0 >> 16) & 0xFFFF) as f32 / 65535.0
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

fn build_world() -> World<EuclideanR2> {
    let mut world = World::new(EuclideanR2);
    register_default_narrowphase(&mut world.narrowphase);
    world.push_field(Box::new(Gravity::new(Vec2::new(0.0, -9.8))));

    // Bucket: floor + two walls. The floor's top face is at y=0.5; its
    // center is at y=0.25 with half-height 0.25.
    world.push_body(static_wall(Vec2::new(0.0, 0.25), Vec2::new(4.8, 0.25)));
    world.push_body(static_wall(Vec2::new(-4.8, 4.0), Vec2::new(0.2, 3.75)));
    world.push_body(static_wall(Vec2::new(4.8, 4.0), Vec2::new(0.2, 3.75)));

    // Drop a random mix of shapes. Staggered y so they don't all spawn
    // at the same height (which would produce instant overlap).
    let mut rng = Rng(0xDEADBEEF);
    for i in 0..14 {
        let x = rng.range(-3.8, 3.8);
        let y = 4.0 + (i as f32) * 0.3;
        let kind = (rng.next_f32() * 5.0) as u32; // 0..=4
        let radius = rng.range(0.22, 0.40);
        let vel = Vec2::new(rng.range(-0.7, 0.7), 0.0);
        match kind {
            0 => {
                world.push_body(sphere_body(Vec2::new(x, y), vel, radius, 1.0));
            }
            k @ 1..=4 => {
                // kind 1→triangle(3), 2→square(4), 3→pentagon(5), 4→hexagon(6)
                world.push_body(polygon_body(Vec2::new(x, y), vel, k + 2, radius, 1.0));
            }
            _ => unreachable!(),
        }
    }

    world
}

fn collect_gpu_bodies(world: &World<EuclideanR2>) -> (BodyBuffer, u32) {
    let mut buf = BodyBuffer([GpuBody::default(); MAX_BODIES]);
    let count = world.bodies.len().min(MAX_BODIES);
    for (i, b) in world.bodies.iter().take(MAX_BODIES).enumerate() {
        // Half-angle rotor → full-angle (cos, sin) for the shader.
        let r = b.orientation.rotation;
        let c = r.a * r.a - r.b * r.b;
        let s = 2.0 * r.a * r.b;

        let (scale, kind, extent) = match &b.collider {
            Collider::Sphere { radius, .. } => (*radius, 0u32, [0.0, 0.0]),
            Collider::Polygon2D { vertices } => {
                // Static-wall convention: inv_mass == 0 with 4 axis-aligned
                // vertices → render as a rectangle (kind 100).
                if b.inv_mass == 0.0 && vertices.len() == 4 {
                    let hx = vertices.iter().map(|v| v.x.abs()).fold(0.0, f32::max);
                    let hy = vertices.iter().map(|v| v.y.abs()).fold(0.0, f32::max);
                    (0.0, 100u32, [hx, hy])
                } else {
                    let circumradius = vertices.iter().map(|v| v.length()).fold(0.0, f32::max);
                    (circumradius, vertices.len() as u32, [0.0, 0.0])
                }
            }
            _ => (0.5, 0u32, [0.0, 0.0]),
        };

        buf.0[i] = GpuBody {
            position: b.position.to_array(),
            rotation: [c, s],
            scale,
            kind,
            extent,
        };
    }
    (buf, count as u32)
}

struct App {
    window: Option<Arc<Window>>,
    rd: Option<RenderDevice>,
    minimized: bool,
    pipeline: Option<wgpu::RenderPipeline>,
    scene_buffer: Option<wgpu::Buffer>,
    body_buffer: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,

    world: World<EuclideanR2>,
    last_frame: Instant,
    accumulator: f32,
    sim_time: f32,
    start: Instant,
    frame_count: u32,
    last_fps: Instant,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            rd: None,
            minimized: false,
            pipeline: None,
            scene_buffer: None,
            body_buffer: None,
            bind_group: None,
            world: build_world(),
            last_frame: Instant::now(),
            accumulator: 0.0,
            sim_time: 0.0,
            start: Instant::now(),
            frame_count: 0,
            last_fps: Instant::now(),
        }
    }
}

impl App {
    fn reset(&mut self) {
        self.world = build_world();
        self.sim_time = 0.0;
        self.accumulator = 0.0;
        self.last_frame = Instant::now();
    }

    fn step_physics(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.25);
        self.last_frame = now;
        self.accumulator += dt;
        while self.accumulator >= FIXED_DT {
            self.world.step(FIXED_DT);
            self.accumulator -= FIXED_DT;
            self.sim_time += FIXED_DT;
        }
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    module: &wgpu::ShaderModule,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("physics2d pipeline layout"),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("physics2d pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_fullscreen"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

impl ApplicationHandler for App {
    fn resumed(&mut self, elwt: &ActiveEventLoop) {
        let win = Arc::new(
            elwt.create_window(
                WindowAttributes::default()
                    .with_title("Rye — 2D Physics")
                    .with_inner_size(winit::dpi::LogicalSize::new(900.0, 720.0))
                    .with_visible(false),
            )
            .expect("create window"),
        );

        let rd = pollster::block_on(RenderDevice::new(win.clone())).expect("render device");

        let shader_src = include_str!("physics2d.wgsl");
        let module = rd
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("physics2d"),
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });

        let scene_buffer = rd.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics2d scene ub"),
            size: std::mem::size_of::<SceneUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let body_buffer = rd.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics2d bodies ub"),
            size: std::mem::size_of::<BodyBuffer>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = rd
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("physics2d bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bind_group = rd.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("physics2d bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: body_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline = create_pipeline(&rd.device, rd.surface_bundle.config.format, &module, &bgl);

        self.window = Some(win.clone());
        self.rd = Some(rd);
        self.pipeline = Some(pipeline);
        self.scene_buffer = Some(scene_buffer);
        self.body_buffer = Some(body_buffer);
        self.bind_group = Some(bind_group);
        self.minimized = false;
        self.last_frame = Instant::now();
        self.start = Instant::now();
        self.last_fps = Instant::now();

        win.set_visible(true);
        win.request_redraw();
    }

    fn window_event(
        &mut self,
        elwt: &ActiveEventLoop,
        _id: winit::window::WindowId,
        ev: WindowEvent,
    ) {
        match ev {
            WindowEvent::CloseRequested => elwt.exit(),

            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && matches!(event.logical_key, Key::Named(NamedKey::Escape)) =>
            {
                elwt.exit();
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && matches!(event.physical_key, PhysicalKey::Code(KeyCode::KeyR)) =>
            {
                self.reset();
            }

            WindowEvent::Resized(size) => {
                self.minimized = size.width == 0 || size.height == 0;
                if !self.minimized {
                    if let Some(rd) = &mut self.rd {
                        rd.resize(size);
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                if self.minimized {
                    return;
                }
                self.step_physics();
                self.render();
                if let Some(w) = &self.window {
                    w.request_redraw();
                }

                // Throttled title-bar FPS/body count for visual feedback.
                self.frame_count += 1;
                let elapsed = self.last_fps.elapsed().as_secs_f32();
                if elapsed >= 0.5 {
                    let fps = self.frame_count as f32 / elapsed;
                    self.frame_count = 0;
                    self.last_fps = Instant::now();
                    if let Some(w) = &self.window {
                        let dynamic_count = self
                            .world
                            .bodies
                            .iter()
                            .filter(|b| b.inv_mass > 0.0)
                            .count();
                        w.set_title(&format!(
                            "Rye — 2D Physics | {fps:.0} fps | {dynamic_count} bodies | sim {:.1}s (R: reset)",
                            self.sim_time
                        ));
                    }
                }
            }

            _ => {}
        }
    }
}

impl App {
    fn render(&mut self) {
        let (Some(rd), Some(pipeline), Some(scene_buffer), Some(body_buffer), Some(bind_group)) = (
            self.rd.as_ref(),
            self.pipeline.as_ref(),
            self.scene_buffer.as_ref(),
            self.body_buffer.as_ref(),
            self.bind_group.as_ref(),
        ) else {
            return;
        };

        let (body_data, body_count) = collect_gpu_bodies(&self.world);

        let scene = SceneUniforms {
            view: [WORLD_X_MIN, WORLD_Y_MIN, WORLD_WIDTH, WORLD_HEIGHT],
            resolution: [
                rd.surface_bundle.config.width as f32,
                rd.surface_bundle.config.height as f32,
            ],
            body_count,
            _pad: 0,
        };
        rd.queue
            .write_buffer(scene_buffer, 0, bytemuck::bytes_of(&scene));
        rd.queue
            .write_buffer(body_buffer, 0, bytemuck::bytes_of(&body_data));

        let Ok((frame, view)) = rd.begin_frame() else {
            return;
        };

        let mut encoder = rd
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("physics2d encoder"),
            });
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("physics2d pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.03,
                            g: 0.03,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rp.set_pipeline(pipeline);
            rp.set_bind_group(0, bind_group, &[]);
            rp.draw(0..3, 0..1);
        }
        rd.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let event_loop = EventLoop::new()?;
    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}
