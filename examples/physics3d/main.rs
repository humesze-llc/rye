//! 3D physics demo — spheres fall onto an infinite floor under gravity,
//! bounce off each other off-center, and tumble via friction-induced
//! angular velocity.
//!
//! Visualization: raymarched per-body sphere SDFs + floor half-space,
//! simple sun-lambert shading. Orbit camera (left-drag to rotate,
//! scroll to zoom). R resets; Esc exits.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use rye_camera::OrbitCamera;
use rye_input::InputState;
use rye_math::EuclideanR3;
use rye_physics::{
    euclidean_r3::{box_body, halfspace_body_r3, register_default_narrowphase, sphere_body_r3},
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

const MAX_BODIES: usize = 32;
const FIXED_DT: f32 = 1.0 / 120.0;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SceneUniforms {
    camera_pos: [f32; 3],
    _pad0: f32,
    camera_forward: [f32; 3],
    fov_y_tan: f32,
    camera_right: [f32; 3],
    _pad1: f32,
    camera_up: [f32; 3],
    _pad2: f32,
    resolution: [f32; 2],
    body_count: u32,
    _pad3: u32,
    floor_normal: [f32; 3],
    floor_offset: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct GpuBody {
    position: [f32; 3],
    /// 0.0 = sphere, 1.0 = box. Packed as float for WGSL alignment.
    kind: f32,
    /// For sphere: (radius, 0, 0). For box: half_extents.
    extents: [f32; 3],
    _pad: f32,
    /// Unit quaternion (x, y, z, w). Identity for spheres.
    rotation: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BodyBuffer([GpuBody; MAX_BODIES]);

// Compile-time layout checks. GpuBody is 48 bytes: vec3+f32 (16) +
// vec3+f32 (16) + vec4 (16). Matches the WGSL Body struct.
const _: () = assert!(std::mem::size_of::<SceneUniforms>() == 96);
const _: () = assert!(std::mem::size_of::<GpuBody>() == 48);
const _: () = assert!(std::mem::size_of::<BodyBuffer>() == 48 * MAX_BODIES);

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

fn build_world() -> World<EuclideanR3> {
    let mut world = World::new(EuclideanR3);
    register_default_narrowphase(&mut world.narrowphase);
    world.push_field(Box::new(Gravity::new(Vec3::new(0.0, -9.8, 0.0))));

    // Infinite floor at y = 0.
    world.push_body(halfspace_body_r3(Vec3::Y, 0.0));

    let mut rng = Rng(0x51A3);
    // Mix of spheres and boxes falling with mild horizontal velocity.
    // Boxes exercise the full GJK+EPA path; spheres stay on the
    // analytical sphere-sphere / sphere-halfspace fast paths.
    for i in 0..14 {
        let x = rng.range(-2.0, 2.0);
        let z = rng.range(-2.0, 2.0);
        let y = 2.5 + (i as f32) * 0.55;
        let vel = Vec3::new(rng.range(-0.4, 0.4), 0.0, rng.range(-0.4, 0.4));
        if i % 2 == 0 {
            let radius = rng.range(0.25, 0.40);
            world.push_body(sphere_body_r3(Vec3::new(x, y, z), vel, radius, 1.0));
        } else {
            let hx = rng.range(0.22, 0.36);
            let hy = rng.range(0.22, 0.36);
            let hz = rng.range(0.22, 0.36);
            world.push_body(box_body(
                Vec3::new(x, y, z),
                vel,
                Vec3::new(hx, hy, hz),
                1.0,
            ));
        }
    }

    world
}

fn collect_gpu_bodies(world: &World<EuclideanR3>) -> (BodyBuffer, u32) {
    let mut buf = BodyBuffer([GpuBody::default(); MAX_BODIES]);
    // Skip the floor; it's rendered via Scene.floor_normal/offset.
    let mut count = 0u32;
    for body in world.bodies.iter() {
        if count as usize >= MAX_BODIES {
            break;
        }
        // Skip bodies whose state went non-finite — feeding NaN
        // positions/quats to wgpu triggers validation panics that
        // look unrelated to the physics bug that produced them.
        if !body.position.is_finite() {
            continue;
        }
        let q = body.orientation.rotation;
        if !q.length_squared().is_finite() {
            continue;
        }
        let rotation = [q.x, q.y, q.z, q.w];

        match &body.collider {
            Collider::Sphere { radius } => {
                buf.0[count as usize] = GpuBody {
                    position: body.position.to_array(),
                    kind: 0.0,
                    extents: [*radius, 0.0, 0.0],
                    _pad: 0.0,
                    rotation,
                };
                count += 1;
            }
            Collider::ConvexPolytope3D { vertices } => {
                // Assume the polytope is a box for visualization
                // purposes: recover half-extents as the AABB of the
                // local vertex set. Works for `box_body()` output; for
                // arbitrary polytopes this renders their OBB instead
                // of the true shape.
                let (mut mn, mut mx) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
                for &v in vertices {
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                let half = (mx - mn) * 0.5;
                buf.0[count as usize] = GpuBody {
                    position: body.position.to_array(),
                    kind: 1.0,
                    extents: half.to_array(),
                    _pad: 0.0,
                    rotation,
                };
                count += 1;
            }
            Collider::HalfSpace { .. }
            | Collider::Polygon2D { .. }
            | Collider::ConvexPolytope4D { .. } => {
                // Halfspace is implicit (scene uniforms); 2D polygons
                // and 4D polytopes shouldn't appear in a 3D world but
                // skip them defensively.
            }
        }
    }
    (buf, count)
}

struct App {
    window: Option<Arc<Window>>,
    rd: Option<RenderDevice>,
    minimized: bool,
    pipeline: Option<wgpu::RenderPipeline>,
    scene_buffer: Option<wgpu::Buffer>,
    body_buffer: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,

    world: World<EuclideanR3>,
    camera: OrbitCamera,
    input: InputState,

    last_frame: Instant,
    accumulator: f32,
    sim_time: f32,
    start: Instant,
    frame_count: u32,
    last_fps: Instant,
}

impl Default for App {
    fn default() -> Self {
        // Seed orbit camera to look at the bucket area from slightly above.
        let mut camera = OrbitCamera::default();
        camera.set_orbit(6.0, -0.45);
        Self {
            window: None,
            rd: None,
            minimized: false,
            pipeline: None,
            scene_buffer: None,
            body_buffer: None,
            bind_group: None,
            world: build_world(),
            camera,
            input: InputState::default(),
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
            // Wrap the physics step in `catch_unwind` to surface any
            // Rust panic explicitly (diagnostic for the "no backtrace
            // on crash" mystery). If the crash is a non-Rust event
            // (stack overflow, driver abort), this won't catch it —
            // but seeing the panic message here would tell us it's a
            // Rust issue.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.world.step(FIXED_DT);
            }));
            match result {
                Ok(()) => {}
                Err(payload) => {
                    let msg = if let Some(s) = payload.downcast_ref::<String>() {
                        s.clone()
                    } else if let Some(s) = payload.downcast_ref::<&'static str>() {
                        (*s).to_string()
                    } else {
                        "<non-string panic payload>".to_string()
                    };
                    eprintln!("PHYSICS STEP PANIC CAUGHT: {msg}");
                    eprintln!(
                        "  sim_time={:.3}s  dynamic_bodies={}",
                        self.sim_time,
                        self.world
                            .bodies
                            .iter()
                            .filter(|b| b.inv_mass > 0.0)
                            .count()
                    );
                    // Resume unwinding so the app-level hook prints the
                    // full backtrace.
                    std::panic::resume_unwind(payload);
                }
            }
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
        label: Some("physics3d pipeline layout"),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("physics3d pipeline"),
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
                    .with_title("Rye — 3D Physics")
                    .with_inner_size(winit::dpi::LogicalSize::new(900.0, 720.0))
                    .with_visible(false),
            )
            .expect("create window"),
        );

        let rd = pollster::block_on(RenderDevice::new(win.clone())).expect("render device");

        let shader_src = include_str!("physics3d.wgsl");
        let module = rd
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("physics3d"),
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });

        let scene_buffer = rd.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics3d scene ub"),
            size: std::mem::size_of::<SceneUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let body_buffer = rd.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics3d bodies ub"),
            size: std::mem::size_of::<BodyBuffer>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = rd
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("physics3d bgl"),
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
            label: Some("physics3d bg"),
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
        let Some(win) = self.window.clone() else {
            return;
        };

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

            WindowEvent::CursorMoved { position, .. } => {
                self.input.cursor_moved(position.x, position.y);
            }
            WindowEvent::CursorLeft { .. } => self.input.cursor_invalidated(),
            WindowEvent::Focused(false) => {
                self.input.cursor_invalidated();
                self.input.release_buttons();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.input.mouse_input(button, state);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.input.mouse_wheel(delta);
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
                // Advance camera from drained input.
                let frame = self.input.take_frame();
                self.camera.advance(frame);

                self.step_physics();
                self.render();
                win.request_redraw();

                self.frame_count += 1;
                let elapsed = self.last_fps.elapsed().as_secs_f32();
                if elapsed >= 0.5 {
                    let fps = self.frame_count as f32 / elapsed;
                    self.frame_count = 0;
                    self.last_fps = Instant::now();
                    let dynamic_count = self
                        .world
                        .bodies
                        .iter()
                        .filter(|b| b.inv_mass > 0.0)
                        .count();
                    win.set_title(&format!(
                        "Rye — 3D Physics | {fps:.0} fps | {dynamic_count} bodies | sim {:.1}s (R: reset, drag: orbit, scroll: zoom)",
                        self.sim_time
                    ));
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

        let view = self.camera.view();
        let (body_data, body_count) = collect_gpu_bodies(&self.world);

        let scene = SceneUniforms {
            camera_pos: view.position.to_array(),
            _pad0: 0.0,
            camera_forward: view.forward.to_array(),
            fov_y_tan: (60.0_f32.to_radians() * 0.5).tan(),
            camera_right: view.right.to_array(),
            _pad1: 0.0,
            camera_up: view.up.to_array(),
            _pad2: 0.0,
            resolution: [
                rd.surface_bundle.config.width as f32,
                rd.surface_bundle.config.height as f32,
            ],
            body_count,
            _pad3: 0,
            floor_normal: [0.0, 1.0, 0.0],
            floor_offset: 0.0,
        };
        rd.queue
            .write_buffer(scene_buffer, 0, bytemuck::bytes_of(&scene));
        rd.queue
            .write_buffer(body_buffer, 0, bytemuck::bytes_of(&body_data));

        let Ok((frame, view_tex)) = rd.begin_frame() else {
            return;
        };

        let mut encoder = rd
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("physics3d encoder"),
            });
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("physics3d pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view_tex,
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

    // Force RUST_BACKTRACE=1 so a panic in the physics step produces
    // an actual stack trace even if the user didn't set the env var.
    // Also install a hook that flushes stderr before the process dies
    // — panics on the main/event-loop thread don't always get a chance
    // to print through the default hook when winit unwinds weirdly.
    std::env::set_var("RUST_BACKTRACE", "1");
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        eprintln!("\n=== PHYSICS3D PANIC ===");
        eprintln!("{info}");
        let bt = std::backtrace::Backtrace::force_capture();
        eprintln!("backtrace:\n{bt}");
        eprintln!("=======================\n");
        use std::io::Write;
        let _ = std::io::stderr().flush();
        default_hook(info);
    }));

    let event_loop = EventLoop::new()?;
    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}
