//! Hypersphere `w`-slice viewer — drop a 4D ball onto a 4D floor and
//! render the 3D cross-section at user-controlled `w₀`.
//!
//! The 4-ball `B = { x ∈ R⁴ : |x − c| ≤ r }` cross-sectioned at
//! `w = w₀` is a 3D ball of radius `sqrt(r² − (w₀ − c.w)²)` for
//! `|w₀ − c.w| < r`, and empty otherwise. As you scrub `w₀` past the
//! body's `w`-coordinate the rendered ball grows from a point, peaks
//! at radius `r` when `w₀ = c.w`, and shrinks back to a point. That
//! growth-and-shrink is the visible signature of "we are slicing a
//! 4D object through three dimensions."
//!
//! Visually this is simpler than the pentatope viewer (a 4-ball has
//! no rotation degrees of freedom worth distinguishing — its
//! cross-section is always a 3-ball) but it pins down the basic
//! 4D-physics-with-3D-rendering pipeline cleanly: drop a body, watch
//! it settle, scrub `w` to confirm it's really 4-dimensional.
//!
//! ## w-slice convention
//!
//! Same as `examples/pentatope_slice`: the slice plane sits at
//! `w_slice = body.w + w_offset`, so offset 0 always cuts through the
//! body's centroid (maximum cross-section radius). Held ↑/↓ scrubs
//! `w_offset` smoothly.
//!
//! ## Controls
//!
//! - Mouse: orbit camera (left-drag), zoom (scroll).
//! - **Space**: pause / resume physics.
//! - **↑ / ↓**: hold to slide the cross-section along `w` (0.6 u/s,
//!   range ±1.5).
//! - **A**: toggle automatic offset sweep (cosine-paced).
//! - **R**: reset — re-spawn the ball at `y = 2.5`, offset = 0.
//! - **Esc**: exit.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};
use rye_asset::AssetWatcher;
use rye_camera::OrbitCamera;
use rye_input::InputState;
use rye_math::{EuclideanR3, EuclideanR4};
use rye_physics::{
    euclidean_r4::{halfspace4_body_r4, register_default_narrowphase, sphere_body_r4},
    field::Gravity,
    World,
};
use rye_render::{device::RenderDevice, graph::RenderNode};
use rye_shader::{ShaderDb, ShaderId};
use rye_time::FixedTimestep;
use wgpu::{util::DeviceExt, *};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, KeyCode, NamedKey, PhysicalKey},
    window::{Window, WindowAttributes},
};

fn shader_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/hypersphere")
}
fn shader_path() -> PathBuf {
    shader_dir().join("hypersphere.wgsl")
}

const TITLE: &str = "Rye — hypersphere w-slice (live)";
const RADIUS_4D: f32 = 1.0;
/// Offset range: ±1.5 covers the full ball (radius 1) plus margin so
/// the user can scrub past the poles and watch the cross-section
/// vanish.
const W_OFFSET_RANGE: f32 = 1.5;
const W_SWEEP_RATE: f32 = 0.6;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SliceUniforms {
    camera_pos: [f32; 3],
    _pad0: f32,
    camera_forward: [f32; 3],
    _pad1: f32,
    camera_right: [f32; 3],
    _pad2: f32,
    camera_up: [f32; 3],
    fov_y_tan: f32,
    resolution: [f32; 2],
    time: f32,
    tick: f32,
    /// `[w_slice, radius4, body_w, _]`.
    params: [f32; 4],
    /// Body center xyz (for use as the cross-section ball's center).
    body_xyz: [f32; 3],
    _pad3: f32,
}

impl Default for SliceUniforms {
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
            params: [0.0; 4],
            body_xyz: [0.0; 3],
            _pad3: 0.0,
        }
    }
}

struct SliceNode {
    pipeline: RenderPipeline,
    uniforms: SliceUniforms,
    uniform_buf: Buffer,
    bind_group: BindGroup,
}

impl SliceNode {
    fn new(device: &Device, surface_format: TextureFormat, shader: &ShaderModule) -> Self {
        let uniforms = SliceUniforms::default();
        let uniform_buf = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("hypersphere uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("hypersphere bgl"),
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
            label: Some("hypersphere bg"),
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("hypersphere pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("hypersphere pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: shader,
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
            uniforms,
            uniform_buf,
            bind_group,
        }
    }

    fn set_uniforms(&mut self, queue: &Queue, uniforms: SliceUniforms) {
        self.uniforms = uniforms;
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&self.uniforms));
    }

    fn execute_frame(&self, rd: &RenderDevice, view: &TextureView) -> Result<()> {
        let mut encoder = rd.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("hypersphere encoder"),
        });
        {
            let mut rp = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("hypersphere pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
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
            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.bind_group, &[]);
            rp.draw(0..3, 0..1);
        }
        rd.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}

impl RenderNode for SliceNode {
    fn name(&self) -> &'static str {
        "hypersphere"
    }
    fn execute(&mut self, rd: &RenderDevice, view: &TextureView) -> Result<()> {
        self.execute_frame(rd, view)
    }
}

fn build_world() -> (World<EuclideanR4>, usize) {
    let mut world = World::new(EuclideanR4);
    register_default_narrowphase(&mut world.narrowphase);
    world.push_field(Box::new(Gravity::new(Vec4::new(0.0, -9.8, 0.0, 0.0))));
    let floor_id = world.push_body(halfspace4_body_r4(Vec4::Y, 0.0));
    world.bodies[floor_id].restitution = 0.0;
    let ball_id = world.push_body(sphere_body_r4(
        Vec4::new(0.0, 2.5, 0.0, 0.0),
        Vec4::ZERO,
        RADIUS_4D,
        1.0,
    ));
    world.bodies[ball_id].restitution = 0.0;
    (world, ball_id)
}

struct App {
    window: Option<Arc<Window>>,
    rd: Option<RenderDevice>,
    minimized: bool,

    shaders: Option<ShaderDb>,
    shader_id: Option<ShaderId>,
    shader_gen: u64,
    watcher: Option<AssetWatcher>,
    node: Option<SliceNode>,

    timestep: FixedTimestep,
    camera: OrbitCamera,
    input: InputState,
    start: Instant,

    world: World<EuclideanR4>,
    ball_id: usize,
    paused: bool,

    w_offset: f32,
    auto_sweep: bool,
    sweep_anchor: Instant,
    slider_up_held: bool,
    slider_down_held: bool,

    frame_count: u32,
    last_fps_update: Instant,
    fps: f32,
}

impl App {
    fn new() -> Self {
        let (world, ball_id) = build_world();
        Self {
            window: None,
            rd: None,
            minimized: false,
            shaders: None,
            shader_id: None,
            shader_gen: 0,
            watcher: None,
            node: None,
            timestep: FixedTimestep::new(60),
            camera: {
                let mut c = OrbitCamera::default();
                c.set_orbit(8.0, -0.35);
                c
            },
            input: InputState::default(),
            start: Instant::now(),
            world,
            ball_id,
            paused: false,
            w_offset: 0.0,
            auto_sweep: false,
            sweep_anchor: Instant::now(),
            slider_up_held: false,
            slider_down_held: false,
            frame_count: 0,
            last_fps_update: Instant::now(),
            fps: 0.0,
        }
    }

    fn reset(&mut self) {
        let (world, ball_id) = build_world();
        self.world = world;
        self.ball_id = ball_id;
        self.w_offset = 0.0;
        self.auto_sweep = false;
        self.sweep_anchor = Instant::now();
        self.paused = false;
    }

    fn advance_auto_sweep(&mut self) {
        if !self.auto_sweep {
            return;
        }
        let phase = (self.sweep_anchor.elapsed().as_secs_f32() / 8.0) * std::f32::consts::TAU;
        self.w_offset = W_OFFSET_RANGE * phase.cos();
    }

    fn advance_slider(&mut self, dt: f32) {
        let dir = (self.slider_up_held as i32 - self.slider_down_held as i32) as f32;
        if dir != 0.0 {
            self.w_offset =
                (self.w_offset + dir * W_SWEEP_RATE * dt).clamp(-W_OFFSET_RANGE, W_OFFSET_RANGE);
        }
    }

    fn handle_keyboard(&mut self, code: PhysicalKey, state: ElementState) {
        let PhysicalKey::Code(kc) = code else {
            return;
        };
        let pressed = state == ElementState::Pressed;
        match kc {
            KeyCode::ArrowUp => {
                if pressed && !self.slider_up_held {
                    self.auto_sweep = false;
                }
                self.slider_up_held = pressed;
                return;
            }
            KeyCode::ArrowDown => {
                if pressed && !self.slider_down_held {
                    self.auto_sweep = false;
                }
                self.slider_down_held = pressed;
                return;
            }
            _ => {}
        }
        if !pressed {
            return;
        }
        match kc {
            KeyCode::KeyA => {
                self.auto_sweep = !self.auto_sweep;
                if self.auto_sweep {
                    self.sweep_anchor = Instant::now();
                }
            }
            KeyCode::KeyR => self.reset(),
            KeyCode::Space => self.paused = !self.paused,
            _ => {}
        }
    }

    fn effective_w_slice(&self) -> f32 {
        self.world.bodies[self.ball_id].position.w + self.w_offset
    }

    fn current_uniforms(&self) -> Option<SliceUniforms> {
        let rd = self.rd.as_ref()?;
        let view = self.camera.view();
        let t = self.start.elapsed().as_secs_f32();
        let body = &self.world.bodies[self.ball_id];
        Some(SliceUniforms {
            camera_pos: view.position.to_array(),
            _pad0: 0.0,
            camera_forward: view.forward.to_array(),
            _pad1: 0.0,
            camera_right: view.right.to_array(),
            _pad2: 0.0,
            camera_up: view.up.to_array(),
            fov_y_tan: (60.0_f32.to_radians() * 0.5).tan(),
            resolution: [
                rd.surface_bundle.config.width as f32,
                rd.surface_bundle.config.height as f32,
            ],
            time: t,
            tick: self.timestep.tick() as f32,
            params: [self.effective_w_slice(), RADIUS_4D, body.position.w, 0.0],
            body_xyz: [body.position.x, body.position.y, body.position.z],
            _pad3: 0.0,
        })
    }

    fn handle_hot_reload(&mut self) {
        let (Some(watcher), Some(shaders), Some(id), Some(rd)) = (
            self.watcher.as_ref(),
            self.shaders.as_mut(),
            self.shader_id,
            self.rd.as_ref(),
        ) else {
            return;
        };
        let events = watcher.poll();
        if events.is_empty() {
            return;
        }
        shaders.apply_events(&events, &EuclideanR3);
        let new_gen = shaders.generation(id);
        if new_gen != self.shader_gen {
            tracing::info!("rebuilding SliceNode for shader gen {new_gen}");
            self.shader_gen = new_gen;
            self.node = Some(SliceNode::new(
                &rd.device,
                rd.surface_bundle.config.format,
                shaders.module(id),
            ));
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, elwt: &ActiveEventLoop) {
        let win = Arc::new(
            elwt.create_window(
                WindowAttributes::default()
                    .with_title(TITLE)
                    .with_visible(false),
            )
            .expect("create window"),
        );
        let rd = pollster::block_on(RenderDevice::new(win.clone())).expect("render device");

        let mut shaders = ShaderDb::new(rd.device.clone());
        let id = shaders
            .load(shader_path(), &EuclideanR3)
            .expect("load hypersphere.wgsl");
        let gen = shaders.generation(id);
        let mut watcher = AssetWatcher::new().expect("asset watcher");
        watcher.watch(shader_dir()).expect("watch shader dir");
        let node = SliceNode::new(
            &rd.device,
            rd.surface_bundle.config.format,
            shaders.module(id),
        );

        self.window = Some(win.clone());
        self.rd = Some(rd);
        self.shaders = Some(shaders);
        self.shader_id = Some(id);
        self.shader_gen = gen;
        self.watcher = Some(watcher);
        self.node = Some(node);
        self.minimized = false;
        self.start = Instant::now();
        self.sweep_anchor = Instant::now();

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
            WindowEvent::KeyboardInput { event, .. } => {
                self.input.key_input(event.physical_key, event.state);
                self.handle_keyboard(event.physical_key, event.state);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input.cursor_moved(position.x, position.y);
            }
            WindowEvent::CursorLeft { .. } => self.input.cursor_invalidated(),
            WindowEvent::Focused(false) => {
                self.input.cursor_invalidated();
                self.input.release_buttons();
                self.slider_up_held = false;
                self.slider_down_held = false;
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
                let ticks = self.timestep.advance(Instant::now());
                let n_ticks = ticks.count();
                if n_ticks > 0 {
                    let frame_input = self.input.take_frame();
                    self.camera.advance(frame_input);
                    self.advance_slider(n_ticks as f32 / 60.0);
                    self.advance_auto_sweep();
                    if !self.paused {
                        for _ in 0..n_ticks.min(4) {
                            self.world.step(1.0 / 60.0);
                        }
                    }
                }
                self.handle_hot_reload();

                self.frame_count += 1;
                let elapsed = self.last_fps_update.elapsed().as_secs_f32();
                if elapsed >= 1.0 {
                    self.fps = self.frame_count as f32 / elapsed;
                    self.frame_count = 0;
                    self.last_fps_update = Instant::now();
                    let body = &self.world.bodies[self.ball_id];
                    let p = body.position;
                    let pause = if self.paused { " [paused]" } else { "" };
                    let mode = if self.auto_sweep { "auto" } else { "manual" };
                    let w_eff = p.w + self.w_offset;
                    // Visible-radius indicator: matches `slice_radius` in
                    // the WGSL — useful for sanity-checking the cross-
                    // section as the user scrubs.
                    let dw = w_eff - p.w;
                    let r3_sq = RADIUS_4D * RADIUS_4D - dw * dw;
                    let r3 = if r3_sq > 0.0 { r3_sq.sqrt() } else { 0.0 };
                    win.set_title(&format!(
                        "{TITLE} | {:.0} fps | offset={:+.2} ({mode}) w₀={:+.2}{pause} | r₃={:.3} | pos.y={:+.2} pos.w={:+.2}",
                        self.fps, self.w_offset, w_eff, r3, p.y, p.w
                    ));
                }

                let Some(uniforms) = self.current_uniforms() else {
                    return;
                };
                let Some(rd) = self.rd.as_ref() else { return };
                if let Some(node) = self.node.as_mut() {
                    node.set_uniforms(&rd.queue, uniforms);
                }
                match rd.begin_frame() {
                    Ok((frame, view)) => {
                        if let Some(node) = self.node.as_mut() {
                            if let Err(e) = node.execute_frame(rd, &view) {
                                tracing::error!("render error: {e:#}");
                            }
                        }
                        frame.present();
                        win.request_redraw();
                    }
                    Err(err) => match err {
                        SurfaceError::Lost | SurfaceError::Outdated => {
                            if let Some(rd) = &mut self.rd {
                                let size = rd.surface_bundle.size;
                                rd.resize(size);
                            }
                            win.request_redraw();
                        }
                        SurfaceError::Timeout => win.request_redraw(),
                        SurfaceError::OutOfMemory => elwt.exit(),
                        SurfaceError::Other => {
                            tracing::error!("surface error: {err:?}");
                            win.request_redraw();
                        }
                    },
                }
            }
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let elwt = EventLoop::new()?;
    elwt.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::new();
    elwt.run_app(&mut app)?;
    let _ = Vec3::ZERO;
    Ok(())
}
