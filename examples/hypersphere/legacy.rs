//! Hypersphere `w`-slice viewer, drop one or many 4D balls onto a
//! 4D floor and render their 3D cross-sections at user-controlled
//! `w₀`.
//!
//! The 4-ball `B = { x ∈ R⁴ : |x − c| ≤ r }` cross-sectioned at
//! `w = w₀` is a 3D ball of radius `sqrt(r² − (w₀ − c.w)²)` for
//! `|w₀ − c.w| < r`, and empty otherwise. As you scrub `w₀` past a
//! body's `w`-coordinate its rendered cross-section grows from a
//! point, peaks at radius `r` when `w₀ = c.w`, and shrinks back to
//! a point. That growth-and-shrink is the visible signature of "we
//! are slicing a 4D object through three dimensions."
//!
//! Visually this is simpler than the pentatope viewer (a 4-ball has
//! no rotation degrees of freedom worth distinguishing, its
//! cross-section is always a 3-ball) but it pins down the basic
//! 4D-physics-with-3D-rendering pipeline cleanly: drop bodies, watch
//! them collide and settle, scrub `w` to confirm they're really 4-
//! dimensional.
//!
//! ## CLI
//!
//! - `-n N` / `--count N`: spawn N hyperspheres (default 1, max 32).
//!   For `N > 1` the bodies are placed in a small lattice with
//!   staggered `y` and `w` so they fall, collide with each other, and
//!   settle as a 4D pile on the floor.
//!
//! ## Viewing modes
//!
//! - **Slice mode** (default), render the 3D cross-section of the
//!   bodies at the current `w₀`. Scrub `w₀` with ↑/↓ to peer at
//!   different cells; bodies outside the slice plane vanish.
//! - **Ghost mode** (toggle with **G**), render the bodies' full 4D
//!   extent simultaneously as a translucent volume. Each point in 3D
//!   gets opacity proportional to the body's `w`-extent through that
//!   xyz column (`2·√(r² − |xyz − c.xyz|²)` for points inside the
//!   xyz-projection, zero outside). A ray through the body's centre
//!   accumulates ≈ `2r` of `w`-thickness; a glancing ray accumulates
//!   less; missing rays accumulate nothing. The result is a smoothly
//!   shaded 3D shadow of the 4-ball that exposes its full extent
//!   without scrubbing `w`.
//!
//! ## Controls
//!
//! - Mouse: orbit camera (left-drag), zoom (scroll).
//! - **Space**: pause / resume physics.
//! - **↑ / ↓**: hold to slide the cross-section along `w` (slice mode
//!   only; 0.6 u/s, range ±1.5 around the first body's `w`).
//! - **A**: toggle automatic offset sweep (slice mode only).
//! - **G**: toggle ghost / slice viewing mode.
//! - **R**: reset, re-spawn all bodies, offset = 0.
//! - **Esc**: exit.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::Vec4;
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
    event_loop::ActiveEventLoop,
    keyboard::{Key, KeyCode, NamedKey, PhysicalKey},
    window::{Window, WindowAttributes},
};

fn shader_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/hypersphere")
}
fn shader_path() -> PathBuf {
    shader_dir().join("hypersphere.wgsl")
}

const TITLE: &str = "Rye - hypersphere w-slice (live)";
const RADIUS_4D: f32 = 1.0;
/// Offset range: ±1.5 covers the full ball (radius 1) plus margin so
/// the user can scrub past the poles and watch the cross-section
/// vanish. For multi-body mode we also extend it once the body pile
/// spreads in `w`, but the default ±1.5 already covers a couple of
/// adjacent bodies' worth of `w`-spread.
const W_OFFSET_RANGE: f32 = 1.5;
const W_SWEEP_RATE: f32 = 0.6;
/// Maximum number of hyperspheres rendered simultaneously. Hard-capped
/// here to keep the uniform layout fixed-size (`array<vec4, MAX_BODIES>`
/// in WGSL) and avoid runtime resource recreation.
const MAX_BODIES: usize = 32;

#[derive(Debug)]
struct Args {
    count: usize,
}

impl Args {
    fn parse() -> Self {
        let mut args = Args { count: 1 };
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "-n" | "--count" => {
                    args.count = iter
                        .next()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(args.count)
                        .clamp(1, MAX_BODIES);
                }
                "-h" | "--help" => {
                    println!("rye hypersphere w-slice viewer");
                    println!();
                    println!("Usage: cargo run --example hypersphere [options]");
                    println!();
                    println!("Options:");
                    println!(
                        "  -n, --count N    spawn N hyperspheres (default 1, max {MAX_BODIES})"
                    );
                    println!("  -h, --help       show this help and exit");
                    std::process::exit(0);
                }
                other => {
                    eprintln!("warning: unknown flag {other:?}; pass --help to see options");
                }
            }
        }
        args
    }
}

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
    /// `[w_slice, radius4, body_count_f, _]`. `body_count_f` is the
    /// number of active entries in `bodies`, encoded as `f32` for
    /// uniform-buffer simplicity (cast to `u32` in WGSL).
    params: [f32; 4],
    /// Up to `MAX_BODIES` body 4-positions (`vec4(x, y, z, w)` each).
    /// Slots `>= body_count` are unread by the shader.
    bodies: [[f32; 4]; MAX_BODIES],
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
            bodies: [[0.0; 4]; MAX_BODIES],
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

/// Spawn-position generator for body `i` of `count`.
///
/// One body: directly above the origin so the single-ball mode looks
/// like the original demo.
///
/// Many bodies: a stack/spiral of small offsets in `xz`, increasing
/// drop height in `y`, and gently varied `w` so the user actually
/// sees `w`-spread when scrubbing the slice plane. The `xz` offset is
/// kept inside ≈1.5× the ball radius so the bodies overlap visually
/// while falling and resolve into a 4D pile on the floor through
/// sphere-sphere collision.
fn spawn_position(i: usize, count: usize) -> Vec4 {
    if count <= 1 {
        return Vec4::new(0.0, 2.5, 0.0, 0.0);
    }
    let r_xz = 1.2 * RADIUS_4D;
    let phi = (i as f32 / count as f32) * std::f32::consts::TAU;
    let x = r_xz * phi.cos();
    let z = r_xz * phi.sin();
    // Each body starts at a different height so they don't all collide
    // simultaneously on the first frame; the higher ones fall onto
    // the lower ones over time.
    let y = 2.5 + 2.5 * RADIUS_4D * i as f32;
    // `w`-spread: alternate ± with a slowly growing magnitude so each
    // body's cross-section appears at a slightly different `w₀` and
    // the user can scrub from one to another. Within ±1.0 so all
    // bodies' cross-sections overlap the default ±1.5 sweep range.
    let w = 0.6 * RADIUS_4D * (i as f32 - 0.5 * (count - 1) as f32) / count.max(1) as f32;
    Vec4::new(x, y, z, w)
}

fn build_world(count: usize) -> (World<EuclideanR4>, Vec<usize>) {
    let mut world = World::new(EuclideanR4);
    register_default_narrowphase(&mut world.narrowphase);
    world.push_field(Box::new(Gravity::new(Vec4::new(0.0, -9.8, 0.0, 0.0))));
    let floor_id = world.push_body(halfspace4_body_r4(Vec4::Y, 0.0));
    world.bodies[floor_id].restitution = 0.0;
    let mut ball_ids = Vec::with_capacity(count);
    for i in 0..count {
        let id = world.push_body(sphere_body_r4(
            spawn_position(i, count),
            Vec4::ZERO,
            RADIUS_4D,
            1.0,
        ));
        world.bodies[id].restitution = 0.0;
        ball_ids.push(id);
    }
    (world, ball_ids)
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
    ball_ids: Vec<usize>,
    paused: bool,

    w_offset: f32,
    auto_sweep: bool,
    sweep_anchor: Instant,
    slider_up_held: bool,
    slider_down_held: bool,
    /// Ghost-mode flag: render the bodies' full 4D extent as a
    /// translucent volume rather than a single `w`-slice.
    ghost_mode: bool,

    frame_count: u32,
    last_fps_update: Instant,
    fps: f32,
}

impl App {
    fn new(count: usize) -> Self {
        let (world, ball_ids) = build_world(count);
        // Pull the camera out a bit further when there are many
        // bodies so the whole stack is in frame at the start.
        let cam_dist = (8.0 + 1.5 * count as f32).min(20.0);
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
                c.set_orbit(cam_dist, -0.35);
                c
            },
            input: InputState::default(),
            start: Instant::now(),
            world,
            ball_ids,
            paused: false,
            w_offset: 0.0,
            auto_sweep: false,
            sweep_anchor: Instant::now(),
            slider_up_held: false,
            slider_down_held: false,
            ghost_mode: false,
            frame_count: 0,
            last_fps_update: Instant::now(),
            fps: 0.0,
        }
    }

    fn reset(&mut self) {
        let count = self.ball_ids.len();
        let (world, ball_ids) = build_world(count);
        self.world = world;
        self.ball_ids = ball_ids;
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
            KeyCode::KeyG => self.ghost_mode = !self.ghost_mode,
            KeyCode::Space => self.paused = !self.paused,
            _ => {}
        }
    }

    /// Reference body for the slice plane. Always the first
    /// hypersphere; multi-body mode picks the *first* spawned body so
    /// the ↑/↓ slider has a stable anchor (other bodies have varied
    /// `w` offsets, see `spawn_position`).
    fn anchor_body_w(&self) -> f32 {
        self.world.bodies[self.ball_ids[0]].position.w
    }

    fn effective_w_slice(&self) -> f32 {
        self.anchor_body_w() + self.w_offset
    }

    fn current_uniforms(&self) -> Option<SliceUniforms> {
        let rd = self.rd.as_ref()?;
        let view = self.camera.view();
        let t = self.start.elapsed().as_secs_f32();

        let mut bodies = [[0.0_f32; 4]; MAX_BODIES];
        let count = self.ball_ids.len().min(MAX_BODIES);
        for (slot, &id) in bodies.iter_mut().zip(self.ball_ids.iter()).take(count) {
            *slot = self.world.bodies[id].position.to_array();
        }

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
            params: [
                self.effective_w_slice(),
                RADIUS_4D,
                count as f32,
                if self.ghost_mode { 1.0 } else { 0.0 },
            ],
            bodies,
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
                    let n = self.ball_ids.len();
                    let anchor = &self.world.bodies[self.ball_ids[0]];
                    let p = anchor.position;
                    let pause = if self.paused { " [paused]" } else { "" };
                    let view_mode = if self.ghost_mode { "ghost" } else { "slice" };
                    if self.ghost_mode {
                        win.set_title(&format!(
                            "{TITLE} | {:.0} fps | n={n} | mode={view_mode}{pause} | pos[0].y={:+.2} pos[0].w={:+.2}",
                            self.fps, p.y, p.w
                        ));
                    } else {
                        let mode = if self.auto_sweep { "auto" } else { "manual" };
                        let w_eff = p.w + self.w_offset;
                        // Visible-radius indicator for the *anchor*
                        // body in slice mode. Matches `slice_radius`
                        // in the WGSL.
                        let dw = w_eff - p.w;
                        let r3_sq = RADIUS_4D * RADIUS_4D - dw * dw;
                        let r3 = if r3_sq > 0.0 { r3_sq.sqrt() } else { 0.0 };
                        win.set_title(&format!(
                            "{TITLE} | {:.0} fps | n={n} | mode={view_mode} offset={:+.2} ({mode}) w₀={:+.2}{pause} | r₃[0]={:.3} | pos[0].y={:+.2} pos[0].w={:+.2}",
                            self.fps, self.w_offset, w_eff, r3, p.y, p.w
                        ));
                    }
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

/// Entry point for the legacy bespoke-shader path. Invoked by the
/// outer `main` when `--legacy` is on the CLI.
///
/// Has the full feature set: slice-mode + ghost-mode (volumetric
/// raymarch), per-body color cycling, on-pause / on-reset hot
/// keys. Doesn't use `rye-app` or `Hyperslice4DNode`; it's the
/// pre-framework hand-rolled implementation that proves the same
/// pipeline works without the framework abstraction.
pub fn run() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .try_init();
    let args = Args::parse();
    let elwt = winit::event_loop::EventLoop::new()?;
    elwt.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::new(args.count);
    elwt.run_app(&mut app)?;
    Ok(())
}
