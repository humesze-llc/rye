//! Pentatope `w`-slice viewer — sweep a 3-hyperplane through a 4D
//! pentatope (5-cell) and watch the 3D cross-section morph.
//!
//! ## Controls
//!
//! - Mouse: orbit camera (left-drag), zoom (scroll).
//! - **↑ / ↓**: nudge `w₀` (the slice level) by ±0.02.
//! - **A**: toggle automatic sweep (cosine-paced) of `w₀`.
//! - **R**: reset `w₀` to 0 and re-enable auto-sweep.
//! - **0–4**: highlight that pentatope cell (the cell index that
//!   contributes the visible face is colored brighter when its index
//!   matches the highlight).
//! - **5 / Space / Esc** combo: 5 or Space clears highlighting; Esc exits.
//!
//! ## What you're seeing
//!
//! The pentatope is the 4D analogue of the tetrahedron. It has 5
//! tetrahedral cells, each one the convex hull of 4 of its 5
//! vertices. Slicing at `w = w₀` produces a convex 3D polyhedron whose
//! faces correspond to the cells the hyperplane crosses. Each face is
//! tinted by which cell contributes it — orange / blue / amber /
//! green / magenta for cells 0..4.
//!
//! At critical `w₀` values (the unique `w` of any pentatope vertex)
//! the cross-section degenerates to a single tetrahedral cell. Sweep
//! through them and you can see all 5 cells appear in turn.
//!
//! Hot-reload the shader by editing `pentatope_slice.wgsl` while the
//! example runs.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use rye_asset::AssetWatcher;
use rye_camera::OrbitCamera;
use rye_input::InputState;
use rye_math::EuclideanR3;
use rye_render::{
    device::RenderDevice,
    graph::RenderNode,
    raymarch::{RayMarchNode, RayMarchUniforms},
};
use rye_shader::{ShaderDb, ShaderId};
use rye_time::FixedTimestep;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, KeyCode, NamedKey, PhysicalKey},
    window::{Window, WindowAttributes},
};

fn shader_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/pentatope_slice")
}

fn shader_path() -> PathBuf {
    shader_dir().join("pentatope_slice.wgsl")
}

const TITLE: &str = "Rye — pentatope w-slice";

/// `w` range the slice can travel through. The pentatope's vertices
/// live at `w = 1` (apex) and `w = −¼` (base 4 vertices), so the
/// nontrivial range is `[−¼, 1]`. We pad it slightly so the user can
/// see the cross-section vanish off either end.
const W_MIN: f32 = -0.35;
const W_MAX: f32 = 1.10;
const NO_HIGHLIGHT: f32 = 5.0;

struct App {
    window: Option<Arc<Window>>,
    rd: Option<RenderDevice>,
    minimized: bool,

    shaders: Option<ShaderDb>,
    shader_id: Option<ShaderId>,
    shader_gen: u64,
    watcher: Option<AssetWatcher>,
    ray_march: Option<RayMarchNode>,

    timestep: FixedTimestep,
    camera: OrbitCamera,
    input: InputState,
    start: Instant,

    // Slice state.
    w_slice: f32,
    auto_sweep: bool,
    highlight: f32,

    // FPS / title bookkeeping.
    frame_count: u32,
    last_fps_update: Instant,
    fps: f32,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            rd: None,
            minimized: false,
            shaders: None,
            shader_id: None,
            shader_gen: 0,
            watcher: None,
            ray_march: None,
            timestep: FixedTimestep::new(60),
            camera: {
                // Pull the camera back so the cross-section
                // (≤ √2 across at any w₀) fits in view comfortably.
                let mut c = OrbitCamera::default();
                c.set_orbit(3.0, 0.45);
                c
            },
            input: InputState::default(),
            start: Instant::now(),
            w_slice: 0.0,
            auto_sweep: true,
            highlight: NO_HIGHLIGHT,
            frame_count: 0,
            last_fps_update: Instant::now(),
            fps: 0.0,
        }
    }

    /// Update `w_slice` per tick when auto-sweep is on. Cosine pacing
    /// makes the slider linger at the extremes (where the
    /// cross-section degenerates to a single cell) so the user can
    /// see those configurations clearly.
    fn advance_auto_sweep(&mut self) {
        if !self.auto_sweep {
            return;
        }
        // 12-second period; eased so the motion slows at the ends.
        let phase = (self.start.elapsed().as_secs_f32() / 12.0) * std::f32::consts::TAU;
        let eased = 0.5 * (1.0 - phase.cos()); // 0..1
        self.w_slice = W_MIN + (W_MAX - W_MIN) * eased;
    }

    fn handle_keyboard(&mut self, code: PhysicalKey, state: ElementState) {
        if state != ElementState::Pressed {
            return;
        }
        let PhysicalKey::Code(kc) = code else {
            return;
        };
        match kc {
            KeyCode::ArrowUp => {
                self.auto_sweep = false;
                self.w_slice = (self.w_slice + 0.02).min(W_MAX);
            }
            KeyCode::ArrowDown => {
                self.auto_sweep = false;
                self.w_slice = (self.w_slice - 0.02).max(W_MIN);
            }
            KeyCode::KeyA => {
                self.auto_sweep = !self.auto_sweep;
                if self.auto_sweep {
                    // Reset start so the cosine phase is consistent.
                    self.start = Instant::now();
                }
            }
            KeyCode::KeyR => {
                self.w_slice = 0.0;
                self.auto_sweep = true;
                self.start = Instant::now();
            }
            KeyCode::Digit0 => self.highlight = 0.0,
            KeyCode::Digit1 => self.highlight = 1.0,
            KeyCode::Digit2 => self.highlight = 2.0,
            KeyCode::Digit3 => self.highlight = 3.0,
            KeyCode::Digit4 => self.highlight = 4.0,
            KeyCode::Digit5 | KeyCode::Space => self.highlight = NO_HIGHLIGHT,
            _ => {}
        }
    }

    fn current_uniforms(&self) -> Option<RayMarchUniforms> {
        let rd = self.rd.as_ref()?;
        let view = self.camera.view();
        let t = self.start.elapsed().as_secs_f32();
        Some(RayMarchUniforms {
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
            params: [self.w_slice, self.highlight, 0.0, 0.0],
        })
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
            .expect("load pentatope_slice.wgsl");
        let gen = shaders.generation(id);

        let mut watcher = AssetWatcher::new().expect("asset watcher");
        watcher.watch(shader_dir()).expect("watch shader dir");

        let ray_march = RayMarchNode::new(
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
        self.ray_march = Some(ray_march);
        self.minimized = false;
        self.start = Instant::now();

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
                if !ticks.is_empty() {
                    let input = self.input.take_frame();
                    self.camera.advance(input);
                    self.advance_auto_sweep();
                }

                self.handle_hot_reload();

                // FPS update + title.
                self.frame_count += 1;
                let elapsed = self.last_fps_update.elapsed().as_secs_f32();
                if elapsed >= 1.0 {
                    self.fps = self.frame_count as f32 / elapsed;
                    self.frame_count = 0;
                    self.last_fps_update = Instant::now();
                    let mode = if self.auto_sweep { "auto" } else { "manual" };
                    let hl = if self.highlight < 5.0 {
                        format!(", cell {}", self.highlight as u32)
                    } else {
                        String::new()
                    };
                    win.set_title(&format!(
                        "{TITLE} | {:.0} fps | w₀={:+.3} ({mode}{hl})",
                        self.fps, self.w_slice
                    ));
                }

                let Some(uniforms) = self.current_uniforms() else {
                    return;
                };
                let Some(rd) = self.rd.as_ref() else { return };
                if let Some(node) = self.ray_march.as_mut() {
                    node.set_uniforms(&rd.queue, uniforms);
                }

                match rd.begin_frame() {
                    Ok((frame, view)) => {
                        if let Some(node) = self.ray_march.as_mut() {
                            if let Err(e) = node.execute(rd, &view) {
                                tracing::error!("render error: {e:#}");
                            }
                        }
                        frame.present();
                        win.request_redraw();
                    }
                    Err(err) => match err {
                        wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated => {
                            if let Some(rd) = &mut self.rd {
                                let size = rd.surface_bundle.size;
                                rd.resize(size);
                            }
                            win.request_redraw();
                        }
                        wgpu::SurfaceError::Timeout => win.request_redraw(),
                        wgpu::SurfaceError::OutOfMemory => elwt.exit(),
                        wgpu::SurfaceError::Other => {
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

impl App {
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
            tracing::info!("rebuilding RayMarchNode for shader gen {new_gen}");
            self.shader_gen = new_gen;
            self.ray_march = Some(RayMarchNode::new(
                &rd.device,
                rd.surface_bundle.config.format,
                shaders.module(id),
            ));
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
    Ok(())
}
