//! Rye's first graphics example: a live Mandelbulb raymarcher.
//!
//! Demonstrates the Phase 1 stack end-to-end:
//! - rye-math's `WgslSpace` → shader prelude (`rye_distance` in WGSL)
//! - rye-asset's file watcher → shader hot reload
//! - rye-shader's ShaderDb → compiled modules, keyed by path
//! - rye-render's RayMarchNode → fullscreen triangle + UBO
//! - rye-time's FixedTimestep → deterministic tick counter
//!
//! Edit `examples/fractal/fractal.wgsl` while the example runs and the
//! scene recompiles on save.
//!
//! ## Modes
//!
//! Pass `--hyperbolic` to swap the WGSL prelude from `EuclideanR3` to
//! `HyperbolicH3`. The Mandelbulb SDF stays in scene coordinates, but
//! ray stepping follows Space geodesics via `rye_exp` and
//! `rye_parallel_transport`, and fog uses `rye_distance`. Distant
//! features dim more aggressively because hyperbolic distances grow
//! faster than Euclidean ones far from the origin.
//!
//! Pass `--spherical` to use `SphericalS3`. Points are interpreted as
//! upper-hemisphere coordinates (`|p|² < 1`), so the scene is rescaled
//! to keep the fractal inside the valid domain. Geodesic ray stepping
//! and fog both run in S³, so trajectories and attenuation can diverge
//! from Euclidean and H³ output as rays approach the equator
//! (`|p|² → 1`).
//!
//! Default (no flag) is Euclidean and produces byte-identical output to
//! prior versions of this example.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use rye_asset::AssetWatcher;
use rye_math::{EuclideanR3, HyperbolicH3, SphericalS3, WgslSpace};
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
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes},
};

mod camera;

use camera::{CameraState, InputState};

fn shader_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/fractal")
}

fn shader_path() -> PathBuf {
    shader_dir().join("fractal.wgsl")
}

/// Typed layout for `RayMarchUniforms::params` in the fractal shader.
///
/// Must match the `power_offset / ball_scale / fog_scale / params_pad`
/// field order in `fractal.wgsl`'s `Uniforms` struct.
struct FractalParams {
    power_offset: f32,
    ball_scale: f32,
    fog_scale: f32,
}

impl FractalParams {
    fn as_array(&self) -> [f32; 4] {
        [self.power_offset, self.ball_scale, self.fog_scale, 0.0]
    }
}

/// Per-mode shader knobs the host pushes into the uniform buffer.
///
/// `ball_scale` maps Euclidean scene coords into the unit Poincaré ball
/// before the H³ prelude consumes them; in Euclidean mode it stays at
/// 1.0. `fog_scale` is the distance at which fog goes opaque; smaller
/// for hyperbolic because geodesic distances grow faster.
#[derive(Clone, Copy)]
struct ShaderKnobs {
    ball_scale: f32,
    fog_scale: f32,
    title: &'static str,
}

const EUCLIDEAN_KNOBS: ShaderKnobs = ShaderKnobs {
    ball_scale: 1.0,
    fog_scale: 12.0,
    title: "Rye — Mandelbulb",
};

const HYPERBOLIC_KNOBS: ShaderKnobs = ShaderKnobs {
    ball_scale: 0.2,
    fog_scale: 4.0,
    title: "Rye — Mandelbulb (H³ fog)",
};

// S³ domain is |p|² < 1 (upper hemisphere). ball_scale compresses the
// fractal into roughly |p| < 0.6 so geodesic wrapping is visible but
// points stay comfortably inside the valid region.
const SPHERICAL_KNOBS: ShaderKnobs = ShaderKnobs {
    ball_scale: 0.15,
    fog_scale: 2.5,
    title: "Rye — Mandelbulb (S³ fog)",
};

struct App<S: WgslSpace + 'static> {
    window: Option<Arc<Window>>,
    rd: Option<RenderDevice>,
    minimized: bool,

    space: S,
    knobs: ShaderKnobs,
    shaders: Option<ShaderDb>,
    shader_id: Option<ShaderId>,
    shader_gen: u64,
    watcher: Option<AssetWatcher>,
    ray_march: Option<RayMarchNode>,

    timestep: FixedTimestep,
    camera: CameraState,
    input: InputState,
    start: Instant,
}

impl<S: WgslSpace + 'static> App<S> {
    fn new(space: S, knobs: ShaderKnobs) -> Self {
        Self {
            window: None,
            rd: None,
            minimized: false,
            space,
            knobs,
            shaders: None,
            shader_id: None,
            shader_gen: 0,
            watcher: None,
            ray_march: None,
            timestep: FixedTimestep::new(60),
            camera: CameraState::default(),
            input: InputState::default(),
            start: Instant::now(),
        }
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
        shaders.apply_events(&events, &self.space);
        let new_gen = shaders.generation(id);
        if new_gen != self.shader_gen {
            tracing::info!("rebuilding RayMarchNode for shader generation {}", new_gen);
            self.shader_gen = new_gen;
            self.ray_march = Some(RayMarchNode::new(
                &rd.device,
                rd.surface_bundle.config.format,
                shaders.module(id),
            ));
        }
    }

    fn current_uniforms(&self) -> Option<RayMarchUniforms> {
        let rd = self.rd.as_ref()?;
        let t = self.start.elapsed().as_secs_f32();
        let camera = self.camera.view();
        let config = &rd.surface_bundle.config;
        Some(RayMarchUniforms {
            camera_pos: camera.position.to_array(),
            _pad0: 0.0,
            camera_forward: camera.forward.to_array(),
            _pad1: 0.0,
            camera_right: camera.right.to_array(),
            _pad2: 0.0,
            camera_up: camera.up.to_array(),
            fov_y_tan: (60.0_f32.to_radians() * 0.5).tan(),
            resolution: [config.width as f32, config.height as f32],
            time: t,
            tick: self.timestep.tick() as f32,
            params: FractalParams {
                power_offset: 0.0,
                ball_scale: self.knobs.ball_scale,
                fog_scale: self.knobs.fog_scale,
            }
            .as_array(),
        })
    }
}

impl<S: WgslSpace + 'static> ApplicationHandler for App<S> {
    fn resumed(&mut self, elwt: &ActiveEventLoop) {
        let win = Arc::new(
            elwt.create_window(
                WindowAttributes::default()
                    .with_title(self.knobs.title)
                    .with_visible(false),
            )
            .expect("create window"),
        );

        let rd = pollster::block_on(RenderDevice::new(win.clone())).expect("render device");

        let mut shaders = ShaderDb::new(rd.device.clone());
        let id = shaders
            .load(shader_path(), &self.space)
            .expect("load fractal.wgsl");
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

            WindowEvent::CursorMoved { position, .. } => {
                self.input.cursor_moved(position.x, position.y);
            }

            WindowEvent::CursorLeft { .. } => {
                self.input.cursor_invalidated();
            }

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
                }
                self.handle_hot_reload();

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
                        wgpu::SurfaceError::Timeout => {
                            win.request_redraw();
                        }
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

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let hyperbolic = args.iter().any(|a| a == "--hyperbolic");
    let spherical = args.iter().any(|a| a == "--spherical");
    let event_loop: EventLoop<()> = EventLoop::new()?;
    if hyperbolic {
        let mut app = App::new(HyperbolicH3, HYPERBOLIC_KNOBS);
        event_loop.run_app(&mut app)?;
    } else if spherical {
        let mut app = App::new(SphericalS3, SPHERICAL_KNOBS);
        event_loop.run_app(&mut app)?;
    } else {
        let mut app = App::new(EuclideanR3, EUCLIDEAN_KNOBS);
        event_loop.run_app(&mut app)?;
    }
    Ok(())
}
