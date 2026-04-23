//! Geodesic lattice — side-by-side E³ / H³ / S³ comparison demo.
//!
//! Renders three panels into a single window: left E³, centre H³, right S³.
//! The same camera orbits the same lattice of spheres; the only difference
//! is the Space prelude injected at shader compile time. The visual difference
//! (evenly-spaced grid vs. tanh-compressed vs. sin-wrapped) is the engine's
//! geometric thesis made visible.
//!
//! `lattice.wgsl` is compiled three times — once per Space — using
//! `WgslSpace::wgsl_impl()` assembled directly (no ShaderDb hot-reload,
//! since the three shaders share a path and ShaderDb is single-path-keyed).
//!
//! ## Flags
//!
//! `--rotate`               — auto-rotate camera
//! `--capture-apng PATH`    — render N frames, save looping APNG, exit
//! `--capture-gif  PATH`    — render N frames, save looping GIF, exit
//! `--capture-frames N`     — frame count (default 300)
//! `--capture-fps N`        — playback fps (default 30)

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use rye_math::{EuclideanR3, HyperbolicH3, SphericalS3, WgslSpace};
use rye_render::{
    device::RenderDevice,
    raymarch::{RayMarchNode, RayMarchUniforms},
};
use rye_sdf::LatticeSphereScene;
use rye_shader::validate_wgsl;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes},
};

#[path = "../fractal/camera.rs"]
mod camera;
#[path = "../capture.rs"]
mod capture;

use camera::{CameraState, InputState};
use capture::FrameCapture;

// BALL_SCALE is baked into lattice.wgsl as a constant; this comment documents
// the value so main.rs and the shader stay in sync. Camera orbit distance ×
// BALL_SCALE must stay < 1.0 for H³/S³ Poincaré ball validity (0.2 × 4.5 = 0.9).

// Fog scale per space (tuned for visual clarity).
const FOG_E3: f32 = 3.0;
const FOG_H3: f32 = 4.2;
const FOG_S3: f32 = 2.4;

// Capture camera parameters (distance * BALL_SCALE must be < 1.0 for H³/S³).
// 3.5 * 0.2 = 0.70, comfortably inside the Poincaré ball.
const CAPTURE_DISTANCE: f32 = 3.5;
const CAPTURE_PITCH: f32 = -0.35;

const ROTATE_YAW_INTERACTIVE: f32 = std::f32::consts::TAU / (60.0 * 20.0);

fn shader_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/lattice/lattice.wgsl")
}

struct CaptureArgs {
    apng_path: Option<PathBuf>,
    gif_path: Option<PathBuf>,
    frames: u32,
    fps: u32,
}

impl CaptureArgs {
    fn any_path(&self) -> Option<&PathBuf> {
        self.apng_path.as_ref().or(self.gif_path.as_ref())
    }
}

/// Assemble Space prelude + scene module + user shader into a single WGSL string.
fn assemble(prelude: &str, scene: &str, user: &str) -> String {
    format!(
        "// ---- rye-math Space prelude ----\n{prelude}\n\
         // ---- rye-sdf scene module ----\n{scene}\n\
         // ---- user shader ----\n{user}"
    )
}

struct App {
    window: Option<Arc<Window>>,
    rd: Option<RenderDevice>,
    minimized: bool,

    node_e3: Option<RayMarchNode>,
    node_h3: Option<RayMarchNode>,
    node_s3: Option<RayMarchNode>,

    timestep: rye_time::FixedTimestep,
    camera: CameraState,
    input: InputState,
    start: Instant,

    rotate: bool,
    rotate_yaw_per_frame: f32,
    capture: Option<FrameCapture>,
}

struct AppRunner {
    app: App,
    capture_args: CaptureArgs,
}

impl AppRunner {
    fn new(rotate: bool, capture_args: CaptureArgs) -> Self {
        let rotate_yaw_per_frame = if capture_args.any_path().is_some() && capture_args.frames > 0 {
            std::f32::consts::TAU / capture_args.frames as f32
        } else {
            ROTATE_YAW_INTERACTIVE
        };
        let app = App {
            window: None,
            rd: None,
            minimized: false,
            node_e3: None,
            node_h3: None,
            node_s3: None,
            timestep: rye_time::FixedTimestep::new(60),
            camera: CameraState::default(),
            input: InputState::default(),
            start: Instant::now(),
            rotate,
            rotate_yaw_per_frame,
            capture: None,
        };
        Self { app, capture_args }
    }

    fn panel_uniforms(
        &self,
        w: u32,
        h: u32,
        x_offset: u32,
        panel_w: u32,
        panel_idx: f32,
        fog_scale: f32,
    ) -> RayMarchUniforms {
        let t = self.app.start.elapsed().as_secs_f32();
        let camera = self.app.camera.view();
        RayMarchUniforms {
            camera_pos: camera.position.to_array(),
            _pad0: 0.0,
            camera_forward: camera.forward.to_array(),
            _pad1: 0.0,
            camera_right: camera.right.to_array(),
            _pad2: 0.0,
            camera_up: camera.up.to_array(),
            fov_y_tan: (60.0_f32.to_radians() * 0.5).tan(),
            resolution: [w as f32, h as f32],
            time: t,
            tick: self.app.timestep.tick() as f32,
            params: [x_offset as f32, panel_idx, panel_w as f32, fog_scale],
        }
    }
}

impl ApplicationHandler for AppRunner {
    fn resumed(&mut self, elwt: &ActiveEventLoop) {
        let app = &mut self.app;

        let win = Arc::new(
            elwt.create_window(
                WindowAttributes::default()
                    .with_title("Rye — Geodesic Lattice (E³ / H³ / S³)")
                    .with_visible(false),
            )
            .expect("create window"),
        );

        let rd = pollster::block_on(RenderDevice::new(win.clone())).expect("render device");

        if let Some(path) = self.capture_args.any_path() {
            let path = path.clone();
            let w = rd.surface_bundle.config.width;
            let h = rd.surface_bundle.config.height;
            let cap = FrameCapture::new(self.capture_args.frames, self.capture_args.fps, w, h);
            tracing::info!(
                "capture mode: {} frames → {:?}  ({}×{} @ {} fps)",
                self.capture_args.frames,
                path,
                w,
                h,
                self.capture_args.fps,
            );
            app.capture = Some(cap);
            app.camera.set_orbit(CAPTURE_DISTANCE, CAPTURE_PITCH);
        }

        // Build lattice scene modules — centers are computed per-Space in Rust.
        let lattice = LatticeSphereScene::default();
        let scene_e3 = lattice.to_wgsl(&EuclideanR3);
        let scene_h3 = lattice.to_wgsl(&HyperbolicH3);
        let scene_s3 = lattice.to_wgsl(&SphericalS3);

        // Read the user WGSL once.
        let user_src = std::fs::read_to_string(shader_path()).expect("read lattice.wgsl");

        // Assemble and compile three distinct shader modules.
        let make_module = |prelude: &str, scene: &str| {
            let full = assemble(prelude, scene, &user_src);
            validate_wgsl(&full).expect("lattice shader should validate");
            rd.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("lattice"),
                source: wgpu::ShaderSource::Wgsl(full.into()),
            })
        };

        let fmt = rd.surface_bundle.config.format;
        let mod_e3 = make_module(&EuclideanR3.wgsl_impl(), &scene_e3);
        let mod_h3 = make_module(&HyperbolicH3.wgsl_impl(), &scene_h3);
        let mod_s3 = make_module(&SphericalS3.wgsl_impl(), &scene_s3);

        app.node_e3 = Some(RayMarchNode::new(&rd.device, fmt, &mod_e3));
        app.node_h3 = Some(RayMarchNode::new(&rd.device, fmt, &mod_h3));
        app.node_s3 = Some(RayMarchNode::new(&rd.device, fmt, &mod_s3));

        app.window = Some(win.clone());
        app.rd = Some(rd);
        app.minimized = false;
        app.start = Instant::now();

        win.set_visible(true);
        win.request_redraw();
    }

    fn window_event(
        &mut self,
        elwt: &ActiveEventLoop,
        _id: winit::window::WindowId,
        ev: WindowEvent,
    ) {
        let Some(win) = self.app.window.clone() else { return };
        let app = &mut self.app;

        match ev {
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && matches!(event.logical_key, Key::Named(NamedKey::Escape)) =>
            {
                elwt.exit();
            }

            WindowEvent::CursorMoved { position, .. } => {
                app.input.cursor_moved(position.x, position.y);
            }
            WindowEvent::CursorLeft { .. } => app.input.cursor_invalidated(),
            WindowEvent::Focused(false) => {
                app.input.cursor_invalidated();
                app.input.release_buttons();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                app.input.mouse_input(button, state);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                app.input.mouse_wheel(delta);
            }

            WindowEvent::Resized(size) => {
                app.minimized = size.width == 0 || size.height == 0;
                if !app.minimized {
                    if let Some(rd) = &mut app.rd {
                        rd.resize(size);
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                if app.minimized { return; }

                let ticks = app.timestep.advance(Instant::now());
                if !ticks.is_empty() {
                    if app.rotate {
                        app.camera.rotate_yaw(app.rotate_yaw_per_frame);
                    } else {
                        let input = app.input.take_frame();
                        app.camera.advance(input);
                    }
                }

                let Some(rd) = app.rd.as_ref() else { return };
                let w = rd.surface_bundle.config.width;
                let h = rd.surface_bundle.config.height;

                // Divide width into three panels (rounding the last to fill any remainder).
                let pw = w / 3;
                let pw2 = w - pw * 2;

                // Upload uniforms for all three panels.
                {
                    let u_e3 = self.panel_uniforms(w, h, 0,        pw,  0.0, FOG_E3);
                    let u_h3 = self.panel_uniforms(w, h, pw,       pw,  1.0, FOG_H3);
                    let u_s3 = self.panel_uniforms(w, h, pw * 2,   pw2, 2.0, FOG_S3);
                    let rd = self.app.rd.as_ref().unwrap();
                    if let Some(n) = &mut self.app.node_e3 { n.set_uniforms(&rd.queue, u_e3); }
                    if let Some(n) = &mut self.app.node_h3 { n.set_uniforms(&rd.queue, u_h3); }
                    if let Some(n) = &mut self.app.node_s3 { n.set_uniforms(&rd.queue, u_s3); }
                }

                let rd = self.app.rd.as_ref().unwrap();
                match rd.begin_frame() {
                    Ok((frame, view)) => {
                        // E³: clear + draw left panel
                        if let Some(n) = &mut self.app.node_e3 {
                            if let Err(e) = n.execute_panel(rd, &view, true, [0, 0, pw, h]) {
                                tracing::error!("E³ render error: {e:#}");
                            }
                        }
                        // H³: load + draw centre panel
                        if let Some(n) = &mut self.app.node_h3 {
                            if let Err(e) = n.execute_panel(rd, &view, false, [pw, 0, pw, h]) {
                                tracing::error!("H³ render error: {e:#}");
                            }
                        }
                        // S³: load + draw right panel
                        if let Some(n) = &mut self.app.node_s3 {
                            if let Err(e) = n.execute_panel(rd, &view, false, [pw * 2, 0, pw2, h]) {
                                tracing::error!("S³ render error: {e:#}");
                            }
                        }

                        if let Some(cap) = &mut self.app.capture {
                            if !cap.is_done() {
                                cap.capture(&rd.device, &rd.queue, &frame);
                                let n = cap.frame_count();
                                let total = self.capture_args.frames;
                                if n % 30 == 0 || n == total as usize {
                                    win.set_title(&format!(
                                        "Rye — Geodesic Lattice — capturing {n}/{total}"
                                    ));
                                }
                            }
                        }

                        frame.present();

                        let done = self.app.capture.as_ref().map_or(false, |c| c.is_done());
                        if done {
                            if let Some(cap) = &self.app.capture {
                                if let Some(path) = &self.capture_args.apng_path {
                                    cap.save_apng(path).expect("save apng");
                                } else if let Some(path) = &self.capture_args.gif_path {
                                    cap.save_gif(path).expect("save gif");
                                }
                            }
                            elwt.exit();
                            return;
                        }

                        win.request_redraw();
                    }
                    Err(err) => match err {
                        wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated => {
                            if let Some(rd) = &mut self.app.rd {
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

fn parse_flag_value<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let rotate = args.iter().any(|a| a == "--rotate");
    let capture_apng: Option<PathBuf> = args
        .windows(2)
        .find(|w| w[0] == "--capture-apng")
        .map(|w| PathBuf::from(&w[1]));
    let capture_gif: Option<PathBuf> = args
        .windows(2)
        .find(|w| w[0] == "--capture-gif")
        .map(|w| PathBuf::from(&w[1]));
    let capture_frames: u32 = parse_flag_value(&args, "--capture-frames", 300);
    let capture_fps: u32 = parse_flag_value(&args, "--capture-fps", 30);

    let rotate = rotate || capture_apng.is_some() || capture_gif.is_some();

    let cap_args = CaptureArgs {
        apng_path: capture_apng,
        gif_path: capture_gif,
        frames: capture_frames,
        fps: capture_fps,
    };

    let event_loop: EventLoop<()> = EventLoop::new()?;
    let mut runner = AppRunner::new(rotate, cap_args);
    event_loop.run_app(&mut runner)?;
    Ok(())
}
