//! Corridor raymarch demo — the README hero image.
//!
//! A rectangular hallway with pillars along both walls, rendered with
//! geodesic ray marching in E³, H³, or S³. Same scene, same shader,
//! different Space prelude. Because the march follows geodesics, the
//! walls read as flat in E³, bowing outward in H³ (parallel geodesics
//! diverge), and converging in S³.
//!
//! Injects:
//! - Space prelude from `rye-math` (`rye_distance`, `rye_exp`, …)
//! - Scene module from `rye-sdf` (`corridor_demo_wgsl`)
//! - User shader from `examples/corridor/corridor.wgsl`
//!
//! ## Flags
//!
//! `--hyperbolic`         — swap Space prelude to HyperbolicH3
//! `--spherical`          — swap Space prelude to SphericalS3
//! `--rotate`             — auto-rotate camera; interactive at 1 rev/20 s
//! `--capture-apng PATH`  — render N frames, save looping APNG, exit
//! `--capture-gif  PATH`  — render N frames, save looping GIF, exit
//! `--capture-frames N`   — frame count (default 300 = 10 s @ 30 fps)
//! `--capture-fps N`      — playback fps baked into APNG (default 30)

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use rye_asset::AssetWatcher;
use rye_math::{EuclideanR3, HyperbolicH3, SphericalS3, WgslSpace};
use rye_render::{
    device::RenderDevice,
    graph::RenderNode,
    raymarch::{GeodesicRayMarchNode, RayMarchUniforms},
};
use rye_sdf::corridor_demo_wgsl;
use rye_shader::{ShaderDb, ShaderId};
use rye_time::FixedTimestep;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes},
};

#[path = "../capture.rs"]
mod capture;

use capture::FrameCapture;
use rye_camera::OrbitCamera;
use rye_input::InputState;

fn shader_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/corridor")
}

fn shader_path() -> PathBuf {
    shader_dir().join("corridor.wgsl")
}

struct SceneParams {
    ball_scale: f32,
    fog_scale: f32,
}

impl SceneParams {
    fn as_array(&self) -> [f32; 4] {
        [0.0, self.ball_scale, self.fog_scale, 0.0]
    }
}

#[derive(Clone, Copy)]
struct ShaderKnobs {
    ball_scale: f32,
    fog_scale: f32,
    title: &'static str,
    capture_distance: f32,
    capture_pitch: f32,
}

// BALL_SCALE maps camera-space orbit units to Space coordinates.
// The corridor walls sit at Space x = ±0.55 and y = ±0.40. The camera
// must stay inside the Poincaré ball for H³/S³ (|p| < 1), so
// capture_distance × ball_scale must be < 1.0 and also less than the
// corridor half-widths. 1.5 × 0.2 = 0.30 — safely inside the hall.
const CORRIDOR_BALL_SCALE: f32 = 0.2;

const EUCLIDEAN_KNOBS: ShaderKnobs = ShaderKnobs {
    ball_scale: CORRIDOR_BALL_SCALE,
    fog_scale: 2.8,
    title: "Rye — Corridor (E³)",
    capture_distance: 1.5,
    capture_pitch: -0.12,
};

const HYPERBOLIC_KNOBS: ShaderKnobs = ShaderKnobs {
    ball_scale: CORRIDOR_BALL_SCALE,
    fog_scale: 2.2,
    title: "Rye — Corridor (H³ geodesics)",
    capture_distance: 1.5,
    capture_pitch: -0.12,
};

const SPHERICAL_KNOBS: ShaderKnobs = ShaderKnobs {
    ball_scale: CORRIDOR_BALL_SCALE,
    fog_scale: 2.4,
    title: "Rye — Corridor (S³ geodesics)",
    capture_distance: 1.5,
    capture_pitch: -0.12,
};

const ROTATE_YAW_INTERACTIVE: f32 = std::f32::consts::TAU / (60.0 * 20.0);

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
    ray_march: Option<GeodesicRayMarchNode>,

    timestep: FixedTimestep,
    camera: OrbitCamera,
    input: InputState,
    start: Instant,

    rotate: bool,
    rotate_yaw_per_frame: f32,
    capture: Option<FrameCapture>,
}

struct AppRunner<S: WgslSpace + 'static> {
    app: App<S>,
    capture_args: CaptureArgs,
}

impl<S: WgslSpace + 'static> AppRunner<S> {
    fn new(space: S, knobs: ShaderKnobs, rotate: bool, capture_args: CaptureArgs) -> Self {
        let rotate_yaw_per_frame = if capture_args.any_path().is_some() && capture_args.frames > 0 {
            std::f32::consts::TAU / capture_args.frames as f32
        } else {
            ROTATE_YAW_INTERACTIVE
        };
        // Seed the camera inside the corridor from the start, so
        // interactive mode doesn't begin looking at the scene from outside.
        let mut camera = OrbitCamera::default();
        camera.set_orbit(knobs.capture_distance, knobs.capture_pitch);
        let app = App {
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
            camera,
            input: InputState::default(),
            start: Instant::now(),
            rotate,
            rotate_yaw_per_frame,
            capture: None,
        };
        Self { app, capture_args }
    }

    fn handle_hot_reload(&mut self) {
        let app = &mut self.app;
        let (Some(watcher), Some(shaders), Some(id), Some(rd)) = (
            app.watcher.as_ref(),
            app.shaders.as_mut(),
            app.shader_id,
            app.rd.as_ref(),
        ) else {
            return;
        };
        let events = watcher.poll();
        if events.is_empty() {
            return;
        }
        shaders.apply_events(&events, &app.space);
        let new_gen = shaders.generation(id);
        if new_gen != app.shader_gen {
            tracing::info!("rebuilding GeodesicRayMarchNode for shader gen {new_gen}");
            app.shader_gen = new_gen;
            app.ray_march = Some(GeodesicRayMarchNode::from_module(
                &rd.device,
                rd.surface_bundle.config.format,
                shaders.module(id),
            ));
        }
    }

    fn current_uniforms(&self) -> Option<RayMarchUniforms> {
        let app = &self.app;
        let rd = app.rd.as_ref()?;
        let t = app.start.elapsed().as_secs_f32();
        let camera = app.camera.view();
        let config = &rd.surface_bundle.config;
        Some(RayMarchUniforms {
            camera_pos: camera.position.to_array(),
            _pad0: 0.0,
            camera_forward: camera.forward.to_array(),
            _pad1: 0.0,
            camera_right: camera.right.to_array(),
            _pad2: 0.0,
            camera_up: camera.up.to_array(),
            fov_y_tan: (70.0_f32.to_radians() * 0.5).tan(),
            resolution: [config.width as f32, config.height as f32],
            time: t,
            tick: app.timestep.tick() as f32,
            params: SceneParams {
                ball_scale: app.knobs.ball_scale,
                fog_scale: app.knobs.fog_scale,
            }
            .as_array(),
        })
    }
}

impl<S: WgslSpace + 'static> ApplicationHandler for AppRunner<S> {
    fn resumed(&mut self, elwt: &ActiveEventLoop) {
        let app = &mut self.app;
        let win = Arc::new(
            elwt.create_window(
                WindowAttributes::default()
                    .with_title(app.knobs.title)
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
            app.camera
                .set_orbit(app.knobs.capture_distance, app.knobs.capture_pitch);
        }

        let mut shaders = ShaderDb::new(rd.device.clone());
        let scene_module = corridor_demo_wgsl();
        let id = shaders
            .load_geodesic_scene(shader_path(), &scene_module, &app.space)
            .expect("load corridor.wgsl");
        let gen = shaders.generation(id);

        let mut watcher = AssetWatcher::new().expect("asset watcher");
        watcher.watch(shader_dir()).expect("watch shader dir");

        let ray_march = GeodesicRayMarchNode::from_module(
            &rd.device,
            rd.surface_bundle.config.format,
            shaders.module(id),
        );

        app.window = Some(win.clone());
        app.rd = Some(rd);
        app.shaders = Some(shaders);
        app.shader_id = Some(id);
        app.shader_gen = gen;
        app.watcher = Some(watcher);
        app.ray_march = Some(ray_march);
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
        let Some(win) = self.app.window.clone() else {
            return;
        };
        let app = &mut self.app;

        match ev {
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && matches!(event.logical_key, Key::Named(NamedKey::Escape)) =>
            {
                elwt.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                app.input.key_input(event.physical_key, event.state);
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
                if app.minimized {
                    return;
                }

                let ticks = app.timestep.advance(Instant::now());
                if !ticks.is_empty() {
                    if app.rotate {
                        app.camera.rotate_yaw(app.rotate_yaw_per_frame);
                    } else {
                        let input = app.input.take_frame();
                        app.camera.advance(input);
                    }
                }
                self.handle_hot_reload();

                let Some(uniforms) = self.current_uniforms() else {
                    return;
                };
                let Some(rd) = self.app.rd.as_ref() else {
                    return;
                };
                if let Some(node) = self.app.ray_march.as_mut() {
                    node.set_uniforms(&rd.queue, uniforms);
                }

                match rd.begin_frame() {
                    Ok((frame, view)) => {
                        if let Some(node) = self.app.ray_march.as_mut() {
                            if let Err(e) = node.execute(rd, &view) {
                                tracing::error!("render error: {e:#}");
                            }
                        }

                        if let Some(cap) = &mut self.app.capture {
                            if !cap.is_done() {
                                cap.capture(&rd.device, &rd.queue, &frame);
                                let n = cap.frame_count();
                                let total = self.capture_args.frames;
                                if n % 30 == 0 || n == total as usize {
                                    win.set_title(&format!(
                                        "{} — capturing {n}/{total}",
                                        self.app.knobs.title
                                    ));
                                }
                            }
                        }

                        frame.present();

                        let done = self.app.capture.as_ref().is_some_and(|c| c.is_done());
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
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let hyperbolic = args.iter().any(|a| a == "--hyperbolic");
    let spherical = args.iter().any(|a| a == "--spherical");
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

    if hyperbolic {
        let mut runner = AppRunner::new(HyperbolicH3, HYPERBOLIC_KNOBS, rotate, cap_args);
        event_loop.run_app(&mut runner)?;
    } else if spherical {
        let mut runner = AppRunner::new(SphericalS3, SPHERICAL_KNOBS, rotate, cap_args);
        event_loop.run_app(&mut runner)?;
    } else {
        let mut runner = AppRunner::new(EuclideanR3, EUCLIDEAN_KNOBS, rotate, cap_args);
        event_loop.run_app(&mut runner)?;
    }
    Ok(())
}
