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

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use glam::Vec3;

use rye_asset::AssetWatcher;
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
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes},
};

fn shader_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/fractal")
}

fn shader_path() -> PathBuf {
    shader_dir().join("fractal.wgsl")
}

struct App {
    window: Option<Arc<Window>>,
    rd: Option<RenderDevice>,
    minimized: bool,

    space: EuclideanR3,
    shaders: Option<ShaderDb>,
    shader_id: Option<ShaderId>,
    shader_gen: u64,
    watcher: Option<AssetWatcher>,
    ray_march: Option<RayMarchNode>,

    timestep: FixedTimestep,
    start: Instant,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            rd: None,
            minimized: false,
            space: EuclideanR3,
            shaders: None,
            shader_id: None,
            shader_gen: 0,
            watcher: None,
            ray_march: None,
            timestep: FixedTimestep::new(60),
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
            tracing::info!(
                "rebuilding RayMarchNode for shader generation {}",
                new_gen
            );
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
        let angle = t * 0.15;
        let radius = 3.5;
        let pos = Vec3::new(
            angle.cos() * radius,
            0.6 + 0.2 * (t * 0.3).sin(),
            angle.sin() * radius,
        );
        let forward = (Vec3::ZERO - pos).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward);
        let config = &rd.surface_bundle.config;
        Some(RayMarchUniforms {
            camera_pos: pos.to_array(),
            _pad0: 0.0,
            camera_forward: forward.to_array(),
            _pad1: 0.0,
            camera_right: right.to_array(),
            _pad2: 0.0,
            camera_up: up.to_array(),
            fov_y_tan: (60.0_f32.to_radians() * 0.5).tan(),
            resolution: [config.width as f32, config.height as f32],
            time: t,
            tick: self.timestep.tick() as f32,
            params: [0.0, 0.0, 0.0, 0.0],
        })
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, elwt: &ActiveEventLoop) {
        let win = Arc::new(
            elwt.create_window(
                WindowAttributes::default()
                    .with_title("Rye — Mandelbulb")
                    .with_visible(false),
            )
            .expect("create window"),
        );

        let rd = pollster::block_on(RenderDevice::new(win.clone()))
            .expect("render device");

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
        let Some(win) = self.window.clone() else { return };

        match ev {
            WindowEvent::CloseRequested => elwt.exit(),

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

                let _ = self.timestep.advance(Instant::now());
                self.handle_hot_reload();

                let Some(uniforms) = self.current_uniforms() else { return };
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
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let event_loop: EventLoop<()> = EventLoop::new()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
