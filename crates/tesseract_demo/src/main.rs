//! One render pipeline (the line rasterizer, no SDF, triangle, point or depth
//! passes) and none of the rotor-composition UI, for a smaller wasm bundle.

use anyhow::Result;
use glam::{Mat4, Vec2, Vec3};
use loam_app::{
    egui, freecam::Freecam, App, Camera, CameraController, FrameCtx, OrbitController, RenderCtx,
    RunConfig, SetupCtx,
};

// Per-frame allocation telemetry surfaced in `frame_trace` + PerfOverlay;
// ~5-10ns per allocation, negligible next to the interop cost being chased.
#[global_allocator]
static GLOBAL: loam_time::alloc::CountingAllocator<std::alloc::System> =
    loam_time::alloc::CountingAllocator::new(std::alloc::System);
use loam_egui::{Console, ConsoleUi};
use loam_math::{Bivector, Bivector4, EuclideanR3, Rotor4};
use loam_render::device::RenderDevice;
use loam_render::{DepthMode, LineRasterStaticR4Node, Viewport};
use loam_shape::polytope::Polytope4;
use loam_shape::LineMesh;
use winit::window::WindowAttributes;

// Vertices live at `w = ±0.5` (unit circumradius); `2.0` keeps the viewer
// outside the polytope so `w = +0.5` projects larger than `w = -0.5`, the
// cube-within-cube look.
const FOCAL_DISTANCE: f32 = 2.0;

// xw rotation sweeps vertex w-coordinates, flipping the projection's
// inner-vs-outer assignment each half turn; that swap is the visible signature
// of 4D rotation, where an xy spin would read as plain 3D rotation.
const SPIN_RATE: f32 = 0.4;

const POLYTOPE_SCALE: f32 = 1.5;

const EDGE_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 0.95];
const EDGE_WIDTH_PX: f32 = 1.6;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CameraMode {
    Orbit,
    FreeRoam,
}

struct TesseractApp {
    /// `DepthMode::Off`: nothing else writes depth. The mesh stays on the GPU
    /// between frames, so per-frame work is one 144-byte uniform write and no
    /// instance-buffer upload.
    lines: LineRasterStaticR4Node,
    /// The 4D rotation is on the geometry side (the rotor), not the camera,
    /// which is plain `EuclideanR3`.
    camera: Camera<EuclideanR3>,
    orbit: OrbitController<EuclideanR3>,
    freecam: Freecam,
    mode: CameraMode,
    rotor: Rotor4,
    omega: Bivector4,
    paused: bool,
    console: Console<()>,
    perf: loam_app::trace::PerfOverlay,
}

impl TesseractApp {
    // One dummy `record` into a 1x1 throwaway texture; compilation is
    // size-independent. Lives in the demo, not the runner, because only the
    // demo knows which pipelines and configs it touches.
    fn warm_pipelines(&mut self, rd: &RenderDevice) {
        let dummy_tex = rd.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tesseract_demo::warmup"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Must match the pipeline's target format, so the warmup pass is
            // format-compatible.
            format: rd.target_format(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let dummy_view = dummy_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = rd
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tesseract_demo::warmup-encoder"),
            });
        {
            let mut ctx = RenderCtx {
                rd,
                view: &dummy_view,
                encoder: &mut encoder,
            };
            let _ = self.record(&mut ctx);
        }
        rd.queue.submit(Some(encoder.finish()));

        tracing::info!("tesseract_demo: warmed render pipelines");
    }
}

impl App for TesseractApp {
    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let topo = Polytope4::Tesseract.topology();

        let args = loam_app::args::Args::current();
        let spin_rate = args.parse::<f32>("spin_rate").unwrap_or(SPIN_RATE);
        let paused = args
            .get("paused")
            .map(|v| matches!(v, "true" | "1" | "yes"))
            .unwrap_or(false);

        let mut lines = LineRasterStaticR4Node::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            DepthMode::Off,
            ctx.rd.sample_count(),
        );

        let mut canonical = LineMesh::<4>::default();
        canonical.segments.reserve(topo.edges.len());
        canonical.colors.reserve(topo.edges.len());
        canonical.widths.reserve(topo.edges.len());
        for &[i, j] in topo.edges {
            let a = topo.vertices[i as usize] * POLYTOPE_SCALE;
            let b = topo.vertices[j as usize] * POLYTOPE_SCALE;
            canonical.segments.push((a.to_array(), b.to_array()));
            canonical.colors.push((EDGE_COLOR, EDGE_COLOR));
            canonical.widths.push(EDGE_WIDTH_PX);
        }
        lines.upload_mesh(&ctx.rd.device, &ctx.rd.queue, &canonical);

        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 1.0, 5.0);
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        orbit.set_orbit(5.0, -0.15);

        let freecam = Freecam::new().with_speed(2.5);

        let mut console = Console::<()>::new();
        loam_app::trace::register_command(&mut console);
        loam_app::fps::register_command(&mut console);
        loam_app::vsync::register_command(&mut console);
        loam_app::version::register_command(
            &mut console,
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
            env!("BUILD_HASH"),
            env!("BUILD_DIRTY"),
        );
        loam_app::log::register_command(&mut console);
        let perf = loam_app::trace::PerfOverlay::new();

        let mut app = Self {
            lines,
            camera,
            orbit,
            freecam,
            mode: CameraMode::Orbit,
            rotor: Rotor4::IDENTITY,
            // basis(2) is the xw plane (Plane4 order: 0=xy,1=xz,2=xw,3=yz,
            // 4=yw,5=zw); the plane whose spin sweeps w through the projection.
            omega: Bivector4::basis(2) * spin_rate,
            paused,
            console,
            perf,
        };

        // Materialize the App's PSOs during setup instead of stalling
        // ~100-500ms on first real draw. egui and composite pipelines are not
        // warmed; see `warm_pipelines`.
        app.warm_pipelines(ctx.rd);

        Ok(app)
    }

    fn apply_command(
        &mut self,
        cmd: &loam_app::command::CommandLine,
        _ctx: &mut loam_app::command::CommandCtx<'_>,
    ) -> Result<()> {
        self.console.dispatch(&cmd.name, &cmd.arg_refs(), &mut ());
        Ok(())
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        // Wall-clock dt, clamped so a multi-second stall does not catapult the
        // rotor through a half-revolution on catch-up.
        let dt = ctx.dt.min(0.1);

        if !self.paused {
            let step = (self.omega * dt).exp();
            self.rotor = (step * self.rotor).normalize();
        }

        match self.mode {
            CameraMode::Orbit => {
                self.orbit
                    .advance(ctx.input, &mut self.camera, &EuclideanR3, dt);
            }
            CameraMode::FreeRoam => {
                self.freecam.advance(ctx.input, &mut self.camera, dt);
            }
        }
    }

    fn on_key(
        &mut self,
        code: winit::keyboard::KeyCode,
        state: winit::event::ElementState,
        ctx: &mut FrameCtx<'_>,
    ) {
        use winit::event::ElementState;
        use winit::keyboard::KeyCode;
        // Do not fire app hotkeys while an egui widget has keyboard focus, or
        // typing `trace` would also toggle pause via `KeyT`.
        if ctx.ui_capture.keyboard {
            return;
        }
        // Alt needs both edges (the freecam preset interprets press + release
        // per its cursor_mode); everything else acts on press only.
        if matches!(code, KeyCode::AltLeft | KeyCode::AltRight)
            && matches!(self.mode, CameraMode::FreeRoam)
        {
            self.freecam.on_alt(matches!(state, ElementState::Pressed));
            return;
        }
        if !matches!(state, ElementState::Pressed) {
            return;
        }
        match code {
            KeyCode::KeyF => {
                self.mode = match self.mode {
                    CameraMode::Orbit => {
                        self.freecam.set_active(true, self.camera.position);
                        CameraMode::FreeRoam
                    }
                    CameraMode::FreeRoam => {
                        self.freecam.set_active(false, self.camera.position);
                        CameraMode::Orbit
                    }
                };
            }
            KeyCode::KeyT => {
                self.paused = !self.paused;
            }
            KeyCode::Space if !matches!(self.mode, CameraMode::FreeRoam) => {
                self.paused = !self.paused;
            }
            KeyCode::KeyR => {
                self.rotor = Rotor4::IDENTITY;
            }
            _ => {}
        }
    }

    fn record(&mut self, ctx: &mut RenderCtx<'_>) -> Result<()> {
        let cfg = &ctx.rd.surface_bundle.config;
        let aspect = cfg.width as f32 / cfg.height.max(1) as f32;
        let proj = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.05, 100.0);
        let view_dir = self.camera.view();
        let view_m = Mat4::look_to_rh(view_dir.position, view_dir.forward, view_dir.up);
        self.lines.set_transform(
            &ctx.rd.queue,
            self.rotor,
            proj * view_m,
            Vec2::new(cfg.width as f32, cfg.height as f32),
            FOCAL_DISTANCE,
        );

        {
            let _clear = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tesseract_demo::clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ctx.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.027,
                            g: 0.027,
                            b: 0.035,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }

        // `LoadOp::Load` preserves the clear; the runner submits the encoder
        // once at end of frame.
        let viewport = Viewport::full([cfg.width, cfg.height]);
        self.lines
            .record(ctx.encoder, ctx.view, None, Some(&viewport));
        Ok(())
    }

    fn ui(&mut self, ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {
        egui::Area::new(egui::Id::new("tesseract-hud"))
            .anchor(egui::Align2::LEFT_TOP, [12.0, 12.0])
            .show(ctx, |ui| {
                let mode = match self.mode {
                    CameraMode::Orbit => "orbit",
                    CameraMode::FreeRoam => "free-roam (WASD + mouse drag, space/shift = up/down)",
                };
                ui.colored_label(
                    egui::Color32::from_rgb(216, 216, 223),
                    format!(
                        "Tesseract \u{00b7} {mode} \u{00b7} F: toggle camera \u{00b7} \
                         T: pause \u{00b7} R: reset \u{00b7} \u{0060}: console \u{00b7} \
                         F3: perf"
                    ),
                );
                if self.paused {
                    ui.colored_label(egui::Color32::from_rgb(220, 180, 90), "[paused]");
                }
            });
        egui::Area::new(egui::Id::new("tesseract-build-id"))
            .anchor(egui::Align2::RIGHT_BOTTOM, [-12.0, -12.0])
            .show(ctx, |ui| {
                ui.colored_label(
                    egui::Color32::from_rgb(120, 120, 130),
                    format!("build {}{}", env!("BUILD_HASH"), env!("BUILD_DIRTY"),),
                );
            });
        self.perf.show(ctx);
        loam_app::log::pump_into(&mut self.console);
        loam_app::command::pump_into(&mut self.console);
        self.console.ui(ctx);
        loam_app::command::forward_pending(&mut self.console);
    }

    fn title(&self, fps: f32) -> std::borrow::Cow<'static, str> {
        format!("tesseract_demo  -  {fps:.0} fps").into()
    }
}

fn main() -> Result<()> {
    loam_app::run::<TesseractApp>(RunConfig {
        window: WindowAttributes::default()
            .with_title("tesseract demo")
            .with_visible(false),
        ..RunConfig::default()
    })
}
