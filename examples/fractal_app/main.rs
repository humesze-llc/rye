//! `fractal_app` — Mandelbulb raymarcher rebuilt on the `rye-app`
//! framework. Same shader and same gameplay as
//! [`examples/fractal`](../fractal/main.rs); the difference is that
//! the winit `ApplicationHandler` impl, hot-reload plumbing, FPS
//! bookkeeping, and surface-error recovery all live in `rye-app`
//! instead of being reimplemented here.
//!
//! ## Flags
//!
//! - `--hyperbolic`  — swap `WgslSpace` prelude to `HyperbolicH3`.
//! - `--spherical`   — swap `WgslSpace` prelude to `SphericalS3`.
//! - `--rotate`      — auto-rotate camera at 1 rev / 20 s.
//!
//! Capture-to-APNG/GIF lives in [`examples/fractal`](../fractal)
//! and is not yet plumbed into the `rye-app` framework. When a
//! game actually wants capture, it'll get its own framework hook.
//!
//! ## What this proves
//!
//! - `App` trait + framework runner work end to end on a real
//!   demo scene.
//! - `App::Space` (used for the shader prelude) and the camera's
//!   own Space are deliberately *decoupled*: this demo runs a
//!   Euclidean orbit camera around a Euclidean fractal scene
//!   while the shader prelude uses H³ / S³ only for the
//!   *geodesic-fog distance metric*. A PAINCARE-style game where
//!   the camera and physics actually live in H³ would use
//!   `Camera<HyperbolicH3>` + `OrbitController<HyperbolicH3>`
//!   end-to-end and orbit along hyperbolic geodesics.
//! - Hot-reload still works: edit `examples/fractal/fractal.wgsl`,
//!   the framework calls back into `on_shader_reload`, and the
//!   `RayMarchNode` rebuilds.
//!
//! ## Why a Euclidean camera here, not `Camera<HyperbolicH3>`
//!
//! `OrbitController<HyperbolicH3>` orbits along H³ geodesics:
//! `camera_pos = exp_target(back · distance)` in the Poincaré
//! ball. With the demo's default distance of 3.5 around the
//! origin, that lands the camera at `|p| ≈ 0.99` — right at the
//! ball's ideal boundary, where the metric explodes and rendering
//! flickers as numerical noise dominates. The legacy
//! `examples/fractal` demo always used a Euclidean orbit; this
//! migration preserves that, while leaving the *honest* H³
//! camera path available for games that actually want it.

use std::path::PathBuf;

use anyhow::Result;
use rye_app::{run_with_config, App, Camera, FrameCtx, OrbitController, RunConfig, SetupCtx};
use rye_math::{EuclideanR3, HyperbolicH3, SphericalS3, WgslSpace};
use rye_render::{
    device::RenderDevice,
    raymarch::{RayMarchNode, RayMarchUniforms},
};
use rye_shader::ShaderId;
use winit::window::WindowAttributes;

/// Shader files live in the original `examples/fractal/` directory
/// so the two examples share one source of truth.
fn shader_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/fractal")
}
fn shader_path() -> PathBuf {
    shader_dir().join("fractal.wgsl")
}

/// Yaw advance per tick at 60 Hz for `--rotate` mode (1 rev / 20 s).
const ROTATE_YAW_PER_TICK: f32 = std::f32::consts::TAU / (60.0 * 20.0);

// ---------------------------------------------------------------------------
// Per-Space knobs — same numbers as the legacy example.
// ---------------------------------------------------------------------------

/// Per-Space tuning constants for the *shader prelude*. The Space
/// only affects the WGSL prelude (geodesic fog metric); the camera
/// is always Euclidean. See the crate-level "Why a Euclidean camera"
/// note for rationale.
trait FractalKnobs: WgslSpace + Default + 'static {
    const BALL_SCALE: f32;
    const FOG_SCALE: f32;
    const TITLE: &'static str;
}

impl FractalKnobs for EuclideanR3 {
    const BALL_SCALE: f32 = 1.0;
    const FOG_SCALE: f32 = 12.0;
    const TITLE: &'static str = "Rye — Mandelbulb (rye-app)";
}
impl FractalKnobs for HyperbolicH3 {
    const BALL_SCALE: f32 = 0.2;
    const FOG_SCALE: f32 = 4.0;
    const TITLE: &'static str = "Rye — Mandelbulb H³ (rye-app)";
}
impl FractalKnobs for SphericalS3 {
    const BALL_SCALE: f32 = 0.15;
    const FOG_SCALE: f32 = 2.5;
    const TITLE: &'static str = "Rye — Mandelbulb S³ (rye-app)";
}

// ---------------------------------------------------------------------------
// FractalApp
// ---------------------------------------------------------------------------

struct FractalApp<S: FractalKnobs> {
    /// Drives the WGSL prelude (geodesic fog metric for H³ / S³;
    /// no-op for E³). Independent of the camera's own geometry.
    space: S,
    /// Camera lives in Euclidean 3-space regardless of `S`. The
    /// shader interprets `camera_pos` as a Poincaré-ball or
    /// 3-sphere point internally for the fog metric, but the
    /// orbit math itself is flat. See the crate-level note.
    camera: Camera<EuclideanR3>,
    orbit: OrbitController<EuclideanR3>,
    shader_id: ShaderId,
    shader_gen: u64,
    ray_march: RayMarchNode,
    rotate: bool,
}

impl<S: FractalKnobs> App for FractalApp<S> {
    type Space = S;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let space = S::default();

        let shader_id = ctx.shader_db.load(shader_path(), &space)?;
        let shader_gen = ctx.shader_db.generation(shader_id);
        let ray_march = RayMarchNode::new(
            &ctx.rd.device,
            ctx.rd.surface_bundle.config.format,
            ctx.shader_db.module(shader_id),
        );

        if let Some(watcher) = ctx.watcher.as_mut() {
            watcher.watch(shader_dir())?;
        }

        let rotate = std::env::args().any(|a| a == "--rotate");

        Ok(Self {
            space,
            camera: Camera::<EuclideanR3>::at_origin(),
            orbit: OrbitController::default(),
            shader_id,
            shader_gen,
            ray_march,
            rotate,
        })
    }

    fn space(&self) -> &S {
        &self.space
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        if self.rotate {
            self.orbit
                .rotate_yaw(ROTATE_YAW_PER_TICK * ctx.n_ticks as f32);
        }
        // Camera always orbits in Euclidean space, even when the
        // shader prelude is H³ / S³.
        self.orbit
            .advance_with_input(ctx.input, &mut self.camera, &EuclideanR3);

        let view = self.camera.view();
        let cfg = &ctx.rd.surface_bundle.config;
        self.ray_march.set_uniforms(
            &ctx.rd.queue,
            RayMarchUniforms {
                camera_pos: view.position.to_array(),
                _pad0: 0.0,
                camera_forward: view.forward.to_array(),
                _pad1: 0.0,
                camera_right: view.right.to_array(),
                _pad2: 0.0,
                camera_up: view.up.to_array(),
                fov_y_tan: (60.0_f32.to_radians() * 0.5).tan(),
                resolution: [cfg.width as f32, cfg.height as f32],
                time: ctx.time,
                tick: ctx.tick as f32,
                params: [0.0, S::BALL_SCALE, S::FOG_SCALE, 0.0],
            },
        );
    }

    fn on_shader_reload(&mut self, ctx: &mut SetupCtx<'_>) {
        let new_gen = ctx.shader_db.generation(self.shader_id);
        if new_gen != self.shader_gen {
            tracing::info!("rebuilding RayMarchNode for shader gen {new_gen}");
            self.shader_gen = new_gen;
            self.ray_march = RayMarchNode::new(
                &ctx.rd.device,
                ctx.rd.surface_bundle.config.format,
                ctx.shader_db.module(self.shader_id),
            );
        }
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        use rye_render::graph::RenderNode;
        self.ray_march.execute(rd, view)
    }

    fn title(&self, fps: f32) -> std::borrow::Cow<'static, str> {
        let p = self.camera.view().position;
        std::borrow::Cow::Owned(format!(
            "{} | {fps:.0} fps | pos ({:.2}, {:.2}, {:.2})",
            S::TITLE,
            p.x,
            p.y,
            p.z
        ))
    }
}

/// Thin wrapper so the `update` call site reads cleanly without
/// dragging in `CameraController` and a meaningless `dt`.
trait OrbitInputExt {
    fn advance_with_input(
        &mut self,
        input: rye_input::FrameInput,
        camera: &mut Camera<EuclideanR3>,
        space: &EuclideanR3,
    );
}

impl OrbitInputExt for OrbitController<EuclideanR3> {
    fn advance_with_input(
        &mut self,
        input: rye_input::FrameInput,
        camera: &mut Camera<EuclideanR3>,
        space: &EuclideanR3,
    ) {
        use rye_camera::CameraController;
        self.advance(input, camera, space, 0.0);
    }
}

// ---------------------------------------------------------------------------
// Entry point — pick the Space at runtime; the App impl is
// monomorphised per-Space at compile time.
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let hyperbolic = args.iter().any(|a| a == "--hyperbolic");
    let spherical = args.iter().any(|a| a == "--spherical");

    let config = RunConfig {
        window: WindowAttributes::default()
            .with_title("Rye — Mandelbulb (rye-app)")
            .with_visible(false),
        ..RunConfig::default()
    };

    if hyperbolic {
        run_with_config::<FractalApp<HyperbolicH3>>(config)
    } else if spherical {
        run_with_config::<FractalApp<SphericalS3>>(config)
    } else {
        run_with_config::<FractalApp<EuclideanR3>>(config)
    }
}
