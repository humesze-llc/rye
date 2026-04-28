//! Geodesic spheres raymarch demo.
//!
//! Injects a scene module from `rye-sdf`:
//! - Space prelude from `rye-math` (`rye_distance`, `rye_exp`, ...)
//! - scene module from `rye-sdf` (`rye_scene_sdf`)
//! - user shader from `examples/geodesic_spheres/spheres.wgsl`
//!
//! ## Flags
//!
//! `--hyperbolic`        : swap Space prelude to HyperbolicH3
//! `--spherical`         : swap Space prelude to SphericalS3
//! `--rotate`            : auto-rotate camera; interactive at 1 rev/20 s
//! `--capture-apng PATH` : render N frames, save looping APNG, exit
//! `--capture-gif  PATH` : render N frames, save looping GIF, exit
//! `--capture-frames N`  : frame count (default 300 = 10 s @ 30 fps)
//! `--capture-fps N`     : playback fps baked into the encoded output
//!
//! Capture mode forces `--rotate` so the captured loop is a
//! complete revolution synced to the frame count.

use std::borrow::Cow;
use std::path::PathBuf;

use anyhow::Result;
use rye_app::{run_with_config, App, CaptureConfig, FrameCtx, RunConfig, SetupCtx};
use rye_camera::OrbitCamera;
use rye_math::{EuclideanR3, HyperbolicH3, SphericalS3, WgslSpace};
use rye_render::{
    device::RenderDevice,
    graph::RenderNode,
    raymarch::{GeodesicRayMarchNode, RayMarchUniforms},
};
use rye_sdf::geodesic_spheres_demo_wgsl;
use rye_shader::ShaderId;
use winit::window::WindowAttributes;

fn shader_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/geodesic_spheres")
}

fn shader_path() -> PathBuf {
    shader_dir().join("spheres.wgsl")
}

trait SphereKnobs: WgslSpace + Default + 'static {
    const BALL_SCALE: f32;
    const FOG_SCALE: f32;
    const TITLE: &'static str;
    /// Camera distance for capture mode. Constrained by the Space's
    /// chart: `distance * BALL_SCALE` must stay inside the Poincaré
    /// ball / spherical hemisphere.
    const CAPTURE_DISTANCE: f32;
}

impl SphereKnobs for EuclideanR3 {
    const BALL_SCALE: f32 = 0.2;
    const FOG_SCALE: f32 = 3.2;
    const TITLE: &'static str = "Rye - Geodesic Spheres";
    const CAPTURE_DISTANCE: f32 = 5.5;
}

impl SphereKnobs for HyperbolicH3 {
    const BALL_SCALE: f32 = 0.2;
    const FOG_SCALE: f32 = 3.0;
    const TITLE: &'static str = "Rye - Geodesic Spheres (H3 fog)";
    // 4.5 * 0.2 = 0.9, safely inside the Poincaré ball.
    const CAPTURE_DISTANCE: f32 = 4.5;
}

impl SphereKnobs for SphericalS3 {
    const BALL_SCALE: f32 = 0.2;
    const FOG_SCALE: f32 = 2.6;
    const TITLE: &'static str = "Rye - Geodesic Spheres (S3 fog)";
    // Same upper-hemisphere constraint as H³: distance * ball_scale < 1.0.
    const CAPTURE_DISTANCE: f32 = 4.5;
}

struct SpheresApp<S: SphereKnobs> {
    space: S,
    camera: OrbitCamera,
    shader_id: ShaderId,
    shader_gen: u64,
    ray_march: GeodesicRayMarchNode,
    rotate: bool,
    /// Yaw advance per tick when `rotate` is on. Interactive: 1 rev /
    /// 20 s at 60 Hz. Capture: `TAU / capture_frames` so the loop
    /// closes on a full revolution.
    rotate_yaw_per_tick: f32,
}

impl<S: SphereKnobs> App for SpheresApp<S> {
    type Space = S;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let space = S::default();
        let scene_module = geodesic_spheres_demo_wgsl();
        let shader_id = ctx
            .shader_db
            .load_geodesic_scene(shader_path(), &scene_module, &space)?;
        let shader_gen = ctx.shader_db.generation(shader_id);
        let ray_march = GeodesicRayMarchNode::from_module(
            &ctx.rd.device,
            ctx.rd.surface_bundle.config.format,
            ctx.shader_db.module(shader_id),
        );

        if let Some(watcher) = ctx.watcher.as_mut() {
            watcher.watch(shader_dir())?;
        }

        let args: Vec<String> = std::env::args().collect();
        let capturing = args.iter().any(|a| a.starts_with("--capture-"));
        let rotate = capturing || args.iter().any(|a| a == "--rotate");

        let mut camera = OrbitCamera::default();
        if capturing {
            camera.set_orbit(S::CAPTURE_DISTANCE, -0.60);
        }

        let rotate_yaw_per_tick = if capturing {
            let frames = arg_value(&args, "--capture-frames")
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(300);
            std::f32::consts::TAU / frames as f32
        } else {
            std::f32::consts::TAU / (60.0 * 20.0)
        };

        Ok(Self {
            space,
            camera,
            shader_id,
            shader_gen,
            ray_march,
            rotate,
            rotate_yaw_per_tick,
        })
    }

    fn space(&self) -> &S {
        &self.space
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        if self.rotate {
            self.camera
                .rotate_yaw(self.rotate_yaw_per_tick * ctx.n_ticks as f32);
        }
        self.camera.advance(ctx.input);

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
            tracing::info!("rebuilding GeodesicRayMarchNode for shader gen {new_gen}");
            self.shader_gen = new_gen;
            self.ray_march = GeodesicRayMarchNode::from_module(
                &ctx.rd.device,
                ctx.rd.surface_bundle.config.format,
                ctx.shader_db.module(self.shader_id),
            );
        }
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        self.ray_march.execute(rd, view)
    }

    fn title(&self, fps: f32) -> Cow<'static, str> {
        Cow::Owned(format!("{} | {fps:.0} fps", S::TITLE))
    }
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    let i = args.iter().position(|a| a == flag)?;
    args.get(i + 1).cloned()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let hyperbolic = args.iter().any(|a| a == "--hyperbolic");
    let spherical = args.iter().any(|a| a == "--spherical");

    let config = RunConfig {
        window: WindowAttributes::default()
            .with_title("Rye - Geodesic Spheres")
            .with_visible(false),
        capture: CaptureConfig::from_env_args(),
        ..RunConfig::default()
    };

    if hyperbolic {
        run_with_config::<SpheresApp<HyperbolicH3>>(config)
    } else if spherical {
        run_with_config::<SpheresApp<SphericalS3>>(config)
    } else {
        run_with_config::<SpheresApp<EuclideanR3>>(config)
    }
}
