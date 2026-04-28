//! Geodesic lattice, side-by-side E³ / H³ / S³ comparison demo.
//!
//! Renders three panels into a single window: left E³, centre H³, right S³.
//! The same camera orbits the same lattice of spheres; the only difference
//! is the Space prelude injected at shader compile time. The visual difference
//! (evenly-spaced grid vs. tanh-compressed vs. sin-wrapped) is the engine's
//! geometric thesis made visible.
//!
//! `lattice.wgsl` is compiled three times, once per Space, using
//! `WgslSpace::wgsl_impl()` assembled directly. The framework's
//! `ShaderDb` is single-path-keyed, so the three shader modules are
//! built outside it; hot-reload is disabled here.
//!
//! ## Flags
//!
//! `--rotate`              : auto-rotate camera
//! `--capture-apng PATH`   : render N frames, save looping APNG, exit
//! `--capture-gif  PATH`   : render N frames, save looping GIF, exit
//! `--capture-frames N`    : frame count (default 300)
//! `--capture-fps N`       : playback fps (default 30)
//!
//! Capture mode forces `--rotate`.

use std::borrow::Cow;
use std::path::PathBuf;

use anyhow::Result;
use rye_app::{run_with_config, App, CaptureConfig, FrameCtx, RunConfig, SetupCtx};
use rye_camera::OrbitCamera;
use rye_math::{EuclideanR3, HyperbolicH3, SphericalS3, WgslSpace};
use rye_render::{
    device::RenderDevice,
    raymarch::{GeodesicRayMarchNode, RayMarchUniforms},
};
use rye_sdf::LatticeSphereScene;
use rye_shader::{validate_wgsl, GEODESIC_MARCH_KERNEL};
use winit::window::WindowAttributes;

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

fn shader_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/lattice/lattice.wgsl")
}

/// Assemble Space prelude + scene SDF + geodesic kernel + user shader.
fn assemble(prelude: &str, scene: &str, user: &str) -> String {
    format!(
        "// ---- rye-math Space prelude ----\n{prelude}\n\
         // ---- rye-sdf scene module ----\n{scene}\n\
         // ---- rye geodesic march kernel ----\n{GEODESIC_MARCH_KERNEL}\n\
         // ---- user shading ----\n{user}"
    )
}

struct LatticeApp {
    /// Required by the trait. ShaderDb hot-reload would re-emit
    /// preludes against this single Space, but the demo manually
    /// assembles three modules per Space, so the field is dormant.
    space: EuclideanR3,
    node_e3: GeodesicRayMarchNode,
    node_h3: GeodesicRayMarchNode,
    node_s3: GeodesicRayMarchNode,
    camera: OrbitCamera,
    rotate: bool,
    rotate_yaw_per_tick: f32,
}

impl LatticeApp {
    fn panel_uniforms(
        &self,
        time: f32,
        tick: u64,
        w: u32,
        h: u32,
        x_offset: u32,
        panel_w: u32,
        panel_idx: f32,
        fog_scale: f32,
    ) -> RayMarchUniforms {
        let camera = self.camera.view();
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
            time,
            tick: tick as f32,
            params: [x_offset as f32, panel_idx, panel_w as f32, fog_scale],
        }
    }
}

impl App for LatticeApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let rd = ctx.rd;

        // Build lattice scene modules; centers are computed per-Space in Rust.
        let lattice = LatticeSphereScene::default();
        let scene_e3 = lattice.to_wgsl(&EuclideanR3);
        let scene_h3 = lattice.to_wgsl(&HyperbolicH3);
        let scene_s3 = lattice.to_wgsl(&SphericalS3);

        // Read the user WGSL once.
        let user_src = std::fs::read_to_string(shader_path())?;

        // Assemble and compile three distinct shader modules.
        let make_module = |prelude: &str, scene: &str| -> Result<wgpu::ShaderModule> {
            let full = assemble(prelude, scene, &user_src);
            validate_wgsl(&full)?;
            Ok(rd
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("lattice"),
                    source: wgpu::ShaderSource::Wgsl(full.into()),
                }))
        };

        let fmt = rd.surface_bundle.config.format;
        let mod_e3 = make_module(&EuclideanR3.wgsl_impl(), &scene_e3)?;
        let mod_h3 = make_module(&HyperbolicH3.wgsl_impl(), &scene_h3)?;
        let mod_s3 = make_module(&SphericalS3.wgsl_impl(), &scene_s3)?;

        let node_e3 = GeodesicRayMarchNode::from_module(&rd.device, fmt, &mod_e3);
        let node_h3 = GeodesicRayMarchNode::from_module(&rd.device, fmt, &mod_h3);
        let node_s3 = GeodesicRayMarchNode::from_module(&rd.device, fmt, &mod_s3);

        let args: Vec<String> = std::env::args().collect();
        let capturing = args.iter().any(|a| a.starts_with("--capture-"));
        let rotate = capturing || args.iter().any(|a| a == "--rotate");

        let mut camera = OrbitCamera::default();
        if capturing {
            camera.set_orbit(CAPTURE_DISTANCE, CAPTURE_PITCH);
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
            space: EuclideanR3,
            node_e3,
            node_h3,
            node_s3,
            camera,
            rotate,
            rotate_yaw_per_tick,
        })
    }

    fn space(&self) -> &EuclideanR3 {
        &self.space
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        if self.rotate {
            self.camera
                .rotate_yaw(self.rotate_yaw_per_tick * ctx.n_ticks as f32);
        }
        self.camera.advance(ctx.input);

        let cfg = &ctx.rd.surface_bundle.config;
        let w = cfg.width;
        let h = cfg.height;
        let pw = w / 3;
        let pw2 = w - pw * 2;

        let u_e3 = self.panel_uniforms(ctx.time, ctx.tick, w, h, 0, pw, 0.0, FOG_E3);
        let u_h3 = self.panel_uniforms(ctx.time, ctx.tick, w, h, pw, pw, 1.0, FOG_H3);
        let u_s3 = self.panel_uniforms(ctx.time, ctx.tick, w, h, pw * 2, pw2, 2.0, FOG_S3);
        self.node_e3.set_uniforms(&ctx.rd.queue, u_e3);
        self.node_h3.set_uniforms(&ctx.rd.queue, u_h3);
        self.node_s3.set_uniforms(&ctx.rd.queue, u_s3);
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        let w = rd.surface_bundle.config.width;
        let h = rd.surface_bundle.config.height;
        let pw = w / 3;
        let pw2 = w - pw * 2;

        // E³: clear + draw left panel.
        self.node_e3.execute_panel(rd, view, true, [0, 0, pw, h])?;
        // H³: load + draw centre panel.
        self.node_h3
            .execute_panel(rd, view, false, [pw, 0, pw, h])?;
        // S³: load + draw right panel.
        self.node_s3
            .execute_panel(rd, view, false, [pw * 2, 0, pw2, h])?;
        Ok(())
    }

    fn title(&self, fps: f32) -> Cow<'static, str> {
        Cow::Owned(format!(
            "Rye - Geodesic Lattice (E³ / H³ / S³) | {fps:.0} fps"
        ))
    }
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    let i = args.iter().position(|a| a == flag)?;
    args.get(i + 1).cloned()
}

fn main() -> Result<()> {
    let config = RunConfig {
        window: WindowAttributes::default()
            .with_title("Rye - Geodesic Lattice (E³ / H³ / S³)")
            .with_visible(false),
        capture: CaptureConfig::from_env_args(),
        ..RunConfig::default()
    };
    run_with_config::<LatticeApp>(config)
}
