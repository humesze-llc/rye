//! `blended` — Phase 4 BlendedSpace milestone demo.
//!
//! Renders a row of spheres above a horizontal floor under
//! `BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX>`.
//! The X axis carries a smooth metric transition: pure Euclidean for
//! `x ≤ −1`, pure Poincaré-H³ for `x ≥ 1`, with a quintic-smootherstep
//! interpolation between. The shading tints surfaces blue→red by
//! blending alpha so the seam is visible.
//!
//! What the rendered image proves:
//! - The geodesic march kernel runs against a non-trivial WGSL prelude
//!   (RK4 integrator inlined per `rye_exp` call, 4 sub-steps per call).
//! - Sphere silhouettes curve in the transition zone — straight lines
//!   bend through the variable metric.
//! - On the H³ side, spheres further from the origin appear smaller
//!   (the chart compresses toward the Poincaré boundary).
//!
//! Controls: left-drag orbit, scroll zoom, R reset orbit, Esc exit.

use std::path::PathBuf;

use anyhow::Result;
use rye_app::{run_with_config, App, Camera, FrameCtx, OrbitController, RunConfig, SetupCtx};
use rye_math::{BlendedSpace, EuclideanR3, HyperbolicH3, LinearBlendX};
use rye_render::{
    device::RenderDevice,
    raymarch::{RayMarchNode, RayMarchUniforms},
};
use rye_shader::ShaderId;
use winit::window::WindowAttributes;

type DemoSpace = BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX>;

fn shader_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/blended")
}
fn shader_path() -> PathBuf {
    shader_dir().join("blended.wgsl")
}

/// Scene SDF: floor at y=0 plus a row of spheres at y=0.6
/// spaced along x ∈ [−2.4, 2.4]. Sphere radius shrinks slightly
/// going right so the H³ chart compression is visible without
/// being confused with the actual scene scale.
///
/// Returns WGSL that defines `rye_scene_sdf(p)` — concatenated by
/// `ShaderDb::load_geodesic_scene` ahead of the geodesic-march
/// kernel and the user shader.
///
/// The scene reads `u.show_spheres` (uniform binding via the
/// concatenated user shader) to optionally suppress the sphere
/// row — `--floor-only` mode is a diagnostic for confirming the
/// floor itself renders cleanly through the variable metric.
fn scene_module_wgsl() -> &'static str {
    r#"
const SPHERE_COUNT: i32 = 7;
fn sphere_x(i: i32) -> f32 { return -2.4 + 0.8 * f32(i); }
fn sphere_radius(i: i32) -> f32 { return 0.22; }

fn rye_scene_sdf(p: vec3<f32>) -> f32 {
    // Floor as a chart-coordinate horizontal plane at y=0.
    var d = p.y;

    // Row of spheres just above the floor (suppressed in
    // diagnostic --floor-only mode).
    if u.show_spheres > 0.5 {
        for (var i: i32 = 0; i < SPHERE_COUNT; i = i + 1) {
            let center = vec3<f32>(sphere_x(i), 0.6, 0.0);
            let s = length(p - center) - sphere_radius(i);
            d = min(d, s);
        }
    }

    return d;
}
"#
}

struct BlendedApp {
    space: DemoSpace,
    camera: Camera<EuclideanR3>,
    orbit: OrbitController<EuclideanR3>,
    shader_id: ShaderId,
    shader_gen: u64,
    ray_march: RayMarchNode,
    show_spheres: f32,
}

impl App for BlendedApp {
    type Space = DemoSpace;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        // Zone width 2.0 centred on origin: pure E³ for x ≤ −1,
        // pure H³ for x ≥ +1, smootherstep between.
        let space = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-1.0, 1.0));

        let shader_id =
            ctx.shader_db
                .load_geodesic_scene(shader_path(), scene_module_wgsl(), &space)?;
        let shader_gen = ctx.shader_db.generation(shader_id);
        let ray_march = RayMarchNode::new(
            &ctx.rd.device,
            ctx.rd.surface_bundle.config.format,
            ctx.shader_db.module(shader_id),
        );

        if let Some(watcher) = ctx.watcher.as_mut() {
            watcher.watch(shader_dir())?;
        }

        let mut orbit = OrbitController::default();
        // Strong downward pitch so the floor occupies most of the
        // lower screen and the spheres sit clearly above it.
        orbit.set_orbit(5.0, -0.55);

        let show_spheres = if std::env::args().any(|a| a == "--floor-only") {
            0.0
        } else {
            1.0
        };

        Ok(Self {
            space,
            camera: Camera::<EuclideanR3>::at_origin(),
            orbit,
            shader_id,
            shader_gen,
            ray_march,
            show_spheres,
        })
    }

    fn space(&self) -> &Self::Space {
        &self.space
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        use rye_camera::CameraController;
        self.orbit
            .advance(ctx.input, &mut self.camera, &EuclideanR3, 0.0);

        let view = self.camera.view();
        let cfg = &ctx.rd.surface_bundle.config;
        // ball_scale: chart coordinates × ball_scale must keep the camera
        // and scene inside the Poincaré ball |p| < 1 on the H³ side.
        // Camera is ~5 units out in chart coords; 0.18 keeps it at
        // ~0.9 ball radius — safely inside.
        let ball_scale = 0.18;
        let fog_scale = 2.4;
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
                params: [0.0, ball_scale, fog_scale, self.show_spheres],
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
            "Rye — BlendedSpace<E3,H3> | {fps:.0} fps | pos ({:.2}, {:.2}, {:.2})",
            p.x, p.y, p.z
        ))
    }
}

fn main() -> Result<()> {
    let config = RunConfig {
        window: WindowAttributes::default()
            .with_title("Rye — BlendedSpace<E3,H3>")
            .with_visible(false),
        ..RunConfig::default()
    };
    run_with_config::<BlendedApp>(config)
}
