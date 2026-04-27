//! Smoke test for the polytope path in `Hyperslice4DNode`. Renders
//! one static pentatope and one static tesseract sitting on a 4D
//! `y = 0` floor, with the user-controllable `w`-slice scrubbing
//! through both shapes.
//!
//! What this verifies (visually):
//!
//! - The kernel's polytope SDF compiles + executes (no naga errors,
//!   no shader runtime panics).
//! - The pentatope's 5-cell cross-sections (triangles, quads,
//!   pentagons) appear as the slice depth changes — distinct from
//!   the tesseract's cube cross-sections.
//! - The dynamic-body uniform array routes both `BODY_KIND_SPHERE`
//!   and `BODY_KIND_POLYTOPE` correctly: the floor (static
//!   `Scene4`) is the third visible surface and shouldn't disappear
//!   regardless of which body is in front.
//!
//! ## Controls
//!
//! - **Mouse left-drag**: orbit camera.
//! - **↑ / ↓**: scrub `w`-slice (0.5 u/s).
//! - **R**: reset slice to 0.
//! - **Esc**: exit.

use anyhow::Result;
use glam::{Vec3, Vec4};
use rye_app::{run_with_config, App, Camera, FrameCtx, OrbitController, RunConfig, SetupCtx};
use rye_math::EuclideanR3;
use rye_render::{
    device::RenderDevice,
    graph::RenderNode,
    raymarch::{BodyUniform, Hyperslice4DNode, HYPERSLICE_KERNEL_WGSL},
};
use rye_sdf::{Scene4, SceneNode4};
use winit::window::WindowAttributes;

/// Pentatope shape index in the kernel's shape table.
const SHAPE_PENTATOPE: u32 = 0;
/// Tesseract shape index in the kernel's shape table.
const SHAPE_TESSERACT: u32 = 1;

/// Slice scrub speed, world units per second.
const W_SCRUB_RATE: f32 = 0.5;
/// Bounded slice range so the user doesn't get lost.
const W_RANGE: f32 = 1.5;

const IDENTITY_ROTOR: [f32; 8] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

struct PolytopeSmokeApp {
    space: EuclideanR3,
    camera: Camera<EuclideanR3>,
    orbit: OrbitController<EuclideanR3>,
    node: Hyperslice4DNode,

    w_slice: f32,
    slider_up_held: bool,
    slider_down_held: bool,
}

impl App for PolytopeSmokeApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        // Static scene: 4D floor at y = 0.
        let scene = Scene4::new(SceneNode4::halfspace(Vec4::Y, 0.0));
        let shader_source = format!(
            "{kernel}\n{scene}\n",
            kernel = HYPERSLICE_KERNEL_WGSL,
            scene = scene.to_hyperslice_wgsl("u.w_slice"),
        );
        let module = ctx
            .rd
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("polytope_smoke shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });
        let mut node =
            Hyperslice4DNode::new(&ctx.rd.device, ctx.rd.surface_bundle.config.format, &module);

        // Two static polytopes sitting just above the floor at
        // distinct x positions so they don't overlap.
        let bodies = [
            BodyUniform::polytope(
                [-1.6, 1.0, 0.0, 0.0],
                SHAPE_PENTATOPE,
                0.9,
                IDENTITY_ROTOR,
                [0.95, 0.55, 0.30], // warm orange
            ),
            BodyUniform::polytope(
                [1.6, 0.5, 0.0, 0.0],
                SHAPE_TESSERACT,
                0.9,
                IDENTITY_ROTOR,
                [0.30, 0.55, 0.95], // cool blue
            ),
        ];
        node.set_bodies(&bodies);

        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 2.5, 6.0);
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        orbit.set_orbit(6.5, -0.30);

        Ok(Self {
            space: EuclideanR3,
            camera,
            orbit,
            node,
            w_slice: 0.0,
            slider_up_held: false,
            slider_down_held: false,
        })
    }

    fn space(&self) -> &EuclideanR3 {
        &self.space
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        // Slice scrub.
        let dir = (self.slider_up_held as i32 - self.slider_down_held as i32) as f32;
        if dir != 0.0 {
            let dt_secs = ctx.n_ticks as f32 / 60.0;
            self.w_slice = (self.w_slice + dir * W_SCRUB_RATE * dt_secs).clamp(-W_RANGE, W_RANGE);
        }

        use rye_camera::CameraController;
        self.orbit
            .advance(ctx.input, &mut self.camera, &EuclideanR3, 0.0);
        let view = self.camera.view();

        let cfg = &ctx.rd.surface_bundle.config;
        {
            let u = self.node.uniforms_mut();
            u.camera_pos = view.position.to_array();
            u.camera_forward = view.forward.to_array();
            u.camera_right = view.right.to_array();
            u.camera_up = view.up.to_array();
            u.fov_y_tan = (60.0_f32.to_radians() * 0.5).tan();
            u.resolution = [cfg.width as f32, cfg.height as f32];
            u.time = ctx.time;
            u.tick = ctx.tick as f32;
            u.w_slice = self.w_slice;
        }
        self.node.flush_uniforms(&ctx.rd.queue);
    }

    fn on_event(&mut self, ev: &winit::event::WindowEvent, _ctx: &mut FrameCtx<'_>) {
        use winit::event::{ElementState, WindowEvent};
        use winit::keyboard::{KeyCode, PhysicalKey};
        let WindowEvent::KeyboardInput { event, .. } = ev else {
            return;
        };
        let PhysicalKey::Code(kc) = event.physical_key else {
            return;
        };
        let pressed = event.state == ElementState::Pressed;
        match kc {
            KeyCode::ArrowUp => self.slider_up_held = pressed,
            KeyCode::ArrowDown => self.slider_down_held = pressed,
            KeyCode::KeyR if pressed => self.w_slice = 0.0,
            _ => {}
        }
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        self.node.execute(rd, view)
    }

    fn title(&self, fps: f32) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Owned(format!(
            "polytope smoke | {fps:.0} fps | w_slice = {:+.3}",
            self.w_slice
        ))
    }
}

fn main() -> Result<()> {
    let config = RunConfig {
        window: WindowAttributes::default()
            .with_title("polytope smoke")
            .with_visible(false),
        ..RunConfig::default()
    };
    run_with_config::<PolytopeSmokeApp>(config)
}
