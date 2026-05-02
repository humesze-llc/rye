//! Static overlapping-SDF test scene. Visual regression rig for the
//! hyperslice marcher (issue #17 follow-up).
//!
//! Renders four hyperspheres on a `y = 0` floor, all at `w = 0`, no
//! rotation, no dynamics. The geometry is deliberately simple so any
//! visible artifact is a marcher / SDF / shading bug, not an
//! interaction with rotor4 or animation.
//!
//! ```text
//! arrangement (top-down view):
//!
//!     z = -1.5  o------o------o          (twin-pair: spheres at -0.6/+0.6
//!                                         in x, both r=0.7 -> overlap at x=0;
//!                                         middle sphere at z=-1.5)
//!
//!     z = +1.5         o                 (solo: sphere at x=0, z=+1.5,
//!                                         r=1.0; isolated, sits on floor)
//!
//!                      ^
//!                  camera looks
//!                  this way
//! ```
//!
//! ## What to look for
//!
//! - **Twin overlap**: the seam between the two left spheres is a
//!   smooth saddle. No background leaks through their shared
//!   silhouette. No dark/light banding at their join.
//! - **Sphere-floor contact**: the floor's checker pattern is
//!   uninterrupted at every sphere's xy-footprint. No dimples,
//!   notches, or color shifts on the floor near a sphere.
//! - **Sky-sphere silhouette**: the outer edge of every sphere is
//!   solid against the sky. No "missing arcs" where the background
//!   shows through.
//! - **w-scrub**: as `w_slice` moves away from 0, the sphere
//!   cross-sections shrink (radius `sqrt(r^2 - w_slice^2)`). At
//!   `|w_slice| > r`, the sphere disappears. The transition should
//!   be smooth; no popping, no leftover ghost surfaces.
//!
//! ## Controls
//!
//! - **Mouse left-drag**: orbit camera.
//! - **Up / Down arrows**: scrub `w_slice` (range +/- 1.5).
//! - **R**: reset `w_slice` to 0.
//! - **Esc**: exit.

use anyhow::Result;
use glam::{Vec3, Vec4};
use rye_app::{run_with_config, App, Camera, FrameCtx, OrbitController, RunConfig, SetupCtx};
use rye_math::EuclideanR3;
use rye_render::{
    device::RenderDevice,
    graph::RenderNode,
    raymarch::{Hyperslice4DNode, HYPERSLICE_KERNEL_WGSL},
};
use rye_sdf::{Scene4, SceneNode4};
use winit::window::WindowAttributes;

const W_SCRUB_RATE: f32 = 0.5;
const W_RANGE: f32 = 1.5;

struct OverlappingSdfsApp {
    space: EuclideanR3,
    camera: Camera<EuclideanR3>,
    orbit: OrbitController<EuclideanR3>,
    node: Hyperslice4DNode,
    w_slice: f32,
    up_held: bool,
    down_held: bool,
}

fn build_scene() -> Scene4 {
    // Twin pair on the left, overlapping at x=0; one solo sphere
    // forward of the camera. All at w=0 so the cross-section at
    // w_slice=0 is the full sphere.
    let twin_l = SceneNode4::hypersphere(Vec4::new(-0.6, 0.7, -1.5, 0.0), 0.7);
    let twin_r = SceneNode4::hypersphere(Vec4::new(0.6, 0.7, -1.5, 0.0), 0.7);
    let solo = SceneNode4::hypersphere(Vec4::new(0.0, 1.0, 1.5, 0.0), 1.0);
    let floor = SceneNode4::halfspace(Vec4::Y, 0.0);

    Scene4::new(twin_l.union(twin_r).union(solo).union(floor))
}

impl App for OverlappingSdfsApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let scene = build_scene();
        let shader_source = format!(
            "{kernel}\n{scene}\n",
            kernel = HYPERSLICE_KERNEL_WGSL,
            scene = scene.to_hyperslice_wgsl("u.w_slice"),
        );
        let module = ctx
            .rd
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("overlapping_sdfs shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });
        let node =
            Hyperslice4DNode::new(&ctx.rd.device, ctx.rd.surface_bundle.config.format, &module);

        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 2.5, 6.5);
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        orbit.set_orbit(7.0, -0.2);

        Ok(Self {
            space: EuclideanR3,
            camera,
            orbit,
            node,
            w_slice: 0.0,
            up_held: false,
            down_held: false,
        })
    }

    fn space(&self) -> &EuclideanR3 {
        &self.space
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        let dt_secs = ctx.n_ticks as f32 / 60.0;

        let dir = (self.up_held as i32 - self.down_held as i32) as f32;
        if dir != 0.0 {
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
            KeyCode::ArrowUp => self.up_held = pressed,
            KeyCode::ArrowDown => self.down_held = pressed,
            KeyCode::KeyR if pressed => {
                self.w_slice = 0.0;
            }
            _ => {}
        }
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        self.node.execute(rd, view)
    }

    fn title(&self, fps: f32) -> std::borrow::Cow<'static, str> {
        format!(
            "overlapping_sdfs  fps={fps:.0}  w={w:+.3}",
            fps = fps,
            w = self.w_slice
        )
        .into()
    }
}

fn main() -> Result<()> {
    let config = RunConfig {
        window: WindowAttributes::default()
            .with_title("overlapping sdfs")
            .with_visible(false),
        ..RunConfig::default()
    };
    run_with_config::<OverlappingSdfsApp>(config)
}
