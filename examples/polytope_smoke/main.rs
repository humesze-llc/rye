//! Smoke test for the polytope path in `Hyperslice4DNode`. Renders
//! a static pentatope and a tesseract sitting on a 4D `y = 0` floor,
//! with user-controllable `w`-slice scrubbing and a 4D rotation
//! mode picker that exercises three different Rotor4 motions.
//!
//! Doubles as the integration showcase for `rye-text` — all live
//! state and the controls help are drawn in-window via the text
//! crate, no window-title stuffing.
//!
//! What this verifies (visually):
//!
//! - The kernel's polytope SDF compiles + executes (no naga errors,
//!   no shader runtime panics).
//! - Pentatope cross-sections morph through the 5-cell's slice
//!   shapes as `w_slice` scrubs.
//! - The Rotor4 inverse-sandwich path correctly transforms world
//!   points into body-local for evaluating the polytope SDF —
//!   single-plane and compound 4D rotations both produce coherent
//!   slice morphs.
//! - `rye-text` renders ASCII glyphs over a hyperslice scene
//!   (atlas + textured-quad pipeline composes cleanly with
//!   `Hyperslice4DNode`).
//!
//! ## Controls
//!
//! - **Mouse left-drag**: orbit camera.
//! - **↑ / ↓**: scrub `w`-slice (0.5 u/s).
//! - **T**: toggle 4D rotation of the tesseract.
//! - **M**: cycle rotation mode (xw / isoclinic xw+yz / compound).
//! - **+ / −**: adjust rotation rate (multiplies base rate).
//! - **R**: reset slice + rotation.
//! - **Esc**: exit.

use std::path::Path;

use anyhow::{anyhow, Result};
use glam::{Vec3, Vec4};
use rye_app::{run_with_config, App, Camera, FrameCtx, OrbitController, RunConfig, SetupCtx};
use rye_math::EuclideanR3;
use rye_render::{
    device::RenderDevice,
    graph::RenderNode,
    raymarch::{BodyUniform, Hyperslice4DNode, HYPERSLICE_KERNEL_WGSL},
};
use rye_sdf::{Scene4, SceneNode4};
use rye_text::TextRenderer;
use winit::window::WindowAttributes;

const SHAPE_PENTATOPE: u32 = 0;
const SHAPE_TESSERACT: u32 = 1;

const W_SCRUB_RATE: f32 = 0.5;
const W_RANGE: f32 = 1.5;

const IDENTITY_ROTOR: [f32; 8] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

/// Base rotation angular rate (rad/s). Scaled by `rate_scale` per
/// frame so +/- can speed it up or slow it down.
const BASE_ROTATION_RATE: f32 = std::f32::consts::TAU * 0.3;
/// Golden ratio — incommensurate ratio for "compound" mode so the
/// rotor never returns to identity (the slice morph never repeats).
const PHI: f32 = 1.618_034;

const TESSERACT_POSITION: [f32; 4] = [1.6, 0.5, 0.0, 0.0];
const TESSERACT_SIZE: f32 = 0.9;
const TESSERACT_COLOR: [f32; 3] = [0.30, 0.55, 0.95];

// ---------------------------------------------------------------------------
// Rotation modes
// ---------------------------------------------------------------------------

/// Which 4D rotation animates the tesseract while `rotate` is on.
#[derive(Copy, Clone, PartialEq, Eq)]
enum RotMode {
    /// Single xw-plane rotation. Slice cross-section = rectangular
    /// prism stretching along x as θ varies. Periodic with period π.
    Xw,
    /// Isoclinic xw + yz at equal rates. Both x stretches AND the
    /// y-z plane tilts. Slice cross-section is a parallelepiped
    /// that morphs continuously. Periodic with period π.
    Isoclinic,
    /// Compound xw + yz at golden-ratio rate ratio. Two
    /// incommensurate planar rotations means the rotor never
    /// returns to identity — the cross-section never repeats.
    Compound,
}

impl RotMode {
    fn name(self) -> &'static str {
        match self {
            RotMode::Xw => "xw single",
            RotMode::Isoclinic => "isoclinic xw+yz",
            RotMode::Compound => "compound xw+phi*yz",
        }
    }
    fn next(self) -> Self {
        match self {
            RotMode::Xw => RotMode::Isoclinic,
            RotMode::Isoclinic => RotMode::Compound,
            RotMode::Compound => RotMode::Xw,
        }
    }
}

/// Build a Rotor4 for `mode` at simulation time `t` scaled by
/// `rate_scale`. Component order matches `BodyUniform.rotor`:
/// `[s, xy, xz, xw, yz, yw, zw, xyzw]`.
fn build_rotor(mode: RotMode, t: f32, rate_scale: f32) -> [f32; 8] {
    let alpha = t * BASE_ROTATION_RATE * rate_scale;
    match mode {
        RotMode::Xw => single_plane_rotor(alpha, 3),
        RotMode::Isoclinic => compound_rotor(alpha, alpha),
        RotMode::Compound => compound_rotor(alpha, alpha * PHI),
    }
}

/// Single-plane rotor: `R = cos(θ/2) + sin(θ/2)·e_plane`.
/// `plane_index` selects which bivector slot (3 = xw, 4 = yz, etc).
fn single_plane_rotor(angle: f32, plane_index: usize) -> [f32; 8] {
    let half = angle * 0.5;
    let (s, c) = half.sin_cos();
    let mut r = [0.0_f32; 8];
    r[0] = c;
    r[plane_index] = s;
    r
}

/// Compound rotor for two perpendicular planes (xw, yz).
/// `R = exp(α/2 · e_xw) · exp(β/2 · e_yz)` =
/// `cos(α/2)cos(β/2) + sin(α/2)cos(β/2)·e_xw + cos(α/2)sin(β/2)·e_yz`
/// `+ sin(α/2)sin(β/2)·e_xyzw`.
///
/// The pseudoscalar (`e_xyzw`) term is what makes compound 4D
/// rotations qualitatively different from any single 3D rotation.
fn compound_rotor(alpha: f32, beta: f32) -> [f32; 8] {
    let (sa, ca) = (alpha * 0.5).sin_cos();
    let (sb, cb) = (beta * 0.5).sin_cos();
    let mut r = [0.0_f32; 8];
    r[0] = ca * cb; // scalar
    r[3] = sa * cb; // xw
    r[4] = ca * sb; // yz
    r[7] = sa * sb; // pseudoscalar
    r
}

// ---------------------------------------------------------------------------
// Font discovery (portable system-font fallback)
// ---------------------------------------------------------------------------

fn load_system_font() -> Result<Vec<u8>> {
    const CANDIDATES: &[&str] = &[
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ];
    for &p in CANDIDATES {
        if Path::new(p).exists() {
            return Ok(std::fs::read(p)?);
        }
    }
    Err(anyhow!(
        "no candidate system font found; tried {CANDIDATES:?}"
    ))
}

// ---------------------------------------------------------------------------
// PolytopeSmokeApp
// ---------------------------------------------------------------------------

struct PolytopeSmokeApp {
    space: EuclideanR3,
    camera: Camera<EuclideanR3>,
    orbit: OrbitController<EuclideanR3>,
    node: Hyperslice4DNode,
    text: TextRenderer,

    w_slice: f32,
    slider_up_held: bool,
    slider_down_held: bool,

    rotate: bool,
    rot_time: f32,
    rot_mode: RotMode,
    rate_scale: f32,
}

impl PolytopeSmokeApp {
    fn write_tesseract(&mut self, rotor: [f32; 8]) {
        let body = BodyUniform::polytope(
            TESSERACT_POSITION,
            SHAPE_TESSERACT,
            TESSERACT_SIZE,
            rotor,
            TESSERACT_COLOR,
        );
        self.node.set_body(1, body);
    }
}

impl App for PolytopeSmokeApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
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

        let bodies = [
            BodyUniform::polytope(
                [-1.6, 1.0, 0.0, 0.0],
                SHAPE_PENTATOPE,
                0.9,
                IDENTITY_ROTOR,
                [0.95, 0.55, 0.30],
            ),
            BodyUniform::polytope(
                TESSERACT_POSITION,
                SHAPE_TESSERACT,
                TESSERACT_SIZE,
                IDENTITY_ROTOR,
                TESSERACT_COLOR,
            ),
        ];
        node.set_bodies(&bodies);

        let font = load_system_font()?;
        let text = TextRenderer::new(
            &ctx.rd.device,
            &ctx.rd.queue,
            ctx.rd.surface_bundle.config.format,
            &font,
            48.0,
        )?;

        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 2.5, 6.0);
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        orbit.set_orbit(6.5, -0.30);

        Ok(Self {
            space: EuclideanR3,
            camera,
            orbit,
            node,
            text,
            w_slice: 0.0,
            slider_up_held: false,
            slider_down_held: false,
            rotate: false,
            rot_time: 0.0,
            rot_mode: RotMode::Xw,
            rate_scale: 1.0,
        })
    }

    fn space(&self) -> &EuclideanR3 {
        &self.space
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        let dt_secs = ctx.n_ticks as f32 / 60.0;

        // Slice scrub.
        let dir = (self.slider_up_held as i32 - self.slider_down_held as i32) as f32;
        if dir != 0.0 {
            self.w_slice = (self.w_slice + dir * W_SCRUB_RATE * dt_secs).clamp(-W_RANGE, W_RANGE);
        }

        // 4D rotation animation.
        if self.rotate {
            self.rot_time += dt_secs;
            let rotor = build_rotor(self.rot_mode, self.rot_time, self.rate_scale);
            self.write_tesseract(rotor);
        }

        // Camera.
        use rye_camera::CameraController;
        self.orbit
            .advance(ctx.input, &mut self.camera, &EuclideanR3, 0.0);
        let view = self.camera.view();

        // Hyperslice uniforms.
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

        // Text overlay — built fresh each frame; rye-text resets
        // its queue on render so leftover labels don't accumulate.
        let w = cfg.width as f32;
        let h = cfg.height as f32;

        // Top-left: live state readout.
        let white = [1.0, 1.0, 1.0, 1.0];
        let dim = [0.85, 0.85, 0.95, 0.85];
        self.text.queue("polytope smoke", [16.0, 16.0], 26.0, white);
        self.text
            .queue(&format!("{:.0} fps", ctx.fps), [16.0, 48.0], 18.0, dim);
        self.text.queue(
            &format!("w_slice = {:+.3}", self.w_slice),
            [16.0, 72.0],
            18.0,
            dim,
        );
        if self.rotate {
            self.text.queue(
                &format!(
                    "spin: {} | rate x{:.2} | t = {:.2} s",
                    self.rot_mode.name(),
                    self.rate_scale,
                    self.rot_time
                ),
                [16.0, 96.0],
                18.0,
                [0.40, 1.0, 0.60, 1.0],
            );
        } else {
            self.text.queue(
                &format!("spin: paused (mode: {})", self.rot_mode.name()),
                [16.0, 96.0],
                18.0,
                [0.95, 0.85, 0.30, 1.0],
            );
        }

        // Bottom-left: controls help.
        let help_lines = [
            "drag-orbit  |  up/dn scrub w  |  t toggle spin",
            "m cycle mode  |  +/- adjust rate  |  r reset",
        ];
        let help_size = 16.0;
        let line_h = help_size * 1.3;
        let bottom = h - 16.0 - line_h * help_lines.len() as f32;
        for (i, line) in help_lines.iter().enumerate() {
            self.text.queue(
                line,
                [16.0, bottom + i as f32 * line_h],
                help_size,
                [0.05, 0.10, 0.45, 1.0],
            );
        }

        // Bottom-right: window size — handy when sizing for
        // screenshots.
        self.text.queue(
            &format!("{w:.0}x{h:.0}"),
            [w - 110.0, h - 28.0],
            14.0,
            [1.0, 1.0, 1.0, 0.5],
        );
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
            KeyCode::KeyR if pressed => {
                self.w_slice = 0.0;
                self.rot_time = 0.0;
                self.rate_scale = 1.0;
                self.write_tesseract(IDENTITY_ROTOR);
            }
            KeyCode::KeyT if pressed => {
                self.rotate = !self.rotate;
                if !self.rotate {
                    self.rot_time = 0.0;
                    self.write_tesseract(IDENTITY_ROTOR);
                }
            }
            KeyCode::KeyM if pressed => {
                self.rot_mode = self.rot_mode.next();
                // Don't reset rot_time — let the new mode pick up
                // mid-animation so cycling feels continuous.
            }
            // Numpad and main-row +/- both adjust rate. Step by 25%
            // multiplicatively so each press is a noticeable change.
            KeyCode::Equal | KeyCode::NumpadAdd if pressed => {
                self.rate_scale = (self.rate_scale * 1.25).min(8.0);
            }
            KeyCode::Minus | KeyCode::NumpadSubtract if pressed => {
                self.rate_scale = (self.rate_scale / 1.25).max(0.05);
            }
            _ => {}
        }
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        // Hyperslice scene first; text overlay on top.
        self.node.execute(rd, view)?;
        let cfg = &rd.surface_bundle.config;
        self.text.render(
            &rd.device,
            &rd.queue,
            view,
            [cfg.width as f32, cfg.height as f32],
        )?;
        Ok(())
    }

    fn title(&self, _fps: f32) -> std::borrow::Cow<'static, str> {
        // Window title is now decorative — all live state is in the
        // overlay. Keep the title static so OS task switchers show
        // a stable label.
        std::borrow::Cow::Borrowed("polytope smoke")
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
