//! Smoke test for the polytope path in `Hyperslice4DNode`. Renders
//! the first four convex regular polychora, 5-cell, tesseract,
//! 16-cell, 24-cell, in a row on a 4D `y = 0` floor, with
//! user-controllable `w`-slice scrubbing and a per-plane toggle
//! UI for arbitrary 4D rotations. The user composes their own
//! motion by toggling individual rotation planes (1..6 -> xy, xz,
//! xw, yz, yw, zw); active planes' bivectors sum into the
//! per-frame angular velocity, which integrates into a rotor via
//! `(ω · dt).exp()`. Sum-of-bivectors composition is commutative,
//! so toggle-order doesn't matter and the result is always
//! predictable from the visible "active" set.
//!
//! All six convex regular 4-polytopes ship; the 120-cell and 600-cell
//! use a Rust-side face-hyperplane generator (their orbit sets are
//! too large to inline as WGSL literals). Their SDFs run a
//! true-Euclidean Wolfe greedy hyperplane projection, not a
//! max-plane lower bound.
//!
//! All live state and controls help are drawn as a `rye-egui`
//! overlay via the `App::ui` hook; the window title stays static.
//!
//! What this verifies (visually):
//!
//! - The kernel's polytope SDF compiles + executes (no naga errors,
//!   no shader runtime panics).
//! - Pentatope cross-sections morph through the 5-cell's slice
//!   shapes as `w_slice` scrubs.
//! - The Rotor4 inverse-sandwich path correctly transforms world
//!   points into body-local for evaluating the polytope SDF,
//!   single-plane and compound 4D rotations both produce coherent
//!   slice morphs.
//! - `rye-egui` overlay composes cleanly on top of
//!   `Hyperslice4DNode`'s output (the framework paints the egui
//!   pass after `App::render` returns).
//!
//! ## Controls
//!
//! - **Mouse left-drag**: orbit camera.
//! - **↑ / ↓**: scrub `w`-slice (0.5 u/s).
//! - **T**: toggle 4D rotation (pause/resume freezes orientation
//!   in place, does NOT snap back to identity).
//! - **1..6**: toggle the corresponding rotation plane on/off.
//!   The mapping is `1=xy, 2=xz, 3=xw, 4=yz, 5=yw, 6=zw`. Active
//!   planes' bivectors sum into the angular velocity. Famous
//!   compositions: `3` alone = single xw stretch; `3+4` =
//!   isoclinic xw+yz; `3+5+6` = three w-planes drift through
//!   SO(4). Pure-3D combinations (`1+2+4`) just rotate the
//!   cross-section as a rigid 3D shape.
//! - **+ / −**: adjust the global rotation rate.
//! - **R**: full reset, slice, rate, all toggles off, AND
//!   orientation back to canonical pose.
//! - **Esc**: exit.
//!
//! ## CLI
//!
//! - `--shapes name1 name2 ...`: choose the polytopes to render
//!   in left-to-right order. Names accepted include the math form
//!   (`5-cell`, `tesseract`, `16-cell`, `24-cell`, `120-cell`,
//!   `600-cell`) and Platonic-slice aliases (`tetrahedron`, `cube`,
//!   `octahedron`, `cuboctahedron`, `dodecahedron`, `icosahedron`).

use anyhow::{anyhow, Result};
use glam::{Vec3, Vec4};
use rye_app::{egui, run_with_config, App, Camera, FrameCtx, OrbitController, RunConfig, SetupCtx};
use rye_math::{Bivector, Bivector4, EuclideanR3, Rotor4};
use rye_render::{
    device::RenderDevice,
    graph::RenderNode,
    raymarch::{
        polytope_extended_sdfs_wgsl, polytope_stub_sdfs_wgsl, BodyUniform, Hyperslice4DNode,
        HYPERSLICE_KERNEL_WGSL, SHAPE_120CELL, SHAPE_16CELL, SHAPE_24CELL, SHAPE_600CELL,
        SHAPE_PENTATOPE, SHAPE_TESSERACT,
    },
};
use rye_sdf::{Scene4, SceneNode4};
use winit::window::WindowAttributes;

const W_SCRUB_RATE: f32 = 0.5;
const W_RANGE: f32 = 1.5;

const IDENTITY_ROTOR: [f32; 8] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

/// Base rotation angular rate (rad/s). Scaled by `rate_scale` per
/// frame so +/- can speed it up or slow it down.
const BASE_ROTATION_RATE: f32 = std::f32::consts::TAU * 0.3;

/// Spacing between body centers along x. Slightly larger than
/// `BODY_SIZE * 2` so rotated bodies can stretch into a neighbor's
/// column without overlap during animation.
const BODY_X_SPACING: f32 = 1.8;
/// Per-body circumradius. Smaller than the `[-2, +2]` first row of
/// shapes was at, letting four shapes fit in view at once.
const BODY_SIZE: f32 = 0.7;
/// Center-y for all bodies; floor is at y=0.
const BODY_Y: f32 = 0.9;

/// One polytope's metadata: shape index in the kernel's table,
/// per-body color, display label (used in the overlay).
#[derive(Copy, Clone)]
struct ShapeEntry {
    shape: u32,
    color: [f32; 3],
    label: &'static str,
}

/// Default row when no `--shapes` argument is given. The four
/// non-H4 polytopes; pass `--shapes 120-cell 600-cell` to render
/// the H4 pair instead.
const DEFAULT_ROW: &[ShapeEntry] = &[
    ShapeEntry {
        shape: SHAPE_PENTATOPE,
        color: [0.95, 0.55, 0.30],
        label: "5-cell",
    },
    ShapeEntry {
        shape: SHAPE_TESSERACT,
        color: [0.30, 0.55, 0.95],
        label: "8-cell",
    },
    ShapeEntry {
        shape: SHAPE_16CELL,
        color: [0.55, 0.95, 0.40],
        label: "16-cell",
    },
    ShapeEntry {
        shape: SHAPE_24CELL,
        color: [0.95, 0.45, 0.85],
        label: "24-cell",
    },
];

/// Catalog of named shapes. Both common math-name aliases (the
/// `n-cell` form) and Platonic-slice aliases (the `tetrahedron` /
/// `cube` / etc. form) resolve to the same shape index.
fn parse_shape_name(name: &str) -> Result<ShapeEntry> {
    let n = name.to_lowercase();
    Ok(match n.as_str() {
        "5-cell" | "5cell" | "pentatope" | "pentachoron" | "tetrahedron" => ShapeEntry {
            shape: SHAPE_PENTATOPE,
            color: [0.95, 0.55, 0.30],
            label: "5-cell",
        },
        "8-cell" | "8cell" | "tesseract" | "hypercube" | "cube" => ShapeEntry {
            shape: SHAPE_TESSERACT,
            color: [0.30, 0.55, 0.95],
            label: "8-cell",
        },
        "16-cell" | "16cell" | "hexadecachoron" | "octahedron" => ShapeEntry {
            shape: SHAPE_16CELL,
            color: [0.55, 0.95, 0.40],
            label: "16-cell",
        },
        "24-cell" | "24cell" | "icositetrachoron" | "cuboctahedron" => ShapeEntry {
            shape: SHAPE_24CELL,
            color: [0.95, 0.45, 0.85],
            label: "24-cell",
        },
        "120-cell" | "120cell" | "hecatonicosachoron" | "dodecahedron" => ShapeEntry {
            shape: SHAPE_120CELL,
            color: [0.40, 0.85, 0.85],
            label: "120-cell",
        },
        "600-cell" | "600cell" | "hexacosichoron" | "icosahedron" => ShapeEntry {
            shape: SHAPE_600CELL,
            color: [0.95, 0.85, 0.40],
            label: "600-cell",
        },
        _ => {
            return Err(anyhow!(
                "unknown shape name {name:?}; valid names: 5-cell, \
                 tesseract, 16-cell, 24-cell, 120-cell, 600-cell \
                 (or Platonic aliases: tetrahedron, cube, octahedron, \
                 cuboctahedron, dodecahedron, icosahedron)"
            ))
        }
    })
}

/// Parse the row from CLI arguments. Looks for `--shapes name1 name2 ...`
/// (consumes everything after the flag). Returns `DEFAULT_ROW` if
/// the flag isn't present.
fn parse_row_from_args() -> Result<Vec<ShapeEntry>> {
    let args: Vec<String> = std::env::args().collect();
    let Some(idx) = args.iter().position(|a| a == "--shapes") else {
        return Ok(DEFAULT_ROW.to_vec());
    };
    let names = &args[idx + 1..];
    if names.is_empty() {
        return Err(anyhow!("--shapes flag passed but no shape names followed"));
    }
    names.iter().map(|n| parse_shape_name(n)).collect()
}

fn body_position(slot: usize, n: usize) -> [f32; 4] {
    let x = (slot as f32 - (n as f32 - 1.0) * 0.5) * BODY_X_SPACING;
    [x, BODY_Y, 0.0, 0.0]
}

// ---------------------------------------------------------------------------
// Rotation planes
// ---------------------------------------------------------------------------

/// One of the six elementary 4D rotation planes. Pressing the
/// matching number key toggles that plane's contribution to the
/// per-frame angular-velocity bivector; multiple planes sum.
///
/// Sum-of-bivectors composition is **commutative** (vector space
/// addition), so toggle order doesn't matter, only the active
/// set does. That sidesteps the "cycle mode produces unpredictable
/// results because rotor multiplication is non-commutative"
/// problem of an earlier sequential-composition design.
///
/// The three w-involving planes (xw, yw, zw) pull visible axes
/// into the hidden dimension and produce the slice-shape changes
/// the viewer is built to show. The three pure-3D planes (xy, xz,
/// yz) just rotate the cross-section as a rigid 3D shape; they're
/// included for completeness and because composing them with
/// w-planes produces non-commuting bivector sums whose `exp` does
/// non-trivial things in SO(4).
#[derive(Copy, Clone, PartialEq, Eq)]
enum Plane {
    Xy = 0,
    Xz = 1,
    Xw = 2,
    Yz = 3,
    Yw = 4,
    Zw = 5,
}

const PLANES: [Plane; 6] = [
    Plane::Xy,
    Plane::Xz,
    Plane::Xw,
    Plane::Yz,
    Plane::Yw,
    Plane::Zw,
];

impl Plane {
    fn label(self) -> &'static str {
        match self {
            Plane::Xy => "xy",
            Plane::Xz => "xz",
            Plane::Xw => "xw",
            Plane::Yz => "yz",
            Plane::Yw => "yw",
            Plane::Zw => "zw",
        }
    }

    /// Unit-rate bivector for this plane. Component order matches
    /// `Bivector4::new(xy, xz, xw, yz, yw, zw)`. The single nonzero
    /// component is at index `self as usize`.
    fn unit_bivector(self) -> Bivector4 {
        let mut c = [0.0_f32; 6];
        c[self as usize] = 1.0;
        Bivector4::new(c[0], c[1], c[2], c[3], c[4], c[5])
    }
}

/// Angular velocity from the active set: sum of unit bivectors of
/// active planes, scaled by base rate × rate_scale.
fn angular_velocity(active: &[bool; 6], rate_scale: f32) -> Bivector4 {
    let mut omega = Bivector4::ZERO;
    for (i, &on) in active.iter().enumerate() {
        if on {
            omega = omega + PLANES[i].unit_bivector();
        }
    }
    omega * (BASE_ROTATION_RATE * rate_scale)
}

/// Name a recognizable combination of active planes. Indices match
/// `PLANES`: `0=xy 1=xz 2=xw 3=yz 4=yw 5=zw`. Order-independent,
/// only the active *set* matters.
///
/// Curated entries cover common 4D-geometry classics: single
/// stretches, the three perpendicular-pair isoclinics (the only
/// commuting bivector pairs in 4D, related to left/right Hopf
/// maps), pure-3D rotations, and the famous "all w-planes"
/// composition that drives the cross-section through its main-
/// diagonal extreme.
fn combo_name(active: &[bool; 6]) -> Option<&'static str> {
    // Build a 6-bit mask for compact pattern matching.
    let mut mask = 0u8;
    for (i, &on) in active.iter().enumerate() {
        if on {
            mask |= 1 << i;
        }
    }
    // Bit positions: 0=xy 1=xz 2=xw 3=yz 4=yw 5=zw
    let xy = 1 << 0;
    let xz = 1 << 1;
    let xw = 1 << 2;
    let yz = 1 << 3;
    let yw = 1 << 4;
    let zw = 1 << 5;
    let m = mask;
    Some(match m {
        0 => return None,
        // Single planes, three w-stretchers and three pure-3D rotations.
        x if x == xw => "x-into-w stretch",
        x if x == yw => "y-into-w stretch",
        x if x == zw => "z-into-w stretch",
        x if x == xy => "xy spin (3D only)",
        x if x == xz => "xz spin (3D only)",
        x if x == yz => "yz spin (3D only)",
        // Perpendicular-pair isoclinics, the only commuting
        // bivector pairs in 4D.
        x if x == xw | yz => "isoclinic xw+yz",
        x if x == xz | yw => "isoclinic xz+yw",
        x if x == xy | zw => "isoclinic xy+zw",
        // Pure-3D combos, equivalent to standard 3D rotations.
        x if x == xy | xz | yz => "full 3D spin",
        // The famous "all w-planes", pulls every visible axis
        // into w simultaneously, drives the tesseract through its
        // main-diagonal cross-section (max-volume octahedron).
        x if x == xw | yw | zw => "main-diagonal spin (all-w)",
        // Maximally compound, every plane active.
        x if x == xy | xz | xw | yz | yw | zw => "chaotic SO(4) drift",
        _ => "compound",
    })
}

/// Pack a `Rotor4` into the `[f32; 8]` slot expected by
/// `BodyUniform.rotor`. Component order:
/// `[s, xy, xz, xw, yz, yw, zw, xyzw]`.
fn rotor_to_slot(r: Rotor4) -> [f32; 8] {
    [r.s, r.xy, r.xz, r.xw, r.yz, r.yw, r.zw, r.xyzw]
}

// ---------------------------------------------------------------------------
// Font discovery (portable system-font fallback)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// PolytopeSmokeApp
// ---------------------------------------------------------------------------

struct PolytopeSmokeApp {
    space: EuclideanR3,
    camera: Camera<EuclideanR3>,
    orbit: OrbitController<EuclideanR3>,
    node: Hyperslice4DNode,
    /// Polytope row built at startup from `--shapes` CLI args (or
    /// `DEFAULT_ROW`); drives both the body uniforms and per-body
    /// label lookups in the overlay.
    row: Vec<ShapeEntry>,

    w_slice: f32,
    slider_up_held: bool,
    slider_down_held: bool,

    rotate: bool,
    rot_state: Rotor4,
    /// Toggle bitmap for the six rotation planes; sum of active
    /// planes' unit bivectors becomes the per-frame angular
    /// velocity. See [`PLANES`] for the index -> plane mapping.
    active: [bool; 6],
    rate_scale: f32,
    /// Accumulated time spent rotating (advances only while
    /// `rotate == true`; resets on **R**). Useful for spotting
    /// periodicities in compound-bivector animations.
    rot_time: f32,
}

impl PolytopeSmokeApp {
    /// Drive every body in the row with the same rotor, lets the
    /// user directly compare slice signatures under identical 4D motion.
    fn write_all(&mut self, rotor: [f32; 8]) {
        let n = self.row.len();
        for (slot, entry) in self.row.iter().enumerate() {
            let body = BodyUniform::polytope(
                body_position(slot, n),
                entry.shape,
                BODY_SIZE,
                rotor,
                entry.color,
            );
            self.node.set_body(slot, body);
        }
    }

    /// Full reset: slice, rate, active set, orientation, time
    /// accumulator. Slice resets to the row's Platonic-named
    /// cross-section when applicable (120/600-cell), else to centre.
    fn reset(&mut self) {
        self.w_slice = if self.row.iter().any(|e| e.shape == SHAPE_120CELL) {
            0.926 * BODY_SIZE
        } else if self.row.iter().any(|e| e.shape == SHAPE_600CELL) {
            0.85 * BODY_SIZE
        } else {
            0.0
        };
        self.rate_scale = 1.0;
        self.active = [false; 6];
        self.rot_state = Rotor4::IDENTITY;
        self.rot_time = 0.0;
        self.write_all(IDENTITY_ROTOR);
    }
}

impl App for PolytopeSmokeApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let row = parse_row_from_args()?;
        if row.is_empty() {
            return Err(anyhow!("--shapes produced an empty row"));
        }

        let scene = Scene4::new(SceneNode4::halfspace(Vec4::Y, 0.0));
        // Only pay the ~24 KB polytope-data compile cost if the row
        // actually contains a 120-cell or 600-cell. Otherwise the
        // kernel's dispatch branches for those shapes are unreachable
        // at runtime and the stubs are enough to satisfy naga.
        let needs_extended = row
            .iter()
            .any(|e| e.shape == SHAPE_120CELL || e.shape == SHAPE_600CELL);
        let polytope_wgsl = if needs_extended {
            polytope_extended_sdfs_wgsl()
        } else {
            polytope_stub_sdfs_wgsl().to_owned()
        };
        let shader_source = format!(
            "{kernel}\n{polytope_wgsl}\n{scene}\n",
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

        let n = row.len();
        let bodies: Vec<BodyUniform> = row
            .iter()
            .enumerate()
            .map(|(slot, entry)| {
                BodyUniform::polytope(
                    body_position(slot, n),
                    entry.shape,
                    BODY_SIZE,
                    IDENTITY_ROTOR,
                    entry.color,
                )
            })
            .collect();
        node.set_bodies(&bodies);

        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 3.0, 9.0);
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        // Wider orbit so all four bodies in the row are visible at
        // default zoom; user can scroll-zoom in.
        orbit.set_orbit(9.5, -0.25);

        // Initial w-slice: pick a value that lands on a "Platonic-named"
        // cross-section when one of the H4-symmetric polytopes is in the
        // row. The 120-cell's dodecahedral face cell is centered at
        // w = inradius * size = 0.926 * size; for the 600-cell, the
        // icosahedral cross-section appears near a vertex (w = size).
        // For 5/8/16/24-cell rows, w=0 already gives the Platonic-shaped
        // central slice, so leave the default at 0.
        let initial_w = if row.iter().any(|e| e.shape == SHAPE_120CELL) {
            0.926 * BODY_SIZE
        } else if row.iter().any(|e| e.shape == SHAPE_600CELL) {
            0.85 * BODY_SIZE
        } else {
            0.0
        };

        Ok(Self {
            space: EuclideanR3,
            camera,
            orbit,
            node,
            row,
            w_slice: initial_w,
            slider_up_held: false,
            slider_down_held: false,
            rotate: false,
            rot_state: Rotor4::IDENTITY,
            active: [false; 6],
            rate_scale: 1.0,
            rot_time: 0.0,
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

        // 4D rotation animation. Both bodies share the same rotor
        // so the user can directly compare their slice signatures
        // under identical 4D motion. Rotor accumulates per-frame
        // (delta = exp(ω · dt)) so pause naturally freezes
        // orientation in place, see KeyT handler.
        if self.rotate {
            self.rot_time += dt_secs;
            let omega = angular_velocity(&self.active, self.rate_scale) * dt_secs;
            // No-op when no planes are active; skip the exp+mul.
            if omega.magnitude_squared() > 0.0 {
                let delta = omega.exp();
                self.rot_state = (delta * self.rot_state).normalize();
                self.write_all(rotor_to_slot(self.rot_state));
            }
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
    }

    fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        egui::SidePanel::left("polytope-smoke-controls")
            .resizable(false)
            .default_width(280.0)
            .show(ctx, |ui| {
                ui.heading("polytope smoke");
                ui.label(format!("{:.0} fps", frame.fps));

                ui.separator();
                ui.label("Slice");
                ui.add(
                    egui::Slider::new(&mut self.w_slice, -W_RANGE..=W_RANGE)
                        .text("w")
                        .fixed_decimals(3),
                );
                ui.weak("Up/Down arrow keys also scrub at 0.5 u/s.");

                ui.separator();
                ui.label("Rotation");
                let (status_word, status_color) = if self.rotate {
                    ("spinning", egui::Color32::from_rgb(102, 255, 153))
                } else {
                    ("paused", egui::Color32::from_rgb(242, 217, 76))
                };
                ui.horizontal(|ui| {
                    ui.colored_label(status_color, status_word);
                    if ui
                        .button(if self.rotate { "Pause" } else { "Spin" })
                        .clicked()
                    {
                        self.rotate = !self.rotate;
                    }
                    ui.label(format!("t = {:.2}s", self.rot_time));
                });
                ui.add(
                    egui::Slider::new(&mut self.rate_scale, 0.0..=8.0)
                        .text("rate")
                        .fixed_decimals(2),
                );

                ui.separator();
                ui.label("Planes");
                ui.weak(
                    "Checkbox: include the plane in the continuous-rotation \
                     active set (commutative bivector sum). Buttons: compose \
                     a fixed-angle rotation onto the current orientation via \
                     rotor multiplication, independent of the spin state.",
                );
                let mut compose: Option<(Plane, f32)> = None;
                for (i, plane) in PLANES.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.active[i], plane.label());
                        if ui.small_button("+90°").clicked() {
                            compose = Some((*plane, std::f32::consts::FRAC_PI_2));
                        }
                        if ui.small_button("-90°").clicked() {
                            compose = Some((*plane, -std::f32::consts::FRAC_PI_2));
                        }
                        if ui.small_button("+30°").clicked() {
                            compose = Some((*plane, std::f32::consts::FRAC_PI_6));
                        }
                    });
                }
                if let Some((plane, theta)) = compose {
                    let delta = (plane.unit_bivector() * theta).exp();
                    self.rot_state = (delta * self.rot_state).normalize();
                    self.write_all(rotor_to_slot(self.rot_state));
                }

                if let Some(name) = combo_name(&self.active) {
                    ui.colored_label(egui::Color32::from_rgb(255, 217, 140), name);
                }

                ui.separator();
                if ui.button("Reset orientation + slice").clicked() {
                    self.reset();
                }

                ui.separator();
                ui.label("Row");
                for entry in &self.row {
                    ui.label(format!("• {}", entry.label));
                }

                ui.separator();
                ui.weak("Mouse drag in the viewport orbits the camera.");
            });

        // Bottom-right: window size, handy for screenshots.
        let cfg = &frame.rd.surface_bundle.config;
        let (w, h) = (cfg.width, cfg.height);
        egui::Area::new(egui::Id::new("polytope-smoke-size"))
            .anchor(egui::Align2::RIGHT_BOTTOM, [-16.0, -16.0])
            .show(ctx, |ui| {
                ui.weak(format!("{w}x{h}"));
            });
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
            KeyCode::KeyR if pressed => self.reset(),
            KeyCode::KeyT if pressed => {
                // Pause / resume only, DO NOT touch rot_state. The
                // bodies keep their current orientation when paused
                // and resume from there when toggled back on.
                self.rotate = !self.rotate;
            }
            // Plane toggles. Sum-of-bivectors composition is
            // commutative, so the order in which planes are toggled
            // doesn't affect the resulting motion, only the active
            // set matters.
            KeyCode::Digit1 | KeyCode::Numpad1 if pressed => self.active[0] = !self.active[0],
            KeyCode::Digit2 | KeyCode::Numpad2 if pressed => self.active[1] = !self.active[1],
            KeyCode::Digit3 | KeyCode::Numpad3 if pressed => self.active[2] = !self.active[2],
            KeyCode::Digit4 | KeyCode::Numpad4 if pressed => self.active[3] = !self.active[3],
            KeyCode::Digit5 | KeyCode::Numpad5 if pressed => self.active[4] = !self.active[4],
            KeyCode::Digit6 | KeyCode::Numpad6 if pressed => self.active[5] = !self.active[5],
            // Numpad and main-row +/- both adjust rate by 0.25
            // additively. Floored at 0.0 (effectively pause via
            // rate=0); ceiling at 8.0 keeps the integration stable.
            KeyCode::Equal | KeyCode::NumpadAdd if pressed => {
                self.rate_scale = (self.rate_scale + 0.25).min(8.0);
            }
            KeyCode::Minus | KeyCode::NumpadSubtract if pressed => {
                self.rate_scale = (self.rate_scale - 0.25).max(0.0);
            }
            _ => {}
        }
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        // Hyperslice scene; the rye-app framework paints the egui overlay
        // on top after this returns.
        self.node.execute(rd, view)
    }

    fn title(&self, _fps: f32) -> std::borrow::Cow<'static, str> {
        // Window title is now decorative, all live state is in the
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
