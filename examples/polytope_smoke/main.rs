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
    raymarch::{
        polytope_extended_sdfs_wgsl, BodyUniform, Hyperslice4DNode, HYPERSLICE_KERNEL_WGSL,
        SHAPE_120CELL, SHAPE_16CELL, SHAPE_24CELL, SHAPE_600CELL, SHAPE_PENTATOPE, SHAPE_TESSERACT,
    },
    Viewport,
};
use rye_sdf::{Scene4, SceneNode4};
use winit::window::WindowAttributes;

/// Cap on shapes per row from the runtime "Add" buttons. Keeps the
/// scene visible without scroll-zoom and bounds the per-frame body
/// loop. The CLI `--shapes` argument can still spawn up to
/// `MAX_BODIES` (32) at startup.
const MAX_ROW_LEN: usize = 8;

/// Width in pixels of the egui side panel; used to compute the
/// scene viewport so the polytope render fills only the area to
/// the right of the panel. Must match the `default_width` passed
/// to `egui::SidePanel::left` below.
const PANEL_WIDTH: u32 = 300;

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
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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

    /// Whether the egui control panel is visible. When false the
    /// scene fills the full window and a tiny "Show controls" button
    /// floats top-left. Toggle with **H** or the panel's "Hide"
    /// button.
    show_panel: bool,

    /// Sequence of [`RotorTerm`]s the user is building in the panel.
    /// Apply composes them onto `rot_state` left-to-right via rotor
    /// multiplication. Singletons multiply their own rotor; groups
    /// sum their bivectors first then exp once.
    seq: Vec<RotorTerm>,
}

/// One term in the rotor-composition sequence.
#[derive(Clone, Debug)]
enum RotorTerm {
    /// Singleton: one plane and an angle in radians.
    /// Composed as `(plane.bivector * angle).exp()`.
    Single { plane: Plane, angle: f32 },
    /// Additive group: two or more (plane, angle) entries summed
    /// as bivectors before the single `exp()`. Order within the
    /// group does not matter (bivector addition is commutative).
    Group(Vec<(Plane, f32)>),
}

impl RotorTerm {
    /// Compose this term as a delta rotor.
    fn delta(&self) -> Rotor4 {
        match self {
            RotorTerm::Single { plane, angle } => (plane.unit_bivector() * *angle).exp(),
            RotorTerm::Group(entries) => {
                let mut sum = Bivector4::ZERO;
                for (plane, angle) in entries {
                    sum = sum + plane.unit_bivector() * *angle;
                }
                sum.exp()
            }
        }
    }
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

    /// Body of the egui control panel, rendered inside a vertical
    /// scroll area so a long panel doesn't clip on small windows.
    fn render_panel_body(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Slice");
        ui.add(
            egui::Slider::new(&mut self.w_slice, -W_RANGE..=W_RANGE)
                .text("w")
                .fixed_decimals(3),
        );

        ui.separator();
        ui.label("Rotation (continuous)");
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
        // Active-set checkboxes: which planes contribute to the
        // angular velocity while spinning. Two columns of three.
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                for (active, plane) in self.active[..3].iter_mut().zip(&PLANES[..3]) {
                    ui.checkbox(active, plane.label());
                }
            });
            ui.vertical(|ui| {
                for (active, plane) in self.active[3..].iter_mut().zip(&PLANES[3..]) {
                    ui.checkbox(active, plane.label());
                }
            });
        });
        if let Some(name) = combo_name(&self.active) {
            ui.colored_label(egui::Color32::from_rgb(255, 217, 140), name);
        }

        ui.separator();
        ui.label("Compose (one-shot)");
        // Add buttons sit at the top of the section so the user
        // builds a sequence by appending downward.
        ui.horizontal_wrapped(|ui| {
            for plane in PLANES.iter() {
                if ui
                    .small_button(format!("+{}", plane.label()))
                    .on_hover_text("Append a 90° term to the sequence")
                    .clicked()
                {
                    self.seq.push(RotorTerm::Single {
                        plane: *plane,
                        angle: std::f32::consts::FRAC_PI_2,
                    });
                }
            }
        });

        // Per-term row: drag handle, plane(s) + angle(s), Group/Ungroup, X.
        let mut moves: Vec<(usize, usize)> = Vec::new();
        let mut remove_term: Option<usize> = None;
        let mut group_with_prev: Option<usize> = None;
        let mut ungroup: Option<usize> = None;
        for (i, term) in self.seq.iter_mut().enumerate() {
            let drag_id = ui.make_persistent_id(("seq-term", i));
            ui.horizontal(|ui| {
                let drag_resp = ui.dnd_drag_source(drag_id, i, |ui| {
                    ui.label(egui::RichText::new("⠿").size(16.0).strong());
                });
                drag_resp
                    .response
                    .on_hover_cursor(egui::CursorIcon::Grab)
                    .on_hover_text("Drag to reorder");
                match term {
                    RotorTerm::Single { plane, angle } => {
                        ui.monospace(plane.label());
                        let mut deg = angle.to_degrees();
                        if ui
                            .add(
                                egui::DragValue::new(&mut deg)
                                    .suffix("°")
                                    .speed(1.0)
                                    .range(-360.0..=360.0),
                            )
                            .on_hover_text("Drag to scrub; double-click to type")
                            .changed()
                        {
                            *angle = deg.to_radians();
                        }
                        if i > 0
                            && ui
                                .small_button("(+)")
                                .on_hover_text("Group with previous (additive bivector sum)")
                                .clicked()
                        {
                            group_with_prev = Some(i);
                        }
                    }
                    RotorTerm::Group(entries) => {
                        ui.monospace("(");
                        for (k, (plane, angle)) in entries.iter_mut().enumerate() {
                            if k > 0 {
                                ui.monospace("+");
                            }
                            ui.monospace(plane.label());
                            let mut deg = angle.to_degrees();
                            if ui
                                .add(
                                    egui::DragValue::new(&mut deg)
                                        .suffix("°")
                                        .speed(1.0)
                                        .range(-360.0..=360.0),
                                )
                                .changed()
                            {
                                *angle = deg.to_radians();
                            }
                        }
                        ui.monospace(")");
                        if ui
                            .small_button("ungrp")
                            .on_hover_text("Split back into singletons")
                            .clicked()
                        {
                            ungroup = Some(i);
                        }
                    }
                }
                if ui.small_button("X").clicked() {
                    remove_term = Some(i);
                }
            });
            // Drop zone after each term.
            let (_, payload) =
                ui.dnd_drop_zone::<usize, ()>(egui::Frame::default().inner_margin(1.0), |ui| {
                    ui.allocate_space(egui::vec2(ui.available_width(), 3.0));
                });
            if let Some(src) = payload {
                let from = *src;
                let to = if from < i { i } else { i + 1 };
                if from != to && from != to.saturating_sub(1) {
                    moves.push((from, to));
                }
            }
        }

        if let Some(i) = group_with_prev {
            let cur = self.seq.remove(i);
            let prev = self.seq.remove(i - 1);
            let mut entries = Vec::new();
            match prev {
                RotorTerm::Single { plane, angle } => entries.push((plane, angle)),
                RotorTerm::Group(items) => entries.extend(items),
            }
            match cur {
                RotorTerm::Single { plane, angle } => entries.push((plane, angle)),
                RotorTerm::Group(items) => entries.extend(items),
            }
            self.seq.insert(i - 1, RotorTerm::Group(entries));
        }
        if let Some(i) = ungroup {
            if let RotorTerm::Group(entries) = self.seq.remove(i) {
                for (k, (plane, angle)) in entries.into_iter().enumerate() {
                    self.seq.insert(i + k, RotorTerm::Single { plane, angle });
                }
            }
        }
        if let Some(i) = remove_term {
            self.seq.remove(i);
        }
        for (from, to) in moves {
            if from < self.seq.len() {
                let item = self.seq.remove(from);
                let dest = if to > from { to - 1 } else { to };
                self.seq.insert(dest.min(self.seq.len()), item);
            }
        }

        ui.horizontal(|ui| {
            let apply = ui
                .add_enabled(!self.seq.is_empty(), egui::Button::new("Apply"))
                .clicked();
            if apply {
                let terms = self.seq.clone();
                for term in &terms {
                    self.rot_state = (term.delta() * self.rot_state).normalize();
                }
                self.write_all(rotor_to_slot(self.rot_state));
            }
            if ui.button("Clear").clicked() {
                self.seq.clear();
            }
        });

        ui.separator();
        ui.label("Shapes (drag to reorder)");
        let has_heavy = self
            .row
            .iter()
            .any(|e| e.shape == SHAPE_120CELL || e.shape == SHAPE_600CELL);
        if has_heavy {
            ui.colored_label(
                egui::Color32::from_rgb(242, 130, 70),
                "120/600-cell SDFs are heavy; expect <60 fps.",
            );
        }
        // Add buttons FIRST so the user enters a shape and sees it
        // appear at the right end of the cards row immediately.
        let mut row_changed = false;
        if self.row.len() < MAX_ROW_LEN {
            ui.horizontal_wrapped(|ui| {
                ui.label("Add:");
                for shape_name in [
                    "5-cell", "8-cell", "16-cell", "24-cell", "120-cell", "600-cell",
                ] {
                    if ui.small_button(shape_name).clicked() {
                        if let Ok(entry) = parse_shape_name(shape_name) {
                            self.row.push(entry);
                            row_changed = true;
                        }
                    }
                }
            });
        }

        let mut remove_idx: Option<usize> = None;
        let mut shape_moves: Vec<(usize, usize)> = Vec::new();
        let row_len = self.row.len();
        ui.horizontal_wrapped(|ui| {
            for (i, entry) in self.row.iter().enumerate() {
                let drag_id = ui.make_persistent_id(("shape-card", i));
                let drag_resp = ui.dnd_drag_source(drag_id, i, |ui| {
                    ui.group(|ui| {
                        ui.vertical(|ui| {
                            ui.label(entry.label);
                            if row_len > 1 && ui.small_button("X").clicked() {
                                remove_idx = Some(i);
                            }
                        });
                    });
                });
                drag_resp.response.on_hover_cursor(egui::CursorIcon::Grab);
                let (_, payload) =
                    ui.dnd_drop_zone::<usize, ()>(egui::Frame::default().inner_margin(2.0), |ui| {
                        ui.allocate_space(egui::vec2(4.0, 30.0));
                    });
                if let Some(src) = payload {
                    let from = *src;
                    let to = if from < i { i } else { i + 1 };
                    if from != to && from != to.saturating_sub(1) {
                        shape_moves.push((from, to));
                    }
                }
            }
        });
        if let Some(i) = remove_idx {
            self.row.remove(i);
            row_changed = true;
        }
        for (from, to) in shape_moves {
            if from < self.row.len() {
                let item = self.row.remove(from);
                let dest = if to > from { to - 1 } else { to };
                self.row.insert(dest.min(self.row.len()), item);
                row_changed = true;
            }
        }
        if row_changed {
            self.rebuild_bodies();
        }
    }

    /// Rebuild the full body uniform array from the current row +
    /// rotor. Use this when the row's length or order changes; the
    /// per-body position depends on the row's `n` and the body's slot
    /// index, so a single body update is not enough.
    fn rebuild_bodies(&mut self) {
        let n = self.row.len();
        let rotor = rotor_to_slot(self.rot_state);
        let bodies: Vec<BodyUniform> = self
            .row
            .iter()
            .enumerate()
            .map(|(slot, entry)| {
                BodyUniform::polytope(
                    body_position(slot, n),
                    entry.shape,
                    BODY_SIZE,
                    rotor,
                    entry.color,
                )
            })
            .collect();
        self.node.set_bodies(&bodies);
    }

    /// Full reset: slice, rate, active set, orientation, time
    /// accumulator. Slice resets to centre regardless of row.
    fn reset(&mut self) {
        self.w_slice = 0.0;
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
        // Always include the extended polytope WGSL so any of the six
        // shapes can be added to the row at runtime via the panel.
        // The ~24 KB const-array cost is fixed per app and acceptable
        // for a viz/demo target.
        let shader_source = format!(
            "{kernel}\n{polytope}\n{scene}\n",
            kernel = HYPERSLICE_KERNEL_WGSL,
            polytope = polytope_extended_sdfs_wgsl(),
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

        // Always start at w=0 regardless of row contents. Auto-shifting
        // to the 120/600-cell's "Platonic-named" cross-section was
        // confusing in mixed rows: the other shapes' slices got pulled
        // off-centre. Users who want the dodecahedral / icosahedral
        // view scrub there with the slider.
        let initial_w = 0.0;

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
            show_panel: true,
            seq: Vec::new(),
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

        // Camera. Gate the orbit on `!ui_has_focus` so dragging the
        // egui w-slice slider doesn't also rotate the camera.
        use rye_camera::CameraController;
        if !ctx.ui_has_focus {
            self.orbit
                .advance(ctx.input, &mut self.camera, &EuclideanR3, 0.0);
        }
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
        if self.show_panel {
            egui::SidePanel::left("polytope-smoke-controls")
                .resizable(false)
                .default_width(PANEL_WIDTH as f32)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("polytope smoke");
                        // Right-aligned cluster: ?-circle and < collapse.
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .add(egui::Button::new("<").min_size(egui::vec2(22.0, 22.0)))
                                .on_hover_text("Hide controls panel (H)")
                                .clicked()
                            {
                                self.show_panel = false;
                            }
                            ui.add(
                                egui::Button::new(egui::RichText::new("?").strong())
                                    .min_size(egui::vec2(22.0, 22.0)),
                            )
                            .on_hover_ui(|ui| {
                                ui.label("Up / Down: scrub w-slice");
                                ui.label("T: toggle spin");
                                ui.label("R: reset");
                                ui.label("1..6: toggle plane in active set (xy xz xw yz yw zw)");
                                ui.label("+ / -: rate scale");
                                ui.label("H: hide / show this panel");
                                ui.label("Esc: exit");
                                ui.label("Mouse drag in viewport: orbit camera");
                            });
                        });
                    });
                    ui.label(format!("{:.0} fps", frame.fps));

                    egui::ScrollArea::vertical()
                        .auto_shrink([false; 2])
                        .show(ui, |ui| self.render_panel_body(ui));
                });
        } else {
            egui::Area::new(egui::Id::new("polytope-smoke-show"))
                .anchor(egui::Align2::LEFT_TOP, [8.0, 8.0])
                .show(ctx, |ui| {
                    if ui
                        .add(egui::Button::new(">").min_size(egui::vec2(24.0, 24.0)))
                        .on_hover_text("Show controls panel (H)")
                        .clicked()
                    {
                        self.show_panel = true;
                    }
                });
        }

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
            KeyCode::KeyH if pressed => self.show_panel = !self.show_panel,
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
        // Hyperslice scene rendered into the area not covered by the
        // egui panel. `u.resolution` is the viewport size and
        // `u.viewport_origin` is its top-left in framebuffer pixels;
        // together they let the kernel compute the correct UV inside
        // the carved-out region. The egui overlay paints on top
        // after this returns.
        let cfg = &rd.surface_bundle.config;
        let viewport = if self.show_panel {
            Viewport::right_of_left_panel(PANEL_WIDTH, [cfg.width, cfg.height])
        } else {
            Viewport::full([cfg.width, cfg.height])
        };
        {
            let u = self.node.uniforms_mut();
            u.resolution = viewport.resolution_f32();
            u.viewport_origin = [viewport.x as f32, viewport.y as f32];
        }
        self.node.flush_uniforms(&rd.queue);
        self.node.execute_in_viewport(rd, view, viewport)
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
