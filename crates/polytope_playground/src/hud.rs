//! Owned by the scene, not by the shell. The pre-shell version hung off the
//! `App` because it had to draw after the demo's own passes; under
//! `loam_app::shell` a scene's `record` is already the last thing before egui
//! paints, so that ordering no longer picks an owner. What does: every value
//! the readout formats is `Demo` state, and the shell hosts scenes whose state
//! it knows nothing about, so a shell-owned HUD would need a `Scene` hook with
//! one implementor plus an atlas and pipeline every other scene pays for.

use std::fmt::Write as _;

use anyhow::Result;
use loam_app::{egui, RenderCtx};
use loam_render::device::RenderDevice;
use loam_text::TextRenderer;

use crate::state::Demo;

/// Draw size, in egui points. Scaled to physical pixels by the frame's
/// pixels-per-point, so the readout keeps its apparent size across displays.
const HUD_SIZE_PT: f32 = 16.0;
/// Atlas rasterization size, in physical pixels: four times the draw size, so
/// the quads still minify at a 4x scale factor. loam-text has no mip chain,
/// and magnification is the visibly worse direction.
const HUD_BAKE_PX: f32 = 4.0 * HUD_SIZE_PT;
/// Inset from the top-left corner of the region the shell's panels leave free.
const HUD_INSET_PT: f32 = 16.0;
const HUD_COLOR: [f32; 4] = [0.92, 0.96, 1.0, 1.0];
/// Drop shadow, offset by [`HUD_SHADOW_OFFSET_PT`]. The readout sits over
/// arbitrary raymarched color, so a single flat text color is not legible on
/// its own.
const HUD_SHADOW_COLOR: [f32; 4] = [0.0, 0.0, 0.0, 0.7];
const HUD_SHADOW_OFFSET_PT: f32 = 1.0;

/// Plane order matches `Plane4::ALL`: xy, xz, xw, yz, yw, zw.
const PLANE_NAMES: [&str; 6] = ["xy", "xz", "xw", "yz", "yw", "zw"];
/// Placeholder for an inactive plane, same width as a plane name so the strip
/// keeps a fixed column layout.
const PLANE_OFF: &str = "..";

/// Live values the readout formats. Copied out of `Demo` so the formatter is
/// testable without a GPU-backed demo.
struct Readout {
    w_slice: f32,
    rot_time: f32,
    rate_scale: f32,
    /// Slot the rotation controls are aimed at, and the row length it sits
    /// in. The plane strip below is that slot's mask, not the row's, so the
    /// readout has to name whose planes it is showing.
    selected: usize,
    slots: usize,
    active: [bool; 6],
}

impl Readout {
    fn from_demo(demo: &Demo) -> Self {
        Self {
            w_slice: demo.w_slice,
            rot_time: demo.rot_time,
            rate_scale: demo.rate_scale,
            selected: demo.selected_slot(),
            slots: demo.render_row().len(),
            active: demo.spins.selected_spin().active,
        }
    }
}

/// Format the readout into `out`, which is cleared first.
///
/// Values are right-aligned in a fixed-width column. loam-text lays out on
/// advance widths only, so column alignment is available exclusively through a
/// monospace face plus padded formatting.
fn write_readout(out: &mut String, r: &Readout) {
    out.clear();
    let _ = writeln!(out, "{:<6} {:>+8.3}", "w", r.w_slice);
    let _ = writeln!(out, "{:<6} {:>7.2}s", "t", r.rot_time);
    let _ = writeln!(out, "{:<6} {:>7.2}x", "rate", r.rate_scale);
    let _ = writeln!(
        out,
        "{:<6} {:>7}",
        "body",
        format!("{}/{}", r.selected, r.slots)
    );
    let _ = write!(out, "{:<6} ", "planes");
    for (i, name) in PLANE_NAMES.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(if r.active[i] { name } else { PLANE_OFF });
    }
}

/// Where the readout sits for one frame, in physical pixels.
///
/// loam-text positions and sizes in physical pixels and has no scale-factor
/// notion, so the point-to-pixel conversion happens here. A fixed pixel inset
/// (what the pre-shell version used) seats the block under the menu bar on any
/// display whose scale factor exceeds one, because the bar's height is in
/// points and the inset was not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HudSeat {
    origin_px: [f32; 2],
    pixels_per_point: f32,
}

impl Default for HudSeat {
    /// The seat a frame that never ran `ui` would draw at: inset from the
    /// viewport corner with no chrome to clear.
    fn default() -> Self {
        Self {
            origin_px: [HUD_INSET_PT, HUD_INSET_PT],
            pixels_per_point: 1.0,
        }
    }
}

impl HudSeat {
    fn size_px(&self) -> f32 {
        HUD_SIZE_PT * self.pixels_per_point
    }
}

/// The readout's top-left in egui points: inset from `free`, the region egui's
/// panels leave (`Context::available_rect`), so the shell's menu bar is cleared
/// by construction rather than by a guessed offset.
fn hud_origin(free: egui::Rect) -> egui::Pos2 {
    free.left_top() + egui::vec2(HUD_INSET_PT, HUD_INSET_PT)
}

/// Convert [`hud_origin`] into the physical pixels loam-text takes.
pub(crate) fn hud_seat(free: egui::Rect, pixels_per_point: f32) -> HudSeat {
    let origin = hud_origin(free);
    HudSeat {
        origin_px: [origin.x * pixels_per_point, origin.y * pixels_per_point],
        pixels_per_point,
    }
}

/// The block the readout occupies, in egui points, for comparison against the
/// chrome's rects. Measurement is linear in size, so measuring at
/// [`HUD_SIZE_PT`] gives points directly whatever the display's scale factor.
///
/// Nothing in the draw path needs the extent, only the origin, so this exists
/// to state the claim the chrome-clearance test checks. It shares
/// [`hud_origin`] with the draw path rather than restating the inset.
///
/// The advance box, not the ink box; see `loam_text::TextMetrics::measure`.
#[cfg(test)]
fn hud_rect(free: egui::Rect, metrics: &loam_text::TextMetrics, readout: &str) -> egui::Rect {
    let [w, h] = metrics.measure(readout, HUD_SIZE_PT);
    egui::Rect::from_min_size(hud_origin(free), egui::vec2(w, h))
}

/// One queued string. loam-text has no draw-call concept, so the readout's
/// draw list is the ordered set of `queue` calls a frame makes; every entry
/// carries the same formatted string.
#[derive(Clone, Copy, Debug, PartialEq)]
struct HudDraw {
    origin_px: [f32; 2],
    size_px: f32,
    color: [f32; 4],
}

/// Shadow first so the body paints over it.
fn draw_list(seat: HudSeat) -> [HudDraw; 2] {
    let offset = HUD_SHADOW_OFFSET_PT * seat.pixels_per_point;
    [
        HudDraw {
            origin_px: [seat.origin_px[0] + offset, seat.origin_px[1] + offset],
            size_px: seat.size_px(),
            color: HUD_SHADOW_COLOR,
        },
        HudDraw {
            origin_px: seat.origin_px,
            size_px: seat.size_px(),
            color: HUD_COLOR,
        },
    ]
}

pub(crate) struct TextHud {
    text: TextRenderer,
    /// Rebuilt every frame with `clear` + `write!`, keeping the allocation.
    line_buf: String,
}

impl TextHud {
    /// Bake the atlas and build the pipeline against the device's current
    /// target format and sample count.
    pub(crate) fn new(rd: &RenderDevice) -> Result<Self> {
        let text = TextRenderer::new(
            &rd.device,
            &rd.queue,
            rd.target_format(),
            hud_font_bytes(),
            HUD_BAKE_PX,
            rd.sample_count(),
        )?;
        Ok(Self {
            text,
            line_buf: String::new(),
        })
    }

    /// Format and record the readout into the frame's encoder. No-op while
    /// `Demo::show_text_hud` is off.
    ///
    /// Recorded, not submitted: a nested submit would reach the GPU before the
    /// scene passes already sitting in this encoder, painting the readout under
    /// the scene instead of over it.
    pub(crate) fn record(&mut self, ctx: &mut RenderCtx<'_>, demo: &Demo, seat: HudSeat) {
        if !demo.show_text_hud {
            return;
        }
        let Self { text, line_buf } = self;
        write_readout(line_buf, &Readout::from_demo(demo));
        for draw in draw_list(seat) {
            text.queue(line_buf, draw.origin_px, draw.size_px, draw.color);
        }
        let cfg = &ctx.rd.surface_bundle.config;
        text.record(
            &ctx.rd.device,
            &ctx.rd.queue,
            ctx.encoder,
            ctx.view,
            [cfg.width as f32, cfg.height as f32],
        );
    }
}

/// Hack Regular, shipped as raw bytes by the `epaint_default_fonts` asset
/// crate. Chosen over committing a TTF: the file is already in the dependency
/// tree (egui bundles it) and monospace is what the aligned columns need. The
/// crate contains no egui code, so this does not re-couple the first-party text
/// path to egui.
fn hud_font_bytes() -> &'static [u8] {
    epaint_default_fonts::HACK_REGULAR
}

#[cfg(test)]
mod tests {
    use super::*;

    fn readout(w_slice: f32, rot_time: f32, rate_scale: f32, active: [bool; 6]) -> Readout {
        Readout {
            w_slice,
            rot_time,
            rate_scale,
            selected: 7,
            slots: 8,
            active,
        }
    }

    /// The widest the readout ever gets: every plane named, and value columns
    /// at full width. Chrome clearance has to hold for this one, not a typical
    /// one.
    fn widest_readout() -> String {
        let mut out = String::new();
        write_readout(&mut out, &readout(-9.999, -99.99, -9.99, [true; 6]));
        out
    }

    fn hud_metrics() -> loam_text::TextMetrics {
        loam_text::TextMetrics::new(hud_font_bytes(), HUD_BAKE_PX)
            .expect("bundled Hack Regular parses")
    }

    #[test]
    fn readout_is_renderable_for_every_extreme_float() {
        let mut out = String::new();
        for &value in &[
            0.0_f32,
            -0.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
            -1.0e-30,
        ] {
            write_readout(&mut out, &readout(value, value, value, [true; 6]));
            assert!(
                loam_text::is_renderable(&out),
                "readout for {value} contains characters loam-text would drop: {out:?}"
            );
        }
    }

    #[test]
    fn readout_value_columns_align_across_magnitudes() {
        let mut out = String::new();
        let mut widths: Option<Vec<usize>> = None;
        for &(w, t, rate) in &[
            (0.0_f32, 0.0_f32, 1.0_f32),
            (-9.999, 99.99, 0.25),
            (9.999, 9.99, 4.0),
        ] {
            write_readout(&mut out, &readout(w, t, rate, [false; 6]));
            let line_widths: Vec<usize> = out.lines().map(|l| l.chars().count()).collect();
            match &widths {
                None => widths = Some(line_widths),
                Some(first) => assert_eq!(
                    first, &line_widths,
                    "line widths drifted for (w={w}, t={t}, rate={rate}): {out:?}"
                ),
            }
        }
    }

    #[test]
    fn readout_line_count_is_fixed() {
        let mut out = String::new();
        write_readout(&mut out, &readout(0.0, 0.0, 1.0, [false; 6]));
        assert_eq!(out.lines().count(), 5);
        write_readout(&mut out, &readout(-1.0, 123.0, 4.0, [true; 6]));
        assert_eq!(out.lines().count(), 5);
    }

    #[test]
    fn plane_strip_names_exactly_the_active_planes() {
        let mut out = String::new();
        let mut active = [false; 6];
        active[2] = true;
        active[3] = true;
        write_readout(&mut out, &readout(0.0, 0.0, 1.0, active));
        let strip = out.lines().last().expect("planes line").to_string();
        assert!(strip.ends_with(".. .. xw yz .. .."), "strip was {strip:?}");

        write_readout(&mut out, &readout(0.0, 0.0, 1.0, [false; 6]));
        let off = out.lines().last().expect("planes line");
        assert_eq!(off.chars().count(), strip.chars().count());
    }

    #[test]
    fn reshelled_draw_list_matches_the_pre_shell_readout_at_unit_scale() {
        let free = egui::Rect::from_min_max(egui::pos2(0.0, 24.0), egui::pos2(1280.0, 720.0));
        assert_eq!(
            draw_list(hud_seat(free, 1.0)),
            [
                HudDraw {
                    origin_px: [17.0, 41.0],
                    size_px: 16.0,
                    color: [0.0, 0.0, 0.0, 0.7],
                },
                HudDraw {
                    origin_px: [16.0, 40.0],
                    size_px: 16.0,
                    color: [0.92, 0.96, 1.0, 1.0],
                },
            ]
        );
    }

    #[test]
    fn draw_list_scales_the_whole_placement_by_pixels_per_point() {
        let free = egui::Rect::from_min_max(egui::pos2(0.0, 24.0), egui::pos2(1280.0, 720.0));
        let unit = draw_list(hud_seat(free, 1.0));
        for ppp in [1.25_f32, 1.5, 2.0, 3.0] {
            let scaled = draw_list(hud_seat(free, ppp));
            for (u, s) in unit.iter().zip(scaled.iter()) {
                assert!(
                    (s.origin_px[0] - u.origin_px[0] * ppp).abs() < 1e-3
                        && (s.origin_px[1] - u.origin_px[1] * ppp).abs() < 1e-3,
                    "at {ppp}x the origin was {:?}, expected {:?} scaled",
                    s.origin_px,
                    u.origin_px
                );
                assert!((s.size_px - u.size_px * ppp).abs() < 1e-3, "size at {ppp}x");
                assert_eq!(s.color, u.color, "scale must not touch color");
            }
        }
    }

    #[test]
    fn hud_rect_top_left_is_the_body_origin_in_points() {
        let metrics = hud_metrics();
        let readout = widest_readout();
        let free = egui::Rect::from_min_max(egui::pos2(3.0, 27.0), egui::pos2(1280.0, 720.0));
        for ppp in [1.0_f32, 1.5, 2.0] {
            let body = draw_list(hud_seat(free, ppp))[1];
            let rect = hud_rect(free, &metrics, &readout);
            assert!(
                (body.origin_px[0] - rect.left() * ppp).abs() < 1e-3
                    && (body.origin_px[1] - rect.top() * ppp).abs() < 1e-3,
                "at {ppp}x the body sits at {:?}, rect top-left is {:?}",
                body.origin_px,
                rect.left_top()
            );
        }
    }

    #[test]
    fn hud_rect_clears_the_menu_bar_and_the_bottom_overlay() {
        /// Share of the viewport height the overlay stand-in claims. The real
        /// controls overlay auto-sizes to well under this even expanded, so
        /// clearing the stand-in clears the real one.
        const OVERLAY_PROBE_FRACTION: f32 = 0.5;

        let metrics = hud_metrics();
        let readout = widest_readout();
        for (w, h) in [
            (640.0_f32, 480.0_f32),
            (800.0, 600.0),
            (1280.0, 720.0),
            (1920.0, 1080.0),
        ] {
            let ctx = egui::Context::default();
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(w, h),
                )),
                ..Default::default()
            };
            let mut rects = None;
            let _ = ctx.run(input, |ctx| {
                let bar = egui::TopBottomPanel::top("shell-menu-bar")
                    .show(ctx, |ui| {
                        egui::MenuBar::new().ui(ui, |ui| {
                            ui.menu_button("Demo", |_| {});
                        });
                    })
                    .response
                    .rect;
                let overlay = egui::Window::new("overlay-probe")
                    .title_bar(false)
                    .resizable(false)
                    .collapsible(false)
                    .pivot(egui::Align2::CENTER_BOTTOM)
                    .default_pos(crate::ui::overlay_seat(ctx))
                    .show(ctx, |ui| {
                        ui.set_min_height(h * OVERLAY_PROBE_FRACTION);
                        ui.label("controls");
                    })
                    .expect("probe window is never collapsed")
                    .response
                    .rect;
                rects = Some((
                    bar,
                    overlay,
                    hud_rect(ctx.available_rect(), &metrics, &readout),
                ));
            });
            let (bar, overlay, hud) = rects.expect("run closure fills the rects");
            // Degenerate chrome would make the disjointness below vacuous.
            assert!(bar.height() > 0.0, "{w}x{h}: menu bar measured empty");
            assert!(
                overlay.height() >= h * OVERLAY_PROBE_FRACTION,
                "{w}x{h}: overlay probe measured {overlay:?}, shorter than requested"
            );
            assert!(hud.area() > 0.0, "{w}x{h}: readout measured empty");
            assert!(
                !hud.intersects(bar),
                "{w}x{h}: readout {hud:?} overlaps the menu bar {bar:?}"
            );
            assert!(
                !hud.intersects(overlay),
                "{w}x{h}: readout {hud:?} overlaps the controls overlay {overlay:?}"
            );
        }
    }
}
