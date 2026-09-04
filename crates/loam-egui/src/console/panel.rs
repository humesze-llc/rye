use egui::{
    text::{LayoutJob, TextFormat},
    Color32, FontId, Frame, Label, Layout, Margin, Order, RichText, ScrollArea, Sense, Stroke,
    TextEdit, TextWrapMode,
};

use loam_console::{Console, HistoryLine, LineKind};

use super::PANEL_HEIGHT_FRACTION;
use crate::media::dock_chevrons;

const COLOR_BG: Color32 = Color32::from_rgba_premultiplied(12, 12, 16, 230);
const COLOR_INPUT_ECHO: Color32 = Color32::from_rgb(230, 230, 235);
const COLOR_OUTPUT: Color32 = Color32::from_rgb(180, 180, 188);
const COLOR_ERROR: Color32 = Color32::from_rgb(245, 130, 130);
const COLOR_SYSTEM: Color32 = Color32::from_rgb(140, 200, 220);
const COLOR_PROMPT: Color32 = Color32::from_rgb(160, 200, 140);
const COLOR_TITLE: Color32 = Color32::from_rgb(200, 200, 210);
const COLOR_SEPARATOR: Color32 = Color32::from_rgb(60, 60, 70);
const COLOR_GHOST: Color32 = Color32::from_rgb(110, 110, 120);
const FONT_SIZE: f32 = 13.0;
const ROW_TITLE_HEIGHT: f32 = 22.0;
const ROW_INPUT_HEIGHT: f32 = 24.0;

// First detach only; egui remembers the window after that.
const DETACHED_DEFAULT_W: f32 = 520.0;
const DETACHED_DEFAULT_H: f32 = 320.0;

pub(super) fn draw<Ctx: 'static>(console: &mut Console<Ctx>, ctx: &egui::Context, progress: f32) {
    if console.is_detached() {
        draw_detached(console, ctx);
    } else {
        draw_docked(console, ctx, progress);
    }
}

fn draw_docked<Ctx: 'static>(console: &mut Console<Ctx>, ctx: &egui::Context, progress: f32) {
    let viewport = ctx.content_rect();
    let panel_height = (viewport.height() * PANEL_HEIGHT_FRACTION).round();
    let y_offset = -panel_height * (1.0 - progress);
    let width = viewport.width();

    // Un-offset rect: a click during the slide must not defocus.
    let panel_rect = egui::Rect::from_min_size(
        egui::pos2(viewport.min.x, viewport.min.y),
        egui::vec2(width, panel_height),
    );
    let pointer_pressed = ctx.input(|i| i.pointer.any_pressed());
    if pointer_pressed {
        if let Some(pos) = ctx.input(|i| i.pointer.interact_pos()) {
            console.set_user_defocused(!panel_rect.contains(pos));
        }
    }

    egui::Area::new(egui::Id::new("loam_console_area"))
        .order(Order::Foreground)
        .fixed_pos(egui::pos2(viewport.min.x, viewport.min.y + y_offset))
        .show(ctx, |ui| {
            let frame = Frame::default()
                .fill(COLOR_BG)
                .inner_margin(Margin::same(0));
            frame.show(ui, |ui| {
                ui.set_min_size(egui::vec2(width, panel_height));
                ui.set_max_size(egui::vec2(width, panel_height));
                ui.allocate_ui_with_layout(
                    egui::vec2(width, panel_height),
                    Layout::top_down(egui::Align::Min),
                    |ui| {
                        let scroll_h = panel_height - ROW_TITLE_HEIGHT - ROW_INPUT_HEIGHT - 2.0;
                        draw_content(ui, console, width, scroll_h);
                    },
                );
            });
        });
}

fn draw_detached<Ctx: 'static>(console: &mut Console<Ctx>, ctx: &egui::Context) {
    let viewport = ctx.content_rect();
    let default_pos = egui::pos2(
        (viewport.right() - DETACHED_DEFAULT_W - 16.0).max(viewport.left() + 16.0),
        viewport.top() + 80.0,
    );
    let frame = Frame::default()
        .fill(COLOR_BG)
        .stroke(Stroke::new(1.0, COLOR_SEPARATOR))
        .inner_margin(Margin::same(0))
        .corner_radius(egui::CornerRadius::same(4));

    egui::Window::new("loam_console_window")
        .id(egui::Id::new("loam_console_window"))
        .title_bar(false)
        .resizable(true)
        .collapsible(false)
        .movable(true)
        .default_pos(default_pos)
        .default_size(egui::vec2(DETACHED_DEFAULT_W, DETACHED_DEFAULT_H))
        .frame(frame)
        .show(ctx, |ui| {
            // Rows must sum to `available_height` or the window auto-grows each frame.
            ui.spacing_mut().item_spacing.y = 0.0;
            let width = ui.available_width();
            let scroll_h =
                (ui.available_height() - ROW_TITLE_HEIGHT - ROW_INPUT_HEIGHT - 2.0).max(60.0);
            draw_content(ui, console, width, scroll_h);
        });
}

fn draw_content<Ctx: 'static>(
    ui: &mut egui::Ui,
    console: &mut Console<Ctx>,
    width: f32,
    scroll_height: f32,
) {
    draw_title_row(ui, console, width);
    draw_separator(ui, width);
    draw_scrollback(ui, console, scroll_height, width);
    draw_separator(ui, width);
    draw_input_row(ui, console, width);
}

fn draw_title_row<Ctx: 'static>(ui: &mut egui::Ui, console: &mut Console<Ctx>, width: f32) {
    ui.allocate_ui_with_layout(
        egui::vec2(width, ROW_TITLE_HEIGHT),
        Layout::left_to_right(egui::Align::Center),
        |ui| {
            ui.add_space(8.0);
            ui.label(
                RichText::new("console")
                    .color(COLOR_TITLE)
                    .font(FontId::monospace(FONT_SIZE))
                    .strong(),
            );
            ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(8.0);
                let detached = console.is_detached();
                let tip = if detached {
                    "Re-attach as the half-screen drop-down"
                } else {
                    "Detach as a draggable window"
                };
                if dock_chevrons(ui, egui::vec2(12.0, 16.0), detached, tip).clicked() {
                    if detached {
                        console.dock();
                    } else {
                        console.detach();
                    }
                }
                ui.add_space(8.0);
                if !console.status().is_empty() {
                    ui.label(
                        RichText::new(console.status())
                            .color(COLOR_TITLE)
                            .font(FontId::monospace(FONT_SIZE)),
                    );
                }
            });
        },
    );
}

fn draw_separator(ui: &mut egui::Ui, width: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 1.0), Sense::hover());
    ui.painter().hline(
        rect.x_range(),
        rect.center().y,
        Stroke::new(1.0, COLOR_SEPARATOR),
    );
}

fn draw_scrollback<Ctx: 'static>(
    ui: &mut egui::Ui,
    console: &Console<Ctx>,
    height: f32,
    width: f32,
) {
    ui.allocate_ui_with_layout(
        egui::vec2(width, height),
        Layout::top_down(egui::Align::Min),
        |ui| {
            ScrollArea::vertical()
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .max_height(height)
                .show(ui, |ui| {
                    Frame::new()
                        .inner_margin(Margin {
                            left: 8,
                            right: 8,
                            top: 4,
                            bottom: 4,
                        })
                        .show(ui, |ui| {
                            ui.add(
                                Label::new(scrollback_layout_job(console.history()))
                                    .wrap_mode(TextWrapMode::Wrap)
                                    .selectable(true),
                            );
                        });
                });
        },
    );
}

fn line_color(kind: LineKind) -> Color32 {
    match kind {
        LineKind::Input => COLOR_INPUT_ECHO,
        LineKind::Output => COLOR_OUTPUT,
        LineKind::Error => COLOR_ERROR,
        LineKind::System => COLOR_SYSTEM,
    }
}

fn scrollback_layout_job(history: &std::collections::VecDeque<HistoryLine>) -> LayoutJob {
    let mut job = LayoutJob::default();
    let font = FontId::monospace(FONT_SIZE);
    let last = history.len().saturating_sub(1);
    for (i, line) in history.iter().enumerate() {
        let fmt = TextFormat {
            font_id: font.clone(),
            color: line_color(line.kind),
            ..Default::default()
        };
        job.append(&line.text, 0.0, fmt);
        if i < last {
            job.append("\n", 0.0, TextFormat::default());
        }
    }
    job
}

fn draw_input_row<Ctx: 'static>(ui: &mut egui::Ui, console: &mut Console<Ctx>, width: f32) {
    ui.allocate_ui_with_layout(
        egui::vec2(width, ROW_INPUT_HEIGHT),
        Layout::left_to_right(egui::Align::Center),
        |ui| {
            ui.add_space(8.0);
            ui.label(
                RichText::new(">")
                    .color(COLOR_PROMPT)
                    .font(FontId::monospace(FONT_SIZE))
                    .strong(),
            );

            let enter_pressed =
                ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Enter));

            let prev_input = console.input().to_string();
            let ghost = console.tab_preview();
            let output = TextEdit::singleline(console.input_mut())
                .font(FontId::monospace(FONT_SIZE))
                .frame(false)
                .desired_width(width - 32.0)
                .text_color(COLOR_INPUT_ECHO)
                .show(ui);
            let response = output.response;

            if let Some(ghost) = ghost {
                let ghost_pos = output.galley_pos + egui::vec2(output.galley.size().x, 0.0);
                ui.painter().text(
                    ghost_pos,
                    egui::Align2::LEFT_TOP,
                    ghost,
                    FontId::monospace(FONT_SIZE),
                    COLOR_GHOST,
                );
            }

            if console.take_pending_cursor_to_end() {
                let mut state = output.state;
                let end = egui::text::CCursor::new(console.input().chars().count());
                state
                    .cursor
                    .set_char_range(Some(egui::text::CCursorRange::one(end)));
                state.store(ui.ctx(), response.id);
            }

            if console.input() != prev_input {
                console.cancel_tab_cycle();
            }

            // Take the one-shot request before the persistent arm can short-circuit it.
            let opened_this_frame = console.take_pending_focus();
            if opened_this_frame || (console.wants_persistent_focus() && !response.has_focus()) {
                response.request_focus();
            }

            if enter_pressed {
                console.submit();
                response.request_focus();
            }
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    fn history(lines: &[(LineKind, &str)]) -> VecDeque<HistoryLine> {
        lines
            .iter()
            .map(|(kind, text)| HistoryLine {
                kind: *kind,
                text: (*text).to_string(),
            })
            .collect()
    }

    #[test]
    fn empty_history_yields_empty_job() {
        let job = scrollback_layout_job(&VecDeque::new());
        assert!(job.text.is_empty());
        assert!(job.sections.is_empty());
    }

    #[test]
    fn multiple_lines_alternate_with_newline_separators() {
        let h = history(&[
            (LineKind::Input, "> reset"),
            (LineKind::Output, "ok"),
            (LineKind::Error, "boom"),
        ]);
        let job = scrollback_layout_job(&h);
        assert_eq!(job.text, "> reset\nok\nboom");
        assert_eq!(job.sections.len(), 5);
    }

    #[test]
    fn per_line_color_matches_line_kind() {
        let h = history(&[
            (LineKind::Input, "in"),
            (LineKind::Output, "out"),
            (LineKind::Error, "err"),
            (LineKind::System, "sys"),
        ]);
        let job = scrollback_layout_job(&h);
        let input_section = &job.sections[0];
        let output_section = &job.sections[2];
        let error_section = &job.sections[4];
        let system_section = &job.sections[6];
        assert_eq!(input_section.format.color, COLOR_INPUT_ECHO);
        assert_eq!(output_section.format.color, COLOR_OUTPUT);
        assert_eq!(error_section.format.color, COLOR_ERROR);
        assert_eq!(system_section.format.color, COLOR_SYSTEM);
    }
}
