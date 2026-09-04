use egui::{
    vec2, Align, Button, CursorIcon, DragValue, Layout, RichText, Slider, SliderClamping, Ui,
};

/// `dragged` is user-on-the-slider this frame; `changed` also covers the popup.
#[derive(Copy, Clone, Debug, Default)]
pub struct SliderInteraction {
    pub changed: bool,
    pub dragged: bool,
}

/// Right-click on the value cell opens a [`DragValue`] popup.
pub fn slider_with_edit(
    ui: &mut Ui,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    formatted: &str,
    edit_suffix: &str,
    edit_decimals: usize,
    value_cell_w: f32,
) -> SliderInteraction {
    let slider_resp = ui.add(
        Slider::new(value, range.clone())
            .show_value(false)
            .smart_aim(false)
            .clamping(SliderClamping::Always),
    );
    let mut popup_changed = false;
    ui.allocate_ui_with_layout(
        vec2(value_cell_w, 14.0),
        Layout::left_to_right(Align::Center),
        |ui| {
            let label_resp = ui.add(
                Button::new(RichText::new(formatted).monospace())
                    .frame(false)
                    .small(),
            );
            label_resp
                .on_hover_cursor(CursorIcon::ContextMenu)
                .on_hover_text("Right-click to edit value")
                .context_menu(|ui| {
                    let drag_resp = ui.add(
                        DragValue::new(value)
                            .range(range)
                            .suffix(edit_suffix)
                            .fixed_decimals(edit_decimals),
                    );
                    if drag_resp.changed() {
                        popup_changed = true;
                    }
                });
        },
    );
    SliderInteraction {
        changed: slider_resp.changed() || popup_changed,
        dragged: slider_resp.dragged(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn screen() -> egui::Rect {
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(800.0, 600.0))
    }

    #[test]
    fn no_input_returns_false_and_preserves_value() {
        let ctx = egui::Context::default();
        let mut value = 0.5_f32;
        let input = egui::RawInput {
            screen_rect: Some(screen()),
            time: Some(0.0),
            ..Default::default()
        };
        let mut interaction = SliderInteraction::default();
        let _ = ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                interaction = slider_with_edit(ui, &mut value, 0.0..=1.0, "0.50", "", 2, 60.0);
            });
        });
        assert!(
            !interaction.changed,
            "no input should not fire a changed event"
        );
        assert!(!interaction.dragged, "no input should not report dragged");
        assert_eq!(value, 0.5);
    }

    #[test]
    fn side_cell_width_is_fixed() {
        let ctx = egui::Context::default();
        let mut total_widths: Vec<f32> = Vec::new();
        let layout_input = egui::RawInput {
            screen_rect: Some(screen()),
            time: Some(0.0),
            ..Default::default()
        };
        for formatted in ["0", "+999.99"] {
            let mut value = 0.5_f32;
            let mut total = 0.0_f32;
            let _ = ctx.run(layout_input.clone(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        let before = ui.next_widget_position().x;
                        slider_with_edit(ui, &mut value, 0.0..=1.0, formatted, "", 2, 60.0);
                        total = ui.next_widget_position().x - before;
                    });
                });
            });
            assert!(total > 60.0, "slider width was not measured: {total}");
            total_widths.push(total);
        }
        let drift = (total_widths[0] - total_widths[1]).abs();
        assert!(
            drift < 1.0,
            "slider total width must be invariant across formatted-string lengths; \
             got {} vs {} (drift {:.2})",
            total_widths[0],
            total_widths[1],
            drift,
        );
    }
}
