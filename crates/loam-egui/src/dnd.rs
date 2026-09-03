use egui::{
    self, emath::TSTransform, vec2, DragAndDrop, Id, LayerId, Order, Rect, Response, Sense, Ui,
    UiBuilder,
};

/// Unlike [`Ui::dnd_drag_source`], the dragged body leaves no slot behind.
pub fn drag_source_collapsing<P>(
    ui: &mut Ui,
    id: Id,
    payload: P,
    body: impl FnOnce(&mut Ui),
) -> Response
where
    P: 'static + Send + Sync,
{
    let ctx = ui.ctx().clone();
    let is_dragged = ctx.is_being_dragged(id);
    if !is_dragged {
        return ui.dnd_drag_source(id, payload, body).response;
    }
    DragAndDrop::set_payload(&ctx, payload);
    let layer_id = LayerId::new(Order::Tooltip, id);
    let mut child = ui.new_child(UiBuilder::new().layer_id(layer_id));
    body(&mut child);
    let body_rect = child.min_rect();
    if let Some(pos) = ctx.pointer_interact_pos() {
        let delta = pos - body_rect.center();
        ctx.transform_layer_shapes(layer_id, TSTransform::from_translation(delta));
    }
    ui.interact(body_rect, id, Sense::hover())
}

/// `true` on a pointer release over the targeted gap.
pub fn make_room_gap(
    ui: &mut Ui,
    is_target: bool,
    slot_id: Id,
    height: f32,
    open_width: f32,
) -> bool {
    let target_w = if is_target { open_width } else { 0.0 };
    let smooth_w = ui.ctx().animate_value_with_time(slot_id, target_w, 0.12);
    if smooth_w >= 0.5 {
        let _ = ui.allocate_exact_size(vec2(smooth_w, height), Sense::hover());
    }
    let dropped = is_target && ui.ctx().input(|i| i.pointer.any_released());
    if dropped {
        let _ = ui.ctx().animate_value_with_time(slot_id, 0.0, 0.0);
    }
    dropped
}

/// `None` when not dragging or more than 40 pt off the row.
pub fn drop_target_idx(
    ctx: &egui::Context,
    is_dragging: bool,
    row_rect: Rect,
    item_count: usize,
) -> Option<usize> {
    if !is_dragging {
        return None;
    }
    let cursor = ctx.input(|i| i.pointer.hover_pos())?;
    let band = row_rect.expand2(vec2(0.0, 40.0));
    if !band.x_range().contains(cursor.x) || !band.y_range().contains(cursor.y) {
        return None;
    }
    let n_slots = item_count + 1;
    let slot_w = (row_rect.width() / n_slots as f32).max(1.0);
    let rel = (cursor.x - row_rect.left()).max(0.0);
    Some(((rel / slot_w) as usize).min(item_count))
}

/// The Tooltip layer never registers hover, so the dragged body dims without this.
pub fn force_opaque_active(ui: &mut Ui) {
    let active = ui.visuals().widgets.active;
    let v = ui.visuals_mut();
    v.widgets.inactive.bg_fill = active.bg_fill;
    v.widgets.inactive.weak_bg_fill = active.weak_bg_fill;
    v.widgets.inactive.fg_stroke = active.fg_stroke;
    v.widgets.inactive.bg_stroke = active.bg_stroke;
    v.widgets.noninteractive.bg_fill = active.bg_fill;
    v.widgets.noninteractive.weak_bg_fill = active.weak_bg_fill;
}

/// In `[0.0, 1.0]`, animated over 120 ms at drag start and drag end.
pub fn pickup_t(ctx: &egui::Context, drag_id: Id) -> f32 {
    let target = if ctx.is_being_dragged(drag_id) {
        1.0
    } else {
        0.0
    };
    ctx.animate_value_with_time(drag_id.with("pickup"), target, 0.12)
}

/// Call before the row's render loop; `gap_id_prefix` and `card_id_prefix`
/// must match the row's `make_persistent_id`s.
pub fn apply_drop_pre_pass<T, P>(
    ui: &mut Ui,
    vec: &mut Vec<T>,
    drop_idx: Option<usize>,
    filter: impl FnOnce(&P) -> Option<usize>,
    gap_id_prefix: &'static str,
    card_id_prefix: &'static str,
    max_count: usize,
) -> bool
where
    P: 'static + Send + Sync,
{
    if !ui.ctx().input(|i| i.pointer.any_released()) {
        return false;
    }
    let Some(to) = drop_idx else {
        return false;
    };
    let Some(arc) = DragAndDrop::payload::<P>(ui.ctx()) else {
        return false;
    };
    let Some(from) = filter(&arc) else {
        return false;
    };
    let _ = DragAndDrop::take_payload::<P>(ui.ctx());
    if from == to || from >= vec.len() {
        return false;
    }
    let item = vec.remove(from);
    let dest = if to > from { to - 1 } else { to };
    vec.insert(dest.min(vec.len()), item);
    let ctx = ui.ctx();
    for i in 0..=max_count {
        let gap_id = ui.make_persistent_id((gap_id_prefix, i));
        let _ = ctx.animate_value_with_time(gap_id, 0.0, 0.0);
        let card_id = ui.make_persistent_id((card_id_prefix, i));
        let _ = ctx.animate_value_with_time(card_id.with("pickup"), 0.0, 0.0);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn screen() -> Rect {
        Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(800.0, 600.0))
    }

    // egui detects a drag only after `time` passes `max_click_duration`.
    fn warmup_input(time: f64) -> egui::RawInput {
        egui::RawInput {
            screen_rect: Some(screen()),
            time: Some(time),
            ..Default::default()
        }
    }

    fn pointer_press(time: f64, pos: egui::Pos2) -> egui::RawInput {
        let mut input = warmup_input(time);
        input.events.push(egui::Event::PointerMoved(pos));
        input.events.push(egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: Default::default(),
        });
        input
    }

    fn pointer_move(time: f64, pos: egui::Pos2) -> egui::RawInput {
        let mut input = warmup_input(time);
        input.events.push(egui::Event::PointerMoved(pos));
        input
    }

    fn pointer_release(time: f64, pos: egui::Pos2) -> egui::RawInput {
        let mut input = warmup_input(time);
        input.events.push(egui::Event::PointerMoved(pos));
        input.events.push(egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: Default::default(),
        });
        input
    }

    #[test]
    fn drag_source_collapsing_starts_drag() {
        let ctx = egui::Context::default();
        let id = Id::new("dnd-test-card");
        let card_pos = egui::pos2(60.0, 30.0);
        let render = |ctx: &egui::Context| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let _ = drag_source_collapsing(ui, id, 42_usize, |ui| {
                    ui.allocate_exact_size(vec2(80.0, 18.0), Sense::hover());
                });
            });
        };
        let _ = ctx.run(warmup_input(0.0), render);
        let _ = ctx.run(pointer_press(0.05, card_pos), render);
        let _ = ctx.run(pointer_move(0.10, card_pos + vec2(20.0, 0.0)), render);
        let _ = ctx.run(pointer_move(0.15, card_pos + vec2(40.0, 0.0)), render);
        assert!(
            ctx.is_being_dragged(id),
            "drag should be active after press + move past threshold"
        );
        assert!(
            DragAndDrop::has_payload_of_type::<usize>(&ctx),
            "drag payload should be set after drag starts"
        );
    }

    #[test]
    fn apply_drop_pre_pass_reorders_vec_on_release() {
        let ctx = egui::Context::default();
        let mut vec = vec!['a', 'b', 'c', 'd'];
        DragAndDrop::set_payload(&ctx, 0_usize);
        let pos = egui::pos2(50.0, 30.0);
        let _ = ctx.run(pointer_release(0.05, pos), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let moved = apply_drop_pre_pass::<char, usize>(
                    ui,
                    &mut vec,
                    Some(3),
                    |p| Some(*p),
                    "test-gap",
                    "test-card",
                    8,
                );
                assert!(
                    moved,
                    "release with valid payload + drop_idx should reorder"
                );
            });
        });
        assert_eq!(vec, vec!['b', 'c', 'a', 'd']);
    }

    #[test]
    fn apply_drop_pre_pass_noop_without_release() {
        let ctx = egui::Context::default();
        let mut vec = vec!['a', 'b', 'c'];
        DragAndDrop::set_payload(&ctx, 0_usize);
        let _ = ctx.run(warmup_input(0.05), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let moved = apply_drop_pre_pass::<char, usize>(
                    ui,
                    &mut vec,
                    Some(2),
                    |p| Some(*p),
                    "test-gap",
                    "test-card",
                    8,
                );
                assert!(!moved);
            });
        });
        assert_eq!(vec, vec!['a', 'b', 'c']);
    }
}
