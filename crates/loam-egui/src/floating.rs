use egui::{Context, Id, Painter, Pos2, Rect, Stroke, Ui, Window};

#[must_use = "FloatingPanelBuilder does nothing until `.show()` is called"]
pub struct FloatingPanelBuilder<'a> {
    ctx: &'a Context,
    id: &'a str,
    title: &'a str,
    open: &'a mut bool,
    resizable: bool,
    collapsible: bool,
    default_size: Option<(f32, f32)>,
    default_width: f32,
    default_pos: Option<Pos2>,
}

impl<'a> FloatingPanelBuilder<'a> {
    pub fn resizable(mut self, on: bool) -> Self {
        self.resizable = on;
        self
    }

    pub fn collapsible(mut self, on: bool) -> Self {
        self.collapsible = on;
        self
    }

    pub fn default_size(mut self, width: f32, height: f32) -> Self {
        self.default_size = Some((width, height));
        self
    }

    pub fn default_width(mut self, width: f32) -> Self {
        self.default_width = width;
        self
    }

    /// First display only.
    pub fn default_pos(mut self, pos: Pos2) -> Self {
        self.default_pos = Some(pos);
        self
    }

    pub fn show<R>(self, content: impl FnOnce(&mut Ui) -> R) -> Option<R> {
        if !*self.open {
            return None;
        }
        let mut local_open = *self.open;
        let mut window = Window::new(self.title)
            .id(Id::new(self.id))
            .open(&mut local_open)
            .collapsible(self.collapsible)
            .resizable(self.resizable);
        if let Some((w, h)) = self.default_size {
            window = window.default_size(egui::vec2(w, h));
        } else {
            window = window.default_width(self.default_width);
        }
        if let Some(pos) = self.default_pos {
            window = window.default_pos(pos);
        }
        let result = window.show(self.ctx, content).and_then(|r| r.inner);
        *self.open = local_open;
        result
    }
}

/// `None` while closed; the close-X clears `open`.
pub fn floating_panel<R>(
    ctx: &Context,
    id: &str,
    title: &str,
    open: &mut bool,
    content: impl FnOnce(&mut Ui) -> R,
) -> Option<R> {
    floating_panel_builder(ctx, id, title, open).show(content)
}

pub fn floating_panel_builder<'a>(
    ctx: &'a Context,
    id: &'a str,
    title: &'a str,
    open: &'a mut bool,
) -> FloatingPanelBuilder<'a> {
    FloatingPanelBuilder {
        ctx,
        id,
        title,
        open,
        resizable: false,
        collapsible: true,
        default_size: None,
        default_width: 260.0,
        default_pos: None,
    }
}

/// Closes only on click-outside or Esc, not on a click inside.
pub fn sticky_menu<R>(
    ui: &mut Ui,
    label: &str,
    add_contents: impl FnOnce(&mut Ui) -> R,
) -> Option<R> {
    let response = ui.button(label);
    egui::Popup::menu(&response)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(add_contents)
        .map(|r| r.inner)
}

#[derive(Clone, Debug)]
pub struct CalloutState {
    pub window_pos: Pos2,
    pub open: bool,
}

impl CalloutState {
    pub fn open_at(window_pos: Pos2) -> Self {
        Self {
            window_pos,
            open: true,
        }
    }
}

/// No-op while `state.open` is false.
pub fn callout(
    ctx: &Context,
    id: &str,
    anchor_screen_pos: Pos2,
    state: &mut CalloutState,
    title: &str,
    content: impl FnOnce(&mut Ui),
) {
    if !state.open {
        return;
    }

    const ANCHOR_RADIUS: f32 = 4.0;
    const LEADER_STROKE: f32 = 1.5;
    const PANEL_DEFAULT_WIDTH: f32 = 220.0;
    let leader_color = ctx.style().visuals.window_fill;
    let anchor_outline = ctx.style().visuals.window_stroke.color;

    // Window first so the leader line can attach to its captured frame rect.
    let mut local_open = state.open;
    let window_response = Window::new(title)
        .id(Id::new(id))
        .open(&mut local_open)
        .collapsible(true)
        .resizable(false)
        .default_width(PANEL_DEFAULT_WIDTH)
        .current_pos(state.window_pos)
        .show(ctx, content);
    state.open = local_open;

    let window_rect: Option<Rect> = window_response.as_ref().map(|r| r.response.rect);
    if let Some(rect) = window_rect {
        state.window_pos = rect.min;
    }

    // Background order: under the window, over the scene.
    let painter_layer = egui::LayerId::new(
        egui::Order::Background,
        Id::new(format!("{id}-callout-overlay")),
    );
    let painter = Painter::new(ctx.clone(), painter_layer, ctx.content_rect());
    if let Some(rect) = window_rect {
        painter.line_segment(
            [rect.center(), anchor_screen_pos],
            Stroke::new(LEADER_STROKE, leader_color),
        );
    }
    painter.circle_filled(anchor_screen_pos, ANCHOR_RADIUS, leader_color);
    painter.circle_stroke(
        anchor_screen_pos,
        ANCHOR_RADIUS + 1.0,
        Stroke::new(1.0, anchor_outline),
    );
}
