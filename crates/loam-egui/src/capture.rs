/// `pointer` and `keyboard` are read at different points in egui's pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UiCapture {
    /// This frame's pointer against the previous build's layout.
    pub pointer: bool,
    /// Focus as of the previous build, minus what this frame's Escape dropped.
    pub keyboard: bool,
}

impl UiCapture {
    /// Valid only inside a pass, before the build: `begin_pass` refreshes the hit
    /// test this reads. Not `wants_pointer_input`, which reads the unbuilt
    /// `unused_rect` and drops hover during a drag (egui 0.33.3 context.rs:2753).
    pub fn read(ctx: &egui::Context) -> Self {
        Self {
            pointer: ctx.is_using_pointer()
                || ctx.interaction_snapshot(|i| !i.contains_pointer.is_empty()),
            keyboard: ctx.wants_keyboard_input(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::UiCapture;

    const SCREEN: egui::Vec2 = egui::vec2(800.0, 600.0);
    const PANEL_BLANK: egui::Pos2 = egui::pos2(400.0, 8.0);
    const FIELD: egui::Pos2 = egui::pos2(120.0, 10.0);
    const OPEN_SCENE: egui::Pos2 = egui::pos2(200.0, 400.0);
    const WINDOW_POS: egui::Pos2 = egui::pos2(400.0, 300.0);
    const IN_WINDOW: egui::Pos2 = egui::pos2(430.0, 320.0);

    fn build(ctx: &egui::Context, text: &mut String) {
        egui::TopBottomPanel::top("bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let _ = ui.button("File");
                ui.add(egui::TextEdit::singleline(text).id(egui::Id::new("edit")));
            });
        });
        egui::Window::new("win")
            .fixed_pos(WINDOW_POS)
            .fixed_size(egui::vec2(100.0, 80.0))
            .show(ctx, |ui| {
                let _ = ui.button("press");
            });
    }

    struct Host {
        ctx: egui::Context,
        text: String,
        stale: bool,
    }

    impl Host {
        // Two warm-ups: hit tests use the previous build, `stale` the one before.
        fn new() -> Self {
            let mut host = Self {
                ctx: egui::Context::default(),
                text: String::new(),
                stale: false,
            };
            host.frame(vec![]);
            host.frame(vec![]);
            host
        }

        fn frame(&mut self, events: Vec<egui::Event>) -> (UiCapture, bool) {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, SCREEN)),
                events,
                ..Default::default()
            };
            let stale = self.stale;
            let mut capture = UiCapture::default();
            let text = &mut self.text;
            let _ = self.ctx.run(input, |ctx| {
                capture = UiCapture::read(ctx);
                build(ctx, text);
            });
            self.stale = self.ctx.wants_pointer_input() || self.ctx.wants_keyboard_input();
            (capture, stale)
        }

        fn hover(&mut self, pos: egui::Pos2) -> (UiCapture, bool) {
            self.frame(vec![egui::Event::PointerMoved(pos)])
        }

        fn press(&mut self, pos: egui::Pos2) -> (UiCapture, bool) {
            self.frame(vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
            ])
        }

        fn click(&mut self, pos: egui::Pos2) -> (UiCapture, bool) {
            self.frame(vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::default(),
                },
            ])
        }

        fn key(&mut self, key: egui::Key) -> (UiCapture, bool) {
            self.frame(vec![egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::default(),
            }])
        }
    }

    #[test]
    fn pointer_capture_reports_a_press_on_the_frame_the_pointer_arrives() {
        let mut host = Host::new();
        host.hover(OPEN_SCENE);
        let (capture, stale) = host.press(IN_WINDOW);
        assert!(capture.pointer);
        assert!(!stale);
    }

    #[test]
    fn pointer_capture_covers_a_hovered_panel_without_claiming_the_keyboard() {
        let mut host = Host::new();
        host.hover(OPEN_SCENE);
        let (capture, _) = host.hover(PANEL_BLANK);
        assert!(capture.pointer);
        assert!(!capture.keyboard);
    }

    #[test]
    fn pointer_capture_survives_a_drag_held_over_a_panel() {
        let mut host = Host::new();
        host.hover(PANEL_BLANK);
        host.press(PANEL_BLANK);
        let (capture, stale) = host.hover(egui::pos2(PANEL_BLANK.x + 30.0, PANEL_BLANK.y));
        assert!(capture.pointer);
        assert!(!stale);
    }

    #[test]
    fn pointer_capture_follows_a_drag_off_the_widget_it_started_on() {
        let mut host = Host::new();
        host.hover(FIELD);
        host.press(FIELD);
        let (capture, _) = host.hover(OPEN_SCENE);
        assert!(capture.pointer);
    }

    #[test]
    fn capture_is_clear_over_open_scene() {
        let mut host = Host::new();
        host.hover(PANEL_BLANK);
        let (capture, _) = host.hover(OPEN_SCENE);
        assert_eq!(capture, UiCapture::default());
    }

    #[test]
    fn a_focused_field_claims_the_keyboard_without_claiming_the_pointer() {
        let mut host = Host::new();
        host.click(egui::pos2(120.0, 10.0));
        let (capture, _) = host.hover(OPEN_SCENE);
        assert!(capture.keyboard);
        assert!(!capture.pointer);
    }

    #[test]
    fn keyboard_capture_trails_the_click_that_focuses_a_field_by_one_build() {
        let mut host = Host::new();
        let (on_click, _) = host.click(egui::pos2(120.0, 10.0));
        assert!(!on_click.keyboard);
        let (next, _) = host.frame(vec![]);
        assert!(next.keyboard);
    }

    #[test]
    fn escape_clears_keyboard_capture_on_the_frame_it_is_pressed() {
        let mut host = Host::new();
        host.click(egui::pos2(120.0, 10.0));
        host.frame(vec![]);
        let (capture, stale) = host.key(egui::Key::Escape);
        assert!(!capture.keyboard);
        assert!(stale);
    }
}
