//! What egui is consuming, on the two clocks an immediate-mode pass can
//! answer.

/// egui's claim on this frame's input, read once per frame between
/// `egui::Context::begin_pass` and the build.
///
/// The two fields are deliberately not one bool: they are measured at
/// different points in egui's pass and a caller that gates on the wrong one
/// gets a defect that shows up on one input device only. Gate pointer-driven
/// gameplay (orbit, click-to-throw, picking) on [`Self::pointer`]; gate
/// hotkeys on [`Self::keyboard`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UiCapture {
    /// This frame's pointer against the previous build's layout, plus any
    /// press or drag egui already owns. As fresh as immediate mode allows:
    /// `begin_pass` hit-tests the new pointer position against the widget
    /// rects the last build left behind, so a pointer that arrives over a
    /// widget and clicks on the same frame reports here on that frame.
    pub pointer: bool,
    /// Focus as of the end of the previous build, minus a focus this frame's
    /// Escape has already dropped. It cannot be fresher: focus is granted
    /// during the widget pass and resolved in `Focus::end_pass`
    /// (egui 0.33.3 memory/mod.rs:584-599), and the only event `begin_pass`
    /// acts on is Escape (memory/mod.rs:561). The residue is one frame, on
    /// the frame focus changes, for Tab and for click-to-focus; neither can
    /// coincide with the gameplay keystroke this gates, because the focusing
    /// input is a click or a Tab.
    pub keyboard: bool,
}

impl UiCapture {
    /// Read both clocks from `ctx`. Valid only inside a pass, and only
    /// before the build: `begin_pass` refreshes the hit test and the
    /// interaction snapshot this reads, and every widget added afterwards
    /// belongs to the layout the *next* frame will be measured against.
    ///
    /// [`Self::pointer`] deliberately does not use
    /// `egui::Context::wants_pointer_input`. That reads
    /// `PassState::unused_rect` through `is_pointer_over_area`
    /// (egui 0.33.3 context.rs:2753-2779), which `begin_pass` resets to the
    /// whole screen and only the build shrinks, so a pointer resting on a
    /// panel reads as free until that panel is added. It also drops the
    /// hover term outright while a button is held, so a drag that began on
    /// a panel reports nothing for its whole duration. The interaction
    /// snapshot's `contains_pointer` is the same hit test without either
    /// gap: `begin_pass` fills it from the previous build's widget rects
    /// (context.rs:474-495).
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
    const OPEN_SCENE: egui::Pos2 = egui::pos2(200.0, 400.0);
    const WINDOW_POS: egui::Pos2 = egui::pos2(400.0, 300.0);
    const IN_WINDOW: egui::Pos2 = egui::pos2(430.0, 320.0);

    /// A menu bar with a button and a text field, plus a floating window:
    /// the shapes every demo in this workspace actually puts on screen.
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

    /// Drives one runner frame: `begin_pass`, read the capture where the
    /// runner reads it (before `App::update`, so before the build), then
    /// build. Returns that capture alongside the value the pre-`UiCapture`
    /// runners used, which was sampled after the *previous* `end_pass`
    /// exactly as `UiIntegration::paint` sampled it.
    struct Host {
        ctx: egui::Context,
        text: String,
        stale: bool,
    }

    impl Host {
        /// Two warm-up builds before any test input: egui hit-tests against
        /// the previous build's widget rects, and the bool this replaces is
        /// sampled from the pass before that one.
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

        /// `(capture read this frame, the one-frame-stale bool it replaces)`.
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

    /// The gate the runner reorder exists for: a press that lands on a
    /// widget the same frame the pointer arrives must not also reach
    /// gameplay. The bool this replaces cannot report it, because it was
    /// computed before the pointer moved.
    #[test]
    fn pointer_capture_reports_a_press_on_the_frame_the_pointer_arrives() {
        let mut host = Host::new();
        host.hover(OPEN_SCENE);
        let (capture, stale) = host.press(IN_WINDOW);
        assert!(capture.pointer);
        assert!(!stale);
    }

    /// Hover with no button held is capture too: the next scroll or press
    /// belongs to the widget, not the camera.
    #[test]
    fn pointer_capture_covers_a_hovered_panel_without_claiming_the_keyboard() {
        let mut host = Host::new();
        host.hover(OPEN_SCENE);
        let (capture, _) = host.hover(PANEL_BLANK);
        assert!(capture.pointer);
        assert!(!capture.keyboard);
    }

    /// A drag that began on a panel keeps the pointer for its duration.
    /// `wants_pointer_input` drops its hover term while a button is down,
    /// which is why the stale bool goes false mid-drag.
    #[test]
    fn pointer_capture_survives_a_drag_held_over_a_panel() {
        let mut host = Host::new();
        host.hover(PANEL_BLANK);
        host.press(PANEL_BLANK);
        let (capture, stale) = host.hover(egui::pos2(PANEL_BLANK.x + 30.0, PANEL_BLANK.y));
        assert!(capture.pointer);
        assert!(!stale);
    }

    /// Nothing under the pointer, nothing focused: gameplay owns the frame.
    #[test]
    fn capture_is_clear_over_open_scene() {
        let mut host = Host::new();
        host.hover(PANEL_BLANK);
        let (capture, _) = host.hover(OPEN_SCENE);
        assert_eq!(capture, UiCapture::default());
    }

    /// The playground's throw defect: a focused text field must not stop a
    /// click landing in open scene. One bool cannot express this.
    #[test]
    fn a_focused_field_claims_the_keyboard_without_claiming_the_pointer() {
        let mut host = Host::new();
        host.click(egui::pos2(120.0, 10.0));
        let (capture, _) = host.hover(OPEN_SCENE);
        assert!(capture.keyboard);
        assert!(!capture.pointer);
    }

    /// The keyboard clock, pinned as documented: focus is granted during the
    /// build, so the click that focuses a field reports keyboard capture on
    /// the following frame, not its own.
    #[test]
    fn keyboard_capture_trails_the_click_that_focuses_a_field_by_one_build() {
        let mut host = Host::new();
        let (on_click, _) = host.click(egui::pos2(120.0, 10.0));
        assert!(!on_click.keyboard);
        let (next, _) = host.frame(vec![]);
        assert!(next.keyboard);
    }

    /// Escape is the one focus edge `begin_pass` resolves, so the frame that
    /// presses it is already free for gameplay. The stale bool still reads
    /// captured there, which is what made Esc-to-exit need a second press.
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
