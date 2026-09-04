//! Console keys are consumed via `InputState::consume_key` before `TextEdit` sees them.

mod key;
mod panel;

pub use loam_console::{
    cmd, console_echo_enabled, parse_line, render_line, set_console_echo, subcommands, Command,
    Console, ConsoleWriter, FnCommand, HistoryLine, Key, LineKind, SubcommandSet,
    MAX_HISTORY_LINES, MAX_INPUT_HISTORY,
};

pub const ANIM_DURATION_SECS: f32 = 0.15;

/// Half the viewport height, the Quake convention.
pub const PANEL_HEIGHT_FRACTION: f32 = 0.5;

pub trait ConsoleUi {
    fn ui(&mut self, egui_ctx: &egui::Context);
}

impl<Ctx: 'static> ConsoleUi for Console<Ctx> {
    fn ui(&mut self, egui_ctx: &egui::Context) {
        // A printable key also emits a Text event; strip it or it leaks into the input.
        let toggle = self.toggle_key();
        let toggle_text = key::key_text(toggle);
        let toggle_pressed = egui_ctx.input_mut(|i| {
            let pressed = i.consume_key(egui::Modifiers::NONE, key::to_egui(toggle));
            if pressed {
                if let Some(t) = toggle_text {
                    i.events
                        .retain(|e| !matches!(e, egui::Event::Text(s) if s == t));
                }
            }
            pressed
        });
        if toggle_pressed {
            self.toggle();
        }

        if !self.is_open() {
            let mut fired: Vec<String> = Vec::new();
            egui_ctx.input_mut(|i| {
                for (bound, line) in self.binds() {
                    if i.consume_key(egui::Modifiers::NONE, key::to_egui(bound)) {
                        fired.push(line.to_string());
                    }
                }
            });
            for line in fired {
                self.execute(&line);
            }
        }

        if self.is_open() {
            handle_panel_keys(self, egui_ctx);
        }

        let target = if self.is_open() { 1.0 } else { 0.0 };
        let progress = egui_ctx.animate_value_with_time(
            egui::Id::new("loam_console_open_progress"),
            target,
            ANIM_DURATION_SECS,
        );

        let visible = if self.is_detached() {
            self.is_open()
        } else {
            progress > 0.0
        };
        if visible {
            panel::draw(self, egui_ctx, progress);
        }
    }
}

fn handle_panel_keys<Ctx: 'static>(console: &mut Console<Ctx>, ctx: &egui::Context) {
    if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
        console.close();
    }
    if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp)) {
        console.history_prev();
    }
    if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown)) {
        console.history_next();
    }
    if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Tab)) {
        console.tab_complete();
    }
    if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::L)) {
        console.clear_history();
    }
    if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::C)) {
        console.clear_input();
    }
}
