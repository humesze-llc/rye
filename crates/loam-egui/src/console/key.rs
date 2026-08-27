use loam_console::Key;

pub(super) fn to_egui(key: Key) -> egui::Key {
    match key {
        Key::Escape => egui::Key::Escape,
        Key::Tab => egui::Key::Tab,
        Key::Backspace => egui::Key::Backspace,
        Key::Enter => egui::Key::Enter,
        Key::Space => egui::Key::Space,
        Key::Delete => egui::Key::Delete,

        Key::ArrowUp => egui::Key::ArrowUp,
        Key::ArrowDown => egui::Key::ArrowDown,
        Key::ArrowLeft => egui::Key::ArrowLeft,
        Key::ArrowRight => egui::Key::ArrowRight,

        Key::Home => egui::Key::Home,
        Key::End => egui::Key::End,
        Key::PageUp => egui::Key::PageUp,
        Key::PageDown => egui::Key::PageDown,

        Key::Backtick => egui::Key::Backtick,
        Key::Minus => egui::Key::Minus,
        Key::Equals => egui::Key::Equals,

        Key::Num0 => egui::Key::Num0,
        Key::Num1 => egui::Key::Num1,
        Key::Num2 => egui::Key::Num2,
        Key::Num3 => egui::Key::Num3,
        Key::Num4 => egui::Key::Num4,
        Key::Num5 => egui::Key::Num5,
        Key::Num6 => egui::Key::Num6,
        Key::Num7 => egui::Key::Num7,
        Key::Num8 => egui::Key::Num8,
        Key::Num9 => egui::Key::Num9,

        Key::A => egui::Key::A,
        Key::B => egui::Key::B,
        Key::C => egui::Key::C,
        Key::D => egui::Key::D,
        Key::E => egui::Key::E,
        Key::F => egui::Key::F,
        Key::G => egui::Key::G,
        Key::H => egui::Key::H,
        Key::I => egui::Key::I,
        Key::J => egui::Key::J,
        Key::K => egui::Key::K,
        Key::L => egui::Key::L,
        Key::M => egui::Key::M,
        Key::N => egui::Key::N,
        Key::O => egui::Key::O,
        Key::P => egui::Key::P,
        Key::Q => egui::Key::Q,
        Key::R => egui::Key::R,
        Key::S => egui::Key::S,
        Key::T => egui::Key::T,
        Key::U => egui::Key::U,
        Key::V => egui::Key::V,
        Key::W => egui::Key::W,
        Key::X => egui::Key::X,
        Key::Y => egui::Key::Y,
        Key::Z => egui::Key::Z,

        Key::F1 => egui::Key::F1,
        Key::F2 => egui::Key::F2,
        Key::F3 => egui::Key::F3,
        Key::F4 => egui::Key::F4,
        Key::F5 => egui::Key::F5,
        Key::F6 => egui::Key::F6,
        Key::F7 => egui::Key::F7,
        Key::F8 => egui::Key::F8,
        Key::F9 => egui::Key::F9,
        Key::F10 => egui::Key::F10,
        Key::F11 => egui::Key::F11,
        Key::F12 => egui::Key::F12,
    }
}

// Strips the matching `egui::Event::Text` after the Key event is consumed,
// so the toggle char does not leak into the input.
pub(super) fn key_text(key: Key) -> Option<&'static str> {
    match key {
        Key::Backtick => Some("`"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn to_egui_is_injective_over_every_console_key() {
        let mut seen: HashSet<egui::Key> = HashSet::new();
        for key in Key::ALL {
            let mapped = to_egui(*key);
            assert!(
                seen.insert(mapped),
                "{key:?} collides with an earlier key on {mapped:?}"
            );
        }
        assert_eq!(seen.len(), Key::ALL.len());
    }

    #[test]
    fn to_egui_preserves_the_key_name() {
        for key in Key::ALL {
            assert_eq!(
                format!("{key:?}"),
                format!("{:?}", to_egui(*key)),
                "{key:?} maps to a differently named egui key"
            );
        }
    }
}
