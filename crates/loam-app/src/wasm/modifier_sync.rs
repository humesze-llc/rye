//! Two ways `loam_input::InputState`'s held-key-derived modifier set goes wrong
//! on the browser path: the OS can swallow a modifier keyup (Alt+Tab, Cmd+Tab)
//! and leave it stuck down, and `keymap::keycode_winit` has no entry for
//! `MetaLeft` / `MetaRight`, so Cmd/Win never reaches the held set at all.

use winit::keyboard::KeyCode;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModifierFlags {
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
    pub meta: bool,
}

/// The modifier state the browser last reported, held so only the divergences
/// produce synthetic key transitions.
#[derive(Debug, Default)]
pub struct ModifierSync {
    applied: ModifierFlags,
}

impl ModifierSync {
    /// A flag going false releases both sides of its pair: the flag says neither
    /// is down, and the keyup for one side is exactly what may have been
    /// swallowed.
    pub fn reconcile(&mut self, flags: ModifierFlags, mut emit: impl FnMut(KeyCode, bool)) {
        let pairs = [
            (
                flags.ctrl,
                self.applied.ctrl,
                KeyCode::ControlLeft,
                KeyCode::ControlRight,
            ),
            (
                flags.shift,
                self.applied.shift,
                KeyCode::ShiftLeft,
                KeyCode::ShiftRight,
            ),
            (
                flags.alt,
                self.applied.alt,
                KeyCode::AltLeft,
                KeyCode::AltRight,
            ),
            (
                flags.meta,
                self.applied.meta,
                KeyCode::SuperLeft,
                KeyCode::SuperRight,
            ),
        ];
        for (now, before, left, right) in pairs {
            match (before, now) {
                (false, true) => emit(left, true),
                (true, false) => {
                    emit(left, false);
                    emit(right, false);
                }
                _ => {}
            }
        }
        self.applied = flags;
    }
}
