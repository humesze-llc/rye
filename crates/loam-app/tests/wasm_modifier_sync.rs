//! `crate::wasm` is wasm32-gated, so the reconciler is included by path and
//! driven against the real `InputState` it exists to correct.

#[path = "../src/wasm/modifier_sync.rs"]
mod modifier_sync;

use loam_input::{InputState, Modifiers};
use modifier_sync::{ModifierFlags, ModifierSync};
use winit::event::ElementState;
use winit::keyboard::{KeyCode, PhysicalKey};

fn all_flag_combinations() -> impl Iterator<Item = ModifierFlags> {
    (0u8..16).map(|bits| ModifierFlags {
        ctrl: bits & 1 != 0,
        shift: bits & 2 != 0,
        alt: bits & 4 != 0,
        meta: bits & 8 != 0,
    })
}

fn expected(flags: ModifierFlags) -> Modifiers {
    Modifiers {
        shift: flags.shift,
        control: flags.ctrl,
        alt: flags.alt,
        super_key: flags.meta,
    }
}

// Mirrors the worker's `InputMessage::Key` branch: the browser's flags reconcile
// first, then the physical key lands (`None` for codes `keymap::keycode_winit`
// does not carry).
fn key_event(
    sync: &mut ModifierSync,
    input: &mut InputState,
    code: Option<KeyCode>,
    pressed: bool,
    flags: ModifierFlags,
) {
    let state = |pressed| {
        if pressed {
            ElementState::Pressed
        } else {
            ElementState::Released
        }
    };
    sync.reconcile(flags, |code, pressed| {
        input.key_input(PhysicalKey::Code(code), state(pressed));
    });
    if let Some(code) = code {
        input.key_input(PhysicalKey::Code(code), state(pressed));
    }
}

#[test]
fn frame_input_modifiers_equal_the_browser_flags_after_any_transition() {
    for from in all_flag_combinations() {
        for to in all_flag_combinations() {
            let mut sync = ModifierSync::default();
            let mut input = InputState::default();
            key_event(&mut sync, &mut input, Some(KeyCode::KeyW), true, from);
            assert_eq!(
                input.take_frame().modifiers,
                expected(from),
                "reaching {from:?} from the default state"
            );
            key_event(&mut sync, &mut input, Some(KeyCode::KeyW), false, to);
            assert_eq!(
                input.take_frame().modifiers,
                expected(to),
                "moving from {from:?} to {to:?}"
            );
        }
    }
}

#[test]
fn a_swallowed_keyup_is_released_by_the_next_contradicting_flag() {
    for side in [KeyCode::AltLeft, KeyCode::AltRight] {
        let mut sync = ModifierSync::default();
        let mut input = InputState::default();
        let alt = ModifierFlags {
            alt: true,
            ..ModifierFlags::default()
        };
        key_event(&mut sync, &mut input, Some(side), true, alt);
        assert!(input.take_frame().modifiers.alt);

        // No keyup for `side` ever arrives; the next unrelated key does.
        key_event(
            &mut sync,
            &mut input,
            Some(KeyCode::KeyW),
            true,
            ModifierFlags::default(),
        );
        assert!(
            !input.take_frame().modifiers.alt,
            "{side:?} held with no keyup must clear on the next flags-say-up event"
        );
    }
}

#[test]
fn meta_reaches_super_key_without_a_mapped_code() {
    let mut sync = ModifierSync::default();
    let mut input = InputState::default();
    key_event(
        &mut sync,
        &mut input,
        None,
        true,
        ModifierFlags {
            meta: true,
            ..ModifierFlags::default()
        },
    );
    assert!(input.take_frame().modifiers.super_key);
}

#[test]
fn unchanged_flags_emit_no_transitions() {
    for flags in all_flag_combinations() {
        let mut sync = ModifierSync::default();
        let mut emitted = Vec::new();
        sync.reconcile(flags, |code, pressed| emitted.push((code, pressed)));
        emitted.clear();
        sync.reconcile(flags, |code, pressed| emitted.push((code, pressed)));
        assert!(emitted.is_empty(), "{flags:?} re-emitted {emitted:?}");
    }
}

#[test]
fn a_held_right_hand_modifier_survives_unrelated_key_events() {
    let mut sync = ModifierSync::default();
    let mut input = InputState::default();
    let shift = ModifierFlags {
        shift: true,
        ..ModifierFlags::default()
    };
    key_event(
        &mut sync,
        &mut input,
        Some(KeyCode::ShiftRight),
        true,
        shift,
    );
    key_event(&mut sync, &mut input, Some(KeyCode::KeyW), true, shift);
    assert!(input.take_frame().modifiers.shift);

    key_event(
        &mut sync,
        &mut input,
        Some(KeyCode::ShiftRight),
        false,
        ModifierFlags::default(),
    );
    assert!(!input.take_frame().modifiers.shift);
}
