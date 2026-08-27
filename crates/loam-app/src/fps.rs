use loam_egui::{cmd, Console};

use crate::frame_pacing;

// 1000 fps (1 ms period) is well past any practical refresh rate; above it a
// stray `fps 999999` would silently make the cap a no-op.
const MAX_ACCEPTED_FPS: f32 = 1000.0;

fn print_current(out: &mut loam_egui::ConsoleWriter) {
    let f = frame_pacing::target_fps();
    if f <= 0.0 {
        out.line("fps: unlimited (uncapped; surface/vsync or browser RAF is the upper bound)");
    } else {
        out.line(format!("fps: target {f:.1}"));
    }
}

pub fn register_command<Ctx: 'static>(console: &mut Console<Ctx>) {
    console.register(
        cmd(
            "fps",
            "show or set the target framerate (default 60; use 'unlimited' to remove the cap)",
            |args, _ctx: &mut Ctx, out| {
                match args.first().copied() {
                    None => print_current(out),
                    Some("unlimited") | Some("off") | Some("0") => {
                        frame_pacing::set_target_fps(0.0);
                        out.line("fps: unlimited (cap removed)");
                    }
                    Some(s) => match s.parse::<f32>() {
                        Ok(f) if f > 0.0 && f <= MAX_ACCEPTED_FPS => {
                            frame_pacing::set_target_fps(f);
                            out.line(format!("fps: target set to {f:.1}"));
                        }
                        _ => {
                            out.line(format!(
                                "usage: fps [<n> | unlimited]  (n in (0, {MAX_ACCEPTED_FPS:.0}])"
                            ));
                        }
                    },
                }
                Ok(())
            },
        )
        .with_args(&[&["unlimited", "off", "30", "60", "120", "144", "240"]]),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_pacing;
    // Tests touch the process-global `frame_pacing` atomics, so they share
    // `frame_pacing::TEST_LOCK` against cargo's parallel runner.
    use crate::frame_pacing::TEST_LOCK;

    fn build_console() -> loam_egui::Console<()> {
        let mut c = loam_egui::Console::<()>::new();
        register_command(&mut c);
        c
    }

    #[test]
    fn fps_numeric_arg_updates_target() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut c = build_console();
        let mut ctx = ();
        crate::command::run_on_console(&mut c, "fps 30", &mut ctx);
        let target = frame_pacing::target_fps();
        assert!(
            (target - 30.0).abs() < 0.1,
            "fps 30 should set target near 30, got {target}"
        );
        frame_pacing::set_target_fps(60.0);
    }

    #[test]
    fn fps_unlimited_removes_cap() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut c = build_console();
        let mut ctx = ();
        crate::command::run_on_console(&mut c, "fps unlimited", &mut ctx);
        assert_eq!(frame_pacing::target_fps(), 0.0);
        assert!(frame_pacing::target_period().is_none());
        frame_pacing::set_target_fps(60.0);
    }

    #[test]
    fn fps_off_alias_matches_unlimited() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut c = build_console();
        let mut ctx = ();
        crate::command::run_on_console(&mut c, "fps off", &mut ctx);
        assert_eq!(frame_pacing::target_fps(), 0.0);
        frame_pacing::set_target_fps(60.0);
    }

    #[test]
    fn fps_zero_alias_matches_unlimited() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut c = build_console();
        let mut ctx = ();
        crate::command::run_on_console(&mut c, "fps 0", &mut ctx);
        assert_eq!(frame_pacing::target_fps(), 0.0);
        frame_pacing::set_target_fps(60.0);
    }

    #[test]
    fn fps_out_of_range_does_not_change_target() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut c = build_console();
        let mut ctx = ();
        let before = frame_pacing::target_fps();
        crate::command::run_on_console(&mut c, "fps 100000", &mut ctx);
        let after = frame_pacing::target_fps();
        assert_eq!(before, after, "out-of-range input should be a no-op");
        frame_pacing::set_target_fps(60.0);
    }
}
