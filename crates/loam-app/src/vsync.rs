//! The [`crate::fps`] cap alone cannot exceed the display refresh rate while the
//! surface uses `PresentMode::Fifo`: the presentation engine paces the loop at
//! vsync, and that block lands on the swapchain acquire in
//! `RenderDevice::begin_frame`, not on `present`, which only queues the flip.
//! `vsync off` swaps to `Mailbox` (or `Immediate`) so the cap can drive cadence
//! above the display rate.

use loam_egui::{cmd, Console, ConsoleWriter};

use crate::frame_pacing;

/// The runner's verb table ([`crate::command`]) reaches this before any App
/// hook, so the verb's behaviour lives here; [`register_command`] exists so
/// `help` and tab completion know the name, and so a console driven without a
/// loam-app runner still works.
pub(crate) fn apply(args: &[&str], out: &mut ConsoleWriter) {
    match args.first().copied() {
        None => {
            // Without access to RenderDevice from the handler we can only report
            // what was last requested.
            out.line("vsync: use 'vsync on' (Fifo) or 'vsync off' (Mailbox/Immediate)");
        }
        Some("on") => {
            frame_pacing::request_vsync_on();
            out.line("vsync: requested ON (Fifo); applies on next frame");
        }
        Some("off") => {
            frame_pacing::request_vsync_off();
            out.line(
                "vsync: requested OFF (Mailbox preferred, Immediate fallback) \
                 ; applies on next frame",
            );
        }
        Some(other) => {
            out.line(format!(
                "vsync: unknown subcommand '{other}' (try 'on' or 'off')"
            ));
        }
    }
}

pub fn register_command<Ctx: 'static>(console: &mut Console<Ctx>) {
    console.register(
        cmd(
            "vsync",
            "show or set the surface present mode (on = Fifo, off = Mailbox/Immediate)",
            |args, _ctx: &mut Ctx, out| {
                apply(args, out);
                Ok(())
            },
        )
        .with_args(&[&["on", "off"]]),
    );
}

#[cfg(test)]
mod tests {
    use super::apply;
    use crate::frame_pacing;
    use crate::frame_pacing::TEST_LOCK;
    use loam_egui::ConsoleWriter;

    // The verb's one body, which both the runner's table and the console
    // registration reach.
    fn run(args: &[&str]) -> Option<bool> {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = frame_pacing::take_pending_vsync();
        let mut out = ConsoleWriter::new();
        apply(args, &mut out);
        frame_pacing::take_pending_vsync()
    }

    #[test]
    fn vsync_on_records_pending_request() {
        assert_eq!(
            run(&["on"]),
            Some(true),
            "vsync on should queue a vsync-on request"
        );
    }

    #[test]
    fn vsync_off_records_pending_request() {
        assert_eq!(
            run(&["off"]),
            Some(false),
            "vsync off should queue a vsync-off request"
        );
    }

    #[test]
    fn vsync_bare_invocation_does_not_change_pending() {
        assert_eq!(
            run(&[]),
            None,
            "bare `vsync` should print help, not request a transition"
        );
    }

    #[test]
    fn vsync_unknown_subcommand_does_not_change_pending() {
        assert_eq!(
            run(&["foo"]),
            None,
            "unknown subcommand should print error, not queue a transition"
        );
    }
}
