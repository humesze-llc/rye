//! Console command that prints the demo's build identity.

use loam_egui::{cmd, Console};

/// Register the `version` console command for a demo.
///
/// Pass `env!()` strings from the demo's own crate so each demo reports
/// its own name + version + build hash. The hash and dirty fields can be
/// empty if the demo doesn't have a `build.rs` baking those env vars; the
/// output collapses to just the crate name + version.
pub fn register_command<Ctx: 'static>(
    console: &mut Console<Ctx>,
    crate_name: &'static str,
    crate_version: &'static str,
    build_hash: &'static str,
    build_dirty: &'static str,
) {
    console.register(cmd(
        "version",
        "show the demo's crate version + git build hash",
        move |_args, _ctx: &mut Ctx, out| {
            let line = if build_hash.is_empty() {
                format!("{crate_name} v{crate_version}")
            } else {
                format!("{crate_name} v{crate_version} ({build_hash}{build_dirty})")
            };
            out.line(line);
            Ok(())
        },
    ));
}
