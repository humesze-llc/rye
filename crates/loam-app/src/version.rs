use loam_egui::{cmd, Console};

/// Pass `env!()` strings from the demo's own crate. The hash and dirty fields
/// may be empty when the demo has no `build.rs` baking those env vars; the
/// output then collapses to crate name and version.
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
