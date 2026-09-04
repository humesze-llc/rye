use loam_egui::{cmd, Console};

/// Empty hash and dirty strings collapse the line to crate name and version.
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
