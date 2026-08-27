use std::process::Command;

fn main() {
    let hash = Command::new("git")
        .args(["rev-parse", "--short=8", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "nogit".to_string());
    println!("cargo:rustc-env=BUILD_HASH={hash}");

    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    let dirty_marker = if dirty { "+dirty" } else { "" };
    println!("cargo:rustc-env=BUILD_DIRTY={dirty_marker}");

    // Re-run this build script when HEAD changes or files get staged/unstaged.
    // `.git/HEAD` covers branch + commit moves; `.git/index` covers staging.
    // Note: editing a tracked file without staging it won't re-trigger this
    // script via cargo's usual src-watch (cargo only watches .rs / Cargo.toml
    // by default), so a dirty-but-not-committed change may show stale.
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");
}
