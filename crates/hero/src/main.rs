//! The wordmark, on its own. One scene, no scene switcher and no control
//! panel: what is on screen is what gets recorded.
//!
//! The sequence runs 870 ticks, 14.5 s at 60 Hz, and holds on its last frame
//! rather than looping. `--record` plays it once into `hero.apng` and closes
//! the window on the last frame; `--record=<dir>` names the directory, and
//! without it the file lands in the shell's capture directory. The run is real
//! time, not offline, so the recording is only as steady as the frame rate the
//! machine holds.
//!
//! APNG is lossless and needs no external encoder, but this writer stores whole
//! frames rather than deltas, so the file is the sum of its frames: 53 MB for
//! this sequence at 800x600. `RECORD_FPS` and `scale` on the capture request
//! are the dials if that has to come down.

use std::path::PathBuf;

use anyhow::{bail, Result};
use loam_app::args::Args;
use loam_app::shell::{SceneEntry, SceneRegistry};
use loam_app::RunConfig;
use winit::window::WindowAttributes;

mod scene;

use scene::RecordRequest;

/// Bare `--record` is the whole flag, so unlike a path-carrying flag it has no
/// syntax to get wrong: the directory is optional and defaults to the shell's.
fn record_request(args: &Args) -> Result<Option<RecordRequest>> {
    let request = if args.has_bare_flag("record") {
        Some(RecordRequest { dir: None })
    } else {
        args.get("record").map(|dir| RecordRequest {
            dir: Some(PathBuf::from(dir)),
        })
    };
    // `capture_stub` accepts every request and writes nothing, so without this
    // the flag plays the whole sequence and exits empty-handed.
    if request.is_some() && !cfg!(feature = "capture") {
        bail!("--record needs the `capture` feature, and this build has it off");
    }
    Ok(request)
}

struct Hero;

impl SceneRegistry for Hero {
    const SCENES: &'static [SceneEntry] = &[SceneEntry {
        slug: "hero",
        label: "LOAM",
        build: |ctx| {
            Ok(Box::new(scene::HeroScene::new(
                ctx,
                record_request(&Args::current())?,
            )?))
        },
    }];
}

fn main() -> Result<()> {
    loam_app::run::<loam_app::shell::SceneShell<Hero>>(RunConfig {
        window: WindowAttributes::default()
            .with_title("loam")
            .with_visible(false),
        ..RunConfig::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_takes_a_directory_or_the_shell_default_and_is_off_otherwise() {
        let ask = |argv: [&str; 2]| record_request(&Args::from_argv(argv)).expect("capture is on");

        assert!(ask(["--seed=7", "--fps=60"]).is_none());
        assert_eq!(
            ask(["--record", "--seed=7"]).expect("bare records").dir,
            None
        );
        assert_eq!(
            ask(["--record=out/hero", "--seed=7"])
                .expect("named records")
                .dir,
            Some(PathBuf::from("out/hero"))
        );

        // The value of a detached `--record out/hero` is a positional the arg
        // parser drops, so the run would record somewhere the caller did not
        // name. Recording to the default is the right answer only because the
        // directory is optional; a required path would have to error here.
        assert_eq!(
            ask(["--record", "out/hero"]).expect("detached records").dir,
            None
        );
    }
}
