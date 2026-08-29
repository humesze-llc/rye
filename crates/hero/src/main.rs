//! The wordmark, on its own. One scene, no scene switcher and no control
//! panel: what is on screen is what gets recorded.
//!
//! The sequence runs 870 ticks, 14.5 s at 60 Hz, and holds on its last frame
//! rather than looping. To record it, capture PNG from the shell's capture
//! panel and encode the sequence:
//!
//! ```text
//! ffmpeg -framerate 60 -i frame%05d.png -c:v libvpx-vp9 \
//!     -crf 30 -b:v 0 -row-mt 1 -pix_fmt yuv420p hero.webm
//! ```
//!
//! `loam-app` writes PNG, GIF and APNG; it has no video encoder and should not
//! grow one for a single artifact.

use anyhow::Result;
use loam_app::shell::{SceneEntry, SceneRegistry};
use loam_app::RunConfig;
use winit::window::WindowAttributes;

mod environment;
mod scene;

struct Hero;

impl SceneRegistry for Hero {
    const SCENES: &'static [SceneEntry] = &[SceneEntry {
        slug: "hero",
        label: "LOAM",
        build: |ctx| Ok(Box::new(scene::HeroScene::new(ctx)?)),
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
