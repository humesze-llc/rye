//! The wordmark, on its own. One scene, no scene switcher and no control
//! panel: what is on screen is what gets recorded.

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
