use loam_app::shell::{SceneEntry, SceneRegistry};

pub(crate) struct Playground;

impl SceneRegistry for Playground {
    const SCENES: &'static [SceneEntry] = &[
        SceneEntry {
            slug: "rotate",
            label: "Rotate polytopes",
            build: |ctx| Ok(Box::new(crate::RotateScene::new(ctx)?)),
        },
        SceneEntry {
            slug: "toybox",
            label: "Toybox: grab and throw in R⁴",
            build: |ctx| Ok(Box::new(crate::toybox::ToyboxScene::new(ctx)?)),
        },
    ];
}
