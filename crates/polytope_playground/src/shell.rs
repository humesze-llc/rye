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
            label: "toybox",
            build: |ctx| Ok(Box::new(crate::toybox::ToyboxScene::new(ctx)?)),
        },
    ];
}
