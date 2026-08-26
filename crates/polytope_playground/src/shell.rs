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
            slug: "s3",
            label: "Polychora on S³",
            build: |ctx| Ok(Box::new(crate::s3::S3Scene::new(ctx)?)),
        },
        SceneEntry {
            slug: "sdf",
            label: "SDF editor",
            build: |ctx| Ok(Box::new(crate::sdf::SdfScene::new(ctx)?)),
        },
        SceneEntry {
            slug: "title",
            label: "Title screen",
            build: |ctx| Ok(Box::new(crate::title::TitleScene::new(ctx)?)),
        },
        SceneEntry {
            slug: "hero",
            label: "Hero: LOAM in the rain",
            build: |ctx| Ok(Box::new(crate::hero::HeroScene::new(ctx)?)),
        },
    ];
}
