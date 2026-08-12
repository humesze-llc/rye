//! The demo's scene registry. Hosting lives in [`loam_app::shell`]: menu bar,
//! `--scene=` / `?scene=` boot selection, `--embed=1` / `?embed=1`, lazy build,
//! and the `scene` console command. This file only names the scenes.

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Both `--scene=` and the `scene` command resolve a slug to its first
    /// match, so a duplicate makes the later entry unreachable; an empty table
    /// has no index 0 for the shell to boot.
    #[test]
    fn every_slug_is_claimed_by_exactly_one_entry() {
        assert!(!Playground::SCENES.is_empty(), "index 0 is the boot scene");
        for (i, entry) in Playground::SCENES.iter().enumerate() {
            assert!(
                !Playground::SCENES[..i]
                    .iter()
                    .any(|prior| prior.slug == entry.slug),
                "slug '{}' is registered twice",
                entry.slug
            );
        }
    }

    /// The curved-space scene is reachable by slug and is not index 0: both
    /// `--scene=` / `?scene=` and the `scene` command resolve by position in
    /// this table, and index 0 doubles as the unknown-slug fallback, so a
    /// scene registered there would answer even to a name nothing claims.
    #[test]
    fn the_s3_scene_is_registered_by_slug_away_from_the_fallback_index() {
        let index = Playground::SCENES
            .iter()
            .position(|entry| entry.slug == "s3")
            .expect("`?scene=s3` must resolve");
        assert_ne!(index, 0);
        assert_eq!(Playground::SCENES[0].slug, "rotate", "boot scene unchanged");
    }

    /// The hero scene answers to `--scene=hero` / `?scene=hero` and does not
    /// sit at index 0: it is a scripted sequence with one seed and no
    /// controls, so a cold boot must not land on it.
    #[test]
    fn the_hero_scene_is_registered_by_slug_away_from_the_fallback_index() {
        let index = Playground::SCENES
            .iter()
            .position(|entry| entry.slug == "hero")
            .expect("`?scene=hero` must resolve");
        assert_ne!(index, 0);
    }

    /// The editor answers to `--scene=sdf` / `?scene=sdf` and does not sit at
    /// index 0, which doubles as the unknown-slug fallback: a cold boot should
    /// land on the demo, not on an authoring tool.
    #[test]
    fn the_sdf_editor_is_registered_by_slug_away_from_the_fallback_index() {
        let index = Playground::SCENES
            .iter()
            .position(|entry| entry.slug == "sdf")
            .expect("`?scene=sdf` must resolve");
        assert_ne!(index, 0);
    }

    /// The title screen answers to `--scene=title` / `?scene=title`. Promoting
    /// it to index 0, where a cold boot lands, is the maintainer's call and not
    /// a scene registration detail, so this pins reachability and leaves the
    /// boot scene where the test above asserts it is.
    #[test]
    fn the_title_screen_is_registered_by_slug() {
        assert!(Playground::SCENES.iter().any(|entry| entry.slug == "title"));
    }
}
