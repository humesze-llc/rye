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
}
