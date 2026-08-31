//! The console vocabulary scenes share, and the lens that lets them.
//!
//! A shared verb is written once against its own state struct; a scene opts in
//! by handing over a `fn(&mut Ctx) -> &mut State` that says where it keeps
//! that state. Same shape as `loam_app::environment::register_ground_command`,
//! generalized. The verbs here are the ones every 4D scene can honour, and
//! they are the surface the scripting layer will bind to, so a verb that means
//! subtly different things in two scenes stays out.

use crate::projections::{apply_projection_selection_defaults, WireframeProjection};
use anyhow::anyhow;
use loam_egui::SubcommandSet;

/// Edge thickness a scene starts at, in pixels.
pub(crate) const DEFAULT_WIREFRAME_WIDTH_PX: f32 = 1.8;

/// Above this the lines read as ribbons rather than a wireframe.
const MAX_WIREFRAME_WIDTH_PX: f32 = 16.0;

/// The wireframe state every 4D scene has. Scene-specific knobs (rotate's
/// per-edge activity gradient, its hyperslice cull) stay on the scene and
/// chain onto [`wireframe_subcommands`] rather than living here: a scene that
/// cannot honour a knob should not answer a verb for it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct WireframeControls {
    pub(crate) enabled: bool,
    pub(crate) width_px: f32,
    pub(crate) alpha: f32,
    pub(crate) projection: WireframeProjection,
}

impl Default for WireframeControls {
    fn default() -> Self {
        Self {
            enabled: false,
            width_px: DEFAULT_WIREFRAME_WIDTH_PX,
            alpha: 1.0,
            projection: WireframeProjection::default(),
        }
    }
}

/// The `wireframe` verb, less whatever the scene adds. Returned rather than
/// registered so a scene can chain its own subcommands on before registering:
/// the set is keyed by name, so a chained entry of the same name replaces the
/// shared one.
pub(crate) fn wireframe_subcommands<Ctx: 'static>(
    reach: fn(&mut Ctx) -> &mut WireframeControls,
) -> SubcommandSet<Ctx> {
    loam_egui::subcommands::<Ctx>("wireframe", "4D hull wireframe overlay")
        .with_long_help(
            "Draws every edge of every posed 4D hull, with each vertex shaded by how far it sits from the current slice. The cross-section alone cannot tell a body turning in a w plane from one sliding; the whole hull can.",
        )
        .on_bare(move |ctx| {
            let w = reach(ctx);
            w.enabled = !w.enabled;
            Ok(())
        })
        .custom(
            "width",
            "edge thickness in pixels (bare reads)",
            &[&[]],
            &[],
            move |ctx, args, out| {
                let w = reach(ctx);
                match args.first().copied() {
                    None => out.line(format!("wireframe width: {:.2} px", w.width_px)),
                    Some(token) => {
                        let px: f32 = token
                            .parse()
                            .map_err(|e| anyhow!("invalid width `{token}`: {e}"))?;
                        if !(px > 0.0 && px <= MAX_WIREFRAME_WIDTH_PX) {
                            return Err(anyhow!(
                                "wireframe width {px} out of range; expected (0, {MAX_WIREFRAME_WIDTH_PX}]"
                            ));
                        }
                        w.width_px = px;
                        out.line(format!("wireframe width: set to {px:.2} px"));
                    }
                }
                Ok(())
            },
        )
        .custom(
            "alpha",
            "uniform edge alpha (bare reads)",
            &[&[]],
            &[],
            move |ctx, args, out| {
                let w = reach(ctx);
                match args.first().copied() {
                    None => out.line(format!("wireframe alpha: {:.3}", w.alpha)),
                    Some(token) => {
                        let a: f32 = token
                            .parse()
                            .map_err(|e| anyhow!("invalid alpha `{token}`: {e}"))?;
                        if !(a > 0.0 && a <= 1.0) {
                            return Err(anyhow!(
                                "wireframe alpha {a} out of range; expected (0, 1]"
                            ));
                        }
                        w.alpha = a;
                        out.line(format!("wireframe alpha: set to {a:.3}"));
                    }
                }
                Ok(())
            },
        )
        .custom(
            "perspective",
            "4D->R³ projection (bare cycles): shadow | w-pinhole | stereographic | hyperslice",
            &[&WireframeProjection::TOKENS],
            &[],
            move |ctx, args, out| {
                let w = reach(ctx);
                w.projection = match args.first().copied() {
                    None => {
                        let all = WireframeProjection::ALL;
                        let i = all
                            .iter()
                            .position(|p| p.same_variant(w.projection))
                            .unwrap_or(0);
                        all[(i + 1) % all.len()]
                    }
                    Some(token) => WireframeProjection::from_token(token).ok_or_else(|| {
                        anyhow!(
                            "unknown projection `{token}` (try {})",
                            WireframeProjection::TOKENS.join("|")
                        )
                    })?,
                };
                apply_projection_selection_defaults(w.projection, &mut w.enabled);
                out.line(format!(
                    "wireframe perspective: {}",
                    w.projection.label().to_lowercase()
                ));
                Ok(())
            },
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_egui::Console;

    #[derive(Default)]
    struct Scene {
        wireframe: WireframeControls,
        untouched: u32,
    }

    fn console() -> Console<Scene> {
        let mut c = Console::<Scene>::new();
        c.register(wireframe_subcommands::<Scene>(|s| &mut s.wireframe));
        c
    }

    // The regression the whole module exists to prevent, stated as a list: a
    // scene that grows its own copy of one of these, or simply forgets to
    // register it, drifts from every other scene's console for no reason a
    // viewer can see.
    #[test]
    fn every_scene_console_carries_the_shared_vocabulary() {
        const SHARED: [&str; 12] = [
            "camera", "capture", "floor", "fps", "ground", "log", "restart", "scene", "trace",
            "version", "vsync", "wireframe",
        ];

        let rotate = crate::RotateScene::build_console();
        let mut toybox = Console::<crate::toybox::ToyboxControls>::new();
        crate::toybox::register_toybox_commands(&mut toybox);

        for verb in SHARED {
            assert!(rotate.has_command(verb), "rotate lost `{verb}`");
            assert!(toybox.has_command(verb), "the toybox never registered `{verb}`");
        }
    }

    #[test]
    fn the_lens_writes_through_to_the_state_the_scene_named() {
        let (mut c, mut scene) = (console(), Scene::default());
        c.dispatch("wireframe", &[], &mut scene);
        c.dispatch("wireframe", &["width", "4.5"], &mut scene);
        c.dispatch("wireframe", &["alpha", "0.25"], &mut scene);
        c.dispatch("wireframe", &["perspective", "w-pinhole"], &mut scene);
        assert_eq!(
            scene.wireframe,
            WireframeControls {
                enabled: true,
                width_px: 4.5,
                alpha: 0.25,
                projection: WireframeProjection::WPinhole,
            }
        );
        assert_eq!(
            scene.untouched, 0,
            "the lens reached past the state it names"
        );
    }

    #[test]
    fn out_of_range_width_and_alpha_leave_the_state_alone() {
        let (mut c, mut scene) = (console(), Scene::default());
        for args in [
            ["width", "0"].as_slice(),
            &["width", "17"],
            &["alpha", "0"],
            &["alpha", "1.5"],
            &["width", "wide"],
        ] {
            c.dispatch("wireframe", args, &mut scene);
        }
        assert_eq!(scene.wireframe, WireframeControls::default());
    }

    // The regression this module exists to prevent: a scene growing its own
    // `wireframe` that answers a different set of subcommands from every
    // other scene's.
    #[test]
    fn every_scene_carrying_wireframe_answers_the_same_subcommands() {
        fn table<Ctx: 'static>(console: &mut Console<Ctx>, ctx: &mut Ctx) -> Vec<String> {
            console.dispatch("help", &["wireframe"], ctx);
            let subs = console
                .history()
                .iter()
                .map(|line| line.text.trim().to_string())
                .filter(|text| text.contains("<args...>") || text.contains("<on|off>"))
                .collect::<Vec<_>>();
            assert!(!subs.is_empty(), "`help wireframe` listed no subcommands");
            subs
        }

        let shared = {
            let (mut c, mut scene) = (console(), Scene::default());
            table(&mut c, &mut scene)
        };
        let mut toybox = Console::<crate::toybox::ToyboxControls>::new();
        crate::toybox::register_toybox_commands(&mut toybox);
        let mut controls = crate::toybox::ToyboxControls::default();
        for sub in &shared {
            assert!(
                table(&mut toybox, &mut controls).contains(sub),
                "the toybox's `wireframe` is missing the shared `{sub}`"
            );
        }
    }

    #[test]
    fn a_bare_perspective_cycles_the_whole_list_back_to_where_it_started() {
        let (mut c, mut scene) = (console(), Scene::default());
        let first = scene.wireframe.projection;
        for _ in 0..WireframeProjection::ALL.len() {
            c.dispatch("wireframe", &["perspective"], &mut scene);
        }
        assert_eq!(scene.wireframe.projection, first);
    }
}
