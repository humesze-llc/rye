//! Multi-scene shell: one [`App`] hosting a registry of [`Scene`]s behind a
//! shared menu bar, with boot selection via `--scene=<slug>` / `?scene=<slug>`
//! and an embed mode (`--embed=1` / `?embed=1`) that hides the bar for page
//! embeds. Embed mode leaves the `scene` console command as the only in-app
//! switcher.
//!
//! A demo supplies the table through [`SceneRegistry`] and runs [`SceneShell`]
//! as its App:
//!
//! ```ignore
//! struct Demos;
//! impl loam_app::shell::SceneRegistry for Demos {
//!     const SCENES: &'static [SceneEntry] = &[SceneEntry {
//!         slug: "rotate",
//!         label: "Rotate polytopes",
//!         build: |ctx| Ok(Box::new(RotateScene::new(ctx)?)),
//!     }];
//! }
//! loam_app::run::<SceneShell<Demos>>(config)
//! ```
//!
//! [`Scene`] names no Space: each scene owns its own and recompiles through its
//! own [`ShaderOwner`](crate::ShaderOwner), so a registry may mix geometries.

use std::borrow::Cow;
use std::marker::PhantomData;
use std::sync::Mutex;

use anyhow::{anyhow, Result};
use loam_egui::{cmd, Console};
use loam_math::EuclideanR3;
use loam_render::device::RenderDevice;

use crate::args::Args;
use crate::{egui, App, AssetEvent, FrameCtx, RenderCtx, SetupCtx, ShaderDb};

/// One hostable scene. The method set is [`App`] minus what the shell owns for
/// every scene: window lifecycle, fixed-step ticks, and the choice of which
/// scene is active.
pub trait Scene {
    /// Recompile this scene's shaders against the Space the scene itself owns,
    /// scoped to the [`ShaderOwner`](crate::ShaderOwner) it took at build time.
    /// Scenes with no shaders of their own keep the no-op default.
    fn apply_shader_events(&mut self, _events: &[AssetEvent], _shader_db: &mut ShaderDb) {}
    /// Contributions to the shared menu bar, rendered after the Demo menu. A
    /// scene with nothing to add keeps the no-op default.
    fn menus(&mut self, _ui: &mut egui::Ui) {}
    fn update(&mut self, ctx: &mut FrameCtx<'_>);
    fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>);
    fn on_key(
        &mut self,
        code: winit::keyboard::KeyCode,
        state: winit::event::ElementState,
        ctx: &mut FrameCtx<'_>,
    );
    /// Record this scene's passes into the runner's frame-wide encoder. Must
    /// not submit; see [`RenderCtx`].
    fn record(&mut self, ctx: &mut RenderCtx<'_>) -> Result<()>;
    fn title(&self, fps: f32) -> Cow<'static, str>;
}

/// Registry row: the slug `--scene=` and the `scene` command select by, the
/// menu-bar label, and the constructor.
pub struct SceneEntry {
    pub slug: &'static str,
    pub label: &'static str,
    pub build: fn(&mut SetupCtx<'_>) -> Result<Box<dyn Scene>>,
}

/// A demo's scene table. A trait rather than a value because [`App::setup`] is
/// a static method: the shell has no instance that could have been handed a
/// registry.
pub trait SceneRegistry: 'static {
    /// Index 0 is the fallback when no `scene` arg selects one, so this must be
    /// non-empty.
    const SCENES: &'static [SceneEntry];
}

/// Shell state a scene's console can reach. A console command's context is
/// the scene's own state, so it cannot see [`SceneShell`]; the shell publishes
/// `active` and drains `pending` around the scene's `ui`. Same publish/drain
/// shape as [`crate::capture`]'s request queue.
struct Switcher {
    active: usize,
    pending: Option<usize>,
}

static SWITCHER: Mutex<Switcher> = Mutex::new(Switcher {
    active: 0,
    pending: None,
});

fn with_switcher<R>(f: impl FnOnce(&mut Switcher) -> R) -> R {
    f(&mut SWITCHER.lock().expect("scene switcher poisoned"))
}

/// Registry index for `slug`, or `None` when no scene claims it.
fn scene_index(scenes: &[SceneEntry], slug: &str) -> Option<usize> {
    scenes.iter().position(|entry| entry.slug == slug)
}

/// Queue a switch to the scene named `slug`, applied after the current
/// frame's scene `ui` returns.
fn request_scene(scenes: &[SceneEntry], slug: &str) -> Result<()> {
    let index = scene_index(scenes, slug).ok_or_else(|| {
        let known = scenes
            .iter()
            .map(|entry| entry.slug)
            .collect::<Vec<_>>()
            .join("|");
        anyhow!("unknown scene `{slug}` (try {known})")
    })?;
    with_switcher(|s| s.pending = Some(index));
    Ok(())
}

/// Register the `scene` command: bare lists the registry and marks the
/// active entry, `scene <slug>` switches. Free of the console's `Ctx`
/// because the shell, not the scene, owns the selection.
pub fn register_command<Ctx: 'static, R: SceneRegistry>(console: &mut Console<Ctx>) {
    let slugs = R::SCENES.iter().map(|entry| entry.slug).collect::<Vec<_>>();
    console.register(
        cmd::<Ctx, _>(
            "scene",
            "list scenes (active marked `*`); `scene <slug>` switches, same slugs as --scene= / ?scene=",
            |args, _ctx: &mut Ctx, out| match args.first().copied() {
                None => {
                    let active = with_switcher(|s| s.active);
                    for (i, entry) in R::SCENES.iter().enumerate() {
                        let mark = if i == active { '*' } else { ' ' };
                        out.line(format!("{mark} {} - {}", entry.slug, entry.label));
                    }
                    Ok(())
                }
                Some(slug) => {
                    request_scene(R::SCENES, slug)?;
                    out.line(format!("scene: switching to `{slug}`"));
                    Ok(())
                }
            },
        )
        .with_args(&[&slugs])
        .with_long_help(
            "Selects which scene the shell renders. The same slugs boot the demo\n\
             directly: `--scene=<slug>` natively, `?scene=<slug>` in the browser.\n\
             \n\
             `--embed=1` / `?embed=1` hides the shell menu bar for page embeds,\n\
             which leaves this command as the only in-app switcher.\n\
             \n\
             Bare `scene` lists every registered slug and marks the active one.",
        ),
    );
}

/// Slot per registry entry, filled on the entry's first activation.
type SceneSlots = Vec<Option<Box<dyn Scene>>>;

const ACTIVE_IS_BUILT: &str = "a scene is built before it becomes active";

/// One empty slot per registry entry with `boot` filled. Deferring the rest is
/// the point: a cold start pays one scene's shader compile and VRAM instead of
/// the whole registry's.
fn build_boot_only(
    count: usize,
    boot: usize,
    build: impl FnOnce() -> Result<Box<dyn Scene>>,
) -> Result<SceneSlots> {
    let mut slots: SceneSlots = std::iter::repeat_with(|| None).take(count).collect();
    slots[boot] = Some(build()?);
    Ok(slots)
}

/// Make `next` the active index, constructing it on first activation and
/// serving the cached instance after. Returns the index actually active:
/// `current` when the build failed, since every frame unwraps the active slot
/// and a switch is not worth tearing the process down for.
fn activate(
    slots: &mut SceneSlots,
    current: usize,
    next: usize,
    slug: &str,
    build: impl FnOnce() -> Result<Box<dyn Scene>>,
) -> usize {
    if slots[next].is_none() {
        match build() {
            Ok(scene) => slots[next] = Some(scene),
            Err(err) => {
                tracing::error!("scene '{slug}' failed to build: {err:#}");
                return current;
            }
        }
    }
    next
}

pub struct SceneShell<R: SceneRegistry> {
    scenes: SceneSlots,
    /// Scene shader entries live here, not in the runner's db: the runner's
    /// `&mut ShaderDb` dies with `App::setup`, and a scene built on a later
    /// switch still has to mint an owner and compile against its own Space.
    /// The runner's db stays empty because the shell loads no shaders itself.
    shader_db: ShaderDb,
    active: usize,
    /// Embed mode: no menu bar; the page chrome owns navigation.
    embed: bool,
    capture_panel: crate::capture::CapturePanel,
    perf: crate::trace::PerfOverlay,
    /// `R` is reachable only through its associated const, so nothing else in
    /// the struct carries it.
    registry: PhantomData<fn() -> R>,
}

impl<R: SceneRegistry> SceneShell<R> {
    fn active_scene(&mut self) -> &mut dyn Scene {
        self.scenes[self.active]
            .as_deref_mut()
            .expect(ACTIVE_IS_BUILT)
    }

    /// Rebuild a `SetupCtx` around the frame's `RenderDevice` and the retained
    /// db. `watcher` is `None`: the runner owns it and only lends it for the
    /// duration of `setup`, so a scene built later cannot register new watch
    /// paths. Reload events still reach it through `apply_shader_events`.
    fn switch_to(&mut self, next: usize, rd: &RenderDevice, time: f32) {
        let Self {
            scenes,
            shader_db,
            active,
            ..
        } = self;
        *active = activate(scenes, *active, next, R::SCENES[next].slug, || {
            let mut setup = SetupCtx {
                rd,
                shader_db,
                watcher: None,
                time,
            };
            (R::SCENES[next].build)(&mut setup)
        });
    }
}

/// (boot scene index, embed). Unknown slugs fall back to scene 0. Takes the
/// registry rather than reading `R::SCENES` so the lookup is exercisable
/// against a multi-entry table.
fn resolve_boot(scenes: &[SceneEntry], args: &Args) -> (usize, bool) {
    let active = match args.get("scene") {
        None => 0,
        Some(slug) => scene_index(scenes, slug).unwrap_or_else(|| {
            tracing::warn!("unknown scene '{slug}'; defaulting to '{}'", scenes[0].slug);
            0
        }),
    };
    let embed = args.get("embed").is_some_and(|v| v != "0" && v != "false");
    (active, embed)
}

impl<R: SceneRegistry> App for SceneShell<R> {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let (active, embed) = resolve_boot(R::SCENES, &Args::current());
        let mut shader_db = ShaderDb::new(ctx.rd.device.clone());
        let scenes = build_boot_only(R::SCENES.len(), active, || {
            let mut setup = SetupCtx {
                rd: ctx.rd,
                shader_db: &mut shader_db,
                watcher: ctx.watcher.as_deref_mut(),
                time: ctx.time,
            };
            (R::SCENES[active].build)(&mut setup)
        })?;
        with_switcher(|s| s.active = active);
        Ok(Self {
            scenes,
            shader_db,
            active,
            embed,
            capture_panel: crate::capture::CapturePanel::new(),
            perf: crate::trace::PerfOverlay::new(),
            registry: PhantomData,
        })
    }

    /// The shell owns no shaders of its own, so this satisfies the prelude-axis
    /// bound and nothing else; recompiles route through `apply_shader_events`.
    fn space(&self) -> &EuclideanR3 {
        &EuclideanR3
    }

    /// Fanned out because each scene's apply is scoped to its own owner: a
    /// scene reached here recompiles only the modules it loaded, against the
    /// Space it holds, never the shell's `EuclideanR3`.
    /// The runner's `shader_db` goes unused; scene owners were minted from the
    /// shell's own db, and applying against the wrong db would find no entries.
    fn apply_shader_events(&mut self, events: &[AssetEvent], _shader_db: &mut ShaderDb) {
        let Self {
            scenes, shader_db, ..
        } = self;
        for scene in scenes.iter_mut().flatten() {
            scene.apply_shader_events(events, shader_db);
        }
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        self.active_scene().update(ctx);
    }

    fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        if !self.embed {
            // Bar renders first so the scene's own windows see it in
            // `available_rect()`. `content_rect()` is the viewport minus OS
            // safe-area insets and never shrinks for a panel.
            let active = self.active;
            let scene = self.active_scene();
            egui::TopBottomPanel::top("shell-menu-bar").show(ctx, |ui| {
                egui::MenuBar::new().ui(ui, |ui| {
                    ui.menu_button("Demo", |ui| {
                        for (i, entry) in R::SCENES.iter().enumerate() {
                            if ui.selectable_label(active == i, entry.label).clicked() {
                                // Queued, not applied: a first activation
                                // constructs the scene, which needs the `&mut
                                // self` this closure has borrowed away.
                                with_switcher(|s| s.pending = Some(i));
                                ui.close_kind(egui::UiKind::Menu);
                            }
                        }
                    });
                    scene.menus(ui);
                });
            });
        }
        self.active_scene().ui(ctx, frame);
        self.capture_panel.show(ctx);
        self.perf.show(ctx);
        // Drained after the scene's `ui` returns: the `scene` command runs
        // inside it, holding the borrow a switch would invalidate. Menu-bar
        // clicks queue through the same slot, so both paths land here and the
        // republish carries the outcome back to the console.
        if let Some(next) = with_switcher(|s| s.pending.take()) {
            self.switch_to(next, frame.rd, frame.time);
            with_switcher(|s| s.active = self.active);
        }
    }

    fn on_key(
        &mut self,
        code: winit::keyboard::KeyCode,
        state: winit::event::ElementState,
        ctx: &mut FrameCtx<'_>,
    ) {
        self.active_scene().on_key(code, state, ctx);
    }

    fn record(&mut self, ctx: &mut RenderCtx<'_>) -> Result<()> {
        self.active_scene().record(ctx)
    }

    fn title(&self, fps: f32) -> Cow<'static, str> {
        self.scenes[self.active]
            .as_deref()
            .expect(ACTIVE_IS_BUILT)
            .title(fps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ShaderOwner;
    use loam_math::HyperbolicH3;
    use std::cell::Cell;

    /// A scene whose geometry is not the shell's, written to be compiled: no
    /// `EuclideanR3` appears anywhere in the impl, and the reload hook recompiles
    /// against the scene's own Space.
    struct HyperbolicScene {
        space: HyperbolicH3,
        owner: ShaderOwner,
    }

    impl Scene for HyperbolicScene {
        fn apply_shader_events(&mut self, events: &[AssetEvent], shader_db: &mut ShaderDb) {
            shader_db.apply_events(self.owner, events, &self.space);
        }

        fn update(&mut self, _ctx: &mut FrameCtx<'_>) {}

        fn ui(&mut self, _ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {}

        fn on_key(
            &mut self,
            _code: winit::keyboard::KeyCode,
            _state: winit::event::ElementState,
            _ctx: &mut FrameCtx<'_>,
        ) {
        }

        fn record(&mut self, _ctx: &mut RenderCtx<'_>) -> Result<()> {
            Ok(())
        }

        fn title(&self, _fps: f32) -> Cow<'static, str> {
            "hyperbolic".into()
        }
    }

    /// Pins that [`Scene`]'s required surface names no Space: the fixture above
    /// erases into a `SceneEntry` the shell resolves like any other. A `Scene`
    /// method returning the shell's Space would make this unwritable. It also
    /// pins that a scene contributing nothing to the menu bar registers without
    /// writing an empty `menus`, which is what a single-purpose demo does.
    #[test]
    fn registry_admits_a_scene_outside_the_shell_space() {
        const ENTRY: SceneEntry = SceneEntry {
            slug: "hyperbolic",
            label: "Hyperbolic",
            build: |ctx| {
                Ok(Box::new(HyperbolicScene {
                    space: HyperbolicH3,
                    owner: ctx.shader_db.new_owner(),
                }))
            },
        };
        assert_eq!(
            resolve_boot(&[ENTRY], &Args::from_pairs([("scene", ENTRY.slug)])).0,
            0
        );
    }

    /// Stand-in registry, three entries deep so a lookup asserted against it is
    /// distinguishable from no lookup at all.
    struct Fixture;

    impl SceneRegistry for Fixture {
        const SCENES: &'static [SceneEntry] = &[
            SceneEntry {
                slug: "first",
                label: "First",
                build: |_| unreachable!("registry fixture is never built"),
            },
            SceneEntry {
                slug: "second",
                label: "Second",
                build: |_| unreachable!("registry fixture is never built"),
            },
            SceneEntry {
                slug: "third",
                label: "Third",
                build: |_| unreachable!("registry fixture is never built"),
            },
        ];
    }

    const REGISTRY: &[SceneEntry] = Fixture::SCENES;

    #[test]
    fn boot_defaults_to_first_scene_without_params() {
        assert_eq!(
            resolve_boot(REGISTRY, &Args::from_pairs::<[(&str, &str); 0], _, _>([])),
            (0, false)
        );
    }

    #[test]
    fn boot_slug_selects_its_own_registry_index() {
        for (index, entry) in REGISTRY.iter().enumerate() {
            assert_eq!(
                resolve_boot(REGISTRY, &Args::from_pairs([("scene", entry.slug)])).0,
                index,
                "slug '{}' should boot registry index {index}",
                entry.slug
            );
        }
    }

    #[test]
    fn boot_falls_back_to_first_scene_on_unknown_slug() {
        assert_eq!(
            resolve_boot(REGISTRY, &Args::from_pairs([("scene", "nope")])).0,
            0
        );
    }

    #[test]
    fn embed_is_truthy_except_zero_and_false() {
        assert!(resolve_boot(REGISTRY, &Args::from_pairs([("embed", "1")])).1);
        assert!(resolve_boot(REGISTRY, &Args::from_pairs([("embed", "true")])).1);
        assert!(!resolve_boot(REGISTRY, &Args::from_pairs([("embed", "0")])).1);
        assert!(!resolve_boot(REGISTRY, &Args::from_pairs([("embed", "false")])).1);
    }

    /// GPU-free stand-in for a built scene; the lazy table's contract is about
    /// when a builder runs, not what it returns.
    fn stub_scene() -> Box<dyn Scene> {
        Box::new(HyperbolicScene {
            space: HyperbolicH3,
            owner: ShaderDb::ROOT_OWNER,
        })
    }

    /// Startup fills exactly one slot. Building the registry eagerly is what
    /// made a cold start pay every demo's shader compile and VRAM.
    #[test]
    fn startup_builds_the_boot_entry_and_no_other() {
        let builds = Cell::new(0);
        let slots = build_boot_only(REGISTRY.len(), 1, || {
            builds.set(builds.get() + 1);
            Ok(stub_scene())
        })
        .expect("boot build");
        assert_eq!(builds.get(), 1);
        assert_eq!(
            slots.iter().map(Option::is_some).collect::<Vec<_>>(),
            [false, true, false]
        );
    }

    /// Re-entering a scene must hit the cache: the builder runs on the first
    /// activation and never again, however much the user switches around.
    #[test]
    fn activation_builds_once_and_serves_the_cache_after() {
        let mut slots =
            build_boot_only(REGISTRY.len(), 0, || Ok(stub_scene())).expect("boot build");
        let builds = Cell::new(0);
        let mut active = 0;
        for _ in 0..3 {
            active = activate(&mut slots, active, 2, REGISTRY[2].slug, || {
                builds.set(builds.get() + 1);
                Ok(stub_scene())
            });
            assert_eq!(active, 2);
            active = activate(&mut slots, active, 0, REGISTRY[0].slug, || {
                unreachable!("the boot scene is already built")
            });
        }
        assert_eq!(builds.get(), 1);
    }

    /// A builder that fails must leave the shell on a filled slot: every frame
    /// unwraps the active scene.
    #[test]
    fn a_failed_build_keeps_the_current_scene_active() {
        let mut slots =
            build_boot_only(REGISTRY.len(), 0, || Ok(stub_scene())).expect("boot build");
        let active = activate(&mut slots, 0, 1, REGISTRY[1].slug, || {
            Err(anyhow!("no device"))
        });
        assert_eq!(active, 0);
        assert!(slots[1].is_none());
    }

    /// The `scene` command must reject a slug no scene claims instead of
    /// queueing a switch the shell would index with.
    #[test]
    fn scene_request_queues_known_slugs_and_rejects_unknown() {
        assert!(request_scene(REGISTRY, "nope").is_err());
        assert!(with_switcher(|s| s.pending).is_none());
        assert!(request_scene(REGISTRY, REGISTRY[1].slug).is_ok());
        assert_eq!(with_switcher(|s| s.pending.take()), Some(1));
    }

    /// The command is keyed to the registry type, not to a scene's state, so a
    /// demo whose console carries any `Ctx` can register the switcher.
    #[test]
    fn scene_command_registers_against_any_console_context() {
        let mut console = Console::<u32>::new();
        register_command::<u32, Fixture>(&mut console);
        assert!(console.has_command("scene"));
    }
}
