//! [`Scene`] names no Space: each scene owns its own and recompiles through its
//! own [`ShaderOwner`](crate::ShaderOwner), so a registry may mix geometries.

use std::borrow::Cow;
use std::marker::PhantomData;
use std::sync::{Mutex, PoisonError};

use anyhow::{anyhow, Result};
use loam_egui::{cmd, Console};
use loam_render::device::RenderDevice;

use crate::args::Args;
use crate::command::{CommandCtx, CommandLine};
use crate::{egui, App, AssetEvent, FrameCtx, RenderCtx, SetupCtx, ShaderDb};

/// [`App::tick`], [`App::on_event`] and [`App::on_shader_reload`] reach no
/// scene: the shell forwards none of them, so a scene needing one cannot be
/// hosted.
pub trait Scene {
    // Scoped to the `ShaderOwner` this scene took at build time, so the db
    // rebuilds against the Space it loaded them with.
    fn apply_shader_events(&mut self, _events: &[AssetEvent], _shader_db: &mut ShaderDb) {}
    // Apply one queued command, before the tick it is stamped for.
    fn apply_command(
        &mut self,
        cmd: &CommandLine,
        _ctx: &mut CommandCtx<'_>,
    ) -> anyhow::Result<()> {
        anyhow::bail!("no command target for `{}`", cmd.name)
    }
    // Contributions to the shared menu bar, rendered after the Demo menu.
    fn menus(&mut self, _ui: &mut egui::Ui) {}
    fn update(&mut self, ctx: &mut FrameCtx<'_>);
    fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>);
    fn on_key(
        &mut self,
        code: winit::keyboard::KeyCode,
        state: winit::event::ElementState,
        ctx: &mut FrameCtx<'_>,
    );
    // Must not submit; see `RenderCtx`.
    fn record(&mut self, ctx: &mut RenderCtx<'_>) -> Result<()>;
    fn title(&self, fps: f32) -> Cow<'static, str>;
}

pub struct SceneEntry {
    pub slug: &'static str,
    pub label: &'static str,
    pub build: fn(&mut SetupCtx<'_>) -> Result<Box<dyn Scene>>,
}

/// A trait rather than a value because [`App::setup`] is a static method: the
/// shell has no instance that could have been handed a registry.
pub trait SceneRegistry: 'static {
    // Index 0 is the fallback when no `scene` arg selects one, so this must be
    // non-empty.
    const SCENES: &'static [SceneEntry];
}

// A console command's context is the scene's own state, so it cannot see
// `SceneShell`; the shell publishes `active` and drains `pending` around the
// scene's `ui`.
struct Switcher {
    active: usize,
    pending: Option<usize>,
}

static SWITCHER: Mutex<Switcher> = Mutex::new(Switcher {
    active: 0,
    pending: None,
});

fn with_switcher<R>(f: impl FnOnce(&mut Switcher) -> R) -> R {
    // Recover from poisoning rather than propagate it: the switcher holds an
    // index and a pending slug, neither of which a panicking thread can leave
    // torn, so a poisoned lock costs one stale scene request.
    f(&mut SWITCHER.lock().unwrap_or_else(PoisonError::into_inner))
}

fn scene_index(scenes: &[SceneEntry], slug: &str) -> Option<usize> {
    scenes.iter().position(|entry| entry.slug == slug)
}

// Applied after the current frame's scene `ui` returns.
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

type SceneSlots = Vec<Option<Box<dyn Scene>>>;

const ACTIVE_IS_BUILT: &str = "a scene is built before it becomes active";

// One empty slot per registry entry with `boot` filled, so a cold start pays
// one scene's shader compile and VRAM instead of the whole registry's.
fn build_boot_only(
    count: usize,
    boot: usize,
    build: impl FnOnce() -> Result<Box<dyn Scene>>,
) -> Result<SceneSlots> {
    let mut slots: SceneSlots = std::iter::repeat_with(|| None).take(count).collect();
    slots[boot] = Some(build()?);
    Ok(slots)
}

// Returns the index actually active: `current` when the build failed, since
// every frame unwraps the active slot and a switch is not worth tearing the
// process down for.
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
    /// Not the runner's db: the runner's `&mut ShaderDb` dies with `App::setup`,
    /// and a scene built on a later switch must mint an owner of its own.
    shader_db: ShaderDb,
    active: usize,
    /// Embed mode: no menu bar; the page chrome owns navigation.
    embed: bool,
    /// Retained from boot: a scene built on a later switch gets a `SetupCtx` the
    /// shell assembles and must see the budget the runner resolved.
    sim_threads: usize,
    capture_panel: crate::capture::CapturePanel,
    perf: crate::trace::PerfOverlay,
    /// `R` is reachable only through its associated const.
    registry: PhantomData<fn() -> R>,
}

impl<R: SceneRegistry> SceneShell<R> {
    fn active_scene(&mut self) -> &mut dyn Scene {
        self.scenes[self.active]
            .as_deref_mut()
            .expect(ACTIVE_IS_BUILT) // ok: activate builds before it sets active
    }

    // `watcher` is `None`: the runner lends it only for the duration of `setup`,
    // so a scene built later cannot register new watch paths.
    fn apply_pending_switch(&mut self, rd: &RenderDevice, time: f32, sim_threads: usize) {
        let Self {
            scenes,
            shader_db,
            active,
            ..
        } = self;
        *active = drain_pending(scenes, *active, R::SCENES, |next| {
            let mut setup = SetupCtx {
                rd,
                shader_db,
                watcher: None,
                time,
                sim_threads,
            };
            (R::SCENES[next].build)(&mut setup)
        });
    }
}

fn drain_pending(
    slots: &mut SceneSlots,
    active: usize,
    scenes: &[SceneEntry],
    build: impl FnOnce(usize) -> Result<Box<dyn Scene>>,
) -> usize {
    let Some(next) = with_switcher(|s| s.pending.take()) else {
        return active;
    };
    let now = activate(slots, active, next, scenes[next].slug, || build(next));
    with_switcher(|s| s.active = now);
    now
}

// (boot scene index, embed). Unknown slugs fall back to scene 0.
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
    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let (active, embed) = resolve_boot(R::SCENES, &Args::current());
        let mut shader_db = ShaderDb::new(ctx.rd.device.clone());
        let scenes = build_boot_only(R::SCENES.len(), active, || {
            let mut setup = SetupCtx {
                rd: ctx.rd,
                shader_db: &mut shader_db,
                watcher: ctx.watcher.as_deref_mut(),
                time: ctx.time,
                sim_threads: ctx.sim_threads,
            };
            (R::SCENES[active].build)(&mut setup)
        })?;
        with_switcher(|s| s.active = active);
        Ok(Self {
            scenes,
            shader_db,
            active,
            embed,
            sim_threads: ctx.sim_threads,
            capture_panel: crate::capture::CapturePanel::new(),
            perf: crate::trace::PerfOverlay::new(),
            registry: PhantomData,
        })
    }

    // Fanned out because each scene's apply is scoped to its own owner: a scene
    // recompiles only the modules it loaded, against their own prelude.
    fn apply_shader_events(&mut self, events: &[AssetEvent], _shader_db: &mut ShaderDb) {
        let Self {
            scenes, shader_db, ..
        } = self;
        for scene in scenes.iter_mut().flatten() {
            scene.apply_shader_events(events, shader_db);
        }
    }

    // Only the active scene.
    fn apply_command(&mut self, cmd: &CommandLine, ctx: &mut CommandCtx<'_>) -> Result<()> {
        self.active_scene().apply_command(cmd, ctx)
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        self.active_scene().update(ctx);
    }

    fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        if !self.embed {
            // Bar renders first so the scene's own windows see it in
            // `available_rect()`; `content_rect()` is the viewport minus OS
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
        // Drained after the scene's `ui` returns: the `scene` command runs inside
        // it, holding the borrow a switch would invalidate. Menu-bar clicks queue
        // through the same slot.
        self.apply_pending_switch(frame.rd, frame.time, self.sim_threads);
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
            .expect(ACTIVE_IS_BUILT) // ok: activate builds before it sets active
            .title(fps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ShaderOwner;
    use loam_math::HyperbolicH3;
    use std::cell::Cell;
    use std::rc::Rc;

    struct HyperbolicScene {
        #[allow(dead_code)]
        space: HyperbolicH3,
        owner: ShaderOwner,
        /// Written from outside and read back through `title`: a cached
        /// activation must return the object holding this cell, not a new one.
        state: Rc<Cell<u32>>,
    }

    impl Scene for HyperbolicScene {
        fn apply_shader_events(&mut self, events: &[AssetEvent], shader_db: &mut ShaderDb) {
            shader_db.apply_events(self.owner, events);
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
            format!("hyperbolic {}", self.state.get()).into()
        }
    }

    #[test]
    fn registry_admits_a_scene_in_a_geometry_of_its_own() {
        const ENTRY: SceneEntry = SceneEntry {
            slug: "hyperbolic",
            label: "Hyperbolic",
            build: |ctx| {
                Ok(Box::new(HyperbolicScene {
                    space: HyperbolicH3,
                    owner: ctx.shader_db.new_owner(),
                    state: Rc::default(),
                }))
            },
        };
        // Index 1, not 0: 0 is also the unknown-slug fallback, so a one-entry
        // table would answer the same under a `resolve_boot` that ignores slugs.
        const TABLE: &[SceneEntry] = &[
            SceneEntry {
                slug: "euclidean",
                label: "Euclidean",
                build: |_| unreachable!("only the hyperbolic entry is resolved"),
            },
            ENTRY,
        ];
        assert_eq!(
            resolve_boot(TABLE, &Args::from_pairs([("scene", ENTRY.slug)])).0,
            1
        );
    }

    // Three entries deep, so a lookup asserted against it is distinguishable
    // from no lookup at all.
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

    // GPU-free stand-in; the lazy table's contract is about when a builder runs,
    // not what it returns.
    fn stub_scene() -> Box<dyn Scene> {
        stateful_stub_scene(Rc::default())
    }

    fn stateful_stub_scene(state: Rc<Cell<u32>>) -> Box<dyn Scene> {
        Box::new(HyperbolicScene {
            space: HyperbolicH3,
            owner: ShaderDb::ROOT_OWNER,
            state,
        })
    }

    // `SWITCHER` is process-global and cargo runs tests on parallel threads, so
    // a test that queues through it has to own it outright.
    static SWITCHER_TESTS: Mutex<()> = Mutex::new(());

    // Held and reset, so no test inherits the index another one published.
    fn with_exclusive_switcher<T>(f: impl FnOnce() -> T) -> T {
        let _held = SWITCHER_TESTS
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        with_switcher(|s| {
            s.active = 0;
            s.pending = None;
        });
        f()
    }

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

    #[test]
    fn scene_request_queues_known_slugs_and_rejects_unknown() {
        with_exclusive_switcher(|| {
            assert!(request_scene(REGISTRY, "nope").is_err());
            assert!(with_switcher(|s| s.pending).is_none());
            assert!(request_scene(REGISTRY, REGISTRY[1].slug).is_ok());
            assert_eq!(with_switcher(|s| s.pending.take()), Some(1));
        });
    }

    // The slug bare `scene` marks with `*`, the console's own view of which
    // scene the shell is rendering.
    fn marked_slug(console: &mut Console<()>) -> String {
        console.clear_history();
        crate::command::run_on_console(console, "scene", &mut ());
        console
            .history()
            .iter()
            .find_map(|line| line.text.strip_prefix("* "))
            .and_then(|marked| marked.split(' ').next())
            .expect("bare `scene` marks exactly one entry")
            .to_string()
    }

    #[test]
    fn the_scene_command_switches_to_a_second_scene_and_back() {
        with_exclusive_switcher(|| {
            let mut slots =
                build_boot_only(REGISTRY.len(), 0, || Ok(stub_scene())).expect("boot build");
            let mut console = Console::<()>::new();
            register_command::<(), Fixture>(&mut console);
            let builds = Cell::new(0);
            let mut active = 0;
            let drain = |slots: &mut SceneSlots, active: usize| {
                drain_pending(slots, active, REGISTRY, |_| {
                    builds.set(builds.get() + 1);
                    Ok(stub_scene())
                })
            };

            crate::command::run_on_console(&mut console, "scene second", &mut ());
            active = drain(&mut slots, active);
            assert_eq!(active, 1, "the queued switch must become active");
            assert_eq!(marked_slug(&mut console), "second");
            assert_eq!(builds.get(), 1);

            crate::command::run_on_console(&mut console, "scene first", &mut ());
            active = drain(&mut slots, active);
            assert_eq!(active, 0, "switching back must return to the boot scene");
            assert_eq!(marked_slug(&mut console), "first");
            assert_eq!(builds.get(), 1, "the boot scene is already built");
        });
    }

    #[test]
    fn an_unknown_slug_leaves_the_active_scene_alone() {
        with_exclusive_switcher(|| {
            let mut slots =
                build_boot_only(REGISTRY.len(), 0, || Ok(stub_scene())).expect("boot build");
            let mut console = Console::<()>::new();
            register_command::<(), Fixture>(&mut console);

            crate::command::run_on_console(&mut console, "scene nope", &mut ());
            let active = drain_pending(&mut slots, 0, REGISTRY, |_| {
                unreachable!("an unknown slug must not reach a builder")
            });
            assert_eq!(active, 0);
            assert_eq!(marked_slug(&mut console), "first");
        });
    }

    #[test]
    fn a_revisited_scene_is_the_cached_instance_with_its_state_intact() {
        with_exclusive_switcher(|| {
            let mut slots =
                build_boot_only(REGISTRY.len(), 0, || Ok(stub_scene())).expect("boot build");
            let builds = Cell::new(0);
            let second_state: Rc<Cell<u32>> = Rc::default();
            let mut active = 0;
            let drain = |slots: &mut SceneSlots, active: usize| {
                drain_pending(slots, active, REGISTRY, |_| {
                    builds.set(builds.get() + 1);
                    Ok(stateful_stub_scene(Rc::clone(&second_state)))
                })
            };

            with_switcher(|s| s.pending = Some(1));
            active = drain(&mut slots, active);
            assert_eq!(builds.get(), 1, "first activation builds");
            second_state.set(7);

            with_switcher(|s| s.pending = Some(0));
            active = drain(&mut slots, active);
            assert_eq!(active, 0);

            with_switcher(|s| s.pending = Some(1));
            active = drain(&mut slots, active);
            assert_eq!(active, 1);
            assert_eq!(builds.get(), 1, "the second visit must hit the cache");
            assert_eq!(
                slots[1].as_deref().expect("built").title(0.0),
                "hyperbolic 7",
                "the cached instance kept the state written while it was active"
            );
        });
    }

    #[test]
    fn a_failed_switch_publishes_the_scene_still_rendering() {
        with_exclusive_switcher(|| {
            let mut slots =
                build_boot_only(REGISTRY.len(), 0, || Ok(stub_scene())).expect("boot build");
            let mut console = Console::<()>::new();
            register_command::<(), Fixture>(&mut console);

            crate::command::run_on_console(&mut console, "scene third", &mut ());
            let active = drain_pending(&mut slots, 0, REGISTRY, |_| Err(anyhow!("no device")));
            assert_eq!(active, 0);
            assert_eq!(marked_slug(&mut console), "first");
        });
    }

    #[test]
    fn scene_command_registers_against_any_console_context() {
        let mut console = Console::<u32>::new();
        register_command::<u32, Fixture>(&mut console);
        assert!(console.has_command("scene"));
    }
}
