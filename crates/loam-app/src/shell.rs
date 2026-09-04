//! [`Scene`] names no Space: each scene owns its own and recompiles through its
//! own [`ShaderOwner`](crate::ShaderOwner), so a registry may mix geometries.

use std::borrow::Cow;
use std::marker::PhantomData;
use std::sync::{Mutex, PoisonError};

use anyhow::{anyhow, bail, Result};
use loam_egui::{cmd, Console, ConsoleWriter};
use loam_render::device::RenderDevice;

use crate::args::Args;
use crate::command::{CommandCtx, CommandLine};
use crate::{egui, App, AssetEvent, FrameCtx, RenderCtx, SetupCtx, ShaderDb};

/// The shell forwards neither [`App::tick`], [`App::on_event`] nor
/// [`App::on_shader_reload`].
pub trait Scene {
    /// Scoped to the `ShaderOwner` the scene took at build time.
    fn apply_shader_events(&mut self, _events: &[AssetEvent], _shader_db: &mut ShaderDb) {}
    fn apply_command(
        &mut self,
        cmd: &CommandLine,
        _ctx: &mut CommandCtx<'_>,
    ) -> anyhow::Result<()> {
        anyhow::bail!("no command target for `{}`", cmd.name)
    }
    /// Contributions to the shared menu bar, rendered after the Demo menu.
    fn menus(&mut self, _ui: &mut egui::Ui) {}
    fn update(&mut self, ctx: &mut FrameCtx<'_>);
    fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>);
    fn on_key(
        &mut self,
        code: winit::keyboard::KeyCode,
        state: winit::event::ElementState,
        ctx: &mut FrameCtx<'_>,
    );
    /// Must not submit; see `RenderCtx`.
    fn record(&mut self, ctx: &mut RenderCtx<'_>) -> Result<()>;
    fn title(&self, fps: f32) -> Cow<'static, str>;
    /// `Some(reason)` makes an unforced restart ask first.
    fn unsaved_work(&self) -> Option<Cow<'static, str>> {
        None
    }
}

pub struct SceneEntry {
    pub slug: &'static str,
    pub label: &'static str,
    pub build: fn(&mut SetupCtx<'_>) -> Result<Box<dyn Scene>>,
}

pub trait SceneRegistry: 'static {
    /// Non-empty; index 0 is the boot fallback.
    const SCENES: &'static [SceneEntry];
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Request {
    Switch(usize),
    Restart { forced: bool },
}

struct Switcher {
    active: usize,
    pending: Option<Request>,
}

static SWITCHER: Mutex<Switcher> = Mutex::new(Switcher {
    active: 0,
    pending: None,
});

fn with_switcher<R>(f: impl FnOnce(&mut Switcher) -> R) -> R {
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
    with_switcher(|s| s.pending = Some(Request::Switch(index)));
    Ok(())
}

fn request_restart(boot_args: &Args, forced: bool) -> Result<()> {
    if boot_args.get("script").is_some() {
        bail!(
            "restart is refused in a scripted run: the rebuilt scene re-reads the \
             script argument and would replay it from the first line, so the run \
             would never exit"
        );
    }
    with_switcher(|s| s.pending = Some(Request::Restart { forced }));
    Ok(())
}

fn run_restart(args: &[&str], boot_args: &Args, out: &mut ConsoleWriter) -> Result<()> {
    let forced = match args.first().copied() {
        None => false,
        Some("force") => true,
        Some(other) => bail!("restart: unknown arg `{other}` (try force)"),
    };
    request_restart(boot_args, forced)?;
    out.line(if forced {
        "restart: rebuilding the active scene"
    } else {
        "restart: rebuilding the active scene unless it reports unsaved work"
    });
    Ok(())
}

fn claims_restart(
    code: winit::keyboard::KeyCode,
    state: winit::event::ElementState,
    ui_captures_keyboard: bool,
) -> bool {
    code == winit::keyboard::KeyCode::KeyR
        && state == winit::event::ElementState::Pressed
        && !ui_captures_keyboard
}

fn queue_restart(forced: bool) {
    if let Err(err) = request_restart(&Args::current(), forced) {
        tracing::warn!("{err:#}");
    }
}

/// Fill with [`crate::build_info`]: the `env!` must expand in the demo's crate.
pub struct BuildInfo {
    pub crate_name: &'static str,
    pub crate_version: &'static str,
    pub build_hash: &'static str,
    pub build_dirty: &'static str,
}

/// Every verb the runner owns, so no scene's console is missing half of them.
pub fn register_shell_commands<Ctx: 'static, R: SceneRegistry>(
    console: &mut Console<Ctx>,
    build: BuildInfo,
) {
    register_scene_commands::<Ctx, R>(console);
    crate::capture::register_commands(console);
    crate::capture::bind_default_hotkeys(console);
    crate::log::register_command(console);
    crate::trace::register_command(console);
    crate::fps::register_command(console);
    crate::vsync::register_command(console);
    crate::version::register_command(
        console,
        build.crate_name,
        build.crate_version,
        build.build_hash,
        build.build_dirty,
    );
}

fn register_scene_commands<Ctx: 'static, R: SceneRegistry>(console: &mut Console<Ctx>) {
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
    console.register(
        cmd::<Ctx, _>(
            "restart",
            "rebuild the active scene at its boot state; `restart force` skips the confirmation",
            |args, _ctx: &mut Ctx, out| run_restart(args, &Args::current(), out),
        )
        .with_args(&[&["force"]])
        .with_long_help(
            "Calls the active scene's registry builder again and swaps the result in,\n\
             so the scene comes back exactly as it boots. R does the same. The\n\
             rebuild costs the scene's shader compile and drops its console\n\
             scrollback.\n\
             \n\
             A scene holding unsaved work asks first; `restart force` rebuilds\n\
             without asking.\n\
             \n\
             Refused under `--script=` / `?script=`: the rebuilt scene re-reads that\n\
             argument and would replay the script from its first line.",
        ),
    );
}

type SceneSlots = Vec<Option<Box<dyn Scene>>>;

const ACTIVE_IS_BUILT: &str = "a scene is built before it becomes active";

fn build_boot_only(
    count: usize,
    boot: usize,
    build: impl FnOnce() -> Result<Box<dyn Scene>>,
) -> Result<SceneSlots> {
    let mut slots: SceneSlots = std::iter::repeat_with(|| None).take(count).collect();
    slots[boot] = Some(build()?);
    Ok(slots)
}

// Returns `current` when the build failed; every frame unwraps the active slot.
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

// Build before clearing the slot, or a failed build panics the next frame.
fn rebuild(
    slots: &mut SceneSlots,
    active: usize,
    slug: &str,
    build: impl FnOnce() -> Result<Box<dyn Scene>>,
) {
    match build() {
        Ok(scene) => slots[active] = Some(scene),
        Err(err) => tracing::error!("scene '{slug}' failed to rebuild: {err:#}"),
    }
}

pub struct SceneShell<R: SceneRegistry> {
    scenes: SceneSlots,
    /// Not the runner's: that `&mut ShaderDb` dies with `App::setup`.
    shader_db: ShaderDb,
    active: usize,
    embed: bool,
    sim_threads: usize,
    capture_panel: crate::capture::CapturePanel,
    perf: crate::trace::PerfOverlay,
    /// `Some` while the restart confirmation is on screen.
    confirm: Option<Cow<'static, str>>,
    script: Option<crate::script::ScriptDriver>,
    registry: PhantomData<fn() -> R>,
}

impl<R: SceneRegistry> SceneShell<R> {
    fn active_scene(&mut self) -> &mut dyn Scene {
        self.scenes[self.active]
            .as_deref_mut()
            .expect(ACTIVE_IS_BUILT)
    }

    // `watcher` is `None`: the runner lends it only for the duration of `setup`.
    fn apply_pending_switch(&mut self, rd: &RenderDevice, time: f32, sim_threads: usize) {
        let Self {
            scenes,
            shader_db,
            active,
            confirm,
            ..
        } = self;
        let drained = drain_pending(scenes, *active, R::SCENES, |next| {
            let mut setup = SetupCtx {
                rd,
                shader_db,
                watcher: None,
                time,
                sim_threads,
            };
            (R::SCENES[next].build)(&mut setup)
        });
        match drained {
            Drain::Nothing => {}
            Drain::Applied(now) => {
                *active = now;
                *confirm = None;
            }
            Drain::Ask(reason) => *confirm = Some(reason),
        }
    }

    // Painted before the drain, so an answer applies in the frame it is given.
    fn show_restart_confirm(&mut self, ctx: &egui::Context) {
        let Some(reason) = self.confirm.clone() else {
            return;
        };
        let mut answer: Option<bool> = None;
        egui::Modal::new(egui::Id::new("shell-restart-confirm")).show(ctx, |ui| {
            ui.heading("Restart this scene?");
            ui.label(reason.as_ref());
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Restart").clicked() {
                    answer = Some(true);
                }
                if ui.button("Cancel").clicked() {
                    answer = Some(false);
                }
            });
        });
        match answer {
            None => {}
            Some(true) => {
                with_switcher(|s| s.pending = Some(Request::Restart { forced: true }));
                self.confirm = None;
            }
            Some(false) => self.confirm = None,
        }
    }
}

enum Drain {
    Nothing,
    Applied(usize),
    Ask(Cow<'static, str>),
}

fn drain_pending(
    slots: &mut SceneSlots,
    active: usize,
    scenes: &[SceneEntry],
    build: impl FnOnce(usize) -> Result<Box<dyn Scene>>,
) -> Drain {
    let Some(request) = with_switcher(|s| s.pending.take()) else {
        return Drain::Nothing;
    };
    match request {
        Request::Switch(next) => {
            let now = activate(slots, active, next, scenes[next].slug, || build(next));
            with_switcher(|s| s.active = now);
            Drain::Applied(now)
        }
        Request::Restart { forced } => {
            let unsaved = slots[active]
                .as_deref()
                .expect(ACTIVE_IS_BUILT)
                .unsaved_work();
            match unsaved {
                Some(reason) if !forced => Drain::Ask(reason),
                _ => {
                    rebuild(slots, active, scenes[active].slug, || build(active));
                    Drain::Applied(active)
                }
            }
        }
    }
}

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
        let args = Args::current();
        let (active, embed) = resolve_boot(R::SCENES, &args);
        let script = crate::script::driver_from_args(&args)?;
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
            confirm: None,
            script,
            registry: PhantomData,
        })
    }

    fn apply_shader_events(&mut self, events: &[AssetEvent], _shader_db: &mut ShaderDb) {
        let Self {
            scenes, shader_db, ..
        } = self;
        for scene in scenes.iter_mut().flatten() {
            scene.apply_shader_events(events, shader_db);
        }
    }

    fn apply_command(&mut self, cmd: &CommandLine, ctx: &mut CommandCtx<'_>) -> Result<()> {
        self.active_scene().apply_command(cmd, ctx)
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        if let Some(driver) = self.script.as_mut() {
            if driver.advance() == crate::script::ScriptStatus::Finished {
                self.script = None;
                crate::script::request_exit();
            }
        }
        self.active_scene().update(ctx);
    }

    fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        if !self.embed {
            // The bar renders first so the scene's windows see it in `available_rect()`.
            let active = self.active;
            let scene = self.active_scene();
            egui::TopBottomPanel::top("shell-menu-bar").show(ctx, |ui| {
                egui::MenuBar::new().ui(ui, |ui| {
                    ui.menu_button("Demo", |ui| {
                        for (i, entry) in R::SCENES.iter().enumerate() {
                            if ui.selectable_label(active == i, entry.label).clicked() {
                                with_switcher(|s| s.pending = Some(Request::Switch(i)));
                                ui.close_kind(egui::UiKind::Menu);
                            }
                        }
                        ui.separator();
                        if ui.button("Restart scene (R)").clicked() {
                            queue_restart(false);
                            ui.close_kind(egui::UiKind::Menu);
                        }
                    });
                    scene.menus(ui);
                });
            });
        }
        self.active_scene().ui(ctx, frame);
        self.capture_panel.show(ctx);
        self.perf.show(ctx);
        self.show_restart_confirm(ctx);
        // Drained after the scene's `ui`: the `scene` verb runs inside it.
        self.apply_pending_switch(frame.rd, frame.time, self.sim_threads);
    }

    fn on_key(
        &mut self,
        code: winit::keyboard::KeyCode,
        state: winit::event::ElementState,
        ctx: &mut FrameCtx<'_>,
    ) {
        if claims_restart(code, state, ctx.ui_capture.keyboard) {
            queue_restart(false);
            return;
        }
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
    use std::rc::Rc;

    struct HyperbolicScene {
        #[allow(dead_code)]
        space: HyperbolicH3,
        owner: ShaderOwner,
        state: Rc<Cell<u32>>,
        unsaved: Option<Cow<'static, str>>,
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

        fn unsaved_work(&self) -> Option<Cow<'static, str>> {
            self.unsaved.clone()
        }
    }

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

    fn stub_scene() -> Box<dyn Scene> {
        stateful_stub_scene(Rc::default())
    }

    fn stateful_stub_scene(state: Rc<Cell<u32>>) -> Box<dyn Scene> {
        Box::new(HyperbolicScene {
            space: HyperbolicH3,
            owner: ShaderDb::ROOT_OWNER,
            state,
            unsaved: None,
        })
    }

    const UNSAVED: &str = "the stub holds work no builder can reproduce";

    fn dirty_stub_scene() -> Box<dyn Scene> {
        Box::new(HyperbolicScene {
            space: HyperbolicH3,
            owner: ShaderDb::ROOT_OWNER,
            state: Rc::default(),
            unsaved: Some(Cow::Borrowed(UNSAVED)),
        })
    }

    #[track_caller]
    fn applied(drain: Drain, active: usize) -> usize {
        match drain {
            Drain::Nothing => active,
            Drain::Applied(now) => now,
            Drain::Ask(reason) => panic!("unexpected confirmation: {reason}"),
        }
    }

    static SWITCHER_TESTS: Mutex<()> = Mutex::new(());

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
            assert_eq!(
                with_switcher(|s| s.pending.take()),
                Some(Request::Switch(1))
            );
        });
    }

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
            register_scene_commands::<(), Fixture>(&mut console);
            let builds = Cell::new(0);
            let mut active = 0;
            let drain = |slots: &mut SceneSlots, active: usize| {
                applied(
                    drain_pending(slots, active, REGISTRY, |_| {
                        builds.set(builds.get() + 1);
                        Ok(stub_scene())
                    }),
                    active,
                )
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
    fn a_revisited_scene_is_the_cached_instance_with_its_state_intact() {
        with_exclusive_switcher(|| {
            let mut slots =
                build_boot_only(REGISTRY.len(), 0, || Ok(stub_scene())).expect("boot build");
            let builds = Cell::new(0);
            let second_state: Rc<Cell<u32>> = Rc::default();
            let mut active = 0;
            let drain = |slots: &mut SceneSlots, active: usize| {
                applied(
                    drain_pending(slots, active, REGISTRY, |_| {
                        builds.set(builds.get() + 1);
                        Ok(stateful_stub_scene(Rc::clone(&second_state)))
                    }),
                    active,
                )
            };

            with_switcher(|s| s.pending = Some(Request::Switch(1)));
            active = drain(&mut slots, active);
            assert_eq!(builds.get(), 1, "first activation builds");
            second_state.set(7);

            with_switcher(|s| s.pending = Some(Request::Switch(0)));
            active = drain(&mut slots, active);
            assert_eq!(active, 0);

            with_switcher(|s| s.pending = Some(Request::Switch(1)));
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
            register_scene_commands::<(), Fixture>(&mut console);

            crate::command::run_on_console(&mut console, "scene third", &mut ());
            let active = applied(
                drain_pending(&mut slots, 0, REGISTRY, |_| Err(anyhow!("no device"))),
                0,
            );
            assert_eq!(active, 0);
            assert_eq!(marked_slug(&mut console), "first");
        });
    }

    #[test]
    fn the_shell_claims_r_on_press_and_only_while_the_ui_is_not_typing() {
        use winit::event::ElementState;
        use winit::keyboard::KeyCode;
        assert!(claims_restart(KeyCode::KeyR, ElementState::Pressed, false));
        assert!(
            !claims_restart(KeyCode::KeyR, ElementState::Pressed, true),
            "typing `restart` into a console would fire the hotkey mid-word"
        );
        assert!(
            !claims_restart(KeyCode::KeyR, ElementState::Released, false),
            "a release would restart a second time on key-up"
        );
        assert!(!claims_restart(KeyCode::KeyT, ElementState::Pressed, false));
    }

    #[test]
    fn a_restart_replaces_the_cached_instance_with_a_freshly_built_one() {
        with_exclusive_switcher(|| {
            let state: Rc<Cell<u32>> = Rc::default();
            let mut slots = build_boot_only(REGISTRY.len(), 0, || {
                Ok(stateful_stub_scene(Rc::clone(&state)))
            })
            .expect("boot build");
            state.set(7);
            let builds = Cell::new(0);

            with_switcher(|s| s.pending = Some(Request::Restart { forced: false }));
            let active = applied(
                drain_pending(&mut slots, 0, REGISTRY, |_| {
                    builds.set(builds.get() + 1);
                    Ok(stub_scene())
                }),
                0,
            );

            assert_eq!(active, 0, "a restart stays on the scene it rebuilt");
            assert_eq!(builds.get(), 1, "the builder is the boot path");
            assert_eq!(
                slots[0].as_deref().expect("built").title(0.0),
                "hyperbolic 0",
                "the state written while the scene ran has to be gone"
            );
        });
    }

    #[test]
    fn a_failed_rebuild_keeps_the_instance_that_was_running() {
        with_exclusive_switcher(|| {
            let state: Rc<Cell<u32>> = Rc::default();
            let mut slots = build_boot_only(REGISTRY.len(), 0, || {
                Ok(stateful_stub_scene(Rc::clone(&state)))
            })
            .expect("boot build");
            state.set(7);

            with_switcher(|s| s.pending = Some(Request::Restart { forced: false }));
            let active = applied(
                drain_pending(&mut slots, 0, REGISTRY, |_| Err(anyhow!("no device"))),
                0,
            );

            assert_eq!(active, 0);
            assert_eq!(
                slots[0].as_deref().expect("built").title(0.0),
                "hyperbolic 7",
                "a cleared slot would leave the next frame's unwrap to panic"
            );
        });
    }

    #[test]
    fn a_scene_with_unsaved_work_is_not_rebuilt_until_the_request_is_forced() {
        with_exclusive_switcher(|| {
            let mut slots =
                build_boot_only(REGISTRY.len(), 0, || Ok(dirty_stub_scene())).expect("boot build");
            let builds = Cell::new(0);
            let drain = |slots: &mut SceneSlots| {
                drain_pending(slots, 0, REGISTRY, |_| {
                    builds.set(builds.get() + 1);
                    Ok(stub_scene())
                })
            };

            with_switcher(|s| s.pending = Some(Request::Restart { forced: false }));
            let asked = drain(&mut slots);
            assert!(
                matches!(&asked, Drain::Ask(reason) if reason == UNSAVED),
                "the scene's own reason has to reach the confirmation"
            );
            assert_eq!(builds.get(), 0);

            with_switcher(|s| s.pending = Some(Request::Restart { forced: true }));
            let forced = drain(&mut slots);
            assert!(matches!(forced, Drain::Applied(0)));
            assert_eq!(builds.get(), 1);
        });
    }

    #[test]
    fn the_restart_verb_queues_through_the_same_slot_as_a_switch() {
        with_exclusive_switcher(|| {
            let mut slots =
                build_boot_only(REGISTRY.len(), 0, || Ok(stub_scene())).expect("boot build");
            let mut console = Console::<()>::new();
            register_scene_commands::<(), Fixture>(&mut console);
            let builds = Cell::new(0);

            crate::command::run_on_console(&mut console, "restart", &mut ());
            let active = applied(
                drain_pending(&mut slots, 0, REGISTRY, |_| {
                    builds.set(builds.get() + 1);
                    Ok(stub_scene())
                }),
                0,
            );

            assert_eq!(active, 0);
            assert_eq!(builds.get(), 1, "the verb reaches the shell's rebuild");
        });
    }

    #[test]
    fn a_restart_in_a_scripted_run_is_refused_rather_than_queued() {
        with_exclusive_switcher(|| {
            let scripted = Args::from_pairs([("script", "console-scripts/x.script")]);
            let mut out = ConsoleWriter::new();
            for line in [&[][..], &["force"][..]] {
                let err = run_restart(line, &scripted, &mut out)
                    .expect_err("a scripted run must refuse a restart");
                assert!(format!("{err:#}").contains("scripted run"), "{err:#}");
                assert!(
                    with_switcher(|s| s.pending).is_none(),
                    "a refused restart must leave the slot empty, or the script \
                     replays from its first line and the run never exits"
                );
            }
            run_restart(&[], &Args::default(), &mut out).expect("an unscripted run restarts");
            assert_eq!(
                with_switcher(|s| s.pending.take()),
                Some(Request::Restart { forced: false })
            );
        });
    }

    #[test]
    fn the_restart_verb_rejects_an_argument_it_does_not_define() {
        with_exclusive_switcher(|| {
            let mut out = ConsoleWriter::new();
            assert!(run_restart(&["now"], &Args::default(), &mut out).is_err());
            assert!(with_switcher(|s| s.pending).is_none());
            run_restart(&["force"], &Args::default(), &mut out).expect("force is the one arg");
            assert_eq!(
                with_switcher(|s| s.pending.take()),
                Some(Request::Restart { forced: true })
            );
        });
    }
}
