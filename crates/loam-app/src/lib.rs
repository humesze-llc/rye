//! `loam-app`: thin App trait + event-loop runner that extracts the winit
//! boilerplate every Loam example would otherwise rewrite.
//!
//! Apps implement [`App`] on a struct that owns their state; the runner [`run`]
//! (or [`run_with_config`]) handles window creation, [`RenderDevice`] +
//! surface-error recovery, [`ShaderDb`]/[`AssetWatcher`] hot-reload,
//! [`InputState`] routing, [`FixedTimestep`]-driven `App::tick`, and FPS/title
//! bookkeeping. It is not an ECS, scene graph, render-graph orchestrator, or
//! camera framework; apps own those directly.
//!
//! A frame-capture pipeline ships behind the `capture` feature (default-on); see
//! [`capture`]. External recorders (OBS) remain better for long sessions.
//!
//! ## Lifecycle
//!
//! ```text
//! run::<MyApp>()
//!   * EventLoop::new
//!   * on `resumed`:
//!         create Window
//!         create RenderDevice
//!         create ShaderDb + AssetWatcher
//!         create UiIntegration (egui)
//!         A::setup(&mut SetupCtx) -> A
//!   * on each redraw:
//!         command::drain -> A::apply_command for each, stamped with the
//!             tick index the drain precedes
//!         FixedTimestep::advance -> ticks
//!         for each tick: A::tick(dt, &mut TickCtx)
//!         input.take_frame()
//!         A::update(&mut FrameCtx)
//!         A::ui(&egui::Context, &mut FrameCtx)
//!         A::on_event(...) for each WindowEvent
//!         poll AssetWatcher -> if events:
//!             A::apply_shader_events(events, &mut ShaderDb)
//!             A::on_shader_reload(&mut SetupCtx)
//!         maybe update title (rate-limited to ~1 Hz)
//!         RenderDevice::begin_frame
//!         A::render(rd, view)
//!         UiIntegration::paint  (egui overlay, LoadOp::Load)
//!         frame.present
//!   * on `Esc` or `CloseRequested`: exit cleanly
//!     (Esc is suppressed when an egui TextEdit has keyboard focus)
//! ```

use std::borrow::Cow;
use std::marker::PhantomData;
use std::sync::Arc;
// `web_time::Instant` works on native (std type) and wasm32 (`performance.now()`);
// `std::time::Instant::now` panics on wasm32, so the swap is mandatory there.
use web_time::Instant;

// Real capture pipeline on native+`capture`; stub elsewhere. Both expose the
// same surface so demos need no `cfg` gates at `loam_app::capture::*` call sites.
#[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
pub mod capture;

pub mod args;
#[cfg(any(not(feature = "capture"), target_arch = "wasm32"))]
#[path = "capture_stub.rs"]
pub mod capture;
pub mod command;
pub mod cursor;
pub mod fps;
pub mod frame_pacing;
pub mod freecam;
pub mod keymap;
pub mod log;
pub mod script;
pub mod shell;
pub mod trace;
pub mod version;
pub mod vsync;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes},
};

use loam_asset::AssetWatcher;
use loam_egui::UiIntegration;
use loam_input::{FrameInput, InputState};
use loam_render::device::RenderDevice;
use loam_time::jobs::JobPool;
use loam_time::FixedTimestep;

use crate::args::Args;

// Convenience re-exports so apps depend on `loam-app` alone for common types.
pub use loam_asset::AssetEvent;
pub use loam_camera::{
    Camera, CameraController, CameraView, FirstPersonController, OrbitController,
};
pub use loam_egui::{egui, world_to_screen, BottomOverlay, LinearIndicator};
pub use loam_input::FrameInput as Input;
pub use loam_shader::{ShaderDb, ShaderOwner};

// ---------------------------------------------------------------------------
// App trait
// ---------------------------------------------------------------------------

/// The framework calls back into your App through this trait. Every method
/// except [`App::setup`] has a default impl.
pub trait App: Sized + 'static {
    /// One-shot construction after `RenderDevice` and `ShaderDb` are ready. Build
    /// render nodes, load shaders, allocate state, and store it all (including
    /// any Spaces and cameras) in the returned `Self`.
    fn setup(ctx: &mut SetupCtx<'_>) -> anyhow::Result<Self>;

    /// Per-tick simulation step at the fixed-timestep rate: usually 0 or 1 per
    /// frame, spiking after a stall to the runner's catch-up cap. The native
    /// runner takes rate and cap from [`RunConfig`]; the wasm worker hardcodes
    /// 60Hz and [`DEFAULT_MAX_TICKS_PER_FRAME`], since `RunConfig` never
    /// crosses its init message.
    fn tick(&mut self, _dt: f32, _ctx: &mut TickCtx<'_>) {}

    /// Apply one command from [`crate::command`]'s queue, before the tick it is
    /// stamped for. The runner's engine-verb table is consulted first, so a name
    /// it claims never arrives here.
    ///
    /// The default refuses rather than swallowing: a command with nowhere to go
    /// is a wiring mistake, and silence is how it would stay one.
    fn apply_command(
        &mut self,
        cmd: &command::CommandLine,
        _ctx: &mut command::CommandCtx<'_>,
    ) -> anyhow::Result<()> {
        anyhow::bail!("no command target for `{}`", cmd.name)
    }

    /// Per-frame update with drained input, after all the frame's ticks. Advance
    /// the camera controller, recompute uniforms, etc.
    fn update(&mut self, _ctx: &mut FrameCtx<'_>) {}

    /// Custom `WindowEvent` handling beyond the framework's input routing.
    ///
    /// Not called for keyboard events in the wasm worker (it can't construct a
    /// `WindowEvent::KeyboardInput`); use [`App::on_key`] for hotkeys, which fires
    /// on both paths.
    fn on_event(&mut self, _ev: &WindowEvent, _ctx: &mut FrameCtx<'_>) {}

    /// Keyboard hotkey hook, fired for every press and release after input
    /// routing. Use for edge-triggered toggles (Space, Tab, digits, etc.);
    /// continuous WASD belongs in [`App::update`] reading held-key axes.
    ///
    /// Separate from [`App::on_event`] because the wasm worker can't construct a
    /// `winit::event::KeyEvent` but can produce `KeyCode + ElementState`, so
    /// `on_key` reaches both native and wasm.
    fn on_key(
        &mut self,
        _code: winit::keyboard::KeyCode,
        _state: ElementState,
        _ctx: &mut FrameCtx<'_>,
    ) {
    }

    /// Recompile the shaders `events` touches. The default covers exactly what
    /// this app loaded under [`ShaderDb::ROOT_OWNER`]; each module rebuilds
    /// against the Space prelude it was loaded with. An app hosting sub-scenes
    /// that took a [`ShaderDb::new_owner`] overrides this to fan out one scoped
    /// apply per sub-scene, so each one recompiles exactly its own modules.
    fn apply_shader_events(&mut self, events: &[AssetEvent], shader_db: &mut ShaderDb) {
        shader_db.apply_events(ShaderDb::ROOT_OWNER, events);
    }

    /// Hot-reload notification, after [`App::apply_shader_events`] has run;
    /// rebuild any consumer pipelines that may be stale.
    fn on_shader_reload(&mut self, _ctx: &mut SetupCtx<'_>) {}

    /// Legacy render path. Implement either this or `App::record`; the runner
    /// always calls `record`, whose default impl calls this. Each `render` call
    /// typically does its own `queue.submit`, so an N-node demo pays N submits;
    /// `record` lets the runner batch into a single per-frame submit. New demos
    /// should override `record`; this remains for the existing examples.
    fn render(&mut self, _rd: &RenderDevice, _view: &wgpu::TextureView) -> anyhow::Result<()> {
        Ok(())
    }

    /// Preferred render path: the runner owns one frame-wide command encoder
    /// (shared via [`RenderCtx::encoder`]) for the demo's passes, ui-paint, and
    /// the wasm composite, reaching the GPU in a single `queue.submit`. Default
    /// impl falls back to [`App::render`], so migration is opt-in.
    ///
    /// Contract:
    /// - Do NOT call `encoder.finish()` or `queue.submit`; the runner does that
    ///   once at end-of-frame.
    /// - Multiple render passes per call are fine on the shared encoder.
    /// - Use `ctx.view` as the color target; the runner already selected the
    ///   right view (MSAA / scene-target / swapchain) for the platform.
    fn record(&mut self, ctx: &mut RenderCtx<'_>) -> anyhow::Result<()> {
        self.render(ctx.rd, ctx.view)
    }

    /// Build this frame's egui UI, after [`App::update`]; painted as a 2D overlay.
    /// Gate gameplay input on [`FrameCtx::ui_has_focus`] so typing into a field
    /// doesn't also fire WASD. For a label following a 3D object, use
    /// [`world_to_screen`] to place an `egui::Area`.
    ///
    /// ```ignore
    /// fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
    ///     egui::Window::new("Settings").show(ctx, |ui| {
    ///         ui.add(egui::Slider::new(&mut self.fov, 30.0..=120.0));
    ///         if ui.button("Reset").clicked() { self.reset(); }
    ///     });
    /// }
    /// ```
    fn ui(&mut self, _ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {}

    /// Title bar text. Override for live readouts; the runner rate-limits the
    /// `set_title` call to ~1 Hz.
    fn title(&self, _fps: f32) -> Cow<'static, str> {
        Cow::Borrowed("loam app")
    }
}

// ---------------------------------------------------------------------------
// Context structs
// ---------------------------------------------------------------------------

/// Run one frame's fixed-timestep ticks: advance the accumulator and call
/// `App::tick` for every tick it yields. Shared by the native runner and the
/// wasm worker, so the accumulator-to-`App::tick` mapping has one definition.
/// The values it is called with are the caller's and do differ: the native
/// runner passes `RunConfig::fixed_hz` and `RunConfig::max_ticks_per_frame`,
/// the worker hardcodes 60Hz and [`DEFAULT_MAX_TICKS_PER_FRAME`], so both the
/// `dt` a tick sees and how many ticks a stalled frame yields diverge once a
/// demo sets either. Returns the tick count (for `FrameCtx::n_ticks`).
///
/// The catch-up cap lives solely in the `FixedTimestep`
/// (`with_max_catch_up`); capping again here would book ticks the accumulator
/// charged but the sim never ran, silently losing sim time and desyncing
/// `tick_index` from `FixedTimestep::tick`.
///
/// `now` reaches the wall clock only through the accumulator, which decides
/// *how many* ticks run. What each tick sees is a pure function of its index.
pub(crate) fn drive_fixed_ticks<A: App>(
    app: &mut A,
    timestep: &mut FixedTimestep,
    tick_index: &mut u64,
    now: Instant,
    fixed_hz: u32,
    jobs: &JobPool,
) -> usize {
    let _scope = loam_time::frame_trace::scope("sim-ticks");
    let ticks = timestep.advance(now);
    let dt = 1.0 / fixed_hz as f32;
    // Bounded by the accumulator's catch-up cap, so the narrowing is exact.
    let n_ticks = (ticks.end - ticks.start) as usize;
    for tick in ticks {
        let mut tctx = TickCtx {
            time: tick as f32 * dt,
            tick,
            jobs,
        };
        app.tick(dt, &mut tctx);
        // Derived from the range rather than incremented independently, so
        // the runner's counter cannot drift from `FixedTimestep::tick`.
        *tick_index = tick + 1;
    }
    n_ticks
}

/// Setup-phase context. Available during [`App::setup`] and [`App::on_shader_reload`].
pub struct SetupCtx<'a> {
    pub rd: &'a RenderDevice,
    pub shader_db: &'a mut ShaderDb,
    /// `None` when filesystem watching failed to init; apps still load shaders
    /// but get no hot-reload.
    pub watcher: Option<&'a mut AssetWatcher>,
    /// Wall-clock seconds since `run`. Always 0 in `setup`.
    pub time: f32,
    /// Sim worker budget the runner resolved once before `setup`, and the
    /// same integer [`TickCtx::jobs`] partitions against. An app that owns a
    /// sim with its own worker-count field writes this into it here; the
    /// runner never reaches into app state to do it, and there is no second
    /// place the count can be set, so the two cannot disagree.
    pub sim_threads: usize,
}

/// Per-tick context. Visible to [`App::tick`]. Deliberately GPU-free so sim code stays
/// bit-deterministic.
pub struct TickCtx<'a> {
    /// Sim time in seconds: `tick` scaled by the runner's fixed-timestep
    /// interval (`1.0 / RunConfig::fixed_hz` natively, 1/60 in the wasm
    /// worker). Derived from the tick index rather than read from the clock, so
    /// replaying the same tick range yields the same bits however the frames
    /// were paced. Wall-clock time lives on [`FrameCtx::time`], outside the
    /// determinism boundary.
    pub time: f32,
    pub tick: u64,
    /// The runner's worker pool, lent for the tick. The one runner-owned
    /// resource a deterministic tick may reach: its partition is a pure
    /// function of (unit count, worker count), so what a stage computes does
    /// not depend on how the pool scheduled it. Its timings do, and nothing
    /// in a tick may branch on them.
    pub jobs: &'a JobPool,
}

/// Render-time context for `App::record`. Owns the shared frame encoder; the
/// runner reuses it for ui-paint and the wasm composite, then submits once.
///
/// `view` is the runner's scene-pass color target for the platform (MSAA
/// attachment, offscreen scene texture on the composite path, or the swapchain
/// view). Pipelines built with [`RenderDevice::target_format`] +
/// [`RenderDevice::sample_count`] match it. The UI pass paints afterwards and
/// is single-sampled either way: into a non-sRGB view of the swapchain on the
/// direct-to-swapchain paths, which under MSAA is the scene resolve's
/// destination rather than this attachment, and into this very view on the
/// composite path.
pub struct RenderCtx<'a> {
    pub rd: &'a RenderDevice,
    pub view: &'a wgpu::TextureView,
    /// Shared command encoder. Open passes, draw, drop; do NOT call `finish()`
    /// or `queue.submit`.
    pub encoder: &'a mut wgpu::CommandEncoder,
}

/// Per-frame context. Visible to [`App::update`] and [`App::on_event`]. Carries the
/// drained input, FPS readout, and the count of ticks the framework just executed.
pub struct FrameCtx<'a> {
    pub rd: &'a RenderDevice,
    pub input: FrameInput,
    pub time: f32,
    pub fps: f32,
    pub n_ticks: usize,
    pub tick: u64,
    /// Wall-clock seconds since the previous `App::update` call. Use this for
    /// variable-rate visual animation (camera smoothing, hover bobs, particles,
    /// continuous rotors driven by user-perceived time). For deterministic sim
    /// state that must be lockstep-reproducible, use [`App::tick`] instead;
    /// `tick`'s `dt` is the fixed-timestep interval regardless of frame rate.
    ///
    /// First call after setup gets the runner's fixed-timestep interval as a
    /// sensible fallback (`1.0 / RunConfig::fixed_hz` natively, 1/60 in the
    /// wasm worker; no prior frame to measure from). Subsequent calls reflect
    /// actual elapsed time, so a 50fps frame gets dt ≈ 0.02 and a stutter-frame
    /// at 15fps gets dt ≈ 0.066.
    pub dt: f32,
    /// `true` if egui is consuming pointer or keyboard input this frame (a widget is
    /// hovered, focused, or accepting text). Gameplay code should gate movement /
    /// mouselook on `!ctx.ui_has_focus` so typing into a settings field doesn't also
    /// fire WASD or rotate the camera.
    pub ui_has_focus: bool,
    /// Phantom for forward-compat: future fields here mustn't silently break code that
    /// pattern-matches on the struct.
    _non_exhaustive: PhantomData<()>,
}

// ---------------------------------------------------------------------------
// RunConfig
// ---------------------------------------------------------------------------

/// Default catch-up ticks a single frame may run before the accumulator's
/// excess is dropped. The native runner reads
/// [`RunConfig::max_ticks_per_frame`], which starts here; the wasm worker
/// hardcodes this constant, since `RunConfig` never crosses the worker's init
/// message, so overriding the cap changes the native stall cadence only.
///
/// Not the only cap constant: [`loam_time::DEFAULT_MAX_CATCH_UP`] is
/// `FixedTimestep`'s own default at a different value, and both runners
/// override it, so it is never the effective cap here.
pub const DEFAULT_MAX_TICKS_PER_FRAME: u32 = 4;

/// [`Args`] key naming the sim worker budget: `--threads=N` natively,
/// `?threads=N` on the page.
///
/// `Args` rather than [`RunConfig`] because `RunConfig` does not cross the
/// wasm worker's postMessage boundary while the page query is forwarded into
/// the worker, so this is the only surface that means the same thing on both
/// runners.
const SIM_THREADS_KEY: &str = "threads";

/// Resolve the sim worker budget. Called once per process, before
/// `App::setup`, and never again: a mid-run change would make the schedule a
/// function of when the operator typed it, so a replay from tick 0 would stop
/// reproducing the run. That is why this is not a console command while
/// `fps` and `vsync` are.
///
/// `Args` beats `configured` ([`RunConfig::sim_threads`], the programmatic
/// default a demo can pin without an argv). Either at `0`, or neither
/// present, asks the runner to pick.
pub(crate) fn resolve_sim_threads(args: &Args, configured: Option<usize>) -> usize {
    let requested = match args.get(SIM_THREADS_KEY) {
        Some(raw) => raw.parse::<usize>().ok().or_else(|| {
            tracing::warn!("--{SIM_THREADS_KEY}={raw} is not a worker count; ignoring");
            None
        }),
        None => {
            // The value a bare `--threads N` meant to carry stays a
            // positional and is dropped, so silence here would read as "the
            // user asked for the default".
            if args.has_bare_flag(SIM_THREADS_KEY) {
                tracing::warn!(
                    "--{SIM_THREADS_KEY} needs an attached value (--{SIM_THREADS_KEY}=N); ignoring"
                );
            }
            None
        }
    };
    match requested.or(configured) {
        Some(n) if n > 0 => n,
        _ => default_sim_threads(),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn default_sim_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// One browser thread, and `frame_trace`'s thread-local state is sound only
/// because of it. A platform fact, not a policy.
#[cfg(target_arch = "wasm32")]
fn default_sim_threads() -> usize {
    1
}

/// Runtime knobs. New fields land with defaults so adding configuration is non-breaking.
pub struct RunConfig {
    pub window: WindowAttributes,
    /// Simulation rate; `App::tick` receives `dt = 1.0 / fixed_hz`. Native
    /// only: `RunConfig` does not cross the worker's postMessage boundary, so
    /// the wasm build simulates at 60Hz whatever this says.
    pub fixed_hz: u32,
    /// Spiral-of-death cap, applied by the native runner's [`FixedTimestep`].
    /// Ticks beyond this in one frame are dropped, not deferred; `0` stops the
    /// sim entirely. Native only: the wasm worker hardcodes
    /// [`DEFAULT_MAX_TICKS_PER_FRAME`] whatever this says.
    pub max_ticks_per_frame: u32,
    /// Programmatic sim worker budget, overridden by `--threads=N` /
    /// `?threads=N`. `None` or `Some(0)` lets the runner pick
    /// (`available_parallelism` natively, 1 on wasm32). Reaches `App::setup`
    /// as [`SetupCtx::sim_threads`] and each tick as [`TickCtx::jobs`].
    pub sim_threads: Option<usize>,
    /// `EnvFilter`-style log filter. `None` means keep whatever `tracing-subscriber`
    /// was already configured with (or the `RUST_LOG` env var); `Some` installs a new
    /// global default subscriber.
    pub log_filter: Option<String>,
    /// When true (default) the framework exits the event loop on `Esc`. Apps that bind
    /// `Esc` to a gameplay action (pause, menu, modal dismiss) set this to false and
    /// handle the key inside [`App::on_event`].
    pub esc_exits: bool,
    /// Bail out after this many consecutive [`App::render`] errors. The last error
    /// surfaces back through [`run_with_config`]'s `Result` instead of looping forever
    /// on a wedged GPU. Reset to zero on any successful frame. `0` disables the budget.
    pub render_error_budget: u32,
    /// Bail out after this many consecutive `wgpu::SurfaceError` results from
    /// `begin_frame`. Larger than [`Self::render_error_budget`] because a sleep /
    /// resume cycle on Windows / DX12 routinely takes several frames to settle
    /// (the surface returns `Outdated` or `Other` until the driver finishes
    /// rebuilding the swapchain). Counts all error variants except
    /// `OutOfMemory`, which exits immediately. Reset to zero on any successful
    /// `begin_frame`. `0` disables the budget.
    pub surface_error_budget: u32,
    /// MSAA sample count requested for the scene render target; the UI pass is
    /// single-sampled whatever this says. `1` disables MSAA. `4` is the
    /// conventional default (good quality / cost tradeoff, supported on every
    /// consumer GPU). Higher counts (8, 16) cost more and yield diminishing
    /// returns on edge antialiasing. The runtime negotiates with the adapter; if the
    /// requested count isn't supported on the chosen surface format, [`RenderDevice`]
    /// falls back to the highest supported lower count and logs a warning.
    pub msaa_samples: u32,
    /// Wasm-specific knobs (DOM IDs the page exposes). Ignored on native. The
    /// defaults match the standard layout loam demos use (`loam-canvas-host`
    /// container, `loam-launch` button, `loam-canvas` canvas); demos that ship
    /// a different HTML layout override here.
    pub wasm: WasmConfig,
}

/// Wasm-only configuration knobs: the DOM element IDs the page uses to host
/// the demo. Defaults match the standard layout in the engine's example
/// `index.html` templates. Demos that need custom IDs override the fields
/// they care about and `..WasmConfig::default()` the rest.
#[derive(Clone)]
pub struct WasmConfig {
    /// Container element with `data-mode="manual"`. Determines whether the
    /// demo enters click-to-start mode (vs auto-launch on page load).
    pub host_id: String,
    /// Launch button. Click handler transfers the canvas to a worker.
    pub button_id: String,
    /// Canvas the worker renders into. The element is
    /// `transferControlToOffscreen()`-ed to the worker on click.
    pub canvas_id: String,
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            host_id: "loam-canvas-host".into(),
            button_id: "loam-launch".into(),
            canvas_id: "loam-canvas".into(),
        }
    }
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            window: WindowAttributes::default()
                .with_title("loam app")
                .with_visible(false),
            fixed_hz: 60,
            max_ticks_per_frame: DEFAULT_MAX_TICKS_PER_FRAME,
            sim_threads: None,
            log_filter: None,
            esc_exits: true,
            render_error_budget: 8,
            surface_error_budget: 32,
            msaa_samples: 1,
            wasm: WasmConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Run a demo. The unified entry point that handles native + wasm
/// (both main-thread fallback and worker mode) in one call.
///
/// On native, this is equivalent to [`run_with_config`]: the function
/// blocks until the event loop exits and returns the deferred error
/// (or `Ok(())`).
///
/// On wasm32:
/// - When invoked inside a `DedicatedWorkerGlobalScope` (worker
///   context), routes to `wasm::worker::run`. Drives the App's
///   lifecycle on a worker-side RAF loop with the `wasm::worker_ui::WorkerUi`
///   egui integration.
/// - When invoked on main thread AND the page's `host_id` element has
///   `data-mode="manual"`, routes to `wasm::launch_on_click`. Wires
///   the launch button to spawn the worker on click.
/// - When invoked on main thread WITHOUT manual mode, falls back to
///   [`run_with_config`] (the legacy windowed-mode wasm path).
///
/// The single function call replaces ~8 lines of dispatch boilerplate
/// in each demo's `main()`. Demos that need finer control over the
/// dispatch (e.g. inspecting `wasm::is_worker_context()` for setup-time
/// side effects) can still call the lower-level entry points directly.
pub fn run<A: App + 'static>(config: RunConfig) -> anyhow::Result<()> {
    #[cfg(target_arch = "wasm32")]
    {
        if wasm::is_worker_context() {
            return wasm::worker::run::<A>();
        }
        if wasm::launch::is_manual_mode(&config.wasm.host_id) {
            return wasm::launch_on_click(
                &config.wasm.host_id,
                &config.wasm.button_id,
                &config.wasm.canvas_id,
            );
        }
        // Fall through to main-thread auto-launch (legacy wasm path).
    }
    run_with_config::<A>(config)
}

/// Run an app with custom config.
///
/// On native the function blocks until the event loop exits, then returns whatever
/// error the runner deferred (or `Ok(())`). On wasm32 there is no blocking event loop;
/// the function returns `Ok(())` synchronously after handing the runner off to
/// `EventLoopExtWebSys::spawn_app`, which keeps a reference alive on the JS heap and
/// drives the loop via `requestAnimationFrame`. The browser's JS runtime owns
/// lifecycle from that point; deferred errors are surfaced through `tracing::error!`
/// (and the console panic hook for unwinding) rather than a return value.
pub fn run_with_config<A: App>(config: RunConfig) -> anyhow::Result<()> {
    // Compose two tracing layers: the standard fmt layer (writes to stdout) and our
    // ConsoleLayer (pushes events into the in-process ring buffer for the dev
    // console). Both subscribe to the same EnvFilter so RUST_LOG / log_filter
    // controls both outputs uniformly. `try_init` is best-effort: if a subscriber is
    // already installed (tests, repeated calls) we silently no-op so the existing
    // sink keeps working.
    //
    // On wasm32 stdout doesn't exist and the env-filter has no `RUST_LOG` to read; we
    // route tracing events into the browser console via `tracing-wasm` instead, and
    // install the panic hook so a Rust panic surfaces a useful stack trace in
    // devtools rather than the default `unreachable executed` from `wasm-bindgen`.
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        tracing_wasm::set_as_global_default();
        // Wire `performance.memory.usedJSHeapSize` (Chromium-only) into
        // frame_trace so each completed frame carries a signed heap delta and
        // spike-warns include `heap_delta=+24.5MB` style annotations. On
        // Firefox / Safari the sampler returns `None` and the field stays
        // empty (no misleading reads).
        loam_time::frame_trace::set_heap_sampler(wasm::js_heap_sampler);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        let filter = match &config.log_filter {
            Some(s) => tracing_subscriber::EnvFilter::new(s.clone()),
            None => tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        };
        let _ = tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer())
            .with(log::ConsoleLayer)
            .try_init();
    }

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    // The one read of the thread knob in the process. Everything downstream
    // reads the resolved count back off the pool.
    let jobs = JobPool::new(resolve_sim_threads(&Args::current(), config.sim_threads));
    let runner = Runner::<A>::new(config, jobs);

    #[cfg(target_arch = "wasm32")]
    {
        // `spawn_app` consumes the runner, parks it on the JS heap, and returns
        // immediately. Errors from the runner (deferred via `self.deferred_error` on
        // setup / render failure) are not visible to this call site because there's no
        // return path to bubble them up to JS; they surface through `tracing::error!`
        // -> browser console instead.
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(runner);
        Ok(())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut runner = runner;
        event_loop.run_app(&mut runner)?;
        runner.finish()
    }
}

// ---------------------------------------------------------------------------
// Runner: internal `ApplicationHandler` impl
// ---------------------------------------------------------------------------

/// Everything the runner needs after device acquisition has completed. Bundled so the
/// native (synchronous) and wasm32 (async via `spawn_local`) paths can both produce a
/// single value that the runner installs into its own fields atomically.
struct InitArtifacts<A: App> {
    rd: RenderDevice,
    shader_db: ShaderDb,
    watcher: Option<AssetWatcher>,
    ui: UiIntegration,
    app: A,
}

/// The UI pass is single-sampled on every surface path: an MSAA scene attachment is
/// resolved into the swapchain before egui paints (see
/// [`RenderDevice::resolve_scene_to_swap`]), so egui never draws multisampled and never
/// carries a resolve.
const UI_PASS_SAMPLE_COUNT: u32 = 1;

/// Format and sample count are the contract between `RenderDevice` (which owns the color
/// attachments and the views onto them) and `UiIntegration` (which builds egui pipelines
/// against exactly one of each). Building `UiIntegration` here, in the same place
/// A::setup runs, keeps that pairing colocated so the two can't drift.
///
/// Free function (not a method on Runner) so it can be called from the wasm `spawn_local`
/// closure where `&mut Runner` isn't available across the await point.
fn setup_after_device<A: App>(
    win: &Arc<Window>,
    rd: RenderDevice,
    jobs: &JobPool,
) -> anyhow::Result<InitArtifacts<A>> {
    let mut shader_db = ShaderDb::new(rd.device.clone());

    // AssetWatcher init failure isn't fatal: apps still work without hot-reload. Log
    // and proceed. On wasm32 the watcher is a no-op stub (see `loam-asset`'s watcher.rs)
    // so this always succeeds and the `.warn` branch is dead code in the browser.
    let mut watcher = match AssetWatcher::new() {
        Ok(w) => Some(w),
        Err(e) => {
            tracing::warn!("AssetWatcher disabled: {e}");
            None
        }
    };

    let mut ctx = SetupCtx {
        rd: &rd,
        shader_db: &mut shader_db,
        watcher: watcher.as_mut(),
        time: 0.0,
        // Read off the pool rather than stored beside it, so the integer an
        // app writes into its sim cannot drift from the one the pool
        // partitions against.
        sim_threads: jobs.threads(),
    };
    let app = A::setup(&mut ctx).map_err(|e| e.context("App::setup"))?;

    // Format is `ui_format`, not `target_format`, because on the direct-to-swapchain
    // paths the UI pass draws through the swapchain's non-sRGB reinterpretation
    // rather than the view the scene pass uses; on the composite path the two
    // formats coincide.
    let mut ui = UiIntegration::new(&rd.device, win, rd.ui_format(), UI_PASS_SAMPLE_COUNT);

    // Runner-side pipeline warming (N3). Forces lazy pipeline compilation for
    // egui-wgpu's shape variants and the browser-WebGPU composite pass during
    // setup, instead of stalling the user's first visible frame for ~50-200ms
    // per first-touched pipeline. App-owned pipelines are warmed inside the
    // app's `setup` (e.g. `tesseract_demo::warm_pipelines`); these two cover
    // the runner-owned ones every demo benefits from.
    //
    // Architectural note: lives here (in setup_after_device, after both ui +
    // rd exist but before the first redraw) so it's truly part of the setup
    // step, not a per-frame check + first-frame-flag pattern. A demo that
    // skips the runner (does its own event loop) would also skip this warm,
    // but no loam demo does that today.
    ui.warm_pipelines(
        &rd.device,
        &rd.queue,
        win,
        rd.ui_format(),
        UI_PASS_SAMPLE_COUNT,
    );
    rd.warm_composite();

    Ok(InitArtifacts {
        rd,
        shader_db,
        watcher,
        ui,
        app,
    })
}

/// On wasm32 the device-acquisition future runs to completion in a JS microtask and
/// hands its result back to the runner through this slot. The runner polls the slot at
/// the top of every event-loop callback (window_event + redraw) and installs the
/// artifacts when they appear.
///
/// `Rc<RefCell<...>>` is fine: wasm32 is single-threaded, the future and the runner
/// never borrow the cell simultaneously (the future borrows it once, on completion, to
/// move the result in; the runner borrows it on each callback to try to take the
/// result out).
#[cfg(target_arch = "wasm32")]
type PendingInit<A> = std::rc::Rc<std::cell::RefCell<Option<anyhow::Result<InitArtifacts<A>>>>>;

/// Attach the canvas backing the winit window to the page's DOM. Without this the
/// canvas exists only as a JS object the surface can target but nothing the user can
/// see; appending it makes the render output visible and lets pointer / keyboard
/// events flow through.
///
/// Host element selection prefers `#loam-canvas-host` when the page provides one
/// (Trunk-generated pages typically have a dedicated container so CSS can layout
/// around the canvas); falls back to `<body>` so a minimal page without any host
/// element still works. Canvas style is set to fill its parent so a flex / grid /
/// percentage-sized container drives the surface size; the next resize observer
/// hookup (TODO) will then forward `ResizeObserver` fires to `winit::WindowEvent`.
#[cfg(target_arch = "wasm32")]
fn attach_canvas_to_dom(win: &winit::window::Window) -> anyhow::Result<()> {
    use winit::platform::web::WindowExtWebSys;

    let canvas = win
        .canvas()
        .ok_or_else(|| anyhow::anyhow!("winit window has no canvas (wasm32 only)"))?;

    let web_window =
        web_sys::window().ok_or_else(|| anyhow::anyhow!("no global `window` object"))?;
    let document = web_window
        .document()
        .ok_or_else(|| anyhow::anyhow!("no `document` on global window"))?;

    let host: web_sys::Element = match document.get_element_by_id("loam-canvas-host") {
        Some(el) => el,
        None => document.body().map(Into::into).ok_or_else(|| {
            anyhow::anyhow!("no canvas host: page is missing both `#loam-canvas-host` and `<body>`")
        })?,
    };

    // Fill the host. Without these the canvas keeps winit's default intrinsic size
    // (typically 1024x768) which usually disagrees with the page layout.
    let style = canvas.style();
    let _ = style.set_property("width", "100%");
    let _ = style.set_property("height", "100%");
    let _ = style.set_property("display", "block");

    host.append_child(&canvas)
        .map_err(|e| anyhow::anyhow!("append canvas to host: {e:?}"))?;

    Ok(())
}

struct Runner<A: App> {
    config: RunConfig,

    /// Resolved once in [`run_with_config`] and immutable for the process.
    /// The only storage of the worker budget: `SetupCtx::sim_threads` and
    /// `TickCtx::jobs` both read it from here.
    jobs: JobPool,
    timestep: FixedTimestep,
    input: InputState,
    start: Instant,

    // Lazy-init: created in `resumed`.
    window: Option<Arc<Window>>,
    rd: Option<RenderDevice>,
    shader_db: Option<ShaderDb>,
    watcher: Option<AssetWatcher>,
    ui: Option<UiIntegration>,
    app: Option<A>,

    /// Wasm32-only: present while the spawned device-acquisition future is in flight;
    /// taken on completion. `None` after the first successful poll (or before `resumed`
    /// has fired). See `PendingInit` for the design rationale.
    #[cfg(target_arch = "wasm32")]
    pending_init: Option<PendingInit<A>>,

    minimized: bool,

    // FPS bookkeeping.
    last_fps_update: Instant,
    frame_count: u32,
    fps: f32,

    /// Timestamp of the previous `App::update` call, used to compute `FrameCtx::dt`
    /// (wall-clock elapsed since the last update). `None` before the first frame so
    /// the first `dt` falls back to the fixed-timestep interval.
    last_update_at: Option<Instant>,

    tick_index: u64,
    /// Timestamp of the previous `redraw` entry. Used by the [`frame_pacing`]
    /// throttle at the top of each `redraw` to decide whether to sleep (native)
    /// or skip-and-rerequest (wasm) before doing this frame's work. `None`
    /// disables throttling until the first frame establishes a reference.
    last_redraw_at: Option<Instant>,
    /// Consecutive `App::render` failures since the last successful frame. Compared
    /// against `RunConfig::render_error_budget`.
    render_error_streak: u32,
    /// Consecutive `wgpu::SurfaceError` results from `begin_frame`, since the
    /// last successful frame. Compared against `RunConfig::surface_error_budget`.
    /// Sleep / resume on DX12 routinely produces several frames of `Outdated` or
    /// `Other` before the swapchain rebuilds; the streak only matters if it
    /// doesn't recover.
    surface_error_streak: u32,
    /// Wall-clock instant of the last surface-error log line. Used to rate-limit
    /// the `tracing::error!` for `SurfaceError::Other` to ~1 Hz so a stuck
    /// surface doesn't spew thousands of log lines (and the corresponding
    /// allocations) per second.
    last_surface_error_log: Option<Instant>,
    /// Surfaced to the user via `finish()` if the runner exited because of a setup or
    /// render error, so callers can propagate it from `main`.
    deferred_error: Option<anyhow::Error>,

    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
    capture: capture::Capture,
}

impl<A: App> Runner<A> {
    fn new(config: RunConfig, jobs: JobPool) -> Self {
        let timestep =
            FixedTimestep::new(config.fixed_hz).with_max_catch_up(config.max_ticks_per_frame);
        Self {
            config,
            jobs,
            timestep,
            input: InputState::default(),
            start: Instant::now(),
            window: None,
            rd: None,
            shader_db: None,
            watcher: None,
            ui: None,
            app: None,
            #[cfg(target_arch = "wasm32")]
            pending_init: None,
            minimized: false,
            last_fps_update: Instant::now(),
            frame_count: 0,
            fps: 0.0,
            last_update_at: None,
            tick_index: 0,
            last_redraw_at: None,
            render_error_streak: 0,
            surface_error_streak: 0,
            last_surface_error_log: None,
            deferred_error: None,

            #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
            capture: capture::Capture::new(),
        }
    }

    /// Drain any error that the runner deferred during the event loop (setup or render
    /// failures cause `elwt.exit()` so the loop returns `Ok`; we surface the real error
    /// here).
    ///
    /// Native-only: on wasm32 the runner is consumed by
    /// `EventLoopExtWebSys::spawn_app` and its lifetime is owned by the JS heap, so
    /// there's no return path to surface deferred errors through. They bubble up via
    /// `tracing::error!` -> the browser console instead.
    #[cfg(not(target_arch = "wasm32"))]
    fn finish(self) -> anyhow::Result<()> {
        match self.deferred_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    fn time(&self) -> f32 {
        self.start.elapsed().as_secs_f32()
    }

    /// Install artifacts produced by `setup_after_device`. Shared by both the native
    /// synchronous path (called from `resumed` directly) and the wasm32 async path
    /// (called from `poll_pending_init` when the future resolves).
    fn install_init(&mut self, win: Arc<Window>, artifacts: InitArtifacts<A>) {
        self.window = Some(win.clone());
        self.rd = Some(artifacts.rd);
        self.shader_db = Some(artifacts.shader_db);
        self.watcher = artifacts.watcher;
        self.ui = Some(artifacts.ui);
        self.app = Some(artifacts.app);
        self.minimized = false;
        self.start = Instant::now();
        self.last_fps_update = Instant::now();

        win.set_visible(true);
        win.request_redraw();
    }

    /// Wasm32-only: try to install the deferred init artifacts. Returns `true` if the
    /// state transitioned from Loading -> Ready (or Loading -> Failed) on this call.
    /// Called at the top of every event-loop callback so the runner can finish
    /// constructing its state as soon as `RenderDevice::new` resolves.
    #[cfg(target_arch = "wasm32")]
    fn poll_pending_init(&mut self, elwt: &ActiveEventLoop) -> bool {
        let Some(cell) = self.pending_init.as_ref() else {
            return false;
        };
        let Some(result) = cell.borrow_mut().take() else {
            return false;
        };
        // Drop the shared cell now that we've consumed its payload; the future itself
        // has already completed.
        self.pending_init = None;
        let Some(win) = self.window.clone() else {
            // Shouldn't happen: `resumed` always sets `self.window` before spawning the
            // future, but be defensive.
            self.deferred_error = Some(anyhow::anyhow!(
                "wasm init future resolved with no window present",
            ));
            elwt.exit();
            return true;
        };
        match result {
            Ok(artifacts) => {
                self.install_init(win, artifacts);
                true
            }
            Err(e) => {
                self.deferred_error = Some(e);
                elwt.exit();
                true
            }
        }
    }
}

/// Read back `texture` and hand the pixels to the capture state machine, which
/// dispatches to the active writer (one-shot PNG, sequence PNG, or GIF encoder).
/// Logs and swallows errors so a transient capture failure doesn't abort the render
/// loop. Free function (not a method on Runner) so the borrow checker can see that
/// `&mut capture` and `&rd` are disjoint borrows.
#[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
fn capture_consume(
    capture: &mut capture::Capture,
    rd: &RenderDevice,
    texture: &wgpu::Texture,
    is_pre: bool,
    captured_at: Instant,
) {
    let img = match capture::read_texture_rgba(
        &rd.device,
        &rd.queue,
        texture,
        rd.surface_bundle.size.width,
        rd.surface_bundle.size.height,
        rd.surface_bundle.config.format,
    ) {
        Ok(i) => i,
        Err(e) => {
            tracing::error!("capture: readback failed: {e:#}");
            return;
        }
    };
    if let Err(e) = capture.consume_frame(is_pre, img.rgba, img.width, img.height, captured_at) {
        tracing::error!("capture: write failed: {e:#}");
    }
}

impl<A: App> ApplicationHandler for Runner<A> {
    #[cfg(target_arch = "wasm32")]
    fn resumed(&mut self, elwt: &ActiveEventLoop) {
        // Wasm32 init is a two-phase dance:
        //
        //   1. Synchronous prelude: create the winit `Window` (this is sync on every
        //      platform; on wasm it constructs an `HtmlCanvasElement` but does NOT
        //      attach it to the DOM, so we do that explicitly). Then we hand a clone
        //      of `Arc<Window>` to the async tail.
        //   2. Async tail: `RenderDevice::new` awaits `request_adapter` /
        //      `request_device`, both of which are JS-promise-backed on wasm. We hand
        //      that future to `wasm_bindgen_futures::spawn_local` and write the
        //      eventual `InitArtifacts` into a shared `Rc<RefCell<...>>` slot.
        //
        // `poll_pending_init` (called at the top of every event-loop callback) then
        // drains the slot and installs the artifacts into `self`.
        //
        // `with_prevent_default(false)` tells winit not to call
        // `event.preventDefault()` on every keyboard / mouse / wheel event the canvas
        // receives. Default-on capture made Ctrl+R / F12 / Ctrl+Shift+I unreachable
        // (the canvas swallowed them before browser chrome could see them); turning
        // it off means winit only consumes the events it actually translates into
        // `WindowEvent`s. App-relevant keys (Esc, arrows, etc.) still flow through
        // the input system because winit listens on those passively.
        use winit::platform::web::WindowAttributesExtWebSys;
        let attrs = self.config.window.clone().with_prevent_default(false);
        let win = match elwt.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                self.deferred_error = Some(anyhow::anyhow!("create_window: {e}"));
                elwt.exit();
                return;
            }
        };

        if let Err(e) = attach_canvas_to_dom(&win) {
            self.deferred_error = Some(e.context("attach canvas to DOM"));
            elwt.exit();
            return;
        }

        let msaa = self.config.msaa_samples;
        let jobs = self.jobs;
        let win_for_future = win.clone();
        let cell: PendingInit<A> = std::rc::Rc::new(std::cell::RefCell::new(None));
        let cell_for_future = cell.clone();

        self.window = Some(win);
        self.pending_init = Some(cell);

        wasm_bindgen_futures::spawn_local(async move {
            let result = async {
                let rd = RenderDevice::new(win_for_future.clone(), msaa)
                    .await
                    .map_err(|e| anyhow::anyhow!("RenderDevice::new: {e:#}"))?;
                setup_after_device::<A>(&win_for_future, rd, &jobs)
            }
            .await;
            *cell_for_future.borrow_mut() = Some(result);
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn resumed(&mut self, elwt: &ActiveEventLoop) {
        let win = match elwt.create_window(self.config.window.clone()) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                self.deferred_error = Some(anyhow::anyhow!("create_window: {e}"));
                elwt.exit();
                return;
            }
        };

        let rd = match pollster::block_on(RenderDevice::new(win.clone(), self.config.msaa_samples))
        {
            Ok(r) => r,
            Err(e) => {
                self.deferred_error = Some(anyhow::anyhow!("RenderDevice::new: {e:#}"));
                elwt.exit();
                return;
            }
        };
        // for debugging negotiated MSAA count; the actual count is in `rd.sample_count()`
        // tracing::info!("scale_factor: {}, msaa: {}x", win.scale_factor(), rd.sample_count());

        let artifacts = match setup_after_device::<A>(&win, rd, &self.jobs) {
            Ok(a) => a,
            Err(e) => {
                self.deferred_error = Some(e);
                elwt.exit();
                return;
            }
        };

        self.install_init(win, artifacts);
    }

    fn window_event(
        &mut self,
        elwt: &ActiveEventLoop,
        _id: winit::window::WindowId,
        ev: WindowEvent,
    ) {
        // Wasm32: drain the deferred-init slot if the spawned device-acquisition future
        // has resolved. On native this is a no-op (the slot doesn't exist).
        #[cfg(target_arch = "wasm32")]
        let _installed = self.poll_pending_init(elwt);

        let Some(win) = self.window.clone() else {
            return;
        };

        // Event-correlation diagnostic. When `log events on` has been issued
        // (see loam_app::log), every meaningful WindowEvent emits a
        // tracing::info! so the spike investigation can cross-reference
        // browser events with the spike-warn timestamps. Cursor-moves are
        // filtered because they fire at 60Hz+ and drown out the signal;
        // everything else goes through. Lives BEFORE the egui forward so
        // RedrawRequested doesn't spam (RedrawRequested fires every frame
        // and would obscure the rare events we care about).
        if log::events_enabled() {
            match &ev {
                // Suppress per-frame noise.
                WindowEvent::CursorMoved { .. }
                | WindowEvent::RedrawRequested
                | WindowEvent::AxisMotion { .. } => {}
                other => {
                    tracing::info!("WindowEvent: {other:?}");
                }
            }
        }

        // Forward to egui first so it can claim hover/focus/clicks
        // before Loam's own routing translates the event for gameplay.
        // egui consuming the event is informational; Loam still sees it.
        if let Some(ui) = self.ui.as_mut() {
            let _ = ui.handle_event(&win, &ev);
        }

        // Esc / close: exit cleanly. (When egui has keyboard focus,
        // e.g. a TextEdit is active, swallow Esc so it dismisses the
        // edit instead of exiting the app.)
        match &ev {
            WindowEvent::CloseRequested => {
                elwt.exit();
                return;
            }
            WindowEvent::KeyboardInput { event, .. }
                if self.config.esc_exits
                    && event.state == ElementState::Pressed
                    && matches!(event.logical_key, Key::Named(NamedKey::Escape))
                    && !self.ui.as_ref().is_some_and(|u| u.ui_has_focus()) =>
            {
                elwt.exit();
                return;
            }
            _ => {}
        }

        // Always route input *first*, before user `on_event` sees
        // it. Means apps can read derived state (e.g. via
        // `FrameCtx::input`) without re-implementing routing.
        match &ev {
            WindowEvent::KeyboardInput { event, .. } => {
                self.input.key_input(event.physical_key, event.state);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input.cursor_moved(position.x, position.y);
            }
            WindowEvent::CursorLeft { .. } => self.input.cursor_invalidated(),
            WindowEvent::Focused(false) => {
                self.input.cursor_invalidated();
                self.input.release_buttons();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.input.mouse_input(*button, *state);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.input.mouse_wheel(*delta);
            }
            WindowEvent::Resized(size) => {
                let was_minimized = self.minimized;
                self.minimized = size.width == 0 || size.height == 0;
                match (was_minimized, self.minimized) {
                    // Just got minimized: park the event loop in `Wait` so we
                    // stop burning CPU on poll iterations that have no work to
                    // do. Without this the loop spins at 100% even though
                    // `redraw` early-returns, which is what makes laptop fans
                    // go nuts on a minimized demo window.
                    (false, true) => elwt.set_control_flow(ControlFlow::Wait),
                    // Just restored: switch back to `Poll` so the redraw loop
                    // re-engages at vsync rate, reconfigure the swapchain at
                    // the new size, and kick off the first redraw. The
                    // `FixedTimestep` spiral cap will discard any accumulated
                    // catch-up so we don't queue hours of ticks.
                    (true, false) => {
                        elwt.set_control_flow(ControlFlow::Poll);
                        if let Some(rd) = &mut self.rd {
                            rd.resize(*size);
                        }
                        win.request_redraw();
                    }
                    // Visible-to-visible resize: just reconfigure the
                    // swapchain. Minimized-to-minimized is impossible (winit
                    // wouldn't emit it) but harmless.
                    (false, false) => {
                        if let Some(rd) = &mut self.rd {
                            rd.resize(*size);
                        }
                    }
                    (true, true) => {}
                }
            }
            _ => {}
        }

        // Notify user of the event *after* our routing has settled.
        if let WindowEvent::RedrawRequested = ev {
            self.redraw(elwt, &win);
            return;
        }

        let now = self.time();
        let fps = self.fps;
        let tick = self.tick_index;
        if let Some(app) = self.app.as_mut() {
            if let Some(rd) = self.rd.as_ref() {
                let ui_has_focus = self.ui.as_ref().is_some_and(|u| u.ui_has_focus());
                let mut ctx = FrameCtx {
                    rd,
                    input: FrameInput::default(),
                    time: now,
                    fps,
                    n_ticks: 0,
                    tick,
                    // dt isn't meaningful for input events (they fire whenever the OS
                    // delivers, not on a frame cadence). Zero is the least-surprising
                    // value; apps that integrate continuous state should do that work
                    // in `update`, not `on_event`.
                    dt: 0.0,
                    ui_has_focus,
                    _non_exhaustive: PhantomData,
                };
                app.on_event(&ev, &mut ctx);
                // Mirror the wasm worker's contract: route keyboard events
                // through `on_key` too, so demos with a single hotkey impl
                // (in `on_key`) work the same on both runners.
                if let WindowEvent::KeyboardInput { event, .. } = &ev {
                    if let winit::keyboard::PhysicalKey::Code(code) = event.physical_key {
                        app.on_key(code, event.state, &mut ctx);
                    }
                }
            }
        }
    }

    /// Fires after every event batch when `ControlFlow::Poll` is set. On wasm32 we use
    /// it to drain the deferred-init slot: the spawned device-acquisition future may
    /// have resolved between callbacks, and without a user-driven event the runner
    /// would otherwise sit idle. Once installed, the normal redraw cycle takes over.
    fn about_to_wait(&mut self, _elwt: &ActiveEventLoop) {
        #[cfg(target_arch = "wasm32")]
        {
            if self.pending_init.is_some() {
                self.poll_pending_init(_elwt);
            }
        }
    }

    /// Device-level events (mouse motion independent of cursor position,
    /// raw scroll wheel, etc.). The only one we currently consume is
    /// `MouseMotion`, which gives uncapped deltas that don't stop when the
    /// cursor hits the screen edge. Routed into `InputState::accumulate_raw_motion`
    /// so consumers (camera controllers, primarily) that grab the cursor can
    /// read it via `FrameInput::mouse_raw_delta`.
    fn device_event(
        &mut self,
        _elwt: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        ev: winit::event::DeviceEvent,
    ) {
        if let winit::event::DeviceEvent::MouseMotion { delta } = ev {
            self.input.accumulate_raw_motion(delta.0, delta.1);
        }
    }
}

/// Section names the frame loops open directly inside their `frame` scope.
/// [`crate::trace`] subtracts these from `frame` to report the remainder as
/// `unscoped`; a scope added to a frame loop and not to this list lands there
/// instead of under its own name.
///
/// Only direct children belong here. Sections a demo opens nest inside
/// `app-record` or `app-ui`, so subtracting this list cannot double-count.
/// `between-frames` and `idle` bracket the frame rather than nesting in it,
/// and `gpu-total` is device time recorded out of band.
pub(crate) const FRAME_LOOP_SECTIONS: &[&str] = &[
    "sim-ticks",
    "app-update",
    "app-ui",
    "hot-reload",
    "surface-acquire",
    "app-record",
    "scene-resolve",
    "ui-paint",
    "composite",
    "present",
];

impl<A: App> Runner<A> {
    fn redraw(&mut self, elwt: &ActiveEventLoop, win: &Arc<Window>) {
        if self.minimized {
            return;
        }
        // A `--script` run holds no `ActiveEventLoop`, so it publishes
        // completion here instead. Read before the frame's work so the last
        // scripted frame is the last one presented.
        if script::exit_requested() {
            elwt.exit();
            return;
        }
        // Pending vsync transitions from the `vsync` console command. Applied
        // here so the next `begin_frame` picks up the new present mode. The
        // off-side picks the best non-Fifo mode the adapter advertised:
        // `Mailbox` first (triple-buffered, no tearing), `Immediate` as
        // fallback (single-buffered, tearing allowed). If neither is offered
        // (typical browser surface), the request silently no-ops; surface
        // configuration is the wrong layer to surface an error in that case.
        if let (Some(want_on), Some(rd)) = (frame_pacing::take_pending_vsync(), self.rd.as_mut()) {
            let target = if want_on {
                wgpu::PresentMode::Fifo
            } else {
                let modes = rd.supported_present_modes();
                if modes.contains(&wgpu::PresentMode::Mailbox) {
                    wgpu::PresentMode::Mailbox
                } else if modes.contains(&wgpu::PresentMode::Immediate) {
                    wgpu::PresentMode::Immediate
                } else {
                    rd.present_mode()
                }
            };
            let _ = rd.set_present_mode(target);
        }

        // Pending cursor grab + visibility transitions from
        // `loam_app::cursor`. Native only; browser Pointer Lock requires a
        // recent user gesture that console commands don't satisfy (the
        // cursor module emits a one-time tracing::warn on wasm).
        //
        // Grab + visibility are independent. The runner translates the
        // engine-level `GrabMode` to winit's `CursorGrabMode` and falls
        // back from `Locked` to `Confined` (or vice versa) if the platform
        // doesn't support the requested mode (macOS + Linux/X11 vary on
        // which they prefer; Windows accepts both). The fallback chain
        // preserves the user's intent (some form of grab) rather than
        // silently dropping to `None`.
        #[cfg(not(target_arch = "wasm32"))]
        {
            use winit::window::CursorGrabMode;
            let (pending_grab, pending_vis) = cursor::take_pending();
            let mut applied = cursor::current_state();
            if let Some(mode) = pending_grab {
                let primary = match mode {
                    cursor::GrabMode::None => CursorGrabMode::None,
                    cursor::GrabMode::Confined => CursorGrabMode::Confined,
                    cursor::GrabMode::Locked => CursorGrabMode::Locked,
                };
                if win.set_cursor_grab(primary).is_err() && mode != cursor::GrabMode::None {
                    let fallback = match mode {
                        cursor::GrabMode::Locked => CursorGrabMode::Confined,
                        cursor::GrabMode::Confined => CursorGrabMode::Locked,
                        cursor::GrabMode::None => CursorGrabMode::None,
                    };
                    let _ = win.set_cursor_grab(fallback);
                }
                applied.grab = mode;
            }
            if let Some(visible) = pending_vis {
                win.set_cursor_visible(visible);
                applied.visible = visible;
            }
            if pending_grab.is_some() || pending_vis.is_some() {
                cursor::mark_applied(applied.grab, applied.visible);
            }
            // Warp-to-center request. Applied AFTER the grab/visibility
            // transition so the new cursor state is in effect first; warping
            // a still-Locked cursor would be a no-op (winit pins it to the
            // center already), but pairing the warp with a release lands the
            // pointer where the user was aiming when they Alt-tabbed out.
            if cursor::take_pending_warp_center() {
                let size = win.inner_size();
                let center = winit::dpi::PhysicalPosition::new(
                    size.width as f64 / 2.0,
                    size.height as f64 / 2.0,
                );
                let _ = win.set_cursor_position(center);
            }
        }

        let Some(rd) = self.rd.as_ref() else { return };

        // Frame-rate cap. The `fps` console command pokes
        // [`frame_pacing::set_target_fps`]; we read it here. Cap is enforced
        // differently per target; native does a precise sleep up to the
        // deadline; wasm skips the RAF callback and re-requests, since we
        // can't block in the browser. With `target_fps = 0` the load returns
        // `None` and we fall through to the surface's native cadence (vsync
        // on native, RAF on wasm).
        //
        // We anchor the deadline on the previous frame's ideal start (not on
        // the actual wake-up) so the cadence stays locked to the period even
        // if individual frames overshoot. If we ran long (work + present took
        // longer than the period), we set `last_redraw_at = now` to "catch
        // up" instead of falling further behind on every subsequent frame.
        let now = Instant::now();
        let frame_anchor = if let (Some(period), Some(last)) =
            (frame_pacing::target_period(), self.last_redraw_at)
        {
            let deadline = last + period;
            if now < deadline {
                #[cfg(target_arch = "wasm32")]
                {
                    win.request_redraw();
                    return;
                }
                // Native: sleep out the remainder and anchor on the
                // ideal deadline, not the wake-up time.
                #[cfg(not(target_arch = "wasm32"))]
                {
                    frame_pacing::precise_sleep_until(deadline);
                    deadline
                }
            } else {
                now
            }
        } else {
            now
        };
        self.last_redraw_at = Some(frame_anchor);

        // Mark frame start for `idle` measurement. `end_frame` subtracts this from
        // the previous `end_frame` timestamp to get `idle` (browser/RAF gap, not
        // our work); separate from `between-frames` (total cadence, our work +
        // idle combined).
        loam_time::frame_trace::begin_frame();

        // Whole-redraw scope. Subsequent named sections sum to less than this; the
        // delta is the small bits in between (FPS bookkeeping, capture status
        // publish, etc.). Used by the `trace` console command to surface "total
        // CPU work this frame" vs. "what's the dominant section."
        let _frame_scope = loam_time::frame_trace::scope("frame");

        // 1. Command queue, drained immediately before the ticks and stamped
        // with the index they start from, so a command's position in sim time
        // is a property of the recording and not of how many catch-up ticks
        // this frame happens to run. The worker drains at the same place for
        // the same reason.
        if let Some(app) = self.app.as_mut() {
            command::apply_drained(app, rd, self.tick_index);
        }

        // 2. Fixed-timestep ticks, through the same `drive_fixed_ticks` the
        // wasm worker uses. The accumulator logic is shared; the rate is not,
        // since the worker hardcodes 60Hz and cannot read `RunConfig`.
        let n_ticks = if let Some(app) = self.app.as_mut() {
            drive_fixed_ticks(
                app,
                &mut self.timestep,
                &mut self.tick_index,
                Instant::now(),
                self.config.fixed_hz,
                &self.jobs,
            )
        } else {
            0
        };

        // 3. Per-frame update with drained input + UI build.
        // egui's focus reading reflects the *previous* frame's state
        // (egui hasn't run yet for this frame). That one-frame
        // staleness is fine: focus changes one frame at a time, and
        // `App::update` needs to know "should I gate gameplay input"
        // before this frame's UI runs.
        let ui_has_focus = self.ui.as_ref().is_some_and(|u| u.ui_has_focus());
        let input = self.input.take_frame();

        // Compute dt (wall-clock seconds since previous update). First frame after
        // setup has no prior `last_update_at`, so we seed with the fixed-timestep
        // interval; better than 0.0, which would zero out any dt-driven animation
        // on its very first integration step.
        let now_inst = Instant::now();
        let dt = match self.last_update_at {
            Some(prev) => now_inst.saturating_duration_since(prev).as_secs_f32(),
            None => 1.0 / self.config.fixed_hz as f32,
        };
        self.last_update_at = Some(now_inst);

        if let Some(app) = self.app.as_mut() {
            let mut fctx = FrameCtx {
                rd,
                input,
                time: self.start.elapsed().as_secs_f32(),
                fps: self.fps,
                n_ticks,
                tick: self.tick_index,
                dt,
                ui_has_focus,
                _non_exhaustive: PhantomData,
            };
            {
                let _scope = loam_time::frame_trace::scope("app-update");
                app.update(&mut fctx);
            }

            // Build this frame's UI. egui captures the widgets;
            // `paint` later renders them after `App::render`.
            if let Some(ui) = self.ui.as_mut() {
                let _scope = loam_time::frame_trace::scope("app-ui");
                let egui_ctx = ui.begin_frame(win.as_ref()).clone();
                app.ui(&egui_ctx, &mut fctx);
            }
        }

        // 4. Hot-reload poll.
        {
            let _scope = loam_time::frame_trace::scope("hot-reload");
            let reload_events = self.watcher.as_mut().map(|w| w.poll()).unwrap_or_default();
            if !reload_events.is_empty() {
                if let (Some(app), Some(shader_db), Some(rd)) =
                    (self.app.as_mut(), self.shader_db.as_mut(), self.rd.as_ref())
                {
                    app.apply_shader_events(&reload_events, shader_db);
                    let mut ctx = SetupCtx {
                        rd,
                        shader_db,
                        watcher: self.watcher.as_mut(),
                        time: self.start.elapsed().as_secs_f32(),
                        sim_threads: self.jobs.threads(),
                    };
                    app.on_shader_reload(&mut ctx);
                }
            }
        }

        // 5. FPS + title (rate-limited to ~1 Hz).
        self.frame_count += 1;
        let elapsed = self.last_fps_update.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            self.fps = self.frame_count as f32 / elapsed;
            self.frame_count = 0;
            self.last_fps_update = Instant::now();
            if let Some(app) = self.app.as_ref() {
                let title = app.title(self.fps);
                // Append capture status when active. 1 Hz refresh matches the title
                // update cadence and gives the user a visible recording counter without
                // wiring it through the demo's UI.
                #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
                let title = match self.capture.status() {
                    Some(status) => format!("{title} [{status}]").into(),
                    None => title,
                };
                win.set_title(&title);
            }
        }

        // 6. Drain any queued capture requests + update the state machine BEFORE the
        // render pass. Requests come from console commands and hotkey binds; they're
        // applied here so this frame can honor them.
        #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
        {
            let requests = capture::drain_requests();
            if !requests.is_empty() {
                let log = self.capture.apply_requests(requests);
                for line in log {
                    tracing::info!("{line}");
                }
            }
        }

        // 7. Render: scene (App::record) then UI overlay.
        //
        // Three attachment topologies, one per surface path. On the two
        // direct-to-swapchain paths the swapchain texture is addressed through
        // two views, an sRGB one for whatever the scene writes into it and its
        // non-sRGB reinterpretation for the UI pass, so egui blends in the
        // gamma space its feathering assumes; where the adapter forbids the
        // reinterpretation the UI takes the sRGB view too and those two
        // topologies are otherwise unchanged.
        //
        //   sRGB swap, MSAA on:  scene into `rd.msaa_view()`, then
        //     `resolve_scene_to_swap` resolves that attachment into the
        //     `begin_frame` view, sRGB at both ends. UI into the reinterpreted
        //     view of that same swapchain texture, single-sampled.
        //   sRGB swap, MSAA off: scene into the `begin_frame` swapchain view,
        //     UI into the reinterpreted view of that same texture. Nothing
        //     resolves.
        //   non-sRGB swap (browser-WebGPU): no reinterpretation and one view
        //     per attachment. Scene and UI both draw into `rd.scene_view()`,
        //     the offscreen scene texture, and the end-of-frame composite pass
        //     gamma-encodes that into the `begin_frame` swapchain view, the
        //     only pass writing through it here. `RenderDevice` forces MSAA off
        //     on this path, so nothing resolves.
        //
        // Capture taps read the swapchain texture, which on the composite path
        // carries none of the frame until `composite_to_swap` runs. Each tap
        // therefore orders a composite ahead of its readback, so it gets
        // gamma-encoded pixels of the stage it names rather than an untouched
        // surface.
        //   - `pre`-egui:  after App::record and the scene resolve, before ui.paint.
        //     The pre tap reads just the 3D pass output.
        //   - `post`-egui: after ui.paint and the frame's composite, before
        //     frame.present. Reads scene + UI. This is what DWM receives.
        // FPS-gate decides whether either tap fires this frame. Computed once before the
        // render pass so the same `now` is used to schedule the next capture interval.
        #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
        let capture_now = Instant::now();
        #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
        let do_capture = self.capture.should_capture(capture_now);

        // Scoped because this is where a vsync-locked frame spends most of
        // its wall time and an unscoped block here is indistinguishable from
        // engine work in the trace table. Under `PresentMode::Fifo` the
        // presentation engine holds the acquire until a swapchain image frees
        // at the next flip, so the backpressure lands here, not in `present`,
        // which only queues the flip and returns.
        let begin_result = {
            let _scope = loam_time::frame_trace::scope("surface-acquire");
            rd.begin_frame()
        };
        if begin_result.is_ok() {
            self.surface_error_streak = 0;
            self.last_surface_error_log = None;
        }
        match begin_result {
            Ok((frame, swap_view)) => {
                let mut last_err: Option<anyhow::Error> = None;
                // Scene-pass target, per the topology above: the multisampled
                // attachment if there is one, else the offscreen scene texture
                // on the composite path, else the swapchain view. The UI pass
                // picks its own target below.
                let render_view = rd.msaa_view().or(rd.scene_view()).unwrap_or(&swap_view);

                // GPU timer start. Tiny dedicated encoder so the timestamp lands in
                // the queue before any of the frame's submitted work. Stays separate
                // from the main frame encoder so the start timestamp is on the GPU
                // BEFORE we begin recording scene passes; merging them would put the
                // start timestamp at the end of the same submit as the work, ruining
                // the measurement. Same logic for the end timer below. No-op when
                // the adapter didn't advertise TIMESTAMP_QUERY.
                if let Some(timer) = rd.gpu_timer.as_ref() {
                    let mut t_enc =
                        rd.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("loam-app::gpu-timer-start"),
                            });
                    timer.write_start(&mut t_enc);
                    rd.queue.submit(Some(t_enc.finish()));
                }

                // THE frame encoder. App::record, ui.paint, and the composite pass
                // all write into this single encoder; the runner submits it once
                // before `frame.present`. The capture taps (native only) need
                // intermediate submits to make readback see the right pixels; those
                // paths split the encoder mid-frame to maintain correctness, at the
                // cost of the single-submit win in capture-active frames.
                let mut encoder =
                    rd.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("loam-app::frame"),
                        });

                if let Some(app) = self.app.as_mut() {
                    let _scope = loam_time::frame_trace::scope("app-record");
                    let mut ctx = RenderCtx {
                        rd,
                        view: render_view,
                        encoder: &mut encoder,
                    };
                    if let Err(e) = app.record(&mut ctx) {
                        tracing::error!("App::record error: {e:#}");
                        last_err = Some(e);
                    }
                }

                // Scene MSAA resolve, in its own pass because `App::record` owns
                // the scene passes. Branch rather than lean on the method's own
                // guard so the MSAA-off steady state pays nothing.
                if rd.sample_count() > 1 {
                    let _scope = loam_time::frame_trace::scope("scene-resolve");
                    rd.resolve_scene_to_swap(&mut encoder, &swap_view);
                }

                // Pre-egui capture tap. Forces a mid-frame submit so the GPU has
                // actually drawn the scene before we read it back; we restart the
                // encoder afterwards for ui+composite.
                #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
                if do_capture && self.capture.wants_pre() {
                    // Nothing has written the swapchain yet on the composite
                    // path; the scene lives in the offscreen target. Encoding
                    // it now is exactly the scene-only image this stage wants,
                    // and the frame's own composite overwrites it after the UI
                    // pass. No-op on the direct paths, where the scene pass (or
                    // the resolve above) already wrote the swapchain.
                    if rd.scene_view().is_some() {
                        rd.composite_to_swap(&mut encoder, &swap_view);
                    }
                    rd.queue.submit(Some(encoder.finish()));
                    encoder = rd
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("loam-app::frame-post-pre-capture"),
                        });
                    capture_consume(&mut self.capture, rd, &frame.texture, true, capture_now);
                }

                if let Some(ui) = self.ui.as_mut() {
                    let _scope = loam_time::frame_trace::scope("ui-paint");
                    let viewport = (rd.surface_bundle.size.width, rd.surface_bundle.size.height);
                    // Direct-to-swapchain paths blend the UI in gamma space via
                    // the swapchain's non-sRGB reinterpretation
                    // (RenderDevice::ui_format). The composite path keeps
                    // painting into the scene texture, which the later composite
                    // pass consumes. Either target is single-sampled, so the UI
                    // pass never carries a resolve.
                    let ui_swap_view =
                        (rd.scene_view().is_none()).then(|| rd.create_ui_swap_view(&frame));
                    let ui_view = match &ui_swap_view {
                        Some(swap) => swap,
                        None => render_view,
                    };
                    ui.paint(
                        &rd.device,
                        &rd.queue,
                        &mut encoder,
                        ui_view,
                        None,
                        win.as_ref(),
                        viewport,
                    );
                }

                // Composite pass: offscreen scene texture -> linear swapchain
                // with manual gamma encoding. Writes into the same encoder as
                // everything else (no separate submit). Ordered before the post
                // tap because that tap reads the swapchain and this is the only
                // pass that writes it on this path. No-op on the direct paths
                // (scene_view() is None and rendering wrote the swapchain).
                if rd.scene_view().is_some() {
                    let _scope = loam_time::frame_trace::scope("composite");
                    rd.composite_to_swap(&mut encoder, &swap_view);
                }

                // Post-egui capture tap. Like the pre-tap, forces a mid-frame submit
                // before the readback so the frame's passes have actually run on the
                // GPU. Restart the encoder so the submit below still has one.
                #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
                if do_capture && self.capture.wants_post() {
                    rd.queue.submit(Some(encoder.finish()));
                    encoder = rd
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("loam-app::frame-post-post-capture"),
                        });
                    capture_consume(&mut self.capture, rd, &frame.texture, false, capture_now);
                }
                #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
                if do_capture {
                    self.capture.advance_frame(capture_now);
                }
                // Publish status every frame so the panel + window title stay current
                // even when do_capture is false (FPS-gated idle frames between writes).
                #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
                capture::publish_status(self.capture.status());

                // GPU timer end + resolve. Stays in a separate small encoder for the
                // same reason as the start timer: ordering vs the frame's main work.
                if let Some(timer) = rd.gpu_timer.as_ref() {
                    rd.queue.submit(Some(encoder.finish()));
                    let mut t_enc =
                        rd.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("loam-app::gpu-timer-end"),
                            });
                    timer.write_end_and_resolve(&mut t_enc);
                    rd.queue.submit(Some(t_enc.finish()));
                } else {
                    // Single submit for the whole frame (the path we're optimizing
                    // for on wasm + native-without-timestamps).
                    rd.queue.submit(Some(encoder.finish()));
                }

                {
                    let _scope = loam_time::frame_trace::scope("present");
                    frame.present();
                }

                // Advance the GPU timer's frame index + drain any completed timings
                // into frame_trace. The slot whose end-timestamp was just resolved
                // gets its map_async scheduled here; the result lands in frame_trace
                // 1-2 frames later via the channel.
                if let Some(timer) = self.rd.as_mut().and_then(|rd| rd.gpu_timer.as_mut()) {
                    timer.tick();
                }
                if let Some(err) = last_err {
                    self.render_error_streak = self.render_error_streak.saturating_add(1);
                    let budget = self.config.render_error_budget;
                    if budget > 0 && self.render_error_streak >= budget {
                        self.deferred_error = Some(err.context(format!(
                            "App::render failed {budget} consecutive frames; aborting"
                        )));
                        elwt.exit();
                        return;
                    }
                } else {
                    self.render_error_streak = 0;
                }
                win.request_redraw();
            }
            Err(err) => {
                // `OutOfMemory` is terminal; bail without budgeting (any retry
                // would just allocate again and fail again, faster).
                if matches!(err, wgpu::SurfaceError::OutOfMemory) {
                    self.deferred_error = Some(anyhow::anyhow!("wgpu surface out of memory"));
                    elwt.exit();
                    return;
                }

                // Recovery: re-configure the swapchain at the current size for
                // the `Lost`/`Outdated`/`Other` variants. `Other` is what DX12
                // returns after sleep/resume when the swapchain is wedged;
                // reconfiguring lets wgpu rebuild it. `Timeout` is transient
                // (driver took too long for one frame) so don't reconfigure.
                match err {
                    wgpu::SurfaceError::Lost
                    | wgpu::SurfaceError::Outdated
                    | wgpu::SurfaceError::Other => {
                        if let Some(rd) = &mut self.rd {
                            let size = rd.surface_bundle.size;
                            rd.resize(size);
                        }
                    }
                    _ => {}
                }

                // Rate-limit the `Other` log to ~1 Hz so a persistently wedged
                // surface doesn't spew thousands of lines/sec (and the
                // corresponding tracing allocations). Other variants are rare
                // enough that the unthrottled debug-level log is fine.
                if matches!(err, wgpu::SurfaceError::Other) {
                    let now = Instant::now();
                    let should_log = self
                        .last_surface_error_log
                        .map(|t| now.duration_since(t).as_secs_f32() >= 1.0)
                        .unwrap_or(true);
                    if should_log {
                        tracing::error!("surface error: {err:?}");
                        self.last_surface_error_log = Some(now);
                    }
                } else {
                    tracing::debug!("surface error: {err:?}");
                }

                self.surface_error_streak = self.surface_error_streak.saturating_add(1);
                let budget = self.config.surface_error_budget;
                if budget > 0 && self.surface_error_streak >= budget {
                    self.deferred_error = Some(anyhow::anyhow!(
                        "wgpu surface error persisted {budget} consecutive frames: {err:?}"
                    ));
                    elwt.exit();
                    return;
                }
                win.request_redraw();
            }
        }

        // End the frame's trace. Must happen after _frame_scope drops (i.e. at the
        // very end of redraw); the surrounding block scope ensures that ordering.
        drop(_frame_scope);
        loam_time::frame_trace::end_frame();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// One tick of the 60 Hz accumulator, as `FixedTimestep` stores it
    /// (nanoseconds derived from the target Hz, truncated).
    const TICK: Duration = Duration::from_nanos(1_000_000_000 / 60);

    #[derive(Default)]
    struct TickRecorder {
        times: Vec<f32>,
        /// Worker budget observed through `TickCtx`, one entry per tick, so a
        /// pool that changed or went missing mid-run is visible.
        workers: Vec<usize>,
    }

    impl App for TickRecorder {
        fn setup(_ctx: &mut SetupCtx<'_>) -> anyhow::Result<Self> {
            Ok(Self::default())
        }

        fn tick(&mut self, _dt: f32, ctx: &mut TickCtx<'_>) {
            self.times.push(ctx.time);
            self.workers.push(ctx.jobs.threads());
        }
    }

    /// Drive frames whose wall-clock instants are `base + offsets[i]`, mirroring
    /// how the runners wire the cap into the accumulator. Returns the
    /// `TickCtx::time` values the app observed, the accumulator, and the
    /// runner-side tick counter. The first offset only primes the accumulator.
    fn drive(
        base: Instant,
        offsets: &[Duration],
        max_catch_up: u32,
    ) -> (Vec<f32>, FixedTimestep, u64) {
        let mut app = TickRecorder::default();
        let mut timestep = FixedTimestep::new(60).with_max_catch_up(max_catch_up);
        let mut tick_index = 0u64;
        let jobs = JobPool::new(1);
        for offset in offsets {
            drive_fixed_ticks(
                &mut app,
                &mut timestep,
                &mut tick_index,
                base + *offset,
                60,
                &jobs,
            );
        }
        (app.times, timestep, tick_index)
    }

    fn tick_times(base: Instant, offsets: &[Duration], max_catch_up: u32) -> Vec<f32> {
        drive(base, offsets, max_catch_up).0
    }

    #[test]
    fn tick_time_sequence_is_independent_of_wall_clock_offset() {
        let offsets: Vec<Duration> = (0..=20).map(|k| TICK * k).collect();
        let early = tick_times(Instant::now(), &offsets, 8);
        let late = tick_times(Instant::now() + Duration::from_secs(3_600), &offsets, 8);

        assert_eq!(
            early.len(),
            20,
            "one tick per frame after the priming frame"
        );
        assert_eq!(
            early, late,
            "tick time must not shift with the run's wall-clock origin"
        );
    }

    #[test]
    fn tick_time_sequence_is_independent_of_frame_pacing() {
        let base = Instant::now();
        let one_per_frame: Vec<Duration> = (0..=60).map(|k| TICK * k).collect();
        let ten_per_frame: Vec<Duration> = (0..=6).map(|k| TICK * (k * 10)).collect();

        let smooth = tick_times(base, &one_per_frame, 10);
        let stuttered = tick_times(base, &ten_per_frame, 10);

        let expected: Vec<f32> = (0..60).map(|i| i as f32 * (1.0 / 60.0)).collect();
        assert_eq!(smooth, expected, "tick time is tick_index * dt from zero");
        assert_eq!(
            stuttered, expected,
            "catching up ten ticks in one frame must yield the same time sequence"
        );
    }

    #[test]
    fn executed_ticks_equal_ticks_charged_to_the_accumulator() {
        // Prime, then a frame arriving ten ticks late under a cap of exactly
        // ten, so all ten are charged and all ten must run: any second cap
        // inside the tick loop books ticks it never simulates.
        const BACKLOG: u32 = 10;
        let offsets = [Duration::ZERO, TICK * BACKLOG];
        let (times, timestep, tick_index) = drive(Instant::now(), &offsets, BACKLOG);

        assert_eq!(
            times.len() as u64,
            timestep.tick(),
            "every tick charged to the accumulator must have run App::tick"
        );
        assert_eq!(times.len(), BACKLOG as usize);
        assert_eq!(tick_index, timestep.tick());
    }

    #[test]
    fn the_runner_caps_catch_up_in_the_accumulator_not_the_tick_loop() {
        let config = RunConfig {
            max_ticks_per_frame: 2,
            ..RunConfig::default()
        };
        let mut runner = Runner::<TickRecorder>::new(config, JobPool::new(1));
        let base = Instant::now();
        runner.timestep.advance(base);
        let ticks = runner.timestep.advance(base + TICK * 10);
        assert_eq!(
            ticks.end - ticks.start,
            2,
            "RunConfig::max_ticks_per_frame must reach the accumulator, \
             which is the only place the cap may be applied"
        );
    }

    #[test]
    fn timestep_tick_and_runner_index_stay_equal_across_a_stall() {
        // Two cap sizes, since a cap re-applied downstream stays invisible
        // whenever the accumulator's own cap is the smaller of the two.
        for cap in [DEFAULT_MAX_TICKS_PER_FRAME, loam_time::DEFAULT_MAX_CATCH_UP] {
            let base = Instant::now();
            let mut app = TickRecorder::default();
            let mut timestep = FixedTimestep::new(60).with_max_catch_up(cap);
            let mut tick_index = 0u64;
            let jobs = JobPool::new(1);

            // Prime, three steady frames, a 30-tick stall, three steady frames.
            let steps = [
                Duration::ZERO,
                TICK,
                TICK,
                TICK,
                TICK * 30,
                TICK,
                TICK,
                TICK,
            ];
            let mut elapsed = Duration::ZERO;
            for step in steps {
                elapsed += step;
                drive_fixed_ticks(
                    &mut app,
                    &mut timestep,
                    &mut tick_index,
                    base + elapsed,
                    60,
                    &jobs,
                );
                assert_eq!(
                    tick_index,
                    timestep.tick(),
                    "cap {cap}: runner tick_index diverged from the accumulator at {elapsed:?}"
                );
                assert_eq!(
                    app.times.len() as u64,
                    timestep.tick(),
                    "cap {cap}: a booked tick was never simulated at {elapsed:?}"
                );
            }

            let expected_ticks = 3 + u64::from(cap) + 3;
            assert_eq!(timestep.tick(), expected_ticks);
            // Dropped backlog must not punch a hole in the sim-time sequence:
            // the stall costs wall-clock time, not tick indices.
            let expected_times: Vec<f32> = (0..expected_ticks)
                .map(|i| i as f32 * (1.0 / 60.0))
                .collect();
            assert_eq!(app.times, expected_times, "cap {cap}");
        }
    }

    #[test]
    fn the_thread_budget_prefers_args_and_reads_zero_as_let_the_runner_pick() {
        let picked = default_sim_threads();
        assert!(picked >= 1, "the runner's own pick must be a legal budget");
        // Offset off the pick rather than literal, so no assertion below can
        // pass by coinciding with the core count of the machine running it.
        let flagged = picked + 1;
        let configured = picked + 2;
        let flagged_arg = flagged.to_string();

        let flag = |value: &str| Args::from_pairs([(SIM_THREADS_KEY, value)]);
        assert_eq!(resolve_sim_threads(&flag(&flagged_arg), None), flagged);
        assert_eq!(
            resolve_sim_threads(&flag(&flagged_arg), Some(configured)),
            flagged,
            "the flag must win over RunConfig, not the other way round"
        );
        assert_eq!(
            resolve_sim_threads(&Args::default(), Some(configured)),
            configured
        );
        assert_eq!(resolve_sim_threads(&Args::default(), None), picked);
        assert_eq!(resolve_sim_threads(&flag("0"), Some(configured)), picked);
        assert_eq!(resolve_sim_threads(&Args::default(), Some(0)), picked);
        // A value that is not a count must fall through to the next source,
        // never be rounded into one.
        assert_eq!(
            resolve_sim_threads(&flag("many"), Some(configured)),
            configured
        );
        assert_eq!(
            resolve_sim_threads(
                &Args::from_argv(["--threads", &flagged_arg]),
                Some(configured)
            ),
            configured,
            "a bare flag drops its value, so it must not read as a request"
        );
    }

    #[test]
    fn every_tick_of_a_run_sees_the_one_pool_the_runner_resolved() {
        // Offset off the platform's pick, so a path that quietly re-resolved
        // instead of reading the runner's pool shows a different number on
        // any machine rather than only on one whose core count differs.
        let budget = default_sim_threads() + 1;

        let jobs = JobPool::new(resolve_sim_threads(
            &Args::from_pairs([(SIM_THREADS_KEY, budget.to_string())]),
            Some(budget + 1),
        ));
        let mut runner = Runner::<TickRecorder>::new(RunConfig::default(), jobs);
        assert_eq!(
            runner.jobs.threads(),
            budget,
            "the resolved budget must be what the runner stores"
        );

        let base = Instant::now();
        let mut app = TickRecorder::default();
        let mut tick_index = 0u64;
        for frame in 0..=5u32 {
            drive_fixed_ticks(
                &mut app,
                &mut runner.timestep,
                &mut tick_index,
                base + TICK * frame,
                runner.config.fixed_hz,
                &runner.jobs,
            );
        }

        assert_eq!(app.workers.len(), 5, "the frames must have produced ticks");
        assert_eq!(
            app.workers,
            vec![budget; app.workers.len()],
            "the pool a tick borrows is the runner's, at the resolved budget"
        );
    }

    // -----------------------------------------------------------------
    // Command queue ordering
    // -----------------------------------------------------------------

    /// What the frame loop did, in the order it did it. The drain and the tick
    /// loop write to one log so their interleaving is the thing under test.
    #[derive(Clone, Debug, PartialEq, Eq)]
    enum FrameEvent {
        Applied { stamp: u64, line: String },
        Ticked(u64),
    }

    #[derive(Default)]
    struct EventRecorder {
        log: std::rc::Rc<std::cell::RefCell<Vec<FrameEvent>>>,
    }

    impl App for EventRecorder {
        fn setup(_ctx: &mut SetupCtx<'_>) -> anyhow::Result<Self> {
            Ok(Self::default())
        }

        fn tick(&mut self, _dt: f32, ctx: &mut TickCtx<'_>) {
            self.log.borrow_mut().push(FrameEvent::Ticked(ctx.tick));
        }
    }

    /// Replay the two runners' frame order: drain and record, then tick. One
    /// command is submitted per frame, standing in for the console line or
    /// bound key that would have been typed during the previous frame's UI.
    /// That the runners really call in this order is a separate assertion,
    /// `command::tests::each_runner_drains_once_and_before_its_ticks`.
    fn drive_with_commands(offsets: &[Duration], max_catch_up: u32) -> Vec<FrameEvent> {
        let _held = command::TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = command::drain(0);

        let mut app = EventRecorder::default();
        let log = app.log.clone();
        let mut timestep = FixedTimestep::new(60).with_max_catch_up(max_catch_up);
        let mut tick_index = 0u64;
        let jobs = JobPool::new(1);
        let base = Instant::now();
        for (frame, offset) in offsets.iter().enumerate() {
            for stamped in command::drain(tick_index) {
                log.borrow_mut().push(FrameEvent::Applied {
                    stamp: stamped.tick,
                    line: stamped.command.name.clone(),
                });
            }
            drive_fixed_ticks(
                &mut app,
                &mut timestep,
                &mut tick_index,
                base + *offset,
                60,
                &jobs,
            );
            command::submit_line(&format!("mark{frame}"));
        }
        let _ = command::drain(tick_index);
        let events = log.borrow().clone();
        events
    }

    /// The contract the whole queue exists for: a command stamped T is applied
    /// after tick T-1 and before tick T, whatever the pacing. Dispatching in
    /// the UI phase, as the console used to, puts the mutation after the
    /// frame's whole tick batch instead, and the batch size is a wall-clock
    /// accident.
    #[test]
    fn a_command_applies_after_the_previous_tick_and_before_the_one_it_is_stamped_for() {
        let smooth: Vec<Duration> = (0..=40).map(|k| TICK * k).collect();
        let stuttered: Vec<Duration> = (0..=8).map(|k| TICK * (k * 5)).collect();
        // Two-thirds of a tick per frame: most frames run none, some run one.
        let subtick: Vec<Duration> = (0..=40).map(|k| (TICK * 2 * k) / 3).collect();

        for (name, offsets) in [
            ("one tick per frame", smooth),
            ("five ticks per frame", stuttered),
            ("two thirds of a tick per frame", subtick),
        ] {
            let events = drive_with_commands(&offsets, 10);
            assert!(
                events
                    .iter()
                    .any(|e| matches!(e, FrameEvent::Applied { .. })),
                "{name}: nothing was applied, so the assertions below are vacuous"
            );
            let mut last_tick: Option<u64> = None;
            for (i, event) in events.iter().enumerate() {
                match event {
                    FrameEvent::Ticked(t) => last_tick = Some(*t),
                    FrameEvent::Applied { stamp, line } => {
                        assert_eq!(
                            last_tick.map(|t| t + 1).unwrap_or(0),
                            *stamp,
                            "{name}: `{line}` is stamped {stamp} but the tick before it \
                             was {last_tick:?}"
                        );
                        let next_tick = events[i + 1..].iter().find_map(|e| match e {
                            FrameEvent::Ticked(t) => Some(*t),
                            FrameEvent::Applied { .. } => None,
                        });
                        if let Some(next) = next_tick {
                            assert_eq!(
                                next, *stamp,
                                "{name}: `{line}` stamped {stamp} must be applied before \
                                 tick {stamp}, not before tick {next}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// A frame that runs zero ticks still drains, and its commands carry the
    /// same stamp as the next frame's. Concatenation preserves order, so one
    /// tick's worth of commands split over many frames arrives in submission
    /// order at one boundary.
    #[test]
    fn commands_split_over_zero_tick_frames_reach_one_boundary_in_order() {
        // A quarter tick per frame: three frames of nothing, then one tick.
        let offsets: Vec<Duration> = (0..=12).map(|k| (TICK * k) / 4).collect();
        let events = drive_with_commands(&offsets, 10);

        let batched: Vec<(u64, String)> = events
            .iter()
            .filter_map(|e| match e {
                FrameEvent::Applied { stamp, line } => Some((*stamp, line.clone())),
                FrameEvent::Ticked(_) => None,
            })
            .collect();
        assert!(
            batched.iter().any(|(stamp, _)| batched
                .iter()
                .filter(|(other, _)| other == stamp)
                .count()
                > 1),
            "the pacing must produce at least one stamp carrying several commands"
        );

        // Submission order is the `markN` order, and stamps never go backwards.
        let submitted: Vec<u32> = batched
            .iter()
            .map(|(_, line)| line.trim_start_matches("mark").parse().expect("markN"))
            .collect();
        let mut sorted = submitted.clone();
        sorted.sort_unstable();
        assert_eq!(submitted, sorted, "the queue reordered submissions");
        assert!(
            batched.windows(2).all(|w| w[0].0 <= w[1].0),
            "stamps must be monotone: {batched:?}"
        );
    }
}
