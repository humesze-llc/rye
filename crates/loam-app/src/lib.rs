use std::borrow::Cow;
use std::marker::PhantomData;
use std::sync::Arc;
// `std::time::Instant::now` panics on wasm32, so the swap is mandatory there.
use web_time::Instant;

// Both cfg arms expose one surface, so demos need no gates at the call sites.
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

pub use loam_asset::AssetEvent;
pub use loam_camera::{
    Camera, CameraController, CameraView, FirstPersonController, OrbitController,
};
pub use loam_egui::{egui, world_to_screen, UiCapture};
pub use loam_input::FrameInput as Input;
pub use loam_shader::{ShaderDb, ShaderOwner};

pub trait App: Sized + 'static {
    fn setup(ctx: &mut SetupCtx<'_>) -> anyhow::Result<Self>;

    /// Runs 0..N times per frame, N bounded by the runner's catch-up cap.
    fn tick(&mut self, _dt: f32, _ctx: &mut TickCtx<'_>) {}

    /// Applied before the tick it is stamped for.
    fn apply_command(
        &mut self,
        cmd: &command::CommandLine,
        _ctx: &mut command::CommandCtx<'_>,
    ) -> anyhow::Result<()> {
        anyhow::bail!("no command target for `{}`", cmd.name)
    }

    /// Runs after all the frame's ticks, with the drained input.
    fn update(&mut self, _ctx: &mut FrameCtx<'_>) {}

    /// Not called for keyboard events in the wasm worker, which cannot construct
    /// a `WindowEvent::KeyboardInput`; `on_key` fires on both paths.
    fn on_event(&mut self, _ev: &WindowEvent, _ctx: &mut FrameCtx<'_>) {}

    /// Fired for every press and release, after input routing.
    fn on_key(
        &mut self,
        _code: winit::keyboard::KeyCode,
        _state: ElementState,
        _ctx: &mut FrameCtx<'_>,
    ) {
    }

    fn apply_shader_events(&mut self, events: &[AssetEvent], shader_db: &mut ShaderDb) {
        shader_db.apply_events(ShaderDb::ROOT_OWNER, events);
    }

    /// Runs after `apply_shader_events`; rebuild any stale consumer pipelines.
    fn on_shader_reload(&mut self, _ctx: &mut SetupCtx<'_>) {}

    /// Implement either this or `record`; the runner always calls `record`.
    fn render(&mut self, _rd: &RenderDevice, _view: &wgpu::TextureView) -> anyhow::Result<()> {
        Ok(())
    }

    /// Do NOT call `encoder.finish()` or `queue.submit`; the runner does that once
    /// at end-of-frame. `ctx.view` is already the right view for the platform.
    fn record(&mut self, ctx: &mut RenderCtx<'_>) -> anyhow::Result<()> {
        self.render(ctx.rd, ctx.view)
    }

    /// Runs after `App::update`; painted as a 2D overlay.
    fn ui(&mut self, _ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {}

    /// The runner rate-limits the `set_title` call to ~1 Hz.
    fn title(&self, _fps: f32) -> Cow<'static, str> {
        Cow::Borrowed("loam app")
    }
}

// The catch-up cap lives solely in the `FixedTimestep`: capping again here would
// book ticks the accumulator charged but the sim never ran.
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
        *tick_index = tick + 1;
    }
    n_ticks
}

pub struct SetupCtx<'a> {
    pub rd: &'a RenderDevice,
    pub shader_db: &'a mut ShaderDb,
    /// `None` when filesystem watching failed to init: shaders load, but no
    /// hot-reload.
    pub watcher: Option<&'a mut AssetWatcher>,
    /// Wall-clock seconds since `run`. Always 0 in `setup`.
    pub time: f32,
    /// Resolved once before `setup` and constant for the run.
    pub sim_threads: usize,
}

/// Deliberately GPU-free so sim code stays bit-deterministic.
pub struct TickCtx<'a> {
    /// Derived from the tick index rather than read from the clock, so replaying
    /// the same tick range yields the same bits.
    pub time: f32,
    pub tick: u64,
    /// The one runner-owned resource a deterministic tick may reach: its
    /// partition is a pure function of (unit count, worker count). Its timings
    /// are not, and nothing in a tick may branch on them.
    pub jobs: &'a JobPool,
}

/// Owns the shared frame encoder, reused for ui-paint and the wasm composite and
/// submitted once. `view` is the platform's scene-pass color target.
pub struct RenderCtx<'a> {
    pub rd: &'a RenderDevice,
    pub view: &'a wgpu::TextureView,
    pub encoder: &'a mut wgpu::CommandEncoder,
}

pub struct FrameCtx<'a> {
    pub rd: &'a RenderDevice,
    pub input: FrameInput,
    pub time: f32,
    pub fps: f32,
    pub n_ticks: usize,
    pub tick: u64,
    /// Wall-clock seconds since the previous `App::update`; the first call
    /// after setup reports the fixed-timestep interval instead.
    pub dt: f32,
    pub ui_capture: UiCapture,
    _non_exhaustive: PhantomData<()>,
}

pub const DEFAULT_MAX_TICKS_PER_FRAME: u32 = 4;

const SIM_THREADS_KEY: &str = "threads";

// Resolved once per process: a mid-run change would make the schedule a function
// of when the operator typed it, so a replay from tick 0 would diverge.
pub(crate) fn resolve_sim_threads(args: &Args, configured: Option<usize>) -> usize {
    let requested = match args.get(SIM_THREADS_KEY) {
        Some(raw) => raw.parse::<usize>().ok().or_else(|| {
            tracing::warn!("--{SIM_THREADS_KEY}={raw} is not a worker count; ignoring");
            None
        }),
        None => {
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

// `frame_trace`'s thread-local state is sound only because this stays one.
#[cfg(target_arch = "wasm32")]
fn default_sim_threads() -> usize {
    1
}

pub struct RunConfig {
    pub window: WindowAttributes,
    /// Native only: `RunConfig` does not cross the worker's postMessage
    /// boundary, so the wasm build simulates at 60Hz whatever this says.
    pub fixed_hz: u32,
    /// Ticks beyond this are dropped, not deferred; `0` stops the sim. Native
    /// only: the wasm worker hardcodes [`DEFAULT_MAX_TICKS_PER_FRAME`].
    pub max_ticks_per_frame: u32,
    /// `None` or `Some(0)` lets the runner pick (`available_parallelism`
    /// natively, 1 on wasm32). `--threads=N` / `?threads=N` overrides it.
    pub sim_threads: Option<usize>,
    /// `None` keeps whatever `tracing-subscriber` already installed (or
    /// `RUST_LOG`); `Some` installs a new global default subscriber.
    pub log_filter: Option<String>,
    pub esc_exits: bool,
    /// `0` disables the budget.
    pub render_error_budget: u32,
    /// Larger than [`Self::render_error_budget`] because a Windows / DX12 sleep
    /// and resume cycle takes several frames to settle.
    pub surface_error_budget: u32,
    /// The UI pass is single-sampled whatever this says. `1` disables MSAA.
    pub msaa_samples: u32,
    /// Ignored on native.
    pub wasm: WasmConfig,
}

#[derive(Clone)]
pub struct WasmConfig {
    /// Must carry `data-mode="manual"`; anything else auto-launches on load.
    pub host_id: String,
    /// Its click handler transfers the canvas to a worker.
    pub button_id: String,
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

/// Dispatches native, wasm main-thread, and wasm worker mode.
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
    }
    run_with_config::<A>(config)
}

/// On native this blocks until the event loop exits.
pub fn run_with_config<A: App>(config: RunConfig) -> anyhow::Result<()> {
    // wasm32 has no stdout and no `RUST_LOG`, so tracing goes to the browser
    // console; the panic hook turns `unreachable executed` into a stack trace.
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        tracing_wasm::set_as_global_default();
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

    let jobs = JobPool::new(resolve_sim_threads(&Args::current(), config.sim_threads));
    let runner = Runner::<A>::new(config, jobs);

    #[cfg(target_arch = "wasm32")]
    {
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

struct InitArtifacts<A: App> {
    rd: RenderDevice,
    shader_db: ShaderDb,
    watcher: Option<AssetWatcher>,
    ui: UiIntegration,
    app: A,
}

// The MSAA scene attachment resolves into the swapchain before egui paints.
const UI_PASS_SAMPLE_COUNT: u32 = 1;

// Free function, not a method on Runner, so the wasm `spawn_local` closure can
// call it where `&mut Runner` is not available across the await point.
fn setup_after_device<A: App>(
    win: &Arc<Window>,
    rd: RenderDevice,
    jobs: &JobPool,
) -> anyhow::Result<InitArtifacts<A>> {
    let mut shader_db = ShaderDb::new(rd.device.clone());

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
        sim_threads: jobs.threads(),
    };
    let app = A::setup(&mut ctx).map_err(|e| e.context("App::setup"))?;

    // `ui_format`, not `target_format`: on the direct-to-swapchain paths the UI
    // pass draws through the swapchain's non-sRGB reinterpretation, not the view
    // the scene pass uses. On the composite path the two coincide.
    let mut ui = UiIntegration::new(&rd.device, win, rd.ui_format(), UI_PASS_SAMPLE_COUNT);

    // Compiles egui-wgpu's shape variants and the browser-WebGPU composite pass
    // now, rather than stalling the first visible frame ~50-200ms per pipeline.
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

#[cfg(target_arch = "wasm32")]
type PendingInit<A> = std::rc::Rc<std::cell::RefCell<Option<anyhow::Result<InitArtifacts<A>>>>>;

// Appending the canvas is what makes the render output visible.
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

    // Without these the canvas keeps winit's 1024x768 intrinsic size.
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

    jobs: JobPool,
    timestep: FixedTimestep,
    input: InputState,
    start: Instant,

    window: Option<Arc<Window>>,
    rd: Option<RenderDevice>,
    shader_db: Option<ShaderDb>,
    watcher: Option<AssetWatcher>,
    ui: Option<UiIntegration>,
    app: Option<A>,

    /// Read at each frame's `begin_frame`. Window events fire between frames,
    /// so `on_event` / `on_key` and the Esc gate all serve this frame's value.
    ui_capture: UiCapture,

    #[cfg(target_arch = "wasm32")]
    pending_init: Option<PendingInit<A>>,

    minimized: bool,

    last_fps_update: Instant,
    frame_count: u32,
    fps: f32,

    last_update_at: Option<Instant>,

    tick_index: u64,
    last_redraw_at: Option<Instant>,
    render_error_streak: u32,
    surface_error_streak: u32,
    last_surface_error_log: Option<Instant>,
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
            ui_capture: UiCapture::default(),
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

    // Setup and render failures call `elwt.exit()`, so the loop returns `Ok`.
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

    #[cfg(target_arch = "wasm32")]
    fn poll_pending_init(&mut self, elwt: &ActiveEventLoop) -> bool {
        let Some(cell) = self.pending_init.as_ref() else {
            return false;
        };
        let Some(result) = cell.borrow_mut().take() else {
            return false;
        };
        self.pending_init = None;
        let Some(win) = self.window.clone() else {
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

// Errors are logged and swallowed so a transient capture failure does not abort
// the render loop. Free function so `&mut capture` and `&rd` stay disjoint.
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
        // Default-on prevent_default made Ctrl+R / F12 / Ctrl+Shift+I unreachable:
        // the canvas swallowed them before browser chrome saw them.
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
        #[cfg(target_arch = "wasm32")]
        let _installed = self.poll_pending_init(elwt);

        let Some(win) = self.window.clone() else {
            return;
        };

        if log::events_enabled() {
            match &ev {
                WindowEvent::CursorMoved { .. }
                | WindowEvent::RedrawRequested
                | WindowEvent::AxisMotion { .. } => {}
                other => {
                    tracing::info!("WindowEvent: {other:?}");
                }
            }
        }

        // Forward to egui first so it claims hover, focus and clicks before
        // Loam's own routing.
        if let Some(ui) = self.ui.as_mut() {
            let _ = ui.handle_event(&win, &ev);
        }

        match &ev {
            WindowEvent::CloseRequested => {
                elwt.exit();
                return;
            }
            WindowEvent::KeyboardInput { event, .. }
                if self.config.esc_exits
                    && event.state == ElementState::Pressed
                    && matches!(event.logical_key, Key::Named(NamedKey::Escape))
                    && !self.ui_capture.keyboard =>
            {
                elwt.exit();
                return;
            }
            _ => {}
        }

        // Always route input *first*, before user `on_event` sees it.
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
                    // Park in `Wait` so a minimized window stops polling.
                    (false, true) => elwt.set_control_flow(ControlFlow::Wait),
                    (true, false) => {
                        elwt.set_control_flow(ControlFlow::Poll);
                        if let Some(rd) = &mut self.rd {
                            rd.resize(*size);
                        }
                        win.request_redraw();
                    }
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

        if let WindowEvent::RedrawRequested = ev {
            self.redraw(elwt, &win);
            return;
        }

        let now = self.time();
        let fps = self.fps;
        let tick = self.tick_index;
        let ui_capture = self.ui_capture;
        if let Some(app) = self.app.as_mut() {
            if let Some(rd) = self.rd.as_ref() {
                let mut ctx = FrameCtx {
                    rd,
                    input: FrameInput::default(),
                    time: now,
                    fps,
                    n_ticks: 0,
                    tick,
                    // dt is meaningless for input events; they fire on OS
                    // delivery, not on a frame cadence.
                    dt: 0.0,
                    ui_capture,
                    _non_exhaustive: PhantomData,
                };
                app.on_event(&ev, &mut ctx);
                // Mirror the wasm worker: keyboard events also reach `on_key`.
                if let WindowEvent::KeyboardInput { event, .. } = &ev {
                    if let winit::keyboard::PhysicalKey::Code(code) = event.physical_key {
                        app.on_key(code, event.state, &mut ctx);
                    }
                }
            }
        }
    }

    // Drains the deferred-init slot: the device-acquisition future may resolve
    // between callbacks, and the runner would otherwise sit idle.
    fn about_to_wait(&mut self, _elwt: &ActiveEventLoop) {
        #[cfg(target_arch = "wasm32")]
        {
            if self.pending_init.is_some() {
                self.poll_pending_init(_elwt);
            }
        }
    }

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

// `crate::trace` subtracts these from `frame` to report the remainder as
// `unscoped`; a frame-loop scope missing here lands in `unscoped` instead.
pub(crate) const FRAME_LOOP_SECTIONS: &[&str] = &[
    "sim-ticks",
    "ui-begin",
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
        // A `--script` run holds no `ActiveEventLoop`, so it publishes completion
        // here. Read before the frame's work so the last scripted frame presents.
        if script::exit_requested() {
            elwt.exit();
            return;
        }
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

        // Native only; browser Pointer Lock needs a recent user gesture that a
        // console command does not satisfy. The `Locked` / `Confined` fallback
        // covers platforms supporting only one (macOS and Linux/X11 vary).
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
            // After the grab transition: warping a still-Locked cursor is a no-op,
            // and pairing the warp with a release lands the pointer where aimed.
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

        // Anchor on the previous frame's ideal start, not the wake-up, so cadence
        // stays locked to the period; a long frame anchors on `now` instead.
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

        loam_time::frame_trace::begin_frame();

        let _frame_scope = loam_time::frame_trace::scope("frame");

        // Drained before the ticks and stamped with the index they start from, so
        // a command's place in sim time is a property of the recording.
        if let Some(app) = self.app.as_mut() {
            command::apply_drained(app, rd, self.tick_index);
        }

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

        // Opened before `App::update` so input hit-tests the last build's layout.
        let egui_ctx = if let Some(ui) = self.ui.as_mut() {
            let _scope = loam_time::frame_trace::scope("ui-begin");
            let ctx = ui.begin_frame(win.as_ref()).clone();
            self.ui_capture = UiCapture::read(&ctx);
            Some(ctx)
        } else {
            None
        };
        let ui_capture = self.ui_capture;
        let input = self.input.take_frame();

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
                ui_capture,
                _non_exhaustive: PhantomData,
            };
            {
                let _scope = loam_time::frame_trace::scope("app-update");
                app.update(&mut fctx);
            }

            if let Some(egui_ctx) = egui_ctx.as_ref() {
                let _scope = loam_time::frame_trace::scope("app-ui");
                app.ui(egui_ctx, &mut fctx);
            }
        }

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

        self.frame_count += 1;
        let elapsed = self.last_fps_update.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            self.fps = self.frame_count as f32 / elapsed;
            self.frame_count = 0;
            self.last_fps_update = Instant::now();
            if let Some(app) = self.app.as_ref() {
                let title = app.title(self.fps);
                #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
                let title = match self.capture.status() {
                    Some(status) => format!("{title} [{status}]").into(),
                    None => title,
                };
                win.set_title(&title);
            }
        }

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

        // On the direct-to-swapchain paths the swapchain is addressed through two
        // views, sRGB for the scene and non-sRGB for the UI pass, so egui blends
        // in the gamma space its feathering assumes. Capture taps read that
        // texture, so each tap orders a composite ahead of its readback.
        #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
        let capture_now = Instant::now();
        #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
        let do_capture = self.capture.should_capture(capture_now);

        // Under `Fifo` the acquire blocks until the next flip, not `present`.
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
                let render_view = rd.msaa_view().or(rd.scene_view()).unwrap_or(&swap_view);

                // Separate encoder so the start timestamp reaches the GPU before
                // the scene passes record, not at the end of the same submit.
                if let Some(timer) = rd.gpu_timer.as_ref() {
                    let mut t_enc =
                        rd.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("loam-app::gpu-timer-start"),
                            });
                    timer.write_start(&mut t_enc);
                    rd.queue.submit(Some(t_enc.finish()));
                }

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

                // Branch here, not in the method's guard, so MSAA-off pays
                // nothing.
                if rd.sample_count() > 1 {
                    let _scope = loam_time::frame_trace::scope("scene-resolve");
                    rd.resolve_scene_to_swap(&mut encoder, &swap_view);
                }

                // Mid-frame submit so the GPU has drawn the scene before the
                // readback.
                #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
                if do_capture && self.capture.wants_pre() {
                    // Nothing has written the swapchain yet on this path.
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

                // Before the post tap: the only pass that writes the swapchain
                // here.
                if rd.scene_view().is_some() {
                    let _scope = loam_time::frame_trace::scope("composite");
                    rd.composite_to_swap(&mut encoder, &swap_view);
                }

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
                #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
                capture::publish_status(self.capture.status());

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
                    rd.queue.submit(Some(encoder.finish()));
                }

                {
                    let _scope = loam_time::frame_trace::scope("present");
                    frame.present();
                }

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
                if matches!(err, wgpu::SurfaceError::OutOfMemory) {
                    self.deferred_error = Some(anyhow::anyhow!("wgpu surface out of memory"));
                    elwt.exit();
                    return;
                }

                // `Other` is what DX12 returns after sleep/resume with a wedged
                // swapchain; `Timeout` is transient, so do not reconfigure.
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

                // Rate-limit to ~1 Hz so a wedged surface cannot spew lines.
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

        drop(_frame_scope);
        loam_time::frame_trace::end_frame();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // Nanoseconds derived from the target Hz and truncated, as `FixedTimestep`
    // stores it.
    const TICK: Duration = Duration::from_nanos(1_000_000_000 / 60);

    #[derive(Default)]
    struct TickRecorder {
        times: Vec<f32>,
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

    // The first offset only primes the accumulator.
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
        // Ten ticks late under a cap of exactly ten: all ten are charged.
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
        // Two cap sizes: a downstream cap hides whenever the accumulator's own
        // cap is the smaller of the two.
        for cap in [DEFAULT_MAX_TICKS_PER_FRAME, loam_time::DEFAULT_MAX_CATCH_UP] {
            let base = Instant::now();
            let mut app = TickRecorder::default();
            let mut timestep = FixedTimestep::new(60).with_max_catch_up(cap);
            let mut tick_index = 0u64;
            let jobs = JobPool::new(1);

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
            // The stall costs wall-clock time, not tick indices.
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
        // Offset off the pick, so no assertion can pass by matching the core
        // count of the machine running it.
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
        // Offset off the platform's pick, so a path that re-resolved instead of
        // reading the runner's pool differs on any machine.
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

    // The drain and the tick loop write to one log; the interleaving is the test.
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

    // Replays the runners' frame order: drain and record, then tick. That they
    // really call in this order is asserted in `command::tests`.
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
