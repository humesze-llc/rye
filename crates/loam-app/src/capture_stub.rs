//! Stub capture API for `wasm32` and `--no-default-features` builds.
//!
//! The real capture pipeline (`capture.rs`) is desktop-only because it leans on
//! `std::fs::File`, `std::thread::spawn`, and a small forest of native-only PNG / GIF /
//! APNG encoder crates. This module mirrors the public API as no-ops so consumers
//! (`loam_app::capture::CapturePanel`, `register_commands`, `bind_default_hotkeys`, etc.)
//! compile against both targets without `cfg`-littering their call sites.

use std::path::PathBuf;

use loam_egui::Console;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CaptureStage {
    Pre,
    Post,
    Both,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CaptureFormat {
    Png,
    Gif,
    Apng,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum PaletteMode {
    #[default]
    Local,
    Global,
}

#[derive(Debug)]
pub enum CaptureRequest {
    OneShot {
        stage: CaptureStage,
        dir: Option<PathBuf>,
        name: Option<String>,
    },
    StartSequence {
        format: CaptureFormat,
        stage: CaptureStage,
        dir: Option<PathBuf>,
        name: Option<String>,
        fps: Option<u16>,
        scale: Option<u32>,
        palette: PaletteMode,
    },
    Stop,
    Toggle {
        format: CaptureFormat,
        stage: CaptureStage,
        dir: Option<PathBuf>,
        name: Option<String>,
        fps: Option<u16>,
        scale: Option<u32>,
        palette: PaletteMode,
    },
}

/// No-op enqueue: silently drops the request.
pub fn enqueue(_req: CaptureRequest) {}

/// Always returns `None` on stubbed builds; the real status string surfaces only when
/// the capture pipeline is running.
pub fn current_status() -> Option<String> {
    None
}

/// Register the `capture` console command as a stub that prints a one-liner explaining
/// frame capture isn't available in this build.
pub fn register_commands<Ctx: 'static>(console: &mut Console<Ctx>) {
    console.register(loam_egui::cmd(
        "capture",
        "frame capture (unavailable in this build)",
        |_args, _ctx, out| {
            out.line(
                "Frame capture is unavailable in this build. Run a native desktop \
                 build to enable PNG / GIF / APNG capture.",
            );
            Ok(())
        },
    ));
}

/// No-op hotkey binding.
pub fn bind_default_hotkeys<Ctx: 'static>(_console: &mut Console<Ctx>) {}

/// Stub `CapturePanel`. `new()` constructs an empty value; `show()` does nothing. The
/// type exists so demos can hold a `capture_panel: CapturePanel` field unconditionally.
#[derive(Default)]
pub struct CapturePanel {
    _private: (),
}

impl CapturePanel {
    pub fn new() -> Self {
        Self { _private: () }
    }

    pub fn show(&mut self, _ctx: &loam_egui::egui::Context) {}
}
