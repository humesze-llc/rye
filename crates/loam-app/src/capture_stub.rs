//! The real capture pipeline (`capture.rs`) is desktop-only: it leans on
//! `std::fs`, `std::thread`, and native-only PNG / GIF / APNG encoder crates.
//! This module mirrors the public API as no-ops so call sites need no `cfg`.

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

pub fn enqueue(_req: CaptureRequest) {}

/// Always `None` on stubbed builds.
pub fn current_status() -> Option<String> {
    None
}

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

pub fn bind_default_hotkeys<Ctx: 'static>(_console: &mut Console<Ctx>) {}

/// The type exists so demos can hold a `capture_panel` field unconditionally.
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
