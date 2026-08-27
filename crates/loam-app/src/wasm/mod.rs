//! Wasm32-only support code.
//!
//! Verified on Chrome and Firefox (desktop). The hard requirements are
//! WebGPU (`navigator.gpu`), `OffscreenCanvas` + `transferControlToOffscreen`,
//! and ES-module workers (`new Worker(url, { type: "module" })`). Safari is
//! UNVERIFIED: WebGPU shipped in Safari 18 but the OffscreenCanvas-in-worker
//! + module-worker combination this path depends on has not been tested here,
//! and older Safari lacks WebGPU entirely. Demos that need a Safari fallback
//! should feature-detect `navigator.gpu` and surface a "WebGPU required"
//! message rather than spawning the worker into a hard failure.

pub mod host_action;
pub mod input_queue;
pub mod launch;
pub mod main_launcher;
pub mod messages;
pub mod modifier_sync;
pub mod worker;
pub mod worker_ui;

pub use main_launcher::launch_on_click;

pub use launch::{is_manual_mode, js_heap_sampler, wait_for_launch};

/// Returns true when the wasm binary is executing inside a
/// `DedicatedWorkerGlobalScope` (i.e., a Web Worker), false on the main
/// page thread. The same wasm binary serves both contexts; this check
/// lets `main` branch into worker entry vs main-thread launcher.
pub fn is_worker_context() -> bool {
    use wasm_bindgen::JsCast;
    js_sys::global()
        .dyn_into::<web_sys::DedicatedWorkerGlobalScope>()
        .is_ok()
}
