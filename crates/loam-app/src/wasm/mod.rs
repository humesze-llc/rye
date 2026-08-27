//! Verified on Chrome and Firefox (desktop). The hard requirements are WebGPU
//! (`navigator.gpu`), `OffscreenCanvas` + `transferControlToOffscreen`, and
//! ES-module workers. Safari is unverified: WebGPU shipped in Safari 18, but
//! the OffscreenCanvas-in-worker plus module-worker combination this path
//! depends on has not been tested, and older Safari lacks WebGPU entirely.

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

/// The same wasm binary serves both contexts, so `main` branches on this into
/// the worker entry or the main-thread launcher.
pub fn is_worker_context() -> bool {
    use wasm_bindgen::JsCast;
    js_sys::global()
        .dyn_into::<web_sys::DedicatedWorkerGlobalScope>()
        .is_ok()
}
