//! `rye-egui`: integration glue between `rye-app` and the [egui]
//! immediate-mode UI library.
//!
//! Named for what it is, not what it abstracts. The crate wraps egui
//! and provides a wgpu paint pass + world-anchored label helper; if a
//! future migration to a different UI framework happens, this crate
//! gets replaced rather than retargeted.
//!
//! ## Surface
//!
//! - [`UiIntegration`]: per-app egui state (Context, winit translator,
//!   wgpu renderer). Owned by `rye_app::Runner`; apps don't construct
//!   it directly.
//! - [`world_to_screen`]: project a world-space point to screen pixel
//!   coordinates via a camera + viewport. The cheap pattern for
//!   "egui label that follows a 3D object."
//!
//! [egui]: https://github.com/emilk/egui
//!
//! Apps interact with the UI by overriding `App::ui(&mut self, ctx,
//! frame)` and writing immediate-mode egui code:
//!
//! ```ignore
//! fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
//!     egui::Window::new("Settings").show(ctx, |ui| {
//!         ui.add(egui::Slider::new(&mut self.fov, 30.0..=120.0).text("FOV"));
//!         if ui.button("Reset").clicked() {
//!             self.reset();
//!         }
//!     });
//! }
//! ```
//!
//! ## Input gating
//!
//! egui consumes input it cares about (clicks on widgets, typing into
//! a focused text input). Gameplay code that reads the same WASD keys
//! or mouse delta should gate on `frame.ui_has_focus()` so a player
//! typing into a settings field doesn't also fire movement.
//!
//! ## Why egui (not iced or a from-scratch UI)
//!
//! Immediate mode matches `rye-app`'s "library-style composition, no
//! ECS" pattern: apps construct and call UI inside `App::ui`. egui
//! integrates with wgpu directly via `egui-wgpu`; no rendering glue
//! beyond what this crate provides. Pure-Rust dependency tree.
//!
//! ## What's deliberately out of scope
//!
//! - **3D-billboard egui** (egui rendered to texture, sampled in the
//!   3D scene with ray-cast interaction). Possible but unnecessary
//!   for current use cases; the screen-space-with-world-anchoring
//!   pattern via [`world_to_screen`] covers labels and HUDs that
//!   follow 3D objects.
//! - **Custom widget set on top of egui.** egui's defaults are fine
//!   for v0; theme tweaks live in the app, not here.

mod integration;
mod world;

pub use integration::UiIntegration;
pub use world::world_to_screen;

// Re-export egui so apps depend on `rye-egui` only and the version
// pin lives in one place.
pub use egui;
