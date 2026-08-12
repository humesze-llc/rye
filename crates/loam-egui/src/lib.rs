//! `loam-egui`: integration glue between `loam-app` and the [egui] immediate-mode UI
//! library.
//!
//! Wraps egui with a wgpu paint pass plus a few widgets; a future UI-framework
//! migration replaces this crate rather than retargeting it.
//!
//! ## Surface
//!
//! - [`UiIntegration`]: per-app egui state, owned by `loam_app::Runner`.
//! - [`world_to_screen`]: egui coordinates for
//!   [`loam_camera::Camera::pixels_from_world`], which anchors a label to a
//!   point in whatever [`Space`](loam_math::Space) the scene lives in.
//! - [`ConsoleUi`]: the egui frontend for [`Console`], whose registry,
//!   scrollback, completion and dispatch live in `loam-console` and carry no
//!   egui types.
//!
//! [egui]: https://github.com/emilk/egui
//!
//! ## Input gating
//!
//! Gameplay code reading WASD or mouse delta should gate on
//! `frame.ui_has_focus()` so typing into a settings field doesn't also fire
//! movement.

mod bivector_matrix;
pub mod console;
pub mod dnd;
mod floating;
mod integration;
pub mod media;
mod slider_edit;
mod world;

pub use bivector_matrix::{bivector_matrix, cell_text as bivector_matrix_cell_text};
pub use console::{
    cmd, console_echo_enabled, parse_line, set_console_echo, subcommands, Command, Console,
    ConsoleUi, ConsoleWriter, HistoryLine, Key, LineKind, SubcommandSet,
};
pub use floating::{
    callout, floating_panel, floating_panel_builder, sticky_menu, CalloutState,
    FloatingPanelBuilder,
};
pub use integration::UiIntegration;
pub use slider_edit::{slider_with_edit, SliderInteraction};
pub use world::world_to_screen;

// Re-export egui so the version pin lives in one place.
pub use egui;
