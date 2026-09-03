//! Loam is a game engine where the geometry of the world is a parameter the
//! game picks. A scene names a [`math::Space`] and rendering, physics, and
//! input run in that space.
//!
//! # Crates
//!
//! The workspace is a DAG. Lower tiers never depend on higher ones, and a
//! change low in the graph ripples up.
//!
//! - `loam-math`: the `Space` trait and its geometries, bivectors and rotors,
//!   projections. The root; nothing sits below it.
//! - `loam-shape`: the `Shape` enum, regular 4-polytope topology,
//!   cross-sections, isovolumes. With `loam-math`, the stable surface.
//! - `loam-time`, `loam-scene`, `loam-physics`, `loam-text`: the fixed
//!   timestep and job pool, typed SDF scenes that emit WGSL, rigid bodies,
//!   and glyph geometry, built on the two above.
//! - `loam-shader`, `loam-render`: WGSL assembly with hot reload, and the wgpu
//!   pipelines (raymarch, line, point, triangle, sky and ground).
//! - `loam-console`, `loam-egui`, `loam-app`: the console model, its egui
//!   frontend, and the shell that hosts both: the `App` trait, the winit
//!   event loop, the scene registry, the command queue, and capture.
//! - `polytope_playground`, `hero`, `tesseract_demo`: demos. Depend on the
//!   engine crates, never on each other.
//!
//! # Capability traits
//!
//! A geometry is anything implementing `Space` (`exp`, `log`, distance,
//! parallel transport) and `IsometryGroup`. What a geometry can do
//! downstream is an opt-in trait, split by how fast each surface changes:
//! `WgslSpace` emits the raymarcher's prelude, `RasterizableSpace` projects
//! for the rasterizer, `SectionableSpace` supports cross-sections, and
//! `PhysicsSpace` supports rigid bodies. A new geometry implements what it
//! supports and nothing else.
//!
//! # Two render paths
//!
//! The raymarch path and the rasterizer path run on the same `Space` impls
//! wherever a space implements both. Over Euclidean R⁴ that is what lets one
//! polytope appear as a raymarched surface, an exact cross-section, and a
//! wireframe from one source of truth.
//!
//! # This crate
//!
//! A facade. It re-exports the crates a consumer is most likely to want under
//! short names; the rest are depended on directly.

pub use loam_asset as asset;
pub use loam_math as math;
pub use loam_render as render;
pub use loam_shader as shader;
pub use loam_time as time;

pub mod prelude;
