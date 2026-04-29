//! Thin wgpu wrapper plus a tiny render-graph harness.
//!
//! - [`device`]: window surface + adapter/device acquisition. One
//!   [`device::RenderDevice`] per app.
//! - [`graph`]: linear list of [`graph::RenderNode`]s executed in
//!   order against a [`wgpu::TextureView`].
//! - [`raymarch`]: ready-made fullscreen-triangle ray-march nodes
//!   (Euclidean, geodesic, hyperslice 4D); the engine's main render
//!   path until rasterised geometry shows up.
//!
//! The crate stays deliberately small: it hands wgpu primitives to
//! callers rather than abstracting them behind a higher-level engine
//! API.

pub mod device;
pub mod graph;
pub mod raymarch;

pub use raymarch::{GeodesicRayMarchNode, RayMarchNode, RayMarchUniforms};
