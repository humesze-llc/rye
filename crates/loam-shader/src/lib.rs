//! `loam-shader`: WGSL loading, Space intrinsic injection, hot reload.
//!
//! When a shader is loaded, the active
//! [`loam_math::WgslSpace`]'s `wgsl_impl` is prepended to the user source (and an optional
//! scene module can be inserted between them), so shader authors can call `loam_distance`,
//! `loam_exp`, etc. without manually importing anything.

mod db;

pub use db::{validate_wgsl, ShaderDb, ShaderId, ShaderOwner, WgslValidationError};

/// Geodesic ray march kernel: `loam_safe_normalize`, `loam_march_geodesic`,
/// `loam_estimate_normal`. Assembled between the scene SDF and user shading
/// WGSL by [`ShaderDb::load_geodesic_scene`].
pub const GEODESIC_MARCH_KERNEL: &str = include_str!("kernel.wgsl");
