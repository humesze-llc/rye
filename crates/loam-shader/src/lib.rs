//! The active [`loam_math::WgslSpace`]'s `wgsl_impl` is prepended to every
//! loaded shader, with an optional scene module between them, so shader
//! authors call `loam_distance` and `loam_exp` without importing anything.

mod db;

pub use db::{validate_wgsl, ShaderDb, ShaderId, ShaderOwner, WgslValidationError};

/// Defines `loam_safe_normalize`, `loam_march_geodesic`, `loam_estimate_normal`.
/// Assembled between the scene SDF and user shading by
/// [`ShaderDb::load_geodesic_scene`].
pub const GEODESIC_MARCH_KERNEL: &str = include_str!("kernel.wgsl");
