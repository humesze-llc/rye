//! `rye-shader` — WGSL loading, Space intrinsic injection, hot reload.
//!
//! [`ShaderDb`] owns compiled `wgpu::ShaderModule`s and rebuilds them on
//! filesystem events from [`rye_asset::AssetWatcher`]. When a shader is
//! loaded, the active [`rye_math::WgslSpace`]'s `wgsl_impl` is prepended
//! to the user source, so shader authors can call `rye_distance`,
//! `rye_exp`, etc. without manually importing anything.
//!
//! ## Scope note
//!
//! v0 is deliberately minimal: plain string concatenation, no
//! preprocessor. When a second shader in this workspace needs to share
//! code with a first, we'll add `naga_oil` for real `#import`
//! resolution. The public API is designed to be stable across that
//! change — only the internal compile path swaps.

mod db;

pub use db::{validate_wgsl, ShaderDb, ShaderId, WgslValidationError};
