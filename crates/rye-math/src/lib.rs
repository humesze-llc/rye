//! `rye-math` — geometric primitives for Rye.
//!
//! The central abstraction is the [`Space`] trait. Every other crate that
//! cares about *where things are* (scene graph, physics, render, shaders)
//! routes through it, so that swapping `EuclideanR3` for `HyperbolicH3` or
//! `Spherical S3` is a type-level decision, not a fork.
//!
//! [`WgslSpace`] is a separate subtrait for the GPU half of the contract;
//! CPU-only consumers never need to implement it.
//!
//! [`Tangent`] bundles a tangent vector with its base point — the
//! recommended holder outside tight numerical kernels.
//!
//! ## Determinism
//!
//! Rye is built for lockstep multiplayer, so all math here must be
//! bit-reproducible across machines that agree on:
//! - target architecture's IEEE-754 f32 semantics (no fast-math, no FMA
//!   contraction unless globally enabled),
//! - this crate's exact version,
//! - call ordering (single-threaded sim).
//!
//! We commit to `f32` for v0. Hyperbolic distances grow as `acosh` and lose
//! precision near the horizon; if that becomes a problem we'll add a
//! `Scalar` associated type rather than rewrite call sites.

pub mod euclidean;
pub mod hyperbolic;
pub mod space;
pub mod tangent;

pub use euclidean::{EuclideanR3, Iso3};
pub use hyperbolic::{HyperbolicH3, Iso3H};
pub use space::{Space, WgslSpace};
pub use tangent::Tangent;
