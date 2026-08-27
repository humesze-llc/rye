//! `loam-math`: geometric primitives for Loam.
//!
//! ## Determinism
//!
//! Loam is built for lockstep multiplayer, so all math here must be bit-reproducible across
//! machines that agree on:
//! - target architecture's IEEE-754 f32 semantics (no fast-math, no FMA contraction unless
//!   globally enabled),
//! - this crate's exact version,
//! - call ordering (single-threaded sim).

#![warn(missing_docs)]

pub mod bivector;
pub mod blended;
pub mod euclidean;
pub mod euclidean_r2;
pub mod euclidean_r4;
pub mod hyperbolic;
pub mod quotient;
pub mod rasterizable;
pub mod sectionable;
pub mod space;
pub mod spherical;
pub mod spherical_embedded;
pub mod tangent;

pub use bivector::{
    Bivector, Bivector2, Bivector3, Bivector4, Plane4, Rotor, Rotor2, Rotor3, Rotor4,
};
pub use blended::{BlendedSpace, BlendingField, ConformallyFlat, LinearBlendX};
pub use euclidean::{EuclideanR3, Iso3};
pub use euclidean_r2::{EuclideanR2, Iso2};
pub use euclidean_r4::{EuclideanR4, Iso4Flat};
pub use hyperbolic::{HyperbolicH3, Iso3H};
pub use quotient::{FlatTorus3, LensSpace, QuotientSpace};
pub use rasterizable::{Projection, RasterizableSpace, STEREOGRAPHIC_POLE_EPSILON};
pub use sectionable::{
    SectionableSpace, WPlane, EDGE_PARALLEL_EPSILON, SLICE_PERTURBATION_EPSILON,
};
pub use space::{IsometryGroup, Space, WgslSpace};
pub use spherical::{Iso4, SphericalS3};
pub use spherical_embedded::SphericalS3Embedded;
pub use tangent::Tangent;
