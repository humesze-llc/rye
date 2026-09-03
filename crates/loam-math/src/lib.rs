//! Sim and math are bit-reproducible for a given binary on a given machine:
//! same inputs, same bits. That is a debugging tool, not a product promise.
//! Not cross-platform: the transcendentals reach libm, whose results differ
//! between wasm and native. `sqrt` is IEEE-exact and does carry over.

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
