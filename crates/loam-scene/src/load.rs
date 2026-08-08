//! Scene descriptions read from RON, the boundary between file text and the
//! emitters.
//!
//! [`Primitive::to_wgsl`](crate::Primitive) asserts on constants WGSL cannot
//! spell, so a description that merely deserializes is not yet safe to emit.
//! Both entry points here run the same check, and every rejection is a
//! [`SceneLoadError`] naming its origin rather than a panic in the emitter.
//!
//! `load` is the native path. wasm32 has no filesystem, so a browser build
//! fetches the text itself and calls `from_ron` with the URL as the origin; the
//! origin is a parameter and not an internal detail of `load` for exactly that
//! reason.
//!
//! Committed examples live in `crates/loam-scene/scenes/`.

use std::path::{Path, PathBuf};

use loam_shape::Shape;
use serde::de::DeserializeOwned;
use thiserror::Error;

use crate::scene::{Scene, SceneNode};
use crate::scene4::{Scene4, SceneNode4};

/// Why a RON scene description did not produce a scene.
#[derive(Debug, Error)]
pub enum SceneLoadError {
    /// The file could not be read.
    #[error("{}: {source}", origin.display())]
    Read {
        /// Path as handed to [`Scene::load`] / [`Scene4::load`].
        origin: PathBuf,
        /// Underlying filesystem failure.
        #[source]
        source: std::io::Error,
    },

    /// The text is not a RON scene description. RON's `line:column` is part of
    /// the message.
    #[error("{}: {source}", origin.display())]
    Parse {
        /// Path or label the text came from.
        origin: PathBuf,
        /// Underlying RON failure, positioned in the source text.
        #[source]
        source: ron::error::SpannedError,
    },

    /// The description deserialized but holds a constant no emitter can bake.
    #[error("{}: {reason}", origin.display())]
    Invalid {
        /// Path or label the text came from.
        origin: PathBuf,
        /// What the description asks for and why it cannot be emitted.
        reason: String,
    },
}

impl Scene {
    /// Read a scene description from a `.ron` file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SceneLoadError> {
        let path = path.as_ref();
        Self::from_ron(path, &read_to_string(path)?)
    }

    /// Deserialize a scene description, attributing failures to `origin` (the
    /// file, URL, or label `src` came from).
    pub fn from_ron(origin: impl AsRef<Path>, src: &str) -> Result<Self, SceneLoadError> {
        let origin = origin.as_ref();
        let scene: Self = parse(origin, src)?;
        check_node(&scene.root).map_err(|reason| SceneLoadError::Invalid {
            origin: origin.to_path_buf(),
            reason,
        })?;
        Ok(scene)
    }

    /// Serialize this scene to a RON string.
    pub fn to_ron(&self) -> Result<String, ron::Error> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
    }
}

impl Scene4 {
    /// Read a 4D scene description from a `.ron` file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SceneLoadError> {
        let path = path.as_ref();
        Self::from_ron(path, &read_to_string(path)?)
    }

    /// Deserialize a 4D scene description, attributing failures to `origin`.
    pub fn from_ron(origin: impl AsRef<Path>, src: &str) -> Result<Self, SceneLoadError> {
        let origin = origin.as_ref();
        let scene: Self = parse(origin, src)?;
        check_node_4d(&scene.root).map_err(|reason| SceneLoadError::Invalid {
            origin: origin.to_path_buf(),
            reason,
        })?;
        Ok(scene)
    }

    /// Serialize this scene to a RON string.
    pub fn to_ron(&self) -> Result<String, ron::Error> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
    }
}

fn read_to_string(path: &Path) -> Result<String, SceneLoadError> {
    std::fs::read_to_string(path).map_err(|source| SceneLoadError::Read {
        origin: path.to_path_buf(),
        source,
    })
}

/// `ron::from_str` caps nesting at 128 frames (`ron::Options` default), which
/// is also what bounds the recursion in [`check_node`] and in every later walk
/// of a loaded tree: file input cannot drive them to a stack overflow.
fn parse<T: DeserializeOwned>(origin: &Path, src: &str) -> Result<T, SceneLoadError> {
    ron::from_str(src).map_err(|source| SceneLoadError::Parse {
        origin: origin.to_path_buf(),
        source,
    })
}

/// First constant in `shape` that has no WGSL literal.
///
/// Every variant is scanned, not only the ones an emitter bakes today: the
/// sentinel set shrinks as closed forms land, and a description that survives
/// loading only because its shape is currently unemittable is a trap laid for
/// that lap.
fn non_finite_constant(shape: &Shape) -> Option<f32> {
    fn first(values: impl IntoIterator<Item = f32>) -> Option<f32> {
        values.into_iter().find(|v| !v.is_finite())
    }
    match shape {
        Shape::Sphere { center, radius } => first(center.to_array().into_iter().chain([*radius])),
        Shape::HalfSpace { normal, offset } => {
            first(normal.to_array().into_iter().chain([*offset]))
        }
        Shape::HalfSpace4D { normal, offset } => {
            first(normal.to_array().into_iter().chain([*offset]))
        }
        Shape::Box3 { half_extents } => first(half_extents.to_array()),
        Shape::Polygon2D { vertices } => first(vertices.iter().flat_map(|v| v.to_array())),
        Shape::ConvexPolytope3D { vertices } => first(vertices.iter().flat_map(|v| v.to_array())),
        Shape::ConvexPolytope4D { vertices } => first(vertices.iter().flat_map(|v| v.to_array())),
        Shape::HyperSphere4D { center, radius } => {
            first(center.to_array().into_iter().chain([*radius]))
        }
    }
}

fn check_leaf(shape: &Shape) -> Result<(), String> {
    match non_finite_constant(shape) {
        None => Ok(()),
        Some(value) => Err(format!(
            "{:?} carries the non-finite constant {value:?}, which has no WGSL literal",
            shape.kind(),
        )),
    }
}

fn check_node(node: &SceneNode) -> Result<(), String> {
    match node {
        SceneNode::Leaf(shape) => check_leaf(shape),
        SceneNode::Union(left, right)
        | SceneNode::Intersection(left, right)
        | SceneNode::Difference(left, right) => check_node(left).and_then(|()| check_node(right)),
        SceneNode::SmoothUnion { k, left, right } => {
            // The emitted `smin` divides by `k` and the CPU twin divides by the
            // same constant, so a zero blend radius is an infinity in the
            // distance field and a negative one inverts the blend into a field
            // that is no longer an underestimate of `min`.
            if !k.is_finite() || *k <= 0.0 {
                return Err(format!(
                    "smooth-union blend radius must be finite and positive, got {k:?}",
                ));
            }
            check_node(left).and_then(|()| check_node(right))
        }
    }
}

fn check_node_4d(node: &SceneNode4) -> Result<(), String> {
    match node {
        SceneNode4::Leaf(shape) => check_leaf(shape),
        SceneNode4::Union(left, right)
        | SceneNode4::Intersection(left, right)
        | SceneNode4::Difference(left, right) => {
            check_node_4d(left).and_then(|()| check_node_4d(right))
        }
    }
}
