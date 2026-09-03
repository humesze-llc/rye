//! [`Primitive::to_wgsl`](crate::Primitive) asserts on constants WGSL cannot
//! spell, so a description that merely deserializes is not yet safe to emit.
//! Both entry points here run the same check.

use std::path::{Path, PathBuf};

use loam_shape::Shape;
use serde::de::DeserializeOwned;
use thiserror::Error;

use crate::scene::{Scene, SceneNode};
use crate::scene4::{Scene4, SceneNode4};

#[derive(Debug, Error)]
pub enum SceneLoadError {
    #[error("{}: {source}", origin.display())]
    Read {
        origin: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// RON's `line:column` is part of the message.
    #[error("{}: {source}", origin.display())]
    Parse {
        origin: PathBuf,
        #[source]
        source: ron::error::SpannedError,
    },

    /// Deserialized, but holds a constant no emitter can bake.
    #[error("{}: {reason}", origin.display())]
    Invalid { origin: PathBuf, reason: String },
}

impl Scene {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SceneLoadError> {
        let path = path.as_ref();
        Self::from_ron(path, &read_to_string(path)?)
    }

    /// `origin` is the file, URL, or label `src` came from; failures name it.
    pub fn from_ron(origin: impl AsRef<Path>, src: &str) -> Result<Self, SceneLoadError> {
        let origin = origin.as_ref();
        let scene: Self = parse(origin, src)?;
        check_node(&scene.root).map_err(|reason| SceneLoadError::Invalid {
            origin: origin.to_path_buf(),
            reason,
        })?;
        Ok(scene)
    }

    pub fn to_ron(&self) -> Result<String, ron::Error> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
    }
}

impl Scene4 {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SceneLoadError> {
        let path = path.as_ref();
        Self::from_ron(path, &read_to_string(path)?)
    }

    pub fn from_ron(origin: impl AsRef<Path>, src: &str) -> Result<Self, SceneLoadError> {
        let origin = origin.as_ref();
        let scene: Self = parse(origin, src)?;
        check_node_4d(&scene.root).map_err(|reason| SceneLoadError::Invalid {
            origin: origin.to_path_buf(),
            reason,
        })?;
        Ok(scene)
    }

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

// `ron::from_str` caps nesting at 128 frames (`ron::Options` default).
fn parse<T: DeserializeOwned>(origin: &Path, src: &str) -> Result<T, SceneLoadError> {
    ron::from_str(src).map_err(|source| SceneLoadError::Parse {
        origin: origin.to_path_buf(),
        source,
    })
}

// Every variant is scanned, not only the ones an emitter bakes today.
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

// Shared with `crate::edit`, so a tree an editor writes is a tree this module
// would have accepted from a file.
pub(crate) fn check_leaf(shape: &Shape) -> Result<(), String> {
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
            check_blend_radius(*k)?;
            check_node(left).and_then(|()| check_node(right))
        }
    }
}

// The emitted `smin` divides by `k`, so a zero blend radius is an infinity.
pub(crate) fn check_blend_radius(k: f32) -> Result<(), String> {
    if !k.is_finite() || k <= 0.0 {
        return Err(format!(
            "smooth-union blend radius must be finite and positive, got {k:?}",
        ));
    }
    Ok(())
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
