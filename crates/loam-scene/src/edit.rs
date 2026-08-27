//! Addresses are positional, so a [`NodePath`] names a position and not an
//! identity; whoever holds one across an [`SceneEdit::Insert`] or
//! [`SceneEdit::Remove`] re-derives it from [`SceneEdit::focus_after`].
//! [`apply`] accepts only what [`Scene::from_ron`] would, through the same
//! [`crate::load`] predicates, so an edited tree is always emittable.

use std::fmt;
use std::str::FromStr;

use glam::Vec3;
use loam_shape::Shape;
use thiserror::Error;

use crate::load::{check_blend_radius, check_leaf};
use crate::scene::{Scene, SceneNode};

/// Space units. Wide enough to read as a blend at the scale of the unit-ish
/// primitives an editor inserts.
pub const DEFAULT_BLEND_RADIUS: f32 = 0.15;

// Below this the three components are at f32 noise scale relative to each
// other, so the direction they encode is not a direction; `dot(p, n) - d` would
// also stop being a distance, since that formula is a signed distance only for
// unit `n`.
const MIN_NORMAL_LENGTH: f32 = 1e-6;

/// The child index taken at each step from the root. Every interior
/// [`SceneNode`] has exactly two children, so each step is 0 or 1. Spelled
/// `root`, `root.0`, `root.1.0` and parsed back from the same text.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct NodePath(Vec<u8>);

impl NodePath {
    pub fn root() -> Self {
        Self(Vec::new())
    }

    pub fn child(&self, index: u8) -> Self {
        let mut steps = self.0.clone();
        steps.push(index & 1);
        Self(steps)
    }

    /// `None` at the root.
    pub fn parent(&self) -> Option<Self> {
        let (_, head) = self.0.split_last()?;
        Some(Self(head.to_vec()))
    }

    /// `None` at the root.
    pub fn last_step(&self) -> Option<u8> {
        self.0.last().copied()
    }

    pub fn steps(&self) -> &[u8] {
        &self.0
    }

    pub fn depth(&self) -> usize {
        self.0.len()
    }
}

impl fmt::Display for NodePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("root")?;
        for step in &self.0 {
            write!(f, ".{step}")?;
        }
        Ok(())
    }
}

impl FromStr for NodePath {
    type Err = EditError;

    fn from_str(text: &str) -> Result<Self, EditError> {
        let rest = text
            .strip_prefix("root")
            .ok_or_else(|| EditError::Syntax(format!("path `{text}` does not start at `root`")))?;
        let mut steps = Vec::new();
        for token in rest.split('.').skip_while(|t| t.is_empty()) {
            match token {
                "0" => steps.push(0),
                "1" => steps.push(1),
                _ => {
                    return Err(EditError::Syntax(format!(
                        "path step `{token}` in `{text}` is not a child index (0 or 1)",
                    )))
                }
            }
        }
        Ok(Self(steps))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Combinator {
    Union,
    Intersection,
    Difference,
    SmoothUnion,
}

impl Combinator {
    pub const ALL: [Combinator; 4] = [
        Combinator::Union,
        Combinator::Intersection,
        Combinator::Difference,
        Combinator::SmoothUnion,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Combinator::Union => "union",
            Combinator::Intersection => "intersection",
            Combinator::Difference => "difference",
            Combinator::SmoothUnion => "smooth-union",
        }
    }

    fn parse(text: &str) -> Result<Self, EditError> {
        Self::ALL
            .into_iter()
            .find(|c| c.name() == text)
            .ok_or_else(|| EditError::Syntax(format!("unknown combinator `{text}`")))
    }

    fn combine(self, left: SceneNode, right: SceneNode) -> SceneNode {
        match self {
            Combinator::Union => left.union(right),
            Combinator::Intersection => left.intersect(right),
            Combinator::Difference => left.subtract(right),
            Combinator::SmoothUnion => left.smooth_union(right, DEFAULT_BLEND_RADIUS),
        }
    }
}

/// The vertex-list shapes are absent on purpose: they carry no scalar parameter
/// to author and their SDF is the sentinel, so inserting one would add an
/// invisible node with nothing to edit.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LeafKind {
    Sphere,
    Box,
    Plane,
}

impl LeafKind {
    pub const ALL: [LeafKind; 3] = [LeafKind::Sphere, LeafKind::Box, LeafKind::Plane];

    pub fn name(self) -> &'static str {
        match self {
            LeafKind::Sphere => "sphere",
            LeafKind::Box => "box",
            LeafKind::Plane => "plane",
        }
    }

    /// Sized against the unit-ish scale the march kernel's step thresholds
    /// assume.
    pub fn shape(self) -> Shape {
        match self {
            LeafKind::Sphere => Shape::sphere_at(Vec3::ZERO, 0.25),
            LeafKind::Box => Shape::Box3 {
                half_extents: Vec3::splat(0.25),
            },
            LeafKind::Plane => Shape::HalfSpace {
                normal: Vec3::Y,
                offset: -0.5,
            },
        }
    }

    fn parse(text: &str) -> Result<Self, EditError> {
        Self::ALL
            .into_iter()
            .find(|k| k.name() == text)
            .ok_or_else(|| EditError::Syntax(format!("unknown leaf kind `{text}`")))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Param {
    Center,
    Radius,
    HalfExtents,
    Normal,
    Offset,
    Blend,
}

impl Param {
    pub fn name(self) -> &'static str {
        match self {
            Param::Center => "center",
            Param::Radius => "radius",
            Param::HalfExtents => "half-extents",
            Param::Normal => "normal",
            Param::Offset => "offset",
            Param::Blend => "blend",
        }
    }

    const ALL: [Param; 6] = [
        Param::Center,
        Param::Radius,
        Param::HalfExtents,
        Param::Normal,
        Param::Offset,
        Param::Blend,
    ];

    fn parse(text: &str) -> Result<Self, EditError> {
        Self::ALL
            .into_iter()
            .find(|p| p.name() == text)
            .ok_or_else(|| EditError::Syntax(format!("unknown parameter `{text}`")))
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum EditValue {
    Scalar(f32),
    Vector(Vec3),
}

impl EditValue {
    /// The order components are written to and read back from a command line.
    pub fn components(&self) -> &[f32] {
        match self {
            EditValue::Scalar(v) => std::slice::from_ref(v),
            EditValue::Vector(v) => AsRef::<[f32; 3]>::as_ref(v),
        }
    }
}

impl fmt::Display for EditValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, c) in self.components().iter().enumerate() {
            if i > 0 {
                f.write_str(" ")?;
            }
            // `Debug`, not `Display`: shortest round-trip in both, but only
            // `Debug` keeps a point or an exponent, which is what makes the
            // text re-parse as the same f32 rather than as an integer.
            write!(f, "{c:?}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SceneEdit {
    Set {
        path: NodePath,
        param: Param,
        value: EditValue,
    },
    /// Replaces the node at `path` with `combinator(that node, new leaf)`: the
    /// subtree already there becomes the left child, the new leaf the right.
    Insert {
        path: NodePath,
        combinator: Combinator,
        leaf: LeafKind,
    },
    /// Collapses the parent into the sibling.
    Remove { path: NodePath },
}

impl SceneEdit {
    pub fn path(&self) -> &NodePath {
        match self {
            SceneEdit::Set { path, .. }
            | SceneEdit::Insert { path, .. }
            | SceneEdit::Remove { path } => path,
        }
    }

    /// Where this edit's node lives once applied. A selection follows this
    /// rather than trying to hold a path across a structural change.
    pub fn focus_after(&self) -> NodePath {
        match self {
            SceneEdit::Set { path, .. } => path.clone(),
            SceneEdit::Insert { path, .. } => path.child(1),
            SceneEdit::Remove { path } => path.parent().unwrap_or_default(),
        }
    }

    pub fn to_args(&self) -> Vec<String> {
        match self {
            SceneEdit::Set { path, param, value } => {
                let mut args = vec![
                    "set".to_string(),
                    path.to_string(),
                    param.name().to_string(),
                ];
                args.extend(value.components().iter().map(|c| format!("{c:?}")));
                args
            }
            SceneEdit::Insert {
                path,
                combinator,
                leaf,
            } => vec![
                "add".to_string(),
                path.to_string(),
                combinator.name().to_string(),
                leaf.name().to_string(),
            ],
            SceneEdit::Remove { path } => vec!["remove".to_string(), path.to_string()],
        }
    }

    pub fn from_args(args: &[&str]) -> Result<Self, EditError> {
        let verb = args
            .first()
            .copied()
            .ok_or_else(|| EditError::Syntax("expected set, add or remove".into()))?;
        let path = |index: usize| -> Result<NodePath, EditError> {
            args.get(index)
                .copied()
                .ok_or_else(|| EditError::Syntax(format!("{verb} needs a node path")))?
                .parse()
        };
        match verb {
            "set" => {
                let path = path(1)?;
                let param = Param::parse(args.get(2).copied().unwrap_or_default())?;
                let value = parse_value(param, &args[3.min(args.len())..])?;
                Ok(SceneEdit::Set { path, param, value })
            }
            "add" => Ok(SceneEdit::Insert {
                path: path(1)?,
                combinator: Combinator::parse(args.get(2).copied().unwrap_or_default())?,
                leaf: LeafKind::parse(args.get(3).copied().unwrap_or_default())?,
            }),
            "remove" => Ok(SceneEdit::Remove { path: path(1)? }),
            other => Err(EditError::Syntax(format!(
                "`{other}` is not one of set, add, remove",
            ))),
        }
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum EditError {
    #[error("no node at `{0}`")]
    NoSuchNode(NodePath),
    #[error("`{param}` is not a parameter of the {node} at `{path}`")]
    NotAParameter {
        path: NodePath,
        node: &'static str,
        param: &'static str,
    },
    #[error("the root cannot be removed: a scene is one root node")]
    RootRemoval,
    #[error("{0}")]
    Rejected(String),
    #[error("{0}")]
    Syntax(String),
}

/// Returns whether the tree actually changed, so a caller that recompiles a
/// shader per change does not pay for a set to the value already held. The
/// comparison is on bits, because the emitter prints the sign of a zero.
pub fn apply(scene: &mut Scene, edit: &SceneEdit) -> Result<bool, EditError> {
    match edit {
        SceneEdit::Set { path, param, value } => {
            let node = node_at_mut(&mut scene.root, path)
                .ok_or_else(|| EditError::NoSuchNode(path.clone()))?;
            set_param(node, path, *param, *value)
        }
        SceneEdit::Insert {
            path,
            combinator,
            leaf,
        } => {
            let node = node_at_mut(&mut scene.root, path)
                .ok_or_else(|| EditError::NoSuchNode(path.clone()))?;
            let shape = leaf.shape();
            check_leaf(&shape).map_err(EditError::Rejected)?;
            // Cloned rather than swapped out through a placeholder, which would
            // be a second tree state an emitter could observe if this ever grew
            // a fallible step.
            *node = combinator.combine(node.clone(), SceneNode::Leaf(shape));
            Ok(true)
        }
        SceneEdit::Remove { path } => {
            let step = path.last_step().ok_or(EditError::RootRemoval)?;
            let parent_path = path.parent().unwrap_or_default();
            let parent = node_at_mut(&mut scene.root, &parent_path)
                .ok_or_else(|| EditError::NoSuchNode(path.clone()))?;
            let sibling = {
                let kids = children(parent).ok_or_else(|| EditError::NoSuchNode(path.clone()))?;
                SceneNode::clone(kids[usize::from(step ^ 1)])
            };
            *parent = sibling;
            Ok(true)
        }
    }
}

/// In panel order. Empty for the vertex-list leaves, whose SDF is the sentinel,
/// and for the boolean combinators, which carry no constant.
pub fn parameters(node: &SceneNode) -> Vec<(Param, EditValue)> {
    match node {
        SceneNode::Leaf(Shape::Sphere { center, radius }) => vec![
            (Param::Center, EditValue::Vector(*center)),
            (Param::Radius, EditValue::Scalar(*radius)),
        ],
        SceneNode::Leaf(Shape::Box3 { half_extents }) => {
            vec![(Param::HalfExtents, EditValue::Vector(*half_extents))]
        }
        SceneNode::Leaf(Shape::HalfSpace { normal, offset }) => vec![
            (Param::Normal, EditValue::Vector(*normal)),
            (Param::Offset, EditValue::Scalar(*offset)),
        ],
        SceneNode::SmoothUnion { k, .. } => vec![(Param::Blend, EditValue::Scalar(*k))],
        SceneNode::Leaf(_)
        | SceneNode::Union(..)
        | SceneNode::Intersection(..)
        | SceneNode::Difference(..) => Vec::new(),
    }
}

pub fn label(node: &SceneNode) -> &'static str {
    match node {
        SceneNode::Leaf(shape) => match shape {
            Shape::Sphere { .. } => "sphere",
            Shape::Box3 { .. } => "box",
            Shape::HalfSpace { .. } => "plane",
            Shape::HalfSpace4D { .. } => "plane-4d",
            Shape::HyperSphere4D { .. } => "hypersphere",
            Shape::Polygon2D { .. } => "polygon",
            Shape::ConvexPolytope3D { .. } => "polytope",
            Shape::ConvexPolytope4D { .. } => "polytope-4d",
        },
        SceneNode::Union(..) => "union",
        SceneNode::Intersection(..) => "intersection",
        SceneNode::Difference(..) => "difference",
        SceneNode::SmoothUnion { .. } => "smooth-union",
    }
}

/// Pre-order. One path buffer is pushed and popped through the whole walk, so a
/// full traversal allocates once however deep the tree.
pub fn for_each_node(root: &SceneNode, mut visit: impl FnMut(&NodePath, &SceneNode)) {
    fn walk(node: &SceneNode, path: &mut NodePath, visit: &mut impl FnMut(&NodePath, &SceneNode)) {
        visit(path, node);
        if let Some(kids) = children(node) {
            for (index, child) in kids.into_iter().enumerate() {
                path.0.push(index as u8);
                walk(child, path, visit);
                path.0.pop();
            }
        }
    }
    walk(root, &mut NodePath::root(), &mut visit);
}

/// `None` when the tree has no such position.
pub fn node_at<'a>(root: &'a SceneNode, path: &NodePath) -> Option<&'a SceneNode> {
    let mut node = root;
    for step in path.steps() {
        node = children(node)?[usize::from(*step)];
    }
    Some(node)
}

fn node_at_mut<'a>(root: &'a mut SceneNode, path: &NodePath) -> Option<&'a mut SceneNode> {
    let mut node = root;
    for step in path.steps() {
        let index = usize::from(*step);
        node = match node {
            SceneNode::Union(left, right)
            | SceneNode::Intersection(left, right)
            | SceneNode::Difference(left, right)
            | SceneNode::SmoothUnion { left, right, .. } => {
                if index == 0 {
                    left
                } else {
                    right
                }
            }
            SceneNode::Leaf(_) => return None,
        };
    }
    Some(node)
}

fn children(node: &SceneNode) -> Option<[&SceneNode; 2]> {
    match node {
        SceneNode::Union(left, right)
        | SceneNode::Intersection(left, right)
        | SceneNode::Difference(left, right)
        | SceneNode::SmoothUnion { left, right, .. } => Some([left, right]),
        SceneNode::Leaf(_) => None,
    }
}

fn set_param(
    node: &mut SceneNode,
    path: &NodePath,
    param: Param,
    value: EditValue,
) -> Result<bool, EditError> {
    let node_label = label(node);
    for component in value.components() {
        if !component.is_finite() {
            return Err(EditError::Rejected(format!(
                "{} takes finite values, got {component:?}",
                param.name(),
            )));
        }
    }
    match (&mut *node, param, value) {
        (SceneNode::Leaf(Shape::Sphere { center, .. }), Param::Center, EditValue::Vector(v)) => {
            Ok(store_vec3(center, v))
        }
        (SceneNode::Leaf(Shape::Sphere { radius, .. }), Param::Radius, EditValue::Scalar(v)) => {
            Ok(store_f32(radius, v))
        }
        (
            SceneNode::Leaf(Shape::Box3 { half_extents }),
            Param::HalfExtents,
            EditValue::Vector(v),
        ) => Ok(store_vec3(half_extents, v)),
        (SceneNode::Leaf(Shape::HalfSpace { normal, .. }), Param::Normal, EditValue::Vector(v)) => {
            Ok(store_vec3(normal, unit_normal(v)?))
        }
        (SceneNode::Leaf(Shape::HalfSpace { offset, .. }), Param::Offset, EditValue::Scalar(v)) => {
            Ok(store_f32(offset, v))
        }
        (SceneNode::SmoothUnion { k, .. }, Param::Blend, EditValue::Scalar(v)) => {
            check_blend_radius(v).map_err(EditError::Rejected)?;
            Ok(store_f32(k, v))
        }
        _ => Err(EditError::NotAParameter {
            path: path.clone(),
            node: node_label,
            param: param.name(),
        }),
    }
}

fn unit_normal(v: Vec3) -> Result<Vec3, EditError> {
    let length = v.length();
    if length < MIN_NORMAL_LENGTH {
        return Err(EditError::Rejected(format!(
            "a half-space normal must be at least {MIN_NORMAL_LENGTH:e} long, got {length:e}",
        )));
    }
    Ok(v / length)
}

// Reports whether the bits moved.
fn store_f32(slot: &mut f32, value: f32) -> bool {
    let changed = slot.to_bits() != value.to_bits();
    *slot = value;
    changed
}

fn store_vec3(slot: &mut Vec3, value: Vec3) -> bool {
    let changed = slot
        .to_array()
        .iter()
        .zip(value.to_array())
        .any(|(a, b)| a.to_bits() != b.to_bits());
    *slot = value;
    changed
}

fn parse_value(param: Param, args: &[&str]) -> Result<EditValue, EditError> {
    let number = |index: usize| -> Result<f32, EditError> {
        args.get(index)
            .copied()
            .ok_or_else(|| {
                EditError::Syntax(format!(
                    "{} takes {} number(s), got {}",
                    param.name(),
                    arity(param),
                    args.len(),
                ))
            })?
            .parse::<f32>()
            .map_err(|e| EditError::Syntax(format!("`{}`: {e}", args[index])))
    };
    match arity(param) {
        1 => Ok(EditValue::Scalar(number(0)?)),
        _ => Ok(EditValue::Vector(Vec3::new(
            number(0)?,
            number(1)?,
            number(2)?,
        ))),
    }
}

fn arity(param: Param) -> usize {
    match param {
        Param::Center | Param::HalfExtents | Param::Normal => 3,
        Param::Radius | Param::Offset | Param::Blend => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_math::EuclideanR3;

    // Four leaves and every combinator: `((sphere ~ box) | plane) - sphere`.
    fn fixture() -> Scene {
        Scene::new(
            SceneNode::sphere(Vec3::new(-0.35, 0.0, 0.0), 0.45)
                .smooth_union(SceneNode::box_(Vec3::splat(0.4)), 0.15)
                .union(SceneNode::plane(Vec3::Y, -0.8))
                .subtract(SceneNode::sphere(Vec3::new(0.3, 0.35, 0.35), 0.35)),
        )
    }

    fn paths(scene: &Scene) -> Vec<NodePath> {
        let mut out = Vec::new();
        for_each_node(&scene.root, |path, _| out.push(path.clone()));
        out
    }

    fn node_count(scene: &Scene) -> usize {
        paths(scene).len()
    }

    #[test]
    fn every_enumerated_path_resolves_to_the_node_it_was_enumerated_with() {
        let scene = fixture();
        let mut seen: Vec<(NodePath, &'static str)> = Vec::new();
        for_each_node(&scene.root, |path, node| {
            seen.push((path.clone(), label(node)));
            assert_eq!(
                label(node_at(&scene.root, path).expect("enumerated path resolves")),
                label(node),
            );
        });
        assert_eq!(
            seen.iter()
                .map(|(p, l)| (p.to_string(), *l))
                .collect::<Vec<_>>(),
            vec![
                ("root".into(), "difference"),
                ("root.0".into(), "union"),
                ("root.0.0".into(), "smooth-union"),
                ("root.0.0.0".into(), "sphere"),
                ("root.0.0.1".into(), "box"),
                ("root.0.1".into(), "plane"),
                ("root.1".into(), "sphere"),
            ],
        );
    }

    #[test]
    fn a_path_round_trips_through_its_text_form() {
        for path in paths(&fixture()) {
            assert_eq!(path.to_string().parse::<NodePath>(), Ok(path.clone()));
        }
        assert!("root.2".parse::<NodePath>().is_err());
        assert!("0.1".parse::<NodePath>().is_err());
        assert_eq!("root".parse::<NodePath>(), Ok(NodePath::root()));
    }

    #[test]
    fn set_is_total_over_every_path_and_never_changes_the_tree_shape() {
        let base = fixture();
        let before = node_count(&base);
        for path in paths(&base) {
            let node = node_at(&base.root, &path).expect("enumerated");
            let advertised: Vec<Param> = parameters(node).into_iter().map(|(p, _)| p).collect();
            for param in Param::ALL {
                let value = match arity(param) {
                    1 => EditValue::Scalar(0.31),
                    _ => EditValue::Vector(Vec3::new(0.31, -0.12, 0.07)),
                };
                let mut scene = base.clone();
                let result = apply(
                    &mut scene,
                    &SceneEdit::Set {
                        path: path.clone(),
                        param,
                        value,
                    },
                );
                assert_eq!(
                    result.is_ok(),
                    advertised.contains(&param),
                    "{path} ({}) and parameter `{}`: {result:?}",
                    label(node),
                    param.name(),
                );
                assert_eq!(node_count(&scene), before, "{path} changed the tree shape");
            }
        }
    }

    #[test]
    fn one_edit_changes_exactly_one_emitted_constant() {
        let mut scene = fixture();
        let before = scene.to_wgsl(&EuclideanR3);
        let path: NodePath = "root.1".parse().expect("path");
        assert!(apply(
            &mut scene,
            &SceneEdit::Set {
                path,
                param: Param::Radius,
                value: EditValue::Scalar(0.125),
            },
        )
        .expect("radius is a sphere parameter"));
        let after = scene.to_wgsl(&EuclideanR3);

        let differing: Vec<(&str, &str)> = before
            .lines()
            .zip(after.lines())
            .filter(|(a, b)| a != b)
            .collect();
        assert_eq!(before.lines().count(), after.lines().count());
        assert_eq!(differing.len(), 1, "changed lines: {differing:?}");
        assert!(differing[0].0.contains("0.35"), "{:?}", differing[0]);
        assert!(differing[0].1.contains("0.125"), "{:?}", differing[0]);
    }

    #[test]
    fn setting_the_value_a_node_already_holds_is_not_a_change() {
        let mut scene = fixture();
        let path: NodePath = "root.1".parse().expect("path");
        let edit = SceneEdit::Set {
            path,
            param: Param::Radius,
            value: EditValue::Scalar(0.2),
        };
        assert_eq!(apply(&mut scene, &edit), Ok(true));
        assert_eq!(apply(&mut scene, &edit), Ok(false));
    }

    #[test]
    fn an_insert_and_the_matching_remove_restore_the_original_tree() {
        let original = fixture();
        for path in paths(&original) {
            for combinator in Combinator::ALL {
                for leaf in LeafKind::ALL {
                    let mut scene = original.clone();
                    let insert = SceneEdit::Insert {
                        path: path.clone(),
                        combinator,
                        leaf,
                    };
                    assert_eq!(apply(&mut scene, &insert), Ok(true));
                    assert_eq!(node_count(&scene), node_count(&original) + 2);
                    let added = insert.focus_after();
                    assert_eq!(
                        label(node_at(&scene.root, &added).expect("the new leaf resolves")),
                        leaf.name(),
                    );

                    assert_eq!(
                        apply(&mut scene, &SceneEdit::Remove { path: added }),
                        Ok(true)
                    );
                    assert_eq!(
                        scene.to_wgsl(&EuclideanR3),
                        original.to_wgsl(&EuclideanR3),
                        "{path} / {} / {}",
                        combinator.name(),
                        leaf.name(),
                    );
                }
            }
        }
    }

    #[test]
    fn a_primitive_added_at_runtime_enters_the_field_moves_within_it_and_leaves_it() {
        let radius = 0.25_f32;
        let elsewhere = Vec3::new(-1.0, 0.0, 0.0);
        let probe = Vec3::new(1.0, 0.0, 0.0);
        let original = Scene::new(SceneNode::sphere(elsewhere, radius));
        let field = |scene: &Scene| scene.eval(&EuclideanR3, probe);
        let empty = field(&original);
        assert!(empty > 0.5, "the probe starts outside everything: {empty}");

        let mut scene = original.clone();
        let insert = SceneEdit::Insert {
            path: NodePath::root(),
            combinator: Combinator::Union,
            leaf: LeafKind::Sphere,
        };
        assert_eq!(apply(&mut scene, &insert), Ok(true));
        let added = insert.focus_after();
        assert!(
            field(&scene) > 0.0,
            "a leaf inserted at the origin must not already cover the probe",
        );

        let at = |p: Vec3| SceneEdit::Set {
            path: added.clone(),
            param: Param::Center,
            value: EditValue::Vector(p),
        };
        assert_eq!(apply(&mut scene, &at(probe)), Ok(true));
        assert!(
            (field(&scene) + radius).abs() < 1e-6,
            "moved onto the probe, the field must read minus the radius: {}",
            field(&scene),
        );

        assert_eq!(apply(&mut scene, &at(Vec3::new(0.0, 1.5, 0.0))), Ok(true));
        assert!(
            field(&scene) > 0.0,
            "moved away, the probe is outside again: {}",
            field(&scene),
        );

        assert_eq!(apply(&mut scene, &at(probe)), Ok(true));
        assert_eq!(
            apply(&mut scene, &SceneEdit::Remove { path: added }),
            Ok(true)
        );
        assert_eq!(
            field(&scene),
            empty,
            "removing the leaf must restore the field it was unioned into",
        );
    }

    #[test]
    fn a_remove_collapses_the_parent_into_the_sibling_at_the_parents_address() {
        let mut scene = fixture();
        let removed: NodePath = "root.0.0.1".parse().expect("path");
        let sibling = node_at(&scene.root, &"root.0.0.0".parse().expect("path"))
            .expect("sibling")
            .clone();
        let edit = SceneEdit::Remove {
            path: removed.clone(),
        };
        let focus = edit.focus_after();
        assert_eq!(focus.to_string(), "root.0.0");
        assert_eq!(apply(&mut scene, &edit), Ok(true));

        let survivor = node_at(&scene.root, &focus).expect("the sibling took the slot");
        assert_eq!(
            Scene::new(survivor.clone()).to_wgsl(&EuclideanR3),
            Scene::new(sibling).to_wgsl(&EuclideanR3),
        );
        assert_eq!(node_count(&scene), 5);
    }

    #[test]
    fn the_root_cannot_be_removed() {
        let mut scene = fixture();
        assert_eq!(
            apply(
                &mut scene,
                &SceneEdit::Remove {
                    path: NodePath::root(),
                },
            ),
            Err(EditError::RootRemoval),
        );
        assert_eq!(node_count(&scene), 7);
    }

    #[test]
    fn an_unresolvable_path_is_refused_without_touching_the_tree() {
        let mut scene = fixture();
        let before = scene.to_wgsl(&EuclideanR3);
        for text in ["root.1.0", "root.0.0.0.1", "root.1.1.1.1"] {
            let path: NodePath = text.parse().expect("path");
            assert_eq!(
                apply(
                    &mut scene,
                    &SceneEdit::Set {
                        path: path.clone(),
                        param: Param::Radius,
                        value: EditValue::Scalar(0.1),
                    },
                ),
                Err(EditError::NoSuchNode(path.clone())),
            );
            assert!(apply(&mut scene, &SceneEdit::Remove { path: path.clone() }).is_err());
            assert!(node_at(&scene.root, &path).is_none());
        }
        assert_eq!(scene.to_wgsl(&EuclideanR3), before);
    }

    #[test]
    fn an_edited_tree_round_trips_through_ron_after_a_sequence_of_edits() {
        let mut scene = fixture();
        for edit in edit_sequence() {
            let _ = apply(&mut scene, &edit).expect("the sequence is applicable");
        }
        let ron = scene.to_ron().expect("serialize");
        let recovered = Scene::from_ron("<round trip>", &ron).expect("deserialize");
        assert_eq!(
            recovered.to_wgsl(&EuclideanR3),
            scene.to_wgsl(&EuclideanR3),
            "{ron}",
        );
    }

    #[test]
    fn a_reloaded_tree_accepts_the_rest_of_the_sequence_identically() {
        let mut live = fixture();
        let mut reloaded = fixture();
        for edit in edit_sequence() {
            let live_result = apply(&mut live, &edit);
            let reloaded_result = apply(&mut reloaded, &edit);
            assert_eq!(live_result, reloaded_result);
            let ron = reloaded.to_ron().expect("serialize");
            reloaded = Scene::from_ron("<step>", &ron).expect("deserialize");
            assert_eq!(
                reloaded.to_wgsl(&EuclideanR3),
                live.to_wgsl(&EuclideanR3),
                "diverged after {:?}",
                edit.to_args(),
            );
        }
    }

    fn edit_sequence() -> Vec<SceneEdit> {
        let path = |text: &str| text.parse::<NodePath>().expect("path");
        vec![
            SceneEdit::Insert {
                path: path("root.0.1"),
                combinator: Combinator::Union,
                leaf: LeafKind::Sphere,
            },
            SceneEdit::Set {
                path: path("root.0.1.1"),
                param: Param::Center,
                value: EditValue::Vector(Vec3::new(0.7, -0.25, 0.4)),
            },
            SceneEdit::Set {
                path: path("root.0.1.1"),
                param: Param::Radius,
                value: EditValue::Scalar(0.18),
            },
            SceneEdit::Set {
                path: path("root.0.0"),
                param: Param::Blend,
                value: EditValue::Scalar(0.02),
            },
            SceneEdit::Set {
                path: path("root.0.1.0"),
                param: Param::Normal,
                value: EditValue::Vector(Vec3::new(0.2, 1.0, -0.3)),
            },
            SceneEdit::Insert {
                path: path("root.1"),
                combinator: Combinator::SmoothUnion,
                leaf: LeafKind::Box,
            },
            SceneEdit::Remove {
                path: path("root.1.1"),
            },
        ]
    }

    #[test]
    fn an_edit_round_trips_through_its_command_line_bit_exactly() {
        let awkward = [
            0.1_f32,
            -0.0,
            1e-7,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            3.4028235e38,
            -1.0 / 3.0,
        ];
        let mut edits = edit_sequence();
        for value in awkward {
            edits.push(SceneEdit::Set {
                path: "root.1".parse().expect("path"),
                param: Param::Radius,
                value: EditValue::Scalar(value),
            });
            edits.push(SceneEdit::Set {
                path: "root.1".parse().expect("path"),
                param: Param::Center,
                value: EditValue::Vector(Vec3::new(value, -value, 0.0)),
            });
        }
        for edit in edits {
            let args = edit.to_args();
            let refs: Vec<&str> = args.iter().map(String::as_str).collect();
            let parsed = SceneEdit::from_args(&refs).expect("its own spelling parses");
            assert_eq!(parsed, edit);
            for (a, b) in parsed
                .to_args()
                .iter()
                .zip(&args)
                .filter(|(a, b)| a.parse::<f32>().is_ok() && b.parse::<f32>().is_ok())
            {
                assert_eq!(
                    a.parse::<f32>().expect("number").to_bits(),
                    b.parse::<f32>().expect("number").to_bits(),
                );
            }
        }
    }

    #[test]
    fn a_malformed_command_line_is_a_syntax_error_rather_than_a_default() {
        for args in [
            vec!["set"],
            vec!["set", "root"],
            vec!["set", "root.9", "radius", "1.0"],
            vec!["set", "root", "girth", "1.0"],
            vec!["set", "root", "center", "1.0", "2.0"],
            vec!["set", "root", "radius", "wide"],
            vec!["add", "root", "union"],
            vec!["add", "root", "blend", "sphere"],
            vec!["add", "root", "union", "torus"],
            vec!["remove"],
            vec!["destroy", "root"],
            vec![],
        ] {
            assert!(
                SceneEdit::from_args(&args).is_err(),
                "`{args:?}` must not parse",
            );
        }
    }

    #[test]
    fn constants_the_emitter_cannot_bake_are_refused() {
        let mut scene = fixture();
        for bad in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            for edit in [
                SceneEdit::Set {
                    path: "root.1".parse().expect("path"),
                    param: Param::Radius,
                    value: EditValue::Scalar(bad),
                },
                SceneEdit::Set {
                    path: "root.1".parse().expect("path"),
                    param: Param::Center,
                    value: EditValue::Vector(Vec3::new(0.0, bad, 0.0)),
                },
                SceneEdit::Set {
                    path: "root.0.0".parse().expect("path"),
                    param: Param::Blend,
                    value: EditValue::Scalar(bad),
                },
            ] {
                assert!(matches!(
                    apply(&mut scene, &edit),
                    Err(EditError::Rejected(_)),
                ));
            }
        }
        // Zero and negative blend radii are finite and still unemittable: the
        // smooth minimum divides by `k`.
        for k in [0.0_f32, -0.1] {
            assert!(matches!(
                apply(
                    &mut scene,
                    &SceneEdit::Set {
                        path: "root.0.0".parse().expect("path"),
                        param: Param::Blend,
                        value: EditValue::Scalar(k),
                    },
                ),
                Err(EditError::Rejected(_)),
            ));
        }
        assert_eq!(scene.to_wgsl(&EuclideanR3), fixture().to_wgsl(&EuclideanR3));
    }

    #[test]
    fn a_half_space_normal_stays_on_the_unit_sphere_under_every_edit() {
        let mut scene = fixture();
        let path: NodePath = "root.0.1".parse().expect("path");
        for raw in [
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1e-3, 0.0, 0.0),
            Vec3::new(3.0, -4.0, 12.0),
        ] {
            // Not asserted as a change: the first raw normal here normalizes to
            // the one the fixture already carries, so it is a no-op.
            assert!(apply(
                &mut scene,
                &SceneEdit::Set {
                    path: path.clone(),
                    param: Param::Normal,
                    value: EditValue::Vector(raw),
                },
            )
            .is_ok());
            let stored = match node_at(&scene.root, &path) {
                Some(SceneNode::Leaf(Shape::HalfSpace { normal, .. })) => *normal,
                other => panic!("expected a half-space, got {other:?}"),
            };
            assert!(
                (stored.length() - 1.0).abs() < 1e-6,
                "normal {stored:?} off the unit sphere",
            );
            assert!(stored.dot(raw.normalize()) > 1.0 - 1e-6);
        }
        for degenerate in [Vec3::ZERO, Vec3::splat(1e-8)] {
            assert!(matches!(
                apply(
                    &mut scene,
                    &SceneEdit::Set {
                        path: path.clone(),
                        param: Param::Normal,
                        value: EditValue::Vector(degenerate),
                    },
                ),
                Err(EditError::Rejected(_)),
            ));
        }
    }

    #[test]
    fn sentinel_leaves_and_boolean_combinators_offer_no_parameters() {
        for shape in [
            Shape::ConvexPolytope3D {
                vertices: vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z],
            },
            Shape::Polygon2D {
                vertices: vec![glam::Vec2::ZERO, glam::Vec2::X, glam::Vec2::Y],
            },
            Shape::HyperSphere4D {
                center: glam::Vec4::ZERO,
                radius: 0.5,
            },
            Shape::HalfSpace4D {
                normal: glam::Vec4::Y,
                offset: 0.0,
            },
            Shape::ConvexPolytope4D {
                vertices: vec![glam::Vec4::ZERO, glam::Vec4::X],
            },
        ] {
            assert!(
                parameters(&SceneNode::Leaf(shape.clone())).is_empty(),
                "{:?} advertises parameters it cannot render",
                shape.kind(),
            );
        }
        let leaf = || SceneNode::sphere(Vec3::ZERO, 0.1);
        for node in [
            leaf().union(leaf()),
            leaf().intersect(leaf()),
            leaf().subtract(leaf()),
        ] {
            assert!(parameters(&node).is_empty());
        }
        assert_eq!(
            parameters(&leaf().smooth_union(leaf(), 0.2)),
            vec![(Param::Blend, EditValue::Scalar(0.2))],
        );
    }

    #[test]
    fn advertised_parameter_values_are_the_ones_the_node_holds() {
        let scene = fixture();
        for path in paths(&scene) {
            let node = node_at(&scene.root, &path).expect("enumerated");
            for (param, value) in parameters(node) {
                let mut probe = scene.clone();
                assert_eq!(
                    apply(
                        &mut probe,
                        &SceneEdit::Set {
                            path: path.clone(),
                            param,
                            value,
                        },
                    ),
                    Ok(false),
                    "{path} `{}` reported a value the node does not hold",
                    param.name(),
                );
            }
        }
    }
}
