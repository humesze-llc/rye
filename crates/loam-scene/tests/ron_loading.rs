//! Scene descriptions arriving as file text: what a `.ron` file must
//! deserialize to, and what every category of malformed input must do instead
//! of panicking.

use std::path::{Path, PathBuf};

use glam::{Vec3, Vec4};
use loam_math::EuclideanR3;
use loam_scene::load::SceneLoadError;
use loam_scene::scene::{Scene, SceneNode};
use loam_scene::scene4::{Scene4, SceneNode4};

fn scenes_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("scenes")
}

/// A file the test owns for the duration of one case. Named after the case so
/// concurrently running tests cannot collide on it.
struct TempScene(PathBuf);

impl TempScene {
    fn new(case: &str, contents: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("loam-scene-{}-{case}.ron", std::process::id()));
        std::fs::write(&path, contents).expect("temp scene is writable");
        Self(path)
    }
}

impl Drop for TempScene {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// The committed example must deserialize to the tree its text spells, not
/// merely to some tree: a swapped operand or a dropped combinator would still
/// parse. Emitted WGSL is the comparison because it carries both the structure
/// and every baked constant.
#[test]
fn committed_scene_file_deserializes_to_the_tree_it_spells() {
    let loaded = Scene::load(scenes_dir().join("sphere_over_floor.ron")).expect("scene loads");
    let expected = Scene::new(
        SceneNode::sphere(Vec3::new(0.0, 0.15, 0.0), 0.3)
            .smooth_union(SceneNode::box_(Vec3::new(0.4, 0.1, 0.4)), 0.08)
            .union(SceneNode::plane(Vec3::Y, -0.5)),
    );
    assert_eq!(loaded.to_wgsl(&EuclideanR3), expected.to_wgsl(&EuclideanR3));
}

#[test]
fn committed_scene4_file_deserializes_to_the_tree_it_spells() {
    let loaded =
        Scene4::load(scenes_dir().join("hypersphere_over_floor.ron")).expect("4D scene loads");
    let expected = Scene4::new(
        SceneNode4::hypersphere(Vec4::new(0.0, 0.2, 0.0, 0.0), 0.45)
            .union(SceneNode4::halfspace(Vec4::Y, -0.5)),
    );
    assert_eq!(loaded.to_wgsl_4d(), expected.to_wgsl_4d());
}

/// Loading is only half the contract: a scene that came from a file has to
/// reach a shader. Both committed examples are emitted and put through naga,
/// so a file that parses but bakes a token WGSL rejects fails here rather than
/// at pipeline creation in a demo.
#[test]
fn a_loaded_scene_emits_wgsl_naga_accepts() {
    use loam_math::WgslSpace;

    fn assert_naga_accepts(source: &str) {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("WGSL parse failed: {e}\n--- source ---\n{source}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("WGSL validation failed: {e:?}\n--- source ---\n{source}"));
    }

    let scene = Scene::load(scenes_dir().join("sphere_over_floor.ron")).expect("scene loads");
    assert_naga_accepts(&format!(
        "{prelude}\n{scene}\n\
         @compute @workgroup_size(1) fn main() {{\n\
         \t_ = loam_scene_sdf(vec3<f32>(0.0));\n\
         }}\n",
        prelude = EuclideanR3.wgsl_impl(),
        scene = scene.to_wgsl(&EuclideanR3),
    ));

    let scene4 =
        Scene4::load(scenes_dir().join("hypersphere_over_floor.ron")).expect("4D scene loads");
    assert_naga_accepts(&format!(
        "{scene}\n\
         @compute @workgroup_size(1) fn main() {{\n\
         \t_ = loam_scene_sdf_4d(vec4<f32>(0.0));\n\
         }}\n",
        scene = scene4.to_wgsl_4d(),
    ));
}

/// A path that does not resolve is a `Read`, distinguishable from a file that
/// exists and is wrong: a caller falling back to a built-in scene when there is
/// no file must not also swallow a typo inside one.
#[test]
fn a_missing_file_is_a_read_error_naming_the_path() {
    let path = scenes_dir().join("no_such_scene.ron");
    let err = Scene::load(&path).expect_err("a missing file cannot load");
    assert!(
        matches!(err, SceneLoadError::Read { .. }),
        "expected a read error, got {err:?}",
    );
    assert!(
        err.to_string().contains("no_such_scene.ron"),
        "error must name the path it tried: {err}",
    );
}

/// The position is the whole point of reporting a syntax error: without it the
/// author of a hundred-line scene file has nowhere to look.
#[test]
fn a_syntax_error_is_reported_with_its_path_and_position() {
    let scene = TempScene::new(
        "syntax",
        "(\n    root: Leaf(Sphere(\n        center: (0.0, 0.0 0.0),\n        radius: 0.3,\n    )),\n)\n",
    );
    let err = Scene::load(&scene.0).expect_err("a missing separator cannot load");
    assert!(
        matches!(err, SceneLoadError::Parse { .. }),
        "expected a parse error, got {err:?}",
    );
    let message = err.to_string();
    assert!(
        message.contains("syntax") && message.contains("3:"),
        "error must name the file and the line: {message}",
    );
}

/// A leaf naming a shape that does not exist is rejected rather than skipped,
/// so a renamed variant fails loudly at the file that still spells the old name.
#[test]
fn an_unknown_leaf_variant_is_rejected() {
    let err = Scene::from_ron(
        "unknown_variant",
        "(root: Leaf(Torus(major: 1.0, minor: 0.2)))",
    )
    .expect_err("Torus is not a Shape");
    assert!(
        matches!(err, SceneLoadError::Parse { .. }),
        "expected a parse error, got {err:?}",
    );
}

/// The emitter asserts on constants WGSL cannot spell, so without a check at
/// the load boundary a one-word edit to a scene file would take the process
/// down inside `to_wgsl`. Swept over both dimensions and over infinity and NaN.
#[test]
fn a_non_finite_constant_is_rejected_at_load_rather_than_reaching_the_emitter() {
    for spelling in ["inf", "-inf", "NaN"] {
        let err = Scene::from_ron(
            "non_finite",
            &format!("(root: Leaf(Sphere(center: (0.0, 0.0, 0.0), radius: {spelling})))"),
        )
        .expect_err("a non-finite radius cannot load");
        assert!(
            matches!(err, SceneLoadError::Invalid { .. }),
            "radius {spelling}: expected an invalid-description error, got {err:?}",
        );
        assert!(
            err.to_string().contains("non-finite"),
            "radius {spelling}: error must say what is wrong: {err}",
        );

        let err = Scene4::from_ron(
            "non_finite_4d",
            &format!("(root: Leaf(HalfSpace4D(normal: (0.0, 1.0, 0.0, {spelling}), offset: 0.0)))"),
        )
        .expect_err("a non-finite normal cannot load");
        assert!(
            matches!(err, SceneLoadError::Invalid { .. }),
            "normal.w {spelling}: expected an invalid-description error, got {err:?}",
        );
    }
}

/// A non-finite constant nested under combinators is still caught: the check
/// walks the tree rather than inspecting the root.
#[test]
fn a_non_finite_constant_below_a_combinator_is_still_rejected() {
    let err = Scene::from_ron(
        "nested_non_finite",
        "(root: Union(\
            Leaf(Sphere(center: (0.0, 0.0, 0.0), radius: 0.3)),\
            Difference(\
                Leaf(Box3(half_extents: (1.0, 1.0, 1.0))),\
                Leaf(HalfSpace(normal: (0.0, 1.0, 0.0), offset: inf)),\
            ),\
        ))",
    )
    .expect_err("a non-finite offset cannot load, however deep it sits");
    assert!(matches!(err, SceneLoadError::Invalid { .. }), "{err:?}");
}

/// `k` divides in both the emitted `smin` and its CPU twin, and a negative `k`
/// stops the result being an underestimate of `min`, so neither is a scene the
/// loader may hand on.
#[test]
fn a_non_positive_blend_radius_is_rejected() {
    for k in ["0.0", "-0.5"] {
        let err = Scene::from_ron(
            "blend_radius",
            &format!(
                "(root: SmoothUnion(k: {k}, \
                 left: Leaf(Sphere(center: (0.0, 0.0, 0.0), radius: 0.3)), \
                 right: Leaf(Box3(half_extents: (0.2, 0.2, 0.2)))))"
            ),
        )
        .expect_err("k must be positive");
        assert!(
            matches!(err, SceneLoadError::Invalid { .. }),
            "k = {k}: expected an invalid-description error, got {err:?}",
        );
        assert!(
            err.to_string().contains("blend radius"),
            "k = {k}: error must name the field: {err}",
        );
    }
}

/// The scene tree is recursive and so is every walk over it, so nesting depth
/// is the one malformed input that could take the process down without an
/// assert to blame. RON's own recursion limit is what bounds it; this pins that
/// the limit is in force on the path the loader actually uses.
#[test]
fn deeply_nested_input_errors_instead_of_exhausting_the_stack() {
    const DEPTH: usize = 5_000;
    let mut src = String::from("(root: ");
    for _ in 0..DEPTH {
        src.push_str("Union(");
    }
    src.push_str("Leaf(Sphere(center: (0.0, 0.0, 0.0), radius: 0.1))");
    for _ in 0..DEPTH {
        src.push_str(", Leaf(Box3(half_extents: (0.1, 0.1, 0.1))))");
    }
    src.push(')');

    let err =
        Scene::from_ron("deep_nesting", &src).expect_err("nesting past the limit cannot load");
    assert!(
        matches!(err, SceneLoadError::Parse { .. }),
        "expected a parse error, got {err:?}",
    );
}

/// Empty input is a parse failure, not an empty scene: a truncated or
/// zero-length file must not deserialize into anything.
#[test]
fn empty_input_is_a_parse_error() {
    let scene = TempScene::new("empty", "");
    let err = Scene::load(&scene.0).expect_err("an empty file is not a scene");
    assert!(matches!(err, SceneLoadError::Parse { .. }), "{err:?}");
}

/// Trailing text after the description is rejected rather than ignored, so a
/// half-finished edit below a valid tree cannot load as if it were not there.
#[test]
fn trailing_text_after_the_description_is_rejected() {
    let err = Scene::from_ron(
        "trailing",
        "(root: Leaf(Sphere(center: (0.0, 0.0, 0.0), radius: 0.3))) and then some",
    )
    .expect_err("trailing text is not part of the scene");
    assert!(matches!(err, SceneLoadError::Parse { .. }), "{err:?}");
}
