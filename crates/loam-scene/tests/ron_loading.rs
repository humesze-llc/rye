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

#[test]
fn a_non_finite_constant_below_any_combinator_arm_is_rejected() {
    const GOOD: &str = "Leaf(Sphere(center: (0.0, 0.0, 0.0), radius: 0.3))";
    const BAD: &str = "Leaf(HalfSpace(normal: (0.0, 1.0, 0.0), offset: inf))";

    for combinator in ["Union", "Intersection", "Difference", "SmoothUnion"] {
        for (inner_side, (left, right)) in [("left", (BAD, GOOD)), ("right", (GOOD, BAD))] {
            let inner = if combinator == "SmoothUnion" {
                format!("SmoothUnion(k: 0.08, left: {left}, right: {right})")
            } else {
                format!("{combinator}({left}, {right})")
            };
            for (outer_side, src) in [
                ("left", format!("(root: Union({inner}, {GOOD}))")),
                ("right", format!("(root: Union({GOOD}, {inner}))")),
            ] {
                let err = Scene::from_ron("nested_non_finite", &src)
                    .expect_err("a non-finite offset cannot load, however deep it sits");
                assert!(
                    matches!(err, SceneLoadError::Invalid { .. }),
                    "{combinator} {inner_side} arm under the outer {outer_side} arm: \
                     expected an invalid-description error, got {err:?}",
                );
            }
        }
    }
}

#[test]
fn a_non_finite_constant_below_any_4d_combinator_arm_is_rejected() {
    const GOOD: &str = "Leaf(HyperSphere4D(center: (0.0, 0.0, 0.0, 0.0), radius: 0.3))";
    const BAD: &str = "Leaf(HalfSpace4D(normal: (0.0, 1.0, 0.0, 0.0), offset: inf))";

    for combinator in ["Union", "Intersection", "Difference"] {
        for (inner_side, (left, right)) in [("left", (BAD, GOOD)), ("right", (GOOD, BAD))] {
            let inner = format!("{combinator}({left}, {right})");
            for (outer_side, src) in [
                ("left", format!("(root: Union({inner}, {GOOD}))")),
                ("right", format!("(root: Union({GOOD}, {inner}))")),
            ] {
                let err = Scene4::from_ron("nested_non_finite_4d", &src)
                    .expect_err("a non-finite offset cannot load, however deep it sits");
                assert!(
                    matches!(err, SceneLoadError::Invalid { .. }),
                    "{combinator} {inner_side} arm under the outer {outer_side} arm: \
                     expected an invalid-description error, got {err:?}",
                );
            }
        }
    }
}

#[test]
fn a_blend_radius_that_is_not_finite_and_positive_is_rejected() {
    for k in ["0.0", "-0.5", "inf", "-inf", "NaN"] {
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

#[test]
fn empty_input_is_a_parse_error() {
    let scene = TempScene::new("empty", "");
    let err = Scene::load(&scene.0).expect_err("an empty file is not a scene");
    assert!(matches!(err, SceneLoadError::Parse { .. }), "{err:?}");
}

#[test]
fn trailing_text_after_the_description_is_rejected() {
    let err = Scene::from_ron(
        "trailing",
        "(root: Leaf(Sphere(center: (0.0, 0.0, 0.0), radius: 0.3))) and then some",
    )
    .expect_err("trailing text is not part of the scene");
    assert!(matches!(err, SceneLoadError::Parse { .. }), "{err:?}");
}
