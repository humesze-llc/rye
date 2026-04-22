//! `rye-sdf` - tiny scene-module builders for Rye.
//!
//! Phase 2 starts with a narrow vertical slice: emit WGSL scene modules
//! that define:
//!
//! `fn rye_scene_sdf(p: vec3<f32>) -> f32`
//!
//! The scene is evaluated in the active Space coordinates, so authors
//! can use `rye_distance` directly and get Euclidean / H3 / S3 behavior
//! from the same module source.

/// Geodesic-spheres demo scene parameters.
///
/// The scene is intentionally simple but non-trivial:
/// - two horizontal planes (`floor` and `ceiling`)
/// - one center sphere
/// - six orbit spheres around the center
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeodesicSpheresScene {
    pub sphere_radius: f32,
    pub orbit_radius: f32,
    pub orbit_height: f32,
    pub floor_y: f32,
    pub ceiling_y: f32,
}

impl Default for GeodesicSpheresScene {
    fn default() -> Self {
        Self {
            sphere_radius: 0.22,
            orbit_radius: 0.62,
            orbit_height: 0.18,
            floor_y: -0.50,
            ceiling_y: 0.82,
        }
    }
}

/// Emit WGSL source defining `rye_scene_sdf`.
pub fn geodesic_spheres_demo_wgsl() -> String {
    GeodesicSpheresScene::default().to_wgsl()
}

impl GeodesicSpheresScene {
    pub fn to_wgsl(&self) -> String {
        let mut out = String::new();
        out.push_str("// ---- rye-sdf scene module: geodesic_spheres_demo ----\n");
        out.push_str(&format!(
            "const RYE_SCENE_SPHERE_RADIUS: f32 = {:.6};\n",
            self.sphere_radius
        ));
        out.push_str(&format!(
            "const RYE_SCENE_ORBIT_RADIUS: f32 = {:.6};\n",
            self.orbit_radius
        ));
        out.push_str(&format!(
            "const RYE_SCENE_ORBIT_HEIGHT: f32 = {:.6};\n",
            self.orbit_height
        ));
        out.push_str(&format!(
            "const RYE_SCENE_FLOOR_Y: f32 = {:.6};\n",
            self.floor_y
        ));
        out.push_str(&format!(
            "const RYE_SCENE_CEILING_Y: f32 = {:.6};\n",
            self.ceiling_y
        ));
        out.push_str(
            r#"
fn rye_scene_orbit_center(i: i32) -> vec3<f32> {
    let a = f32(i) * 1.0471976; // 2*pi/6
    return vec3<f32>(
        cos(a) * RYE_SCENE_ORBIT_RADIUS,
        RYE_SCENE_ORBIT_HEIGHT,
        sin(a) * RYE_SCENE_ORBIT_RADIUS
    );
}

fn rye_scene_sdf(p: vec3<f32>) -> f32 {
    var d = 1e9;

    // Horizontal slab keeps content in view.
    d = min(d, p.y - RYE_SCENE_FLOOR_Y);
    d = min(d, RYE_SCENE_CEILING_Y - p.y);

    // Center geodesic sphere.
    d = min(d, rye_distance(p, vec3<f32>(0.0, 0.12, 0.0)) - RYE_SCENE_SPHERE_RADIUS);

    // Six orbit spheres.
    for (var i = 0; i < 6; i = i + 1) {
        let c = rye_scene_orbit_center(i);
        d = min(d, rye_distance(p, c) - RYE_SCENE_SPHERE_RADIUS);
    }

    return d;
}
"#,
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_required_scene_entrypoint() {
        let src = geodesic_spheres_demo_wgsl();
        assert!(src.contains("fn rye_scene_sdf"));
        assert!(src.contains("rye_distance"));
    }
}
