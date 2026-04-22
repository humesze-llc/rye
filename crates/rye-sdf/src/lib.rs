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
/// The default scene is seven geodesic spheres:
/// - one center sphere
/// - six orbit spheres around the center
///
/// Optionally, an Euclidean-y slab can be enabled as a visual cage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeodesicSpheresScene {
    pub sphere_radius: f32,
    pub orbit_radius: f32,
    pub orbit_height: f32,
    pub include_slab: bool,
    pub floor_y: f32,
    pub ceiling_y: f32,
}

impl Default for GeodesicSpheresScene {
    fn default() -> Self {
        Self {
            sphere_radius: 0.22,
            orbit_radius: 0.62,
            orbit_height: 0.18,
            include_slab: false,
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
    pub fn with_slab(mut self, floor_y: f32, ceiling_y: f32) -> Self {
        self.include_slab = true;
        self.floor_y = floor_y;
        self.ceiling_y = ceiling_y;
        self
    }

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
        if self.include_slab {
            out.push_str(&format!(
                "const RYE_SCENE_FLOOR_Y: f32 = {:.6};\n",
                self.floor_y
            ));
            out.push_str(&format!(
                "const RYE_SCENE_CEILING_Y: f32 = {:.6};\n",
                self.ceiling_y
            ));
        }
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
"#,
        );
        if self.include_slab {
            out.push_str(
                r#"
fn rye_scene_slab_sdf(p: vec3<f32>) -> f32 {
    let floor_d = p.y - RYE_SCENE_FLOOR_Y;
    let ceiling_d = RYE_SCENE_CEILING_Y - p.y;
    return min(floor_d, ceiling_d);
}
"#,
            );
        }
        out.push_str("\nfn rye_scene_sdf(p: vec3<f32>) -> f32 {\n");
        out.push_str("    var d = 1e9;\n\n");
        out.push_str("    // Center geodesic sphere.\n");
        out.push_str(
            "    d = min(d, rye_distance(p, vec3<f32>(0.0, 0.12, 0.0)) - RYE_SCENE_SPHERE_RADIUS);\n\n",
        );
        out.push_str("    // Six orbit spheres.\n");
        out.push_str("    for (var i = 0; i < 6; i = i + 1) {\n");
        out.push_str("        let c = rye_scene_orbit_center(i);\n");
        out.push_str("        d = min(d, rye_distance(p, c) - RYE_SCENE_SPHERE_RADIUS);\n");
        out.push_str("    }\n\n");
        if self.include_slab {
            out.push_str("    d = min(d, rye_scene_slab_sdf(p));\n\n");
        }
        out.push_str("    return d;\n");
        out.push_str("}\n");
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

    #[test]
    fn default_scene_has_no_slab_constants() {
        let src = geodesic_spheres_demo_wgsl();
        assert!(!src.contains("RYE_SCENE_FLOOR_Y"));
        assert!(!src.contains("RYE_SCENE_CEILING_Y"));
    }

    #[test]
    fn slab_scene_emits_slab_constants() {
        let src = GeodesicSpheresScene::default()
            .with_slab(-0.5, 0.8)
            .to_wgsl();
        assert!(src.contains("RYE_SCENE_FLOOR_Y"));
        assert!(src.contains("RYE_SCENE_CEILING_Y"));
    }
}
