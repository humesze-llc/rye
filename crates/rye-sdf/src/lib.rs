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

use glam::Vec3;
use rye_math::Space;

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

/// Periodic geodesic lattice scene.
///
/// Emits a `rye_scene_sdf` that places a sphere at the origin and at
/// geodesic lattice positions along the ±X, ±Y, ±Z axes. The lattice
/// centers are computed in Rust by calling `space.exp` so that they
/// live at evenly-spaced geodesic intervals in the given Space:
///
/// - E³: evenly-spaced Euclidean grid
/// - H³: tanh-compressed toward the Poincaré boundary
/// - S³: sin-based, wrapping past π/2
///
/// The visual difference between the three spaces emerges naturally
/// when the same shader is compiled against each Space prelude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatticeSphereScene {
    /// Number of shells along each axis direction (1..=N).
    pub steps: u32,
    /// Geodesic arc length between consecutive shells.
    pub geodesic_spacing: f32,
    /// Sphere radius in Space coordinates.
    pub sphere_radius: f32,
}

impl Default for LatticeSphereScene {
    fn default() -> Self {
        Self {
            steps: 2,
            geodesic_spacing: 0.45,
            sphere_radius: 0.12,
        }
    }
}

impl LatticeSphereScene {
    /// Emit a WGSL scene module for the given Space.
    ///
    /// Centers are computed via `space.exp` and embedded as literal
    /// `vec3<f32>` constants — no `rye_exp` calls inside the march loop.
    /// The emitted `rye_scene_sdf` calls only `rye_distance`, so the
    /// spatial metric is fully Space-aware at runtime.
    pub fn to_wgsl<S>(&self, space: &S) -> String
    where
        S: Space<Point = Vec3, Vector = Vec3>,
    {
        let mut centers: Vec<Vec3> = vec![Vec3::ZERO];

        let axes = [
            Vec3::X, Vec3::NEG_X,
            Vec3::Y, Vec3::NEG_Y,
            Vec3::Z, Vec3::NEG_Z,
        ];
        for axis in axes {
            for k in 1..=self.steps {
                let tangent = axis * (k as f32 * self.geodesic_spacing);
                centers.push(space.exp(Vec3::ZERO, tangent));
            }
        }

        let mut out = String::new();
        out.push_str("// ---- rye-sdf scene module: lattice_spheres ----\n");
        out.push_str(&format!(
            "const RYE_LATTICE_RADIUS: f32 = {:.6};\n\n",
            self.sphere_radius
        ));
        out.push_str("fn rye_scene_sdf(p: vec3<f32>) -> f32 {\n");
        out.push_str("    var d = 1e9;\n");
        for c in &centers {
            out.push_str(&format!(
                "    d = min(d, rye_distance(p, vec3<f32>({:.6}, {:.6}, {:.6})) - RYE_LATTICE_RADIUS);\n",
                c.x, c.y, c.z
            ));
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

    #[test]
    fn lattice_scene_emits_required_entrypoint() {
        use rye_math::EuclideanR3;
        let src = LatticeSphereScene::default().to_wgsl(&EuclideanR3);
        assert!(src.contains("fn rye_scene_sdf"));
        assert!(src.contains("rye_distance"));
        assert!(src.contains("RYE_LATTICE_RADIUS"));
    }

    #[test]
    fn lattice_scene_euclidean_centers_are_evenly_spaced() {
        use rye_math::EuclideanR3;
        let scene = LatticeSphereScene {
            steps: 1,
            geodesic_spacing: 0.5,
            sphere_radius: 0.1,
        };
        let src = scene.to_wgsl(&EuclideanR3);
        assert!(src.contains("0.500000, 0.000000, 0.000000"));
    }

    #[test]
    fn lattice_scene_hyperbolic_centers_are_compressed() {
        use rye_math::HyperbolicH3;
        let scene = LatticeSphereScene {
            steps: 1,
            geodesic_spacing: 0.5,
            sphere_radius: 0.1,
        };
        let src = scene.to_wgsl(&HyperbolicH3);
        assert!(!src.contains("0.500000, 0.000000, 0.000000"));
    }
}
