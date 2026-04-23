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

/// A rectangular corridor oriented along the Z axis.
///
/// Floor, ceiling, and side walls are Euclidean-coordinate planes in the
/// active Space chart (`p.y + H`, `H - p.y`, `p.x + W`, `W - p.x`). They
/// are *not* geodesic surfaces — and that is the point. With geodesic ray
/// marching, the rays bend according to the metric, so the same axis-aligned
/// planes appear flat in E³, bowed outward in H³ (parallel geodesics diverge),
/// and converging in S³ (parallel geodesics meet).
///
/// Rows of geodesic-sphere pillars line both sides of the corridor along
/// the floor, providing depth cues so the curvature is visually obvious.
/// Pillars use `rye_distance` so they are space-aware (perfect spheres in
/// each metric). The walls use raw coordinate arithmetic so the difference
/// between metrics is carried entirely by the ray bending.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorridorScene {
    /// Half-width of the corridor along the Space X axis.
    pub half_width: f32,
    /// Half-height of the corridor along the Space Y axis.
    pub half_height: f32,
    /// Half-length along Z. End walls close the corridor so every ray
    /// hits geometry before reaching the Poincaré ball boundary in H³/S³.
    /// An open corridor (no end walls) lets rays drift into the
    /// saturation shell where Euclidean-coord wall SDFs produce noise.
    pub half_depth: f32,
    /// Geodesic radius of each pillar sphere.
    pub pillar_radius: f32,
    /// Distance from centerline to each pillar row (Space X).
    pub pillar_x_offset: f32,
    /// Y coordinate of pillar centers (typically `-half_height + pillar_radius`).
    pub pillar_y: f32,
    /// Geodesic spacing between consecutive pillars along Z.
    pub pillar_z_spacing: f32,
    /// Total pillars per row. Must be odd (symmetric about z = 0).
    pub pillars_per_row: u32,
}

impl Default for CorridorScene {
    fn default() -> Self {
        Self {
            half_width: 0.55,
            half_height: 0.40,
            // 0.70 keeps the end walls well inside the Poincaré ball
            // (|p| ≤ 0.70 has conformal factor ~3.9, still well-conditioned).
            half_depth: 0.70,
            pillar_radius: 0.07,
            pillar_x_offset: 0.40,
            pillar_y: -0.33,
            pillar_z_spacing: 0.24,
            pillars_per_row: 7,
        }
    }
}

impl CorridorScene {
    pub fn to_wgsl(&self) -> String {
        assert!(
            self.pillars_per_row % 2 == 1,
            "pillars_per_row must be odd so the row is symmetric about z=0"
        );
        let mut out = String::new();
        out.push_str("// ---- rye-sdf scene module: corridor ----\n");
        out.push_str(&format!(
            "const RYE_CORR_HALF_W: f32 = {:.6};\n",
            self.half_width
        ));
        out.push_str(&format!(
            "const RYE_CORR_HALF_H: f32 = {:.6};\n",
            self.half_height
        ));
        out.push_str(&format!(
            "const RYE_CORR_HALF_D: f32 = {:.6};\n",
            self.half_depth
        ));
        out.push_str(&format!(
            "const RYE_CORR_PILLAR_R: f32 = {:.6};\n",
            self.pillar_radius
        ));
        out.push_str(&format!(
            "const RYE_CORR_PILLAR_X: f32 = {:.6};\n",
            self.pillar_x_offset
        ));
        out.push_str(&format!(
            "const RYE_CORR_PILLAR_Y: f32 = {:.6};\n",
            self.pillar_y
        ));
        out.push_str(&format!(
            "const RYE_CORR_PILLAR_DZ: f32 = {:.6};\n",
            self.pillar_z_spacing
        ));
        out.push_str(&format!(
            "const RYE_CORR_PILLAR_HALF: i32 = {};\n\n",
            (self.pillars_per_row - 1) / 2
        ));

        out.push_str(
            r#"fn rye_scene_sdf(p: vec3<f32>) -> f32 {
    // Floor / ceiling / side walls: Euclidean-coord half-spaces in Space.
    // With geodesic ray marching the rays bend in H³/S³, so these flat
    // planes read as curved walls.
    let floor_d   = p.y + RYE_CORR_HALF_H;
    let ceiling_d = RYE_CORR_HALF_H - p.y;
    let left_d    = p.x + RYE_CORR_HALF_W;
    let right_d   = RYE_CORR_HALF_W - p.x;
    let back_d    = p.z + RYE_CORR_HALF_D;
    let front_d   = RYE_CORR_HALF_D - p.z;
    var d = min(min(min(floor_d, ceiling_d), min(left_d, right_d)), min(back_d, front_d));

    // Two rows of geodesic-sphere pillars along the floor, symmetric about z=0.
    for (var i = -RYE_CORR_PILLAR_HALF; i <= RYE_CORR_PILLAR_HALF; i = i + 1) {
        let z  = f32(i) * RYE_CORR_PILLAR_DZ;
        let pl = vec3<f32>(-RYE_CORR_PILLAR_X, RYE_CORR_PILLAR_Y, z);
        let pr = vec3<f32>( RYE_CORR_PILLAR_X, RYE_CORR_PILLAR_Y, z);
        let dl = rye_distance(p, pl) - RYE_CORR_PILLAR_R;
        let dr = rye_distance(p, pr) - RYE_CORR_PILLAR_R;
        d = min(d, min(dl, dr));
    }

    return d;
}
"#,
        );
        out
    }
}

/// Emit WGSL source for the default `CorridorScene`.
pub fn corridor_demo_wgsl() -> String {
    CorridorScene::default().to_wgsl()
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
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
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
        // E³ step-1 center along +X should be at (0.5, 0, 0)
        assert!(src.contains("0.500000, 0.000000, 0.000000"));
    }

    #[test]
    fn corridor_scene_emits_required_entrypoint() {
        let src = corridor_demo_wgsl();
        assert!(src.contains("fn rye_scene_sdf"));
        assert!(src.contains("RYE_CORR_HALF_W"));
        assert!(src.contains("RYE_CORR_HALF_H"));
        assert!(src.contains("rye_distance"));
    }

    #[test]
    #[should_panic(expected = "pillars_per_row must be odd")]
    fn corridor_scene_rejects_even_pillar_count() {
        let scene = CorridorScene {
            pillars_per_row: 6,
            ..Default::default()
        };
        let _ = scene.to_wgsl();
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
        // H³ step-1 center along +X: tanh(0.25) ≈ 0.2449, which is < 0.5.
        // The WGSL source should not contain "0.500000" as an x-coord.
        assert!(!src.contains("0.500000, 0.000000, 0.000000"));
    }
}
