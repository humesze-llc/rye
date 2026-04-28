//! `rye-sdf`: signed-distance field primitives and scene builders for Rye.
//!
//! [`Primitive`] is the typed abstraction for geometric objects. Every
//! primitive emits a WGSL function `fn {name}(p: vec3<f32>) -> f32` that
//! uses only `rye_*` Space-prelude functions, guaranteeing correctness
//! across E³, H³, and S³.
//!
//! [`combinator`] provides Space-agnostic combinators (union, intersection,
//! smooth-min) that operate on the scalar distances returned by primitive SDFs.
//!
//! [`GeodesicSpheresScene`], [`CorridorScene`], and [`LatticeSphereScene`]
//! are demo-shaped scene builders consumed by the corresponding examples.
//! They are constructed on top of the typed primitive layer (each has a
//! `to_scene` method that returns a [`scene::Scene`]).

pub mod combinator;
pub mod primitive;
pub mod primitive4;
pub mod scene;
pub mod scene4;

pub use primitive::Primitive;
pub use primitive4::Primitive4;
pub use rye_shape::Shape;
pub use scene::{PrimitiveKind, Scene, SceneNode};
pub use scene4::{Scene4, SceneNode4};

use std::f32::consts::PI;

use glam::Vec3;
use rye_math::{EuclideanR3, Space, WgslSpace};

/// Geodesic-spheres demo scene parameters.
///
/// The default scene is seven geodesic spheres:
/// - one center sphere
/// - six orbit spheres around the center
///
/// Optionally, a Euclidean-y slab (floor / ceiling planes) can be
/// enabled as a visual cage. Note: half-space SDF emission currently
/// returns the `+1e9` sentinel (see [`Primitive`]), so the slab
/// renders invisible until a closed-form geodesic-plane SDF lands.
/// The geodesic spheres themselves render correctly in every Space.
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

    /// Build the typed scene tree. Orbit centers are pre-computed in Rust
    /// and embedded as literals; `rye_distance` carries the Space metric.
    pub fn to_scene(&self) -> Scene {
        let mut node = SceneNode::sphere(Vec3::new(0.0, 0.12, 0.0), self.sphere_radius);
        for i in 0..6 {
            let a = (i as f32) * PI / 3.0;
            let center = Vec3::new(
                a.cos() * self.orbit_radius,
                self.orbit_height,
                a.sin() * self.orbit_radius,
            );
            node = node.union(SceneNode::sphere(center, self.sphere_radius));
        }
        if self.include_slab {
            // floor: SDF = p.y - floor_y (positive above floor)
            // ceiling: SDF = ceiling_y - p.y (positive below ceiling)
            // union = min(floor_sdf, ceiling_sdf): positive inside slab, terminates march at walls
            let floor = SceneNode::plane(Vec3::Y, self.floor_y);
            let ceiling = SceneNode::plane(Vec3::NEG_Y, -self.ceiling_y);
            node = node.union(floor.union(ceiling));
        }
        Scene::new(node)
    }

    pub fn to_wgsl(&self) -> String {
        self.to_scene().to_wgsl(&EuclideanR3)
    }
}

/// A rectangular corridor oriented along the Z axis, lined with
/// rows of geodesic spheres for depth cues.
///
/// Pillars use `rye_distance` so they are space-aware (perfect
/// spheres in every metric, with the curvature carried by ray
/// bending). Floor, ceiling, and side walls were originally
/// Euclidean-coordinate planes (`p.y + H`, `H - p.y`, etc.)
/// chosen specifically to visualise the chart-vs-geodesic
/// difference. That emission has since been gated to the `+1e9`
/// sentinel (see [`Primitive`]), so the walls currently render
/// invisible. The pillars still tell the curvature story; the
/// surrounding cage is dormant pending a closed-form
/// geodesic-plane SDF.
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
    /// Build the typed scene tree for the corridor.
    ///
    /// Walls are constructed as `SceneNode::plane(...)` leaves, but
    /// `Primitive::to_wgsl` currently sentinels half-space emission
    /// (the chart-coord `dot(p, n) - d` form was a doc lie in
    /// curved Spaces); the walls render invisible until a
    /// closed-form geodesic-plane SDF replaces the sentinel.
    /// Pillars are geodesic spheres (`rye_distance`) and render
    /// honestly in every Space.
    pub fn to_scene(&self) -> Scene {
        assert!(
            self.pillars_per_row % 2 == 1,
            "pillars_per_row must be odd so the row is symmetric about z=0"
        );
        // Six walls as Euclidean half-space planes.
        // Each plane's SDF is positive on the interior side of that wall.
        let walls = SceneNode::plane(Vec3::Y, -self.half_height) // floor: p.y + H
            .union(SceneNode::plane(Vec3::NEG_Y, -self.half_height)) // ceiling: H - p.y
            .union(SceneNode::plane(Vec3::X, -self.half_width)) // left: p.x + W
            .union(SceneNode::plane(Vec3::NEG_X, -self.half_width)) // right: W - p.x
            .union(SceneNode::plane(Vec3::Z, -self.half_depth)) // back: p.z + D
            .union(SceneNode::plane(Vec3::NEG_Z, -self.half_depth)); // front: D - p.z

        let half = (self.pillars_per_row - 1) / 2;
        let mut root = walls;
        for i in -(half as i32)..=(half as i32) {
            let z = (i as f32) * self.pillar_z_spacing;
            root = root.union(SceneNode::sphere(
                Vec3::new(-self.pillar_x_offset, self.pillar_y, z),
                self.pillar_radius,
            ));
            root = root.union(SceneNode::sphere(
                Vec3::new(self.pillar_x_offset, self.pillar_y, z),
                self.pillar_radius,
            ));
        }
        Scene::new(root)
    }

    pub fn to_wgsl(&self) -> String {
        self.to_scene().to_wgsl(&EuclideanR3)
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
    /// Build the typed scene tree for the given Space.
    ///
    /// Centers are computed via `space.exp` and stored as literal positions in
    /// `Sphere` primitives. The emitted `rye_scene_sdf` calls only
    /// `rye_distance`, so the spatial metric is fully Space-aware at runtime.
    pub fn to_scene<S>(&self, space: &S) -> Scene
    where
        S: Space<Point = Vec3, Vector = Vec3> + WgslSpace,
    {
        let axes = [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ];
        let mut node = SceneNode::sphere(Vec3::ZERO, self.sphere_radius);
        for axis in axes {
            for k in 1..=self.steps {
                let tangent = axis * (k as f32 * self.geodesic_spacing);
                node = node.union(SceneNode::sphere(
                    space.exp(Vec3::ZERO, tangent),
                    self.sphere_radius,
                ));
            }
        }
        Scene::new(node)
    }

    pub fn to_wgsl<S>(&self, space: &S) -> String
    where
        S: Space<Point = Vec3, Vector = Vec3> + WgslSpace,
    {
        self.to_scene(space).to_wgsl(space)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Primitive trait tests -------------------------------------------

    #[test]
    fn sphere_emits_rye_distance_call() {
        use rye_math::EuclideanR3;
        let s = Shape::sphere_at(Vec3::ZERO, 0.25);
        let src = s.to_wgsl(&EuclideanR3, "sdf_0");
        assert!(src.contains("fn sdf_0(p: vec3<f32>) -> f32"));
        assert!(src.contains("rye_distance"));
        assert!(src.contains("0.250000"));
    }

    #[test]
    fn sphere_wgsl_is_space_agnostic() {
        use rye_math::{EuclideanR3, HyperbolicH3, SphericalS3};
        let s = Shape::sphere_at_origin(0.3);
        let e3 = s.to_wgsl(&EuclideanR3, "sdf_0");
        let h3 = s.to_wgsl(&HyperbolicH3, "sdf_0");
        let s3 = s.to_wgsl(&SphericalS3, "sdf_0");
        // The emitted body must be identical across spaces, only
        // rye_distance differs at prelude link time, not in the
        // emitted text.
        assert_eq!(e3, h3);
        assert_eq!(h3, s3);
    }

    /// `HalfSpace` no longer emits a chart-coordinate `dot(p, n) -
    /// d`: the trait rule forbids raw coordinate arithmetic, and the
    /// chart-coord form rendered visibly wrong floors in H³ / S³.
    /// Until a closed-form geodesic-plane SDF lands, the variant
    /// emits the `+1e9` invisible-far-away sentinel. Pinned here so
    /// a future regression that re-enables raw `dot()` fails loud.
    #[test]
    fn halfspace_emits_sentinel_sdf() {
        use rye_math::EuclideanR3;
        let p = Shape::HalfSpace {
            normal: Vec3::Y,
            offset: -0.5,
        };
        let src = p.to_wgsl(&EuclideanR3, "sdf_floor");
        assert!(src.contains("fn sdf_floor(_p: vec3<f32>) -> f32"));
        assert!(src.contains("return 1e9"));
        assert!(
            !src.contains("dot(p,"),
            "HalfSpace must not emit raw chart-coord dot product",
        );
    }

    #[test]
    fn box_emits_euclidean_box_sdf() {
        use rye_math::EuclideanR3;
        let b = Shape::Box3 {
            half_extents: Vec3::splat(0.4),
        };
        let src = b.to_wgsl(&EuclideanR3, "sdf_box");
        assert!(src.contains("fn sdf_box(p: vec3<f32>) -> f32"));
        assert!(src.contains("abs(p)"));
        assert!(src.contains("0.400000"));
    }

    #[test]
    fn combinator_union_expr() {
        use combinator::union_expr;
        let expr = union_expr("da", "db");
        assert_eq!(expr, "min(da, db)");
    }

    #[test]
    fn combinator_smooth_min_fn_compiles() {
        use combinator::smooth_min_fn;
        let src = smooth_min_fn("smin", 0.08);
        assert!(src.contains("fn smin(a: f32, b: f32) -> f32"));
        assert!(src.contains("0.080000"));
        assert!(src.contains("clamp"));
        assert!(src.contains("mix"));
    }

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

    /// Slab planes (floor + ceiling) currently sentinel through
    /// `Primitive::HalfSpace`, so the floor / ceiling literals
    /// no longer appear in the emitted WGSL. The slab still
    /// participates in the typed scene tree (a future
    /// geodesic-plane SDF will re-enable the rendering); the
    /// scene assembly just doesn't carry the chart-coord values
    /// through any longer.
    #[test]
    fn slab_scene_emits_sentinel_for_planes_without_floor_constants() {
        let src = GeodesicSpheresScene::default()
            .with_slab(-0.5, 0.8)
            .to_wgsl();
        assert!(src.contains("fn rye_scene_sdf"));
        assert!(src.contains("rye_distance")); // spheres still emit honestly
        assert!(
            !src.contains("-0.500000"),
            "floor_y must not leak as a chart-coord literal while HalfSpace sentinels",
        );
        assert!(
            src.contains("return 1e9"),
            "slab planes should resolve to the +1e9 sentinel SDF",
        );
    }

    #[test]
    fn lattice_scene_emits_required_entrypoint() {
        use rye_math::EuclideanR3;
        let src = LatticeSphereScene::default().to_wgsl(&EuclideanR3);
        assert!(src.contains("fn rye_scene_sdf"));
        assert!(src.contains("rye_distance"));
        // Sphere radius is now inlined per-sphere, not a named constant.
        assert!(src.contains("0.120000")); // default sphere_radius
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

    /// Corridor walls currently sentinel through
    /// `Primitive::HalfSpace`; only the pillar spheres emit
    /// honest geometry. The half-width / half-height literals
    /// no longer surface in the WGSL because they were used only
    /// to construct the wall plane normals.
    #[test]
    fn corridor_scene_emits_required_entrypoint_with_sentinel_walls() {
        let src = corridor_demo_wgsl();
        assert!(src.contains("fn rye_scene_sdf"));
        assert!(src.contains("rye_distance")); // pillars still honest
        assert!(
            src.contains("return 1e9"),
            "corridor walls should resolve to the +1e9 sentinel SDF",
        );
        assert!(
            !src.contains("0.550000"),
            "half_width must not leak as a chart-coord literal while HalfSpace sentinels",
        );
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
