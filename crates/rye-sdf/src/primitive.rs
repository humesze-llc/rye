use glam::Vec3;
use rye_math::WgslSpace;

/// A geometric primitive that emits its signed-distance function as WGSL.
///
/// The emitted function has signature `fn {name}(p: vec3<f32>) -> f32` and
/// must call only `rye_*` functions from the Space prelude — never raw
/// coordinate arithmetic. This ensures correctness across E³, H³, and S³.
///
/// # Object safety
///
/// This trait is intentionally not object-safe (generic method). The Scene
/// tree composes primitives via the concrete [`PrimitiveKind`] enum rather
/// than `dyn Primitive`.
pub trait Primitive {
    /// Emit a WGSL function named `name` that returns the signed distance
    /// from `p` to this primitive in the given Space.
    fn to_wgsl<S: WgslSpace>(&self, space: &S, name: &str) -> String;
}

/// A geodesic sphere: the locus of points at geodesic distance `radius`
/// from `center`.
///
/// The SDF is `rye_distance(p, center) - radius` in every Space.
/// No per-space specialization is needed because `rye_distance` resolves
/// to the correct metric at shader link time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

impl Sphere {
    pub fn new(center: Vec3, radius: f32) -> Self {
        Self { center, radius }
    }

    pub fn at_origin(radius: f32) -> Self {
        Self {
            center: Vec3::ZERO,
            radius,
        }
    }
}

impl Primitive for Sphere {
    fn to_wgsl<S: WgslSpace>(&self, _space: &S, name: &str) -> String {
        format!(
            "fn {name}(p: vec3<f32>) -> f32 {{\n\
             \treturn rye_distance(p, vec3<f32>({cx:.6}, {cy:.6}, {cz:.6})) - {r:.6};\n\
             }}\n",
            name = name,
            cx = self.center.x,
            cy = self.center.y,
            cz = self.center.z,
            r = self.radius,
        )
    }
}

/// A totally-geodesic plane.
///
/// In E³: the half-space `dot(p, normal) - offset`, where `normal` is a
/// unit vector and `offset` is the signed distance from the origin.
///
/// In H³ (Poincaré ball): a geodesic plane is a Euclidean sphere or plane
/// orthogonal to the boundary sphere. Only planes through the origin
/// (geodesic planes that are Euclidean flat) have a clean closed form
/// (`dot(p, normal)`); off-origin geodesic planes require the
/// artanh-of-Möbius formulation and will be added in a follow-on step.
///
/// In S³ (upper hemisphere): a geodesic plane is a great 2-sphere; the SDF
/// is chord distance to the intersection hyperplane.
///
/// For the initial implementation all spaces emit the same Euclidean dot
/// formula, which is exact in E³ and gives the Euclidean-coordinate
/// approximation in H³/S³. The space-correct formulas are tracked in the
/// Phase 2 plan and will replace this once the `SpaceKind` dispatch pattern
/// is established.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    /// Outward unit normal (Euclidean).
    pub normal: Vec3,
    /// Signed distance of the plane from the origin along the normal.
    pub offset: f32,
}

impl Plane {
    pub fn new(normal: Vec3, offset: f32) -> Self {
        Self { normal, offset }
    }

    /// Horizontal floor at y = `y`.
    pub fn floor(y: f32) -> Self {
        Self {
            normal: Vec3::Y,
            offset: y,
        }
    }

    /// Horizontal ceiling at y = `y`.
    pub fn ceiling(y: f32) -> Self {
        Self {
            normal: Vec3::NEG_Y,
            offset: -y,
        }
    }
}

impl Primitive for Plane {
    fn to_wgsl<S: WgslSpace>(&self, _space: &S, name: &str) -> String {
        format!(
            "fn {name}(p: vec3<f32>) -> f32 {{\n\
             \treturn dot(p, vec3<f32>({nx:.6}, {ny:.6}, {nz:.6})) - {d:.6};\n\
             }}\n",
            name = name,
            nx = self.normal.x,
            ny = self.normal.y,
            nz = self.normal.z,
            d = self.offset,
        )
    }
}

/// An axis-aligned box centered at the origin.
///
/// Uses the standard Euclidean box SDF (`q = abs(p) - half_extents`,
/// `length(max(q, 0)) + min(max(q.x, q.y, q.z), 0)`). The box geometry is
/// defined in Space coordinates, so in H³/S³ the corners appear
/// compressed/expanded according to the metric — the same way Euclidean-coord
/// walls in the corridor demo appear to bow.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Box {
    pub half_extents: Vec3,
}

impl Box {
    pub fn new(half_extents: Vec3) -> Self {
        Self { half_extents }
    }

    pub fn cube(half_side: f32) -> Self {
        Self {
            half_extents: Vec3::splat(half_side),
        }
    }
}

impl Primitive for Box {
    fn to_wgsl<S: WgslSpace>(&self, _space: &S, name: &str) -> String {
        format!(
            "fn {name}(p: vec3<f32>) -> f32 {{\n\
             \tlet b = vec3<f32>({hx:.6}, {hy:.6}, {hz:.6});\n\
             \tlet q = abs(p) - b;\n\
             \treturn length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);\n\
             }}\n",
            name = name,
            hx = self.half_extents.x,
            hy = self.half_extents.y,
            hz = self.half_extents.z,
        )
    }
}
