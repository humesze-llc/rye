//! Shader database with hot-reload support.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use loam_asset::{AssetEvent, AssetEventKind};
use loam_math::WgslSpace;
use wgpu::{Device, ShaderModule, ShaderModuleDescriptor, ShaderSource};

/// WGSL parse or validation failure.
#[derive(Debug, thiserror::Error)]
pub enum WgslValidationError {
    #[error("WGSL parse error: {0}")]
    Parse(#[from] naga::front::wgsl::ParseError),
    #[error("WGSL validation error: {0}")]
    Validate(Box<naga::WithSpan<naga::valid::ValidationError>>),
}

/// Opaque handle to a shader in a [`ShaderDb`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShaderId(u32);

/// Identifies who loaded a shader. The Space prelude is a property of the
/// loader, not of the file, so hot-reload is scoped by owner: recompiling
/// another owner's module against this one's Space would silently retune its
/// metric.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShaderOwner(u32);

struct Entry {
    path: PathBuf,
    module: ShaderModule,
    scene_source: Option<String>,
    /// Bumped on every successful (re)compile; render code rebuilds its pipeline
    /// on a generation mismatch.
    generation: u64,
    /// Debug label for the module; reused on recompile.
    label: String,
}

/// Cache of compiled shaders, invalidated on asset events. A failed hot-reload
/// (or removed file) keeps the last good module rather than crashing; a later
/// create/modify restores it.
pub struct ShaderDb {
    device: Device,
    entries: HashMap<ShaderId, Entry>,
    /// Path index per owner rather than one shared map: two owners loading the
    /// same file need two entries, since a module carries exactly one prelude.
    path_index: HashMap<ShaderOwner, HashMap<PathBuf, ShaderId>>,
    next_id: u32,
    next_owner: u32,
}

impl ShaderDb {
    /// The owner every db starts with, for a host that loads all its shaders
    /// against one Space. Sub-scenes with Spaces of their own take an owner
    /// from [`ShaderDb::new_owner`] instead.
    pub const ROOT_OWNER: ShaderOwner = ShaderOwner(0);

    /// Construct. `device` is cloned on recompile (wgpu's Device is refcounted).
    pub fn new(device: Device) -> Self {
        Self {
            device,
            entries: HashMap::new(),
            path_index: HashMap::new(),
            next_id: 0,
            next_owner: Self::ROOT_OWNER.0 + 1,
        }
    }

    /// Mint an owner distinct from [`ShaderDb::ROOT_OWNER`] and from every other
    /// owner this db has issued.
    pub fn new_owner(&mut self) -> ShaderOwner {
        let owner = ShaderOwner(self.next_owner);
        self.next_owner += 1;
        owner
    }

    /// Load a shader from disk, prepending the Space's WGSL prelude. The returned
    /// [`ShaderId`] is stable across reloads; the same `owner` loading the same
    /// path again yields the same ID and a fresh compilation, while a different
    /// owner loading it gets its own ID and its own prelude.
    pub fn load<S: WgslSpace>(
        &mut self,
        owner: ShaderOwner,
        path: impl AsRef<Path>,
        space: &S,
    ) -> Result<ShaderId> {
        self.load_inner(owner, path, None, space)
    }

    /// Load a shader plus a scene module; the scene source is stored and reused on
    /// reloads of the shader file.
    pub fn load_with_scene<S: WgslSpace>(
        &mut self,
        owner: ShaderOwner,
        path: impl AsRef<Path>,
        scene_source: &str,
        space: &S,
    ) -> Result<ShaderId> {
        self.load_inner(owner, path, Some(scene_source), space)
    }

    /// Load a shader for geodesic ray marching: assembles Space prelude + scene
    /// SDF + geodesic march kernel ([`crate::GEODESIC_MARCH_KERNEL`], which defines
    /// `loam_march_geodesic` / `loam_estimate_normal` / `loam_safe_normalize`) + user
    /// shading. The scene + kernel is stored and reused on reloads.
    pub fn load_geodesic_scene<S: WgslSpace>(
        &mut self,
        owner: ShaderOwner,
        path: impl AsRef<Path>,
        scene_source: &str,
        space: &S,
    ) -> Result<ShaderId> {
        let scene_with_kernel = format!(
            "{scene_source}// ---- loam geodesic march kernel ----\n{}",
            crate::GEODESIC_MARCH_KERNEL
        );
        self.load_inner(owner, path, Some(&scene_with_kernel), space)
    }

    fn load_inner<S: WgslSpace>(
        &mut self,
        owner: ShaderOwner,
        path: impl AsRef<Path>,
        scene_source: Option<&str>,
        space: &S,
    ) -> Result<ShaderId> {
        let path = canonicalize(path.as_ref())?;
        let source = std::fs::read_to_string(&path)
            .with_context(|| format!("reading shader {}", path.display()))?;
        let module = self.compile(&path, &source, scene_source, space)?;

        let label = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());

        if let Some(id) = self.lookup(owner, &path) {
            let entry = self.entries.get_mut(&id).expect("path_index out of sync");
            entry.module = module;
            entry.scene_source = scene_source.map(str::to_owned);
            entry.generation += 1;
            entry.label = label;
            Ok(id)
        } else {
            let id = ShaderId(self.next_id);
            self.next_id += 1;
            self.path_index
                .entry(owner)
                .or_default()
                .insert(path.clone(), id);
            self.entries.insert(
                id,
                Entry {
                    path,
                    module,
                    scene_source: scene_source.map(str::to_owned),
                    generation: 1,
                    label,
                },
            );
            Ok(id)
        }
    }

    fn lookup(&self, owner: ShaderOwner, path: &Path) -> Option<ShaderId> {
        self.path_index.get(&owner)?.get(path).copied()
    }

    /// Borrow the current compiled module for `id`.
    pub fn module(&self, id: ShaderId) -> &ShaderModule {
        &self
            .entries
            .get(&id)
            .expect("unknown ShaderId - was it loaded by this ShaderDb?")
            .module
    }

    /// Generation counter for `id`; bumps on every successful (re)compile so
    /// render code can rebuild its pipeline on mismatch.
    pub fn generation(&self, id: ShaderId) -> u64 {
        self.entries.get(&id).map(|e| e.generation).unwrap_or(0)
    }

    /// Apply filesystem events to the shaders `owner` loaded, recompiling them
    /// against `owner`'s `space`. Entries under any other owner are untouched,
    /// including entries for the same path. Compile errors are logged but keep
    /// the last good module; rendering continues until fixed.
    pub fn apply_events<S: WgslSpace>(
        &mut self,
        owner: ShaderOwner,
        events: &[AssetEvent],
        space: &S,
    ) {
        for event in events {
            let canonical = match canonicalize(&event.path) {
                Ok(p) => p,
                // Removed files can't be canonicalized; look up by raw path.
                Err(_) => event.path.clone(),
            };
            let Some(id) = self.lookup(owner, &canonical) else {
                continue;
            };
            match event.kind {
                AssetEventKind::Created | AssetEventKind::Modified => {
                    if let Err(e) = self.reload(id, space) {
                        tracing::warn!("shader reload failed for {}: {e:#}", canonical.display());
                    } else {
                        tracing::info!("reloaded shader {}", canonical.display());
                    }
                }
                AssetEventKind::Removed => {
                    tracing::warn!(
                        "shader file removed; keeping stale module: {}",
                        canonical.display()
                    );
                }
            }
        }
    }

    fn reload<S: WgslSpace>(&mut self, id: ShaderId, space: &S) -> Result<()> {
        let path = self.entries[&id].path.clone();
        let scene_source = self.entries[&id].scene_source.clone();
        let source = std::fs::read_to_string(&path)
            .with_context(|| format!("reading shader {}", path.display()))?;
        let module = self.compile(&path, &source, scene_source.as_deref(), space)?;
        let entry = self.entries.get_mut(&id).expect("id just looked up");
        entry.module = module;
        entry.generation += 1;
        Ok(())
    }

    fn compile<S: WgslSpace>(
        &self,
        path: &Path,
        user_source: &str,
        scene_source: Option<&str>,
        space: &S,
    ) -> Result<ShaderModule> {
        let full = assemble_source_with_scene(&space.wgsl_impl(), scene_source, user_source);
        validate_wgsl(&full).with_context(|| format!("validating shader {}", path.display()))?;
        let label = path.file_name().and_then(|n| n.to_str());
        Ok(self.device.create_shader_module(ShaderModuleDescriptor {
            label,
            source: ShaderSource::Wgsl(full.into()),
        }))
    }
}

fn canonicalize(path: &Path) -> Result<PathBuf> {
    path.canonicalize()
        .with_context(|| format!("canonicalizing {}", path.display()))
}

/// Concatenate the Space's WGSL prelude with the user shader source. Extracted for
/// testability; the Device-free part of the hot-reload path.
#[cfg(test)]
pub(crate) fn assemble_source(space_wgsl: &str, user_source: &str) -> String {
    assemble_source_with_scene(space_wgsl, None, user_source)
}

pub(crate) fn assemble_source_with_scene(
    space_wgsl: &str,
    scene_wgsl: Option<&str>,
    user_source: &str,
) -> String {
    let scene_len = scene_wgsl.map(str::len).unwrap_or(0);
    let mut out = String::with_capacity(space_wgsl.len() + scene_len + user_source.len() + 96);
    out.push_str("// ---- loam-math Space prelude ----\n");
    out.push_str(space_wgsl);
    if !space_wgsl.ends_with('\n') {
        out.push('\n');
    }
    if let Some(scene_wgsl) = scene_wgsl {
        out.push_str("// ---- loam-scene scene module ----\n");
        out.push_str(scene_wgsl);
        if !scene_wgsl.ends_with('\n') {
            out.push('\n');
        }
    }
    out.push_str("// ---- user shader ----\n");
    out.push_str(user_source);
    out
}

/// Parse and validate a complete WGSL module. Headless: rejects a broken
/// [`loam_math::WgslSpace`] prelude or user shader without a GPU adapter. `wgpu`
/// still does its own backend validation at module creation.
pub fn validate_wgsl(source: &str) -> std::result::Result<(), WgslValidationError> {
    let module = naga::front::wgsl::parse_str(source)?;
    let flags = naga::valid::ValidationFlags::all();
    let caps = naga::valid::Capabilities::empty();
    naga::valid::Validator::new(flags, caps)
        .validate(&module)
        .map_err(|e| WgslValidationError::Validate(Box::new(e)))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::{Pod, Zeroable};
    use glam::{Vec3, Vec4};
    use loam_math::{
        BlendedSpace, ConformallyFlat, EuclideanR3, EuclideanR4, HyperbolicH3, LinearBlendX, Space,
        SphericalS3, WgslSpace,
    };

    const ABI_PROBE: &str = r#"
@compute @workgroup_size(1)
fn main() {
    let a = vec3<f32>(0.1, 0.2, 0.3);
    let b = vec3<f32>(0.2, -0.1, 0.05);
    let v = vec3<f32>(0.01, 0.02, -0.03);
    _ = loam_distance(a, b);
    _ = loam_origin_distance(a);
    _ = loam_exp(a, v);
    _ = loam_log(a, b);
    _ = loam_parallel_transport(a, b, v);
    _ = LOAM_MAX_ARC;
}
"#;

    // 4D variant of `ABI_PROBE`; `EuclideanR4` uses `vec4<f32>` throughout, v0 ABI
    // otherwise identical.
    const ABI_PROBE_VEC4: &str = r#"
@compute @workgroup_size(1)
fn main() {
    let a = vec4<f32>(0.1, 0.2, 0.3, 0.0);
    let b = vec4<f32>(0.2, -0.1, 0.05, 0.4);
    let v = vec4<f32>(0.01, 0.02, -0.03, 0.05);
    _ = loam_distance(a, b);
    _ = loam_origin_distance(a);
    _ = loam_exp(a, v);
    _ = loam_log(a, b);
    _ = loam_parallel_transport(a, b, v);
    _ = LOAM_MAX_ARC;
}
"#;

    #[test]
    fn assemble_includes_both_sources() {
        let s = assemble_source(
            "fn loam_distance(a: f32, b: f32) -> f32 { return 0.0; }",
            "@fragment fn main() {}",
        );
        assert!(s.contains("loam_distance"));
        assert!(s.contains("@fragment fn main"));
        assert!(s.find("loam_distance").unwrap() < s.find("@fragment fn main").unwrap());
    }

    #[test]
    fn assemble_adds_newline_between_sources() {
        let s = assemble_source("fn a() {}", "fn b() {}");
        // prelude line then newline then user section marker then user code.
        assert!(s.contains("fn a() {}\n// ---- user shader ----"));
    }

    #[test]
    fn assemble_handles_trailing_newline_in_prelude() {
        let s = assemble_source("fn a() {}\n", "fn b() {}");
        // No double newline from our side; the input one is sufficient.
        assert!(!s.contains("fn a() {}\n\n// ---- user shader ----"));
        assert!(s.contains("fn a() {}\n// ---- user shader ----"));
    }

    #[test]
    fn assemble_includes_scene_between_space_and_user() {
        let s = assemble_source_with_scene("fn space() {}", Some("fn scene() {}"), "fn user() {}");
        let i_space = s.find("fn space() {}").expect("space chunk present");
        let i_scene = s.find("fn scene() {}").expect("scene chunk present");
        let i_user = s.find("fn user() {}").expect("user chunk present");
        assert!(i_space < i_scene && i_scene < i_user);
    }

    #[test]
    fn euclidean_space_prelude_validates_against_abi_probe() {
        let src = assemble_source(&EuclideanR3.wgsl_impl(), ABI_PROBE);
        validate_wgsl(&src).expect("EuclideanR3 WGSL prelude should validate");
    }

    #[test]
    fn hyperbolic_space_prelude_validates_against_abi_probe() {
        let src = assemble_source(&HyperbolicH3.wgsl_impl(), ABI_PROBE);
        validate_wgsl(&src).expect("HyperbolicH3 WGSL prelude should validate");
    }

    #[test]
    fn spherical_space_prelude_validates_against_abi_probe() {
        let src = assemble_source(&SphericalS3.wgsl_impl(), ABI_PROBE);
        validate_wgsl(&src).expect("SphericalS3 WGSL prelude should validate");
    }

    /// `EuclideanR4`'s prelude is the v0 ABI in `vec4<f32>`. No render node
    /// consumes it yet (4D ships via hyperslice, not native geodesic march), but
    /// the flat-space ℝ⁴ math is honest, so naga validation pins the contract.
    #[test]
    fn euclidean_r4_space_prelude_validates_against_abi_probe() {
        let src = assemble_source(&EuclideanR4.wgsl_impl(), ABI_PROBE_VEC4);
        validate_wgsl(&src).expect("EuclideanR4 WGSL prelude should validate");
    }

    // Minimal stub of loam_scene_sdf so the kernel has something to call.
    const KERNEL_SCENE: &str = r#"
fn loam_scene_sdf(p: vec3<f32>) -> f32 {
    return loam_distance(p, vec3<f32>(0.0, 0.0, 0.0)) - 0.25;
}
"#;

    // Compute probe that exercises all three kernel entry points.
    const KERNEL_PROBE: &str = r#"
@compute @workgroup_size(1)
fn main() {
    let ro = vec3<f32>(0.0, 0.0, 2.0);
    let rd = vec3<f32>(0.0, 0.0, -1.0);
    _ = loam_march_geodesic(ro, rd, 0.2);
    _ = loam_estimate_normal(vec3<f32>(0.0, 0.0, 0.0), 0.2);
    _ = loam_safe_normalize(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0));
}
"#;

    fn assemble_geodesic_probe(space_wgsl: &str) -> String {
        assemble_source_with_scene(
            space_wgsl,
            Some(&format!("{KERNEL_SCENE}{}", crate::GEODESIC_MARCH_KERNEL)),
            KERNEL_PROBE,
        )
    }

    #[test]
    fn euclidean_geodesic_kernel_validates() {
        let src = assemble_geodesic_probe(&EuclideanR3.wgsl_impl());
        validate_wgsl(&src).expect("EuclideanR3 + geodesic kernel should validate");
    }

    #[test]
    fn hyperbolic_geodesic_kernel_validates() {
        let src = assemble_geodesic_probe(&HyperbolicH3.wgsl_impl());
        validate_wgsl(&src).expect("HyperbolicH3 + geodesic kernel should validate");
    }

    #[test]
    fn spherical_geodesic_kernel_validates() {
        let src = assemble_geodesic_probe(&SphericalS3.wgsl_impl());
        validate_wgsl(&src).expect("SphericalS3 + geodesic kernel should validate");
    }

    /// The boundary escape splits its numbers across two crates: `LOAM_MAX_ARC`
    /// and `LOAM_S3_R2_MAX` are loam-math prelude data, `0.92` is kernel policy
    /// buffering the S3 chart's saturating `loam_origin_distance`. The escape
    /// can only fire if their product stays under the largest arc that chart
    /// reports, `asin(√LOAM_S3_R2_MAX)`; above it a ray leaves the domain and
    /// only the arc budget terminates the march. Every literal below is read
    /// back out of one of the three pinned strings, so neither crate can move
    /// its number without failing here.
    #[test]
    fn spherical_boundary_escape_fires_below_the_saturated_chart_radius() {
        let prelude = SphericalS3.wgsl_impl();
        assert!(prelude.contains("const LOAM_MAX_ARC: f32 = 1.5;"));
        assert!(prelude.contains("const LOAM_S3_R2_MAX: f32 = 0.999999;"));
        assert!(
            crate::GEODESIC_MARCH_KERNEL.contains("loam_origin_distance(p) > LOAM_MAX_ARC * 0.92")
        );
        assert!(0.92_f32 * 1.5 < 0.999999_f32.sqrt().asin());
    }

    #[test]
    fn blended_e3_h3_prelude_validates_against_abi_probe() {
        let bs = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-2.0, 2.0));
        let src = assemble_source(&bs.wgsl_impl(), ABI_PROBE);
        validate_wgsl(&src).expect("BlendedSpace<E3,H3,LinearBlendX> WGSL prelude should validate");
    }

    #[test]
    fn blended_e3_h3_geodesic_kernel_validates() {
        let bs = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-2.0, 2.0));
        let src = assemble_geodesic_probe(&bs.wgsl_impl());
        validate_wgsl(&src)
            .expect("BlendedSpace<E3,H3,LinearBlendX> + geodesic kernel should validate");
    }

    // `loam_origin_distance` as the Space preludes ship it, verbatim. The S3
    // saturation constant is pinned next to the body because the body only names
    // it: the two live on different lines of the prelude and could otherwise
    // drift apart at the shell without either pin noticing.
    const R3_ORIGIN_DISTANCE_FN: &str =
        "fn loam_origin_distance(p: vec3<f32>) -> f32 { return length(p); }";
    const S3_R2_MAX_DECL: &str = "const LOAM_S3_R2_MAX: f32 = 0.999999;";
    const S3_ORIGIN_DISTANCE_BODY: &str =
        "    let r2 = min(dot(p, p), LOAM_S3_R2_MAX);\n    return asin(sqrt(r2));\n}";
    const S3_R2_MAX: f32 = 0.999999;

    /// CPU port of `EuclideanR3`'s shipped `loam_origin_distance`.
    ///
    /// The text pin lives in the constructor rather than in a standalone test so
    /// that a prelude which moves out from under the port fails the `cpu_march_*`
    /// tests themselves; a port that can silently stop mirroring the shader is
    /// the defect this indirection exists to prevent.
    fn euclidean_origin_distance_mirror() -> impl Fn(Vec3) -> f32 {
        assert!(
            EuclideanR3.wgsl_impl().contains(R3_ORIGIN_DISTANCE_FN),
            "EuclideanR3 loam_origin_distance drifted from its CPU port",
        );
        |p: Vec3| p.length()
    }

    /// CPU port of `SphericalS3`'s shipped `loam_origin_distance`, expression for
    /// expression. See [`euclidean_origin_distance_mirror`] for why the pin runs
    /// here.
    fn spherical_origin_distance_mirror() -> impl Fn(Vec3) -> f32 {
        let prelude = SphericalS3.wgsl_impl();
        assert!(
            prelude.contains(S3_R2_MAX_DECL) && prelude.contains(S3_ORIGIN_DISTANCE_BODY),
            "SphericalS3 loam_origin_distance drifted from its CPU port",
        );
        |p: Vec3| {
            let r2 = p.length_squared().min(S3_R2_MAX);
            r2.sqrt().asin()
        }
    }

    // CPU port of `kernel.wgsl::loam_march_geodesic`, mirrored line-for-line, so
    // the `cpu_march_*` tests can check hit points against a known SDF without a
    // GPU adapter. `loam_origin_distance` and `loam_max_arc` are parameters here
    // (the kernel reads both from the Space prelude) so each test supplies the
    // pinned mirror of the one and the value it exercises for the other.
    fn march_geodesic_cpu<S: Space<Point = Vec3, Vector = Vec3>>(
        space: &S,
        sdf: impl Fn(Vec3) -> f32,
        origin_distance: impl Fn(Vec3) -> f32,
        ro: Vec3,
        rd: Vec3,
        ball_scale: f32,
        loam_max_arc: f32,
    ) -> Option<(Vec3, f32)> {
        let scale = ball_scale.max(1e-5);
        let mut p = ro * scale;

        let rd_unit = rd.try_normalize().unwrap_or(Vec3::new(0.0, 0.0, -1.0));
        let probe_eps = 1e-4_f32;
        let probed = space.exp(p, rd_unit * probe_eps);
        let riem_norm = space.distance(p, probed) / probe_eps;
        let mut v = rd_unit / riem_norm.max(1e-7);

        let mut t_scene = 0.0_f32;
        let mut t_arc = 0.0_f32;
        let hit_eps = 0.001 * scale;
        let min_step = 0.0001 * scale;

        for _ in 0..256 {
            if origin_distance(p) > loam_max_arc * 0.92 {
                return None;
            }
            let d = sdf(p);
            if d < hit_eps {
                return Some((p, t_scene));
            }
            if t_scene > 40.0 || t_arc > loam_max_arc {
                return None;
            }
            let step = (d * 0.85).max(min_step);
            let next_p = space.exp(p, v * step);
            let next_v = space.parallel_transport(p, next_p, v);
            p = next_p;
            if next_v.length_squared() > 1e-12 {
                v = next_v;
            }
            t_scene += step / scale;
            t_arc += step;
        }
        None
    }

    /// Sphere-trace a unit ray against an origin-centered sphere in EuclideanR3;
    /// the hit should land on the surface within `hit_eps` and `t_scene` equal the
    /// camera-space distance to it.
    #[test]
    fn cpu_march_hits_centered_sphere_in_euclidean_r3() {
        let space = EuclideanR3;
        let sphere_radius = 0.5_f32;
        let sdf = |p: Vec3| p.length() - sphere_radius;
        let ro = Vec3::new(0.0, 0.0, 2.0);
        let rd = Vec3::new(0.0, 0.0, -1.0);
        let (hit, t) = march_geodesic_cpu(
            &space,
            sdf,
            euclidean_origin_distance_mirror(),
            ro,
            rd,
            1.0,
            1.0e9,
        )
        .expect("ray should hit centered sphere");

        // Front of the sphere along -Z: (0, 0, 0.5).
        let expected = Vec3::new(0.0, 0.0, sphere_radius);
        let position_drift = (hit - expected).length();
        assert!(
            position_drift < 5e-3,
            "hit {hit:?} should be within hit_eps of expected {expected:?} (drift {position_drift})",
        );

        // t_scene = ro.z - radius = 1.5; tolerance covers last-step overshoot
        // (up to one hit_eps) plus float-stepping noise.
        let expected_t = 1.5_f32;
        assert!(
            (t - expected_t).abs() < 5e-3,
            "t_scene {t} should be ~{expected_t} (within last-step overshoot)",
        );
    }

    /// A ray pointing away from the only object must miss (exit on `t_scene > 40`
    /// or the 256-iteration cap).
    #[test]
    fn cpu_march_misses_when_ray_points_away_in_euclidean_r3() {
        let space = EuclideanR3;
        let sdf = |p: Vec3| p.length() - 0.5;
        let ro = Vec3::new(0.0, 0.0, 2.0);
        let rd = Vec3::new(0.0, 0.0, 1.0); // away from sphere
        let result = march_geodesic_cpu(
            &space,
            sdf,
            euclidean_origin_distance_mirror(),
            ro,
            rd,
            1.0,
            1.0e9,
        );
        assert!(
            result.is_none(),
            "ray pointing away from sphere should miss; got {result:?}",
        );
    }

    /// The kernel tests the boundary escape before it samples the scene, so
    /// against an everywhere-solid SDF the arc budget alone decides hit vs miss
    /// and the march reports exactly what `loam_origin_distance` told it.
    ///
    /// |p| = 1e-4 sits below the ≈1.73e-4 radius where `acos(√(1−|p|²))`
    /// collapses to exactly 0 in f32, so the escape is pinned in the regime
    /// where an ill-conditioned origin distance reads as "at the origin" and the
    /// boundary silently never fires.
    #[test]
    fn cpu_march_arc_escape_tracks_origin_distance_near_the_s3_origin() {
        let solid = |_: Vec3| -1.0_f32;
        let ro = Vec3::new(0.0, 0.0, 1e-4);
        let rd = Vec3::new(0.0, 0.0, -1.0);

        let escaped = march_geodesic_cpu(
            &SphericalS3,
            solid,
            spherical_origin_distance_mirror(),
            ro,
            rd,
            1.0,
            1e-4,
        );
        assert!(
            escaped.is_none(),
            "|p| = 1e-4 is past a 1e-4 arc budget (0.92 buffer); \
             the march must escape before sampling, got {escaped:?}",
        );

        let inside = march_geodesic_cpu(
            &SphericalS3,
            solid,
            spherical_origin_distance_mirror(),
            ro,
            rd,
            1.0,
            1.0,
        );
        assert!(
            inside.is_some(),
            "|p| = 1e-4 is well inside a 1.0 arc budget; the march must sample the scene",
        );
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, Pod, Zeroable)]
    struct GpuCase {
        a: [f32; 4],
        b: [f32; 4],
        v: [f32; 4],
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, Pod, Zeroable)]
    struct GpuOut {
        /// `loam_distance(a, b)`, then `loam_origin_distance` of `a`, of `b`.
        scalars: [f32; 4],
        exp_point: [f32; 4],
        log_vec: [f32; 4],
        transported: [f32; 4],
    }

    const PROBE_IO: &str = r#"
struct Case {
    a: vec4<f32>,
    b: vec4<f32>,
    v: vec4<f32>,
};

struct ProbeOut {
    scalars: vec4<f32>,
    exp_point: vec4<f32>,
    log_vec: vec4<f32>,
    transported: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> cases: array<Case>;
@group(0) @binding(1) var<storage, read_write> out: array<ProbeOut>;
"#;

    const GPU_PROBE: &str = r#"
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let c = cases[i];
    let a = c.a.xyz;
    let b = c.b.xyz;
    let v = c.v.xyz;
    out[i].scalars = vec4<f32>(
        loam_distance(a, b),
        loam_origin_distance(a),
        loam_origin_distance(b),
        0.0);
    out[i].exp_point = vec4<f32>(loam_exp(a, v), 0.0);
    out[i].log_vec = vec4<f32>(loam_log(a, b), 0.0);
    out[i].transported = vec4<f32>(loam_parallel_transport(a, b, v), 0.0);
}
"#;

    // `EuclideanR4` ships the same v0 ABI over `vec4<f32>`, so it needs its own
    // probe body; the buffer layout is unchanged.
    const GPU_PROBE_VEC4: &str = r#"
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let c = cases[i];
    out[i].scalars = vec4<f32>(
        loam_distance(c.a, c.b),
        loam_origin_distance(c.a),
        loam_origin_distance(c.b),
        0.0);
    out[i].exp_point = loam_exp(c.a, c.v);
    out[i].log_vec = loam_log(c.a, c.b);
    out[i].transported = loam_parallel_transport(c.a, c.b, c.v);
}
"#;

    /// One probe triple, labelled by the corner of the chart's domain it sits
    /// in so a failure names the regime instead of an array index.
    struct ParityCase {
        corner: &'static str,
        a: Vec3,
        b: Vec3,
        v: Vec3,
    }

    fn corner(corner: &'static str, a: Vec3, b: Vec3, v: Vec3) -> ParityCase {
        ParityCase { corner, a, b, v }
    }

    /// Corners for the two flat charts: coincident, separated below one ulp of
    /// the coordinates, at 1e-4 radius, generic, and widely separated. Nothing
    /// here is a domain boundary because ℝⁿ has none, so these pin the plumbing
    /// and the exp/log/transport algebra rather than a conditioning class.
    ///
    /// The ℝ⁴ probe reuses these and supplies `w` from [`FLAT_CORNER_W`].
    fn flat_corners() -> Vec<ParityCase> {
        vec![
            corner(
                "coincident at the origin",
                Vec3::ZERO,
                Vec3::ZERO,
                Vec3::new(0.01, 0.02, -0.03),
            ),
            corner(
                "separation below one coordinate ulp",
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(1.0 + 1e-8, 1.0, 1.0),
                Vec3::new(1e-8, 0.0, 0.0),
            ),
            corner(
                "small radius",
                Vec3::new(1e-4, 0.0, 0.0),
                Vec3::new(0.0, -1e-4, 0.0),
                Vec3::new(1e-5, 0.0, 2e-5),
            ),
            corner(
                "generic interior",
                Vec3::new(0.1, 0.2, 0.3),
                Vec3::new(0.5, -0.1, 0.0),
                Vec3::new(0.01, 0.02, -0.03),
            ),
            corner(
                "wide separation, tangent past the unit ball",
                Vec3::new(-2.0, 4.0, 0.5),
                Vec3::new(1.5, 0.25, -3.0),
                Vec3::new(30.0, -12.0, 7.0),
            ),
        ]
    }

    /// Interior corners of the S³ upper-hemisphere chart: the pole, a radius
    /// small enough to expose an ill-conditioned origin distance, and the
    /// near-antipodal pairs whose transport denominator is the smallest the
    /// chart holds. Outside the chart is [`out_of_domain_corners`].
    ///
    /// Radii are literals rather than offsets from the prelude's saturation
    /// constant: a fixture that reads the constant it probes retunes with it
    /// and stops failing. `0.9999` is inside today's shell; a retune that moves
    /// the shell inside it turns this into an out-of-domain fixture, which is
    /// visible to a reader in a way a derived radius would not be.
    fn hemisphere_corners() -> Vec<ParityCase> {
        vec![
            corner(
                "at the pole",
                Vec3::ZERO,
                Vec3::new(0.2, -0.1, 0.05),
                Vec3::new(0.01, 0.02, -0.03),
            ),
            // Below the radius where `acos(sqrt(1 - r2))` collapses to zero in
            // f32; the regime the shipped origin distance was wrong in.
            corner(
                "small radius",
                Vec3::new(1e-4, 0.0, 0.0),
                Vec3::new(0.0, 1e-4, 0.0),
                Vec3::new(1e-5, -2e-5, 0.0),
            ),
            corner(
                "generic interior",
                Vec3::new(0.1, 0.2, 0.3),
                Vec3::new(0.2, -0.1, 0.05),
                Vec3::new(0.01, 0.02, -0.03),
            ),
            // Two points near the equator on opposite sides: arc ≈ π − 0.028,
            // driving the transport denominator `|qf + qt|²/2 = 2w²` to 4e-4,
            // the smallest of any corner here. Deliberately short of the
            // chart's own extreme (π − 0.002, at the saturation shell), so a
            // retune of that shell cannot turn this into a clamp probe.
            corner(
                "near-antipodal across the equator",
                Vec3::new(0.9999, 0.0, 0.0),
                Vec3::new(-0.9999, 0.0, 0.0),
                Vec3::new(0.02, 0.03, -0.01),
            ),
            corner(
                "near-antipodal, oblique",
                Vec3::new(0.7, 0.7, 0.1),
                Vec3::new(-0.7, -0.7, -0.1),
                Vec3::new(0.05, -0.02, 0.03),
            ),
        ]
    }

    /// Interior corners of the H³ Poincaré-ball chart. Same regimes and the
    /// same literal-radius discipline as [`hemisphere_corners`]; near-antipodal
    /// here means opposite sides of the ideal boundary, along a segment whose
    /// conformal factor runs from 4 at its centre to 1.0e4 at each endpoint.
    fn ball_corners() -> Vec<ParityCase> {
        vec![
            corner(
                "at the ball centre",
                Vec3::ZERO,
                Vec3::new(0.2, -0.1, 0.08),
                Vec3::new(0.01, 0.02, -0.015),
            ),
            corner(
                "small radius",
                Vec3::new(1e-4, 0.0, 0.0),
                Vec3::new(0.0, 1e-4, 0.0),
                Vec3::new(1e-5, -2e-5, 0.0),
            ),
            corner(
                "generic interior",
                Vec3::new(0.1, 0.2, 0.05),
                Vec3::new(0.2, -0.1, 0.08),
                Vec3::new(0.01, 0.02, -0.015),
            ),
            corner(
                "near-antipodal across the ideal boundary",
                Vec3::new(0.99, 0.0, 0.0),
                Vec3::new(-0.99, 0.0, 0.0),
                Vec3::new(0.02, 0.03, -0.01),
            ),
            corner(
                "both endpoints at r = 0.9999",
                Vec3::new(0.9999, 0.0, 0.0),
                Vec3::new(0.0, -0.9999, 0.0),
                Vec3::new(0.02, -0.015, 0.01),
            ),
        ]
    }

    /// Out-of-domain probes for both `|p| < 1` charts: the unit boundary
    /// itself, one ulp past it, and radii far outside, crossed with directions
    /// that are axis-aligned, oblique and irrational, and with in-domain,
    /// on-boundary and far-outside partners.
    ///
    /// The radii are literals rather than offsets from either chart's
    /// saturation constant. Both charts clamp onto a shell a few f32 ulps
    /// inside `|p| = 1`; a fixture parameterised on that shell would move with
    /// a retune and stop being out of domain at all.
    fn out_of_domain_corners() -> Vec<ParityCase> {
        let directions = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.6, 0.8, 0.0),
            Vec3::new(0.5773503, 0.5773503, 0.5773503),
            Vec3::new(0.26726124, -0.5345225, 0.8017837),
            Vec3::new(-0.35856858, 0.5976143, -0.71713716),
        ];
        let radii = [1.0_f32, 1.0000001, 1.5, 8.0, 1e4];
        let partners = [
            Vec3::new(0.3, 0.1, 0.0),
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(-2.0, 0.5, 0.25),
        ];
        let tangents = [
            Vec3::new(0.01, 0.02, -0.03),
            Vec3::new(2.0, -1.0, 0.5),
            Vec3::ZERO,
        ];
        let mut cases = Vec::new();
        for (i, direction) in directions.iter().enumerate() {
            for (j, radius) in radii.iter().enumerate() {
                let outside = *direction * *radius;
                let partner = partners[(i + j) % partners.len()];
                let tangent = tangents[(i + j) % tangents.len()];
                cases.push(corner("out of domain, source", outside, partner, tangent));
                cases.push(corner("out of domain, target", partner, outside, tangent));
                cases.push(corner("out of domain, both", outside, -outside, tangent));
            }
        }
        cases
    }

    fn gpu_case(a: Vec3, b: Vec3, v: Vec3) -> GpuCase {
        GpuCase {
            a: a.extend(0.0).to_array(),
            b: b.extend(0.0).to_array(),
            v: v.extend(0.0).to_array(),
        }
    }

    async fn run_gpu_probe<S: WgslSpace>(
        space: &S,
        cases: &[GpuCase],
    ) -> Result<Vec<GpuOut>, String> {
        run_probe_body(space, GPU_PROBE, cases).await
    }

    async fn run_probe_body<S: WgslSpace>(
        space: &S,
        body: &str,
        cases: &[GpuCase],
    ) -> Result<Vec<GpuOut>, String> {
        run_compute_probe(
            &assemble_source(&space.wgsl_impl(), &format!("{PROBE_IO}{body}")),
            "loam-space-gpu-probe",
            cases,
        )
        .await
    }

    /// Flat chart, so the only gap is f32 rounding order between the two
    /// compilers.
    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing Space WGSL"]
    fn euclidean_r3_wgsl_matches_the_rust_space_at_the_domain_corners() {
        assert_prelude_matches_cpu(&EuclideanR3, &flat_corners(), 1e-6);
    }

    /// The `w` component each [`flat_corners`] entry carries in ℝ⁴, as
    /// `[a, b, v]` in that fixture's order. Per-corner rather than one constant
    /// triple because a constant one destroys the two regimes the fixture
    /// exists for: it separates the coincident pair and swamps the sub-ulp one.
    /// Away from those two, `a`, `b` and `v` carry distinct nonzero `w`, so a
    /// dropped or duplicated component still cannot pass.
    const FLAT_CORNER_W: [[f32; 3]; 5] = [
        [0.0, 0.0, 0.125],
        [1.0, 1.0 + 1e-8, 1e-8],
        [1e-4, -1e-4, 2e-5],
        [-0.75, 2.5, 0.125],
        [-6.0, 9.0, 40.0],
    ];

    /// Same chart in `vec4<f32>`. No render node consumes this prelude yet, so
    /// this is the only thing standing between it and a silent divergence.
    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing Space WGSL"]
    fn euclidean_r4_wgsl_matches_the_rust_space_at_the_domain_corners() {
        let space = EuclideanR4;
        let corners = flat_corners();
        assert_eq!(corners.len(), FLAT_CORNER_W.len());
        let cases: Vec<GpuCase> = corners
            .iter()
            .zip(FLAT_CORNER_W)
            .map(|(c, w)| GpuCase {
                a: c.a.extend(w[0]).to_array(),
                b: c.b.extend(w[1]).to_array(),
                v: c.v.extend(w[2]).to_array(),
            })
            .collect();
        let rows = pollster::block_on(run_probe_body(&space, GPU_PROBE_VEC4, &cases))
            .expect("EuclideanR4 GPU probe");
        for ((corner, case), row) in corners.iter().zip(&cases).zip(&rows) {
            let (a, b, v) = (
                Vec4::from_array(case.a),
                Vec4::from_array(case.b),
                Vec4::from_array(case.v),
            );
            let at = |what| format!("{} :: {what}", corner.corner);
            assert_near(&at("distance"), row.scalars[0], space.distance(a, b), 1e-6);
            assert_near(
                &at("origin_distance(a)"),
                row.scalars[1],
                space.distance(Vec4::ZERO, a),
                1e-6,
            );
            assert_near(
                &at("origin_distance(b)"),
                row.scalars[2],
                space.distance(Vec4::ZERO, b),
                1e-6,
            );
            assert_vec_near(&at("exp"), row.exp_point, space.exp(a, v), 1e-6);
            assert_vec_near(&at("log"), row.log_vec, space.log(a, b), 1e-6);
            assert_vec_near(
                &at("parallel_transport"),
                row.transported,
                space.parallel_transport(a, b, v),
                1e-6,
            );
        }
    }

    /// Tolerance 2e-4 relative: `asin` and the chord half-angle are evaluated
    /// by two different compilers' intrinsics, and the near-antipodal corner
    /// divides by the transport denominator `2w²`, which the equator case
    /// drives to 4e-4.
    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing Space WGSL"]
    fn spherical_s3_wgsl_matches_the_rust_space_at_the_domain_corners() {
        assert_prelude_matches_cpu(&SphericalS3, &hemisphere_corners(), 2e-4);
    }

    /// Tolerance 2e-4 relative, for `artanh`'s two-compiler gap; at the
    /// outermost corner the `2·artanh` origin distance carries a derivative of
    /// 1e4 in `r`, so the relative form of the bound is doing the work here.
    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing Space WGSL"]
    fn hyperbolic_h3_wgsl_matches_the_rust_space_at_the_domain_corners() {
        assert_prelude_matches_cpu(&HyperbolicH3, &ball_corners(), 2e-4);
    }

    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing Space WGSL"]
    fn spherical_s3_wgsl_degrades_finitely_outside_the_hemisphere() {
        assert_prelude_survives_out_of_domain(&SphericalS3, "SphericalS3");
    }

    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing Space WGSL"]
    fn hyperbolic_h3_wgsl_degrades_finitely_outside_the_ball() {
        assert_prelude_survives_out_of_domain(&HyperbolicH3, "HyperbolicH3");
    }

    /// Dispatch one workgroup per element of `inputs` against a two-binding
    /// compute shader (`read` at 0, `read_write` at 1) and read the output back.
    /// Shared by the Space-ABI probe and the scene-SDF probe.
    async fn run_compute_probe<In: Pod, Out: Pod>(
        source: &str,
        label: &str,
        inputs: &[In],
    ) -> Result<Vec<Out>, String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("request_adapter failed: {e}"))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some(label),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
                experimental_features: Default::default(),
            })
            .await
            .map_err(|e| format!("request_device failed: {e}"))?;

        validate_wgsl(source).map_err(|e| e.to_string())?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.to_owned().into()),
        });

        let input_size = std::mem::size_of_val(inputs) as u64;
        let output_size = (inputs.len() * std::mem::size_of::<Out>()) as u64;

        let input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}-input")),
            size: input_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        input
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(inputs));
        input.unmap();

        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}-output")),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}-staging")),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label}-bgl")),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}-bg")),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label}-layout")),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{label}-pipeline")),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("{label}-encoder")),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{label}-pass")),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(inputs.len() as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output_size);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).expect("map callback receiver should exist");
        });
        // `submission_index = None` waits on the most recent submission;
        // `timeout = None` waits indefinitely.
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| e.to_string())?;
        rx.recv()
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;

        let data = slice.get_mapped_range();
        let rows = bytemuck::cast_slice::<u8, Out>(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(rows)
    }

    /// Every entry point of `space`'s prelude against the Rust `Space` it
    /// mirrors, at each corner of the chart's domain.
    ///
    /// `loam_origin_distance` is checked against `distance` from the chart
    /// origin because that identity is the entire content of the function. Its
    /// absence from this probe is why S³ shipped an `acos` form that reads zero
    /// for every radius under 1.73e-4.
    fn assert_prelude_matches_cpu<S>(space: &S, cases: &[ParityCase], eps: f32)
    where
        S: WgslSpace + Space<Point = Vec3, Vector = Vec3>,
    {
        let gpu: Vec<GpuCase> = cases.iter().map(|c| gpu_case(c.a, c.b, c.v)).collect();
        let rows = pollster::block_on(run_gpu_probe(space, &gpu)).expect("GPU probe");
        assert_eq!(cases.len(), rows.len());
        for (case, row) in cases.iter().zip(&rows) {
            let (a, b, v) = (case.a, case.b, case.v);
            let at = |what| format!("{} :: {what}", case.corner);
            assert_near(&at("distance"), row.scalars[0], space.distance(a, b), eps);
            assert_near(
                &at("origin_distance(a)"),
                row.scalars[1],
                space.distance(Vec3::ZERO, a),
                eps,
            );
            assert_near(
                &at("origin_distance(b)"),
                row.scalars[2],
                space.distance(Vec3::ZERO, b),
                eps,
            );
            assert_vec_near(&at("exp"), row.exp_point, space.exp(a, v).extend(0.0), eps);
            assert_vec_near(&at("log"), row.log_vec, space.log(a, b).extend(0.0), eps);
            assert_vec_near(
                &at("parallel_transport"),
                row.transported,
                space.parallel_transport(a, b, v).extend(0.0),
                eps,
            );
        }
    }

    /// Outside a `|p| < 1` chart neither twin contracts a value, only a
    /// degraded-but-finite one. Both clamp onto a saturation shell whose
    /// thickness is a few f32 ulps, and `artanh`, the `1/w` tangent lift and
    /// the conformal ratio all carry derivatives past 10⁶ with respect to where
    /// exactly that clamp lands, so two compilers rounding the clamp
    /// differently disagree by whole percent on the outputs. No parity budget
    /// separates that from a transcription error, and one written loose enough
    /// to pass would not fail on one either.
    ///
    /// What does survive out there is the contract the chart modules state:
    /// finite, never NaN, and a returned point still inside the chart. That is
    /// what a missing clamp, an unfloored divisor or the gyration's
    /// zero-denominator all break, so it is what this gates on. The divergence
    /// it declines to gate on is printed instead of inferred.
    fn assert_prelude_survives_out_of_domain<S>(space: &S, label: &str)
    where
        S: WgslSpace + Space<Point = Vec3, Vector = Vec3>,
    {
        let cases = out_of_domain_corners();
        let gpu: Vec<GpuCase> = cases.iter().map(|c| gpu_case(c.a, c.b, c.v)).collect();
        let rows = pollster::block_on(run_gpu_probe(space, &gpu)).expect("GPU probe");
        let mut worst = 0.0_f32;
        let mut worst_at = String::new();
        for (case, row) in cases.iter().zip(&rows) {
            let (a, b, v) = (case.a, case.b, case.v);
            let cpu = [
                Vec4::new(
                    space.distance(a, b),
                    space.distance(Vec3::ZERO, a),
                    space.distance(Vec3::ZERO, b),
                    0.0,
                ),
                space.exp(a, v).extend(0.0),
                space.log(a, b).extend(0.0),
                space.parallel_transport(a, b, v).extend(0.0),
            ];
            let gpu = [row.scalars, row.exp_point, row.log_vec, row.transported];
            let names = ["scalars", "exp", "log", "parallel_transport"];
            for ((name, cpu), gpu) in names.iter().zip(cpu).zip(gpu) {
                let where_ = || format!("{label}/{} a={a:?} b={b:?} v={v:?} {name}", case.corner);
                for (lane, (cpu, gpu)) in cpu.to_array().iter().zip(gpu).enumerate() {
                    assert!(
                        gpu.is_finite(),
                        "{}: GPU lane {lane} is {gpu}, not finite",
                        where_()
                    );
                    assert!(
                        cpu.is_finite(),
                        "{}: CPU lane {lane} is {cpu}, not finite",
                        where_()
                    );
                    let divergence = (gpu - cpu).abs() / cpu.abs().max(1.0);
                    if divergence > worst {
                        worst = divergence;
                        worst_at = format!("{} lane {lane}", where_());
                    }
                }
            }
            // Both charts are the open unit ball and both preludes end
            // `loam_exp` in their own clamp, so a result outside it is the clamp
            // failing. Excluded when the tangent is zero: there the tiny-
            // tangent early-out returns the base point, in domain or not.
            let exp_gpu = Vec3::new(row.exp_point[0], row.exp_point[1], row.exp_point[2]);
            assert!(
                v == Vec3::ZERO || exp_gpu.length_squared() <= 1.0,
                "{label}/{}: loam_exp({a:?}, {v:?}) returned {exp_gpu:?}, outside the chart",
                case.corner,
            );
        }
        println!("{label} out-of-domain CPU/GPU divergence: worst {worst} at {worst_at}");
    }

    /// CPU/GPU parity for `BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX>`,
    /// restricted to `loam_exp` (the highest-leverage method; transport has its own
    /// probe). `loam_log` and `loam_distance` are intentionally divergent (chart-diff
    /// vs Gauss-Newton; midpoint chord-metric vs full Riemannian) and the kernel
    /// doesn't call `loam_log`.
    ///
    /// Tolerance 5e-3: GPU runs 16 RK4 sub-steps, CPU 32; both 4th-order, so
    /// cumulative drift grows ~16x but stays small for the smooth conformal factor
    /// and the small `v` well inside the H3 ball.
    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing BlendedSpace WGSL"]
    fn blended_e3_h3_gpu_probe_exp_matches_cpu() {
        let space = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-0.5, 0.5));
        let cases = [
            // Pure E3 (alpha = 0): straight-line motion.
            gpu_case(
                Vec3::new(-1.0, 0.05, 0.0),
                Vec3::new(-0.8, 0.05, 0.0),
                Vec3::new(0.1, 0.0, 0.0),
            ),
            // Mid-zone (alpha ~ 0.5): exercises the conformal factor's gradient.
            gpu_case(
                Vec3::new(0.0, 0.05, 0.0),
                Vec3::new(0.1, 0.05, 0.0),
                Vec3::new(0.05, 0.0, 0.0),
            ),
            // Pure H3 (alpha = 1) at r=0.7: f(p) ~ 15.4x identity, non-linear.
            gpu_case(
                Vec3::new(0.7, 0.0, 0.0),
                Vec3::new(0.71, 0.05, 0.0),
                Vec3::new(0.02, 0.02, 0.0),
            ),
            // Straddles the `|v|² < 1e-14` early-out both sides share; the
            // RK4 loop and the identity return must agree here.
            gpu_case(
                Vec3::new(0.0, 0.05, 0.0),
                Vec3::new(0.0, 0.05, 0.0),
                Vec3::new(1e-7, 0.0, 0.0),
            ),
            // Outside the H3 unit ball but inside the blended chart: the
            // field pins alpha to 0 there, so the metric is flat and the
            // H3 side's saturation shell is never consulted.
            gpu_case(
                Vec3::new(-2.0, 0.3, 0.1),
                Vec3::new(-1.6, 0.3, 0.1),
                Vec3::new(0.4, 0.05, 0.0),
            ),
        ];
        let out =
            pollster::block_on(run_gpu_probe(&space, &cases)).expect("BlendedSpace GPU probe");

        for (case, row) in cases.iter().zip(&out) {
            let a = Vec3::from_array([case.a[0], case.a[1], case.a[2]]);
            let v = Vec3::from_array([case.v[0], case.v[1], case.v[2]]);
            let cpu = space.exp(a, v);
            let gpu = Vec3::new(row.exp_point[0], row.exp_point[1], row.exp_point[2]);
            let diff = (cpu - gpu).length();
            assert!(
                diff < 5e-3,
                "BlendedSpace exp parity failed at a={a:?} v={v:?}: cpu={cpu:?} gpu={gpu:?} diff={diff}",
            );
        }
    }

    /// CPU/GPU parity for
    /// `BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX>::parallel_transport`.
    /// Both sides run 8 RK4 sub-steps along the chart-coordinate line `a` to `b`.
    /// Paths sample pure E3 (identity transport), the mid-zone, and pure H3 at
    /// moderate radius. Tolerance matches the exp probe's 5e-3.
    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing BlendedSpace WGSL"]
    fn blended_e3_h3_gpu_probe_transport_matches_cpu() {
        let space = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-0.5, 0.5));
        let cases = [
            // Pure E3: transport is identity; drift is pure GPU-vs-CPU noise.
            gpu_case(
                Vec3::new(-1.0, 0.05, 0.0),
                Vec3::new(-0.8, 0.05, 0.0),
                Vec3::new(0.1, 0.0, 0.0),
            ),
            // Long traversal across the transition zone into H3 at r ~ 0.7; the
            // case that discriminates 8-step RK4 from single-step Euler.
            gpu_case(
                Vec3::new(-0.6, 0.0, 0.0),
                Vec3::new(0.7, 0.0, 0.0),
                Vec3::new(0.5, 0.5, 0.0),
            ),
            // Pure H3 at r ~ 0.7 where f(p) ~ 15.4x identity.
            gpu_case(
                Vec3::new(0.7, 0.0, 0.0),
                Vec3::new(0.72, 0.05, 0.0),
                Vec3::new(0.02, 0.02, 0.0),
            ),
            // Straddles the `|p_to - p_from|² < 1e-14` early-out.
            gpu_case(
                Vec3::new(0.0, 0.05, 0.0),
                Vec3::new(1e-7, 0.05, 0.0),
                Vec3::new(0.05, -0.02, 0.01),
            ),
            // Outside the H3 unit ball, inside the blended chart's flat zone.
            gpu_case(
                Vec3::new(-2.0, 0.3, 0.1),
                Vec3::new(-1.6, 0.3, 0.1),
                Vec3::new(0.4, 0.05, 0.0),
            ),
        ];
        let out =
            pollster::block_on(run_gpu_probe(&space, &cases)).expect("BlendedSpace GPU probe");

        for (case, row) in cases.iter().zip(&out) {
            let a = Vec3::from_array([case.a[0], case.a[1], case.a[2]]);
            let b = Vec3::from_array([case.b[0], case.b[1], case.b[2]]);
            let v = Vec3::from_array([case.v[0], case.v[1], case.v[2]]);
            let cpu = space.parallel_transport(a, b, v);
            let gpu = Vec3::new(row.transported[0], row.transported[1], row.transported[2]);
            let diff = (cpu - gpu).length();
            assert!(
                diff < 5e-3,
                "BlendedSpace transport parity failed at a={a:?} b={b:?} v={v:?}: cpu={cpu:?} gpu={gpu:?} diff={diff}",
            );
        }
    }

    /// The other half of `BlendedSpace`'s ABI. `loam_log` and `loam_distance`
    /// are not ports of `Space::log` and `Space::distance`; the prelude
    /// documents them as the chart-coordinate difference and the midpoint chord
    /// metric `√f((a+b)/2)·|b − a|`, and the Rust side runs Gauss-Newton
    /// shooting and a full Riemannian length. Asserting agreement between those
    /// would be asserting a falsehood, so the reference here is what the
    /// prelude claims to compute, evaluated through the shipped
    /// [`ConformallyFlat`] factor rather than a transcription of it. That still
    /// catches a swapped operand, a dropped `√`, or a midpoint that is not the
    /// midpoint, which is what the emitted text can plausibly get wrong.
    ///
    /// `loam_origin_distance` is `loam_distance` from the chart origin, so the
    /// same reference covers it and pins that the two stay consistent.
    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing BlendedSpace WGSL"]
    fn blended_e3_h3_log_and_distance_match_the_closed_forms_the_prelude_documents() {
        let space = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-0.5, 0.5));
        let corners = [
            corner(
                "pure E3 zone",
                Vec3::new(-1.0, 0.05, 0.0),
                Vec3::new(-0.8, 0.05, 0.0),
                Vec3::ZERO,
            ),
            corner(
                "mid-zone, small separation",
                Vec3::new(0.0, 0.05, 0.0),
                Vec3::new(1e-4, 0.05, 0.0),
                Vec3::ZERO,
            ),
            corner(
                "pure H3 zone at r = 0.7",
                Vec3::new(0.7, 0.0, 0.0),
                Vec3::new(0.71, 0.05, 0.0),
                Vec3::ZERO,
            ),
            corner(
                "across the whole transition zone",
                Vec3::new(-0.6, 0.0, 0.0),
                Vec3::new(0.7, 0.0, 0.0),
                Vec3::ZERO,
            ),
            corner(
                "outside the H3 unit ball, in the flat zone",
                Vec3::new(-2.0, 0.3, 0.1),
                Vec3::new(-1.6, 0.3, 0.1),
                Vec3::ZERO,
            ),
        ];
        let gpu: Vec<GpuCase> = corners.iter().map(|c| gpu_case(c.a, c.b, c.v)).collect();
        let rows = pollster::block_on(run_gpu_probe(&space, &gpu)).expect("BlendedSpace GPU probe");

        let chord = |a: Vec3, b: Vec3| {
            space.conformal_factor((a + b) * 0.5).max(0.0).sqrt() * (b - a).length()
        };
        for (case, row) in corners.iter().zip(&rows) {
            let (a, b) = (case.a, case.b);
            let at = |what| format!("{} :: {what}", case.corner);
            assert_near(&at("distance"), row.scalars[0], chord(a, b), 1e-5);
            assert_near(
                &at("origin_distance(a)"),
                row.scalars[1],
                chord(Vec3::ZERO, a),
                1e-5,
            );
            assert_near(
                &at("origin_distance(b)"),
                row.scalars[2],
                chord(Vec3::ZERO, b),
                1e-5,
            );
            assert_vec_near(&at("log"), row.log_vec, (b - a).extend(0.0), 1e-6);
        }
    }

    // ---- Scene SDF: CPU evaluator vs emitted shader -----------------------

    /// Writes `loam_scene_sdf(p.xyz)` per sample point. Assembled as
    /// prelude + `Scene::to_wgsl` + this.
    const SCENE_SDF_PROBE: &str = r#"
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    out[i] = vec4<f32>(loam_scene_sdf(points[i].xyz), 0.0, 0.0, 0.0);
}
"#;

    /// 4D counterpart: writes both fields of `loam_scene_at`, so the probe
    /// covers `Scene4::eval_at`'s kind tracking and not only its distance.
    const SCENE4_HIT_PROBE: &str = r#"
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let hit = loam_scene_at(points[i].xyz);
    out[i] = vec4<f32>(hit.dist, f32(hit.kind), 0.0, 0.0);
}
"#;

    // Geometry shared by the probe scenes and the explicit boundary sample
    // points, so "on the surface" and "in the blend band" stay true when a
    // constant is retuned.
    const PROBE_BALL_A: (Vec3, f32) = (Vec3::new(0.10, 0.00, 0.05), 0.22);
    const PROBE_BALL_B: (Vec3, f32) = (Vec3::new(-0.15, 0.08, 0.00), 0.18);
    const PROBE_BOX_HALF_EXTENTS: Vec3 = Vec3::new(0.20, 0.15, 0.25);
    const PROBE_PLANE_OFFSET: f32 = -0.30;
    /// The two smooth-union blend radii, an order of magnitude apart, so the
    /// probe sees both a wide active band and a nearly hard `min`.
    const PROBE_SMOOTH_K: [f32; 2] = [0.12, 0.012];

    /// One scene per emit feature so a parity failure localises to a single
    /// combinator or leaf rather than to "the tree walk".
    fn probe_scenes() -> Vec<(&'static str, loam_scene::Scene)> {
        use loam_scene::{Scene, SceneNode};
        let ball_a = || SceneNode::sphere(PROBE_BALL_A.0, PROBE_BALL_A.1);
        let ball_b = || SceneNode::sphere(PROBE_BALL_B.0, PROBE_BALL_B.1);
        let box3 = || SceneNode::box_(PROBE_BOX_HALF_EXTENTS);
        let plane = || SceneNode::plane(Vec3::Y, PROBE_PLANE_OFFSET);
        vec![
            ("sphere", Scene::new(ball_a())),
            ("sphere union plane", Scene::new(ball_a().union(plane()))),
            (
                "sphere intersect box",
                Scene::new(ball_a().intersect(box3())),
            ),
            (
                "sphere minus sphere",
                Scene::new(ball_a().subtract(ball_b())),
            ),
            (
                "smooth union k=0.12",
                Scene::new(ball_a().smooth_union(ball_b(), PROBE_SMOOTH_K[0])),
            ),
            (
                "smooth union k=0.012",
                Scene::new(ball_a().smooth_union(ball_b(), PROBE_SMOOTH_K[1])),
            ),
            (
                "three-deep nested tree",
                Scene::new(
                    ball_a()
                        .smooth_union(box3(), PROBE_SMOOTH_K[0])
                        .union(ball_b().subtract(plane()))
                        .intersect(SceneNode::cube(0.6)),
                ),
            ),
        ]
    }

    /// Seeded lattice plus the analytically interesting points: leaf centres
    /// (where a sphere reads `-r`), leaf surfaces, the `Difference` seam, the
    /// midpoint of the two balls (inside every blend band), and the box corner.
    /// `extent` shrinks the random cloud for charts with a boundary shell.
    fn scene_probe_points(extent: f32) -> Vec<[f32; 4]> {
        let mut points: Vec<Vec3> = vec![
            Vec3::ZERO,
            PROBE_BALL_A.0,
            PROBE_BALL_B.0,
            PROBE_BALL_A.0 + Vec3::X * PROBE_BALL_A.1,
            PROBE_BALL_A.0 - Vec3::Y * PROBE_BALL_A.1,
            PROBE_BALL_B.0 + Vec3::Z * PROBE_BALL_B.1,
            (PROBE_BALL_A.0 + PROBE_BALL_B.0) * 0.5,
            PROBE_BOX_HALF_EXTENTS,
            PROBE_BOX_HALF_EXTENTS * Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(0.0, PROBE_PLANE_OFFSET, 0.0),
        ];
        let mut state: u32 = 0x517E_5DF0;
        let mut next_f32 = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for _ in 0..118 {
            points.push(Vec3::new(
                next_f32() * extent,
                next_f32() * extent,
                next_f32() * extent,
            ));
        }
        points
            .into_iter()
            .map(|p| p.extend(0.0).to_array())
            .collect()
    }

    /// Run every probe scene through the GPU and against `Scene::eval`,
    /// returning the largest absolute residual seen. Fails on the first sample
    /// exceeding `tolerance`.
    fn assert_scene_parity<S>(space: &S, label: &str, extent: f32, tolerance: f32) -> f32
    where
        S: WgslSpace + Space<Point = Vec3, Vector = Vec3>,
    {
        let points = scene_probe_points(extent);
        let mut worst = 0.0_f32;
        for (name, scene) in probe_scenes() {
            let source = assemble_source_with_scene(
                &space.wgsl_impl(),
                Some(&scene.to_wgsl(space)),
                SCENE_SDF_PROBE,
            );
            let rows: Vec<[f32; 4]> =
                pollster::block_on(run_compute_probe(&source, "loam-scene-gpu-probe", &points))
                    .expect("scene GPU probe");
            for (point, row) in points.iter().zip(&rows) {
                let p = Vec3::new(point[0], point[1], point[2]);
                let cpu = scene.eval(space, p);
                let residual = (cpu - row[0]).abs();
                worst = worst.max(residual);
                assert!(
                    residual <= tolerance,
                    "{label}/{name}: CPU {cpu} vs GPU {} at {p:?} differ by {residual} \
                     (tolerance {tolerance})",
                    row[0],
                );
            }
        }
        worst
    }

    /// Flat chart, so the only structural gap is the emitter's `{:.6}` constant
    /// rounding (bounded near 5e-7) plus GPU rounding. Tolerance matches the
    /// Space-ABI probe's `EuclideanR3` figure.
    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing scene emit or eval"]
    fn scene_sdf_gpu_probe_matches_cpu_in_euclidean_r3() {
        let worst = assert_scene_parity(&EuclideanR3, "EuclideanR3", 0.9, 1e-5);
        println!("EuclideanR3 scene parity: worst residual {worst}");
    }

    /// Curved chart: `Sphere` routes through `loam_distance` on the GPU and
    /// `Space::distance` on the CPU, which agree by construction for H³, so the
    /// residual is the artanh/Möbius rounding difference between the two
    /// implementations. Tolerance matches the Space-ABI probe's H³ figure.
    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing scene emit or eval"]
    fn scene_sdf_gpu_probe_matches_cpu_in_hyperbolic_h3() {
        let worst = assert_scene_parity(&HyperbolicH3, "HyperbolicH3", 0.30, 2e-4);
        println!("HyperbolicH3 scene parity: worst residual {worst}");
    }

    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing scene emit or eval"]
    fn scene_sdf_gpu_probe_matches_cpu_in_spherical_s3() {
        let worst = assert_scene_parity(&SphericalS3, "SphericalS3", 0.30, 2e-4);
        println!("SphericalS3 scene parity: worst residual {worst}");
    }

    /// `BlendedSpace` is the one Space whose Rust `distance` (Gauss-Newton
    /// shooting `log` plus conformal rescale) and WGSL `loam_distance` (midpoint
    /// chord metric, first-order accurate for nearby points) are deliberately
    /// different functions. The CPU evaluator is built on the Rust side because
    /// that is the validated reference; the consequence is that the CPU and GPU
    /// scene SDFs are different scalar fields here. This test therefore records
    /// a bound rather than gating on agreement: the number it prints is the
    /// input to "can a baked collider serve this Space".
    #[test]
    #[ignore = "requires a working wgpu adapter; records the BlendedSpace CPU/GPU divergence"]
    fn scene_sdf_gpu_probe_records_blended_space_divergence() {
        let space = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-0.5, 0.5));
        let worst = assert_scene_parity(&space, "BlendedSpace<E3,H3>", 0.30, 5e-2);
        println!("BlendedSpace<E3,H3> scene parity: worst residual {worst}");
    }

    /// The hyperslice path, asserting both fields of `LoamSceneHit`. `w_slice`
    /// is baked as a literal so the probe needs no uniform buffer. The Space
    /// prelude is unused by a `Scene4` module but `assemble_source_with_scene`
    /// wants one.
    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing scene4 emit or eval"]
    fn scene4_hyperslice_gpu_probe_matches_cpu() {
        use glam::Vec4;
        use loam_scene::{Scene4, SceneNode4};

        const W_SLICE: f32 = 0.25;
        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::new(0.1, 0.0, -0.05, 0.0), 0.5)
                .union(SceneNode4::halfspace(Vec4::Y, -0.4))
                .subtract(SceneNode4::hypersphere(Vec4::new(0.3, 0.1, 0.0, 0.1), 0.2))
                .intersect(SceneNode4::hypersphere(Vec4::ZERO, 1.2)),
        );
        let source = assemble_source_with_scene(
            &EuclideanR3.wgsl_impl(),
            Some(&scene.to_hyperslice_wgsl(&format!("{W_SLICE}"))),
            SCENE4_HIT_PROBE,
        );
        let points = scene_probe_points(0.9);
        let rows: Vec<[f32; 4]> =
            pollster::block_on(run_compute_probe(&source, "loam-scene4-gpu-probe", &points))
                .expect("scene4 GPU probe");

        let mut worst = 0.0_f32;
        for (point, row) in points.iter().zip(&rows) {
            let p = Vec3::new(point[0], point[1], point[2]);
            let (cpu_dist, cpu_kind) = scene.eval_at(p, W_SLICE, true);
            let residual = (cpu_dist - row[0]).abs();
            worst = worst.max(residual);
            assert!(
                residual <= 1e-5,
                "scene4: CPU {cpu_dist} vs GPU {} at {p:?} differ by {residual}",
                row[0],
            );
            assert_eq!(
                cpu_kind, row[1] as u32,
                "scene4: kind mismatch at {p:?} (CPU {cpu_kind}, GPU {})",
                row[1],
            );
        }
        println!("Scene4 hyperslice parity: worst residual {worst}");
    }

    fn assert_vec_near(what: &str, actual: [f32; 4], expected: Vec4, eps: f32) {
        for (lane, (actual, expected)) in actual.iter().zip(expected.to_array()).enumerate() {
            assert_near(&format!("{what}[{lane}]"), *actual, expected, eps);
        }
    }

    /// Absolute floor under the relative budget. Two f32 pipelines that agree
    /// to their last bits still differ by ~1e-7 on a coordinate of order 1, and
    /// a lane whose true value is 0 has no relative scale to be held to. One
    /// order above that single-ulp figure and no higher: the small-radius
    /// corner works at 1e-4, so a floor of 1e-6 still holds it to 1% and the
    /// `acos(sqrt(1 − r²))` origin distance, which reads exactly 0 there,
    /// cannot pass under it.
    const PARITY_ABSOLUTE_FLOOR: f32 = 1e-6;

    /// Relative, because the curved charts carry conformal factors reaching
    /// 10⁷ near their boundaries and metric quantities down at 1e-4 near their
    /// origins; a flat absolute bound would either pass everything at one end
    /// or fail everything at the other.
    fn assert_near(what: &str, actual: f32, expected: f32, eps: f32) {
        let budget = eps * expected.abs() + PARITY_ABSOLUTE_FLOOR;
        assert!(
            (actual - expected).abs() <= budget,
            "{what}: GPU {actual} differs from CPU {expected} by more than {budget}",
        );
    }

    // ---- Owner scoping ----------------------------------------------------

    /// A `Device` from wgpu's noop backend. `create_shader_module` is a stub
    /// there, which is exactly the layer the owner-scoping tests do not care
    /// about: naga still validates every assembled source inside
    /// [`ShaderDb::compile`], and the entry bookkeeping under test is the db's
    /// own. Buys these tests a real `ShaderDb` with no GPU, so they run
    /// unconditionally instead of behind `#[ignore]`.
    fn noop_device() -> Device {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::NOOP,
            backend_options: wgpu::BackendOptions {
                noop: wgpu::NoopBackendOptions { enable: true },
                ..Default::default()
            },
            ..Default::default()
        });
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("noop backend always yields an adapter");
        let (device, _queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("loam-shader-owner-scope-tests"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
                experimental_features: Default::default(),
            }))
            .expect("noop adapter always yields a device");
        device
    }

    fn modified(path: &Path) -> Vec<AssetEvent> {
        vec![AssetEvent {
            path: path.to_path_buf(),
            kind: AssetEventKind::Modified,
        }]
    }

    /// Rewrite `path` with source that still validates but differs byte-wise,
    /// so a recompile is observable as a generation bump rather than a no-op.
    fn touch_with_edit(path: &Path) {
        let previous = std::fs::read_to_string(path).unwrap();
        let edited = previous.replace("vec3<f32>(0.1, 0.2, 0.3)", "vec3<f32>(0.15, 0.25, 0.35)");
        assert_ne!(previous, edited, "the edit must actually change the file");
        std::fs::write(path, edited).unwrap();
    }

    /// The prelude is a property of the loader, not of the file: two owners
    /// naming the same shader path in different Spaces must get two entries,
    /// because one module cannot carry both preludes. A path-keyed db hands
    /// back one shared ID and recompiles it out from under the first owner.
    #[test]
    fn two_owners_of_one_path_get_independent_modules() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shared.wgsl");
        std::fs::write(&path, ABI_PROBE).unwrap();

        let mut db = ShaderDb::new(noop_device());
        let spherical = db.new_owner();
        let hyperbolic = db.new_owner();
        let in_s3 = db.load(spherical, &path, &SphericalS3).unwrap();
        let in_h3 = db.load(hyperbolic, &path, &HyperbolicH3).unwrap();

        assert_ne!(
            in_s3, in_h3,
            "one path loaded by two owners must not collapse to one module",
        );
        assert_eq!(db.generation(in_s3), 1, "loading H3's copy recompiled S3's");
        assert_eq!(db.generation(in_h3), 1);
    }

    /// The property the shell's fan-out rests on: one owner's apply reloads
    /// that owner's entry for the edited path and leaves every other owner's
    /// entry, for the same path, at the generation it had.
    #[test]
    fn an_apply_bumps_only_the_generation_of_its_own_owner() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shared.wgsl");
        std::fs::write(&path, ABI_PROBE).unwrap();

        let mut db = ShaderDb::new(noop_device());
        let spherical = db.new_owner();
        let hyperbolic = db.new_owner();
        let in_s3 = db.load(spherical, &path, &SphericalS3).unwrap();
        let in_h3 = db.load(hyperbolic, &path, &HyperbolicH3).unwrap();

        touch_with_edit(&path);
        let events = modified(&path);

        db.apply_events(spherical, &events, &SphericalS3);
        assert_eq!(db.generation(in_s3), 2, "S3's owner asked for this reload");
        assert_eq!(
            db.generation(in_h3),
            1,
            "H3's module must be untouched by an apply scoped to S3's owner",
        );

        db.apply_events(hyperbolic, &events, &HyperbolicH3);
        assert_eq!(
            db.generation(in_s3),
            2,
            "H3's owner must not recompile S3's module against H3",
        );
        assert_eq!(db.generation(in_h3), 2);
    }

    /// A host fans the same event slice at every owner it holds. Each shader
    /// must therefore be recompiled exactly once, by the owner that loaded it:
    /// a db-wide apply would rebuild the edited module once per owner, leaving
    /// it compiled against whichever Space fanned out last.
    #[test]
    fn a_fan_out_recompiles_each_edited_shader_exactly_once() {
        let dir = tempfile::tempdir().unwrap();
        let s3_path = dir.path().join("spherical.wgsl");
        let h3_path = dir.path().join("hyperbolic.wgsl");
        std::fs::write(&s3_path, ABI_PROBE).unwrap();
        std::fs::write(&h3_path, ABI_PROBE).unwrap();

        let mut db = ShaderDb::new(noop_device());
        let spherical = db.new_owner();
        let hyperbolic = db.new_owner();
        let in_s3 = db.load(spherical, &s3_path, &SphericalS3).unwrap();
        let in_h3 = db.load(hyperbolic, &h3_path, &HyperbolicH3).unwrap();

        touch_with_edit(&s3_path);
        let events = modified(&s3_path);
        db.apply_events(spherical, &events, &SphericalS3);
        db.apply_events(hyperbolic, &events, &HyperbolicH3);

        assert_eq!(
            db.generation(in_s3),
            2,
            "the edited shader must recompile once, under its own Space, not once per owner",
        );
        assert_eq!(db.generation(in_h3), 1, "no event named H3's shader");
    }

    /// [`ShaderDb::load`]'s stability contract, now conditional on the owner:
    /// one owner naming one path twice must land on its existing entry and
    /// recompile it. Minting a second ID instead would strand the first behind
    /// whatever pipeline already holds it, unreachable by any later apply.
    #[test]
    fn a_second_load_by_the_same_owner_keeps_the_id_and_recompiles() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shared.wgsl");
        std::fs::write(&path, ABI_PROBE).unwrap();

        let mut db = ShaderDb::new(noop_device());
        let scene = db.new_owner();
        let first = db.load(scene, &path, &SphericalS3).unwrap();
        let second = db.load(scene, &path, &SphericalS3).unwrap();

        assert_eq!(first, second, "one owner's path must map to one entry");
        assert_eq!(db.generation(first), 2, "the second load must recompile");
    }

    /// The shipped pairing: a host applies under [`ShaderDb::ROOT_OWNER`] via
    /// the `App` default while a sub-scene applies under a minted owner. A
    /// `new_owner` that handed back the root would put both in one path space,
    /// which is the collision owner scoping exists to remove, and no assertion
    /// over minted owners alone would notice.
    #[test]
    fn a_minted_owner_shares_no_entry_with_the_root_owner() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shared.wgsl");
        std::fs::write(&path, ABI_PROBE).unwrap();

        let mut db = ShaderDb::new(noop_device());
        let scene = db.new_owner();
        let in_root = db.load(ShaderDb::ROOT_OWNER, &path, &EuclideanR3).unwrap();
        let in_scene = db.load(scene, &path, &HyperbolicH3).unwrap();
        assert_ne!(
            in_root, in_scene,
            "a minted owner must not land in the root's path space",
        );

        touch_with_edit(&path);
        db.apply_events(ShaderDb::ROOT_OWNER, &modified(&path), &EuclideanR3);
        assert_eq!(db.generation(in_root), 2);
        assert_eq!(
            db.generation(in_scene),
            1,
            "the host's apply must not recompile a sub-scene's module against R3",
        );
    }

    /// Hot-reload's text path: read, assemble against a Space prelude, validate via
    /// naga, mutate, repeat. Pins the Device-free I/O + assembly + validation that
    /// [`ShaderDb::reload`] depends on (the Device-bound layer is just
    /// `create_shader_module` over the same validated source).
    #[test]
    fn hot_reload_pipeline_reads_assembles_and_validates_mutated_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("probe.wgsl");

        let v1 = ABI_PROBE;
        std::fs::write(&path, v1).unwrap();

        let read1 = std::fs::read_to_string(&path).unwrap();
        let src1 = assemble_source(&EuclideanR3.wgsl_impl(), &read1);
        validate_wgsl(&src1).expect("v1 must validate");

        // Mutate in place (same path, changed bytes, as the watcher sees a save);
        // tweak a constant, not structure, so v2 still validates.
        let v2 = ABI_PROBE.replace("vec3<f32>(0.1, 0.2, 0.3)", "vec3<f32>(0.4, 0.5, 0.6)");
        assert_ne!(v1, v2, "test mutation should produce different source");
        std::fs::write(&path, &v2).unwrap();

        let read2 = std::fs::read_to_string(&path).unwrap();
        assert_ne!(read1, read2, "file mutation should change bytes");
        let src2 = assemble_source(&EuclideanR3.wgsl_impl(), &read2);
        validate_wgsl(&src2).expect("v2 must validate after mutation");
    }
}
