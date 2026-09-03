use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use loam_asset::{AssetEvent, AssetEventKind};
use loam_math::WgslSpace;
use wgpu::{Device, ShaderModule, ShaderModuleDescriptor, ShaderSource};

#[derive(Debug, thiserror::Error)]
pub enum WgslValidationError {
    #[error("WGSL parse error: {0}")]
    Parse(#[from] naga::front::wgsl::ParseError),
    #[error("WGSL validation error: {0}")]
    Validate(Box<naga::WithSpan<naga::valid::ValidationError>>),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShaderId(u32);

/// One path under two owners is two modules; hot-reload is scoped by owner.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShaderOwner(u32);

struct Entry {
    path: PathBuf,
    module: ShaderModule,
    scene_source: Option<String>,
    prelude: Cow<'static, str>,
    generation: u64,
    label: String,
}

/// A failed reload or a removed file keeps the last good module.
pub struct ShaderDb {
    device: Device,
    entries: HashMap<ShaderId, Entry>,
    path_index: HashMap<ShaderOwner, HashMap<PathBuf, ShaderId>>,
    next_id: u32,
    next_owner: u32,
}

impl ShaderDb {
    pub const ROOT_OWNER: ShaderOwner = ShaderOwner(0);

    pub fn new(device: Device) -> Self {
        Self {
            device,
            entries: HashMap::new(),
            path_index: HashMap::new(),
            next_id: 0,
            next_owner: Self::ROOT_OWNER.0 + 1,
        }
    }

    pub fn new_owner(&mut self) -> ShaderOwner {
        let owner = ShaderOwner(self.next_owner);
        self.next_owner += 1;
        owner
    }

    /// The id is stable across reloads for one owner and path.
    pub fn load<S: WgslSpace>(
        &mut self,
        owner: ShaderOwner,
        path: impl AsRef<Path>,
        space: &S,
    ) -> Result<ShaderId> {
        self.load_inner(owner, path, None, space)
    }

    /// The scene source is stored and reused on reloads of the shader file.
    pub fn load_with_scene<S: WgslSpace>(
        &mut self,
        owner: ShaderOwner,
        path: impl AsRef<Path>,
        scene_source: &str,
        space: &S,
    ) -> Result<ShaderId> {
        self.load_inner(owner, path, Some(scene_source), space)
    }

    /// Prelude, scene SDF, [`crate::GEODESIC_MARCH_KERNEL`], user shading, in that order.
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
        let prelude = space.wgsl_impl();
        let module = self.compile(&path, &source, scene_source, &prelude)?;

        let label = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());

        if let Some(id) = self.lookup(owner, &path) {
            let entry = self.entries.get_mut(&id).expect("path_index out of sync");
            entry.module = module;
            entry.scene_source = scene_source.map(str::to_owned);
            entry.prelude = prelude;
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
                    prelude,
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

    pub fn module(&self, id: ShaderId) -> &ShaderModule {
        &self
            .entries
            .get(&id)
            .expect("unknown ShaderId - was it loaded by this ShaderDb?")
            .module
    }

    /// Bumps on every successful compile.
    pub fn generation(&self, id: ShaderId) -> u64 {
        self.entries.get(&id).map(|e| e.generation).unwrap_or(0)
    }

    /// Other owners are untouched, including entries for the same path.
    pub fn apply_events(&mut self, owner: ShaderOwner, events: &[AssetEvent]) {
        for event in events {
            let canonical = match canonicalize(&event.path) {
                Ok(p) => p,
                Err(_) => event.path.clone(),
            };
            let Some(id) = self.lookup(owner, &canonical) else {
                continue;
            };
            match event.kind {
                AssetEventKind::Created | AssetEventKind::Modified => {
                    if let Err(e) = self.reload(id) {
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

    fn reload(&mut self, id: ShaderId) -> Result<()> {
        let entry = &self.entries[&id];
        let source = std::fs::read_to_string(&entry.path)
            .with_context(|| format!("reading shader {}", entry.path.display()))?;
        let module = self.compile(
            &entry.path,
            &source,
            entry.scene_source.as_deref(),
            &entry.prelude,
        )?;
        let entry = self.entries.get_mut(&id).expect("id just looked up");
        entry.module = module;
        entry.generation += 1;
        Ok(())
    }

    fn compile(
        &self,
        path: &Path,
        user_source: &str,
        scene_source: Option<&str>,
        prelude: &str,
    ) -> Result<ShaderModule> {
        let full = assemble_source_with_scene(prelude, scene_source, user_source);
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

/// Headless; `wgpu` still validates at module creation.
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

    #[test]
    fn euclidean_r4_space_prelude_validates_against_abi_probe() {
        let src = assemble_source(&EuclideanR4.wgsl_impl(), ABI_PROBE_VEC4);
        validate_wgsl(&src).expect("EuclideanR4 WGSL prelude should validate");
    }

    const KERNEL_SCENE: &str = r#"
fn loam_scene_sdf(p: vec3<f32>) -> f32 {
    return loam_distance(p, vec3<f32>(0.0, 0.0, 0.0)) - 0.25;
}
"#;

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

    struct ParityCase {
        corner: &'static str,
        a: Vec3,
        b: Vec3,
        v: Vec3,
    }

    fn corner(corner: &'static str, a: Vec3, b: Vec3, v: Vec3) -> ParityCase {
        ParityCase { corner, a, b, v }
    }

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

    // Literal radii: a fixture that reads the shell constant retunes with it.
    fn hemisphere_corners() -> Vec<ParityCase> {
        vec![
            corner(
                "at the pole",
                Vec3::ZERO,
                Vec3::new(0.2, -0.1, 0.05),
                Vec3::new(0.01, 0.02, -0.03),
            ),
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

    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing Space WGSL"]
    fn euclidean_r3_wgsl_matches_the_rust_space_at_the_domain_corners() {
        assert_prelude_matches_cpu(&EuclideanR3, &flat_corners(), 1e-6);
    }

    // Per corner: one constant `w` would separate the coincident pair.
    const FLAT_CORNER_W: [[f32; 3]; 5] = [
        [0.0, 0.0, 0.125],
        [1.0, 1.0 + 1e-8, 1e-8],
        [1e-4, -1e-4, 2e-5],
        [-0.75, 2.5, 0.125],
        [-6.0, 9.0, 40.0],
    ];

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

    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing Space WGSL"]
    fn spherical_s3_wgsl_matches_the_rust_space_at_the_domain_corners() {
        assert_prelude_matches_cpu(&SphericalS3, &hemisphere_corners(), 2e-4);
    }

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

    // Two-binding compute shader: `read` at 0, `read_write` at 1.
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

    // No parity budget survives the shell clamp; gate on finite and in-chart instead.
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
            // A zero tangent returns the base point, in domain or not.
            let exp_gpu = Vec3::new(row.exp_point[0], row.exp_point[1], row.exp_point[2]);
            assert!(
                v == Vec3::ZERO || exp_gpu.length_squared() <= 1.0,
                "{label}/{}: loam_exp({a:?}, {v:?}) returned {exp_gpu:?}, outside the chart",
                case.corner,
            );
        }
        println!("{label} out-of-domain CPU/GPU divergence: worst {worst} at {worst_at}");
    }

    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing BlendedSpace WGSL"]
    fn blended_e3_h3_gpu_probe_exp_matches_cpu() {
        let space = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-0.5, 0.5));
        let cases = [
            // Pure E3.
            gpu_case(
                Vec3::new(-1.0, 0.05, 0.0),
                Vec3::new(-0.8, 0.05, 0.0),
                Vec3::new(0.1, 0.0, 0.0),
            ),
            // Mid-zone.
            gpu_case(
                Vec3::new(0.0, 0.05, 0.0),
                Vec3::new(0.1, 0.05, 0.0),
                Vec3::new(0.05, 0.0, 0.0),
            ),
            // Pure H3 at r = 0.7.
            gpu_case(
                Vec3::new(0.7, 0.0, 0.0),
                Vec3::new(0.71, 0.05, 0.0),
                Vec3::new(0.02, 0.02, 0.0),
            ),
            // Straddles the `|v|² < 1e-14` early-out.
            gpu_case(
                Vec3::new(0.0, 0.05, 0.0),
                Vec3::new(0.0, 0.05, 0.0),
                Vec3::new(1e-7, 0.0, 0.0),
            ),
            // Outside the H3 ball, in the flat zone.
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

    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing BlendedSpace WGSL"]
    fn blended_e3_h3_gpu_probe_transport_matches_cpu() {
        let space = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-0.5, 0.5));
        let cases = [
            // Pure E3.
            gpu_case(
                Vec3::new(-1.0, 0.05, 0.0),
                Vec3::new(-0.8, 0.05, 0.0),
                Vec3::new(0.1, 0.0, 0.0),
            ),
            // Across the transition zone into H3.
            gpu_case(
                Vec3::new(-0.6, 0.0, 0.0),
                Vec3::new(0.7, 0.0, 0.0),
                Vec3::new(0.5, 0.5, 0.0),
            ),
            // Pure H3 at r = 0.7.
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
            // Outside the H3 ball, in the flat zone.
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

    // Assembled as prelude + `Scene::to_wgsl` + this.
    const SCENE_SDF_PROBE: &str = r#"
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    out[i] = vec4<f32>(loam_scene_sdf(points[i].xyz), 0.0, 0.0, 0.0);
}
"#;

    // Covers `Scene4::eval_at`'s kind tracking, not only its distance.
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

    // Shared by the scenes and the boundary sample points.
    const PROBE_BALL_A: (Vec3, f32) = (Vec3::new(0.10, 0.00, 0.05), 0.22);
    const PROBE_BALL_B: (Vec3, f32) = (Vec3::new(-0.15, 0.08, 0.00), 0.18);
    const PROBE_BOX_HALF_EXTENTS: Vec3 = Vec3::new(0.20, 0.15, 0.25);
    const PROBE_PLANE_OFFSET: f32 = -0.30;
    // A wide band and a nearly hard `min`.
    const PROBE_SMOOTH_K: [f32; 2] = [0.12, 0.012];

    // One scene per emit feature, so a parity failure localises to one combinator.
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

    // `extent` shrinks the random cloud for charts with a boundary shell.
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

    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing scene emit or eval"]
    fn scene_sdf_gpu_probe_matches_cpu_in_euclidean_r3() {
        let worst = assert_scene_parity(&EuclideanR3, "EuclideanR3", 0.9, 1e-5);
        println!("EuclideanR3 scene parity: worst residual {worst}");
    }

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

    #[test]
    #[ignore = "requires a working wgpu adapter; records the BlendedSpace CPU/GPU divergence"]
    fn scene_sdf_gpu_probe_records_blended_space_divergence() {
        let space = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-0.5, 0.5));
        let worst = assert_scene_parity(&space, "BlendedSpace<E3,H3>", 0.30, 5e-2);
        println!("BlendedSpace<E3,H3> scene parity: worst residual {worst}");
    }

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

    // One order above f32 pipeline noise on a coordinate of order 1.
    const PARITY_ABSOLUTE_FLOOR: f32 = 1e-6;

    // Relative: conformal factors reach 1e7 near the boundaries.
    fn assert_near(what: &str, actual: f32, expected: f32, eps: f32) {
        let budget = eps * expected.abs() + PARITY_ABSOLUTE_FLOOR;
        assert!(
            (actual - expected).abs() <= budget,
            "{what}: GPU {actual} differs from CPU {expected} by more than {budget}",
        );
    }

    // Noop backend: naga still validates inside `ShaderDb::compile`.
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

    // Still validates; the needle appears in both probes.
    fn touch_with_edit(path: &Path) {
        let previous = std::fs::read_to_string(path).unwrap();
        let edited = previous.replace("0.1, 0.2, 0.3", "0.15, 0.25, 0.35");
        assert_ne!(previous, edited, "the edit must actually change the file");
        std::fs::write(path, edited).unwrap();
    }

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

        db.apply_events(spherical, &events);
        assert_eq!(db.generation(in_s3), 2, "S3's owner asked for this reload");
        assert_eq!(
            db.generation(in_h3),
            1,
            "H3's module must be untouched by an apply scoped to S3's owner",
        );

        db.apply_events(hyperbolic, &events);
        assert_eq!(
            db.generation(in_s3),
            2,
            "H3's owner must not touch S3's module",
        );
        assert_eq!(db.generation(in_h3), 2);
    }

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
        db.apply_events(spherical, &events);
        db.apply_events(hyperbolic, &events);

        assert_eq!(
            db.generation(in_s3),
            2,
            "the edited shader must recompile once, not once per owner",
        );
        assert_eq!(db.generation(in_h3), 1, "no event named H3's shader");
    }

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
        db.apply_events(ShaderDb::ROOT_OWNER, &modified(&path));
        assert_eq!(db.generation(in_root), 2);
        assert_eq!(
            db.generation(in_scene),
            1,
            "the host's apply must not recompile a sub-scene's module",
        );
    }

    #[test]
    fn a_reload_rebuilds_each_entry_against_the_prelude_it_was_loaded_with() {
        let dir = tempfile::tempdir().unwrap();
        let flat3 = dir.path().join("flat3.wgsl");
        let flat4 = dir.path().join("flat4.wgsl");
        std::fs::write(&flat3, ABI_PROBE).unwrap();
        std::fs::write(&flat4, ABI_PROBE_VEC4).unwrap();

        let mut db = ShaderDb::new(noop_device());
        let owner = ShaderDb::ROOT_OWNER;
        let in_r3 = db.load(owner, &flat3, &EuclideanR3).unwrap();
        let in_r4 = db.load(owner, &flat4, &EuclideanR4).unwrap();

        touch_with_edit(&flat3);
        touch_with_edit(&flat4);
        let events = [modified(&flat3), modified(&flat4)].concat();
        db.apply_events(owner, &events);

        assert_eq!(db.generation(in_r3), 2);
        assert_eq!(
            db.generation(in_r4),
            2,
            "the ℝ⁴ entry must reassemble under the vec4 prelude it was loaded with",
        );
    }

    #[test]
    fn re_loading_under_a_new_space_repoints_later_reloads_at_the_new_prelude() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("respecialized.wgsl");
        std::fs::write(&path, ABI_PROBE).unwrap();

        let mut db = ShaderDb::new(noop_device());
        let owner = ShaderDb::ROOT_OWNER;
        let id = db.load(owner, &path, &EuclideanR3).unwrap();

        std::fs::write(&path, ABI_PROBE_VEC4).unwrap();
        assert_eq!(
            db.load(owner, &path, &EuclideanR4).unwrap(),
            id,
            "one owner's path stays one entry across a Space change",
        );
        assert_eq!(db.generation(id), 2);

        touch_with_edit(&path);
        db.apply_events(owner, &modified(&path));
        assert_eq!(
            db.generation(id),
            3,
            "the reload must use ℝ⁴'s prelude; ℝ³'s no longer validates this source",
        );
    }
}
