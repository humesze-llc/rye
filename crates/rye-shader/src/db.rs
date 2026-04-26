//! Shader database with hot-reload support.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rye_asset::{AssetEvent, AssetEventKind};
use rye_math::WgslSpace;
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

struct Entry {
    path: PathBuf,
    module: ShaderModule,
    scene_source: Option<String>,
    /// Incremented on every successful (re)compile. Render code caches
    /// the generation it last built a pipeline against; mismatch means
    /// the pipeline needs rebuilding.
    generation: u64,
    /// Debug label for the module; reused on recompile.
    label: String,
}

/// Cache of compiled shaders, invalidated on asset events.
///
/// Hot-reload failures preserve the previous successful module: the
/// user sees stale output and a log line, not a crash. When a shader
/// file is removed, the entry is retained (stale) until a create or
/// modify event restores it.
pub struct ShaderDb {
    device: Device,
    entries: HashMap<ShaderId, Entry>,
    path_index: HashMap<PathBuf, ShaderId>,
    next_id: u32,
}

impl ShaderDb {
    /// Construct. `device` is cloned internally on recompile; wgpu's
    /// Device is cheap to clone (internally reference-counted).
    pub fn new(device: Device) -> Self {
        Self {
            device,
            entries: HashMap::new(),
            path_index: HashMap::new(),
            next_id: 0,
        }
    }

    /// Load a shader from disk, prepending the Space's WGSL prelude.
    ///
    /// Returns a [`ShaderId`] that remains valid across hot reloads of
    /// the same path. Call [`ShaderDb::load`] twice with the same path
    /// and you get the same ID and a fresh compilation.
    pub fn load<S: WgslSpace>(&mut self, path: impl AsRef<Path>, space: &S) -> Result<ShaderId> {
        self.load_inner(path, None, space)
    }

    /// Load a shader from disk with an additional scene module.
    ///
    /// The scene source is stored with the entry and reused on hot
    /// reloads of the shader file.
    pub fn load_with_scene<S: WgslSpace>(
        &mut self,
        path: impl AsRef<Path>,
        scene_source: &str,
        space: &S,
    ) -> Result<ShaderId> {
        self.load_inner(path, Some(scene_source), space)
    }

    /// Load a shader from disk for geodesic ray marching.
    ///
    /// Assembles four layers: Space prelude + scene SDF + geodesic march
    /// kernel ([`crate::GEODESIC_MARCH_KERNEL`]) + user shading WGSL.
    /// The kernel defines `rye_march_geodesic`, `rye_estimate_normal`, and
    /// `rye_safe_normalize` for the user shading fragment to call.
    ///
    /// The assembled scene + kernel is stored and reused on hot reloads of
    /// the user shading file.
    pub fn load_geodesic_scene<S: WgslSpace>(
        &mut self,
        path: impl AsRef<Path>,
        scene_source: &str,
        space: &S,
    ) -> Result<ShaderId> {
        let scene_with_kernel = format!(
            "{scene_source}// ---- rye geodesic march kernel ----\n{}",
            crate::GEODESIC_MARCH_KERNEL
        );
        self.load_inner(path, Some(&scene_with_kernel), space)
    }

    fn load_inner<S: WgslSpace>(
        &mut self,
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

        if let Some(&id) = self.path_index.get(&path) {
            let entry = self.entries.get_mut(&id).expect("path_index out of sync");
            entry.module = module;
            entry.scene_source = scene_source.map(str::to_owned);
            entry.generation += 1;
            entry.label = label;
            Ok(id)
        } else {
            let id = ShaderId(self.next_id);
            self.next_id += 1;
            self.entries.insert(
                id,
                Entry {
                    path: path.clone(),
                    module,
                    scene_source: scene_source.map(str::to_owned),
                    generation: 1,
                    label,
                },
            );
            self.path_index.insert(path, id);
            Ok(id)
        }
    }

    /// Borrow the current compiled module for `id`.
    pub fn module(&self, id: ShaderId) -> &ShaderModule {
        &self
            .entries
            .get(&id)
            .expect("unknown ShaderId — was it loaded by this ShaderDb?")
            .module
    }

    /// Generation counter for `id`. Increments on every successful
    /// (re)compile. Render code caches the value it last built a
    /// pipeline against and rebuilds on mismatch.
    pub fn generation(&self, id: ShaderId) -> u64 {
        self.entries.get(&id).map(|e| e.generation).unwrap_or(0)
    }

    /// Apply filesystem events, recompiling affected shaders.
    ///
    /// Compile errors are logged but do not remove the stale module;
    /// rendering continues against the last good compile until the
    /// source file is fixed.
    pub fn apply_events<S: WgslSpace>(&mut self, events: &[AssetEvent], space: &S) {
        for event in events {
            let canonical = match canonicalize(&event.path) {
                Ok(p) => p,
                // Removed files can't be canonicalized; fall back to the
                // raw path for lookup.
                Err(_) => event.path.clone(),
            };
            let Some(&id) = self.path_index.get(&canonical) else {
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

/// Concatenate the Space's WGSL prelude with the user shader source.
///
/// Extracted for testability — this is the hot-reloadable logic that
/// doesn't require a wgpu Device.
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
    out.push_str("// ---- rye-math Space prelude ----\n");
    out.push_str(space_wgsl);
    if !space_wgsl.ends_with('\n') {
        out.push('\n');
    }
    if let Some(scene_wgsl) = scene_wgsl {
        out.push_str("// ---- rye-sdf scene module ----\n");
        out.push_str(scene_wgsl);
        if !scene_wgsl.ends_with('\n') {
            out.push('\n');
        }
    }
    out.push_str("// ---- user shader ----\n");
    out.push_str(user_source);
    out
}

/// Parse and validate a complete WGSL module.
///
/// This is intentionally headless: `rye-shader` can reject a broken
/// [`rye_math::WgslSpace`] prelude or user shader without requiring a GPU
/// adapter, window, or render pipeline. `wgpu` still performs backend
/// validation when the module is created.
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
    use glam::Vec3;
    use rye_math::{
        BlendedSpace, EuclideanR3, HyperbolicH3, LinearBlendX, Space, SphericalS3, WgslSpace,
    };

    const ABI_PROBE: &str = r#"
@compute @workgroup_size(1)
fn main() {
    let a = vec3<f32>(0.1, 0.2, 0.3);
    let b = vec3<f32>(0.2, -0.1, 0.05);
    let v = vec3<f32>(0.01, 0.02, -0.03);
    _ = rye_distance(a, b);
    _ = rye_origin_distance(a);
    _ = rye_exp(a, v);
    _ = rye_log(a, b);
    _ = rye_parallel_transport(a, b, v);
    _ = RYE_MAX_ARC;
}
"#;

    #[test]
    fn assemble_includes_both_sources() {
        let s = assemble_source(
            "fn rye_distance(a: f32, b: f32) -> f32 { return 0.0; }",
            "@fragment fn main() {}",
        );
        assert!(s.contains("rye_distance"));
        assert!(s.contains("@fragment fn main"));
        assert!(s.find("rye_distance").unwrap() < s.find("@fragment fn main").unwrap());
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

    // Minimal stub of rye_scene_sdf so the kernel has something to call.
    const KERNEL_SCENE: &str = r#"
fn rye_scene_sdf(p: vec3<f32>) -> f32 {
    return rye_distance(p, vec3<f32>(0.0, 0.0, 0.0)) - 0.25;
}
"#;

    // Compute probe that exercises all three kernel entry points.
    const KERNEL_PROBE: &str = r#"
@compute @workgroup_size(1)
fn main() {
    let ro = vec3<f32>(0.0, 0.0, 2.0);
    let rd = vec3<f32>(0.0, 0.0, -1.0);
    _ = rye_march_geodesic(ro, rd, 0.2);
    _ = rye_estimate_normal(vec3<f32>(0.0, 0.0, 0.0), 0.2);
    _ = rye_safe_normalize(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0));
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
        distance: [f32; 4],
        exp_point: [f32; 4],
        log_vec: [f32; 4],
        transported: [f32; 4],
    }

    const GPU_PROBE: &str = r#"
struct Case {
    a: vec4<f32>,
    b: vec4<f32>,
    v: vec4<f32>,
};

struct ProbeOut {
    distance: vec4<f32>,
    exp_point: vec4<f32>,
    log_vec: vec4<f32>,
    transported: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> cases: array<Case>;
@group(0) @binding(1) var<storage, read_write> out: array<ProbeOut>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let c = cases[i];
    let a = c.a.xyz;
    let b = c.b.xyz;
    let v = c.v.xyz;
    out[i].distance = vec4<f32>(rye_distance(a, b), 0.0, 0.0, 0.0);
    out[i].exp_point = vec4<f32>(rye_exp(a, v), 0.0);
    out[i].log_vec = vec4<f32>(rye_log(a, b), 0.0);
    out[i].transported = vec4<f32>(rye_parallel_transport(a, b, v), 0.0);
}
"#;

    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing Space WGSL"]
    fn euclidean_space_gpu_probe_matches_cpu() {
        let space = EuclideanR3;
        let cases = [
            gpu_case(
                Vec3::new(0.1, 0.2, 0.3),
                Vec3::new(0.5, -0.1, 0.0),
                Vec3::new(0.01, 0.02, -0.03),
            ),
            gpu_case(
                Vec3::new(-2.0, 4.0, 0.5),
                Vec3::new(1.5, 0.25, -3.0),
                Vec3::new(0.5, -0.25, 0.125),
            ),
        ];
        let out = pollster::block_on(run_gpu_probe(&space, &cases)).expect("GPU probe");
        assert_probe_matches_cpu(&space, &cases, &out, 1e-5);
    }

    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing Space WGSL"]
    fn spherical_space_gpu_probe_matches_cpu() {
        let space = SphericalS3;
        // Points must be inside the upper hemisphere: |p|² < 1.
        let cases = [
            gpu_case(
                Vec3::new(0.1, 0.2, 0.3),
                Vec3::new(0.2, -0.1, 0.05),
                Vec3::new(0.01, 0.02, -0.03),
            ),
            gpu_case(
                Vec3::new(-0.3, 0.4, 0.1),
                Vec3::new(0.2, 0.3, -0.2),
                Vec3::new(0.02, -0.01, 0.015),
            ),
        ];
        let out = pollster::block_on(run_gpu_probe(&space, &cases)).expect("GPU probe");
        assert_probe_matches_cpu(&space, &cases, &out, 2e-4);
    }

    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing Space WGSL"]
    fn hyperbolic_space_gpu_probe_matches_cpu() {
        let space = HyperbolicH3;
        let cases = [
            gpu_case(
                Vec3::new(0.1, 0.2, 0.05),
                Vec3::new(0.2, -0.1, 0.08),
                Vec3::new(0.01, 0.02, -0.015),
            ),
            gpu_case(
                Vec3::new(-0.25, 0.1, 0.05),
                Vec3::new(0.15, 0.2, -0.1),
                Vec3::new(0.02, -0.015, 0.01),
            ),
        ];
        let out = pollster::block_on(run_gpu_probe(&space, &cases)).expect("GPU probe");
        assert_probe_matches_cpu(&space, &cases, &out, 2e-4);
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
                label: Some("rye-space-gpu-probe"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| format!("request_device failed: {e}"))?;

        let source = assemble_source(&space.wgsl_impl(), GPU_PROBE);
        validate_wgsl(&source).map_err(|e| e.to_string())?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rye-space-gpu-probe"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let input_size = std::mem::size_of_val(cases) as u64;
        let output_size = (cases.len() * std::mem::size_of::<GpuOut>()) as u64;

        let input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rye-space-gpu-probe-input"),
            size: input_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        input
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(cases));
        input.unmap();

        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rye-space-gpu-probe-output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rye-space-gpu-probe-staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rye-space-gpu-probe-bgl"),
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
            label: Some("rye-space-gpu-probe-bg"),
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
            label: Some("rye-space-gpu-probe-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rye-space-gpu-probe-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rye-space-gpu-probe-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rye-space-gpu-probe-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(cases.len() as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output_size);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).expect("map callback receiver should exist");
        });
        device
            .poll(wgpu::PollType::Wait)
            .map_err(|e| e.to_string())?;
        rx.recv()
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;

        let data = slice.get_mapped_range();
        let rows = bytemuck::cast_slice::<u8, GpuOut>(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(rows)
    }

    fn assert_probe_matches_cpu<S>(space: &S, cases: &[GpuCase], out: &[GpuOut], eps: f32)
    where
        S: Space<Point = Vec3, Vector = Vec3>,
    {
        assert_eq!(cases.len(), out.len());
        for (case, row) in cases.iter().zip(out) {
            let a = Vec3::from_array([case.a[0], case.a[1], case.a[2]]);
            let b = Vec3::from_array([case.b[0], case.b[1], case.b[2]]);
            let v = Vec3::from_array([case.v[0], case.v[1], case.v[2]]);

            assert_near(row.distance[0], space.distance(a, b), eps);
            assert_vec3_near(row.exp_point, space.exp(a, v), eps);
            assert_vec3_near(row.log_vec, space.log(a, b), eps);
            assert_vec3_near(row.transported, space.parallel_transport(a, b, v), eps);
        }
    }

    fn assert_vec3_near(actual: [f32; 4], expected: Vec3, eps: f32) {
        assert_near(actual[0], expected.x, eps);
        assert_near(actual[1], expected.y, eps);
        assert_near(actual[2], expected.z, eps);
    }

    fn assert_near(actual: f32, expected: f32, eps: f32) {
        assert!(
            (actual - expected).abs() <= eps,
            "actual {actual} differs from expected {expected} by more than {eps}",
        );
    }
}
