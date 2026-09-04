//! The 3D raymarch chain, `loam_scene::Scene` -> WGSL -> `ShaderDb` ->
//! `RayMarchNode`: nothing else in the workspace instantiates it. Headless
//! except the `gpu_probe`.

use glam::Vec3;
use loam_math::{EuclideanR3, WgslSpace};
use loam_render::{GeodesicRayMarchNode, RayMarchNode, RayMarchUniforms};
use loam_scene::{Scene, SceneNode};
use loam_shader::{validate_wgsl, ShaderDb, GEODESIC_MARCH_KERNEL};
use wgpu::{Device, TextureFormat};

fn probe_scene() -> Scene {
    Scene::new(SceneNode::sphere(Vec3::ZERO, 0.5).union(SceneNode::plane(Vec3::Y, -0.5)))
}

// Mirror of `RayMarchUniforms`; WGSL's `vec3` alignment matches the hand padding.
const UNIFORMS_WGSL: &str = r#"
struct RayMarchUniforms {
    camera_pos: vec3<f32>,
    camera_forward: vec3<f32>,
    camera_right: vec3<f32>,
    camera_up: vec3<f32>,
    fov_y_tan: f32,
    resolution: vec2<f32>,
    time: f32,
    tick: f32,
    params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: RayMarchUniforms;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vid) / 2) * 4.0 - 1.0;
    let y = f32(i32(vid) & 1) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn ray_direction(pos: vec4<f32>) -> vec3<f32> {
    let ndc = pos.xy / u.resolution * 2.0 - vec2<f32>(1.0, 1.0);
    return u.camera_forward
        + u.camera_right * ndc.x * u.fov_y_tan
        + u.camera_up * ndc.y * u.fov_y_tan;
}
"#;

// Calls every symbol `GEODESIC_MARCH_KERNEL` exports.
const GEODESIC_SHADING_WGSL: &str = r#"
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dir = loam_safe_normalize(ray_direction(pos), vec3<f32>(0.0, 0.0, -1.0));
    let hit = loam_march_geodesic(u.camera_pos, dir, 1.0);
    if hit.w < 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let n = loam_estimate_normal(hit.xyz, 1.0);
    return vec4<f32>(n * 0.5 + vec3<f32>(0.5), 1.0);
}
"#;

const SCENE_SHADING_WGSL: &str = r#"
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = u.camera_pos + ray_direction(pos);
    let lit = step(loam_scene_sdf(p), 0.0);
    return vec4<f32>(vec3<f32>(lit), 1.0);
}
"#;

fn geodesic_user_shader() -> String {
    format!("{UNIFORMS_WGSL}{GEODESIC_SHADING_WGSL}")
}

fn scene_user_shader() -> String {
    format!("{UNIFORMS_WGSL}{SCENE_SHADING_WGSL}")
}

// `ShaderDb`'s assembler is crate-private; this repeats its ordering.
fn assemble(space_wgsl: &str, scene_wgsl: &str, user_wgsl: &str) -> String {
    format!("{space_wgsl}\n{scene_wgsl}\n{user_wgsl}")
}

#[test]
fn geodesic_chain_assembles_into_valid_wgsl() {
    let scene_wgsl = probe_scene().to_wgsl(&EuclideanR3);
    let source = assemble(
        &EuclideanR3.wgsl_impl(),
        &format!("{scene_wgsl}{GEODESIC_MARCH_KERNEL}"),
        &geodesic_user_shader(),
    );
    validate_wgsl(&source).expect("geodesic raymarch chain should validate");
}

#[test]
fn scene_chain_assembles_into_valid_wgsl() {
    let scene_wgsl = probe_scene().to_wgsl(&EuclideanR3);
    let source = assemble(&EuclideanR3.wgsl_impl(), &scene_wgsl, &scene_user_shader());
    validate_wgsl(&source).expect("scene raymarch chain should validate");
}

fn build_geodesic_node(
    device: &Device,
    surface_format: TextureFormat,
    shader_path: &std::path::Path,
    scene: &Scene,
) -> anyhow::Result<GeodesicRayMarchNode> {
    let mut db = ShaderDb::new(device.clone());
    let id = db.load_geodesic_scene(
        ShaderDb::ROOT_OWNER,
        shader_path,
        &scene.to_wgsl(&EuclideanR3),
        &EuclideanR3,
    )?;
    Ok(GeodesicRayMarchNode::from_module(
        device,
        surface_format,
        db.module(id),
        1,
    ))
}

fn build_raymarch_node(
    device: &Device,
    surface_format: TextureFormat,
    shader_path: &std::path::Path,
    scene: &Scene,
) -> anyhow::Result<RayMarchNode> {
    let mut db = ShaderDb::new(device.clone());
    let id = db.load_with_scene(
        ShaderDb::ROOT_OWNER,
        shader_path,
        &scene.to_wgsl(&EuclideanR3),
        &EuclideanR3,
    )?;
    Ok(RayMarchNode::new(device, surface_format, db.module(id), 1))
}

fn frame_uniforms() -> RayMarchUniforms {
    RayMarchUniforms {
        resolution: [640.0, 360.0],
        ..Default::default()
    }
}

async fn request_device() -> Result<(wgpu::Device, wgpu::Queue), String> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| format!("request_adapter failed: {e}"))?;
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("raymarch-chain-smoke"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: Default::default(),
        })
        .await
        .map_err(|e| format!("request_device failed: {e}"))
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn both_nodes_build_from_a_real_shader_db_gpu_probe() {
    let (device, queue) = pollster::block_on(request_device()).expect("wgpu device");
    let surface_format = TextureFormat::Rgba8UnormSrgb;
    let scene = probe_scene();

    let dir = tempfile::tempdir().expect("tempdir");

    let geodesic_path = dir.path().join("geodesic.wgsl");
    std::fs::write(&geodesic_path, geodesic_user_shader()).expect("write geodesic shader");
    let mut geodesic = build_geodesic_node(&device, surface_format, &geodesic_path, &scene)
        .expect("geodesic chain should build a pipeline");
    geodesic.set_uniforms(&queue, frame_uniforms());

    let scene_path = dir.path().join("scene.wgsl");
    std::fs::write(&scene_path, scene_user_shader()).expect("write scene shader");
    let mut plain = build_raymarch_node(&device, surface_format, &scene_path, &scene)
        .expect("scene chain should build a pipeline");
    plain.set_uniforms(&queue, frame_uniforms());
}
