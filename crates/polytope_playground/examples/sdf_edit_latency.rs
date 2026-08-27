//! Run: `cargo run -p polytope_playground --release --example sdf_edit_latency`
//! Headless: an adapter and a device, no surface and no window.

use std::time::Instant;

use glam::Vec3;
use loam_math::{EuclideanR3, WgslSpace};
use loam_scene::edit::{self, EditValue, NodePath, Param, SceneEdit};
use loam_scene::{Scene, SceneNode};
use loam_shader::GEODESIC_MARCH_KERNEL;
use wgpu::*;

// Enough that a pipeline cache warming up shows as a gap between the median
// and the maximum rather than hiding in the mean.
const FRAMES: usize = 120;

const FRAME_BUDGET_MS: f64 = 1000.0 / 60.0;

// Kept in step with `sdf::boot_scene` by shape rather than by import: an
// example must not force a demo's internals public.
fn boot_scene() -> Scene {
    Scene::new(
        SceneNode::sphere(Vec3::new(-0.35, 0.0, 0.0), 0.45)
            .smooth_union(SceneNode::box_(Vec3::splat(0.4)), 0.15)
            .union(SceneNode::plane(Vec3::Y, -0.8))
            .subtract(SceneNode::sphere(Vec3::new(0.3, 0.35, 0.35), 0.35)),
    )
}

// Its own text, not the editor's: what is being timed is the size and shape of
// the assembled module, and the fragment body is a constant in both.
const SHADING: &str = r#"
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

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let ndc = frag_pos.xy / u.resolution * 2.0 - vec2<f32>(1.0, 1.0);
    let dir = loam_safe_normalize(u.camera_forward + u.camera_right * ndc.x, vec3<f32>(0.0, 0.0, -1.0));
    let hit = loam_march_geodesic(u.camera_pos, dir, 1.0);
    if hit.w < 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return vec4<f32>(loam_estimate_normal(hit.xyz, 1.0) * 0.5 + 0.5, 1.0);
}
"#;

fn assemble(scene: &Scene) -> String {
    format!(
        "{prelude}\n{emit}\n{kernel}\n{SHADING}",
        prelude = EuclideanR3.wgsl_impl(),
        emit = scene.to_wgsl(&EuclideanR3),
        kernel = GEODESIC_MARCH_KERNEL,
    )
}

fn main() {
    let (device, _queue) = pollster::block_on(request_device());
    let mut scene = boot_scene();
    let path: NodePath = "root.1".parse().expect("the carve sphere");

    let mut emit = Vec::with_capacity(FRAMES);
    let mut module = Vec::with_capacity(FRAMES);
    let mut pipeline = Vec::with_capacity(FRAMES);

    for frame in 0..FRAMES {
        // One radius edit per frame, which is what a held slider produces.
        let radius = 0.2 + 0.15 * (frame as f32 / FRAMES as f32);
        let edit = SceneEdit::Set {
            path: path.clone(),
            param: Param::Radius,
            value: EditValue::Scalar(radius),
        };
        assert!(
            edit::apply(&mut scene, &edit).expect("the drag applies"),
            "frame {frame} moved nothing",
        );

        let started = Instant::now();
        let source = assemble(&scene);
        emit.push(started.elapsed().as_secs_f64() * 1000.0);

        let started = Instant::now();
        let compiled = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("sdf edit latency"),
            source: ShaderSource::Wgsl(source.into()),
        });
        module.push(started.elapsed().as_secs_f64() * 1000.0);

        let started = Instant::now();
        let node = loam_render::GeodesicRayMarchNode::from_module(
            &device,
            TextureFormat::Rgba8UnormSrgb,
            &compiled,
            1,
        );
        pipeline.push(started.elapsed().as_secs_f64() * 1000.0);
        drop(node);
    }

    let total: Vec<f64> = (0..FRAMES)
        .map(|i| emit[i] + module[i] + pipeline[i])
        .collect();
    println!("{FRAMES} single-parameter edits on the boot scene, milliseconds:");
    for (label, mut samples) in [
        ("emit", emit),
        ("shader module", module),
        ("pipeline", pipeline),
        ("total per edit", total),
    ] {
        samples.sort_by(f64::total_cmp);
        println!(
            "  {label:<14} p50 {:.3}  p95 {:.3}  max {:.3}",
            samples[FRAMES / 2],
            samples[FRAMES * 95 / 100],
            samples[FRAMES - 1],
        );
        if label == "total per edit" {
            println!(
                "  p95 is {:.0}% of a {FRAME_BUDGET_MS:.1}ms frame at 60Hz",
                100.0 * samples[FRAMES * 95 / 100] / FRAME_BUDGET_MS,
            );
        }
    }
}

async fn request_device() -> (Device, Queue) {
    let instance = Instance::default();
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("wgpu adapter");
    println!("adapter: {:?}", adapter.get_info());
    adapter
        .request_device(&DeviceDescriptor {
            label: Some("sdf edit latency"),
            required_features: Features::empty(),
            required_limits: Limits::default(),
            memory_hints: MemoryHints::default(),
            trace: Trace::Off,
            experimental_features: Default::default(),
        })
        .await
        .expect("wgpu device")
}
