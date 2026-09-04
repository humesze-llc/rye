//! The flat 3-torus through the real GPU chain. `loam-math` pins the quotient
//! on the CPU; what only execution can pin is that the unmodified march kernel
//! crosses the gluing by calling `loam_exp` alone.

use glam::Vec3;
use loam_math::{FlatTorus3, QuotientSpace, Space, WgslSpace};
use loam_scene::{Scene, SceneNode};
use loam_shader::{validate_wgsl, GEODESIC_MARCH_KERNEL};
use wgpu::util::DeviceExt;

// Unequal sides, so a swapped axis in the wrap reads as a wrong distance.
const CELL: Vec3 = Vec3::new(2.0, 2.6, 1.7);

// Spheres only: `Shape::Sphere` routes through `loam_distance` and is lattice-periodic.
fn room() -> Scene {
    Scene::new(
        SceneNode::sphere(Vec3::ZERO, 0.36)
            .union(SceneNode::sphere(Vec3::new(0.72, -0.85, 0.45), 0.2))
            .union(SceneNode::sphere(Vec3::new(-0.6, 0.8, -0.55), 0.26)),
    )
}

fn torus() -> FlatTorus3 {
    FlatTorus3::new(CELL)
}

// Per ray: drift of the marched wrap from the covering ray's wrap, and the hit arclength.
const PROBE_WGSL: &str = r#"
struct Ray {
    origin: vec3<f32>,
    _pad0: f32,
    direction: vec3<f32>,
    _pad1: f32,
};

@group(0) @binding(0) var<storage, read> rays: array<Ray>;
// x: |wrap(covering point) - wrap(marched point)|, negative on a miss.
// y: geodesic arclength of the hit, negative on a miss.
@group(0) @binding(1) var<storage, read_write> out: array<vec2<f32>>;

@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&rays) { return; }
    let ray = rays[i];
    let dir = loam_safe_normalize(ray.direction, vec3<f32>(0.0, 0.0, -1.0));
    let hit = loam_march_geodesic(ray.origin, dir, 1.0);
    if hit.w < 0.0 {
        out[i] = vec2<f32>(-1.0, -1.0);
        return;
    }
    // The kernel's parameter is in units of its finite-difference tangent (~1e-3
    // relative error), not arclength.
    let probe_eps = 1e-4;
    let speed = loam_distance(ray.origin, loam_exp(ray.origin, dir * probe_eps)) / probe_eps;
    let arclength = hit.w / speed;

    // Geodesics of E³/L lift to straight lines of the cover.
    let cover = ray.origin + dir * arclength;
    out[i] = vec2<f32>(length(loam_torus_wrap(cover) - loam_torus_wrap(hit.xyz)), arclength);
}
"#;

fn assemble_probe_source() -> String {
    let space = torus();
    format!(
        "{}\n{}\n{}\n{}",
        space.wgsl_impl(),
        room().to_wgsl(&space),
        GEODESIC_MARCH_KERNEL,
        PROBE_WGSL
    )
}

#[test]
fn the_unmodified_geodesic_kernel_accepts_the_quotient_prelude() {
    let source = assemble_probe_source();
    validate_wgsl(&source).expect("flat-torus march chain should validate");
    assert!(source.contains("fn loam_torus_wrap("));
    assert!(source.contains("fn loam_scene_sdf("));
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuRay {
    origin: [f32; 3],
    _pad0: f32,
    direction: [f32; 3],
    _pad1: f32,
}

// Fixed lattice origins and a spiral of directions, identical on every machine.
fn probe_rays(origin_offset: Vec3) -> Vec<GpuRay> {
    let mut rays = Vec::with_capacity(16 * 64);
    for oi in 0..16 {
        let s = oi as f32 / 16.0;
        let origin = Vec3::new(
            (s - 0.5) * CELL.x * 0.8,
            (s * 3.0 % 1.0 - 0.5) * CELL.y * 0.8,
            (s * 7.0 % 1.0 - 0.5) * CELL.z * 0.8,
        ) + origin_offset;
        for di in 0..64 {
            let t = (di as f32 + 0.5) / 64.0;
            let cos_theta = 1.0 - 2.0 * t;
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
            let phi = di as f32 * 2.399_963_2;
            rays.push(GpuRay {
                origin: origin.into(),
                _pad0: 0.0,
                direction: [sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta],
                _pad1: 0.0,
            });
        }
    }
    rays
}

async fn request_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("wgpu adapter");
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("flat-torus-room probe"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: Default::default(),
        })
        .await
        .expect("wgpu device")
}

fn dispatch<I: bytemuck::Pod, O: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &str,
    entry_point: &str,
    input: &[I],
) -> Vec<O> {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(entry_point),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("probe input"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_size = (input.len() * std::mem::size_of::<O>()) as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe staging"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: None,
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(entry_point),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(input.len().div_ceil(64) as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, output_size);
    queue.submit(Some(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    rx.recv().expect("map").expect("map succeeded");
    let results = bytemuck::cast_slice::<u8, O>(&staging.slice(..).get_mapped_range()).to_vec();
    staging.unmap();
    results
}

fn run_probe(device: &wgpu::Device, queue: &wgpu::Queue, rays: &[GpuRay]) -> Vec<[f32; 2]> {
    dispatch(device, queue, &assemble_probe_source(), "probe", rays)
}

const WRAP_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> lifts: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> wrapped: array<vec4<f32>>;

@compute @workgroup_size(64)
fn wrap_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&lifts) { return; }
    wrapped[i] = vec4<f32>(loam_torus_wrap(lifts[i].xyz), 0.0);
}
"#;

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn the_emitted_prelude_wraps_to_the_same_quotient_point_as_the_rust_impl_gpu_probe() {
    let (device, queue) = pollster::block_on(request_device());
    let space = torus();

    let half = 0.5 * CELL;
    let mut lifts: Vec<[f32; 4]> = Vec::new();
    for axis in 0..3 {
        for sign in [-1.0_f32, 1.0] {
            let edge = sign * half[axis];
            for coordinate in [
                edge,
                f32::from_bits(edge.to_bits() - 1),
                f32::from_bits(edge.to_bits() + 1),
            ] {
                for copy in -4..=4 {
                    let mut p = Vec3::new(0.31, -0.47, 0.19);
                    p[axis] = coordinate + copy as f32 * CELL[axis];
                    lifts.push([p.x, p.y, p.z, 0.0]);
                }
            }
        }
    }
    for i in 0..24 {
        for j in 0..24 {
            for k in 0..24 {
                let p = Vec3::new(i as f32 - 11.5, j as f32 - 11.5, k as f32 - 11.5) * 0.7;
                lifts.push([p.x, p.y, p.z, 0.0]);
            }
        }
    }

    let source = format!("{}\n{}", space.wgsl_impl(), WRAP_WGSL);
    let gpu: Vec<[f32; 4]> = dispatch(&device, &queue, &source, "wrap_probe", &lifts);

    let mut worst_off_face = 0.0_f32;
    let mut worst_quotient = 0.0_f32;
    let mut on_face = 0;
    for (lift, wrapped) in lifts.iter().zip(&gpu) {
        let p = Vec3::new(lift[0], lift[1], lift[2]);
        let cpu = space.wrap_to_domain(p).0;
        let gpu = Vec3::new(wrapped[0], wrapped[1], wrapped[2]);
        assert!(
            space.in_fundamental_domain(gpu),
            "the prelude put wrap({p:?}) = {gpu:?} outside the domain"
        );
        worst_quotient = worst_quotient.max(space.distance(cpu, gpu));

        if (half - cpu.abs()).min_element() > 1e-4 {
            worst_off_face = worst_off_face.max((cpu - gpu).abs().max_element());
        } else {
            on_face += 1;
        }
    }

    assert!(
        on_face > 0,
        "no lift landed on a face; the boundary case went untested"
    );
    // Measured: one ulp at the cell scale, 2.4e-7.
    assert!(
        worst_off_face < 1e-6,
        "away from the faces the prelude and the Rust wrap disagreed by \
         {worst_off_face}, past one ulp of the cell"
    );
    assert!(
        worst_quotient < 1e-5,
        "the prelude wrapped to a different point of the quotient, by {worst_quotient}"
    );
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn marched_position_agrees_with_the_covering_ray_gpu_probe() {
    let (device, queue) = pollster::block_on(request_device());
    let rays = probe_rays(Vec3::ZERO);
    let results = run_probe(&device, &queue, &rays);

    let hits = results.iter().filter(|r| r[1] >= 0.0).count();
    assert!(
        hits * 10 >= rays.len() * 9,
        "only {hits}/{} rays hit; the probe would be vacuous",
        rays.len()
    );

    let worst = results
        .iter()
        .filter(|r| r[1] >= 0.0)
        .fold(0.0_f32, |acc, r| acc.max(r[0]));
    // Measured worst over this fan: 1.4e-5.
    assert!(worst < 1e-4, "worst covering-ray disagreement was {worst}");
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn translating_by_a_lattice_vector_leaves_the_view_unchanged_gpu_probe() {
    let (device, queue) = pollster::block_on(request_device());
    let baseline = run_probe(&device, &queue, &probe_rays(Vec3::ZERO));

    for axis in 0..3 {
        let mut offset = Vec3::ZERO;
        offset[axis] = CELL[axis];
        let translated = run_probe(&device, &queue, &probe_rays(offset));

        let mut worst = 0.0_f32;
        for (index, (base, moved)) in baseline.iter().zip(&translated).enumerate() {
            assert_eq!(
                base[1] >= 0.0,
                moved[1] >= 0.0,
                "ray {index} changed hit/miss after a lattice translation on axis {axis}"
            );
            worst = worst.max((base[1] - moved[1]).abs() / base[1].max(1.0));
        }
        // Measured worst over this fan: 4.3e-5; a dropped axis is O(1).
        assert!(
            worst < 1e-3,
            "lattice translation on axis {axis} moved a hit by {worst} relative"
        );
    }
}
