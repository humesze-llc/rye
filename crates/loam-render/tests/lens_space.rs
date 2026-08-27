//! The lens space prelude through the real GPU chain.
//!
//! `loam-math` pins the quotient's algebra on the CPU. What cannot be pinned
//! there is that the emitted WGSL is the same arithmetic: if the two halves
//! drift, the simulation walks through one manifold and the shader draws
//! another, and neither half fails on its own.
//!
//! The `gpu_probe`-suffixed test needs an adapter and is ignored by default,
//! matching `flat_torus_room.rs`; run with `--include-ignored`.

use glam::Vec4;
use loam_math::{IsometryGroup, LensSpace, QuotientSpace, Space};
use loam_shader::validate_wgsl;
use wgpu::util::DeviceExt;

const P: u32 = 5;
const Q: u32 = 2;

// A centre off both degenerate circles, so its deck orbit is p distinct points
// and the distance minimisation has something to choose between.
fn probe_centre() -> Vec4 {
    Vec4::new(0.62, 0.18, 0.55, 0.52).normalize()
}

// A fixed spiral over S³ plus the wedge walls and an ulp either side of them,
// which is where the wrap's correction step decides. Constructed rather than
// sampled, so the probe set is identical on every machine.
fn probe_lifts() -> Vec<Vec4> {
    let mut lifts = Vec::new();
    for i in 0..192 {
        let t = i as f32 / 192.0;
        lifts.push(
            Vec4::new(
                (t * 11.0).cos(),
                (t * 7.0).sin(),
                (t * 5.0 + 1.0).cos() * 0.7,
                (t * 3.0 + 2.0).sin() * 0.7,
            )
            .normalize(),
        );
    }
    for wedge in 0..P as i32 {
        let wall = (wedge as f32 + 0.5) * std::f32::consts::TAU / P as f32;
        for angle in [
            wall,
            f32::from_bits(wall.to_bits() - 1),
            f32::from_bits(wall.to_bits() + 1),
        ] {
            let (sin, cos) = angle.sin_cos();
            lifts.push(Vec4::new(cos * 0.8, sin * 0.8, 0.5, 0.331_662_5).normalize());
        }
    }
    lifts
}

const PROBE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> lifts: array<vec4<f32>>;
// xyz w: the wrapped lift. The second vector carries the quotient distance to
// LENS_PROBE_CENTRE in x and the deck power the minimisation picked in y.
@group(0) @binding(1) var<storage, read_write> out: array<array<vec4<f32>, 2>>;

@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&lifts) { return; }
    let x = lifts[i];
    out[i][0] = loam_lens_wrap(x);
    out[i][1] = vec4<f32>(
        loam_lens_distance(x, LENS_PROBE_CENTRE),
        f32(loam_lens_nearest_power(x, LENS_PROBE_CENTRE)),
        0.0,
        0.0,
    );
}
"#;

fn probe_source() -> String {
    let centre = probe_centre();
    format!(
        "{}\nconst LENS_PROBE_CENTRE = vec4<f32>({:?}, {:?}, {:?}, {:?});\n{}",
        LensSpace::new(P, Q).wgsl_prelude(),
        centre.x,
        centre.y,
        centre.z,
        centre.w,
        PROBE_WGSL
    )
}

#[test]
fn the_emitted_prelude_validates_and_exports_the_names_a_marcher_calls() {
    let source = probe_source();
    validate_wgsl(&source).expect("the lens prelude should validate");
    for symbol in [
        "fn loam_lens_apply(",
        "fn loam_lens_wedge_offset(",
        "fn loam_lens_wrap(",
        "fn loam_lens_distance(",
        "fn loam_lens_nearest_power(",
    ] {
        assert!(source.contains(symbol), "the prelude lost {symbol}");
    }
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn the_emitted_prelude_wraps_and_measures_as_the_rust_impl_does_gpu_probe() {
    let (device, queue) = pollster::block_on(request_device());
    let lens = LensSpace::new(P, Q);
    let centre = probe_centre();
    let lifts = probe_lifts();

    let results: Vec<[[f32; 4]; 2]> = dispatch(
        &device,
        &queue,
        &probe_source(),
        "probe",
        &lifts.iter().map(|p| p.to_array()).collect::<Vec<_>>(),
    );

    // Half a wedge wall's worth of argument, four decades over the rotation's
    // own rounding: outside it the representative is determined.
    let wall_band = 1e-4;
    let half_wedge = std::f32::consts::PI / P as f32;
    for (lift, result) in lifts.iter().zip(&results) {
        let gpu_wrapped = Vec4::from_array(result[0]);
        let (cpu_wrapped, _) = lens.wrap_to_domain(*lift);
        assert!(
            lens.distance(gpu_wrapped, cpu_wrapped) < 1e-4,
            "the GPU wrapped {lift:?} to a different point of the quotient"
        );
        if (lift.y.atan2(lift.x).abs() - half_wedge).abs() > wall_band {
            assert!(
                (gpu_wrapped - cpu_wrapped).length() < 1e-4,
                "the GPU wrapped {lift:?} to a different lift: {gpu_wrapped:?} against \
                 {cpu_wrapped:?}"
            );
        }

        let cpu_distance = lens.distance(*lift, centre);
        assert!(
            (result[1][0] - cpu_distance).abs() < 1e-4,
            "the GPU measured {} to the centre against the CPU's {cpu_distance}",
            result[1][0]
        );
        // The power is an integer, so this is exact or it is wrong: it selects
        // the lift the closed-form surface normal is taken against.
        let power = result[1][1] as i32;
        let selected = lens.iso_apply(lens.deck(power), centre);
        assert!(
            (lens.distance(*lift, selected) - cpu_distance).abs() < 1e-5,
            "the GPU picked deck power {power}, which is not the nearest lift"
        );
    }
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
            label: Some("lens-space probe"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: Default::default(),
        })
        .await
        .expect("wgpu device")
}

// One dispatch of a `@workgroup_size(64)` entry point over `input` at binding
// 0, reading one `O` per element back from binding 1.
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
