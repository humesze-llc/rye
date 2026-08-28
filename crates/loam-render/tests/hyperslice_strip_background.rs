//! What a filmstrip cell shows where the kernel draws nothing.
//!
//! `execute_strip` owns its attachment and its clear, and the kernel discards
//! on a miss and on the floor rather than painting either, so the clear is the
//! only thing standing in for the sky in a comparison grid. A cell of
//! `Color::BLACK` is the regression this pins.
//!
//! The `gpu_probe` suffix is what CI's software-adapter job selects on.

use loam_render::raymarch::{
    polytope_stub_sdfs_wgsl, BodyUniform, Hyperslice4DNode, HYPERSLICE_KERNEL_WGSL,
};
use loam_render::sky_ground::SKY_HORIZON;
use loam_render::Viewport;
use wgpu::*;

// 64 * 4 bytes per row hits `COPY_BYTES_PER_ROW_ALIGNMENT` exactly, so the
// readback needs no row unpadding.
const SIZE: [u32; 2] = [64, 64];

const CELLS: u32 = 4;

// Slack for the float-to-unorm round trip in the clear.
const TOLERANCE: u8 = 2;

// The demo's floor and nothing else, so every ray either misses or lands on
// the half-space: both are `discard` paths, so the whole grid is clear.
const FLOOR_SCENE_WGSL: &str = r#"
const LOAM_PRIM_HYPERSPHERE4D: u32 = 0u;
const LOAM_PRIM_HALFSPACE4D: u32 = 1u;
const LOAM_PRIM_OTHER: u32 = 255u;
struct LoamSceneHit { dist: f32, kind: u32 }
fn loam_scene_at(p: vec3<f32>) -> LoamSceneHit {
    return LoamSceneHit(p.y, LOAM_PRIM_HALFSPACE4D);
}
fn loam_scene_sdf(p: vec3<f32>) -> f32 {
    return loam_scene_at(p).dist;
}
fn loam_scene_max_t(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    if (rd.y > -1.0e-6) { return 1.0e9; }
    return -ro.y / rd.y;
}
"#;

fn unorm_byte(linear: f64) -> u8 {
    (linear * 255.0).round() as u8
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn every_filmstrip_cell_clears_to_the_sky_rather_than_black_gpu_probe() {
    let instance = Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("request_adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: Some("hyperslice-strip-background"),
        required_features: Features::empty(),
        required_limits: Limits::default(),
        memory_hints: MemoryHints::default(),
        trace: Trace::Off,
        experimental_features: Default::default(),
    }))
    .expect("request_device");

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("hyperslice-strip-background"),
        source: ShaderSource::Wgsl(
            format!(
                "{HYPERSLICE_KERNEL_WGSL}\n{}\n{FLOOR_SCENE_WGSL}",
                polytope_stub_sdfs_wgsl()
            )
            .into(),
        ),
    });
    let target = device.create_texture(&TextureDescriptor {
        label: Some("hyperslice-strip-background target"),
        size: Extent3d {
            width: SIZE[0],
            height: SIZE[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&TextureViewDescriptor::default());

    let mut node = Hyperslice4DNode::new(&device, TextureFormat::Rgba8Unorm, &module, 1);
    {
        let u = node.uniforms_mut();
        u.camera_pos = [0.0, 1.8, 6.0];
        u.resolution = [SIZE[0] as f32, SIZE[1] as f32];
    }
    let cells: Vec<(Viewport, f32, BodyUniform)> = Viewport::full(SIZE)
        .split_horizontal(CELLS)
        .into_iter()
        .map(|vp| (vp, 0.0, BodyUniform::default()))
        .collect();
    node.execute_strip(&device, &queue, &view, &cells)
        .expect("strip draw");

    let bytes_per_row = SIZE[0] * 4;
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("hyperslice-strip-background readback"),
        size: (bytes_per_row * SIZE[1]) as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("hyperslice-strip-background readback"),
    });
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &readback,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        Extent3d {
            width: SIZE[0],
            height: SIZE[1],
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));
    let slice = readback.slice(..);
    slice.map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("readback poll");
    let pixels = slice.get_mapped_range().to_vec();
    readback.unmap();

    let expected = [
        unorm_byte(SKY_HORIZON.r),
        unorm_byte(SKY_HORIZON.g),
        unorm_byte(SKY_HORIZON.b),
    ];
    assert_ne!(
        expected,
        [0, 0, 0],
        "the sky must not be black to begin with"
    );
    for (index, texel) in pixels.chunks_exact(4).enumerate() {
        for channel in 0..3 {
            assert!(
                texel[channel].abs_diff(expected[channel]) <= TOLERANCE,
                "texel {index} of the {CELLS}-cell strip is {texel:?}, not the \
                 sky's horizon {expected:?}"
            );
        }
    }
}
