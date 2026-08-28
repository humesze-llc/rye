//! Whether raster content composites over the analytic background at the right
//! depth, both from above the ground plane and from below it.
//!
//! A quad straddling `y = 0` is drawn after `SkyGroundNode` into one 64x64
//! attachment pair. The half on the camera's side of the plane must survive and
//! the half behind it must not, whichever side the camera is on. The control
//! arm turns the ground off through the same uniform: it isolates the ground's
//! depth as the thing doing the occluding, so a pass that merely failed to draw
//! cannot pass this.
//!
//! The `gpu_probe` suffix is what CI's software-adapter job selects on.

use glam::{Mat4, Vec3};
use loam_math::{EuclideanR3, Projection};
use loam_render::{
    DepthMode, FragmentShading, Ground, SkyGroundNode, SkyGroundUniforms, TriangleRasterNode,
    Viewport,
};
use loam_shape::TriangleMesh;
use wgpu::*;

// 64 * 4 bytes per row hits `COPY_BYTES_PER_ROW_ALIGNMENT` exactly, so the
// readback needs no row unpadding.
const SIZE: u32 = 64;

const TARGET_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

// Pure red against a ground whose channels all sit near 0.2 linear, so one
// channel separates the two without any tolerance argument.
const QUAD_COLOR: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

// The quad's sample points, one either side of the plane, and far enough from
// it that no rasterisation edge case lands on the seam.
const ABOVE: Vec3 = Vec3::new(0.0, 0.5, 0.0);
const BELOW: Vec3 = Vec3::new(0.0, -0.5, 0.0);

fn straddling_quad() -> TriangleMesh<3> {
    let mut mesh = TriangleMesh::<3>::default();
    for corner in [
        [-1.5_f32, -1.0, 0.0],
        [1.5, -1.0, 0.0],
        [1.5, 1.0, 0.0],
        [-1.5, 1.0, 0.0],
    ] {
        mesh.vertices.push(corner);
        mesh.colors.push(QUAD_COLOR);
    }
    mesh.indices.push([0, 1, 2]);
    mesh.indices.push([0, 2, 3]);
    mesh
}

fn view_proj(eye: Vec3) -> Mat4 {
    Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0)
        * Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y)
}

fn pixel_of(view_proj: Mat4, world: Vec3) -> (u32, u32) {
    let clip = view_proj * world.extend(1.0);
    let ndc = clip.truncate() / clip.w;
    let x = ((ndc.x * 0.5 + 0.5) * SIZE as f32) as u32;
    let y = ((0.5 - ndc.y * 0.5) * SIZE as f32) as u32;
    assert!(x < SIZE && y < SIZE, "{world} projects off the probe frame");
    (x, y)
}

fn texel(pixels: &[u8], (x, y): (u32, u32)) -> [u8; 4] {
    let i = ((y * SIZE + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

fn request_device() -> (Device, Queue) {
    let instance = Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("request_adapter");
    pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: Some("sky-ground-composite"),
        required_features: Features::empty(),
        required_limits: Limits::default(),
        memory_hints: MemoryHints::default(),
        trace: Trace::Off,
        experimental_features: Default::default(),
    }))
    .expect("request_device")
}

fn render(device: &Device, queue: &Queue, eye: Vec3, show_ground: bool) -> Vec<u8> {
    let color = device.create_texture(&TextureDescriptor {
        label: Some("sky-ground-composite color"),
        size: Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TARGET_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let depth = device.create_texture(&TextureDescriptor {
        label: Some("sky-ground-composite depth"),
        size: Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let color_view = color.create_view(&TextureViewDescriptor::default());
    let depth_view = depth.create_view(&TextureViewDescriptor::default());

    let background = SkyGroundNode::new(device, TARGET_FORMAT, DEPTH_FORMAT, 1);
    background.set_uniforms(
        queue,
        &SkyGroundUniforms::new(
            view_proj(eye),
            Viewport::full([SIZE, SIZE]),
            Ground {
                y: 0.0,
                dark: loam_render::sky_ground::GROUND_DARK_GREY,
                light: loam_render::sky_ground::GROUND_LIGHT_GREY,
                visible: show_ground,
            },
        ),
    );

    let mut quad = TriangleRasterNode::new(
        device,
        TARGET_FORMAT,
        DepthMode::ReadWrite {
            format: DEPTH_FORMAT,
        },
        FragmentShading::Flat,
        1,
    );
    quad.set_camera(queue, view_proj(eye));
    quad.upload::<EuclideanR3, 3>(device, queue, &straddling_quad(), &Projection::Identity);

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("sky-ground-composite"),
    });
    background.record(&mut encoder, &color_view, &depth_view, None);
    quad.record(&mut encoder, &color_view, Some(&depth_view), None);

    let bytes_per_row = SIZE * 4;
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("sky-ground-composite readback"),
        size: (bytes_per_row * SIZE) as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &color,
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
            width: SIZE,
            height: SIZE,
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
    let data = slice.get_mapped_range().to_vec();
    readback.unmap();
    data
}

// Linear red at 1.0 stores as 255 in a non-sRGB target, and every ground
// channel is near 0.2 linear, so these two thresholds cannot both be met by
// one surface.
const QUAD_RED_MIN: u8 = 200;
const GROUND_RED_MAX: u8 = 120;

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn ground_occludes_raster_content_behind_the_plane_gpu_probe() {
    let (device, queue) = request_device();
    // Above the plane the near half of the quad is the one over `y = 0`;
    // below it, the one under. In both the far half is behind the ground.
    for (eye, visible, occluded) in [
        (Vec3::new(0.0, 2.0, 4.0), ABOVE, BELOW),
        (Vec3::new(0.0, -2.0, 4.0), BELOW, ABOVE),
    ] {
        let vp = view_proj(eye);
        let with_ground = render(&device, &queue, eye, true);
        let seen = texel(&with_ground, pixel_of(vp, visible));
        let hidden = texel(&with_ground, pixel_of(vp, occluded));
        assert!(
            seen[0] >= QUAD_RED_MIN,
            "eye {eye}: the quad at {visible} is on the camera's side of the \
             ground and must survive it, got {seen:?}"
        );
        assert!(
            hidden[0] <= GROUND_RED_MAX,
            "eye {eye}: the quad at {occluded} is behind the ground and must \
             be cut by its depth, got {hidden:?}"
        );

        // Same frame, ground off: if the far half now draws, the cut above was
        // the ground's depth and not a lost draw or a clipped vertex.
        let without_ground = render(&device, &queue, eye, false);
        let uncovered = texel(&without_ground, pixel_of(vp, occluded));
        assert!(
            uncovered[0] >= QUAD_RED_MIN,
            "eye {eye}: with the ground off the quad at {occluded} must draw, \
             got {uncovered:?}; the occlusion test above proves nothing"
        );
    }
}
