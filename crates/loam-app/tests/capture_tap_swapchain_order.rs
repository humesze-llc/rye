//! What a capture tap can read out of the swapchain on the non-sRGB surface
//! path, where the frame's pixels reach it only through the composite pass.
//!
//! `Runner`'s frame loop needs a window, so these pin the GPU-side contract it
//! is ordered against rather than the loop itself: the swapchain still holds
//! whatever it held before the frame's passes until `CompositeNode` runs, and
//! once it has run the swapchain holds the scene's sRGB-encoded bits, which is
//! what the readback in `capture::read_texture_rgba` turns into PNG bytes. A
//! tap ordered ahead of the composite therefore captures the first state; the
//! runner puts a composite in front of both taps.
//!
//! The `gpu_probe` suffix is what CI's software-adapter job selects on.

use loam_render::composite::CompositeNode;
use wgpu::{
    Color, CommandEncoderDescriptor, Device, Extent3d, LoadOp, MapMode, Operations, Origin3d,
    PollType, Queue, RenderPassColorAttachment, RenderPassDescriptor, StoreOp, TexelCopyBufferInfo,
    TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};

/// Square side in texels. 64 * 4 bytes hits `COPY_BYTES_PER_ROW_ALIGNMENT`
/// exactly, so the readback needs no row unpadding.
const SIZE: u32 = 64;

/// The linear canvas format browser-WebGPU advertises, and the sRGB sibling
/// `RenderDevice` gives the offscreen scene target (`add_srgb_suffix`).
const SWAP_FORMAT: TextureFormat = TextureFormat::Bgra8Unorm;
const SCENE_FORMAT: TextureFormat = TextureFormat::Bgra8UnormSrgb;

/// Scene colour in linear space, per channel. Three distinct values so a
/// channel swap cannot pass, none of them near the transfer function's
/// linear segment.
const SCENE_LINEAR: [f64; 3] = [0.25, 0.5, 0.75];

/// Swapchain contents before any of the frame's passes touch it, in the
/// attachment's BGRA byte order. Opaque magenta: nothing the scene colour
/// could be confused with.
const UNWRITTEN_BGRA: [u8; 4] = [255, 0, 255, 255];

/// One 8-bit code point of slack, spent on the encode-decode-encode
/// round trip: the scene attachment stores encoded bytes, the composite's
/// sampler decodes them, and the shader encodes again.
const TOLERANCE: u8 = 1;

/// IEC 61966-2-1 linear -> sRGB, mirroring `composite.wgsl`'s `linear_to_srgb`.
/// Restated here so the expectation is a closed form rather than a second
/// reading of the shader under test.
fn linear_to_srgb(c: f64) -> f64 {
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn encoded_byte(linear: f64) -> u8 {
    (linear_to_srgb(linear) * 255.0).round() as u8
}

/// `SCENE_LINEAR` after encoding, in the attachments' BGRA byte order.
fn expected_bgra() -> [u8; 4] {
    [
        encoded_byte(SCENE_LINEAR[2]),
        encoded_byte(SCENE_LINEAR[1]),
        encoded_byte(SCENE_LINEAR[0]),
        255,
    ]
}

struct Probe {
    device: Device,
    queue: Queue,
    scene: Texture,
    scene_view: TextureView,
    swap: Texture,
    swap_view: TextureView,
    composite: CompositeNode,
}

impl Probe {
    fn new() -> Self {
        let (device, queue) = pollster::block_on(request_device()).expect("wgpu device");
        let scene = create_target(&device, SCENE_FORMAT, "scene");
        let swap = create_target(&device, SWAP_FORMAT, "swap");
        let scene_view = scene.create_view(&TextureViewDescriptor::default());
        let swap_view = swap.create_view(&TextureViewDescriptor::default());
        let mut composite = CompositeNode::new(&device, SWAP_FORMAT);
        composite.set_scene_view(&device, &scene_view);
        Self {
            device,
            queue,
            scene,
            scene_view,
            swap,
            swap_view,
            composite,
        }
    }

    /// Stand-in for `App::record`: fill a target with a flat colour. On the
    /// sRGB scene target the clear value is linear and the hardware encodes it
    /// on write, which is the same path a shader's linear output takes.
    fn fill(&self, view: &TextureView, color: Color) {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("capture-order probe fill"),
            });
        encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("capture-order probe fill"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(color),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        self.queue.submit(Some(encoder.finish()));
    }

    /// The state both attachments are in once the scene pass has run and
    /// before anything writes the swapchain.
    fn record_scene(&self) {
        self.fill(&self.swap_view, unwritten_color());
        self.fill(
            &self.scene_view,
            Color {
                r: SCENE_LINEAR[0],
                g: SCENE_LINEAR[1],
                b: SCENE_LINEAR[2],
                a: 1.0,
            },
        );
    }

    fn run_composite(&self) {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("capture-order probe composite"),
            });
        self.composite.run(&mut encoder, &self.swap_view);
        self.queue.submit(Some(encoder.finish()));
    }
}

/// `UNWRITTEN_BGRA` as the clear value that produces it. `Bgra8Unorm` stores
/// clear channels verbatim, so 1.0 and 0.0 land on 255 and 0.
fn unwritten_color() -> Color {
    Color {
        r: 1.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    }
}

fn create_target(device: &Device, format: TextureFormat, tag: &str) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: Some(&format!("capture-order probe {tag}")),
        size: Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

async fn request_device() -> Result<(Device, Queue), String> {
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
            label: Some("capture-tap-swapchain-order"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: Default::default(),
        })
        .await
        .map_err(|e| format!("request_device failed: {e}"))
}

/// Raw stored bytes, no format interpretation: the same thing
/// `capture::read_texture_rgba` copies out before its BGRA swizzle.
fn read_back(probe: &Probe, texture: &Texture) -> Vec<u8> {
    let bytes_per_row = SIZE * 4;
    let buffer = probe.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("capture-order probe readback"),
        size: (bytes_per_row * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = probe
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("capture-order probe readback"),
        });
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &buffer,
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
    probe.queue.submit(Some(encoder.finish()));
    let slice = buffer.slice(..);
    slice.map_async(MapMode::Read, |_| {});
    probe
        .device
        .poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("readback poll");
    let data = slice.get_mapped_range().to_vec();
    buffer.unmap();
    data
}

/// Every texel, so a pass that covered part of the target fails here rather
/// than passing on a lucky sample.
fn assert_every_texel(pixels: &[u8], expected: [u8; 4], tolerance: u8, what: &str) {
    for (index, texel) in pixels.chunks_exact(4).enumerate() {
        for channel in 0..4 {
            let delta = texel[channel].abs_diff(expected[channel]);
            assert!(
                delta <= tolerance,
                "{what}: texel {index} channel {channel} is {}, expected {} \
                 (delta {delta} > {tolerance}); full texel {texel:?}",
                texel[channel],
                expected[channel],
            );
        }
    }
}

/// The composite is the only pass that writes the swapchain on this path, so a
/// tap that submits and reads before it gets the surface's prior contents and
/// none of the frame. This is the failure the tap ordering exists to avoid.
#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn swapchain_holds_no_scene_pixels_until_the_composite_runs_gpu_probe() {
    let probe = Probe::new();
    probe.record_scene();

    let before = read_back(&probe, &probe.swap);
    assert_every_texel(&before, UNWRITTEN_BGRA, 0, "swapchain before the composite");

    probe.run_composite();
    let after = read_back(&probe, &probe.swap);
    assert_every_texel(
        &after,
        expected_bgra(),
        TOLERANCE,
        "swapchain after the composite",
    );
}

/// After the composite the swapchain holds sRGB-encoded bits, byte for byte
/// what the scene target stores, which is also what an sRGB swapchain holds on
/// the direct path. That equality is why a capture taken after the composite is
/// colour-correct and one taken from the raw linear values would not be: the
/// unencoded bytes for these channels are 64, 128 and 191.
#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn composite_leaves_the_swapchain_holding_srgb_encoded_channels_gpu_probe() {
    let probe = Probe::new();
    probe.record_scene();
    probe.run_composite();

    let expected = expected_bgra();
    assert_eq!(
        expected,
        [225, 188, 137, 255],
        "closed-form expectation drifted from the literals this test was written against"
    );
    assert_every_texel(
        &read_back(&probe, &probe.scene),
        expected,
        TOLERANCE,
        "scene target",
    );
    assert_every_texel(
        &read_back(&probe, &probe.swap),
        expected,
        TOLERANCE,
        "swapchain",
    );
}

/// The pre-egui tap composites into a swapchain the frame's own composite
/// overwrites a pass later. Running it twice has to land on the same bits, or
/// the diagnostic tap would be paid for with a corrupted presented frame.
#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn a_repeated_composite_leaves_the_swapchain_unchanged_gpu_probe() {
    let probe = Probe::new();
    probe.record_scene();
    probe.run_composite();
    let once = read_back(&probe, &probe.swap);
    probe.run_composite();
    let twice = read_back(&probe, &probe.swap);
    assert_eq!(once, twice, "a second composite changed the swapchain");
}
