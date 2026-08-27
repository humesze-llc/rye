//! Smoke pin for the `TextRenderer` chain: construct, queue, record.
//!
//! The atlas and layout halves are pure-CPU and pinned in the crate's own unit
//! tests. Only the wgpu half needs a device, and only it is `#[ignore]`d. The
//! `gpu_probe` suffix is what CI's software-adapter job selects on.

use loam_text::TextRenderer;
use wgpu::{Device, Queue, TextureFormat, TextureView};

const TARGET_FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;
const VIEWPORT: [f32; 2] = [1280.0, 720.0];

fn draw_hud_frame(
    device: &Device,
    queue: &Queue,
    view: &TextureView,
    font_bytes: &[u8],
) -> anyhow::Result<()> {
    let mut text = TextRenderer::new(device, queue, TARGET_FORMAT, font_bytes, 48.0, 1)?;
    text.queue("fps 240", [16.0, 16.0], 32.0, [1.0, 1.0, 1.0, 1.0]);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hud-chain-smoke frame"),
    });
    text.record(device, queue, &mut encoder, view, VIEWPORT);
    queue.submit(Some(encoder.finish()));
    Ok(())
}

/// First readable TTF from the well-known system font directories. The crate
/// ships no font asset, so a machine without one panics naming the paths it
/// probed: libtest reports an early return as a pass, which would leave this
/// test green while asserting nothing.
fn system_font() -> Vec<u8> {
    const CANDIDATES: &[&str] = &[
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ];
    CANDIDATES
        .iter()
        .find_map(|p| std::fs::read(p).ok())
        .unwrap_or_else(|| panic!("no readable system font; probed {CANDIDATES:?}"))
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
            label: Some("hud-chain-smoke"),
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
fn hud_frame_renders_into_an_offscreen_target_gpu_probe() {
    let font_bytes = system_font();
    let (device, queue) = pollster::block_on(request_device()).expect("wgpu device");

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hud-chain-smoke target"),
        size: wgpu::Extent3d {
            width: VIEWPORT[0] as u32,
            height: VIEWPORT[1] as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    draw_hud_frame(&device, &queue, &view, &font_bytes).expect("HUD frame should render");
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("queue drain");
}
