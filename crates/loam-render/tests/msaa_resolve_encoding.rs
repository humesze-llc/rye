//! A 4x attachment with one dark and three bright samples, resolved through a
//! view of its own format: an sRGB view averages linearized samples; a non-sRGB
//! view averages encoded bytes and lands ~34 code points darker.

use wgpu::*;

// 64 * 4 bytes per row meets `COPY_BYTES_PER_ROW_ALIGNMENT`, so no row unpadding.
const SIZE: u32 = 64;

const SAMPLES: u32 = 4;

// Samples 1..3 take the draw; sample 0 keeps the clear.
const BRIGHT_SAMPLE_MASK: u64 = 0b1110;

// Linear mean of the four samples: three at 1.0, one at 0.0.
const LINEAR_MEAN: f64 = 0.75;

// (0 + 3 * 255) / 4.
const ENCODED_MEAN_BYTE: u8 = 191;

// Slack for the encode-average-encode round trip through 8-bit storage.
const TOLERANCE: u8 = 2;

// IEC 61966-2-1 linear -> sRGB.
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

fn resolve_through(format: TextureFormat) -> Vec<u8> {
    let instance = Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("request_adapter");
    let flags = adapter.get_texture_format_features(format).flags;
    assert!(
        flags.contains(TextureFormatFeatureFlags::MULTISAMPLE_X4),
        "{format:?}: adapter cannot render 4x multisampled"
    );
    assert!(
        flags.contains(TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE),
        "{format:?}: adapter cannot resolve"
    );
    let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: Some("msaa-resolve-encoding"),
        required_features: Features::empty(),
        required_limits: Limits::default(),
        memory_hints: MemoryHints::default(),
        trace: Trace::Off,
        experimental_features: Default::default(),
    }))
    .expect("request_device");

    let msaa = create_target(&device, format, SAMPLES, TextureUsages::RENDER_ATTACHMENT);
    let resolved = create_target(
        &device,
        format,
        1,
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
    );
    let msaa_view = msaa.create_view(&TextureViewDescriptor::default());
    let resolved_view = resolved.create_view(&TextureViewDescriptor::default());
    let pipeline = bright_pipeline(&device, format);

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("msaa-resolve-encoding"),
    });
    {
        let mut scene = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("msaa-resolve-encoding scene"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &msaa_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        scene.set_pipeline(&pipeline);
        scene.draw(0..3, 0..1);
    }
    {
        let _resolve = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("msaa-resolve-encoding resolve"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &msaa_view,
                depth_slice: None,
                resolve_target: Some(&resolved_view),
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
    }
    queue.submit(Some(encoder.finish()));
    read_back(&device, &queue, &resolved)
}

fn create_target(
    device: &Device,
    format: TextureFormat,
    sample_count: u32,
    usage: TextureUsages,
) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: Some("msaa-resolve-encoding target"),
        size: Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

// Writes 1.0 into the samples `BRIGHT_SAMPLE_MASK` selects.
fn bright_pipeline(device: &Device, format: TextureFormat) -> RenderPipeline {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("msaa-resolve-encoding"),
        source: ShaderSource::Wgsl(
            r#"
@vertex
fn vs(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vertex << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vertex & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
"#
            .into(),
        ),
    });
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("msaa-resolve-encoding"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("msaa-resolve-encoding"),
        layout: Some(&layout),
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(ColorTargetState {
                format,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState {
            count: SAMPLES,
            mask: BRIGHT_SAMPLE_MASK,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
}

fn read_back(device: &Device, queue: &Queue, texture: &Texture) -> Vec<u8> {
    let bytes_per_row = SIZE * 4;
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("msaa-resolve-encoding readback"),
        size: (bytes_per_row * SIZE) as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("msaa-resolve-encoding readback"),
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
    queue.submit(Some(encoder.finish()));
    let slice = buffer.slice(..);
    slice.map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("readback poll");
    let data = slice.get_mapped_range().to_vec();
    buffer.unmap();
    data
}

fn assert_every_texel(pixels: &[u8], grey: u8, what: &str) {
    for (index, texel) in pixels.chunks_exact(4).enumerate() {
        for channel in 0..3 {
            let delta = texel[channel].abs_diff(grey);
            assert!(
                delta <= TOLERANCE,
                "{what}: texel {index} channel {channel} is {}, expected {grey} \
                 (delta {delta} > {TOLERANCE}); full texel {texel:?}",
                texel[channel],
            );
        }
        assert_eq!(texel[3], 255, "{what}: texel {index} alpha");
    }
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn srgb_view_resolve_averages_linearized_samples_gpu_probe() {
    let resolved = resolve_through(TextureFormat::Bgra8UnormSrgb);
    assert_every_texel(&resolved, encoded_byte(LINEAR_MEAN), "sRGB resolve");
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn non_srgb_view_resolve_averages_encoded_samples_gpu_probe() {
    let resolved = resolve_through(TextureFormat::Bgra8Unorm);
    assert_every_texel(&resolved, ENCODED_MEAN_BYTE, "gamma resolve");
    assert!(
        encoded_byte(LINEAR_MEAN).abs_diff(ENCODED_MEAN_BYTE) > 8 * TOLERANCE,
        "the two resolves must be distinguishable for this pair to mean anything"
    );
}
