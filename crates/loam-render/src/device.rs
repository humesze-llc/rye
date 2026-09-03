//! sRGB surface: the scene draws and resolves through sRGB views; the UI
//! paints through the swapchain's non-sRGB twin. Non-sRGB surface: MSAA off,
//! scene and UI draw into [`OffscreenTarget`] and the composite encodes them.

use anyhow::Result;
use std::sync::Arc;
use wgpu::*;
use winit::window::Window;

pub struct SurfaceBundle {
    pub surface: Surface<'static>,
    pub config: SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
}

pub struct MsaaTarget {
    // Keeps the GPU allocation alive for the lifetime of `view`.
    #[allow(dead_code)]
    texture: Texture,
    pub view: TextureView,
}

pub struct OffscreenTarget {
    // Keeps the GPU allocation alive for the lifetime of `view`.
    #[allow(dead_code)]
    texture: Texture,
    pub view: TextureView,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct UiTargetFormats {
    pub ui_format: TextureFormat,
    /// `None`: the UI paints through the swapchain's own format.
    pub swap_view_format: Option<TextureFormat>,
}

// A view format outside `SURFACE_VIEW_FORMATS` fails validation at `surface.configure`.
fn ui_target_formats(surface_format: TextureFormat, downlevel: DownlevelFlags) -> UiTargetFormats {
    if !surface_format.is_srgb() {
        return UiTargetFormats {
            ui_format: surface_format.add_srgb_suffix(),
            swap_view_format: None,
        };
    }
    if !downlevel.contains(DownlevelFlags::SURFACE_VIEW_FORMATS) {
        return UiTargetFormats {
            ui_format: surface_format,
            swap_view_format: None,
        };
    }
    let gamma = surface_format.remove_srgb_suffix();
    UiTargetFormats {
        ui_format: gamma,
        swap_view_format: Some(gamma),
    }
}

fn surface_configuration(
    format: TextureFormat,
    size: winit::dpi::PhysicalSize<u32>,
    alpha_mode: CompositeAlphaMode,
    ui_targets: UiTargetFormats,
) -> SurfaceConfiguration {
    SurfaceConfiguration {
        // COPY_SRC for headless screenshot readback.
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        format,
        width: size.width,
        height: size.height,
        present_mode: PresentMode::Fifo,
        alpha_mode,
        view_formats: ui_targets.swap_view_format.into_iter().collect(),
        desired_maximum_frame_latency: 2,
    }
}

fn ui_view_descriptor(ui_view_format: Option<TextureFormat>) -> TextureViewDescriptor<'static> {
    TextureViewDescriptor {
        format: ui_view_format,
        ..Default::default()
    }
}

// Both ends of the MSAA resolve take the target's own format; a non-sRGB view
// would average encoded bytes.
fn scene_view_descriptor() -> TextureViewDescriptor<'static> {
    TextureViewDescriptor::default()
}

pub struct RenderDevice {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub surface_bundle: SurfaceBundle,
    sample_count: u32,
    msaa_target: Option<MsaaTarget>,
    pub gpu_timer: Option<crate::gpu_timer::GpuTimer>,
    scene_target: Option<OffscreenTarget>,
    composite: Option<crate::composite::CompositeNode>,
    scene_format: Option<TextureFormat>,
    present_modes: Vec<PresentMode>,
    ui_targets: UiTargetFormats,
}

impl RenderDevice {
    pub async fn new(window: Arc<Window>, requested_msaa_samples: u32) -> Result<Self> {
        let instance = Instance::default();
        let surface = instance.create_surface(window.clone())?;
        let size = window.inner_size();
        Self::from_surface(instance, surface, size, requested_msaa_samples).await
    }

    pub async fn from_surface(
        instance: Instance,
        surface: Surface<'static>,
        size: winit::dpi::PhysicalSize<u32>,
        requested_msaa_samples: u32,
    ) -> Result<Self> {
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
            })
            .await?;

        // Some browsers advertise TIMESTAMP_QUERY without INSIDE_ENCODERS, where
        // `write_timestamp` panics.
        let needed = Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let timestamps_ok = adapter.features().contains(needed);
        let required_features = if timestamps_ok {
            needed
        } else {
            Features::empty()
        };
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("Loam Device"),
                required_features,
                required_limits: Limits::default(),
                memory_hints: MemoryHints::default(),
                trace: Trace::Off,
                experimental_features: Default::default(),
            })
            .await?;
        if timestamps_ok {
            tracing::info!("GPU timestamp queries enabled (TIMESTAMP_QUERY + INSIDE_ENCODERS)");
        } else {
            tracing::info!(
                "GPU timestamp queries unavailable (adapter features: {:?})",
                adapter.features()
            );
        }

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        tracing::info!(
            "surface picked format={format:?} (advertised={:?})",
            caps.formats
        );

        // Browser `PreMultiplied` composites alpha < 1 against the page.
        let alpha_mode = caps
            .alpha_modes
            .iter()
            .copied()
            .find(|m| *m == CompositeAlphaMode::Opaque)
            .unwrap_or(caps.alpha_modes[0]);

        let ui_targets = ui_target_formats(format, adapter.get_downlevel_capabilities().flags);
        if format.is_srgb() && ui_targets.swap_view_format.is_none() {
            tracing::warn!(
                "adapter lacks SURFACE_VIEW_FORMATS; UI blends in linear space and \
                 egui feathering will look thin on hairlines"
            );
        }
        let config = surface_configuration(format, size, alpha_mode, ui_targets);

        surface.configure(&device, &config);

        let needs_composite = !format.is_srgb();
        let effective_msaa = surface_msaa_request(format, requested_msaa_samples);
        if effective_msaa < requested_msaa_samples {
            tracing::warn!(
                "MSAA={requested_msaa_samples}x ignored: composite pass for sRGB \
                 gamma encoding (browser-WebGPU linear surface) is incompatible \
                 with MSAA in v1; falling back to sample_count=1",
            );
        }

        let sample_count = negotiate_sample_count(&adapter, format, effective_msaa);
        let msaa_target = (sample_count > 1)
            .then(|| create_msaa_target(&device, format, size.width, size.height, sample_count));

        let (scene_target, composite, scene_format) = if needs_composite {
            let scene_fmt = format.add_srgb_suffix();
            tracing::info!(
                "non-sRGB surface; rendering through offscreen scene target {scene_fmt:?} \
                 with composite pass to {format:?} swapchain"
            );
            let scene = create_scene_target(&device, scene_fmt, size.width, size.height);
            let mut comp = crate::composite::CompositeNode::new(&device, format);
            comp.set_scene_view(&device, &scene.view);
            (Some(scene), Some(comp), Some(scene_fmt))
        } else {
            (None, None, None)
        };

        let gpu_timer = crate::gpu_timer::GpuTimer::new(&device, &queue);

        let present_modes = caps.present_modes.clone();
        tracing::info!("surface present modes advertised: {present_modes:?}");

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface_bundle: SurfaceBundle {
                surface,
                config,
                size,
            },
            sample_count,
            msaa_target,
            gpu_timer,
            scene_target,
            composite,
            scene_format,
            present_modes,
            ui_targets,
        })
    }

    /// No-op on a zero dimension, which wgpu rejects.
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.surface_bundle.size = new_size;
        self.surface_bundle.config.width = new_size.width;
        self.surface_bundle.config.height = new_size.height;
        self.surface_bundle
            .surface
            .configure(&self.device, &self.surface_bundle.config);
        if self.sample_count > 1 {
            self.msaa_target = Some(create_msaa_target(
                &self.device,
                self.surface_bundle.config.format,
                new_size.width,
                new_size.height,
                self.sample_count,
            ));
        }
        if let (Some(scene_fmt), Some(composite)) = (self.scene_format, self.composite.as_mut()) {
            let scene =
                create_scene_target(&self.device, scene_fmt, new_size.width, new_size.height);
            composite.set_scene_view(&self.device, &scene.view);
            self.scene_target = Some(scene);
        }
    }

    pub fn begin_frame(
        &self,
    ) -> std::result::Result<(SurfaceTexture, TextureView), wgpu::SurfaceError> {
        let frame = self.surface_bundle.surface.get_current_texture()?;
        let view = frame.texture.create_view(&scene_view_descriptor());
        Ok((frame, view))
    }

    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    pub fn present_mode(&self) -> PresentMode {
        self.surface_bundle.config.present_mode
    }

    pub fn supported_present_modes(&self) -> &[PresentMode] {
        &self.present_modes
    }

    /// `Fifo` is the only browser-WebGPU mode.
    pub fn set_present_mode(&mut self, mode: PresentMode) -> std::result::Result<(), PresentMode> {
        if !self.present_modes.contains(&mode) {
            return Err(mode);
        }
        if self.surface_bundle.config.present_mode == mode {
            return Ok(());
        }
        self.surface_bundle.config.present_mode = mode;
        self.surface_bundle
            .surface
            .configure(&self.device, &self.surface_bundle.config);
        tracing::info!("surface present_mode -> {mode:?}");
        Ok(())
    }

    pub fn msaa_view(&self) -> Option<&TextureView> {
        self.msaa_target.as_ref().map(|t| &t.view)
    }

    /// Scene-pass target priority: `msaa_view`, then this, then the swapchain view.
    pub fn scene_view(&self) -> Option<&TextureView> {
        self.scene_target.as_ref().map(|t| &t.view)
    }

    /// Pipeline constructors take this, not the surface format.
    pub fn target_format(&self) -> TextureFormat {
        self.scene_format
            .unwrap_or(self.surface_bundle.config.format)
    }

    /// The format of every view the UI pass renders into.
    pub fn ui_format(&self) -> TextureFormat {
        self.ui_targets.ui_format
    }

    /// Never a `resolve_target`; the UI pass is single-sampled on every path.
    pub fn create_ui_swap_view(&self, frame: &SurfaceTexture) -> TextureView {
        frame
            .texture
            .create_view(&ui_view_descriptor(self.ui_targets.swap_view_format))
    }

    /// No-op with MSAA off; both ends take the target's own sRGB format so the
    /// resolve averages linear samples.
    pub fn resolve_scene_to_swap(&self, encoder: &mut CommandEncoder, swap_view: &TextureView) {
        let Some(msaa) = self.msaa_target.as_ref() else {
            return;
        };
        // Nothing to draw: the resolve runs at pass end.
        let _resolve_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("loam-render::scene-msaa-resolve"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &msaa.view,
                depth_slice: None,
                resolve_target: Some(swap_view),
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

    pub fn composite_to_swap(&self, encoder: &mut wgpu::CommandEncoder, swap_view: &TextureView) {
        if let Some(composite) = self.composite.as_ref() {
            composite.run(encoder, swap_view);
        }
    }

    /// Compiles the composite PSO at setup rather than on the first frame.
    pub fn warm_composite(&self) {
        if self.composite.is_none() {
            return;
        }
        let format = self.surface_bundle.config.format;
        let dummy = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("loam-render::composite::warm dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let dummy_view = dummy.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("loam-render::composite::warm encoder"),
            });
        self.composite_to_swap(&mut encoder, &dummy_view);
        self.queue.submit(Some(encoder.finish()));
    }
}

fn create_scene_target(
    device: &Device,
    format: TextureFormat,
    width: u32,
    height: u32,
) -> OffscreenTarget {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("loam-render::scene_target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    OffscreenTarget { texture, view }
}

// The composite pass has no multisampled input.
fn surface_msaa_request(surface_format: TextureFormat, requested: u32) -> u32 {
    if surface_format.is_srgb() {
        requested
    } else {
        1
    }
}

fn negotiate_sample_count(adapter: &Adapter, format: TextureFormat, requested: u32) -> u32 {
    if requested <= 1 {
        return 1;
    }
    let features = adapter.get_texture_format_features(format);
    let flags = features.flags;
    for count in [16u32, 8, 4, 2] {
        if count > requested {
            continue;
        }
        let supported = match count {
            2 => flags.contains(TextureFormatFeatureFlags::MULTISAMPLE_X2),
            4 => flags.contains(TextureFormatFeatureFlags::MULTISAMPLE_X4),
            8 => flags.contains(TextureFormatFeatureFlags::MULTISAMPLE_X8),
            16 => flags.contains(TextureFormatFeatureFlags::MULTISAMPLE_X16),
            _ => false,
        };
        if supported {
            if count != requested {
                tracing::warn!(
                    "requested MSAA {requested}x not supported on {format:?}; falling back to {count}x"
                );
            }
            return count;
        }
    }
    tracing::warn!("no multisampled count supported on {format:?}; MSAA disabled");
    1
}

fn msaa_texture_descriptor(
    format: TextureFormat,
    width: u32,
    height: u32,
    sample_count: u32,
) -> TextureDescriptor<'static> {
    TextureDescriptor {
        label: Some("loam-render::msaa-color"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    }
}

fn create_msaa_target(
    device: &Device,
    format: TextureFormat,
    width: u32,
    height: u32,
    sample_count: u32,
) -> MsaaTarget {
    let texture = device.create_texture(&msaa_texture_descriptor(
        format,
        width,
        height,
        sample_count,
    ));
    let view = texture.create_view(&scene_view_descriptor());
    MsaaTarget { texture, view }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOTH: DownlevelFlags =
        DownlevelFlags::SURFACE_VIEW_FORMATS.union(DownlevelFlags::VIEW_FORMATS);

    const SURFACES: [TextureFormat; 4] = [
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba16Float,
    ];

    const DOWNLEVELS: [DownlevelFlags; 5] = [
        DownlevelFlags::empty(),
        DownlevelFlags::SURFACE_VIEW_FORMATS,
        DownlevelFlags::VIEW_FORMATS,
        BOTH,
        DownlevelFlags::all(),
    ];

    const SIZE: winit::dpi::PhysicalSize<u32> = winit::dpi::PhysicalSize {
        width: 800,
        height: 600,
    };

    #[test]
    fn srgb_surface_registers_the_gamma_twin_only_with_surface_view_formats() {
        let srgb = TextureFormat::Bgra8UnormSrgb;
        let gamma = TextureFormat::Bgra8Unorm;
        let table = [
            (DownlevelFlags::empty(), None),
            (DownlevelFlags::SURFACE_VIEW_FORMATS, Some(gamma)),
            (DownlevelFlags::VIEW_FORMATS, None),
            (BOTH, Some(gamma)),
            (DownlevelFlags::all(), Some(gamma)),
        ];
        for (downlevel, expected) in table {
            let targets = ui_target_formats(srgb, downlevel);
            assert_eq!(targets.swap_view_format, expected, "{downlevel:?}");
            assert_eq!(targets.ui_format, expected.unwrap_or(srgb), "{downlevel:?}");
        }
    }

    #[test]
    fn composite_path_registers_no_view_formats_and_targets_the_scene_format() {
        let table = [
            (TextureFormat::Bgra8Unorm, TextureFormat::Bgra8UnormSrgb),
            (TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb),
            (TextureFormat::Rgba16Float, TextureFormat::Rgba16Float),
        ];
        for (surface, scene) in table {
            for downlevel in [DownlevelFlags::empty(), DownlevelFlags::all()] {
                let targets = ui_target_formats(surface, downlevel);
                assert_eq!(targets.swap_view_format, None, "{surface:?} {downlevel:?}");
                assert_eq!(targets.ui_format, scene, "{surface:?} {downlevel:?}");
            }
        }
    }

    #[test]
    fn ui_format_matches_every_view_the_ui_pass_renders_into() {
        for surface in SURFACES {
            for downlevel in DOWNLEVELS {
                let targets = ui_target_formats(surface, downlevel);
                let case = format!("{surface:?} {downlevel:?}");
                if surface.is_srgb() {
                    let swap = targets.swap_view_format.unwrap_or(surface);
                    assert_eq!(targets.ui_format, swap, "{case}");
                } else {
                    assert_eq!(targets.ui_format, surface.add_srgb_suffix(), "{case}");
                }
            }
        }
    }

    #[test]
    fn scene_msaa_resolve_runs_through_the_srgb_view_pair() {
        assert_eq!(
            scene_view_descriptor().format,
            None,
            "scene views take their target's own format"
        );
        for surface in SURFACES {
            for sample_count in [2u32, 4, 8, 16] {
                let case = format!("{surface:?} {sample_count}x");
                let msaa = msaa_texture_descriptor(surface, SIZE.width, SIZE.height, sample_count);
                assert_eq!(msaa.format, surface, "{case}");
                assert!(msaa.view_formats.is_empty(), "{case}");
            }
        }
        for surface in SURFACES {
            for downlevel in DOWNLEVELS {
                let case = format!("{surface:?} {downlevel:?}");
                let targets = ui_target_formats(surface, downlevel);
                let Some(gamma) = targets.swap_view_format else {
                    continue;
                };
                assert!(!gamma.is_srgb(), "{case}");
                assert_eq!(
                    ui_view_descriptor(targets.swap_view_format).format,
                    Some(gamma),
                    "the UI pass keeps the twin: {case}"
                );
                assert_ne!(
                    scene_view_descriptor().format,
                    Some(gamma),
                    "the scene resolve declines it: {case}"
                );
            }
        }
    }

    #[test]
    fn descriptors_register_only_the_sanctioned_reinterpretation() {
        for surface in SURFACES {
            for downlevel in DOWNLEVELS {
                let case = format!("{surface:?} {downlevel:?}");
                let sanctioned: Vec<TextureFormat> = if surface.is_srgb()
                    && downlevel.contains(DownlevelFlags::SURFACE_VIEW_FORMATS)
                {
                    vec![surface.remove_srgb_suffix()]
                } else {
                    vec![]
                };
                let targets = ui_target_formats(surface, downlevel);
                let config =
                    surface_configuration(surface, SIZE, CompositeAlphaMode::Opaque, targets);
                assert_eq!(config.view_formats, sanctioned, "swapchain: {case}");
            }
        }
    }

    #[test]
    fn ui_view_requests_match_their_target_registration_in_both_arms() {
        for surface in SURFACES {
            for downlevel in DOWNLEVELS {
                let case = format!("{surface:?} {downlevel:?}");
                let expected = (surface.is_srgb()
                    && downlevel.contains(DownlevelFlags::SURFACE_VIEW_FORMATS))
                .then(|| surface.remove_srgb_suffix());
                let targets = ui_target_formats(surface, downlevel);

                let swap_request = ui_view_descriptor(targets.swap_view_format).format;
                assert_eq!(swap_request, expected, "swapchain request: {case}");
                let config =
                    surface_configuration(surface, SIZE, CompositeAlphaMode::Opaque, targets);
                assert_eq!(
                    config.view_formats,
                    swap_request.into_iter().collect::<Vec<_>>(),
                    "swapchain registration: {case}"
                );
            }
        }
    }
}
