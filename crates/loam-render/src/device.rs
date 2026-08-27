//! sRGB surface: the scene pass draws through the target's own sRGB view
//! ([`RenderDevice::msaa_view`], else the `begin_frame` view), and under MSAA
//! [`RenderDevice::resolve_scene_to_swap`] resolves it into the `begin_frame`
//! view, sRGB at both ends. The UI pass paints single-sampled through the
//! swapchain's non-sRGB reinterpretation ([`RenderDevice::create_ui_swap_view`])
//! so egui blends in gamma space, or through the swapchain's own view where
//! the adapter forbids the reinterpretation (see `ui_target_formats`).
//!
//! Non-sRGB surface: MSAA off, scene and UI both draw into [`OffscreenTarget`],
//! and [`crate::composite::CompositeNode`] gamma-encodes that into the
//! swapchain view. The offscreen target takes the surface format's sRGB
//! sibling, so the UI blends in linear space. A format with no sibling
//! (`Rgba16Float`) keeps its own format, which lands egui on the
//! gamma-framebuffer path whose already-encoded output the composite encodes
//! again.

use anyhow::Result;
use std::sync::Arc;
use wgpu::*;
use winit::window::Window;

pub struct SurfaceBundle {
    pub surface: Surface<'static>,
    pub config: SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
}

/// Written by the scene pass alone, through [`MsaaTarget::view`], and resolved
/// into the swapchain by [`RenderDevice::resolve_scene_to_swap`] before the UI
/// pass runs.
pub struct MsaaTarget {
    // Keeps the GPU allocation alive for the lifetime of `view`.
    #[allow(dead_code)]
    texture: Texture,
    pub view: TextureView,
}

/// Target for the non-sRGB-surface path: scene and UI render here, then
/// [`crate::composite::CompositeNode`] gamma-encodes into the linear
/// swapchain. Carries the surface format's sRGB sibling, or the surface
/// format itself where it has none.
pub struct OffscreenTarget {
    // Keeps the GPU allocation alive for the lifetime of `view`.
    #[allow(dead_code)]
    texture: Texture,
    pub view: TextureView,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct UiTargetFormats {
    pub ui_format: TextureFormat,
    /// Registered as the swapchain's view format and used for the UI pass's
    /// swapchain view. `None` means the UI uses the swapchain's own format.
    pub swap_view_format: Option<TextureFormat>,
}

// wgpu-core gates a swapchain texture's view formats behind the
// `SURFACE_VIEW_FORMATS` downlevel flag (checked at `surface.configure`).
// GL/WebGL never advertises it and Vulkan advertises it only with
// `VK_KHR_swapchain_mutable_format`, so an unguarded entry is a validation
// failure, not a degraded render.
//
// The swapchain is the only target reinterpreted: the multisampled
// attachment's resolve has to stay sRGB at both ends (see
// [`RenderDevice::resolve_scene_to_swap`]), so it registers nothing.
fn ui_target_formats(surface_format: TextureFormat, downlevel: DownlevelFlags) -> UiTargetFormats {
    // Composite path: the UI paints into the offscreen scene texture and
    // `CompositeNode` encodes gamma afterwards, so nothing is reinterpreted.
    // `add_srgb_suffix` is the identity for formats without an sRGB sibling.
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

// Split out of `RenderDevice::from_surface` so the `view_formats`
// registration, the field wgpu validates against the downlevel flags, is
// checkable without a device.
fn surface_configuration(
    format: TextureFormat,
    size: winit::dpi::PhysicalSize<u32>,
    alpha_mode: CompositeAlphaMode,
    ui_targets: UiTargetFormats,
) -> SurfaceConfiguration {
    SurfaceConfiguration {
        // COPY_SRC keeps headless screenshot readback open at negligible cost.
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

// `None` requests the target's own format, the only legal request when
// nothing was registered as a reinterpretation.
fn ui_view_descriptor(ui_view_format: Option<TextureFormat>) -> TextureViewDescriptor<'static> {
    TextureViewDescriptor {
        format: ui_view_format,
        ..Default::default()
    }
}

// The target's own format, never the UI pass's non-sRGB twin. Both ends of
// the MSAA resolve take it, which is what keeps the averaging linear.
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
    /// `Some` only when the adapter advertised the timestamp features and we
    /// requested them. The runner owns the per-frame lifecycle.
    pub gpu_timer: Option<crate::gpu_timer::GpuTimer>,
    /// Present only on the non-sRGB swapchain path; `None` on native.
    scene_target: Option<OffscreenTarget>,
    composite: Option<crate::composite::CompositeNode>,
    /// The surface format's sRGB sibling where it has one, else the surface
    /// format. `None` when `scene_target` is `None`.
    scene_format: Option<TextureFormat>,
    /// Cached so the `vsync` command validates without re-querying
    /// `get_surface_capabilities`.
    present_modes: Vec<PresentMode>,
    ui_targets: UiTargetFormats,
}

impl RenderDevice {
    /// `requested_msaa_samples` of 1 means no MSAA; the effective count (see
    /// [`RenderDevice::sample_count`]) falls back to 1 if unsupported.
    pub async fn new(window: Arc<Window>, requested_msaa_samples: u32) -> Result<Self> {
        let instance = Instance::default();
        let surface = instance.create_surface(window.clone())?;
        let size = window.inner_size();
        Self::from_surface(instance, surface, size, requested_msaa_samples).await
    }

    /// [`Self::new`] variant taking a wgpu [`Surface`] directly, for callers
    /// without a winit [`Window`]. Keeps `loam-render` decoupled from `web-sys`:
    /// the caller owns surface creation. `size` is in surface pixels.
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

        // wgpu 27 splits timestamps into TIMESTAMP_QUERY (pass-attached) and
        // TIMESTAMP_QUERY_INSIDE_ENCODERS (free-floating write_timestamp, our path
        // since App::render owns its passes). Some browser builds advertise only
        // the former, where write_timestamp then panics; require both or skip the
        // timer entirely.
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

        // Prefer `Opaque` over the browser-advertised `PreMultiplied`, which
        // composites alpha < 1 shader output against the page and darkens it on
        // non-white backgrounds. Fall back to whatever is advertised first.
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

        // sRGB swapchains render directly; linear ones (browser-WebGPU) need an
        // offscreen scene texture plus a gamma-encoding composite pass.
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

    /// No-ops on a zero dimension, the minimized case wgpu rejects. Recreates the
    /// MSAA and offscreen-scene textures, rewiring the composite bind group, when
    /// those paths are active.
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

    /// Returns the wgpu surface error directly so callers can branch on `Lost`,
    /// `Outdated` and `Timeout`.
    pub fn begin_frame(
        &self,
    ) -> std::result::Result<(SurfaceTexture, TextureView), wgpu::SurfaceError> {
        let frame = self.surface_bundle.surface.get_current_texture()?;
        let view = frame.texture.create_view(&scene_view_descriptor());
        Ok((frame, view))
    }

    /// 1 = off. Below the requested count where the adapter could not honor it.
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    pub fn present_mode(&self) -> PresentMode {
        self.surface_bundle.config.present_mode
    }

    /// A mode outside this list is a wgpu validation error at
    /// `surface.configure`.
    pub fn supported_present_modes(&self) -> &[PresentMode] {
        &self.present_modes
    }

    /// `Err(mode)` and no change where the adapter does not advertise it;
    /// otherwise reconfigures in place for the next `begin_frame`. `Fifo` is the
    /// only browser-WebGPU mode.
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

    /// `None` when MSAA is off.
    pub fn msaa_view(&self) -> Option<&TextureView> {
        self.msaa_target.as_ref().map(|t| &t.view)
    }

    /// `None` on native. Scene-pass target priority: `msaa_view()`, then
    /// `scene_view()`, then the swapchain view directly. The UI pass paints here
    /// too on this path, since the composite is what reaches the swapchain.
    pub fn scene_view(&self) -> Option<&TextureView> {
        self.scene_target.as_ref().map(|t| &t.view)
    }

    /// On the composite path the offscreen texture's format, not the surface
    /// format. Pipeline constructors take this rather than reading the surface.
    pub fn target_format(&self) -> TextureFormat {
        self.scene_format
            .unwrap_or(self.surface_bundle.config.format)
    }

    /// Direct-to-swapchain: the swapchain format with the sRGB suffix stripped,
    /// so egui blends in the gamma space its feathering assumes; the sRGB format
    /// instead where the adapter's downlevel capabilities forbid reinterpreting
    /// swapchain views.
    ///
    /// Composite: the scene texture's format, the surface format's sRGB sibling
    /// where it has one, so blending is linear. Where there is no sibling the
    /// format is not sRGB and egui takes its gamma-framebuffer path into a target
    /// the composite encodes again; see this module's doc.
    pub fn ui_format(&self) -> TextureFormat {
        self.ui_targets.ui_format
    }

    /// The non-sRGB reinterpretation where the adapter supports it, the texture's
    /// own format otherwise. Always a color attachment and never a
    /// `resolve_target`, since the UI pass is single-sampled on every path.
    /// Unused on the composite path, where the UI never touches the swapchain.
    pub fn create_ui_swap_view(&self, frame: &SurfaceTexture) -> TextureView {
        frame
            .texture
            .create_view(&ui_view_descriptor(self.ui_targets.swap_view_format))
    }

    /// No-op with MSAA off. Caller submits the encoder.
    ///
    /// Its own pass because `App::record` owns the scene passes and a resolve can
    /// only ride on the frame's last write to the attachment. `StoreOp::Store`
    /// rather than `Discard` for the same reason: the attachment's load ops
    /// belong to the app, which may carry content across frames.
    ///
    /// Both ends carry the target's own sRGB format. A resolve averages samples
    /// in the encoding its views name, so routing either end through the UI
    /// pass's gamma twin would average sRGB-encoded values and darken every
    /// antialiased edge; wgpu also rejects a resolve whose target format differs
    /// from the attachment view's (`MismatchedResolveTextureFormat`), which is
    /// why the attachment registers no reinterpretation (see `ui_target_formats`).
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

    /// Caller submits the encoder. No-op when `scene_view()` is `None`.
    pub fn composite_to_swap(&self, encoder: &mut wgpu::CommandEncoder, swap_view: &TextureView) {
        if let Some(composite) = self.composite.as_ref() {
            composite.run(encoder, swap_view);
        }
    }

    /// Forces the driver to compile the composite PSO during setup rather than on
    /// the first real frame. No-op on the native path.
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

// `RENDER_ATTACHMENT` to draw into, `TEXTURE_BINDING` for the composite's
// sample.
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

// The composite path has no multisampled input, so a non-sRGB surface renders
// single-sampled and `MsaaTarget`, hence `RenderDevice::msaa_view`, stays
// `None` there.
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

// No `view_formats`: the scene pass is the attachment's only consumer and its
// resolve stays sRGB at both ends, so the gamma twin is never viewable here.
// Split out so that registration is checkable without a device.
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
    fn composite_path_never_requests_multisampling() {
        for surface in SURFACES {
            for requested in [1u32, 2, 4, 8, 16, u32::MAX] {
                let effective = surface_msaa_request(surface, requested);
                let case = format!("{surface:?} {requested}");
                if surface.is_srgb() {
                    assert_eq!(effective, requested, "{case}");
                } else {
                    assert_eq!(effective, 1, "{case}");
                }
            }
        }
    }

    #[test]
    fn descriptors_register_only_the_sanctioned_reinterpretation() {
        for surface in SURFACES {
            for downlevel in DOWNLEVELS {
                let case = format!("{surface:?} {downlevel:?}");
                // Derived from the flags rather than from `ui_target_formats`,
                // so a descriptor cannot drift in step with the decision.
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
                // Derived from the flags rather than from `ui_target_formats`,
                // so a view descriptor cannot drift in step with the decision.
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
