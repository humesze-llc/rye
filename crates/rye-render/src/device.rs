//! Window surface + wgpu adapter/device acquisition.
//!
//! [`RenderDevice::new`] picks a high-performance adapter and an sRGB
//! surface format when available; resize is handled by
//! [`RenderDevice::resize`]. [`RenderDevice::begin_frame`] returns the
//! per-frame `(SurfaceTexture, TextureView)` pair the render graph
//! draws into.

use anyhow::Result;
use std::sync::Arc;
use wgpu::*;
use winit::window::Window;

/// Surface + per-frame configuration. Owned by [`RenderDevice`]; held
/// out as a struct so resize-aware code can read the current size and
/// format without poking at private fields.
pub struct SurfaceBundle {
    pub surface: Surface<'static>,
    pub config: SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
}

/// All wgpu state the engine carries: shared `Instance`, the chosen
/// `Adapter`, the logical `Device`, the submission `Queue`, and the
/// current surface bundle. One per app; cloning this is not supported.
pub struct RenderDevice {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub surface_bundle: SurfaceBundle,
}

impl RenderDevice {
    /// Acquire a surface for `window`, request a high-performance
    /// adapter, and configure the surface for sRGB rendering when
    /// the platform supports it. Async because both adapter and
    /// device creation are async on every wgpu backend.
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let instance = Instance::default();

        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("Rye Device"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
                memory_hints: MemoryHints::default(),
                trace: Trace::Off,
                // wgpu v27 requires opting in to experimental features explicitly;
                // we don't use any.
                experimental_features: Default::default(),
            })
            .await?;

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = SurfaceConfiguration {
            // COPY_SRC keeps texture readback open for headless screenshot tools
            // and any future capture path; cost is negligible vs. the headache of
            // re-creating the surface to enable it later.
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            format,
            width: size.width,
            height: size.height,
            present_mode: PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

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
        })
    }

    /// Reconfigure the surface for the new window size. No-ops on
    /// width or height of zero (the minimized-window case wgpu rejects
    /// outright).
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
    }

    /// Acquire the next swapchain texture and its default view, ready
    /// for a render pass. Returns the wgpu surface error directly so
    /// callers can branch on `Lost` / `Outdated` / `Timeout` without
    /// extra wrapping.
    pub fn begin_frame(
        &self,
    ) -> std::result::Result<(SurfaceTexture, TextureView), wgpu::SurfaceError> {
        let frame = self.surface_bundle.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        Ok((frame, view))
    }
}
