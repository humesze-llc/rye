use anyhow::Result;
use std::sync::Arc;
use wgpu::*;
use winit::window::Window;

pub struct SurfaceBundle {
    pub surface: Surface<'static>,
    pub config: SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
}

pub struct RenderDevice {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub surface_bundle: SurfaceBundle,
}

impl RenderDevice {
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
            .request_device(
                &DeviceDescriptor {
                    label: Some("Mars Device"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    memory_hints: MemoryHints::default(), // NEW in v26
                    trace: Trace::Off,                    // NEW in v26
                }
            )
            .await?;

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
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
            surface_bundle: SurfaceBundle { surface, config, size },
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 { return; }
        self.surface_bundle.size = new_size;
        self.surface_bundle.config.width = new_size.width;
        self.surface_bundle.config.height = new_size.height;
        self.surface_bundle.surface.configure(&self.device, &self.surface_bundle.config);
    }

    pub fn begin_frame(&self) -> std::result::Result<(SurfaceTexture, TextureView), wgpu::SurfaceError> {
        let frame = self.surface_bundle.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok((frame, view))
    }
}