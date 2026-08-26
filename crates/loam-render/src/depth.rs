//! Sample count must match the color attachment's MSAA configuration.
//!
//! The framework doesn't surface a resize hook on `App`, so [`DepthBuffer::ensure`] checks
//! size + sample count each frame and recreates the texture only when they change. Holds
//! the [`wgpu::TextureView`] only; the underlying texture stays alive via wgpu's internal
//! Arc reference held by the view.

use wgpu::{
    Device, Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureView, TextureViewDescriptor,
};

/// Owns a depth texture view sized to the swapchain, recreated on resize.
pub struct DepthBuffer {
    pub view: TextureView,
    pub format: TextureFormat,
    size: (u32, u32),
    sample_count: u32,
}

impl DepthBuffer {
    pub fn new(
        device: &Device,
        format: TextureFormat,
        size: (u32, u32),
        sample_count: u32,
    ) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("loam-render DepthBuffer"),
            size: Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        Self {
            view,
            format,
            size,
            sample_count,
        }
    }

    /// Intended to be called once per frame at the top of the render function.
    pub fn ensure(
        slot: &mut Option<DepthBuffer>,
        device: &Device,
        format: TextureFormat,
        size: (u32, u32),
        sample_count: u32,
    ) {
        let needs_recreate = match slot {
            Some(b) => b.format != format || b.size != size || b.sample_count != sample_count,
            None => true,
        };
        if needs_recreate {
            *slot = Some(DepthBuffer::new(device, format, size, sample_count));
        }
    }
}
