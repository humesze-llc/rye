use anyhow::Result;
use wgpu::*;

use crate::device::RenderDevice;
use crate::graph::RenderNode;
use crate::raymarch::{RayMarchNode, RayMarchUniforms};

/// Build the module with `loam_shader::ShaderDb::load_geodesic_scene`.
pub struct GeodesicRayMarchNode(RayMarchNode);

impl GeodesicRayMarchNode {
    pub fn from_module(
        device: &Device,
        surface_format: TextureFormat,
        module: &ShaderModule,
        sample_count: u32,
    ) -> Self {
        Self(RayMarchNode::new(
            device,
            surface_format,
            module,
            sample_count,
        ))
    }

    pub fn uniforms(&self) -> &RayMarchUniforms {
        self.0.uniforms()
    }

    pub fn uniforms_mut(&mut self) -> &mut RayMarchUniforms {
        self.0.uniforms_mut()
    }

    pub fn set_uniforms(&mut self, queue: &Queue, uniforms: RayMarchUniforms) {
        self.0.set_uniforms(queue, uniforms);
    }

    pub fn flush_uniforms(&self, queue: &Queue) {
        self.0.flush_uniforms(queue);
    }

    pub fn record_in_viewport(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        viewport: crate::Viewport,
    ) {
        self.0.record_in_viewport(encoder, view, viewport);
    }

    pub fn execute_panel(
        &mut self,
        rd: &RenderDevice,
        view: &wgpu::TextureView,
        clear: bool,
        scissor: [u32; 4],
    ) -> Result<()> {
        self.0.execute_panel(rd, view, clear, scissor)
    }
}

const _: fn(
    &mut GeodesicRayMarchNode,
    &RenderDevice,
    &wgpu::TextureView,
    bool,
    [u32; 4],
) -> Result<()> = GeodesicRayMarchNode::execute_panel;

impl RenderNode for GeodesicRayMarchNode {
    fn name(&self) -> &'static str {
        "geodesic_raymarch"
    }

    fn execute(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        RenderNode::execute(&mut self.0, rd, view)
    }
}
