use anyhow::Result;
use crate::device::RenderDevice;

pub trait RenderNode {
    fn name(&self) -> &'static str;
    fn execute(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()>;
}

pub struct RenderGraph {
    nodes: Vec<Box<dyn RenderNode>>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn add_node<N: RenderNode + 'static>(mut self, node: N) -> Self {
        self.nodes.push(Box::new(node));
        self
    }

    pub fn run(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        for n in &mut self.nodes {
            n.execute(rd, view)?;
        }
        Ok(())
    }
}