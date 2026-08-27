use crate::device::RenderDevice;
use anyhow::Result;

pub trait RenderNode {
    fn name(&self) -> &'static str;

    /// Errors abort the rest of the graph.
    fn execute(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()>;
}

#[derive(Default)]
pub struct RenderGraph {
    nodes: Vec<Box<dyn RenderNode>>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node<N: RenderNode + 'static>(mut self, node: N) -> Self {
        self.nodes.push(Box::new(node));
        self
    }

    /// Insertion order; the first error aborts the remaining nodes.
    pub fn run(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        for n in &mut self.nodes {
            n.execute(rd, view)?;
        }
        Ok(())
    }
}
