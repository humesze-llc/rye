use crate::device::RenderDevice;
use anyhow::Result;

/// One render-pass-or-equivalent unit of work the graph runs.
pub trait RenderNode {
    /// Static label used for tracing / debug output.
    fn name(&self) -> &'static str;

    /// Errors abort the rest of the graph.
    fn execute(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()>;
}

/// Linear sequence of [`RenderNode`]s.
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

    /// Run every node in insertion order, propagating the first error the graph encounters and
    /// aborting subsequent nodes.
    pub fn run(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        for n in &mut self.nodes {
            n.execute(rd, view)?;
        }
        Ok(())
    }
}
