//! Linear render-graph harness. Owns a [`Vec<Box<dyn RenderNode>>`]
//! and runs each node's `execute` in order against a single view.
//!
//! No dependency tracking, no cross-node resource sharing, no
//! parallel execution. The trait is intentionally a one-method
//! `execute(rd, view)`: complex apps that need scheduling or
//! aliasing should compose nodes manually rather than waiting for
//! this graph to grow features.

use crate::device::RenderDevice;
use anyhow::Result;

pub trait RenderNode {
    fn name(&self) -> &'static str;
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

    pub fn run(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        for n in &mut self.nodes {
            n.execute(rd, view)?;
        }
        Ok(())
    }
}
