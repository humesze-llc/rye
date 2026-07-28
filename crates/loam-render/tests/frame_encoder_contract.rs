//! Signature pins for the per-frame draw entry points.
//!
//! The runner opens one command encoder per frame and submits it once
//! (`loam_app::App::record`). A node that took a `RenderDevice` here could
//! reach `queue.submit` and reintroduce a mid-frame submit, so the contract is
//! carried by the types: `record` receives the caller's encoder, has no device
//! to submit through, and returns `()` rather than a `Result` a caller might
//! read as "this ran."
//!
//! The calls themselves need a real surface, so a coercion pins what a call
//! cannot; same device-free technique as `raymarch_chain_smoke`.

use loam_render::raymarch::Hyperslice4DNode;
use loam_render::{LineRasterNode, PointRasterNode, TriangleRasterNode};

/// The three rasterizer nodes share one recording shape: caller's encoder,
/// color view, optional depth view, optional viewport, no return.
type RecordRaster<N> = fn(
    &N,
    &mut wgpu::CommandEncoder,
    &wgpu::TextureView,
    Option<&wgpu::TextureView>,
    Option<&loam_render::Viewport>,
);

#[test]
fn raster_draw_entry_points_record_into_the_callers_encoder() {
    let _: RecordRaster<LineRasterNode> = LineRasterNode::record;
    let _: RecordRaster<PointRasterNode> = PointRasterNode::record;
    let _: RecordRaster<TriangleRasterNode> = TriangleRasterNode::record;
}

/// The SDF pass is the one the playground draws every frame, and it takes the
/// viewport by value rather than by option: it always covers a rect.
#[test]
fn hyperslice_viewport_draw_records_into_the_callers_encoder() {
    type RecordInViewport = fn(
        &mut Hyperslice4DNode,
        &mut wgpu::CommandEncoder,
        &wgpu::TextureView,
        loam_render::Viewport,
    );
    let _: RecordInViewport = Hyperslice4DNode::record_in_viewport;
}
