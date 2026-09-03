use std::sync::Arc;

use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor};
use winit::window::Window;

pub struct UiIntegration {
    ctx: egui::Context,
    winit_state: egui_winit::State,
    renderer: Renderer,
    pixels_per_point: f32,
}

impl UiIntegration {
    /// `surface_format` must be the format of the view [`UiIntegration::paint`] receives.
    pub fn new(
        device: &wgpu::Device,
        window: &Arc<Window>,
        surface_format: wgpu::TextureFormat,
        msaa_samples: u32,
    ) -> Self {
        let ctx = egui::Context::default();
        let winit_state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            window.theme(),
            None,
        );
        let renderer = Renderer::new(
            device,
            surface_format,
            RendererOptions {
                msaa_samples,
                ..Default::default()
            },
        );
        let pixels_per_point = window.scale_factor() as f32;
        Self {
            ctx,
            winit_state,
            renderer,
            pixels_per_point,
        }
    }

    pub fn handle_event(
        &mut self,
        window: &Window,
        event: &winit::event::WindowEvent,
    ) -> egui_winit::EventResponse {
        self.winit_state.on_window_event(window, event)
    }

    /// `target_format` and `sample_count` must match the values passed to
    /// [`UiIntegration::new`], or the warm compiles the wrong pipeline variant.
    pub fn warm_pipelines(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        window: &Window,
        target_format: wgpu::TextureFormat,
        sample_count: u32,
    ) {
        let dummy_color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("loam-egui::warm color"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let dummy_view = dummy_color.create_view(&wgpu::TextureViewDescriptor::default());

        let resolve_tex = if sample_count > 1 {
            Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("loam-egui::warm resolve"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: target_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            }))
        } else {
            None
        };
        let resolve_view = resolve_tex
            .as_ref()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("loam-egui::warm encoder"),
        });

        // Text, filled rect and line stroke cover the pipeline variants the demos use.
        let ctx = self.begin_frame(window);
        let ctx = ctx.clone();
        egui::Area::new(egui::Id::new("loam-egui::warm")).show(&ctx, |ui| {
            ui.label("warm");
            ui.separator();
            ui.painter().line_segment(
                [egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)],
                egui::Stroke::new(1.0, egui::Color32::WHITE),
            );
        });
        self.paint(
            device,
            queue,
            &mut encoder,
            &dummy_view,
            resolve_view.as_ref(),
            window,
            (1, 1),
        );

        queue.submit(Some(encoder.finish()));
    }

    /// Refreshes the hit test [`crate::UiCapture`] reads; call before `App::update`.
    pub fn begin_frame(&mut self, window: &Window) -> &egui::Context {
        let raw_input = self.winit_state.take_egui_input(window);
        self.pixels_per_point = window.scale_factor() as f32;
        self.ctx.begin_pass(raw_input);
        &self.ctx
    }

    /// Loads the attachment, so the scene must already be in `view`. `viewport`
    /// is in pixels; `resolve_target` is `Some` only when `view` is multisampled.
    #[allow(clippy::too_many_arguments)]
    pub fn paint(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        resolve_target: Option<&wgpu::TextureView>,
        window: &Window,
        viewport: (u32, u32),
    ) {
        let full_output = self.ctx.end_pass();

        self.winit_state
            .handle_platform_output(window, full_output.platform_output);

        let primitives = self
            .ctx
            .tessellate(full_output.shapes, self.pixels_per_point);

        let screen = ScreenDescriptor {
            size_in_pixels: [viewport.0, viewport.1],
            pixels_per_point: self.pixels_per_point,
        };

        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer
                .update_texture(device, queue, *id, image_delta);
        }
        self.renderer
            .update_buffers(device, queue, encoder, &primitives, &screen);

        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("loam-egui::paint"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            // egui-wgpu wants a `RenderPass<'static>`.
            self.renderer
                .render(&mut pass.forget_lifetime(), &primitives, &screen);
        }

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }
}
