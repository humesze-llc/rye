use anyhow::Result;
use std::sync::Arc;

use rye_render::{device::RenderDevice, graph::{RenderGraph, RenderNode}};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes},
};

mod hud;
use hud::Hud;

struct ClearNode;
impl RenderNode for ClearNode {
    fn name(&self) -> &'static str { "clear" }
    fn execute(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        let mut encoder = rd.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Clear Encoder"),
        });
        {
            let _rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        rd.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}

struct App {
    window: Option<Arc<Window>>,
    rd: Option<RenderDevice>,
    graph: RenderGraph,
    minimized: bool,

    hud: Option<Hud>,
    static_lines: String,

    last_frame_t: std::time::Instant,
    ema_fps: f32,
    dropped_timeouts: u32,
    dropped_lost: u32,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, elwt: &ActiveEventLoop) {
        // Create window hidden to avoid initial white flash
        let win = Arc::new(elwt.create_window(
            WindowAttributes::default().with_title("Mars Hello").with_visible(false)
        ).expect("create window"));

        // GPU objects
        let rd = pollster::block_on(RenderDevice::new(win.clone())).expect("render device");

        // ===== build static metrics text =====
        let info = rd.adapter.get_info();
        let fmt  = rd.surface_bundle.config.format;
        let pm   = rd.surface_bundle.config.present_mode;
        let am   = rd.surface_bundle.config.alpha_mode;
        let lim  = rd.adapter.limits();
        let size = win.inner_size();
        let scale= win.scale_factor();

        let (x,y,z) = (
            lim.max_compute_workgroup_size_x,
            lim.max_compute_workgroup_size_y,
            lim.max_compute_workgroup_size_z,
        );

        self.static_lines = format!(
"GPU: {} ({:?}, {:?})
Driver: {} [{}]
Vendor/Device: {:#06x}/{:#06x}
Surface: {:?} (sRGB={}) Mode: {:?} Alpha: {:?}
Limits: max_tex2D={}, bind_groups={}, max_buf={}
Align: UBO={}, SSBO={}
Compute: wg_size=({},{},{}) invocations={}
Window: {}×{} @ scale {:.2}",
            info.name, info.backend, info.device_type,
            info.driver, info.driver_info,
            info.vendor, info.device,
            fmt, fmt.is_srgb(), pm, am,
            lim.max_texture_dimension_2d, lim.max_bind_groups, lim.max_buffer_size,
            lim.min_uniform_buffer_offset_alignment, lim.min_storage_buffer_offset_alignment,
            x, y, z, lim.max_compute_invocations_per_workgroup,
            size.width, size.height, scale
        );

        // HUD
        let mut hud = Hud::new(&rd.device, fmt);
        hud.update_vertices(&rd.queue, rd.surface_bundle.config.width, rd.surface_bundle.config.height, 8);

        // First frame (so we show with HUD already drawn)
        self.window = Some(win.clone());
        self.rd = Some(rd);
        self.hud = Some(hud);
        self.minimized = false;
        self.last_frame_t = std::time::Instant::now();
        self.ema_fps = 0.0;
        self.dropped_timeouts = 0;
        self.dropped_lost = 0;

        // Build first text & draw once before showing
        if let (Some(rd), Some(hud)) = (&self.rd, &self.hud) {
            let dyn_line = format!("Frame: {:>.2} ms FPS≈{:>.1} dropped={}/{}", 0.0f32, 0.0f32, 0, 0);
            let text = format!("{}\n{}", self.static_lines, dyn_line);
            hud.upload_text(&rd.queue, &text);

            if let Ok((frame, view)) = rd.begin_frame() {
                let mut encoder = rd.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("First Frame") });
                {
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Clear+HUD"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    hud.draw(&mut rp);
                }
                rd.queue.submit(Some(encoder.finish()));
                frame.present();
            }
        }

        win.set_visible(true);
        if let Some(w) = &self.window { w.request_redraw(); }
    }

    fn window_event(&mut self, elwt: &ActiveEventLoop, _id: winit::window::WindowId, ev: WindowEvent) {
        let (Some(win), Some(rd), Some(hud)) = (&self.window, &mut self.rd, &mut self.hud) else { return; };

        match ev {
            WindowEvent::CloseRequested => elwt.exit(),

            WindowEvent::Resized(size) => {
                self.minimized = size.width == 0 || size.height == 0;
                if !self.minimized {
                    rd.resize(size);
                    hud.update_vertices(&rd.queue, rd.surface_bundle.config.width, rd.surface_bundle.config.height, 8);
                }
            }

            WindowEvent::ScaleFactorChanged { .. } => {
                hud.update_vertices(&rd.queue, rd.surface_bundle.config.width, rd.surface_bundle.config.height, 8);
            }

            WindowEvent::RedrawRequested => {
                if self.minimized { return; }

                // timing
                let now = std::time::Instant::now();
                let dt  = now.duration_since(self.last_frame_t).as_secs_f32();
                self.last_frame_t = now;
                let frame_ms = (dt * 1000.0).clamp(0.0, 1000.0);
                let fps_now  = if dt > 0.0 { 1.0 / dt } else { 0.0 };
                self.ema_fps = if self.ema_fps == 0.0 { fps_now } else { 0.9*self.ema_fps + 0.1*fps_now };

                match rd.begin_frame() {
                    Ok((frame, view)) => {
                        // Build live line and upload
                        let dyn_line = format!("Frame: {:>.2} ms FPS≈{:>.1} dropped={}/{}",
                                               frame_ms, self.ema_fps, self.dropped_timeouts, self.dropped_lost);
                        let text = format!("{}\n{}", self.static_lines, dyn_line);
                        hud.upload_text(&rd.queue, &text);

                        // Clear + HUD
                        let mut encoder = rd.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Frame Encoder") });
                        {
                            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("Clear Pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    depth_slice: None,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });
                            hud.draw(&mut rp);
                        }
                        rd.queue.submit(Some(encoder.finish()));
                        frame.present();
                        win.request_redraw();
                    }
                    Err(err) => {
                        match err {
                            wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated => {
                                self.dropped_lost += 1;
                                rd.resize(rd.surface_bundle.size);
                                hud.update_vertices(&rd.queue, rd.surface_bundle.config.width, rd.surface_bundle.config.height, 8);
                            }
                            wgpu::SurfaceError::Timeout => {
                                self.dropped_timeouts += 1;
                                // just skip this frame
                            }
                            wgpu::SurfaceError::OutOfMemory => elwt.exit(),
                            wgpu::SurfaceError::Other => eprintln!("surface error: {:?}", err),
                        }
                        win.request_redraw();
                    }
                }
            }
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    let event_loop: EventLoop<()> = EventLoop::new()?;
    let app = &mut App {
        window: None,
        rd: None,
        graph: RenderGraph::new().add_node(ClearNode), // your other nodes can follow later
        minimized: false,
        hud: None,
        static_lines: String::new(),
        last_frame_t: std::time::Instant::now(),
        ema_fps: 0.0,
        dropped_timeouts: 0,
        dropped_lost: 0,
    };
    event_loop.run_app(app)?;
    Ok(())
}