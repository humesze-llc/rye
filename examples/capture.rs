//! Frame capture and APNG export for Rye examples.
//!
//! Usage:
//!   1. Add `COPY_SRC` to the surface texture usage (done in `RenderDevice`).
//!   2. After the render pass but BEFORE `frame.present()`, call
//!      `FrameCapture::capture(device, queue, &frame)`.
//!   3. When `is_done()`, call `save_apng` or `save_pngs`.

use anyhow::Result;
use std::path::Path;
use wgpu::{Device, Queue, SurfaceTexture, COPY_BYTES_PER_ROW_ALIGNMENT};

pub struct FrameCapture {
    max_frames: u32,
    fps: u32,
    width: u32,
    height: u32,
    frames: Vec<Vec<u8>>,
}

impl FrameCapture {
    pub fn new(max_frames: u32, fps: u32, width: u32, height: u32) -> Self {
        Self {
            max_frames,
            fps,
            width,
            height,
            frames: Vec::with_capacity(max_frames as usize),
        }
    }

    pub fn is_done(&self) -> bool {
        self.frames.len() >= self.max_frames as usize
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Copy the current surface texture into the frame buffer.
    ///
    /// Must be called AFTER the render pass and BEFORE `frame.present()`.
    /// The surface must have been configured with `TextureUsages::COPY_SRC`
    /// (see `RenderDevice::new`).
    pub fn capture(&mut self, device: &Device, queue: &Queue, surface_tex: &SurfaceTexture) {
        if self.is_done() {
            return;
        }

        let width = self.width;
        let height = self.height;

        // bytes_per_row must be a multiple of COPY_BYTES_PER_ROW_ALIGNMENT (256).
        let bytes_per_row_unaligned = width * 4;
        let bytes_per_row = (bytes_per_row_unaligned + COPY_BYTES_PER_ROW_ALIGNMENT - 1)
            / COPY_BYTES_PER_ROW_ALIGNMENT
            * COPY_BYTES_PER_ROW_ALIGNMENT;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame capture readback"),
            size: (bytes_per_row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame capture encoder"),
        });
        encoder.copy_texture_to_buffer(
            surface_tex.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        // Synchronous map: blocks until the GPU copy is done.
        let (tx, rx) = std::sync::mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |res| tx.send(res).ok().unwrap());
        let _ = device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap().unwrap();

        let raw = buffer.slice(..).get_mapped_range();

        // Surface on Windows is typically Bgra8UnormSrgb; swap to RGBA for PNG.
        let is_bgra = matches!(
            surface_tex.texture.format(),
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );

        let mut frame: Vec<u8> = Vec::with_capacity((width * height * 4) as usize);
        for row in raw.chunks(bytes_per_row as usize) {
            let pixels = &row[..width as usize * 4];
            if is_bgra {
                for px in pixels.chunks(4) {
                    frame.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
                }
            } else {
                frame.extend_from_slice(pixels);
            }
        }

        drop(raw);
        buffer.unmap();

        self.frames.push(frame);
    }

    /// Encode all captured frames as a looping APNG.
    pub fn save_apng(&self, path: &Path) -> Result<()> {
        use std::fs::File;
        use std::io::BufWriter;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = BufWriter::new(File::create(path)?);
        let mut encoder = png::Encoder::new(file, self.width, self.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_animated(self.frames.len() as u32, 0)?;

        let mut writer = encoder.write_header()?;
        for frame in &self.frames {
            // delay_num/delay_den: 1/fps seconds per frame.
            writer.set_frame_delay(1, self.fps as u16)?;
            writer.write_image_data(frame)?;
        }
        drop(writer);

        tracing::info!(
            "saved {} frames → {:?} ({}×{} @ {} fps)",
            self.frames.len(),
            path,
            self.width,
            self.height,
            self.fps,
        );
        Ok(())
    }

    /// Encode all captured frames as a looping GIF (256-color palette per frame).
    pub fn save_gif(&self, path: &Path) -> Result<()> {
        use std::fs::File;
        use std::io::BufWriter;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = BufWriter::new(File::create(path)?);
        let mut encoder = gif::Encoder::new(file, self.width as u16, self.height as u16, &[])?;
        encoder.set_repeat(gif::Repeat::Infinite)?;

        // Centiseconds per frame (GIF delay unit).
        let delay_cs = (100u16).saturating_div(self.fps as u16).max(1);

        for rgba in &self.frames {
            let mut frame =
                gif::Frame::from_rgba_speed(self.width as u16, self.height as u16, &mut rgba.clone(), 10);
            frame.delay = delay_cs;
            encoder.write_frame(&frame)?;
        }

        tracing::info!(
            "saved {} frames → {:?} ({}×{} @ {} fps, GIF)",
            self.frames.len(),
            path,
            self.width,
            self.height,
            self.fps,
        );
        Ok(())
    }

    /// Save all frames as numbered PNGs for offline processing (rye-diag, ffmpeg).
    #[allow(dead_code)]
    pub fn save_pngs(&self, dir: &Path) -> Result<()> {
        use std::fs::File;
        use std::io::BufWriter;

        std::fs::create_dir_all(dir)?;
        for (i, frame) in self.frames.iter().enumerate() {
            let path = dir.join(format!("frame_{i:04}.png"));
            let file = BufWriter::new(File::create(&path)?);
            let mut encoder = png::Encoder::new(file, self.width, self.height);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header()?;
            writer.write_image_data(frame)?;
            drop(writer);
        }
        tracing::info!("saved {} PNGs → {:?}", self.frames.len(), dir);
        Ok(())
    }
}
