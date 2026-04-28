//! Frame capture: copy the surface texture into RAM each frame, then
//! encode the buffered frames as APNG, GIF, or numbered PNGs once N
//! frames have been captured.
//!
//! Wired into the [`crate::Runner`] when [`crate::RunConfig::capture`]
//! is `Some`. Examples opt in by parsing `--capture-apng` /
//! `--capture-gif` / `--capture-frames` / `--capture-fps` from
//! `std::env::args()` via [`CaptureConfig::from_env_args`].

use std::path::{Path, PathBuf};

use anyhow::Result;
use wgpu::{Device, Queue, SurfaceTexture, COPY_BYTES_PER_ROW_ALIGNMENT};

/// Output format for a capture session.
#[derive(Clone, Debug)]
pub enum CaptureFormat {
    /// Animated PNG, lossless. Larger files than GIF; perfect quality.
    Apng,
    /// Animated GIF, palette-quantised. Smaller files; lossy colour.
    Gif,
}

/// User-facing capture parameters: where to write, how many frames,
/// at what playback rate. Built either by parsing CLI args
/// ([`CaptureConfig::from_env_args`]) or constructed directly.
#[derive(Clone, Debug)]
pub struct CaptureConfig {
    pub output_path: PathBuf,
    pub format: CaptureFormat,
    /// Number of frames to capture before saving + exiting.
    pub frames: u32,
    /// Playback frame rate baked into the encoded output. Does not
    /// throttle capture itself; the runner captures every rendered
    /// frame until `frames` has been reached.
    pub fps: u32,
}

impl CaptureConfig {
    /// Parse `--capture-apng PATH` / `--capture-gif PATH` /
    /// `--capture-frames N` / `--capture-fps N` from `std::env::args()`.
    /// Returns `None` if neither `--capture-apng` nor `--capture-gif`
    /// is present.
    ///
    /// Defaults: `frames = 300`, `fps = 30` (10 s of playback).
    pub fn from_env_args() -> Option<Self> {
        let args: Vec<String> = std::env::args().collect();
        let apng = arg_value(&args, "--capture-apng").map(PathBuf::from);
        let gif = arg_value(&args, "--capture-gif").map(PathBuf::from);
        let (output_path, format) = match (apng, gif) {
            (Some(p), _) => (p, CaptureFormat::Apng),
            (_, Some(p)) => (p, CaptureFormat::Gif),
            (None, None) => return None,
        };
        let frames = arg_value(&args, "--capture-frames")
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(300);
        let fps = arg_value(&args, "--capture-fps")
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(30);
        Some(Self {
            output_path,
            format,
            frames,
            fps,
        })
    }
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    let i = args.iter().position(|a| a == flag)?;
    args.get(i + 1).cloned()
}

/// Active capture session. Constructed by the [`crate::Runner`] from
/// a [`CaptureConfig`] once the surface size is known. Not meant to
/// be constructed directly by user code.
pub(crate) struct FrameCapture {
    config: CaptureConfig,
    width: u32,
    height: u32,
    frames: Vec<Vec<u8>>,
}

impl FrameCapture {
    pub(crate) fn new(config: CaptureConfig, width: u32, height: u32) -> Self {
        let cap = config.frames as usize;
        Self {
            config,
            width,
            height,
            frames: Vec::with_capacity(cap),
        }
    }

    pub(crate) fn is_done(&self) -> bool {
        self.frames.len() >= self.config.frames as usize
    }

    /// Copy the current surface texture into the frame buffer.
    ///
    /// Must be called AFTER the render pass and BEFORE
    /// `frame.present()`. The surface must have been configured with
    /// `TextureUsages::COPY_SRC` (which `RenderDevice::new` does
    /// unconditionally).
    pub(crate) fn capture(&mut self, device: &Device, queue: &Queue, surface_tex: &SurfaceTexture) {
        if self.is_done() {
            return;
        }

        let width = self.width;
        let height = self.height;

        // bytes_per_row must be a multiple of COPY_BYTES_PER_ROW_ALIGNMENT (256).
        let bytes_per_row_unaligned = width * 4;
        let bytes_per_row = bytes_per_row_unaligned.div_ceil(COPY_BYTES_PER_ROW_ALIGNMENT)
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

    /// Save the captured frames according to the session's
    /// configured format. Called by the runner once `is_done()`.
    pub(crate) fn save(&self) -> Result<()> {
        match self.config.format {
            CaptureFormat::Apng => self.save_apng(&self.config.output_path),
            CaptureFormat::Gif => self.save_gif(&self.config.output_path),
        }
    }

    fn save_apng(&self, path: &Path) -> Result<()> {
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
            writer.set_frame_delay(1, self.config.fps as u16)?;
            writer.write_image_data(frame)?;
        }
        drop(writer);

        tracing::info!(
            "saved {} frames to {:?} ({}x{} @ {} fps, APNG)",
            self.frames.len(),
            path,
            self.width,
            self.height,
            self.config.fps,
        );
        Ok(())
    }

    fn save_gif(&self, path: &Path) -> Result<()> {
        use std::fs::File;
        use std::io::BufWriter;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = BufWriter::new(File::create(path)?);
        let mut encoder = gif::Encoder::new(file, self.width as u16, self.height as u16, &[])?;
        encoder.set_repeat(gif::Repeat::Infinite)?;

        // Centiseconds per frame (GIF delay unit).
        let delay_cs = (100u16).saturating_div(self.config.fps as u16).max(1);

        for rgba in &self.frames {
            let mut frame = gif::Frame::from_rgba_speed(
                self.width as u16,
                self.height as u16,
                &mut rgba.clone(),
                10,
            );
            frame.delay = delay_cs;
            encoder.write_frame(&frame)?;
        }

        tracing::info!(
            "saved {} frames to {:?} ({}x{} @ {} fps, GIF)",
            self.frames.len(),
            path,
            self.width,
            self.height,
            self.config.fps,
        );
        Ok(())
    }
}
