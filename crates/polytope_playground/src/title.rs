//! [`crate::director::Playback`] refuses a position track outright, because a
//! row slot's place belongs to its rigid body and a track writing it would run
//! a second clock against the solver. Neither half of that holds here. A title
//! letter is not a solver body, and its place in R³ is the typesetting's: this
//! module reads exactly one component of the position channel, `w`, which
//! decides whether the letter meets the slice at all rather than where it sits.
//! [`Transit::new`] enforces that by refusing any key with a nonzero `x`, `y`
//! or `z`, so a file cannot quietly claim a place the layout owns.

use std::borrow::Cow;

use ab_glyph::{Font, FontRef};
use anyhow::{anyhow, Result};
use glam::{Mat4, Vec3};
use loam_app::{egui, Camera, CameraController, FrameCtx, OrbitController, RenderCtx, SetupCtx};
use loam_egui::{Console, ConsoleUi};
use loam_math::{EuclideanR3, Projection};
use loam_render::{
    device::RenderDevice, DepthBuffer, DepthMode, FragmentShading, TriangleRasterNode,
};
use loam_shape::{TriangleMesh, Visualizable};
use loam_text::glyph::{layout_word, GlyphParams};
use loam_time::{Director, Drive};

/// The title, one line per entry. Letters are indexed across the whole title,
/// so the timeline addresses `PLAYGROUND`'s `P` as `letter8`.
const TITLE_LINES: [&str; 2] = ["POLYTOPE", "PLAYGROUND"];

/// The authored transit, compiled in rather than read from a path: the browser
/// build has no filesystem, and a title screen that only animates natively is
/// not a title screen.
const TIMELINE_RON: &str = include_str!("../timelines/title.ron");

const BODY_PREFIX: &str = "letter";

const EM_SIZE: f32 = 1.0;

const LETTER_DEPTH: f32 = 0.25;

const GLYPH_RESOLUTION: u32 = 48;

/// Local `w` from the apex of a letter's solid to its full-size section.
const APEX_DEPTH: f32 = 1.2;

/// Local `w` the solid continues past full size as a prism. The transit parks
/// the slice inside this, so the landed title holds still instead of shrinking
/// back out the far side.
const HOLD_DEPTH: f32 = 0.6;

const SLICE_W: f32 = 0.0;

const TITLE_COLOR: [f32; 4] = [0.90, 0.93, 1.00, 1.0];

const BACKGROUND: wgpu::Color = wgpu::Color {
    r: 0.020,
    g: 0.022,
    b: 0.032,
    a: 1.0,
};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

const BOOT_ORBIT_DISTANCE: f32 = 5.0;
const BOOT_ORBIT_PITCH: f32 = -0.10;

#[derive(Clone, Debug)]
struct Letter {
    base: TriangleMesh<3>,
    apex: Vec3,
}

/// Scale of a letter's section when the slice cuts its solid at `local_w`, or
/// `None` when the slice misses the solid and there is nothing to draw.
///
/// `local_w` is measured in the letter's frame: `SLICE_W - position.w`. The
/// apex is at `-APEX_DEPTH`, the full-size section at `0`, the end of the prism
/// at `+HOLD_DEPTH`.
fn section_scale(local_w: f32) -> Option<f32> {
    if !(-APEX_DEPTH..=HOLD_DEPTH).contains(&local_w) {
        return None;
    }
    let scale = ((local_w + APEX_DEPTH) / APEX_DEPTH).min(1.0);
    (scale > 0.0).then_some(scale)
}

fn build_section(letters: &[Letter], letter_w: &[f32], w_slice: f32, out: &mut TriangleMesh<3>) {
    out.vertices.clear();
    out.indices.clear();
    out.colors.clear();
    for (letter, &w) in letters.iter().zip(letter_w) {
        let Some(scale) = section_scale(w_slice - w) else {
            continue;
        };
        let base = out.vertices.len() as u32;
        for (vertex, color) in letter.base.vertices.iter().zip(&letter.base.colors) {
            let placed = letter.apex + scale * (Vec3::from_array(*vertex) - letter.apex);
            out.vertices.push(placed.to_array());
            out.colors.push(*color);
        }
        out.indices.extend(
            letter
                .base
                .indices
                .iter()
                .map(|[i, j, k]| [i + base, j + base, k + base]),
        );
    }
}

fn typeset(font: &FontRef<'_>) -> Result<Vec<Letter>> {
    let params = GlyphParams {
        em_size: EM_SIZE,
        depth: LETTER_DEPTH,
        resolution: GLYPH_RESOLUTION,
        color: TITLE_COLOR,
        ..GlyphParams::default()
    };
    let units_per_em = font
        .units_per_em()
        .ok_or_else(|| anyhow!("title font declares no units_per_em"))?;
    let line_advance = EM_SIZE * (font.height_unscaled() + font.line_gap_unscaled()) / units_per_em;

    let mut letters = Vec::new();
    for (line, text) in TITLE_LINES.iter().enumerate() {
        let solids = layout_word(font, text, &params)?;
        let width = solids
            .last()
            .map_or(0.0, |last| last.pen_origin().x + last.advance());
        let origin = Vec3::new(-0.5 * width, -(line as f32) * line_advance, 0.0);
        for solid in &solids {
            let mut base = Visualizable::<3>::to_triangles(solid)
                .map_err(|why| anyhow!("title letter {:?} has no geometry: {why:?}", solid.ch()))?;
            for vertex in &mut base.vertices {
                *vertex = (Vec3::from_array(*vertex) + origin).to_array();
            }
            let (lo, hi) = bounds(&base)
                .ok_or_else(|| anyhow!("title letter {:?} baked no vertices", solid.ch()))?;
            letters.push(Letter {
                base,
                apex: 0.5 * (lo + hi),
            });
        }
    }

    let (lo, hi) = letters
        .iter()
        .filter_map(|letter| bounds(&letter.base))
        .reduce(|(lo, hi), (l, h)| (lo.min(l), hi.max(h)))
        .ok_or_else(|| anyhow!("the title laid out with no ink"))?;
    let centre = 0.5 * (lo + hi);
    for letter in &mut letters {
        for vertex in &mut letter.base.vertices {
            *vertex = (Vec3::from_array(*vertex) - centre).to_array();
        }
        letter.apex -= centre;
    }
    Ok(letters)
}

fn bounds(mesh: &TriangleMesh<3>) -> Option<(Vec3, Vec3)> {
    mesh.vertices
        .iter()
        .map(|v| (Vec3::from_array(*v), Vec3::from_array(*v)))
        .reduce(|(lo, hi), (l, h)| (lo.min(l), hi.max(h)))
}

#[derive(Debug)]
struct Transit {
    director: Director,
    bound: Vec<usize>,
    /// Where each letter sits along `w`. The only thing the timeline is allowed
    /// to say about a letter, and the only thing read back out of it.
    letter_w: Vec<f32>,
}

impl Transit {
    /// Bind `director` to a title of `letters`, refusing everything that would
    /// leave a letter undriven or claim a channel this scene does not own.
    fn new(director: Director, letters: usize) -> Result<Self> {
        let mut bound = Vec::with_capacity(director.timeline().bodies.len());
        let mut driven = vec![false; letters];
        for body in &director.timeline().bodies {
            let letter = body
                .name
                .strip_prefix(BODY_PREFIX)
                .and_then(|index| index.parse::<usize>().ok())
                .ok_or_else(|| {
                    anyhow!(
                        "timeline body `{}` does not name a title letter; expected \
                         `{BODY_PREFIX}<index>`",
                        body.name
                    )
                })?;
            if letter >= letters {
                return Err(anyhow!(
                    "timeline body `{}` names letter {letter} of a {letters}-letter title",
                    body.name
                ));
            }
            if body.orientation.is_some() {
                return Err(anyhow!(
                    "timeline body `{}` has an orientation track, and nothing here poses a \
                     letter: the title reads a letter's `w` and takes the rest from the layout",
                    body.name
                ));
            }
            let track = body.position.as_ref().ok_or_else(|| {
                anyhow!(
                    "timeline body `{}` drives no channel; a letter with no `w` transit never \
                     crosses the slice",
                    body.name
                )
            })?;
            for (index, key) in track.keys().iter().enumerate() {
                if key.value.truncate() != Vec3::ZERO {
                    return Err(anyhow!(
                        "timeline body `{}` position key {index} moves the letter in R³, and a \
                         letter's place there belongs to the typesetting; only `w` is the \
                         timeline's",
                        body.name
                    ));
                }
            }
            driven[letter] = true;
            bound.push(letter);
        }
        if let Some(missing) = driven.iter().position(|driven| !driven) {
            return Err(anyhow!(
                "no timeline body drives letter {missing}, so it never crosses the slice"
            ));
        }

        let mut transit = Self {
            director,
            bound,
            letter_w: vec![0.0; letters],
        };
        transit.sample();
        Ok(transit)
    }

    fn sample(&mut self) {
        let Self {
            director,
            bound,
            letter_w,
        } = self;
        for (body, &letter) in director.bodies().zip(bound.iter()) {
            if let Drive::Directed(position) = director.position(body) {
                letter_w[letter] = position.w;
            }
        }
    }

    /// One frame of playback. The playhead is an integer frame index and reads
    /// no wall-clock delta, so the title is the same motion on any machine.
    fn advance(&mut self) {
        self.director.advance();
        self.sample();
    }

    fn seek(&mut self, frame: u32) {
        self.director.seek(frame);
        self.sample();
    }

    fn letter_w(&self) -> &[f32] {
        &self.letter_w
    }
}

pub(crate) struct TitleScene {
    camera: Camera<EuclideanR3>,
    orbit: OrbitController<EuclideanR3>,
    /// Only the shell's `scene` command, as in [`crate::s3`]: without it,
    /// booting `?scene=title&embed=1` would be a one-way trip.
    console: Console<()>,
    triangles: TriangleRasterNode,
    depth: Option<DepthBuffer>,
    letters: Vec<Letter>,
    transit: Transit,
    section: TriangleMesh<3>,
    playing: bool,
}

impl TitleScene {
    pub(crate) fn new(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let mut console = Console::<()>::new();
        loam_app::shell::register_command::<(), crate::shell::Playground>(&mut console);

        let letters = typeset(&title_font()?)?;
        let transit = Transit::new(Director::from_ron(TIMELINE_RON)?, letters.len())?;

        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 0.0, BOOT_ORBIT_DISTANCE);
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        orbit.set_orbit(BOOT_ORBIT_DISTANCE, BOOT_ORBIT_PITCH);

        Ok(Self {
            camera,
            orbit,
            console,
            triangles: TriangleRasterNode::new(
                &ctx.rd.device,
                ctx.rd.target_format(),
                DepthMode::ReadWrite {
                    format: DEPTH_FORMAT,
                },
                FragmentShading::FaceNormalLambert,
                ctx.rd.sample_count(),
            ),
            depth: None,
            letters,
            transit,
            section: TriangleMesh::default(),
            playing: true,
        })
    }
}

fn title_font() -> Result<FontRef<'static>> {
    FontRef::try_from_slice(epaint_default_fonts::HACK_REGULAR)
        .map_err(|why| anyhow!("bundled title font failed to parse: {why}"))
}

impl loam_app::shell::Scene for TitleScene {
    fn apply_command(
        &mut self,
        cmd: &loam_app::command::CommandLine,
        _ctx: &mut loam_app::command::CommandCtx<'_>,
    ) -> anyhow::Result<()> {
        self.console.dispatch(&cmd.name, &cmd.arg_refs(), &mut ());
        Ok(())
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        if self.playing {
            self.transit.advance();
        }
        let cfg = &ctx.rd.surface_bundle.config;
        self.camera.aspect = cfg.width as f32 / cfg.height.max(1) as f32;
        if !ctx.ui_capture.pointer {
            self.orbit
                .advance(ctx.input, &mut self.camera, &EuclideanR3, ctx.dt);
        }
    }

    fn ui(&mut self, ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {
        loam_app::log::pump_into(&mut self.console);
        loam_app::command::pump_into(&mut self.console);
        self.console.ui(ctx);
        loam_app::command::forward_pending(&mut self.console);
    }

    fn on_key(
        &mut self,
        code: winit::keyboard::KeyCode,
        state: winit::event::ElementState,
        ctx: &mut FrameCtx<'_>,
    ) {
        use winit::event::ElementState;
        use winit::keyboard::KeyCode;
        if ctx.ui_capture.keyboard || state != ElementState::Pressed {
            return;
        }
        match code {
            KeyCode::Space => self.playing = !self.playing,
            KeyCode::KeyR => {
                self.transit.seek(0);
                self.playing = true;
            }
            _ => {}
        }
    }

    fn record(&mut self, ctx: &mut RenderCtx<'_>) -> Result<()> {
        let rd: &RenderDevice = ctx.rd;
        let cfg = &rd.surface_bundle.config;

        // This scene owns the clear: nothing runs before it in the frame's
        // encoder, and the raster node loads rather than clears.
        let _ = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("title clear pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(BACKGROUND),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        DepthBuffer::ensure(
            &mut self.depth,
            &rd.device,
            DEPTH_FORMAT,
            (cfg.width, cfg.height),
            rd.sample_count(),
        );
        let depth = self.depth.as_ref().expect("ensure() guarantees Some");
        let _ = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("title depth clear pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        build_section(
            &self.letters,
            self.transit.letter_w(),
            SLICE_W,
            &mut self.section,
        );
        self.triangles.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &self.section,
            &Projection::Identity,
        );

        let view = self.camera.view();
        let aspect = cfg.width as f32 / cfg.height.max(1) as f32;
        let view_mat = Mat4::look_to_rh(view.position, view.forward, view.up);
        let proj_mat = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.1, 100.0);
        self.triangles.set_camera(&rd.queue, proj_mat * view_mat);
        self.triangles
            .record(ctx.encoder, ctx.view, Some(&depth.view), None);
        Ok(())
    }

    fn title(&self, _fps: f32) -> Cow<'static, str> {
        Cow::Borrowed("polytope playground - title (space: pause, R: replay)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec4;
    use loam_time::director::{BodyTrack, Ease, Timeline, Track};

    fn shipped() -> (Vec<Letter>, Transit) {
        let letters = typeset(&title_font().expect("bundled font")).expect("typeset");
        let director = Director::from_ron(TIMELINE_RON).expect("shipped timeline parses");
        let transit = Transit::new(director, letters.len()).expect("shipped timeline binds");
        (letters, transit)
    }

    /// Surface area rather than enclosed volume because the glyph pipeline's
    /// mesh is not closed for every letter (`P` at this em and pitch has a
    /// missing side wall of net oriented area 0.052), and the divergence
    /// theorem needs closure. Area needs none, and scales exactly as the square
    /// of the section scale.
    fn area(mesh: &TriangleMesh<3>) -> f32 {
        mesh.indices
            .iter()
            .map(|[i, j, k]| {
                let a = Vec3::from_array(mesh.vertices[*i as usize]);
                let b = Vec3::from_array(mesh.vertices[*j as usize]);
                let c = Vec3::from_array(mesh.vertices[*k as usize]);
                0.5 * (b - a).cross(c - a).length()
            })
            .sum()
    }

    fn extent(mesh: &TriangleMesh<3>) -> f32 {
        bounds(mesh).map_or(0.0, |(lo, hi)| (hi - lo).max_element())
    }

    fn section_at(letters: &[Letter], transit: &Transit) -> TriangleMesh<3> {
        let mut mesh = TriangleMesh::default();
        build_section(letters, transit.letter_w(), SLICE_W, &mut mesh);
        mesh
    }

    fn timeline_of(letters: usize, key: Vec4) -> Timeline {
        Timeline {
            fps: 60,
            frames: 61,
            w_slice: None,
            bodies: (0..letters)
                .map(|letter| BodyTrack {
                    name: format!("{BODY_PREFIX}{letter}"),
                    position: Some(Track::new().key(0.0, key, Ease::Linear)),
                    orientation: None,
                })
                .collect(),
        }
    }

    #[test]
    fn the_transit_moves_letters_along_w_and_never_through_r3() {
        let (letters, mut transit) = shipped();
        let mut w_moved = vec![false; letters.len()];
        let mut seen = vec![f32::NAN; letters.len()];
        let mut one = TriangleMesh::default();

        for frame in 0..transit.director.timeline().frames {
            transit.seek(frame);
            for (body, &letter) in transit.director.bodies().zip(&transit.bound) {
                let Drive::Directed(position) = transit.director.position(body) else {
                    panic!("{body} lost its position track at frame {frame}");
                };
                assert_eq!(
                    position.truncate(),
                    Vec3::ZERO,
                    "{body} claimed an R³ place at frame {frame}"
                );
                w_moved[letter] |= seen[letter].is_finite() && seen[letter] != position.w;
                seen[letter] = position.w;
            }
            let index = frame as usize % letters.len();
            let w = transit.letter_w()[index];
            let Some(scale) = section_scale(SLICE_W - w) else {
                continue;
            };
            build_section(&letters[index..=index], &[w], SLICE_W, &mut one);
            let (lo, hi) = bounds(&one).expect("a scaled section has vertices");
            let centre = 0.5 * (lo + hi);
            assert!(
                (centre - letters[index].apex).length() < 1e-5,
                "letter {index} at scale {scale} sits at {centre}, off its {}",
                letters[index].apex
            );
        }
        assert!(
            w_moved.iter().all(|moved| *moved),
            "some letter never moved along w: {w_moved:?}"
        );
    }

    #[test]
    fn a_position_key_that_moves_a_letter_through_r3_is_refused_at_load() {
        for offset in [Vec4::X, Vec4::Y, Vec4::Z, Vec4::new(0.0, -1e-3, 0.0, 2.0)] {
            let director = Director::new(timeline_of(2, offset)).expect("valid timeline");
            let error = Transit::new(director, 2).expect_err("R³ motion must be refused");
            assert!(
                format!("{error:#}").contains("moves the letter in R³"),
                "{offset}: {error:#}"
            );
        }
        let director = Director::new(timeline_of(2, Vec4::W * 0.5)).expect("valid timeline");
        let transit = Transit::new(director, 2).expect("a pure w key binds");
        assert_eq!(transit.letter_w(), [0.5, 0.5]);
    }

    /// Frame index a key is authored at. `key.t * fps` is not the way round to
    /// ask: `2.1 * 60` is 125.99999 in f32 while the sampler's `126 / 60` is
    /// exactly the `2.1` the file holds, so the frame is recovered by rounding
    /// and checked by mapping it back through the sampler's own division.
    fn key_frame(t: f32, fps: f32) -> u32 {
        let frame = (t * fps).round();
        assert_eq!(frame / fps, t, "key at t = {t} does not land on a frame");
        frame as u32
    }

    #[test]
    fn every_authored_key_lands_on_a_whole_frame() {
        let (_, transit) = shipped();
        let fps = transit.director.timeline().fps as f32;
        for body in &transit.director.timeline().bodies {
            for key in body.position.as_ref().expect("position track").keys() {
                let frame = key_frame(key.t, fps);
                assert!(
                    frame < transit.director.timeline().frames,
                    "{} keys at frame {frame}, past the timeline",
                    body.name
                );
            }
        }
    }

    #[test]
    fn every_letter_reaches_its_mark_at_the_keyframe_the_file_states() {
        let (_, mut transit) = shipped();
        let fps = transit.director.timeline().fps as f32;
        let marks: Vec<(String, Vec<(u32, Vec4)>)> = transit
            .director
            .timeline()
            .bodies
            .iter()
            .map(|body| {
                let keys = body
                    .position
                    .as_ref()
                    .expect("position track")
                    .keys()
                    .iter()
                    .map(|key| (key_frame(key.t, fps), key.value))
                    .collect();
                (body.name.clone(), keys)
            })
            .collect();

        for (name, keys) in &marks {
            for (frame, value) in keys {
                transit.seek(*frame);
                assert_eq!(
                    transit.director.position(name),
                    Drive::Directed(*value),
                    "{name} missed its mark at frame {frame}"
                );
            }
            let (entry, exit) = (keys.first().expect("keys"), keys.last().expect("keys"));
            assert_eq!(
                section_scale(SLICE_W - entry.1.w),
                None,
                "{name} is already visible at its first key"
            );
            assert_eq!(
                section_scale(SLICE_W - exit.1.w),
                Some(1.0),
                "{name} does not land at full size"
            );
        }
    }

    #[test]
    fn a_mid_transit_section_is_non_empty_and_smaller_than_the_landed_one() {
        let (letters, mut transit) = shipped();
        let frames = transit.director.timeline().frames;

        transit.seek(frames - 1);
        let landed = area(&section_at(&letters, &transit));
        assert!(landed > 0.0, "the landed title has no geometry");

        transit.seek(frames / 4);
        let mid = section_at(&letters, &transit);
        let mid_area = area(&mid);
        assert!(!mid.vertices.is_empty(), "mid-transit draws nothing at all");
        assert!(
            mid_area > 0.0 && mid_area < landed,
            "mid-transit area {mid_area} against a landed {landed}"
        );

        let first = &letters[..1];
        let mut mesh = TriangleMesh::default();
        let mut measure = |transit: &mut Transit, frame: u32| {
            transit.seek(frame);
            build_section(first, &transit.letter_w()[..1], SLICE_W, &mut mesh);
            (area(&mesh), extent(&mesh))
        };
        let (part_area, part_extent) = measure(&mut transit, 15);
        let (whole_area, whole_extent) = measure(&mut transit, 60);
        assert!(
            part_area > 0.0 && part_area < 0.9 * whole_area,
            "one letter mid-transit is {part_area} against its landed {whole_area}"
        );
        assert!(
            part_extent > 0.0 && part_extent < 0.9 * whole_extent,
            "one letter mid-transit spans {part_extent} against its landed {whole_extent}"
        );
    }

    #[test]
    fn the_section_never_recedes_over_the_run() {
        let (letters, mut transit) = shipped();
        let frames = transit.director.timeline().frames;
        let mut previous_scale = vec![0.0; letters.len()];
        let mut previous_area = 0.0;
        for frame in 0..frames {
            transit.seek(frame);
            for (previous, &w) in previous_scale.iter_mut().zip(transit.letter_w()) {
                let scale = section_scale(SLICE_W - w).unwrap_or(0.0);
                assert!(
                    scale >= *previous,
                    "frame {frame}: {scale} after {previous}"
                );
                *previous = scale;
            }
            if frame % 15 == 0 || frame == frames - 1 {
                let area = area(&section_at(&letters, &transit));
                assert!(
                    area >= previous_area,
                    "frame {frame} lost geometry: {area} after {previous_area}"
                );
                previous_area = area;
            }
        }
        assert!(previous_area > 0.0, "the title never arrived");
        assert!(previous_scale.iter().all(|scale| *scale == 1.0));
    }

    #[test]
    fn the_section_depends_only_on_the_gap_between_a_letter_and_the_slice() {
        const ELSEWHERE: f32 = 0.75;

        let (letters, _) = shipped();
        let one = &letters[3..4];
        let (mut here, mut elsewhere) = (TriangleMesh::default(), TriangleMesh::default());
        let mut drawn = 0;
        for step in 0..40 {
            let gap = -APEX_DEPTH + step as f32 * 0.05;
            build_section(one, &[SLICE_W - gap], SLICE_W, &mut here);
            build_section(one, &[ELSEWHERE - gap], ELSEWHERE, &mut elsewhere);
            assert_eq!(here.indices, elsewhere.indices, "gap {gap}");
            assert_eq!(here.vertices.len(), elsewhere.vertices.len(), "gap {gap}");
            for (a, b) in here.vertices.iter().zip(&elsewhere.vertices) {
                // Not bit-equal: the two ways of spelling the same gap differ
                // by an f32 rounding in the subtraction, which the scale then
                // carries into the coordinates.
                let (a, b) = (Vec3::from_array(*a), Vec3::from_array(*b));
                assert!((a - b).length() < 1e-6, "gap {gap}: {a} against {b}");
            }
            drawn += usize::from(!here.vertices.is_empty());
        }
        assert!(drawn > 30, "only {drawn} of 40 gaps drew anything");
    }

    #[test]
    fn the_letters_carry_one_opaque_colour_at_every_frame() {
        let (letters, mut transit) = shipped();
        for frame in (0..transit.director.timeline().frames).step_by(9) {
            transit.seek(frame);
            let mesh = section_at(&letters, &transit);
            assert_eq!(mesh.colors.len(), mesh.vertices.len());
            for color in &mesh.colors {
                assert_eq!(*color, TITLE_COLOR);
            }
        }
    }

    #[test]
    fn a_body_the_title_cannot_host_is_refused_at_load() {
        let named = |name: &str| Timeline {
            fps: 60,
            frames: 61,
            w_slice: None,
            bodies: vec![BodyTrack {
                name: name.to_owned(),
                position: Some(Track::new().key(0.0, Vec4::W, Ease::Linear)),
                orientation: None,
            }],
        };
        for name in ["slot0", "letter", "letterx", "0"] {
            let director = Director::new(named(name)).expect("valid timeline");
            let error = Transit::new(director, 4).expect_err("not a letter name");
            assert!(
                format!("{error:#}").contains("title letter"),
                "{name}: {error:#}"
            );
        }
        let director = Director::new(named("letter4")).expect("valid timeline");
        let error = Transit::new(director, 4).expect_err("letter 4 of a 4-letter title");
        assert!(format!("{error:#}").contains("4-letter title"), "{error:#}");
    }

    #[test]
    fn a_letter_the_timeline_leaves_undriven_is_refused_at_load() {
        let director = Director::new(Timeline {
            fps: 60,
            frames: 61,
            w_slice: None,
            bodies: vec![BodyTrack {
                name: format!("{BODY_PREFIX}0"),
                position: None,
                orientation: None,
            }],
        })
        .expect("valid timeline");
        let error = Transit::new(director, 1).expect_err("no channel");
        assert!(
            format!("{error:#}").contains("drives no channel"),
            "{error:#}"
        );

        let director = Director::new(timeline_of(1, Vec4::W)).expect("valid timeline");
        let error = Transit::new(director, 3).expect_err("letters 1 and 2 undriven");
        assert!(
            format!("{error:#}").contains("drives letter 1"),
            "{error:#}"
        );
    }

    #[test]
    fn an_orientation_track_is_refused_because_the_title_poses_no_letter() {
        let director = Director::new(Timeline {
            fps: 60,
            frames: 61,
            w_slice: None,
            bodies: vec![BodyTrack {
                name: format!("{BODY_PREFIX}0"),
                position: Some(Track::new().key(0.0, Vec4::W, Ease::Linear)),
                orientation: Some(Track::new().key(0.0, loam_math::Rotor4::IDENTITY, Ease::Linear)),
            }],
        })
        .expect("valid timeline");
        let error = Transit::new(director, 1).expect_err("nothing poses a letter");
        assert!(
            format!("{error:#}").contains("orientation track"),
            "{error:#}"
        );
    }

    #[test]
    fn the_section_scale_opens_from_the_apex_and_holds_across_the_prism() {
        assert_eq!(section_scale(-APEX_DEPTH - 0.01), None);
        assert_eq!(section_scale(-APEX_DEPTH), None, "the apex is a point");
        assert_eq!(section_scale(-0.5 * APEX_DEPTH), Some(0.5));
        assert_eq!(section_scale(0.0), Some(1.0));
        assert_eq!(section_scale(HOLD_DEPTH), Some(1.0));
        assert_eq!(section_scale(HOLD_DEPTH + 0.01), None);
    }

    #[test]
    fn the_title_stays_inside_its_per_frame_upload_budget() {
        let (letters, mut transit) = shipped();
        transit.seek(transit.director.timeline().frames - 1);
        let mesh = section_at(&letters, &transit);
        assert_eq!(
            mesh.vertices.len(),
            letters
                .iter()
                .map(|letter| letter.base.vertices.len())
                .sum::<usize>(),
            "the landed title draws every letter"
        );
        assert!(
            mesh.vertices.len() < 100_000,
            "the title is {} vertices, past its upload budget",
            mesh.vertices.len()
        );
    }

    #[test]
    fn the_title_is_laid_out_left_to_right_and_centred_on_its_ink() {
        let letters = typeset(&title_font().expect("bundled font")).expect("typeset");
        assert_eq!(
            letters.len(),
            TITLE_LINES.iter().map(|line| line.chars().count()).sum()
        );
        let (lo, hi) = letters
            .iter()
            .filter_map(|letter| bounds(&letter.base))
            .reduce(|(lo, hi), (l, h)| (lo.min(l), hi.max(h)))
            .expect("ink");
        let centre = 0.5 * (lo + hi);
        assert!(
            centre.length() < 1e-5,
            "the title is off centre at {centre}"
        );
        assert!(hi.x - lo.x > 4.0, "the title is only {} wide", hi.x - lo.x);

        let mut line_start = 0;
        for line in TITLE_LINES {
            let run = &letters[line_start..line_start + line.chars().count()];
            for pair in run.windows(2) {
                assert!(
                    pair[1].apex.x > pair[0].apex.x,
                    "letters of `{line}` do not advance"
                );
                assert!(
                    (pair[1].apex.y - pair[0].apex.y).abs() < 0.1 * EM_SIZE,
                    "`{line}` is not one line"
                );
            }
            line_start += run.len();
        }
        assert!(
            letters[TITLE_LINES[0].len()].apex.y < letters[0].apex.y - 0.5 * EM_SIZE,
            "the second line is not clear of the first"
        );
    }

    #[test]
    fn the_section_build_is_bit_reproducible() {
        let (letters, mut first) = shipped();
        let (_, mut second) = shipped();
        for frame in [0u32, 37, 96, 180] {
            first.seek(frame);
            second.seek(frame);
            assert_eq!(first.letter_w(), second.letter_w());
            let (a, b) = (section_at(&letters, &first), section_at(&letters, &second));
            assert_eq!(a.vertices, b.vertices, "frame {frame}");
            assert_eq!(a.indices, b.indices, "frame {frame}");
        }
    }

    #[test]
    fn playback_advances_one_frame_per_update_and_holds_the_landed_title() {
        let (letters, mut transit) = shipped();
        let frames = transit.director.timeline().frames;
        for expected in 0..frames {
            assert_eq!(transit.director.frame(), expected);
            transit.advance();
        }
        let landed = section_at(&letters, &transit);
        for _ in 0..30 {
            transit.advance();
            assert_eq!(transit.director.frame(), frames - 1);
        }
        assert_eq!(section_at(&letters, &transit).vertices, landed.vertices);
    }
}
