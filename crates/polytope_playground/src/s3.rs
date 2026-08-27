//! `loam-shape` stores every 4-polytope at unit circumradius, so its vertices
//! are already points of S³. The motion is isoclinic: equal angles in the two
//! orthogonal invariant planes `e₁∧e₂` and `e₃∧e₄`. On S³ that is a Clifford
//! translation. For `B = θ(e₁₂ + e₃₄)` and any unit `v`,
//!
//! ```text
//!   ⟨v, Rv⟩ = cos θ·(v₁² + v₂²) + cos θ·(v₃² + v₄²) = cos θ
//! ```
//!
//! so every point is displaced by exactly `θ`, with no fixed point and no axis.
//! <https://en.wikipedia.org/wiki/Rotations_in_4-dimensional_Euclidean_space>

use std::borrow::Cow;

use anyhow::Result;
use glam::{Mat4, Vec2, Vec3, Vec4};
use loam_app::{egui, Camera, CameraController, FrameCtx, OrbitController, RenderCtx, SetupCtx};
use loam_egui::{Console, ConsoleUi};
use loam_math::{
    Bivector, Bivector4, EuclideanR3, Iso4, IsometryGroup, Projection, Rotor4, SphericalS3Embedded,
};
use loam_render::{device::RenderDevice, DepthMode, LineRasterNode};
use loam_shape::polytope::Polytope4;
use loam_shape::LineMesh;

use crate::catalog::SHAPE_CATALOG;
use crate::color::w_depth_color;
use crate::wireframe_geom::push_blended_edge;

const PROJECTION: Projection<4> = Projection::Stereographic { pole: Vec4::W };

const DEFAULT_RATE: f32 = 0.35;

// Every polychoron needs it here, not just the 16-cell the flat scene
// special-cases: the pole is ambient-fixed while the figure turns under it, so
// each vertex in turn sweeps the pole and its image diverges.
const CLIP_RADIUS_FRACTION: f32 = 0.75;

// Floor on the cut, so a close zoom never eats the figure: a unit-circumradius
// polychoron's honest stereographic image already reaches radius ~1.7 at a
// `w = 0.5` vertex.
const CLIP_RADIUS_FLOOR: f32 = 2.5;

// Past this the arcs run into the steep near-pole region, where the fixed
// sample count `push_blended_edge` uses is too coarse and consecutive samples
// jump several-fold, faceting the curve.
const CLIP_RADIUS_MAX: f32 = 10.0;

const LINE_WIDTH_PX: f32 = 1.4;

const BACKGROUND: wgpu::Color = wgpu::Color {
    r: 0.020,
    g: 0.022,
    b: 0.032,
    a: 1.0,
};

const BOOT_ORBIT_DISTANCE: f32 = 6.0;
const BOOT_ORBIT_PITCH: f32 = -0.18;

// Equal magnitudes on a pair of orthogonal planes is what makes the
// exponential land on `Bivector4::exp`'s isoclinic branch.
fn clifford_generator(theta: f32) -> Bivector4 {
    Bivector4::new(theta, 0.0, 0.0, 0.0, 0.0, theta)
}

// `Rotor4::to_mat4` is column-major to match glam, so `Iso4`'s SO(4) matrix
// takes it verbatim.
fn iso_from_rotor(rotor: Rotor4) -> Iso4 {
    Iso4 {
        matrix: Mat4::from_cols_array_2d(&rotor.to_mat4()),
    }
}

// `iso_apply` re-normalizes, so a pose is an S³ point by construction however
// far the angle has run.
fn pose_vertices(iso: Iso4, source: &[Vec4], out: &mut Vec<Vec4>) {
    out.clear();
    out.extend(
        source
            .iter()
            .map(|&v| SphericalS3Embedded.iso_apply(iso, v)),
    );
}

fn clip_radius(camera_distance: f32) -> f32 {
    (camera_distance * CLIP_RADIUS_FRACTION).clamp(CLIP_RADIUS_FLOOR, CLIP_RADIUS_MAX)
}

fn build_wireframe(
    polytope: Polytope4,
    posed: &[Vec4],
    curved: bool,
    clip_radius: f32,
    mesh: &mut LineMesh<3>,
    scratch: &mut Vec<Vec4>,
) {
    mesh.segments.clear();
    mesh.colors.clear();
    mesh.widths.clear();
    let blend = if curved { 1.0 } else { 0.0 };
    for &[i, j] in polytope.topology().edges {
        let a = posed[i as usize];
        let b = posed[j as usize];
        push_blended_edge(
            mesh,
            a,
            b,
            // The posed vertices are points of the ambient S³ itself, so the
            // arc centre is the origin.
            Vec4::ZERO,
            w_depth_color(a.w, 1.0),
            w_depth_color(b.w, 1.0),
            LINE_WIDTH_PX,
            blend,
            &PROJECTION,
            Vec3::ZERO,
            scratch,
            clip_radius,
        );
    }
}

// One entry point so the pose and the mesh always come from the same
// polychoron: a pose left over from a larger one would index past the new
// vertex list.
fn build_frame(
    polytope: Polytope4,
    angle: f32,
    curved: bool,
    clip_radius: f32,
    posed: &mut Vec<Vec4>,
    mesh: &mut LineMesh<3>,
    scratch: &mut Vec<Vec4>,
) {
    let iso = iso_from_rotor(clifford_generator(angle).exp());
    pose_vertices(iso, polytope.topology().vertices, posed);
    build_wireframe(polytope, posed, curved, clip_radius, mesh, scratch);
}

pub(crate) struct S3Scene {
    camera: Camera<EuclideanR3>,
    orbit: OrbitController<EuclideanR3>,
    console: Console<()>,
    lines: LineRasterNode,
    mesh: LineMesh<3>,
    posed: Vec<Vec4>,
    slerp_scratch: Vec<Vec4>,
    polytope: Polytope4,
    /// Accumulated isoclinic angle per invariant plane. The rotor is rebuilt
    /// from this each frame rather than composed incrementally, so a long run
    /// carries no accumulated product drift.
    angle: f32,
    rate: f32,
    spin: bool,
    curved: bool,
}

impl S3Scene {
    pub(crate) fn new(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let mut console = Console::<()>::new();
        loam_app::shell::register_command::<(), crate::shell::Playground>(&mut console);

        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 1.0, BOOT_ORBIT_DISTANCE);
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        orbit.set_orbit(BOOT_ORBIT_DISTANCE, BOOT_ORBIT_PITCH);

        Ok(Self {
            camera,
            orbit,
            console,
            lines: LineRasterNode::new(
                &ctx.rd.device,
                ctx.rd.target_format(),
                DepthMode::Off,
                ctx.rd.sample_count(),
            ),
            mesh: LineMesh::<3>::default(),
            posed: Vec::new(),
            slerp_scratch: Vec::new(),
            polytope: Polytope4::Cell600,
            angle: 0.0,
            rate: DEFAULT_RATE,
            spin: true,
            curved: true,
        })
    }

    fn panel(&mut self, ctx: &egui::Context) {
        egui::Window::new("S³")
            .id(egui::Id::new("s3-scene-controls"))
            .default_pos(egui::pos2(16.0, 48.0))
            .resizable(false)
            .show(ctx, |ui| {
                egui::ComboBox::from_label("polychoron")
                    .selected_text(polytope_label(self.polytope))
                    .show_ui(ui, |ui| {
                        for p in Polytope4::ALL {
                            ui.selectable_value(&mut self.polytope, p, polytope_label(p));
                        }
                    });
                ui.checkbox(&mut self.spin, "Clifford spin (Space)");
                ui.add(
                    egui::Slider::new(&mut self.rate, 0.0..=1.5)
                        .text("rad/s per plane")
                        .fixed_decimals(2),
                );
                ui.checkbox(&mut self.curved, "Great-circle arcs (C)");
                ui.separator();
                let topology = self.polytope.topology();
                ui.label(
                    egui::RichText::new(format!(
                        "{} vertices, {} edges, all at |v| = 1",
                        topology.vertices.len(),
                        topology.edges.len()
                    ))
                    .small()
                    .weak(),
                );
                ui.label(
                    egui::RichText::new(format!(
                        "isoclinic: every point moves {:.2} rad/s, all equally",
                        if self.spin { self.rate } else { 0.0 }
                    ))
                    .small()
                    .weak(),
                );
            });
    }
}

fn polytope_label(p: Polytope4) -> &'static str {
    SHAPE_CATALOG
        .iter()
        .find(|entry| entry.shape.polytope4() == Some(p))
        .expect("the catalog carries every convex regular polychoron")
        .label
}

impl loam_app::shell::Scene for S3Scene {
    fn apply_command(
        &mut self,
        cmd: &loam_app::command::CommandLine,
        _ctx: &mut loam_app::command::CommandCtx<'_>,
    ) -> anyhow::Result<()> {
        self.console.dispatch(&cmd.name, &cmd.arg_refs(), &mut ());
        Ok(())
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        if self.spin {
            self.angle += self.rate * ctx.dt;
            // Wrap at a full turn: the Clifford translation is 2π-periodic in
            // the angle, so an unbounded accumulator only costs f32 precision.
            self.angle = self.angle.rem_euclid(std::f32::consts::TAU);
        }
        let cfg = &ctx.rd.surface_bundle.config;
        self.camera.aspect = cfg.width as f32 / cfg.height.max(1) as f32;
        if !ctx.ui_capture.pointer {
            self.orbit
                .advance(ctx.input, &mut self.camera, &EuclideanR3, ctx.dt);
        }
    }

    fn ui(&mut self, ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {
        self.panel(ctx);
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
            KeyCode::Space => self.spin = !self.spin,
            KeyCode::KeyC => self.curved = !self.curved,
            KeyCode::KeyR => {
                self.angle = 0.0;
                self.rate = DEFAULT_RATE;
                self.orbit.set_orbit(BOOT_ORBIT_DISTANCE, BOOT_ORBIT_PITCH);
            }
            _ => {}
        }
    }

    fn record(&mut self, ctx: &mut RenderCtx<'_>) -> Result<()> {
        let rd: &RenderDevice = ctx.rd;
        let cfg = &rd.surface_bundle.config;

        // This scene owns the clear: nothing runs before it in the frame's
        // encoder, and `LineRasterNode::record` loads rather than clears.
        let _ = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("s3 clear pass"),
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

        build_frame(
            self.polytope,
            self.angle,
            self.curved,
            clip_radius(self.orbit.distance),
            &mut self.posed,
            &mut self.mesh,
            &mut self.slerp_scratch,
        );
        self.lines.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &self.mesh,
            &Projection::Identity,
            1,
        );

        let view_dir = self.camera.view();
        let aspect = cfg.width as f32 / cfg.height.max(1) as f32;
        let view_mat = Mat4::look_to_rh(view_dir.position, view_dir.forward, view_dir.up);
        let proj_mat = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.1, 100.0);
        self.lines.set_camera(
            &rd.queue,
            proj_mat * view_mat,
            Vec2::new(cfg.width as f32, cfg.height as f32),
        );
        self.lines.record(ctx.encoder, ctx.view, None, None);
        Ok(())
    }

    fn title(&self, _fps: f32) -> Cow<'static, str> {
        Cow::Borrowed("polytope playground - S³")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_math::{Rotor, Space};

    const TEST_ANGLE: f32 = 0.7;

    fn posed_at(polytope: Polytope4, angle: f32) -> Vec<Vec4> {
        let iso = iso_from_rotor(clifford_generator(angle).exp());
        let mut out = Vec::new();
        pose_vertices(iso, polytope.topology().vertices, &mut out);
        out
    }

    #[test]
    fn the_isoclinic_rotor_displaces_every_vertex_equally_and_a_simple_one_does_not() {
        let vertices = Polytope4::Cell600.topology().vertices;
        let posed = posed_at(Polytope4::Cell600, TEST_ANGLE);
        let mut spread = 0.0_f32;
        for (v, p) in vertices.iter().zip(&posed) {
            let d = SphericalS3Embedded.distance(*v, *p);
            assert!(
                (d - TEST_ANGLE).abs() < 1e-4,
                "isoclinic displacement {d} should equal the generator angle {TEST_ANGLE}"
            );
            spread = spread.max((d - TEST_ANGLE).abs());
        }
        assert!(spread < 1e-4, "displacement spread {spread} must vanish");

        let simple = iso_from_rotor(Bivector4::new(TEST_ANGLE, 0.0, 0.0, 0.0, 0.0, 0.0).exp());
        let mut simple_posed = Vec::new();
        pose_vertices(simple, vertices, &mut simple_posed);
        let displacements: Vec<f32> = vertices
            .iter()
            .zip(&simple_posed)
            .map(|(v, p)| SphericalS3Embedded.distance(*v, *p))
            .collect();
        let lo = displacements.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = displacements
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            hi - lo > 0.1,
            "a simple rotation must displace vertices unequally, got spread {}",
            hi - lo
        );
    }

    #[test]
    fn the_space_isometry_and_the_rotor_sandwich_agree_on_every_vertex() {
        let rotor = clifford_generator(TEST_ANGLE).exp();
        let iso = iso_from_rotor(rotor);
        for &v in Polytope4::Cell24.topology().vertices {
            let by_space = SphericalS3Embedded.iso_apply(iso, v);
            let by_rotor = Rotor::apply(&rotor, v);
            assert!(
                (by_space - by_rotor).length() < 1e-5,
                "space isometry {by_space:?} should match rotor sandwich {by_rotor:?}"
            );
        }
    }

    #[test]
    fn posed_vertices_stay_on_the_unit_three_sphere_at_every_angle() {
        for step in 0..257 {
            let angle = step as f32 * 0.0245;
            for p in posed_at(Polytope4::Cell24, angle) {
                assert!(
                    (p.length() - 1.0).abs() < 1e-5,
                    "posed vertex off S³ at angle {angle}: |p| = {}",
                    p.length()
                );
            }
        }
    }

    #[test]
    fn the_pose_preserves_every_pairwise_separation() {
        const CUT_LOCUS_MARGIN: f32 = 0.05;
        let vertices = Polytope4::Cell24.topology().vertices;
        let posed = posed_at(Polytope4::Cell24, TEST_ANGLE);
        let mut cut_locus_pairs = 0usize;
        for i in 0..vertices.len() {
            for j in (i + 1)..vertices.len() {
                let dot_before = vertices[i].dot(vertices[j]);
                let dot_after = posed[i].dot(posed[j]);
                assert!(
                    (dot_before - dot_after).abs() < 1e-5,
                    "inner product {dot_before} -> {dot_after} for pair ({i}, {j})"
                );
                let before = SphericalS3Embedded.distance(vertices[i], vertices[j]);
                if before > std::f32::consts::PI - CUT_LOCUS_MARGIN {
                    cut_locus_pairs += 1;
                    continue;
                }
                let after = SphericalS3Embedded.distance(posed[i], posed[j]);
                assert!(
                    (before - after).abs() < 1e-4,
                    "distance {before} -> {after} for vertex pair ({i}, {j})"
                );
            }
        }
        assert_eq!(cut_locus_pairs, vertices.len() / 2);
    }

    #[test]
    fn every_polychoron_has_a_catalog_label() {
        for p in Polytope4::ALL {
            assert!(!polytope_label(p).is_empty());
        }
    }

    #[test]
    fn a_polychoron_switch_reposes_before_it_meshes() {
        let mut posed = Vec::new();
        let mut mesh = LineMesh::<3>::default();
        let mut scratch = Vec::new();
        let sequence = [Polytope4::Cell600, Polytope4::Pentatope, Polytope4::Cell120];
        for polytope in sequence {
            build_frame(
                polytope,
                TEST_ANGLE,
                true,
                f32::INFINITY,
                &mut posed,
                &mut mesh,
                &mut scratch,
            );
            assert_eq!(posed.len(), polytope.topology().vertices.len());
            assert_eq!(
                mesh.segments.len(),
                polytope.topology().edges.len() * crate::consts::SPACE_TESSELLATION_SAMPLES
            );
        }
    }

    fn mesh_length(mesh: &LineMesh<3>) -> f32 {
        mesh.segments
            .iter()
            .map(|(a, b)| (Vec3::from_array(*b) - Vec3::from_array(*a)).length())
            .sum()
    }

    #[test]
    fn great_circle_arcs_are_longer_than_the_chords_through_the_same_seam() {
        let posed = posed_at(Polytope4::Cell24, TEST_ANGLE);
        let mut scratch = Vec::new();
        let mut arcs = LineMesh::<3>::default();
        let mut chords = LineMesh::<3>::default();
        build_wireframe(
            Polytope4::Cell24,
            &posed,
            true,
            f32::INFINITY,
            &mut arcs,
            &mut scratch,
        );
        build_wireframe(
            Polytope4::Cell24,
            &posed,
            false,
            f32::INFINITY,
            &mut chords,
            &mut scratch,
        );
        let (arc_len, chord_len) = (mesh_length(&arcs), mesh_length(&chords));
        assert!(
            arc_len > chord_len * 1.01,
            "arc length {arc_len} should exceed chord length {chord_len}"
        );
    }

    #[test]
    fn every_edge_contributes_geometry_when_nothing_is_clipped() {
        let edges = Polytope4::Cell24.topology().edges.len();
        let posed = posed_at(Polytope4::Cell24, TEST_ANGLE);
        let mut scratch = Vec::new();
        let mut mesh = LineMesh::<3>::default();

        build_wireframe(
            Polytope4::Cell24,
            &posed,
            false,
            f32::INFINITY,
            &mut mesh,
            &mut scratch,
        );
        assert_eq!(mesh.segments.len(), edges);

        build_wireframe(
            Polytope4::Cell24,
            &posed,
            true,
            f32::INFINITY,
            &mut mesh,
            &mut scratch,
        );
        assert_eq!(
            mesh.segments.len(),
            edges * crate::consts::SPACE_TESSELLATION_SAMPLES
        );
        assert_eq!(mesh.colors.len(), mesh.segments.len());
        assert_eq!(mesh.widths.len(), mesh.segments.len());
    }

    #[test]
    fn no_sample_escapes_the_cut_as_vertices_sweep_the_pole() {
        let radius = clip_radius(BOOT_ORBIT_DISTANCE);
        let edges = Polytope4::Cell16.topology().edges.len();
        let mut scratch = Vec::new();
        let mut mesh = LineMesh::<3>::default();
        for step in 0..129 {
            let angle = step as f32 * (std::f32::consts::TAU / 128.0);
            let posed = posed_at(Polytope4::Cell16, angle);
            build_wireframe(
                Polytope4::Cell16,
                &posed,
                true,
                radius,
                &mut mesh,
                &mut scratch,
            );
            assert!(
                mesh.segments.len() >= edges,
                "only {} segments survived the cut at angle {angle}",
                mesh.segments.len()
            );
            for (a, b) in &mesh.segments {
                for p in [Vec3::from_array(*a), Vec3::from_array(*b)] {
                    assert!(p.is_finite(), "non-finite sample {p:?} at angle {angle}");
                    assert!(
                        p.length() <= radius + 1e-3,
                        "sample {p:?} escaped the cut radius {radius} at angle {angle}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_wireframe_build_is_bit_reproducible() {
        let posed_a = posed_at(Polytope4::Cell24, TEST_ANGLE);
        let posed_b = posed_at(Polytope4::Cell24, TEST_ANGLE);
        assert_eq!(posed_a, posed_b);

        let mut scratch = Vec::new();
        let mut first = LineMesh::<3>::default();
        let mut second = LineMesh::<3>::default();
        build_wireframe(
            Polytope4::Cell24,
            &posed_a,
            true,
            clip_radius(BOOT_ORBIT_DISTANCE),
            &mut first,
            &mut scratch,
        );
        build_wireframe(
            Polytope4::Cell24,
            &posed_b,
            true,
            clip_radius(BOOT_ORBIT_DISTANCE),
            &mut second,
            &mut scratch,
        );
        assert_eq!(first.segments, second.segments);
        assert_eq!(first.colors, second.colors);
        assert_eq!(first.widths, second.widths);
    }

    #[test]
    fn the_cut_radius_tracks_the_camera_between_its_floor_and_ceiling() {
        assert_eq!(clip_radius(0.5), CLIP_RADIUS_FLOOR);
        assert_eq!(clip_radius(1000.0), CLIP_RADIUS_MAX);
        let mid = clip_radius(8.0);
        assert!(mid > CLIP_RADIUS_FLOOR && mid < CLIP_RADIUS_MAX);
        assert!((mid - 8.0 * CLIP_RADIUS_FRACTION).abs() < 1e-6);
    }
}
