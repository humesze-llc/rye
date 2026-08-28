use std::collections::HashMap;

use glam::{Vec3, Vec4};
use loam_app::{freecam::Freecam, Camera, OrbitController};
use loam_math::{Bivector, Bivector4, EuclideanR3, Plane4, Projection, Rotor, Rotor4};
use loam_render::raymarch::{BodyUniform, Hyperslice4DNode};
use loam_render::SkyGroundNode;
use loam_shape::polytope::Polytope4;

use crate::catalog::ShapeEntry;
use crate::consts::{BASE_ROTATION_RATE, BODY_SIZE, BODY_X_SPACING, BODY_Y, T_SLIDER_INITIAL};
use crate::director::Playback;
use crate::physics::{BodyPose, PlaygroundPhysics};
use crate::spins::{is_directed, SlotSpins};

pub(crate) use crate::projections::*;
pub(crate) use crate::sections::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum RotationMode {
    Active,
    Composer,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ViewMode {
    Shapes,
    Single,
    Filmstrip,
}

pub(crate) fn render_row_entries<'a>(
    view_mode: ViewMode,
    row: &'a [ShapeEntry],
    strip_subject: &'a ShapeEntry,
) -> &'a [ShapeEntry] {
    match view_mode {
        ViewMode::Single => std::slice::from_ref(strip_subject),
        ViewMode::Shapes | ViewMode::Filmstrip => row,
    }
}

// `bodies_moving` must be read BEFORE the physics step: a body that comes to
// rest during the step still has a final pose to upload, and an at-rest world
// is an exact fixpoint of the integrator, so it has none. The rotor test is
// over the WHOLE row, because each slot carries its own orientation.
pub(crate) fn body_upload_needed(
    spins: &SlotSpins,
    uploaded_rotors: &[Rotor4],
    bodies_moving: bool,
) -> bool {
    bodies_moving || spins.rotors_differ_from(uploaded_rotors)
}

pub(crate) fn set_if_changed<T: PartialEq>(slot: &mut T, value: T) -> bool {
    if *slot == value {
        return false;
    }
    *slot = value;
    true
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub(crate) enum WireframeColorMode {
    #[default]
    VertexGradient,
    UniqueEdge,
    WDepth,
    Active,
}

impl WireframeColorMode {
    pub(crate) const ALL: [Self; 4] = [
        Self::VertexGradient,
        Self::UniqueEdge,
        Self::WDepth,
        Self::Active,
    ];

    pub(crate) fn from_token(token: &str) -> Option<Self> {
        match token {
            "vertex-gradient" => Some(Self::VertexGradient),
            "unique-edge" => Some(Self::UniqueEdge),
            "w-depth" => Some(Self::WDepth),
            "active" => Some(Self::Active),
            _ => None,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::VertexGradient => "Vertex gradient",
            Self::UniqueEdge => "Unique edge",
            Self::WDepth => "W-depth",
            Self::Active => "Active",
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(crate) enum CameraMode {
    #[default]
    Orbit,
    FreeRoam,
}

// Bivector addition within a term is commutative, so plane order inside a term
// is irrelevant; rotor multiplication between terms is not.
#[derive(Clone, Debug, Default)]
pub(crate) struct RotorTerm {
    pub(crate) planes: Vec<Plane4>,
    pub(crate) scalar: Option<f32>,
}

pub(crate) fn render_plane_sum(
    ui: &mut loam_app::egui::Ui,
    planes: &[Plane4],
    mut render_plane: impl FnMut(&mut loam_app::egui::Ui, usize, Plane4),
) {
    let multi = planes.len() > 1;
    if multi {
        ui.monospace("(");
    }
    for (i, plane) in planes.iter().enumerate() {
        if i > 0 {
            ui.monospace("+");
        }
        render_plane(ui, i, *plane);
    }
    if multi {
        ui.monospace(")");
    }
}

pub(crate) fn render_term(term: &RotorTerm) -> String {
    let plane_str = term
        .planes
        .iter()
        .map(|p| p.label())
        .collect::<Vec<_>>()
        .join(" + ");
    let bivec = if term.planes.len() > 1 {
        format!("({plane_str})")
    } else {
        plane_str
    };
    match term.scalar {
        Some(phi) => format!("{:.0}° · {}", phi.to_degrees(), bivec),
        None => bivec,
    }
}

pub(crate) fn render_bivector_sum(parts: &[String]) -> Option<String> {
    match parts {
        [] => None,
        [only] => Some(only.clone()),
        many => Some(format!("({})", many.join(" + "))),
    }
}

pub(crate) fn active_plane_angle(base: f32, active: bool, t: f32) -> f32 {
    base + if active { t * BASE_ROTATION_RATE } else { 0.0 }
}

// The ORDERED PRODUCT `∏ᵢ exp(planeᵢ · active_plane_angle(base[i], active[i],
// t))` in `Plane4::ALL` order. A product (independent sliders), not `exp(sum)`,
// which would reintroduce BCH coupling.
pub(crate) fn compose_active_rotor(base_angles: &[f32; 6], active: &[bool; 6], t: f32) -> Rotor4 {
    let mut r = Rotor4::IDENTITY;
    for i in 0..6 {
        let angle = active_plane_angle(base_angles[i], active[i], t);
        if angle != 0.0 {
            let bivec = Plane4::ALL[i].unit_bivector() * angle;
            r = bivec.exp() * r;
        }
    }
    r.normalize()
}

pub(crate) fn angular_velocity_from_seq(seq: &[RotorTerm], rate_scale: f32) -> Bivector4 {
    let mut omega = Bivector4::ZERO;
    for term in seq {
        let phi = term.scalar.unwrap_or(1.0);
        for plane in &term.planes {
            omega = omega + plane.unit_bivector() * phi;
        }
    }
    omega * (BASE_ROTATION_RATE * rate_scale)
}

// Applying mid-render lays out the rest of the frame against the new state
// while the rows already emitted used the old.
#[derive(Clone, Debug)]
pub(crate) enum DeferredAction {
    DraftPush(Plane4),
    SeqCommitDraft,
    DraftClear,
    SeqPushTerm(RotorTerm),
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum DragPayload {
    Term(usize),
    Entry(usize, usize),
}

pub(crate) fn body_position(slot: usize, n: usize) -> [f32; 4] {
    let x = (slot as f32 - (n as f32 - 1.0) * 0.5) * BODY_X_SPACING;
    [x, BODY_Y, 0.0, 0.0]
}

pub(crate) fn sdf_body_uniform(
    physics: &PlaygroundPhysics,
    entry: &ShapeEntry,
    slot: usize,
    slots: usize,
    spin: Rotor4,
    size: f32,
    surface_mode: SurfaceMode,
) -> BodyUniform {
    // The 120-cell and 600-cell have NO authoritative SDF: their
    // `cell{120,600}_face_planes` are the known-wrong dual-vertex
    // approximation (see `loam_shape::polytope_geom`), wrong on 96 normals.
    // Never raymarch them, on any platform or mode.
    if matches!(
        entry.shape.polytope4(),
        Some(Polytope4::Cell120 | Polytope4::Cell600)
    ) {
        return BodyUniform::default();
    }
    if !surface_mode.uses_sdf_for_polychora() && entry.shape.polytope4().is_some() {
        return BodyUniform::default();
    }
    let pose = physics.pose(slot, slots, spin);
    BodyUniform::polytope_with_rotor(
        pose.position.to_array(),
        entry.shape.shape_id(),
        size,
        pose.rotor,
        entry.body_color,
    )
}

// Each reader takes ALL of its per-body geometry from one `PlaygroundPhysics`,
// so no pass can quietly fall back to the authored spin over the static layout.
pub(crate) struct RowFrame<'a> {
    pub(crate) physics: &'a PlaygroundPhysics,
    pub(crate) row: &'a [ShapeEntry],
    pub(crate) spins: &'a SlotSpins,
    pub(crate) body_size: f32,
    pub(crate) projection: Projection<4>,
    pub(crate) w_slice: f32,
    pub(crate) camera_distance: f32,
}

impl RowFrame<'_> {
    pub(crate) fn pose(&self, slot: usize) -> BodyPose {
        self.physics
            .pose(slot, self.row.len(), self.spins.rotor(slot))
    }

    pub(crate) fn body_local(
        &self,
        slot: usize,
        canonical: &[Vec4],
        scale: f32,
        out: &mut Vec<Vec4>,
    ) -> Vec3 {
        self.physics.body_frame(
            slot,
            self.row.len(),
            self.spins.rotor(slot),
            canonical,
            scale,
            out,
        )
    }

    pub(crate) fn anchor_r3(&self, slot: usize, canonical: Vec4) -> Vec3 {
        let pose = self.pose(slot);
        <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(
            pose.body_local(canonical, self.body_size),
            &self.projection,
        ) + pose.position_r3()
    }
}

pub(crate) struct Demo {
    // The row's pose and collider store, not a simulation: no path in this
    // scene writes a body velocity, so `step` never leaves `at_rest`.
    pub(crate) physics: PlaygroundPhysics,
    pub(crate) left_was_down: bool,
    pub(crate) gimbal: crate::hypergimbal::GimbalUi,
    pub(crate) gimbal_node: loam_render::LineRasterNode,
    pub(crate) camera: Camera<EuclideanR3>,
    pub(crate) orbit: OrbitController<EuclideanR3>,
    pub(crate) freecam: Freecam,
    pub(crate) camera_mode: CameraMode,
    pub(crate) node: Hyperslice4DNode,
    /// Owns the frame's colour and depth clear, so it is recorded first and
    /// unconditionally; every later pass loads.
    pub(crate) sky_ground: SkyGroundNode,
    pub(crate) sdf_upload_pending: bool,
    pub(crate) uploaded_rotors: Vec<Rotor4>,
    pub(crate) section_edges: loam_render::LineRasterNode,
    pub(crate) parent_wireframe: loam_render::LineRasterNode,
    pub(crate) wireframe_enabled: bool,
    pub(crate) wireframe_nearest_active: bool,
    pub(crate) cross_section: SectionLayer,
    pub(crate) projected_cap: SectionLayer,
    pub(crate) wireframe_color_mode: WireframeColorMode,
    pub(crate) wireframe_projection: WireframeProjection,
    pub(crate) schlegel_params: Option<SchlegelParams>,
    pub(crate) stereographic_pole: glam::Vec4,
    pub(crate) wireframe_hyperslice: bool,
    pub(crate) wireframe_hyperslice_thickness: f32,
    pub(crate) wireframe_width_px: f32,
    pub(crate) wireframe_alpha: f32,
    pub(crate) unique_edge_palette_cache: HashMap<Polytope4, Vec<[f32; 4]>>,
    pub(crate) cell_centers_cache: HashMap<Polytope4, Vec<Vec4>>,
    pub(crate) surface_scale: f32,
    pub(crate) floor_enabled: bool,
    pub(crate) environment: crate::environment::Environment,
    pub(crate) section_faces: loam_render::TriangleRasterNode,
    pub(crate) section_faces_translucent: loam_render::TriangleRasterNode,
    pub(crate) points_node: loam_render::PointRasterNode,
    pub(crate) points_enabled: bool,
    pub(crate) points_show_vertices: bool,
    pub(crate) points_show_cell_centers: bool,
    pub(crate) points_size_px: f32,
    pub(crate) points_mesh_scratch: loam_shape::PointMesh<3>,
    pub(crate) section_faces_depth: Option<loam_render::DepthBuffer>,
    pub(crate) section_world_vertices_scratch: Vec<glam::Vec4>,
    pub(crate) section_faces_mesh_scratch: loam_shape::TriangleMesh<3>,
    pub(crate) section_faces_projected_scratch: loam_shape::TriangleMesh<3>,
    pub(crate) section_clip_projected_scratch: Vec<glam::Vec3>,
    pub(crate) body_uniform_scratch: Vec<BodyUniform>,
    pub(crate) slerp_scratch: Vec<glam::Vec4>,
    /// Reused across frames: worst measured case (an eight-slot row of
    /// 600-cells, both perimeters on) is ~12k segments, about 0.7 MB.
    pub(crate) wireframe_section_edges_scratch: loam_shape::LineMesh<3>,
    pub(crate) body_perimeter_scratch: loam_shape::LineMesh<3>,
    // The two are called from different render passes, which is why
    // `Demo::record_section_faces` running before
    // `Demo::record_wireframe_overlay` is load-bearing: each takes this buffer
    // and restores it before the other runs.
    pub(crate) section_cap_scratch: loam_shape::polytope::SectionScratch,
    pub(crate) wireframe_parent_lines_scratch: loam_shape::LineMesh<3>,
    pub(crate) overlay_local_vertices_scratch: Vec<glam::Vec4>,
    pub(crate) overlay_center_locals_scratch: Vec<glam::Vec4>,
    pub(crate) overlay_cell_strengths_scratch: Vec<f32>,
    pub(crate) surface_mode: SurfaceMode,
    pub(crate) row: Vec<ShapeEntry>,

    pub(crate) w_slice: f32,
    pub(crate) slider_up_held: bool,
    pub(crate) slider_down_held: bool,
    pub(crate) slider_left_held: bool,
    pub(crate) slider_right_held: bool,

    pub(crate) rotate: bool,
    pub(crate) spins: SlotSpins,
    pub(crate) playback: Option<Playback>,
    pub(crate) rate_scale: f32,
    pub(crate) rot_time: f32,
    pub(crate) t_slider_max: f32,

    pub(crate) expanded: bool,

    pub(crate) show_help: bool,
    pub(crate) show_render_panel: bool,
    pub(crate) example_callout: loam_egui::CalloutState,

    pub(crate) mode_annotation_open: loam_egui::CalloutState,

    pub(crate) show_formula: bool,

    pub(crate) show_controls: bool,

    pub(crate) show_text_hud: bool,

    pub(crate) view_mode: ViewMode,
    pub(crate) strip_w: bool,
    pub(crate) strip_t: bool,
    pub(crate) strip_swap_axes: bool,
    pub(crate) strip_count_w: usize,
    pub(crate) strip_count_t: usize,
    pub(crate) strip_t_extent: f32,
    pub(crate) strip_subject: ShapeEntry,

    pub(crate) rotation_mode: RotationMode,

    pub(crate) pending_mode: Option<RotationMode>,

    pub(crate) pending_view_mode: Option<ViewMode>,

    pub(crate) pending_actions: Vec<DeferredAction>,

    pub(crate) seq: Vec<RotorTerm>,
    pub(crate) draft: Vec<Plane4>,

    pub(crate) formula_input: String,
    pub(crate) formula_error: Option<String>,
}

impl Demo {
    pub(crate) fn compose_omega(&self) -> Bivector4 {
        let mut omega = Bivector4::ZERO;
        for term in &self.seq {
            let phi = term.scalar.unwrap_or(1.0);
            for plane in &term.planes {
                omega = omega + plane.unit_bivector() * phi;
            }
        }
        omega
    }

    pub(crate) fn omega_animation(&self) -> Bivector4 {
        match self.rotation_mode {
            RotationMode::Active => {
                let mut omega = Bivector4::ZERO;
                for (i, &on) in self.spins.selected_spin().active.iter().enumerate() {
                    if on {
                        omega = omega + Plane4::ALL[i].unit_bivector();
                    }
                }
                omega * BASE_ROTATION_RATE
            }
            RotationMode::Composer => angular_velocity_from_seq(&self.seq, 1.0),
        }
    }

    pub(crate) fn selected_slot(&self) -> usize {
        self.spins.selected()
    }

    pub(crate) fn selected_rotor(&self) -> Rotor4 {
        self.spins.rotor(self.spins.selected())
    }

    pub(crate) fn active_angle_at(&self, plane_idx: usize, t: f32) -> f32 {
        self.spins.selected_spin().angle_at(plane_idx, t)
    }

    pub(crate) fn active_displayed_angle(&self, plane_idx: usize) -> f32 {
        self.active_angle_at(plane_idx, self.rot_time)
    }

    pub(crate) fn rotor_at_time(&self, t: f32) -> Rotor4 {
        match self.rotation_mode {
            RotationMode::Active => self.spins.selected_spin().active_rotor_at(t),
            RotationMode::Composer => (self.omega_animation() * t).exp().normalize(),
        }
    }

    pub(crate) fn recompose_spins_at(&mut self, t: f32) {
        let directed = self.playback.as_ref().map_or(&[][..], Playback::directed);
        match self.rotation_mode {
            RotationMode::Active => self.spins.recompose_active(t, directed),
            RotationMode::Composer => {
                if !is_directed(directed, self.spins.selected()) {
                    let rotor = (self.omega_animation() * t).exp().normalize();
                    self.spins.selected_spin_mut().rotor = rotor;
                }
            }
        }
    }

    pub(crate) fn effective_body_size(&self) -> f32 {
        BODY_SIZE * self.surface_scale
    }

    // A 120-cell or 600-cell in the live SDF kernel crashes the browser tab;
    // see [`row_blocks_sdf`].
    pub(crate) fn sdf_blocked_by_heavy_polychora(&self) -> bool {
        row_blocks_sdf(self.render_row())
    }

    pub(crate) fn effective_w_range(&self) -> f32 {
        crate::consts::W_RANGE * self.surface_scale
    }

    pub(crate) fn render_row(&self) -> &[ShapeEntry] {
        render_row_entries(self.view_mode, &self.row, &self.strip_subject)
    }

    pub(crate) fn row_frame(&self) -> RowFrame<'_> {
        RowFrame {
            physics: &self.physics,
            row: self.render_row(),
            spins: &self.spins,
            body_size: self.effective_body_size(),
            projection: self.resolved_wireframe_projection(),
            w_slice: self.w_slice,
            camera_distance: self.camera_distance_to_focus(),
        }
    }

    pub(crate) fn schlegel_subject(&self) -> Option<Polytope4> {
        self.render_row().iter().find_map(|e| e.shape.polytope4())
    }

    fn schlegel_slot(&self) -> Option<usize> {
        self.render_row()
            .iter()
            .position(|e| e.shape.polytope4().is_some())
    }

    // Call at every cell-SELECT point so the per-frame path never reruns the
    // `LazyLock`-backed [`Polytope4::face_planes`] fit.
    pub(crate) fn resolve_schlegel_cache(&mut self) {
        let (projection, cache) =
            synced_schlegel_projection(self.wireframe_projection, self.schlegel_subject());
        self.wireframe_projection = projection;
        self.schlegel_params = cache;
    }

    pub(crate) fn resolved_wireframe_projection(&self) -> Projection<4> {
        match self.wireframe_projection {
            WireframeProjection::Schlegel { .. } => {
                match (self.schlegel_params, self.schlegel_slot()) {
                    (Some(p), Some(slot)) => {
                        let body_size = self.effective_body_size();
                        let subject = self.spins.rotor(slot);
                        let cell_normal = subject.apply(p.cell_normal);
                        let basis = p.cell_basis.map(|axis| subject.apply(axis));
                        Projection::schlegel_with_basis(
                            cell_normal,
                            p.cell_offset * body_size,
                            p.viewpoint_distance * body_size,
                            basis,
                        )
                    }
                    _ => Projection::Identity,
                }
            }
            WireframeProjection::Stereographic => Projection::Stereographic {
                pole: self.stereographic_pole,
            },
            other => other.to_projection(),
        }
    }

    pub(crate) fn hyperslice_cull_active(&self) -> bool {
        hyperslice_cull_active(self.wireframe_hyperslice, self.wireframe_projection)
    }

    pub(crate) fn toggle_selected_plane(&mut self, plane_idx: usize) {
        let spin = self.spins.selected_spin_mut();
        spin.active[plane_idx] = !spin.active[plane_idx];
    }

    pub(crate) fn apply_selected_active_edit(&mut self) {
        let t = self.rot_time;
        let spin = self.spins.selected_spin_mut();
        spin.rotor = spin.active_rotor_at(t);
        self.rebuild_bodies();
    }

    pub(crate) fn rebuild_bodies(&mut self) {
        self.sdf_upload_pending = true;
        let n = self.render_row().len();
        let size = self.effective_body_size();
        self.sync_physics_row();
        self.spins.record_rotors(&mut self.uploaded_rotors);
        let mut scratch = std::mem::take(&mut self.body_uniform_scratch);
        scratch.clear();
        for slot in 0..n {
            let entry = &self.render_row()[slot];
            scratch.push(sdf_body_uniform(
                &self.physics,
                entry,
                slot,
                n,
                self.spins.rotor(slot),
                size,
                self.surface_mode,
            ));
        }
        self.node.set_bodies(&scratch);
        self.body_uniform_scratch = scratch;
    }

    // Called from `Demo::update` AHEAD of the physics step, so a tick collides
    // the shape the frame draws (up to 0.088 of rim lag at `rate_scale` 4).
    pub(crate) fn sync_physics_row(&mut self) {
        let size = self.effective_body_size();
        let row = render_row_entries(self.view_mode, &self.row, &self.strip_subject);
        self.spins.sync(row.len());
        self.physics.sync(row, &self.spins, size);
    }

    pub(crate) fn formula_string(&self) -> String {
        let parts: Vec<String> = match self.rotation_mode {
            RotationMode::Active => Plane4::ALL
                .iter()
                .zip(self.spins.selected_spin().active.iter())
                .filter(|(_, on)| **on)
                .map(|(p, _)| p.label().to_string())
                .collect(),
            RotationMode::Composer => self.seq.iter().map(render_term).collect(),
        };
        match render_bivector_sum(&parts) {
            Some(bivec) => format!(
                "exp({} · {:.2}·t)",
                bivec,
                BASE_ROTATION_RATE / std::f32::consts::TAU
            ),
            None => String::new(),
        }
    }

    pub(crate) fn reset(&mut self) {
        self.rotate = false;
        self.w_slice = 0.0;
        self.rate_scale = 1.0;
        self.spins.reset();
        self.rot_time = 0.0;
        // A loaded timeline is the other half of `rot_time`, so the reset that
        // zeroes one has to rewind the other or half the row would restart.
        if let Some(playback) = &mut self.playback {
            playback.rewind();
        }
        self.t_slider_max = T_SLIDER_INITIAL;
        self.cross_section = SectionLayer::CROSS_SECTION_DEFAULT;
        self.projected_cap = SectionLayer::PROJECTED_CAP_DEFAULT;
        self.draft.clear();
        // Drop a handle drag in progress: its slot names a body the respawn
        // below despawns.
        self.gimbal.drag = None;
        let slots = self.render_row().len();
        self.physics.respawn(slots, self.effective_body_size());
        self.rebuild_bodies();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        active_plane_angle, body_position, body_upload_needed, compose_active_rotor,
        mode_annotation, render_row_entries, resolve_schlegel_params, row_blocks_sdf,
        sdf_body_uniform, section_layer_projection, set_if_changed, synced_schlegel_projection,
        SectionLayer, SurfaceMode, ViewMode, WireframeProjection, BASE_ROTATION_RATE, BODY_SIZE,
        STEREOGRAPHIC_DEFAULT_POLE,
    };
    use crate::catalog::ShapeEntry;
    use crate::physics::{composed_rotor, PlaygroundPhysics};
    use crate::spins::SlotSpins;
    use glam::Vec4;
    use loam_math::{Bivector, EuclideanR4, Plane4, Projection, Rotor, Rotor4};
    use loam_render::raymarch::{BodyUniform, RaymarchShape};
    use loam_shape::polytope::Polytope4;
    use std::collections::HashSet;

    fn entry(shape: RaymarchShape) -> ShapeEntry {
        ShapeEntry {
            shape,
            body_color: [0.5, 0.5, 0.5],
            label: "test",
            long_name: "test shape",
        }
    }

    const NONE: [bool; 6] = [false; 6];

    fn rotor_close(a: Rotor4, b: Rotor4, eps: f32) -> bool {
        (a.s - b.s).abs() < eps
            && (a.xy - b.xy).abs() < eps
            && (a.xz - b.xz).abs() < eps
            && (a.xw - b.xw).abs() < eps
            && (a.yz - b.yz).abs() < eps
            && (a.yw - b.yw).abs() < eps
            && (a.zw - b.zw).abs() < eps
            && (a.xyzw - b.xyzw).abs() < eps
    }

    #[test]
    fn active_plane_angle_adds_spin_only_when_active() {
        assert_eq!(active_plane_angle(0.5, false, 3.0), 0.5);
        assert_eq!(
            active_plane_angle(0.5, true, 2.0),
            0.5 + 2.0 * BASE_ROTATION_RATE
        );
        assert_eq!(active_plane_angle(0.5, true, 0.0), 0.5);
    }

    #[test]
    fn toggling_active_preserves_displayed_angle() {
        // Flipping `active` re-solves `base` so the displayed angle is continuous
        // across the toggle: `base = displayed_before - spin(active_after)`.
        let t = 7.5_f32;
        let resolve = |base_old: f32, active_before: bool| {
            let displayed_before = active_plane_angle(base_old, active_before, t);
            let active_after = !active_before;
            let spin_after = if active_after {
                t * BASE_ROTATION_RATE
            } else {
                0.0
            };
            let base_new = displayed_before - spin_after;
            active_plane_angle(base_new, active_after, t)
        };
        let base_off = 0.42_f32;
        let displayed_off = active_plane_angle(base_off, false, t);
        assert!(
            (resolve(base_off, false) - displayed_off).abs() < 1e-6,
            "toggle on changed the displayed angle"
        );
        let base_on = -1.1_f32;
        let displayed_on = active_plane_angle(base_on, true, t);
        assert!(
            (resolve(base_on, true) - displayed_on).abs() < 1e-6,
            "toggle off changed the displayed angle"
        );
    }

    #[test]
    fn compose_all_zero_is_identity() {
        let r = compose_active_rotor(&[0.0; 6], &NONE, 0.0);
        assert!(rotor_close(r, Rotor4::IDENTITY, 1e-6), "got {r:?}");
    }

    #[test]
    fn compose_is_always_unit_norm() {
        let base = [0.3, -1.1, 2.0, 0.7, -0.4, 1.6];
        let active = [true, false, true, true, false, true];
        for &t in &[0.0_f32, 0.5, 3.0, 50.0] {
            let n2 = compose_active_rotor(&base, &active, t).norm_squared();
            assert!((n2 - 1.0).abs() < 1e-5, "t={t} norm_squared={n2}");
        }
    }

    #[test]
    fn compose_single_plane_equals_direct_exp() {
        let theta = 0.8_f32;
        let mut base = [0.0; 6];
        base[2] = theta; // Plane4::Xw
        let composed = compose_active_rotor(&base, &NONE, 0.0);
        let direct = (Plane4::Xw.unit_bivector() * theta).exp().normalize();
        assert!(
            rotor_close(composed, direct, 1e-6),
            "{composed:?} vs {direct:?}"
        );
    }

    #[test]
    fn compose_orthogonal_pair_is_order_independent() {
        let (a, b) = (0.6_f32, -0.9_f32);
        let mut base = [0.0; 6];
        base[0] = a; // xy
        base[5] = b; // zw
        let composed = compose_active_rotor(&base, &NONE, 0.0);
        let xy = (Plane4::Xy.unit_bivector() * a).exp();
        let zw = (Plane4::Zw.unit_bivector() * b).exp();
        let reverse = (xy * zw).normalize();
        assert!(
            rotor_close(composed, reverse, 1e-6),
            "{composed:?} vs {reverse:?}"
        );
    }

    #[test]
    fn row_blocks_sdf_only_for_heavy_polychora() {
        assert!(!row_blocks_sdf(&[]));
        let light = [
            entry(RaymarchShape::Polytope(Polytope4::Tesseract)),
            entry(RaymarchShape::Polytope(Polytope4::Cell24)),
        ];
        assert!(!row_blocks_sdf(&light));
        let with_120 = [
            entry(RaymarchShape::Polytope(Polytope4::Tesseract)),
            entry(RaymarchShape::Polytope(Polytope4::Cell120)),
        ];
        assert!(row_blocks_sdf(&with_120));
        let with_600 = [entry(RaymarchShape::Polytope(Polytope4::Cell600))];
        assert!(row_blocks_sdf(&with_600));
    }

    #[test]
    fn annotation_text_present_per_mode() {
        assert!(
            mode_annotation(WireframeProjection::Shadow).is_none(),
            "drop-w must not annotate the default view"
        );

        let cases = [
            WireframeProjection::WPinhole,
            WireframeProjection::Schlegel { cell_index: 0 },
            WireframeProjection::Stereographic,
            WireframeProjection::Hyperslice,
        ];
        let mut bodies = HashSet::new();
        for projection in cases {
            let annotation = mode_annotation(projection)
                .unwrap_or_else(|| panic!("{projection:?} must annotate"));
            assert!(
                !annotation.title.is_empty(),
                "{projection:?}: title must be non-empty"
            );
            assert!(
                !annotation.body.is_empty(),
                "{projection:?}: body must be non-empty"
            );
            assert!(
                bodies.insert(annotation.body.clone()),
                "{projection:?}: body duplicates another mode's copy"
            );
        }
    }

    #[test]
    fn sdf_gate_follows_single_subject_not_row() {
        let heavy = entry(RaymarchShape::Polytope(Polytope4::Cell600));
        let light = entry(RaymarchShape::Polytope(Polytope4::Tesseract));

        let light_row = [light, entry(RaymarchShape::Polytope(Polytope4::Cell24))];
        assert!(
            row_blocks_sdf(render_row_entries(ViewMode::Single, &light_row, &heavy)),
            "heavy Single subject blocks SDF even over an all-light row",
        );
        let heavy_row = [entry(RaymarchShape::Polytope(Polytope4::Cell120))];
        assert!(
            !row_blocks_sdf(render_row_entries(ViewMode::Single, &heavy_row, &light)),
            "light Single subject keeps SDF available even over a heavy row",
        );
        assert!(
            row_blocks_sdf(render_row_entries(ViewMode::Shapes, &heavy_row, &light)),
            "Shapes mode blocks SDF on a heavy row member",
        );
    }

    #[test]
    fn wireframe_projection_from_token_round_trips() {
        assert_eq!(
            WireframeProjection::ALL.len(),
            4,
            "ALL must list every selectable projection mode (Schlegel is excluded)"
        );
        assert_eq!(
            WireframeProjection::from_token("schlegel"),
            None,
            "Schlegel is deliberately not a selectable playground mode"
        );
        for mode in WireframeProjection::ALL {
            let token = match mode {
                WireframeProjection::Shadow => "shadow",
                WireframeProjection::WPinhole => "w-pinhole",
                WireframeProjection::Stereographic => "stereographic",
                WireframeProjection::Hyperslice => "hyperslice",
                WireframeProjection::Schlegel { .. } => {
                    unreachable!("Schlegel must not be in ALL while it is unwired")
                }
            };
            assert_eq!(
                WireframeProjection::from_token(token),
                Some(mode),
                "token `{token}` must parse back to {mode:?}"
            );
        }
        assert_eq!(
            WireframeProjection::Shadow.to_projection(),
            Projection::Identity
        );
        assert_eq!(
            WireframeProjection::WPinhole.to_projection(),
            Projection::Perspective4D {
                focal_distance: 2.0
            }
        );
        assert_eq!(
            WireframeProjection::Stereographic.to_projection(),
            Projection::Stereographic {
                pole: STEREOGRAPHIC_DEFAULT_POLE
            }
        );
        assert_eq!(
            WireframeProjection::Hyperslice.to_projection(),
            Projection::Identity
        );
        assert_eq!(
            WireframeProjection::Schlegel { cell_index: 0 }.to_projection(),
            Projection::Identity
        );
    }

    #[test]
    fn stereographic_plus_w_pole_scale_depends_only_on_w() {
        let pole = STEREOGRAPHIC_DEFAULT_POLE;
        let p = Vec4::new(0.8, 0.0, 0.0, 0.6);
        let q = Vec4::new(0.0, 0.48, 0.64, 0.6);
        assert!((p.w - q.w).abs() < 1e-6, "fixture points must share w");
        let denom_p = 1.0 - p.dot(pole);
        let denom_q = 1.0 - q.dot(pole);
        assert!(
            (denom_p - denom_q).abs() < 1e-6,
            "+w pole must give equal scale at equal w ({denom_p} vs {denom_q})"
        );
        let off_w = Vec4::splat(0.5);
        assert!(
            (1.0 - p.dot(off_w) - (1.0 - q.dot(off_w))).abs() > 1e-3,
            "off-w pole must spread a single w-slice (the skew we reverted)"
        );
    }

    #[test]
    fn schlegel_params_from_face_planes_not_dual() {
        use loam_physics::euclidean_r4::cell600_face_planes;
        let polytope = Polytope4::Cell600;
        let (topo_normals, _) = polytope.face_planes();
        let (dual_normals, _) = cell600_face_planes();
        let divergent = (0..topo_normals.len() as u32).find(|&i| {
            let n = topo_normals[i as usize];
            dual_normals
                .iter()
                .all(|d| (n - *d).length() > 1e-3 && (n + *d).length() > 1e-3)
        });
        let cell_index = divergent.expect(
            "the 600-cell must have a golden-ratio face that diverges from the dual-vertex set",
        );
        let params = resolve_schlegel_params(polytope, cell_index);
        assert_eq!(params.cell_normal, topo_normals[cell_index as usize]);
        for d in &dual_normals {
            let n = params.cell_normal;
            assert!(
                (n - *d).length() > 1e-3 && (n + *d).length() > 1e-3,
                "resolved Schlegel normal must not coincide with any dual-vertex normal"
            );
        }
    }

    #[test]
    fn schlegel_cell_index_clamped_to_cell_count() {
        let polytope = Polytope4::Pentatope; // 5 cells.
        let last = polytope.cell_count() as u32 - 1;
        let params = resolve_schlegel_params(polytope, 9999);
        assert_eq!(params.cell_index, last);
        let at_last = resolve_schlegel_params(polytope, last);
        assert_eq!(params, at_last);
    }

    #[test]
    fn schlegel_resolve_syncs_carried_index_to_clamp() {
        let polytope = Polytope4::Pentatope; // 5 cells; indices 0..=4.
        let last = polytope.cell_count() as u32 - 1;
        let overrun = WireframeProjection::Schlegel { cell_index: 300 };
        let (projection, cache) = synced_schlegel_projection(overrun, Some(polytope));
        let cache = cache.expect("a Schlegel mode with a subject must produce a cache");
        match projection {
            WireframeProjection::Schlegel { cell_index } => {
                assert_eq!(
                    cell_index, cache.cell_index,
                    "carried index must match cache"
                );
                assert_eq!(cell_index, last, "overrun must clamp to the last cell");
            }
            other => panic!("Schlegel input must stay Schlegel, got {other:?}"),
        }
        let (again, _) = synced_schlegel_projection(projection, Some(polytope));
        assert_eq!(again, projection, "re-resolve must be a fixed point");
    }

    #[test]
    fn synced_schlegel_passes_through_non_schlegel_and_subjectless() {
        let (proj, cache) =
            synced_schlegel_projection(WireframeProjection::Stereographic, Some(Polytope4::Cell24));
        assert_eq!(proj, WireframeProjection::Stereographic);
        assert!(cache.is_none(), "non-Schlegel mode must clear the cache");
        let schlegel = WireframeProjection::Schlegel { cell_index: 7 };
        let (proj, cache) = synced_schlegel_projection(schlegel, None);
        assert_eq!(proj, schlegel, "no subject means no clamp, index stays put");
        assert!(cache.is_none(), "no subject means no cache");
    }

    #[test]
    fn schlegel_resolution_is_bit_deterministic() {
        for polytope in Polytope4::ALL {
            let cell_index = (polytope.cell_count() / 2) as u32;
            let a = resolve_schlegel_params(polytope, cell_index);
            let b = resolve_schlegel_params(polytope, cell_index);
            assert_eq!(a.cell_normal, b.cell_normal, "{polytope:?} normal");
            assert_eq!(a.cell_basis, b.cell_basis, "{polytope:?} basis");
            assert_eq!(a.cell_offset, b.cell_offset, "{polytope:?} offset");
            assert_eq!(
                a.viewpoint_distance, b.viewpoint_distance,
                "{polytope:?} viewpoint"
            );
        }
    }

    #[test]
    fn schlegel_cell_basis_spans_chosen_cell() {
        for polytope in Polytope4::ALL {
            let cell_index = (polytope.cell_count() / 2) as u32;
            let params = resolve_schlegel_params(polytope, cell_index);

            for (i, axis) in params.cell_basis.iter().enumerate() {
                assert!(
                    (axis.length() - 1.0).abs() < 1e-5,
                    "{polytope:?} basis axis {i} must be unit"
                );
                assert!(
                    axis.dot(params.cell_normal).abs() < 1e-5,
                    "{polytope:?} basis axis {i} must be in the cell plane"
                );
            }
            for i in 0..3 {
                for j in (i + 1)..3 {
                    assert!(
                        params.cell_basis[i].dot(params.cell_basis[j]).abs() < 1e-5,
                        "{polytope:?} basis axes {i}/{j} must be orthogonal"
                    );
                }
            }

            let topo = polytope.topology();
            let cell = topo.cells[params.cell_index as usize];
            let anchor = topo.vertices[cell[0] as usize];
            for &vi in cell {
                let delta = topo.vertices[vi as usize] - anchor;
                let reconstructed = params
                    .cell_basis
                    .iter()
                    .fold(Vec4::ZERO, |acc, &axis| acc + delta.dot(axis) * axis);
                let residual = delta - reconstructed;
                assert!(
                    residual.length() < 5e-4,
                    "{polytope:?} cell vertex {vi} must reconstruct in the basis"
                );
            }
        }
    }

    #[test]
    fn schlegel_frame_rotates_with_body() {
        let polytope = Polytope4::Tesseract;
        let cell_index = 0;
        let params = resolve_schlegel_params(polytope, cell_index);
        let rot = (Plane4::Xw.unit_bivector() * 0.7).exp().normalize();
        let rotated = rot.apply(params.cell_normal);
        assert!(
            (rotated - params.cell_normal).length() > 1e-3,
            "rotation must actually move the normal for this test to bite"
        );
        assert!((rotated.length() - 1.0).abs() < 1e-5);
        let rotated_basis = params.cell_basis.map(|axis| rot.apply(axis));
        for (i, axis) in rotated_basis.iter().enumerate() {
            assert!((axis.length() - 1.0).abs() < 1e-5, "axis {i} unit");
            assert!(
                axis.dot(rotated).abs() < 1e-5,
                "axis {i} remains perpendicular to the rotated normal"
            );
        }
        let topo = polytope.topology();
        let cell = topo.cells[cell_index as usize];
        let anchor = topo.vertices[cell[0] as usize];
        let rotated_anchor = rot.apply(anchor);
        for &vi in cell {
            let v = topo.vertices[vi as usize];
            let lhs = rotated.dot(rot.apply(v));
            assert!(
                (lhs - params.cell_offset).abs() < 1e-4,
                "rotated cell vertex must stay on the rotated boundary hyperplane: {lhs} vs {}",
                params.cell_offset
            );
            let delta = v - anchor;
            let rotated_delta = rot.apply(v) - rotated_anchor;
            for (axis, rotated_axis) in params.cell_basis.iter().zip(rotated_basis) {
                let want = delta.dot(*axis);
                let got = rotated_delta.dot(rotated_axis);
                assert!(
                    (got - want).abs() < 1e-5,
                    "rotated basis coordinates must match canonical coordinates"
                );
            }
        }
    }

    #[test]
    fn single_mode_renders_one_subject() {
        let subject = entry(RaymarchShape::Polytope(Polytope4::Cell600));
        let row = [
            entry(RaymarchShape::Polytope(Polytope4::Tesseract)),
            entry(RaymarchShape::Polytope(Polytope4::Cell24)),
            entry(RaymarchShape::Polytope(Polytope4::Pentatope)),
        ];
        let single = render_row_entries(ViewMode::Single, &row, &subject);
        assert_eq!(single.len(), 1, "Single renders exactly one body");
        assert_eq!(single[0], subject, "the single body is the strip_subject");
        assert!(
            !row.contains(&subject),
            "test setup: subject must differ from every row entry",
        );
        for mode in [ViewMode::Shapes, ViewMode::Filmstrip] {
            let full = render_row_entries(mode, &row, &subject);
            assert_eq!(full, &row[..], "{mode:?} renders the full row");
        }
    }

    #[test]
    fn single_mode_schlegel_cell_bound_from_subject() {
        let subject = entry(RaymarchShape::Polytope(Polytope4::Cell600));
        let row = [
            entry(RaymarchShape::Polytope(Polytope4::Pentatope)), // 5 cells
            entry(RaymarchShape::Polytope(Polytope4::Cell24)),
        ];
        let subject_poly = render_row_entries(ViewMode::Single, &row, &subject)
            .iter()
            .find_map(|e| e.shape.polytope4())
            .expect("the single subject is a polychoron");
        assert_eq!(subject_poly, Polytope4::Cell600);
        assert_eq!(
            subject_poly.cell_count(),
            Polytope4::Cell600.cell_count(),
            "the cell-index bound is the subject's cell count (600), not the row's leading 5",
        );
        let row_poly = render_row_entries(ViewMode::Shapes, &row, &subject)
            .iter()
            .find_map(|e| e.shape.polytope4())
            .expect("the row has a polychoron");
        assert_eq!(row_poly, Polytope4::Pentatope);
        assert_ne!(
            subject_poly.cell_count(),
            row_poly.cell_count(),
            "test setup: subject and row-leader must differ in cell count",
        );
    }

    #[test]
    fn section_layer_projection_honest_ignores_projected_follows() {
        let actives = [
            Projection::Identity,
            Projection::Perspective4D {
                focal_distance: 2.0,
            },
            Projection::Stereographic { pole: Vec4::W },
            Projection::schlegel(Vec4::W, 0.5, 0.75),
        ];
        for active in actives {
            assert_eq!(
                section_layer_projection(true, active),
                Projection::Identity,
                "honest cross-section must stay drop-w under active {active:?}"
            );
            assert_eq!(
                section_layer_projection(false, active),
                active,
                "projected cap must follow the active projection {active:?}"
            );
        }
    }

    #[test]
    fn section_layer_fill_visible_at_positive_alpha_only() {
        assert!(!SectionLayer {
            perimeter: true,
            surface_alpha: 0.0
        }
        .fill_visible());
        assert!(SectionLayer {
            perimeter: false,
            surface_alpha: 0.01
        }
        .fill_visible());
        assert!(SectionLayer {
            perimeter: false,
            surface_alpha: 1.0
        }
        .fill_visible());
    }

    fn tumbling(slots: usize) -> PlaygroundPhysics {
        let mut physics = PlaygroundPhysics::new(slots, BODY_SIZE);
        let layout = Vec4::from_array(body_position(1, slots));
        physics.world.bodies[1].apply_impulse_at_point(
            &EuclideanR4,
            Vec4::new(0.0, 0.0, 0.0, 1.2),
            layout + Vec4::X * 0.5,
        );
        physics.step(24);
        physics
    }

    #[test]
    fn two_slots_upload_two_different_orientations_in_one_frame() {
        const SLOTS: usize = 3;
        let physics = PlaygroundPhysics::new(SLOTS, BODY_SIZE);
        let shape = entry(RaymarchShape::Polytope(Polytope4::Cell24));
        let mut spins = SlotSpins::new(SLOTS);
        spins.select_picked(Some(0));
        spins.selected_spin_mut().active = [true, false, false, false, false, false];
        spins.select_picked(Some(1));
        spins.selected_spin_mut().active = [false, false, false, false, false, true];
        spins.recompose_active(1.7, &[]);

        let uniform = |slot: usize| {
            sdf_body_uniform(
                &physics,
                &shape,
                slot,
                SLOTS,
                spins.rotor(slot),
                BODY_SIZE,
                SurfaceMode::Sdf,
            )
        };
        let (first, second, control) = (uniform(0), uniform(1), uniform(2));
        assert_ne!(
            first.rotor, second.rotor,
            "both bodies uploaded the same orientation"
        );
        assert_ne!(first.rotor, control.rotor);
        assert_ne!(second.rotor, control.rotor);
        assert_eq!(first.radius_or_shape, second.radius_or_shape);
        assert_eq!(first.position, body_position(0, SLOTS));
        assert_eq!(second.position, body_position(1, SLOTS));
        for (slot, uniform) in [first, second, control].into_iter().enumerate() {
            let norm_squared: f32 = uniform.rotor.iter().map(|c| c * c).sum();
            assert!(
                (norm_squared - 1.0).abs() < 1e-5,
                "slot {slot} uploaded a rotor off the unit sphere: |R|² = {norm_squared}"
            );
        }
    }

    #[test]
    fn sdf_body_uniform_reads_the_physics_pose_not_the_authored_spin() {
        let slots = 3;
        let physics = tumbling(slots);
        let shape = entry(RaymarchShape::Polytope(Polytope4::Cell24));
        let spin = (Plane4::Xy.unit_bivector() * 0.7).exp().normalize();

        let uniform = sdf_body_uniform(
            &physics,
            &shape,
            1,
            slots,
            spin,
            BODY_SIZE,
            SurfaceMode::Sdf,
        );

        let body = &physics.world.bodies[1];
        assert_ne!(
            body.orientation.rotation,
            Rotor4::IDENTITY,
            "throw produced no rotation, so the rotor pin below is vacuous"
        );
        assert_eq!(uniform.position, body.position.to_array());
        assert_ne!(
            uniform.position,
            body_position(1, slots),
            "uniform centre still reads the static layout"
        );
        assert_eq!(
            uniform.rotor,
            <[f32; 8]>::from(composed_rotor(spin, body.orientation.rotation))
        );
        assert_ne!(
            uniform.rotor,
            <[f32; 8]>::from(spin),
            "uniform rotor still reads the authored spin alone"
        );
    }

    #[test]
    fn sdf_body_uniform_of_an_untouched_slot_is_the_authored_layout_and_spin() {
        let slots = 3;
        let physics = tumbling(slots);
        let shape = entry(RaymarchShape::Polytope(Polytope4::Cell24));
        let spin = (Plane4::Zw.unit_bivector() * -1.1).exp().normalize();

        let uniform = sdf_body_uniform(
            &physics,
            &shape,
            2,
            slots,
            spin,
            BODY_SIZE,
            SurfaceMode::Sdf,
        );
        assert_eq!(uniform.position, body_position(2, slots));
        assert_eq!(uniform.rotor, <[f32; 8]>::from(spin));
    }

    #[test]
    fn row_frame_anchor_reads_the_live_pose_not_the_authored_spin() {
        let slots = 3;
        let physics = tumbling(slots);
        let row = [entry(RaymarchShape::Polytope(Polytope4::Cell24)); 3];
        let spin = (Plane4::Xy.unit_bivector() * 0.7).exp().normalize();
        let spins = SlotSpins::uniform(row.len(), spin);
        let frame = super::RowFrame {
            physics: &physics,
            row: &row,
            spins: &spins,
            body_size: BODY_SIZE,
            projection: Projection::Identity,
            w_slice: 0.0,
            camera_distance: 4.0,
        };

        let canonical = Polytope4::Cell24.topology().vertices[0];
        let pose = physics.pose(1, slots, spin);
        assert_eq!(
            frame.anchor_r3(1, canonical),
            pose.body_local(canonical, BODY_SIZE).truncate() + pose.position_r3()
        );
        assert_ne!(
            frame.anchor_r3(1, canonical),
            (BODY_SIZE * spin.apply(canonical)).truncate()
                + Vec4::from_array(body_position(1, slots)).truncate(),
            "anchor still reads the authored spin over the static layout"
        );
    }

    #[test]
    fn sdf_body_uniform_keeps_the_polychora_opt_out() {
        let slots = 3;
        let physics = tumbling(slots);
        let inert = BodyUniform::default().kind;
        let live = |shape, mode| {
            sdf_body_uniform(
                &physics,
                &entry(shape),
                1,
                slots,
                Rotor4::IDENTITY,
                BODY_SIZE,
                mode,
            )
            .kind
                != inert
        };

        for mode in [SurfaceMode::Sdf, SurfaceMode::Raster, SurfaceMode::Off] {
            for heavy in [Polytope4::Cell120, Polytope4::Cell600] {
                assert!(
                    !live(RaymarchShape::Polytope(heavy), mode),
                    "{heavy:?} raymarched in {mode:?}"
                );
            }
            assert_eq!(
                live(RaymarchShape::Polytope(Polytope4::Cell24), mode),
                mode.uses_sdf_for_polychora(),
                "24-cell SDF dispatch disagrees with {mode:?}"
            );
            assert!(
                live(RaymarchShape::CliffordTorus, mode),
                "smooth surface opted out in {mode:?}"
            );
        }
    }

    #[test]
    fn body_upload_gate_fires_on_every_pose_change_and_nothing_else() {
        let spun = (Plane4::Xw.unit_bivector() * 0.4).exp().normalize();
        let still = SlotSpins::uniform(4, Rotor4::IDENTITY);
        let mut uploaded = Vec::new();
        still.record_rotors(&mut uploaded);
        assert!(!body_upload_needed(&still, &uploaded, false));
        assert!(body_upload_needed(&still, &uploaded, true));

        for slot in 0..4 {
            let mut one_turned = SlotSpins::uniform(4, Rotor4::IDENTITY);
            one_turned.select_picked(Some(slot));
            one_turned.selected_spin_mut().rotor = spun;
            assert!(
                body_upload_needed(&one_turned, &uploaded, false),
                "rotating slot {slot} alone left the gate closed"
            );
            let mut turned_upload = Vec::new();
            one_turned.record_rotors(&mut turned_upload);
            assert!(!body_upload_needed(&one_turned, &turned_upload, false));
            assert!(
                body_upload_needed(&still, &turned_upload, false),
                "unrotating slot {slot} left the gate closed"
            );
        }
    }

    #[test]
    fn set_if_changed_reports_a_change_only_when_the_value_moves() {
        let mut w_slice = 0.0_f32;
        assert!(!set_if_changed(&mut w_slice, 0.0));
        assert!(set_if_changed(&mut w_slice, 0.25));
        assert_eq!(w_slice, 0.25);
        assert!(!set_if_changed(&mut w_slice, 0.25));

        let mut camera_pos = [0.0_f32, 0.0, 5.0];
        assert!(!set_if_changed(&mut camera_pos, [0.0, 0.0, 5.0]));
        assert!(set_if_changed(&mut camera_pos, [0.0, 1e-7, 5.0]));
        assert_eq!(camera_pos, [0.0, 1e-7, 5.0]);
    }
}
