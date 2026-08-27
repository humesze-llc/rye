//! Every pass records into the runner's frame-wide encoder, which the runner
//! submits once (`loam_app::App::record`). Two consequences bind the code here:
//! nothing may call `queue.submit`, and no node may be uploaded twice in a
//! frame, because `Queue::write_buffer` lands before the whole command buffer
//! and the second upload would feed both passes.

use crate::*;

impl Demo {
    pub(crate) fn record(
        &mut self,
        rd: &RenderDevice,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) -> Result<()> {
        let cfg = &rd.surface_bundle.config;
        let viewport = Viewport::full([cfg.width, cfg.height]);
        if self.view_mode == ViewMode::Filmstrip {
            let entry = self.strip_subject;
            let strip_w_extent = self.effective_body_size();
            let (cols, rows, w_on_cols) = match (self.strip_w, self.strip_t) {
                (true, true) => {
                    if self.strip_swap_axes {
                        (self.strip_count_t, self.strip_count_w, false)
                    } else {
                        (self.strip_count_w, self.strip_count_t, true)
                    }
                }
                (true, false) => (self.strip_count_w, 1, true),
                (false, true) => (1, self.strip_count_t, false),
                (false, false) => (1, 1, true),
            };
            let col_vps = viewport.split_horizontal(cols as u32);
            let mut grid_cells: Vec<(Viewport, f32, BodyUniform)> = Vec::with_capacity(cols * rows);
            for (col_idx, col_vp) in col_vps.into_iter().enumerate() {
                let row_vps = col_vp.split_vertical(rows as u32);
                for (row_idx, cell_vp) in row_vps.into_iter().enumerate() {
                    let (w_idx, w_n, t_idx, t_n) = if w_on_cols {
                        (col_idx, cols, row_idx, rows)
                    } else {
                        (row_idx, rows, col_idx, cols)
                    };
                    let w_t = if w_n <= 1 {
                        0.5
                    } else {
                        w_idx as f32 / (w_n - 1) as f32
                    };
                    let w_offset = -strip_w_extent + w_t * (2.0 * strip_w_extent);
                    let cell_w_slice = self.w_slice + w_offset;
                    let t_offset = if !self.strip_t || t_n <= 1 {
                        0.0
                    } else {
                        let t_norm = t_idx as f32 / (t_n - 1) as f32;
                        t_norm * self.strip_t_extent
                    };
                    let cell_rotor = if t_offset == 0.0 {
                        self.selected_rotor()
                    } else {
                        self.rotor_at_time(self.rot_time + t_offset)
                    };
                    let body = BodyUniform::polytope_with_rotor(
                        [0.0, BODY_Y, 0.0, 0.0],
                        entry.shape.shape_id(),
                        self.effective_body_size(),
                        cell_rotor,
                        entry.body_color,
                    );
                    grid_cells.push((cell_vp, cell_w_slice, body));
                }
            }
            // The one path that still submits mid-frame: `execute_strip` owns
            // its encoder and submit rather than recording into this one. That
            // submit lands before the runner's, so the strip still composites
            // under the UI.
            let result = self
                .node
                .execute_strip(&rd.device, &rd.queue, view, &grid_cells);
            self.rebuild_bodies();
            result
        } else {
            {
                let _scope = loam_time::frame_trace::scope("pp-sdf");
                {
                    let mut changed = false;
                    let u = self.node.uniforms_mut();
                    changed |= set_if_changed(&mut u.resolution, viewport.resolution_f32());
                    changed |= set_if_changed(
                        &mut u.viewport_origin,
                        [viewport.x as f32, viewport.y as f32],
                    );
                    self.sdf_upload_pending |= changed;
                }
                // The only flush on this path: `update` leaves the dirty flag
                // rather than uploading a buffer this would overwrite.
                if self.sdf_upload_pending {
                    self.node.flush_uniforms(&rd.queue);
                    self.sdf_upload_pending = false;
                }
                self.node.record_in_viewport(encoder, view, viewport);
            }
            // Shared depth for the section pass + the wireframe's depth-test.
            // Order: SDF (color only) -> section_faces (writes depth in Raster) ->
            // wireframe (tests, no write). In SDF mode no pass writes depth, so the
            // cleared `1.0` lets every wireframe fragment pass.
            if shared_depth_is_read(self.surface_mode, self.wireframe_enabled) {
                self.ensure_and_clear_shared_depth(rd, encoder);
            }
            if matches!(self.surface_mode, SurfaceMode::Raster) {
                let _scope = loam_time::frame_trace::scope("pp-section-faces");
                self.record_section_faces(rd, encoder, view);
            }
            if self.wireframe_enabled {
                let _scope = loam_time::frame_trace::scope("pp-wireframe");
                self.record_wireframe_overlay(rd, encoder, view);
            }
            if self.points_enabled {
                let _scope = loam_time::frame_trace::scope("pp-points");
                self.record_points(rd, encoder, view);
            }
            if self.physics_overlay.any_layer() {
                let _scope = loam_time::frame_trace::scope("pp-physics-overlay");
                self.record_physics_overlay(rd, encoder, view);
            }
            {
                let _scope = loam_time::frame_trace::scope("pp-gimbal");
                self.record_gimbal(rd, encoder, view);
            }
            Ok(())
        }
    }

    fn ensure_and_clear_shared_depth(
        &mut self,
        rd: &RenderDevice,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let cfg = &rd.surface_bundle.config;
        DepthBuffer::ensure(
            &mut self.section_faces_depth,
            &rd.device,
            SECTION_FACES_DEPTH_FORMAT,
            (cfg.width, cfg.height),
            rd.sample_count(),
        );
        let depth = self
            .section_faces_depth
            .as_ref()
            .expect("ensure() guarantees Some");
        let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shared depth clear pass"),
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
    }

    fn record_points(
        &mut self,
        rd: &RenderDevice,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        let cfg = &rd.surface_bundle.config;
        let style = PointsStyle {
            color_mode: self.wireframe_color_mode,
            show_vertices: self.points_show_vertices,
            show_cell_centers: self.points_show_cell_centers,
            size_px: self.points_size_px,
        };
        // Mesh + scratch taken out of `self` for the build: the row frame
        // borrows the physics world for as long as the buffers it fills are
        // borrowed. Put back so their capacity persists across frames.
        let mut mesh = std::mem::take(&mut self.points_mesh_scratch);
        let mut centers_cache = std::mem::take(&mut self.cell_centers_cache);
        let mut local_vertices = std::mem::take(&mut self.overlay_local_vertices_scratch);
        let mut center_locals = std::mem::take(&mut self.overlay_center_locals_scratch);
        let mut cell_strengths = std::mem::take(&mut self.overlay_cell_strengths_scratch);
        build_points_mesh(
            &self.row_frame(),
            &style,
            &mut centers_cache,
            &mut local_vertices,
            &mut center_locals,
            &mut cell_strengths,
            &mut mesh,
        );
        self.cell_centers_cache = centers_cache;
        self.overlay_local_vertices_scratch = local_vertices;
        self.overlay_center_locals_scratch = center_locals;
        self.overlay_cell_strengths_scratch = cell_strengths;

        let view_dir = self.camera.view();
        let aspect = cfg.width as f32 / cfg.height as f32;
        let view_mat = Mat4::look_to_rh(view_dir.position, view_dir.forward, view_dir.up);
        let proj_mat = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.1, 100.0);
        let view_proj = proj_mat * view_mat;
        let vp_size = Vec2::new(cfg.width as f32, cfg.height as f32);
        self.points_node.set_camera(&rd.queue, view_proj, vp_size);
        self.points_node.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &mesh,
            &loam_math::Projection::Identity,
        );
        self.points_mesh_scratch = mesh;
        // No depth attachment: see `PointRasterNode::new` (drop-w + ReadOnly
        // LessEqual occluded non-w=0 vertices behind their own caps).
        self.points_node.record(encoder, view, None, None);
    }

    fn record_physics_overlay(
        &mut self,
        rd: &RenderDevice,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        let cfg = &rd.surface_bundle.config;
        let overlay = self.physics_overlay;
        let mut mesh = std::mem::take(&mut self.physics_overlay_mesh_scratch);
        build_physics_overlay_mesh(&self.row_frame(), &overlay, &mut mesh);

        let view_dir = self.camera.view();
        let aspect = cfg.width as f32 / cfg.height as f32;
        let view_mat = Mat4::look_to_rh(view_dir.position, view_dir.forward, view_dir.up);
        let proj_mat = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.1, 100.0);
        let view_proj = proj_mat * view_mat;
        let vp_size = Vec2::new(cfg.width as f32, cfg.height as f32);
        self.physics_overlay_node
            .set_camera(&rd.queue, view_proj, vp_size);
        self.physics_overlay_node.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &mesh,
            &loam_math::Projection::Identity,
            1,
        );
        self.physics_overlay_mesh_scratch = mesh;
        self.physics_overlay_node.record(encoder, view, None, None);
    }

    /// The honest layer draws first so
    /// the opt-in projected cap composites over it when both are on. Defaults draw
    /// only the honest layer, so selecting a distorting projection never silently
    /// reshapes the slice the user reads as "the cross-section."
    ///
    /// When both layers land on the same node ([`section_layers_share_a_node`])
    /// they are merged into one pass, cap indices after cross indices, which is
    /// pixel-identical to two passes on one pipeline with one set of depth ops.
    fn record_section_faces(
        &mut self,
        rd: &RenderDevice,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        let cfg = &rd.surface_bundle.config;
        let cross = self.cross_section;
        let cap = self.projected_cap;
        if !cross.fill_visible() && !cap.fill_visible() {
            return;
        }

        let mut local_vertices = std::mem::take(&mut self.section_world_vertices_scratch);
        let mut proj_scratch = std::mem::take(&mut self.section_clip_projected_scratch);
        let mut cross_mesh = std::mem::take(&mut self.section_faces_mesh_scratch);
        let mut cap_mesh = std::mem::take(&mut self.section_faces_projected_scratch);
        let mut section_scratch = std::mem::take(&mut self.section_cap_scratch);
        build_section_layer_meshes(
            &self.row_frame(),
            cross,
            cap,
            SectionBuffers {
                local_vertices: &mut local_vertices,
                proj_scratch: &mut proj_scratch,
                cross_mesh: &mut cross_mesh,
                cap_mesh: &mut cap_mesh,
                section_scratch: &mut section_scratch,
            },
        );
        let merged = section_layers_share_a_node(cross, cap);
        if merged {
            append_triangle_mesh(&mut cross_mesh, &cap_mesh);
        }
        self.section_cap_scratch = section_scratch;
        self.section_world_vertices_scratch = local_vertices;
        self.section_clip_projected_scratch = proj_scratch;
        self.section_faces_mesh_scratch = cross_mesh;
        self.section_faces_projected_scratch = cap_mesh;

        let view_dir = self.camera.view();
        let aspect = cfg.width as f32 / cfg.height as f32;
        let view_mat = Mat4::look_to_rh(view_dir.position, view_dir.forward, view_dir.up);
        let proj_mat = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.1, 100.0);
        let view_proj = proj_mat * view_mat;

        if cross.fill_visible() {
            self.record_section_layer(rd, encoder, view, view_proj, cross.surface_alpha, true);
        }
        if cap.fill_visible() && !merged {
            self.record_section_layer(rd, encoder, view, view_proj, cap.surface_alpha, false);
        }
    }

    fn record_section_layer(
        &mut self,
        rd: &RenderDevice,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        view_proj: Mat4,
        alpha: f32,
        is_cross_section: bool,
    ) {
        let depth_view = &self
            .section_faces_depth
            .as_ref()
            .expect("shared depth buffer must be ensured before section_faces")
            .view;
        let mesh = if is_cross_section {
            &self.section_faces_mesh_scratch
        } else {
            &self.section_faces_projected_scratch
        };
        let node = if section_alpha_is_opaque(alpha) {
            &mut self.section_faces
        } else {
            &mut self.section_faces_translucent
        };
        node.set_camera(&rd.queue, view_proj);
        node.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            mesh,
            &loam_math::Projection::Identity,
        );
        node.record(encoder, view, Some(depth_view), None);
    }

    pub(crate) fn camera_distance_to_focus(&self) -> f32 {
        (self.camera.position - self.orbit.target).length()
    }

    fn record_wireframe_overlay(
        &mut self,
        rd: &RenderDevice,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        let cfg = &rd.surface_bundle.config;
        let style = WireframeStyle {
            color_mode: self.wireframe_color_mode,
            alpha: self.wireframe_alpha,
            width_px: self.wireframe_width_px,
            nearest_active: self.wireframe_nearest_active,
            space_blend: state::default_edge_blend(self.wireframe_projection),
            hyperslice: self
                .hyperslice_cull_active()
                .then_some(self.wireframe_hyperslice_thickness),
        };
        let cross = self.cross_section;
        let cap = self.projected_cap;
        let mut palette_cache = std::mem::take(&mut self.unique_edge_palette_cache);
        let mut slerp_scratch = std::mem::take(&mut self.slerp_scratch);
        let mut local_vertices = std::mem::take(&mut self.overlay_local_vertices_scratch);
        let mut cell_strengths = std::mem::take(&mut self.overlay_cell_strengths_scratch);
        let mut section_scratch = std::mem::take(&mut self.section_cap_scratch);
        let mut body_perimeter = std::mem::take(&mut self.body_perimeter_scratch);
        let mut section_edges = std::mem::take(&mut self.wireframe_section_edges_scratch);
        let mut parent_lines = std::mem::take(&mut self.wireframe_parent_lines_scratch);
        build_wireframe_meshes(
            &self.row_frame(),
            &style,
            cross,
            cap,
            &mut palette_cache,
            &mut slerp_scratch,
            &mut local_vertices,
            &mut cell_strengths,
            &mut section_scratch,
            &mut body_perimeter,
            &mut section_edges,
            &mut parent_lines,
        );
        self.unique_edge_palette_cache = palette_cache;
        self.slerp_scratch = slerp_scratch;
        self.overlay_local_vertices_scratch = local_vertices;
        self.overlay_cell_strengths_scratch = cell_strengths;
        self.section_cap_scratch = section_scratch;
        self.body_perimeter_scratch = body_perimeter;

        self.section_edges.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &section_edges,
            &loam_math::Projection::Identity,
            1,
        );
        self.parent_wireframe.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &parent_lines,
            &loam_math::Projection::Identity,
            1,
        );
        self.wireframe_section_edges_scratch = section_edges;
        self.wireframe_parent_lines_scratch = parent_lines;

        let view_dir = self.camera.view();
        let aspect = cfg.width as f32 / cfg.height as f32;
        let view_mat = Mat4::look_to_rh(view_dir.position, view_dir.forward, view_dir.up);
        let proj_mat = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.1, 100.0);
        let view_proj = proj_mat * view_mat;
        let vp_size = Vec2::new(cfg.width as f32, cfg.height as f32);
        self.section_edges.set_camera(&rd.queue, view_proj, vp_size);
        self.parent_wireframe
            .set_camera(&rd.queue, view_proj, vp_size);

        let depth_view = self
            .section_faces_depth
            .as_ref()
            .map(|b| &b.view)
            .expect("shared depth buffer must be ensured before wireframe overlay");
        self.section_edges
            .record(encoder, view, Some(depth_view), None);
        self.parent_wireframe
            .record(encoder, view, Some(depth_view), None);
    }
}

pub(crate) fn section_alpha_is_opaque(alpha: f32) -> bool {
    alpha >= 1.0
}

/// Whether both section layers are visible AND draw through the same
/// [`loam_render::TriangleRasterNode`], which forces them into one mesh and one
/// pass: a node owns one vertex buffer, and every `Queue::write_buffer` of a
/// frame is applied before the frame's single command buffer runs, so uploading
/// twice would leave BOTH passes reading the second mesh (the honest
/// cross-section would silently render as the projected cap).
pub(crate) fn section_layers_share_a_node(
    cross: state::SectionLayer,
    cap: state::SectionLayer,
) -> bool {
    cross.fill_visible()
        && cap.fill_visible()
        && section_alpha_is_opaque(cross.surface_alpha)
            == section_alpha_is_opaque(cap.surface_alpha)
}

/// Order is preserved, so a merged pass rasterizes `dst`'s
/// triangles before `src`'s exactly as two passes in that order would.
pub(crate) fn append_triangle_mesh(
    dst: &mut loam_shape::TriangleMesh<3>,
    src: &loam_shape::TriangleMesh<3>,
) {
    let base = dst.vertices.len() as u32;
    dst.vertices.extend_from_slice(&src.vertices);
    dst.colors.extend_from_slice(&src.colors);
    dst.indices.extend(
        src.indices
            .iter()
            .map(|&[i, j, k]| [i + base, j + base, k + base]),
    );
}

pub(crate) fn shared_depth_is_read(surface_mode: SurfaceMode, wireframe_enabled: bool) -> bool {
    matches!(surface_mode, SurfaceMode::Raster) || wireframe_enabled
}

#[derive(Copy, Clone)]
pub(crate) struct PointsStyle {
    pub(crate) color_mode: WireframeColorMode,
    pub(crate) show_vertices: bool,
    pub(crate) show_cell_centers: bool,
    pub(crate) size_px: f32,
}

pub(crate) fn build_points_mesh(
    frame: &RowFrame<'_>,
    style: &PointsStyle,
    centers_cache: &mut std::collections::HashMap<loam_shape::polytope::Polytope4, Vec<Vec4>>,
    local_vertices: &mut Vec<Vec4>,
    center_locals: &mut Vec<Vec4>,
    cell_strengths: &mut Vec<f32>,
    mesh: &mut loam_shape::PointMesh<3>,
) {
    const ACTIVE_GREEN: [f32; 4] = [0.40, 1.00, 0.55, 1.0];
    const INACTIVE_GRAY: [f32; 4] = [0.55, 0.55, 0.58, 0.85];

    mesh.positions.clear();
    mesh.colors.clear();
    mesh.sizes.clear();

    for (slot, entry) in frame.row.iter().enumerate() {
        let Some(polytope) = entry.shape.polytope4() else {
            continue;
        };
        let points_clip_radius = stereographic_clip_radius(
            &frame.projection,
            stereographic_view_radius(polytope, frame.camera_distance),
        );
        let topo = polytope.topology();

        let body_pos_r3 = frame.body_local(slot, topo.vertices, frame.body_size, local_vertices);
        let w_extent_local: f32 = if matches!(style.color_mode, WireframeColorMode::WDepth) {
            let canonical_max_w = topo
                .vertices
                .iter()
                .map(|v| v.w.abs())
                .fold(0.0_f32, f32::max)
                .max(1e-6);
            canonical_max_w * frame.body_size
        } else {
            1.0
        };
        if matches!(style.color_mode, WireframeColorMode::Active) {
            compute_cell_strengths(topo.cells, local_vertices, frame.w_slice, cell_strengths);
        } else {
            cell_strengths.clear();
        }
        let vertex_is_active = |vi: usize| -> bool {
            topo.cells
                .iter()
                .zip(cell_strengths.iter())
                .any(|(cell, &s)| s > 0.0 && cell.contains(&(vi as u32)))
        };

        if style.show_vertices {
            for (vi, v) in topo.vertices.iter().enumerate() {
                let v_local = local_vertices[vi];
                let v3_local =
                    <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(
                        v_local,
                        &frame.projection,
                    );
                if !sample_in_radius(v3_local, points_clip_radius) {
                    continue;
                }
                let v_world = v3_local + body_pos_r3;
                let color = match style.color_mode {
                    WireframeColorMode::VertexGradient | WireframeColorMode::UniqueEdge => {
                        vertex_color_by_position(*v)
                    }
                    WireframeColorMode::WDepth => w_depth_color(v_local.w, w_extent_local),
                    WireframeColorMode::Active => {
                        if vertex_is_active(vi) {
                            ACTIVE_GREEN
                        } else {
                            INACTIVE_GRAY
                        }
                    }
                };
                mesh.positions.push(v_world.to_array());
                mesh.colors.push(color);
                mesh.sizes.push(style.size_px);
            }
        }
        if style.show_cell_centers {
            // Pull centroids inward by `CELL_CENTER_INSET` so they read as
            // interior markers. `cell_centers()` returns centroids at the
            // inradius, which is the DUAL's vertex set; at full inradius the
            // sprites look like the wrong polytope, so inset them inside the cap.
            const CELL_CENTER_INSET: f32 = 0.5;
            let centers: &[Vec4] = centers_cache
                .entry(polytope)
                .or_insert_with(|| polytope.cell_centers());
            frame.body_local(
                slot,
                centers,
                frame.body_size * CELL_CENTER_INSET,
                center_locals,
            );
            for (ci, c) in centers.iter().enumerate() {
                let c_local = center_locals[ci];
                let c3_local =
                    <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(
                        c_local,
                        &frame.projection,
                    );
                if !sample_in_radius(c3_local, points_clip_radius) {
                    continue;
                }
                let c_world = c3_local + body_pos_r3;
                let color = match style.color_mode {
                    WireframeColorMode::VertexGradient | WireframeColorMode::UniqueEdge => {
                        vertex_color_by_position(*c)
                    }
                    WireframeColorMode::WDepth => w_depth_color(c_local.w, w_extent_local),
                    WireframeColorMode::Active => {
                        let s = cell_strengths.get(ci).copied().unwrap_or(0.0);
                        if s > 0.0 {
                            ACTIVE_GREEN
                        } else {
                            INACTIVE_GRAY
                        }
                    }
                };
                mesh.positions.push(c_world.to_array());
                mesh.colors.push(color);
                mesh.sizes.push(style.size_px * 0.5);
            }
        }
    }
}

pub(crate) struct SectionBuffers<'a> {
    pub(crate) local_vertices: &'a mut Vec<Vec4>,
    pub(crate) proj_scratch: &'a mut Vec<Vec3>,
    pub(crate) cross_mesh: &'a mut loam_shape::TriangleMesh<3>,
    pub(crate) cap_mesh: &'a mut loam_shape::TriangleMesh<3>,
    pub(crate) section_scratch: &'a mut SectionScratch,
}

pub(crate) fn build_section_layer_meshes(
    frame: &RowFrame<'_>,
    cross: state::SectionLayer,
    cap: state::SectionLayer,
    buffers: SectionBuffers<'_>,
) {
    let SectionBuffers {
        local_vertices,
        proj_scratch,
        cross_mesh,
        cap_mesh,
        section_scratch,
    } = buffers;
    let w_slice = frame.w_slice;
    let cross_projection = state::section_layer_projection(true, frame.projection);
    let cap_projection = state::section_layer_projection(false, frame.projection);
    let cross_scale = perspective_scale_at_w(w_slice, &cross_projection);
    let cap_scale = perspective_scale_at_w(w_slice, &cap_projection);

    cross_mesh.vertices.clear();
    cross_mesh.colors.clear();
    cross_mesh.indices.clear();
    cap_mesh.vertices.clear();
    cap_mesh.colors.clear();
    cap_mesh.indices.clear();

    for (slot, entry) in frame.row.iter().enumerate() {
        let Some(polytope) = entry.shape.polytope4() else {
            continue;
        };
        let view_radius = stereographic_view_radius(polytope, frame.camera_distance);
        let cross_clip = stereographic_clip_radius(&cross_projection, view_radius);
        let cap_clip = stereographic_clip_radius(&cap_projection, view_radius);
        let topo = polytope.topology();

        let body_pos_r3 = frame.body_local(slot, topo.vertices, frame.body_size, local_vertices);
        let cap_vertices: &[Vec4] = local_vertices;

        let [r, g, b] = entry.body_color;

        let append_layer = |mesh: &mut loam_shape::TriangleMesh<3>,
                            proj_scratch: &mut Vec<Vec3>,
                            scratch: &mut SectionScratch,
                            alpha: f32,
                            projection: &loam_math::Projection<4>,
                            scale: Option<f32>,
                            clip_radius: Option<f32>| {
            let start_v = mesh.vertices.len();
            let start_i = mesh.indices.len();
            polytope_section_faces_append(
                topo.edges,
                topo.cells,
                cap_vertices,
                WPlane::new(w_slice),
                [r, g, b, alpha],
                scratch,
                mesh,
            );
            proj_scratch.clear();
            for v in &mut mesh.vertices[start_v..] {
                let (projected, world) =
                    cap_vertex_projected_and_world(*v, w_slice, scale, projection, body_pos_r3);
                *v = world;
                proj_scratch.push(projected);
            }
            retain_in_radius_triangles(
                &mut mesh.indices,
                start_i,
                start_v,
                proj_scratch,
                clip_radius,
            );
        };

        if cross.fill_visible() {
            append_layer(
                cross_mesh,
                proj_scratch,
                section_scratch,
                cross.surface_alpha,
                &cross_projection,
                cross_scale,
                cross_clip,
            );
        }
        if cap.fill_visible() {
            append_layer(
                cap_mesh,
                proj_scratch,
                section_scratch,
                cap.surface_alpha,
                &cap_projection,
                cap_scale,
                cap_clip,
            );
        }
    }
}

#[derive(Copy, Clone)]
pub(crate) struct WireframeStyle {
    pub(crate) color_mode: WireframeColorMode,
    pub(crate) alpha: f32,
    pub(crate) width_px: f32,
    pub(crate) nearest_active: bool,
    pub(crate) space_blend: f32,
    /// `Some(thickness)` runs the Hyperslice cull: keep only edges whose
    /// body-local w-interval intersects the slab around the slice, thinning the
    /// graph to what the slice passes through. A demo-side per-edge filter,
    /// deliberately not a `Projection` variant (the projection has discarded w,
    /// so it cannot carry a keep/drop signal).
    pub(crate) hyperslice: Option<f32>,
}

/// Both meshes and every scratch buffer are the caller's, cleared on entry and
/// refilled. Under Stereographic the parent mesh reaches ~92k segments for a
/// full row of 600-cells, so owning them here would be megabytes of
/// grow-and-copy per frame.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_wireframe_meshes(
    frame: &RowFrame<'_>,
    style: &WireframeStyle,
    cross: state::SectionLayer,
    cap: state::SectionLayer,
    palette_cache: &mut std::collections::HashMap<loam_shape::polytope::Polytope4, Vec<[f32; 4]>>,
    slerp_scratch: &mut Vec<Vec4>,
    local_vertices: &mut Vec<Vec4>,
    cell_strengths: &mut Vec<f32>,
    section_scratch: &mut SectionScratch,
    body_perimeter: &mut LineMesh<3>,
    section_edges: &mut LineMesh<3>,
    parent_lines: &mut LineMesh<3>,
) {
    section_edges.segments.clear();
    section_edges.colors.clear();
    section_edges.widths.clear();
    parent_lines.segments.clear();
    parent_lines.colors.clear();
    parent_lines.widths.clear();
    const PARENT_ALPHA_DIM: f32 = 0.10;
    const PARENT_ALPHA_BRIGHT: f32 = 0.85;
    const ACTIVE_GREEN: [f32; 4] = [0.40, 1.00, 0.55, 1.0];
    const INACTIVE_GRAY: [f32; 4] = [0.55, 0.55, 0.58, 1.0];
    let w_slice = frame.w_slice;
    let cross_section_projection = state::section_layer_projection(true, frame.projection);

    for (slot, entry) in frame.row.iter().enumerate() {
        let Some(polytope) = entry.shape.polytope4() else {
            continue;
        };
        let view_radius = stereographic_view_radius(polytope, frame.camera_distance);
        let topo = polytope.topology();
        // The body's `w` rides inside the body-local frame (see
        // `BodyPose::body_local`); projecting there and translating in R³ AFTER
        // keeps its apparent x-position stable when Perspective4D scales
        // (x, y, z) by `focal / (focal - w)`.
        let body_pos_r3 = frame.body_local(slot, topo.vertices, frame.body_size, local_vertices);
        // Centre of the circumsphere the arc path bows onto: where the body's
        // own origin lands in that frame, which is the `w` offset once physics
        // has pushed the body off the slice. Taken from `body_local` itself so
        // the arc cannot drift from the vertices it joins.
        let arc_center = frame.pose(slot).body_local(Vec4::ZERO, frame.body_size);

        if cross.perimeter || cap.perimeter {
            body_perimeter.segments.clear();
            body_perimeter.colors.clear();
            body_perimeter.widths.clear();
            polytope_section_perimeter_append(
                topo.edges,
                topo.cells,
                local_vertices,
                WPlane::new(w_slice),
                section_scratch,
                body_perimeter,
            );
            let mut push_perimeter = |projection: &loam_math::Projection<4>| {
                let section_scale = perspective_scale_at_w(w_slice, projection);
                let clip_radius = stereographic_clip_radius(projection, view_radius);
                for ((a, b), (color, width)) in body_perimeter.segments.iter().zip(
                    body_perimeter
                        .colors
                        .iter()
                        .zip(body_perimeter.widths.iter()),
                ) {
                    let (pa, wa) = cap_vertex_projected_and_world(
                        *a,
                        w_slice,
                        section_scale,
                        projection,
                        body_pos_r3,
                    );
                    let (pb, wb) = cap_vertex_projected_and_world(
                        *b,
                        w_slice,
                        section_scale,
                        projection,
                        body_pos_r3,
                    );
                    if !sample_in_radius(pa, clip_radius) || !sample_in_radius(pb, clip_radius) {
                        continue;
                    }
                    section_edges.segments.push((wa, wb));
                    section_edges.colors.push(*color);
                    section_edges.widths.push(*width);
                }
            };
            if cross.perimeter {
                push_perimeter(&cross_section_projection);
            }
            if cap.perimeter {
                push_perimeter(&frame.projection);
            }
        }

        compute_cell_strengths(topo.cells, local_vertices, w_slice, cell_strengths);

        let edge_strength = |i: u32, j: u32| -> f32 {
            let mut best = 0.0_f32;
            for (cell, strength) in topo.cells.iter().zip(cell_strengths.iter()) {
                if cell.contains(&i) && cell.contains(&j) && *strength > best {
                    best = *strength;
                }
            }
            best
        };

        let edge_is_active = |i: u32, j: u32| -> bool {
            topo.cells
                .iter()
                .zip(cell_strengths.iter())
                .any(|(cell, &s)| s > 0.0 && cell.contains(&i) && cell.contains(&j))
        };

        // Hyperslice cull, CELL-level to match `edge_is_active`: an edge survives
        // iff some cell containing both endpoints has its w-range overlapping the
        // slab. The edge-level test would cull a far-side edge of an active cell
        // that the cell-level coloring paints green.
        let edge_in_slab_cell = |i: u32, j: u32, thickness: f32| -> bool {
            topo.cells.iter().any(|cell| {
                if !(cell.contains(&i) && cell.contains(&j)) {
                    return false;
                }
                let (w_min, w_max) = cell_w_range(cell, local_vertices);
                slab_overlaps(w_min, w_max, w_slice, thickness)
            })
        };

        // UniqueEdge palette is topology-only; memoize by `Polytope4` so the
        // 600-cell's ~520k pair-checks run once per launch, not per frame.
        let edge_palette: &[[f32; 4]] =
            if matches!(style.color_mode, WireframeColorMode::UniqueEdge) {
                palette_cache
                    .entry(polytope)
                    .or_insert_with(|| unique_edge_palette(topo.edges))
            } else {
                &[]
            };
        // WDepth normalizes against the CANONICAL max |w| (not the rotated
        // per-frame max), so the color stays temporally stable as the rotor
        // swings a vertex from -w to +w. Per-polytope because the band differs
        // (the tesseract's `[-0.5, +0.5]` does not fit the 16-cell, 5-cell).
        let w_extent_local: f32 = if matches!(style.color_mode, WireframeColorMode::WDepth) {
            let canonical_max_w = topo
                .vertices
                .iter()
                .map(|v| v.w.abs())
                .fold(0.0_f32, f32::max)
                .max(1e-6);
            canonical_max_w * frame.body_size
        } else {
            1.0
        };
        for (edge_idx, &[i, j]) in topo.edges.iter().enumerate() {
            let ia = i as usize;
            let ja = j as usize;
            let a = local_vertices[ia];
            let b = local_vertices[ja];
            if style
                .hyperslice
                .is_some_and(|thickness| !edge_in_slab_cell(i, j, thickness))
            {
                continue;
            }
            let (mut color_a, mut color_b) = match style.color_mode {
                WireframeColorMode::VertexGradient => (
                    vertex_color_by_position(topo.vertices[ia]),
                    vertex_color_by_position(topo.vertices[ja]),
                ),
                WireframeColorMode::UniqueEdge => {
                    let c = edge_palette[edge_idx];
                    (c, c)
                }
                WireframeColorMode::WDepth => (
                    w_depth_color(a.w, w_extent_local),
                    w_depth_color(b.w, w_extent_local),
                ),
                WireframeColorMode::Active => {
                    let c = if edge_is_active(i, j) {
                        ACTIVE_GREEN
                    } else {
                        INACTIVE_GRAY
                    };
                    (c, c)
                }
            };
            let alpha = if style.nearest_active {
                let s = edge_strength(i, j);
                PARENT_ALPHA_DIM + (PARENT_ALPHA_BRIGHT - PARENT_ALPHA_DIM) * s
            } else {
                style.alpha
            };
            color_a[3] = alpha;
            color_b[3] = alpha;
            push_blended_edge(
                parent_lines,
                a,
                b,
                arc_center,
                color_a,
                color_b,
                style.width_px,
                style.space_blend,
                &frame.projection,
                body_pos_r3,
                slerp_scratch,
                view_radius,
            );
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct PhysicsOverlay {
    pub(crate) contacts: bool,
    pub(crate) normals: bool,
    pub(crate) impulses: bool,
    pub(crate) islands: bool,
    pub(crate) impulse_scale: f32,
    pub(crate) width_px: f32,
}

/// Bar length per unit impulse. A head-on flick at
/// [`crate::physics::MAX_THROW_SPEED`] against an equal mass peaks at about
/// 5.2 units of accumulated impulse in one contact, which this puts at rather
/// more than a third of a body radius. Measured, not derived: the solver is
/// not elastic, so the peak is well under the 8.1 of momentum the throw
/// carries in. `a_full_speed_flick_draws_its_bar_at_a_third_of_a_body_radius`
/// holds this sentence to the code.
const DEFAULT_IMPULSE_SCALE: f32 = 0.05;

/// Contact-marker arm half-length, as a fraction of the body radius. Small
/// enough that a full four-slot manifold reads as four marks and not a blob.
const CONTACT_CROSS_FRACTION: f32 = 0.15;

/// Drawn normal length, as a fraction of the body radius. Taken off the body
/// rather than fixed so `surface scale` cannot bury the arrows inside the
/// geometry they point away from.
const NORMAL_LEN_FRACTION: f32 = 0.9;

/// Island-marker arm half-length, as a fraction of the body radius: the marker
/// spans the body it labels.
const ISLAND_CROSS_FRACTION: f32 = 1.0;

const CONTACT_COLOR: [f32; 4] = [1.00, 0.95, 0.35, 1.0];
/// Normals run dark at the contact and bright at the tip, so the sign of the
/// normal reads off one segment without an arrowhead the line raster has no
/// primitive for.
const NORMAL_TAIL_COLOR: [f32; 4] = [0.06, 0.24, 0.42, 1.0];
const NORMAL_TIP_COLOR: [f32; 4] = [0.40, 0.95, 1.00, 1.0];
const NORMAL_IMPULSE_COLOR: [f32; 4] = [1.00, 0.30, 0.22, 1.0];
const TANGENT_IMPULSE_COLOR: [f32; 4] = [0.70, 0.40, 1.00, 1.0];

/// Island colours, indexed by the island's ordinal in the ascending partition
/// and wrapped past the end. Six hues, because the partition is read by eye and
/// a longer palette stops being separable at overlay alpha; a scene with more
/// than six simultaneous islands repeats a colour, which misreads two islands
/// as one, so the count is the ceiling on what this layer can claim.
const ISLAND_PALETTE: [[f32; 4]; 6] = [
    [0.35, 0.85, 0.45, 1.0],
    [0.95, 0.55, 0.20, 1.0],
    [0.45, 0.60, 1.00, 1.0],
    [0.95, 0.40, 0.75, 1.0],
    [0.90, 0.90, 0.35, 1.0],
    [0.35, 0.90, 0.90, 1.0],
];

impl Default for PhysicsOverlay {
    fn default() -> Self {
        Self {
            contacts: false,
            normals: false,
            impulses: false,
            islands: false,
            impulse_scale: DEFAULT_IMPULSE_SCALE,
            width_px: 2.0,
        }
    }
}

impl PhysicsOverlay {
    pub(crate) fn any_layer(self) -> bool {
        self.contacts || self.normals || self.impulses || self.islands
    }
}

fn push_overlay_segment(
    mesh: &mut LineMesh<3>,
    from: Vec3,
    to: Vec3,
    from_color: [f32; 4],
    to_color: [f32; 4],
    width: f32,
) {
    mesh.segments.push((from.to_array(), to.to_array()));
    mesh.colors.push((from_color, to_color));
    mesh.widths.push(width);
}

fn push_axis_cross(mesh: &mut LineMesh<3>, centre: Vec3, half: f32, color: [f32; 4], width: f32) {
    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
        push_overlay_segment(
            mesh,
            centre - axis * half,
            centre + axis * half,
            color,
            color,
            width,
        );
    }
}

pub(crate) fn build_physics_overlay_mesh(
    frame: &RowFrame<'_>,
    overlay: &PhysicsOverlay,
    mesh: &mut LineMesh<3>,
) {
    mesh.segments.clear();
    mesh.colors.clear();
    mesh.widths.clear();
    if !overlay.any_layer() {
        return;
    }

    let world = &frame.physics.world;
    let radius = frame.body_size;
    let width = overlay.width_px;

    if overlay.contacts || overlay.normals || overlay.impulses {
        for manifold in world.manifolds.values() {
            for cp in &manifold.points {
                let point = cp.world_point.truncate();
                let normal = cp.normal.truncate();
                if overlay.contacts {
                    push_axis_cross(
                        mesh,
                        point,
                        CONTACT_CROSS_FRACTION * radius,
                        CONTACT_COLOR,
                        width,
                    );
                }
                if overlay.normals {
                    push_overlay_segment(
                        mesh,
                        point,
                        point + normal * (NORMAL_LEN_FRACTION * radius),
                        NORMAL_TAIL_COLOR,
                        NORMAL_TIP_COLOR,
                        width,
                    );
                }
                if overlay.impulses {
                    // Drawn along −normal, which is the direction the
                    // accumulated impulse pushes A, and which keeps the bar off
                    // the normal ray above so a short bar under a long arrow
                    // still reads.
                    push_overlay_segment(
                        mesh,
                        point,
                        point - normal * (cp.normal_impulse * overlay.impulse_scale),
                        NORMAL_IMPULSE_COLOR,
                        NORMAL_IMPULSE_COLOR,
                        width,
                    );
                    // `tangent_dir` is B's slide relative to A, so −tangent_dir
                    // is the direction friction resists it in. That is the
                    // impulse on B, not on A: the two bars name opposite bodies
                    // and do not compose into a net impulse on either.
                    push_overlay_segment(
                        mesh,
                        point,
                        point
                            - cp.tangent_dir.truncate()
                                * (cp.tangent_impulse * overlay.impulse_scale),
                        TANGENT_IMPULSE_COLOR,
                        TANGENT_IMPULSE_COLOR,
                        width,
                    );
                }
            }
        }
    }

    if overlay.islands {
        // `islands()` allocates its partition per call. It is the only form
        // that yields the components ascending, which is what makes an island's
        // colour a function of the partition rather than of whichever body the
        // union-find left as a root; the layer is off by default, so the cost
        // is paid only by a frame that asked for it.
        for (ordinal, island) in world.islands().iter().enumerate() {
            let color = ISLAND_PALETTE[ordinal % ISLAND_PALETTE.len()];
            for &id in &island.bodies {
                push_axis_cross(
                    mesh,
                    world.bodies[id].position.truncate(),
                    ISLAND_CROSS_FRACTION * radius,
                    color,
                    width,
                );
            }
            for &(a, b) in &island.constraints {
                push_overlay_segment(
                    mesh,
                    world.bodies[a].position.truncate(),
                    world.bodies[b].position.truncate(),
                    color,
                    color,
                    width,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::ShapeEntry;
    use crate::physics::{PlaygroundPhysics, MAX_THROW_SPEED};
    use crate::spins::SlotSpins;
    use crate::state::{body_position, RowFrame, SectionLayer};
    use loam_math::{EuclideanR4, Plane4, Projection};
    use loam_render::raymarch::RaymarchShape;
    use loam_shape::polytope::Polytope4;

    #[test]
    fn shared_depth_is_cleared_exactly_when_a_pass_reads_it() {
        for wireframe_enabled in [false, true] {
            assert!(shared_depth_is_read(SurfaceMode::Raster, wireframe_enabled));
        }
        for surface_mode in [SurfaceMode::Sdf, SurfaceMode::Off] {
            assert!(shared_depth_is_read(surface_mode, true));
            assert!(!shared_depth_is_read(surface_mode, false));
        }
    }

    #[test]
    fn section_layers_merge_exactly_when_they_share_a_node() {
        const ALPHAS: [f32; 4] = [0.0, 0.5, 1.0, 1.5];
        for cross_alpha in ALPHAS {
            for cap_alpha in ALPHAS {
                let layer = |surface_alpha| SectionLayer {
                    perimeter: false,
                    surface_alpha,
                };
                let cross = layer(cross_alpha);
                let cap = layer(cap_alpha);
                let both_drawn = cross.fill_visible() && cap.fill_visible();
                let same_node =
                    section_alpha_is_opaque(cross_alpha) == section_alpha_is_opaque(cap_alpha);
                assert_eq!(
                    section_layers_share_a_node(cross, cap),
                    both_drawn && same_node,
                    "cross alpha {cross_alpha}, cap alpha {cap_alpha}"
                );
            }
        }
    }

    #[test]
    fn appending_a_section_mesh_rebases_its_indices_and_keeps_draw_order() {
        let mesh = |offset: f32, tris: &[[u32; 3]], verts: usize| loam_shape::TriangleMesh::<3> {
            vertices: (0..verts).map(|i| [offset + i as f32, 0.0, 0.0]).collect(),
            colors: (0..verts).map(|_| [offset, 0.0, 0.0, 1.0]).collect(),
            indices: tris.to_vec(),
        };
        let mut dst = mesh(0.0, &[[0, 1, 2]], 3);
        let src = mesh(10.0, &[[1, 2, 3], [0, 1, 2]], 4);
        append_triangle_mesh(&mut dst, &src);

        assert_eq!(dst.vertices.len(), 7);
        assert_eq!(dst.colors.len(), dst.vertices.len());
        assert_eq!(dst.indices, vec![[0, 1, 2], [4, 5, 6], [3, 4, 5]]);
        for (tri, src_tri) in dst.indices[1..].iter().zip(&src.indices) {
            for (&i, &si) in tri.iter().zip(src_tri.iter()) {
                assert_eq!(dst.vertices[i as usize], src.vertices[si as usize]);
            }
        }
    }

    #[test]
    fn a_merged_opaque_frame_draws_both_layers_triangles() {
        let physics = PlaygroundPhysics::new(1, BODY_SIZE);
        let frame = frame_of(
            &physics,
            ROW_16,
            Rotor4::IDENTITY,
            Projection::Perspective4D {
                focal_distance: 3.0,
            },
            SLICE_W,
            CAMERA_DISTANCE,
        );
        let opaque = SectionLayer {
            perimeter: false,
            surface_alpha: 1.0,
        };
        assert!(section_layers_share_a_node(opaque, opaque));

        let mut buffers = OverlayBuffers::default();
        buffers.sections(&frame, opaque, opaque);
        let cross_tris = buffers.cross_faces.indices.len();
        let cap_tris = buffers.cap_faces.indices.len();
        assert!(
            cross_tris > 0 && cap_tris > 0,
            "fixture must build both layers, got {cross_tris} and {cap_tris}"
        );

        append_triangle_mesh(&mut buffers.cross_faces, &buffers.cap_faces);
        assert_eq!(buffers.cross_faces.indices.len(), cross_tris + cap_tris);
        for tri in &buffers.cross_faces.indices {
            for &i in tri {
                assert!((i as usize) < buffers.cross_faces.vertices.len());
            }
        }
    }

    const ROW: &[ShapeEntry] = &[ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Pentatope),
        body_color: [0.95, 0.55, 0.30],
        label: "5-cell",
        long_name: "pentachoron",
    }];

    const CELL16: ShapeEntry = ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Cell16),
        body_color: [0.30, 0.70, 0.95],
        label: "16-cell",
        long_name: "hexadecachoron",
    };
    const ROW_16: &[ShapeEntry] = &[CELL16];
    const ROW_16_PAIR: &[ShapeEntry] = &[CELL16, CELL16];
    const ROW_MIXED: &[ShapeEntry] = &[CELL16, ROW[0]];

    const SLICE_W: f32 = 0.2;

    const CAMERA_DISTANCE: f32 = 4.0;

    const TRANSLATE_TOL: f32 = 1e-5;

    const STEPS_TO_CROSS_THE_GAP: usize = 120;

    struct TranslatedPair {
        thrown: PlaygroundPhysics,
        at_rest: PlaygroundPhysics,
        spin: Rotor4,
        composed: Rotor4,
        delta: Vec3,
    }

    fn translated_pair() -> TranslatedPair {
        let spin = rotor_at(Plane4::Xy, 0.7);
        let mut thrown = PlaygroundPhysics::new(1, BODY_SIZE);
        let layout = Vec4::from_array(body_position(0, 1));
        thrown.world.bodies[0].apply_impulse_at_point(
            &EuclideanR4,
            Vec4::X * 0.5,
            layout + Vec4::W * 0.5,
        );
        thrown.step(24);

        let pose = thrown.pose(0, 1, spin);
        assert_eq!(
            pose.position.w, layout.w,
            "throw left the layout w, so the two worlds cut different geometry"
        );
        let delta = pose.position_r3() - layout.truncate();
        assert!(
            delta.length() > 0.05,
            "throw did not move the body's R³ centre"
        );
        let probe = Vec4::new(0.3, -0.2, 0.9, 0.1);
        assert!(
            (pose.rotor.apply(probe) - spin.apply(probe)).length() > 1e-2,
            "throw produced no visible rotation, so the rotor half of the pins is vacuous"
        );
        TranslatedPair {
            at_rest: PlaygroundPhysics::new(1, BODY_SIZE),
            composed: pose.rotor,
            thrown,
            spin,
            delta,
        }
    }

    fn frame(physics: &PlaygroundPhysics, spin: Rotor4) -> RowFrame<'_> {
        frame_of(
            physics,
            ROW,
            spin,
            Projection::Identity,
            0.0,
            CAMERA_DISTANCE,
        )
    }

    fn uniform_spins(slots: usize, rotor: Rotor4) -> &'static SlotSpins {
        Box::leak(Box::new(SlotSpins::uniform(slots, rotor)))
    }

    fn frame_of<'a>(
        physics: &'a PlaygroundPhysics,
        row: &'a [ShapeEntry],
        spin: Rotor4,
        projection: Projection<4>,
        w_slice: f32,
        camera_distance: f32,
    ) -> RowFrame<'a> {
        RowFrame {
            physics,
            row,
            spins: uniform_spins(row.len(), spin),
            body_size: BODY_SIZE,
            projection,
            w_slice,
            camera_distance,
        }
    }

    fn rotor_at(plane: Plane4, angle: f32) -> Rotor4 {
        (plane.unit_bivector() * angle).exp().normalize()
    }

    fn slice_colored_style() -> WireframeStyle {
        WireframeStyle {
            color_mode: WireframeColorMode::Active,
            alpha: 1.0,
            width_px: 1.8,
            nearest_active: false,
            space_blend: 0.0,
            hyperslice: None,
        }
    }

    struct BuiltRow {
        edges: Vec<[f32; 3]>,
        perimeter: Vec<[f32; 3]>,
        sprites: Vec<[f32; 3]>,
        sprite_colors: Vec<[f32; 4]>,
        cross_caps: Vec<[f32; 3]>,
        projected_caps: Vec<[f32; 3]>,
        projected_cap_triangles: Vec<[u32; 3]>,
        cross_cap_triangles: Vec<[u32; 3]>,
    }

    #[derive(Default)]
    struct OverlayBuffers {
        palette_cache: std::collections::HashMap<Polytope4, Vec<[f32; 4]>>,
        centers_cache: std::collections::HashMap<Polytope4, Vec<Vec4>>,
        slerp: Vec<Vec4>,
        local_vertices: Vec<Vec4>,
        center_locals: Vec<Vec4>,
        cell_strengths: Vec<f32>,
        section_scratch: SectionScratch,
        cross_faces: loam_shape::TriangleMesh<3>,
        cap_faces: loam_shape::TriangleMesh<3>,
        proj: Vec<Vec3>,
        body_perimeter: LineMesh<3>,
        section_edges: LineMesh<3>,
        parent_lines: LineMesh<3>,
        sprites: loam_shape::PointMesh<3>,
    }

    impl OverlayBuffers {
        fn wireframe(
            &mut self,
            frame: &RowFrame<'_>,
            style: &WireframeStyle,
            cross: SectionLayer,
            cap: SectionLayer,
        ) {
            build_wireframe_meshes(
                frame,
                style,
                cross,
                cap,
                &mut self.palette_cache,
                &mut self.slerp,
                &mut self.local_vertices,
                &mut self.cell_strengths,
                &mut self.section_scratch,
                &mut self.body_perimeter,
                &mut self.section_edges,
                &mut self.parent_lines,
            );
        }

        fn sections(&mut self, frame: &RowFrame<'_>, cross: SectionLayer, cap: SectionLayer) {
            build_section_layer_meshes(
                frame,
                cross,
                cap,
                SectionBuffers {
                    local_vertices: &mut self.local_vertices,
                    proj_scratch: &mut self.proj,
                    cross_mesh: &mut self.cross_faces,
                    cap_mesh: &mut self.cap_faces,
                    section_scratch: &mut self.section_scratch,
                },
            );
        }

        fn points(&mut self, frame: &RowFrame<'_>, style: &PointsStyle) {
            build_points_mesh(
                frame,
                style,
                &mut self.centers_cache,
                &mut self.local_vertices,
                &mut self.center_locals,
                &mut self.cell_strengths,
                &mut self.sprites,
            );
        }
    }

    fn points_style_for(style: &WireframeStyle) -> PointsStyle {
        PointsStyle {
            color_mode: style.color_mode,
            show_vertices: true,
            show_cell_centers: true,
            size_px: 6.0,
        }
    }

    fn build_row(frame: &RowFrame<'_>, style: &WireframeStyle) -> BuiltRow {
        let cross = SectionLayer::CROSS_SECTION_DEFAULT;
        let cap = SectionLayer {
            perimeter: true,
            surface_alpha: 0.5,
        };
        let mut buffers = OverlayBuffers::default();
        buffers.wireframe(frame, style, cross, cap);
        buffers.points(frame, &points_style_for(style));

        let mut local_vertices = Vec::new();
        let mut proj_scratch = Vec::new();
        let mut cross_mesh = loam_shape::TriangleMesh::<3>::default();
        let mut cap_mesh = loam_shape::TriangleMesh::<3>::default();
        build_section_layer_meshes(
            frame,
            cross,
            cap,
            SectionBuffers {
                local_vertices: &mut local_vertices,
                proj_scratch: &mut proj_scratch,
                cross_mesh: &mut cross_mesh,
                cap_mesh: &mut cap_mesh,
                section_scratch: &mut SectionScratch::default(),
            },
        );

        BuiltRow {
            edges: segment_points(&buffers.parent_lines),
            perimeter: segment_points(&buffers.section_edges),
            sprites: buffers.sprites.positions,
            sprite_colors: buffers.sprites.colors,
            cross_caps: cross_mesh.vertices,
            projected_caps: cap_mesh.vertices,
            projected_cap_triangles: cap_mesh.indices,
            cross_cap_triangles: cross_mesh.indices,
        }
    }

    impl BuiltRow {
        fn meshes(&self) -> [(&str, &[[f32; 3]]); 5] {
            [
                ("parent wireframe", &self.edges),
                ("section perimeter", &self.perimeter),
                ("point sprites", &self.sprites),
                ("cross-section caps", &self.cross_caps),
                ("projected caps", &self.projected_caps),
            ]
        }

        fn assert_carried_from(&self, rest: &BuiltRow, delta: Vec3) {
            for ((what, live), (_, at_rest)) in self.meshes().iter().zip(rest.meshes().iter()) {
                assert_translated(live, at_rest, delta, what);
            }
            assert_eq!(
                self.sprite_colors, rest.sprite_colors,
                "sprite coloring diverged between the two worlds"
            );
            assert_eq!(
                (&self.cross_cap_triangles, &self.projected_cap_triangles),
                (&rest.cross_cap_triangles, &rest.projected_cap_triangles),
                "cap triangulation diverged, so the vertex pins compare unrelated points"
            );
        }

        fn assert_slots_separated_by(&self, delta: Vec3) {
            for (what, all) in self.meshes() {
                assert!(
                    !all.is_empty(),
                    "{what}: nothing was emitted, so the pin is vacuous"
                );
                let (first, second) = all.split_at(all.len() / 2);
                assert_translated(second, first, delta, what);
            }
        }
    }

    fn segment_points(mesh: &LineMesh<3>) -> Vec<[f32; 3]> {
        mesh.segments.iter().flat_map(|(a, b)| [*a, *b]).collect()
    }

    fn assert_translated(live: &[[f32; 3]], rest: &[[f32; 3]], delta: Vec3, what: &str) {
        assert!(
            !live.is_empty(),
            "{what}: nothing was emitted, so the pin is vacuous"
        );
        assert_eq!(
            live.len(),
            rest.len(),
            "{what}: the two worlds emitted different geometry"
        );
        for (i, (l, r)) in live.iter().zip(rest).enumerate() {
            let expected = Vec3::from_array(*r) + delta;
            assert!(
                (Vec3::from_array(*l) - expected).length() < TRANSLATE_TOL,
                "{what} point {i}: {l:?} is not the at-rest body carried to the live centre {expected:?}"
            );
        }
    }

    #[test]
    fn wireframe_meshes_follow_the_physics_pose() {
        let pair = translated_pair();
        let style = WireframeStyle {
            color_mode: WireframeColorMode::VertexGradient,
            alpha: 1.0,
            width_px: 1.8,
            nearest_active: false,
            space_blend: 0.0,
            hyperslice: None,
        };
        let cross = SectionLayer::CROSS_SECTION_DEFAULT;
        let cap = SectionLayer::PROJECTED_CAP_DEFAULT;
        let mut live = OverlayBuffers::default();
        let mut rest = OverlayBuffers::default();

        live.wireframe(&frame(&pair.thrown, pair.spin), &style, cross, cap);
        rest.wireframe(&frame(&pair.at_rest, pair.composed), &style, cross, cap);

        assert_translated(
            &segment_points(&live.parent_lines),
            &segment_points(&rest.parent_lines),
            pair.delta,
            "parent wireframe",
        );
        assert_translated(
            &segment_points(&live.section_edges),
            &segment_points(&rest.section_edges),
            pair.delta,
            "section perimeter",
        );
    }

    #[test]
    fn point_sprites_follow_the_physics_pose() {
        let pair = translated_pair();
        let style = PointsStyle {
            color_mode: WireframeColorMode::VertexGradient,
            show_vertices: true,
            show_cell_centers: true,
            size_px: 6.0,
        };
        let mut live = OverlayBuffers::default();
        let mut rest = OverlayBuffers::default();

        live.points(&frame(&pair.thrown, pair.spin), &style);
        rest.points(&frame(&pair.at_rest, pair.composed), &style);

        assert_translated(
            &live.sprites.positions,
            &rest.sprites.positions,
            pair.delta,
            "point sprites",
        );
    }

    #[test]
    fn section_caps_follow_the_physics_pose() {
        let pair = translated_pair();
        let cross = SectionLayer::CROSS_SECTION_DEFAULT;
        let cap = SectionLayer {
            perimeter: true,
            surface_alpha: 0.5,
        };
        let mut local_vertices = Vec::new();
        let mut proj_scratch = Vec::new();
        let mut live_cross = loam_shape::TriangleMesh::<3>::default();
        let mut live_cap = loam_shape::TriangleMesh::<3>::default();
        let mut rest_cross = loam_shape::TriangleMesh::<3>::default();
        let mut rest_cap = loam_shape::TriangleMesh::<3>::default();

        build_section_layer_meshes(
            &frame(&pair.thrown, pair.spin),
            cross,
            cap,
            SectionBuffers {
                local_vertices: &mut local_vertices,
                proj_scratch: &mut proj_scratch,
                cross_mesh: &mut live_cross,
                cap_mesh: &mut live_cap,
                section_scratch: &mut SectionScratch::default(),
            },
        );
        build_section_layer_meshes(
            &frame(&pair.at_rest, pair.composed),
            cross,
            cap,
            SectionBuffers {
                local_vertices: &mut local_vertices,
                proj_scratch: &mut proj_scratch,
                cross_mesh: &mut rest_cross,
                cap_mesh: &mut rest_cap,
                section_scratch: &mut SectionScratch::default(),
            },
        );

        assert_translated(
            &live_cross.vertices,
            &rest_cross.vertices,
            pair.delta,
            "cross-section caps",
        );
        assert_translated(
            &live_cap.vertices,
            &rest_cap.vertices,
            pair.delta,
            "projected caps",
        );
        assert_eq!(
            live_cross.indices, rest_cross.indices,
            "cap triangulation diverged, so the vertex pin above compares unrelated points"
        );
    }

    #[test]
    fn each_slot_renders_at_its_own_body() {
        let physics = PlaygroundPhysics::new(2, BODY_SIZE);
        let frame = frame_of(
            &physics,
            ROW_16_PAIR,
            rotor_at(Plane4::Xz, 0.5),
            Projection::Identity,
            SLICE_W,
            CAMERA_DISTANCE,
        );
        let built = build_row(&frame, &slice_colored_style());

        let layout = Vec4::from_array(body_position(1, 2)).truncate()
            - Vec4::from_array(body_position(0, 2)).truncate();
        assert!(layout.length() > 0.5, "the two slots share a position");
        built.assert_slots_separated_by(layout);
        let (first, second) = built.sprite_colors.split_at(built.sprite_colors.len() / 2);
        assert_eq!(
            first, second,
            "the two slots colored the same shape differently"
        );
    }

    #[test]
    fn a_body_lifted_off_the_slice_is_cut_where_physics_put_it() {
        let (lifted, lift) = thrown_along_w();
        let spin = rotor_at(Plane4::Xz, 0.5);
        let at_rest = PlaygroundPhysics::new(1, BODY_SIZE);
        let style = slice_colored_style();
        let build_at = |physics, w_slice| {
            build_row(
                &frame_of(
                    physics,
                    ROW_16,
                    spin,
                    Projection::Identity,
                    w_slice,
                    CAMERA_DISTANCE,
                ),
                &style,
            )
        };

        let live = build_at(&lifted, SLICE_W + lift);
        let rest = build_at(&at_rest, SLICE_W);
        live.assert_carried_from(&rest, Vec3::ZERO);

        let off_slice = build_at(&lifted, SLICE_W);
        assert_ne!(
            off_slice.cross_caps, rest.cross_caps,
            "the cap did not move with the slice, so the pin above is vacuous"
        );
    }

    #[test]
    fn arcs_bow_onto_the_circumsphere_of_a_body_off_the_slice() {
        let (lifted, lift) = thrown_along_w();
        let spin = rotor_at(Plane4::Xz, 0.5);
        let at_rest = PlaygroundPhysics::new(1, BODY_SIZE);
        let mut style = slice_colored_style();
        style.space_blend = 1.0;
        let build_at = |physics, w_slice| {
            build_row(
                &frame_of(
                    physics,
                    ROW_16,
                    spin,
                    Projection::Identity,
                    w_slice,
                    CAMERA_DISTANCE,
                ),
                &style,
            )
        };

        let live = build_at(&lifted, SLICE_W + lift);
        let rest = build_at(&at_rest, SLICE_W);
        assert_eq!(
            live.edges.len(),
            Polytope4::Cell16.topology().edges.len() * SPACE_TESSELLATION_SAMPLES * 2,
            "blend 1 did not subdivide the edges, so the arc path is not under test"
        );
        live.assert_carried_from(&rest, Vec3::ZERO);

        let origin_centred = (BODY_SIZE * BODY_SIZE + lift * lift).sqrt();
        assert!(
            origin_centred - BODY_SIZE > 100.0 * TRANSLATE_TOL,
            "lift {lift} inflates the origin-centred sphere by only {}",
            origin_centred - BODY_SIZE
        );
    }

    #[test]
    fn the_honest_layer_ignores_the_projection_the_cap_scales_by_it() {
        const FOCAL: f32 = 2.0;
        let physics = PlaygroundPhysics::new(1, BODY_SIZE);
        let frame = frame_of(
            &physics,
            ROW_16,
            rotor_at(Plane4::Xz, 0.5),
            Projection::Perspective4D {
                focal_distance: FOCAL,
            },
            SLICE_W,
            CAMERA_DISTANCE,
        );
        let built = build_row(&frame, &slice_colored_style());

        let scale = FOCAL / (FOCAL - SLICE_W);
        assert!(
            (scale - 1.0).abs() > 0.1,
            "the two layers would coincide at this focal distance"
        );
        let centre = Vec4::from_array(body_position(0, 1)).truncate();
        assert_scaled_about(
            &built.projected_caps,
            &built.cross_caps,
            centre,
            scale,
            "section cap fill",
        );
        let (honest, cap) = built.perimeter.split_at(built.perimeter.len() / 2);
        assert_scaled_about(cap, honest, centre, scale, "section perimeter");
    }

    #[test]
    fn hyperslice_keeps_the_edges_whose_cells_cross_the_slab() {
        const THICKNESS: f32 = 0.2;
        const TILT: f32 = 0.5;
        // Slab `[0.4, 0.6]`: above the tilted `e_z` w-extent
        // (`BODY_SIZE · sin TILT` = 0.34) and below the `e_w` one
        // (`BODY_SIZE · cos TILT` = 0.61), so a cell's w-range straddles the
        // near boundary exactly when the cell holds `+e_w`, and the edges
        // reaching only `-e_w` cells are the ones culled.
        const SLAB_CENTRE_W: f32 = 0.5;
        let physics = PlaygroundPhysics::new(1, BODY_SIZE);
        let frame = frame_of(
            &physics,
            ROW_16,
            rotor_at(Plane4::Zw, TILT),
            Projection::Identity,
            SLAB_CENTRE_W,
            CAMERA_DISTANCE,
        );
        let mut style = slice_colored_style();
        style.hyperslice = Some(THICKNESS);

        let topo = Polytope4::Cell16.topology();
        let mut local = Vec::new();
        let centre = frame.body_local(0, topo.vertices, BODY_SIZE, &mut local);
        let mesh_under = |keep: fn(bool) -> bool| -> Vec<([f32; 3], [f32; 3])> {
            topo.edges
                .iter()
                .filter(|[i, j]| {
                    topo.cells.iter().any(|cell| {
                        cell.contains(i) && cell.contains(j) && {
                            let (w_min, w_max) = cell_w_range(cell, &local);
                            keep(slab_overlaps(w_min, w_max, SLAB_CENTRE_W, THICKNESS))
                        }
                    })
                })
                .map(|&[i, j]| {
                    (
                        (local[i as usize].truncate() + centre).to_array(),
                        (local[j as usize].truncate() + centre).to_array(),
                    )
                })
                .collect()
        };
        let expected = mesh_under(|overlaps| overlaps);
        assert!(
            !expected.is_empty() && expected.len() < topo.edges.len(),
            "the slab kept {} of {} edges, so the cull is not under test",
            expected.len(),
            topo.edges.len()
        );
        assert_ne!(
            mesh_under(|overlaps| !overlaps),
            expected,
            "the inverted cull emits the same mesh here, so the pin below \
             cannot see the predicate's polarity"
        );

        let mut buffers = OverlayBuffers::default();
        buffers.wireframe(
            &frame,
            &style,
            SectionLayer::CROSS_SECTION_DEFAULT,
            SectionLayer::PROJECTED_CAP_DEFAULT,
        );
        assert_eq!(
            buffers.parent_lines.segments, expected,
            "the cull kept a different edge set than the slab crosses"
        );
    }

    #[test]
    fn space_blend_one_bows_edges_onto_the_circumsphere() {
        let physics = PlaygroundPhysics::new(1, BODY_SIZE);
        let frame = frame_of(
            &physics,
            ROW_16,
            Rotor4::IDENTITY,
            Projection::Identity,
            SLICE_W,
            CAMERA_DISTANCE,
        );
        let topo = Polytope4::Cell16.topology();
        let centre = Vec4::from_array(body_position(0, 1)).truncate();

        let mut chord_style = slice_colored_style();
        chord_style.space_blend = 0.0;
        let chords = build_row(&frame, &chord_style).edges;
        assert_eq!(chords.len(), topo.edges.len() * 2);

        let mut arc_style = slice_colored_style();
        arc_style.space_blend = 1.0;
        let arcs = build_row(&frame, &arc_style).edges;
        assert_eq!(
            arcs.len(),
            topo.edges.len() * SPACE_TESSELLATION_SAMPLES * 2,
            "blend 1 did not subdivide the edges"
        );

        let equatorial: Vec<usize> = topo
            .edges
            .iter()
            .enumerate()
            .filter(|(_, &[i, j])| {
                topo.vertices[i as usize].w == 0.0 && topo.vertices[j as usize].w == 0.0
            })
            .map(|(e, _)| e)
            .collect();
        assert!(!equatorial.is_empty(), "no edge lies in w = 0");
        let block = SPACE_TESSELLATION_SAMPLES * 2;
        for e in equatorial {
            for (k, p) in arcs[e * block..(e + 1) * block].iter().enumerate() {
                let radius = (Vec3::from_array(*p) - centre).length();
                assert!(
                    (radius - BODY_SIZE).abs() < 1e-4,
                    "edge {e} sample {k} sits at {radius}, off the circumsphere {BODY_SIZE}"
                );
            }
            let mid = (Vec3::from_array(chords[e * 2]) + Vec3::from_array(chords[e * 2 + 1])) * 0.5;
            assert!(
                (mid - centre).length() < BODY_SIZE - 0.05,
                "edge {e}'s chord already lies on the circumsphere"
            );
        }
    }

    #[test]
    fn a_nearer_camera_clips_the_16cell_harder() {
        const NEAR: f32 = 4.0;
        const FAR: f32 = 12.0;
        let near_radius = stereographic_view_radius(Polytope4::Cell16, NEAR);
        let far_radius = stereographic_view_radius(Polytope4::Cell16, FAR);
        assert!(
            near_radius < far_radius,
            "both distances resolve to the same clip radius"
        );

        let physics = PlaygroundPhysics::new(1, BODY_SIZE);
        let spin = rotor_at(Plane4::Zw, 1.1);
        let built_at = |camera_distance| {
            build_row(
                &frame_of(
                    &physics,
                    ROW_16,
                    spin,
                    Projection::Stereographic { pole: Vec4::W },
                    0.45,
                    camera_distance,
                ),
                &slice_colored_style(),
            )
        };
        let near = built_at(NEAR);
        let far = built_at(FAR);
        let centre = Vec4::from_array(body_position(0, 1)).truncate();

        for (what, near_mesh, far_mesh) in [
            ("parent wireframe", &near.edges, &far.edges),
            ("section perimeter", &near.perimeter, &far.perimeter),
            ("point sprites", &near.sprites, &far.sprites),
        ] {
            let near_max = max_local_radius(near_mesh, centre);
            let far_max = max_local_radius(far_mesh, centre);
            assert!(
                near_max <= near_radius,
                "{what}: kept a sample at {near_max}, past the near clip {near_radius}"
            );
            assert!(
                far_max > near_radius && far_max <= far_radius,
                "{what}: the far view's {far_max} is not between the two clip radii"
            );
        }
        assert!(
            near.projected_cap_triangles.len() < far.projected_cap_triangles.len(),
            "the near camera dropped no cap triangles"
        );
    }

    fn max_local_radius(points: &[[f32; 3]], centre: Vec3) -> f32 {
        points
            .iter()
            .map(|p| (Vec3::from_array(*p) - centre).length())
            .fold(0.0_f32, f32::max)
    }

    fn thrown_along_w() -> (PlaygroundPhysics, f32) {
        let mut physics = PlaygroundPhysics::new(1, BODY_SIZE);
        physics.world.bodies[0].apply_impulse(Vec4::W);
        physics.step(15);
        let pose = physics.pose(0, 1, Rotor4::IDENTITY);
        let layout = Vec4::from_array(body_position(0, 1));
        assert_eq!(pose.rotor, Rotor4::IDENTITY, "the throw rotated the body");
        assert_eq!(pose.position.truncate(), layout.truncate());
        assert!(
            pose.position.w > 0.05 && pose.position.w < BODY_SIZE,
            "lift {} left the body's own w-extent",
            pose.position.w
        );
        let lift = pose.position.w;
        (physics, lift)
    }

    /// Counts what the thread running a probe asks of the allocator. The test
    /// runner gives each test its own thread, so a probe never sees a
    /// concurrent test's allocations; the counter is thread-local rather than
    /// global for exactly that reason. Const-initialised so reading it inside
    /// `alloc` cannot itself allocate.
    mod alloc_probe {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        thread_local! {
            static BYTES: Cell<usize> = const { Cell::new(0) };
        }

        pub struct Counting;

        unsafe impl GlobalAlloc for Counting {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                let _ = BYTES.try_with(|bytes| bytes.set(bytes.get() + layout.size()));
                System.alloc(layout)
            }

            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                System.dealloc(ptr, layout)
            }

            unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
                let _ = BYTES.try_with(|bytes| bytes.set(bytes.get() + new_size));
                System.realloc(ptr, layout, new_size)
            }
        }

        pub fn bytes_allocated_by(body: impl FnOnce()) -> usize {
            let before = BYTES.with(Cell::get);
            body();
            BYTES.with(Cell::get) - before
        }
    }

    // Crate-wide, not test-wide: `#[global_allocator]` is a per-binary
    // singleton, so every test in this crate allocates through `Counting` and a
    // second declaration anywhere in it is an E0152 hard error, not a merge
    // conflict.
    #[global_allocator]
    static COUNTING_ALLOCATOR: alloc_probe::Counting = alloc_probe::Counting;

    #[test]
    fn a_warm_overlay_frame_reaches_the_allocator_zero_times() {
        let physics = PlaygroundPhysics::new(2, BODY_SIZE);
        let frame = frame_of(
            &physics,
            ROW_16_PAIR,
            rotor_at(Plane4::Zw, 0.4),
            Projection::Stereographic { pole: Vec4::W },
            SLICE_W,
            CAMERA_DISTANCE,
        );
        let mut style = slice_colored_style();
        style.space_blend = 1.0;
        let points_style = points_style_for(&style);
        let no_perimeter = SectionLayer {
            perimeter: false,
            surface_alpha: 0.0,
        };

        let mut buffers = OverlayBuffers::default();
        buffers.wireframe(&frame, &style, no_perimeter, no_perimeter);
        buffers.points(&frame, &points_style);
        assert!(
            !buffers.parent_lines.segments.is_empty() && !buffers.sprites.positions.is_empty(),
            "the fixture emitted no geometry, so the pin is vacuous"
        );

        let bytes = alloc_probe::bytes_allocated_by(|| {
            buffers.wireframe(&frame, &style, no_perimeter, no_perimeter);
            buffers.points(&frame, &points_style);
        });
        assert_eq!(
            bytes, 0,
            "a warm frame asked the allocator for {bytes} bytes through the overlay builders"
        );
    }

    #[test]
    fn the_perimeter_path_reaches_the_allocator_zero_times() {
        let physics = PlaygroundPhysics::new(1, BODY_SIZE);
        let frame = frame_of(
            &physics,
            ROW_16,
            rotor_at(Plane4::Xz, 0.5),
            Projection::Identity,
            SLICE_W,
            CAMERA_DISTANCE,
        );
        let style = slice_colored_style();
        let cross = SectionLayer::CROSS_SECTION_DEFAULT;
        let cap = SectionLayer {
            perimeter: true,
            surface_alpha: 0.5,
        };

        let mut buffers = OverlayBuffers::default();
        buffers.wireframe(&frame, &style, cross, cap);
        assert!(
            !buffers.section_edges.segments.is_empty(),
            "no perimeter was emitted, so the pin is vacuous"
        );

        let warm =
            alloc_probe::bytes_allocated_by(|| buffers.wireframe(&frame, &style, cross, cap));
        assert_eq!(
            warm, 0,
            "a warm frame with both perimeters on asked the allocator for {warm} bytes"
        );
    }

    #[test]
    fn cell_centre_sprites_come_from_each_slots_own_polytope() {
        let physics = PlaygroundPhysics::new(2, BODY_SIZE);
        let frame = frame_of(
            &physics,
            ROW_MIXED,
            rotor_at(Plane4::Xz, 0.5),
            Projection::Identity,
            SLICE_W,
            CAMERA_DISTANCE,
        );
        let mut style = points_style_for(&slice_colored_style());
        style.show_vertices = false;
        style.show_cell_centers = true;

        let mut buffers = OverlayBuffers::default();
        buffers.points(&frame, &style);

        let expected =
            Polytope4::Cell16.topology().cells.len() + Polytope4::Pentatope.topology().cells.len();
        assert_ne!(
            Polytope4::Cell16.topology().cells.len(),
            Polytope4::Pentatope.topology().cells.len(),
            "the two fixture shapes have the same cell count, so this pin              cannot see a memo that served one slot the other's table"
        );
        assert_eq!(
            buffers.sprites.positions.len(),
            expected,
            "the cell-centre memo served a slot the wrong polytope's cells"
        );
    }

    #[test]
    fn the_fill_path_reaches_the_allocator_zero_times() {
        let physics = PlaygroundPhysics::new(1, BODY_SIZE);
        let frame = frame_of(
            &physics,
            ROW_16,
            rotor_at(Plane4::Xz, 0.5),
            Projection::Identity,
            SLICE_W,
            CAMERA_DISTANCE,
        );
        let cross = SectionLayer::CROSS_SECTION_DEFAULT;
        let cap = SectionLayer {
            perimeter: true,
            surface_alpha: 0.5,
        };

        let mut buffers = OverlayBuffers::default();
        buffers.sections(&frame, cross, cap);
        assert!(
            !buffers.cross_faces.vertices.is_empty() && !buffers.cap_faces.vertices.is_empty(),
            "no fill was emitted, so the pin is vacuous"
        );

        let warm = alloc_probe::bytes_allocated_by(|| buffers.sections(&frame, cross, cap));
        assert_eq!(
            warm, 0,
            "a warm frame with both fills on asked the allocator for {warm} bytes"
        );
    }

    fn assert_scaled_about(
        scaled: &[[f32; 3]],
        base: &[[f32; 3]],
        centre: Vec3,
        scale: f32,
        what: &str,
    ) {
        assert!(
            !base.is_empty(),
            "{what}: nothing was emitted, so the pin is vacuous"
        );
        assert_eq!(
            scaled.len(),
            base.len(),
            "{what}: the two layers emitted different geometry"
        );
        for (i, (s, b)) in scaled.iter().zip(base).enumerate() {
            let expected = (Vec3::from_array(*b) - centre) * scale + centre;
            assert!(
                (Vec3::from_array(*s) - expected).length() < TRANSLATE_TOL,
                "{what} point {i}: {s:?} is not {b:?} scaled by {scale} about {centre:?}"
            );
        }
    }

    const FIXTURE_DT: f32 = 1.0 / 60.0;

    /// Past `PENETRATION_SLOP`, so the narrowphase reports rather than
    /// grazes.
    const FIXTURE_OVERLAP: f32 = 0.2;

    /// Far past the sum of two bounding radii, so the broadphase cannot
    /// couple two pairs into one island.
    const FIXTURE_GROUP_GAP: f32 = 20.0;

    /// Fast enough that the Coulomb clamp `|jt| <= mu*jn` binds on the
    /// first step, so the tangent accumulator is the cap rather than a
    /// rounding artefact.
    const FIXTURE_SLIDE_SPEED: f32 = 3.0;

    fn contacting_world(pairs: usize) -> PlaygroundPhysics {
        let mut physics = PlaygroundPhysics::new(2 * pairs, BODY_SIZE);
        for pair in 0..pairs {
            let base = Vec4::X * (FIXTURE_GROUP_GAP * pair as f32);
            physics.world.bodies[2 * pair].position = base;
            physics.world.bodies[2 * pair + 1].position =
                base + Vec4::X * (2.0 * BODY_SIZE - FIXTURE_OVERLAP);
        }
        physics.world.step(FIXTURE_DT);
        assert_eq!(
            physics.world.manifolds.len(),
            pairs,
            "the fixture layout did not produce one manifold per pair"
        );
        physics
    }

    fn every_layer() -> PhysicsOverlay {
        PhysicsOverlay {
            contacts: true,
            normals: true,
            impulses: true,
            islands: true,
            ..PhysicsOverlay::default()
        }
    }

    fn only(layer: fn(&mut PhysicsOverlay)) -> PhysicsOverlay {
        let mut overlay = PhysicsOverlay::default();
        layer(&mut overlay);
        overlay
    }

    const ROW_PAIR: &[ShapeEntry] = &[ROW[0], ROW[0]];
    const ROW_FOUR: &[ShapeEntry] = &[ROW[0], ROW[0], ROW[0], ROW[0]];

    fn overlay_frame(physics: &PlaygroundPhysics) -> RowFrame<'_> {
        let row = match physics.world.bodies.len() {
            2 => ROW_PAIR,
            4 => ROW_FOUR,
            n => panic!("no fixture row of {n} slots"),
        };
        RowFrame {
            physics,
            row,
            spins: uniform_spins(row.len(), Rotor4::IDENTITY),
            body_size: BODY_SIZE,
            projection: Projection::Identity,
            w_slice: 0.0,
            camera_distance: CAMERA_DISTANCE,
        }
    }

    fn overlay_mesh(physics: &PlaygroundPhysics, overlay: &PhysicsOverlay) -> LineMesh<3> {
        let mut mesh = LineMesh::<3>::default();
        build_physics_overlay_mesh(&overlay_frame(physics), overlay, &mut mesh);
        assert_eq!(mesh.colors.len(), mesh.segments.len());
        assert_eq!(mesh.widths.len(), mesh.segments.len());
        mesh
    }

    #[test]
    fn every_contact_emits_one_normal_along_the_stored_direction() {
        let physics = contacting_world(2);
        let mesh = overlay_mesh(&physics, &only(|o| o.normals = true));
        let world = &physics.world;

        let contacts: usize = world.manifolds.values().map(|m| m.points.len()).sum();
        assert!(contacts > 0, "fixture produced no contacts");
        assert_eq!(
            mesh.segments.len(),
            contacts,
            "the normals layer emitted {} segments for {contacts} contacts",
            mesh.segments.len()
        );

        let mut segments = mesh.segments.iter();
        for (&(a, b), manifold) in world.manifolds.iter() {
            let separation = (world.bodies[b].position - world.bodies[a].position).truncate();
            for cp in &manifold.points {
                let &(from, to) = segments.next().expect("one segment per contact");
                let point = cp.world_point.truncate();
                assert!(
                    (Vec3::from_array(from) - point).length() < TRANSLATE_TOL,
                    "normal starts at {from:?}, not at the contact point {point:?}"
                );
                let drawn = Vec3::from_array(to) - Vec3::from_array(from);
                let expected = cp.normal.truncate() * (NORMAL_LEN_FRACTION * BODY_SIZE);
                assert!(
                    (drawn - expected).length() < TRANSLATE_TOL,
                    "normal drawn as {drawn:?}, not {expected:?}"
                );
                assert!(
                    drawn.dot(separation) > 0.0,
                    "normal runs from A toward B; the layer would misreport a flipped normal"
                );
            }
        }
    }

    #[test]
    fn impulse_bar_length_tracks_the_accumulated_impulse() {
        let physics = contacting_world(1);
        let world = &physics.world;
        let contacts: usize = world.manifolds.values().map(|m| m.points.len()).sum();
        let solved: f32 = world
            .manifolds
            .values()
            .flat_map(|m| m.points.iter())
            .map(|cp| cp.normal_impulse)
            .sum();
        assert!(
            solved > 0.0,
            "fixture accumulated no normal impulse, so the length pin is vacuous"
        );

        let base = only(|o| o.impulses = true);
        let mesh = overlay_mesh(&physics, &base);
        assert_eq!(
            mesh.segments.len(),
            2 * contacts,
            "the impulses layer emits a normal and a tangent bar per contact"
        );

        let doubled = overlay_mesh(
            &physics,
            &PhysicsOverlay {
                impulse_scale: base.impulse_scale * 2.0,
                ..base
            },
        );
        for (i, (&(from, to), &(from2, to2))) in
            mesh.segments.iter().zip(&doubled.segments).enumerate()
        {
            let short = Vec3::from_array(to) - Vec3::from_array(from);
            let long = Vec3::from_array(to2) - Vec3::from_array(from2);
            assert!(
                (long - short * 2.0).length() < TRANSLATE_TOL,
                "bar {i} did not scale with impulse_scale: {short:?} then {long:?}"
            );
        }

        let points = world.manifolds.values().flat_map(|m| m.points.iter());
        for (cp, chunk) in points.zip(mesh.segments.chunks_exact(2)) {
            let bar = Vec3::from_array(chunk[0].1) - Vec3::from_array(chunk[0].0);
            let expected = cp.normal.truncate() * (-cp.normal_impulse * base.impulse_scale);
            assert!(
                (bar - expected).length() < TRANSLATE_TOL,
                "normal-impulse bar drawn as {bar:?}, not {expected:?}"
            );
        }
    }

    #[test]
    fn a_sliding_pair_draws_its_friction_bar_against_the_slide() {
        let mut physics = PlaygroundPhysics::new(2, BODY_SIZE);
        physics.world.bodies[0].position = Vec4::ZERO;
        physics.world.bodies[1].position = Vec4::X * (2.0 * BODY_SIZE - FIXTURE_OVERLAP);
        physics.world.bodies[0].velocity = Vec4::Y * FIXTURE_SLIDE_SPEED;
        physics.world.step(FIXTURE_DT);

        let overlay = only(|o| o.impulses = true);
        let mesh = overlay_mesh(&physics, &overlay);
        let world = &physics.world;
        let contacts: usize = world.manifolds.values().map(|m| m.points.len()).sum();
        assert_eq!(mesh.segments.len(), 2 * contacts);

        let mut chunks = mesh.segments.chunks_exact(2);
        let mut braked = 0;
        for (&(a, b), manifold) in world.manifolds.iter() {
            let lead = (world.bodies[a].velocity - world.bodies[b].velocity).truncate();
            for cp in &manifold.points {
                let chunk = chunks.next().expect("two bars per contact");
                let bar = Vec3::from_array(chunk[1].1) - Vec3::from_array(chunk[1].0);
                let expected = cp.tangent_impulse * overlay.impulse_scale;
                assert!(
                    (bar.length() - expected).abs() < TRANSLATE_TOL,
                    "friction bar runs {} world units for a {} accumulator",
                    bar.length(),
                    cp.tangent_impulse
                );
                if cp.tangent_impulse > 0.0 {
                    braked += 1;
                    assert!(
                        bar.dot(lead) > 0.0,
                        "friction bar {bar:?} runs with the slide {lead:?} it should brake"
                    );
                }
            }
        }
        assert!(
            braked > 0,
            "no contact accumulated friction, so the sign pin is vacuous"
        );
    }

    #[test]
    fn a_full_speed_flick_draws_its_bar_at_a_third_of_a_body_radius() {
        let mut physics = PlaygroundPhysics::new(2, BODY_SIZE);
        let from = Vec4::from_array(body_position(0, 2));
        let to = Vec4::from_array(body_position(1, 2));
        physics.throw(0, (to - from).normalize() * MAX_THROW_SPEED);

        let mut peak = 0.0f32;
        for _ in 0..STEPS_TO_CROSS_THE_GAP {
            physics.step(1);
            for manifold in physics.world.manifolds.values() {
                for cp in &manifold.points {
                    peak = peak.max(cp.normal_impulse);
                }
            }
        }

        assert!(
            (4.5..5.5).contains(&peak),
            "peak normal impulse {peak}: the prose at DEFAULT_IMPULSE_SCALE is now stale, so recompute the bar length before touching this bound"
        );
        let bar = peak * DEFAULT_IMPULSE_SCALE;
        assert!(
            (0.30..0.42).contains(&(bar / BODY_SIZE)),
            "bar runs {bar} world units, {} of a body radius",
            bar / BODY_SIZE
        );
    }

    #[test]
    fn one_island_marks_its_bodies_in_one_colour() {
        let physics = contacting_world(2);
        let world = &physics.world;
        let islands = world.islands();
        assert_eq!(islands.len(), 2, "fixture did not split into two islands");

        let mesh = overlay_mesh(&physics, &only(|o| o.islands = true));
        let expected: usize = islands
            .iter()
            .map(|i| 3 * i.bodies.len() + i.constraints.len())
            .sum();
        assert_eq!(mesh.segments.len(), expected);

        let mut per_island: Vec<[u32; 4]> = Vec::new();
        for island in &islands {
            let mut colors = std::collections::BTreeSet::new();
            for &id in &island.bodies {
                let centre = world.bodies[id].position.truncate();
                let mut arms = 0;
                for (&(from, to), &(color, _)) in mesh.segments.iter().zip(&mesh.colors) {
                    let mid = (Vec3::from_array(from) + Vec3::from_array(to)) * 0.5;
                    if (mid - centre).length() < TRANSLATE_TOL
                        && Vec3::from_array(from) != Vec3::from_array(to)
                    {
                        arms += 1;
                        colors.insert(color.map(f32::to_bits));
                    }
                }
                assert_eq!(arms, 3, "body {id:?} is not marked by a three-arm cross");
            }
            assert_eq!(
                colors.len(),
                1,
                "island {:?} marked its bodies in {} colours",
                island.id,
                colors.len()
            );
            let color = *colors.iter().next().expect("one colour");
            assert!(
                !per_island.contains(&color),
                "two islands share a colour, so the partition cannot be read off the overlay"
            );
            per_island.push(color);
        }
    }

    #[test]
    fn a_world_at_rest_draws_no_physics_overlay() {
        let physics = PlaygroundPhysics::new(4, BODY_SIZE);
        assert!(physics.at_rest());
        assert!(physics.world.manifolds.is_empty());
        let mesh = overlay_mesh(&physics, &every_layer());
        assert!(
            mesh.segments.is_empty(),
            "a resting world emitted {} segments",
            mesh.segments.len()
        );
    }

    #[test]
    fn each_overlay_layer_draws_only_its_own_geometry() {
        let physics = contacting_world(2);
        let contacts: usize = physics
            .world
            .manifolds
            .values()
            .map(|m| m.points.len())
            .sum();

        let layers: [(&str, PhysicsOverlay, usize); 4] = [
            ("contacts", only(|o| o.contacts = true), 3 * contacts),
            ("normals", only(|o| o.normals = true), contacts),
            ("impulses", only(|o| o.impulses = true), 2 * contacts),
            ("islands", only(|o| o.islands = true), {
                let islands = physics.world.islands();
                islands
                    .iter()
                    .map(|i| 3 * i.bodies.len() + i.constraints.len())
                    .sum()
            }),
        ];

        let all = overlay_mesh(&physics, &every_layer());
        let mut total = 0;
        for (name, overlay, expected) in layers {
            let mesh = overlay_mesh(&physics, &overlay);
            assert_eq!(
                mesh.segments.len(),
                expected,
                "the {name} layer alone emitted {} segments, expected {expected}",
                mesh.segments.len()
            );
            for segment in &mesh.segments {
                assert!(
                    all.segments.contains(segment),
                    "the {name} layer's {segment:?} is missing when every layer is on"
                );
            }
            total += expected;
        }
        assert_eq!(
            all.segments.len(),
            total,
            "the all-layers build is not exactly the four layers"
        );
    }

    #[test]
    fn a_hidden_physics_overlay_reaches_the_allocator_zero_times() {
        let physics = contacting_world(2);
        let frame = overlay_frame(&physics);
        let mut mesh = LineMesh::<3>::default();

        build_physics_overlay_mesh(&frame, &every_layer(), &mut mesh);
        assert!(!mesh.segments.is_empty(), "the fixture emitted nothing");
        let warm = mesh.segments.capacity();

        let hidden = PhysicsOverlay::default();
        assert!(!hidden.any_layer(), "the overlay ships with a layer on");
        let bytes = alloc_probe::bytes_allocated_by(|| {
            build_physics_overlay_mesh(&frame, &hidden, &mut mesh)
        });
        assert_eq!(
            bytes, 0,
            "a hidden overlay asked the allocator for {bytes} bytes"
        );
        assert!(mesh.segments.is_empty());
        assert_eq!(
            mesh.segments.capacity(),
            warm,
            "the hidden path dropped the buffer it will need again"
        );
    }

    #[test]
    fn the_contact_overlay_layers_reach_the_allocator_zero_times() {
        let physics = contacting_world(2);
        let frame = overlay_frame(&physics);
        let overlay = PhysicsOverlay {
            contacts: true,
            normals: true,
            impulses: true,
            ..PhysicsOverlay::default()
        };
        let mut mesh = LineMesh::<3>::default();
        build_physics_overlay_mesh(&frame, &overlay, &mut mesh);
        assert!(!mesh.segments.is_empty(), "the fixture emitted nothing");

        let warm = alloc_probe::bytes_allocated_by(|| {
            build_physics_overlay_mesh(&frame, &overlay, &mut mesh)
        });
        assert_eq!(
            warm, 0,
            "a warm contact overlay asked the allocator for {warm} bytes"
        );
    }
}
