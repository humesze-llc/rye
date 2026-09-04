use crate::verbs::WireframeControls;
use loam_app::egui;
use loam_app::shell::SceneRegistry;
use loam_egui::{
    media::{chevron_button, play_pause_button, refresh_button},
    slider_with_edit,
};

use crate::consts::{CONTROL_H, CONTROL_W, PLAY_PAUSE_W};
use crate::state::{
    apply_projection_selection_defaults, hyperslice_cull_active, DeferredAction, Demo,
    RotationMode, RotorTerm, SectionLayer, SurfaceMode, ViewMode, WireframeColorMode,
    WireframeProjection,
};

const OVERLAY_PAD: f32 = 16.0;

// The window pivots at `CENTER_BOTTOM`, so this is its bottom edge.
pub(crate) fn overlay_seat(ctx: &egui::Context) -> egui::Pos2 {
    let screen = ctx.content_rect();
    egui::pos2(screen.center().x, screen.bottom() - OVERLAY_PAD)
}

fn section_layer_controls(
    ui: &mut egui::Ui,
    title: &str,
    tooltip: &str,
    layer: &mut SectionLayer,
    wireframe_enabled: bool,
) {
    ui.label(egui::RichText::new(title).italics())
        .on_hover_text(tooltip);
    let perimeter_resp = ui
        .checkbox(&mut layer.perimeter, "Perimeter outline")
        .on_hover_text(
            "Cap-boundary outline. Draws in the wireframe overlay, so enable Wireframe to see it.",
        );
    if layer.perimeter && !wireframe_enabled {
        perimeter_resp
            .on_hover_text("Wireframe overlay is off, so the outline is not currently drawn.");
    }
    ui.horizontal(|ui| {
        ui.label("Fill alpha");
        ui.add(egui::Slider::new(&mut layer.surface_alpha, 0.0..=1.0).fixed_decimals(2))
            .on_hover_text("0 = off; below 1.0 composites translucently over the layers behind");
    });
}

impl Demo {
    pub(crate) fn render_expanded_body(&mut self, ui: &mut egui::Ui) {
        self.render_view_tab_row(ui);
        match self.view_mode {
            ViewMode::Shapes => self.render_shapes_section(ui),
            ViewMode::Single => self.render_single_body(ui),
            ViewMode::Filmstrip => self.render_filmstrip_body(ui),
        }
        ui.separator();
        self.render_rotation_tab_row(ui);
        if self.rotation_mode == RotationMode::Active {
            self.render_active_mode(ui);
        } else {
            self.render_composer_mode(ui);
        }
    }

    pub(crate) fn render_view_tab_row(&mut self, ui: &mut egui::Ui) {
        let mut staged = self.view_mode;
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut staged, ViewMode::Shapes, "Shapes")
                .on_hover_text("Compare shapes at the same w slice");
            ui.selectable_value(&mut staged, ViewMode::Single, "Single")
                .on_hover_text("Explore one shape and choose its projection");
            ui.selectable_value(&mut staged, ViewMode::Filmstrip, "Filmstrip")
                .on_hover_text("Compare one shape across slices and rotation times");
            ui.checkbox(&mut self.show_formula, "Formula")
                .on_hover_text("Show the current rotation formula");
        });
        if staged != self.view_mode {
            self.pending_view_mode = Some(staged);
        }
    }

    pub(crate) fn render_rotation_tab_row(&mut self, ui: &mut egui::Ui) {
        let mut staged = self.rotation_mode;
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut staged, RotationMode::Active, "Active set")
                .on_hover_text("Six checkbox-toggled bivectors (xy, xz, ...)");
            ui.selectable_value(&mut staged, RotationMode::Composer, "Composer")
                .on_hover_text("Sum of bivectors from the composed sequence");
        });
        if staged != self.rotation_mode {
            self.pending_mode = Some(staged);
        }
    }

    pub(crate) fn menu_contents(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Edit", |ui| {
            if ui
                .button("Reset orientation")
                .on_hover_text("Return every body in the row to its unrotated pose")
                .clicked()
            {
                self.spins.clear_orientation();
                self.rebuild_bodies();
                ui.close_kind(egui::UiKind::Menu);
            }
            if ui.button("Reset all").clicked() {
                self.reset();
                ui.close_kind(egui::UiKind::Menu);
            }
        });
        loam_egui::sticky_menu(ui, "View", |ui| {
            ui.checkbox(&mut self.show_controls, "Rotation controls (H)");
            ui.checkbox(&mut self.show_formula, "Formula popup");
            ui.checkbox(&mut self.example_callout.open, "Example callout");
            ui.separator();
            if ui.button("About this program").clicked() {
                self.show_help = true;
                egui::Popup::close_all(ui.ctx());
            }
        });
    }

    pub(crate) fn render_render_panel(&mut self, ctx: &egui::Context) {
        let prev_surface = self.surface_mode;
        let prev_projection = self.wireframe.projection;
        let sdf_disabled = self.sdf_blocked_by_heavy_polychora();
        let schlegel_cell_count = self.schlegel_subject().map(|p| p.cell_count() as u32);
        let Self {
            show_render_panel,
            surface_mode,
            cross_section,
            projected_cap,
            wireframe,
            wireframe_nearest_active,
            wireframe_color_mode,
            wireframe_hyperslice,
            wireframe_hyperslice_thickness,
            points_enabled,
            points_show_vertices,
            points_show_cell_centers,
            points_size_px,
            ..
        } = self;
        let WireframeControls {
            enabled: wireframe_enabled,
            projection: wireframe_projection,
            ..
        } = wireframe;
        loam_egui::floating_panel(
            ctx,
            "polytope-playground-render",
            "Render",
            show_render_panel,
            |ui| {
                ui.label(egui::RichText::new("Surface").strong());
                ui.radio_value(surface_mode, SurfaceMode::Raster, "Raster (default)");
                ui.add_enabled_ui(!sdf_disabled, |ui| {
                    let resp = ui.radio_value(surface_mode, SurfaceMode::Sdf, "SDF raymarch");
                    if sdf_disabled {
                        resp.on_disabled_hover_text(
                            "Disabled: 120-cell/600-cell SDFs crash the browser tab. \
                             Remove the heavy polychora to re-enable.",
                        );
                    }
                });
                ui.radio_value(surface_mode, SurfaceMode::Off, "Off");
                ui.separator();

                ui.label(egui::RichText::new("Cross-section").strong());
                section_layer_controls(
                    ui,
                    "Honest (drop-w)",
                    "The drop-w slice, never reprojected: the geometry the SDF \
                     shows. On by default so a projection change never distorts \
                     the slice.",
                    cross_section,
                    *wireframe_enabled,
                );
                section_layer_controls(
                    ui,
                    "Projected cap",
                    "The same slice reprojected through the active wireframe \
                     projection, so it can sit on a Schlegel / stereographic \
                     wireframe. Off by default.",
                    projected_cap,
                    *wireframe_enabled,
                );
                ui.separator();

                ui.label(egui::RichText::new("Wireframe").strong());
                ui.checkbox(wireframe_enabled, "Enabled");
                ui.add_enabled_ui(*wireframe_enabled, |ui| {
                    ui.checkbox(wireframe_nearest_active, "Nearest-active gradient");
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Color");
                        for mode in WireframeColorMode::ALL {
                            ui.radio_value(wireframe_color_mode, mode, mode.label());
                        }
                    });
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Projection");
                        for mode in WireframeProjection::ALL {
                            let selected = wireframe_projection.same_variant(mode);
                            if ui.radio(selected, mode.label()).clicked() && !selected {
                                let next = match (mode, *wireframe_projection) {
                                    (
                                        WireframeProjection::Schlegel { .. },
                                        WireframeProjection::Schlegel { cell_index },
                                    ) => WireframeProjection::Schlegel { cell_index },
                                    (m, _) => m,
                                };
                                *wireframe_projection = next;
                                apply_projection_selection_defaults(next, wireframe_enabled);
                            }
                        }
                    });
                    if let WireframeProjection::Schlegel { cell_index } = wireframe_projection {
                        ui.horizontal(|ui| {
                            ui.label("Boundary cell");
                            match schlegel_cell_count {
                                Some(count) => {
                                    ui.add(
                                        egui::DragValue::new(cell_index)
                                            .range(0..=count - 1)
                                            .speed(0.25),
                                    );
                                    ui.label(format!("of {count}"));
                                }
                                None => {
                                    ui.add_enabled(false, egui::DragValue::new(cell_index))
                                        .on_disabled_hover_text(
                                            "No polychoron in the row to project",
                                        );
                                }
                            }
                        });
                    }
                    ui.checkbox(wireframe_hyperslice, "Hyperslice cull (w-slab)");
                    let cull_active =
                        hyperslice_cull_active(*wireframe_hyperslice, *wireframe_projection);
                    ui.add_enabled_ui(cull_active, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Slab width");
                            ui.add(
                                egui::DragValue::new(wireframe_hyperslice_thickness)
                                    .range(
                                        crate::consts::HYPERSLICE_MIN_THICKNESS
                                            ..=2.0 * crate::consts::W_RANGE,
                                    )
                                    .speed(0.01),
                            );
                        });
                    });
                });
                ui.separator();

                ui.label(egui::RichText::new("Points").strong());
                ui.checkbox(points_enabled, "Enabled");
                ui.add_enabled_ui(*points_enabled, |ui| {
                    ui.checkbox(points_show_vertices, "Vertex markers");
                    ui.checkbox(points_show_cell_centers, "Cell centers");
                    ui.horizontal(|ui| {
                        ui.label("Size (px)");
                        ui.add(
                            egui::DragValue::new(points_size_px)
                                .range(1.0..=32.0)
                                .speed(0.25),
                        );
                    });
                });
            },
        );
        if self.surface_mode != prev_surface {
            self.rebuild_bodies();
        }
        if self.wireframe.projection != prev_projection {
            self.resolve_schlegel_cache();
        }
    }

    pub(crate) fn render_help_window(&mut self, ctx: &egui::Context) {
        loam_egui::floating_panel_builder(
            ctx,
            "polytope-playground-about",
            "About Polytope Playground",
            &mut self.show_help,
        )
        .resizable(true)
        .collapsible(false)
        .default_size(560.0, 460.0)
        .default_pos(egui::pos2(80.0, 80.0))
        .show(|ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Polytope Playground");
                ui.label("Rotate 4D shapes and explore their 3D cross-sections.");
                ui.add_space(8.0);
                egui::Grid::new("playground-shortcuts")
                    .num_columns(2)
                    .spacing([24.0, 8.0])
                    .show(ui, |ui| {
                        for (key, action) in [
                            ("Space / T", "Play or pause rotation"),
                            ("Up / Down", "Move the w slice"),
                            ("Left / Right", "Scrub rotation time"),
                            ("1 to 6", "Toggle rotation planes"),
                            ("H", "Expand or collapse controls"),
                            ("R", "Restart the scene"),
                            ("Backtick", "Open the console"),
                            ("Esc", "Exit"),
                        ] {
                            ui.monospace(key);
                            ui.label(action);
                            ui.end_row();
                        }
                    });
                ui.add_space(8.0);
                ui.label("Drag in the viewport to orbit. Right-click a slider value to type it.");
                ui.label("Drag the controls panel or formula window to move it.");
                ui.separator();
                ui.collapsing("Views and rotation", |ui| {
                    ui.label("Shapes shows a row. Drag the shape cards to reorder them.");
                    ui.label("Single shows one shape. Filmstrip compares slices and rotation times.");
                    ui.label("Active set selects rotation planes. Composer builds a sequence of terms.");
                    ui.label("The xy, xz, and yz planes rotate within 3D. The xw, yw, and zw planes turn through the fourth dimension.");
                    ui.label("Rotation controls affect the whole row. A timeline can hold a separate pose for one body.");
                });
                ui.collapsing("Shapes", |ui| {
                    ui.label("All six convex regular 4-polytopes are available:");
                    for label in [
                        "5-cell: 5 tetrahedra",
                        "8-cell (tesseract): 8 cubes",
                        "16-cell: 16 tetrahedra",
                        "24-cell: 24 octahedra",
                        "120-cell: 120 dodecahedra",
                        "600-cell: 600 tetrahedra",
                    ] {
                        ui.label(label);
                    }
                });
                ui.collapsing("Rotation handles", |ui| {
                    ui.label("Enter handles in the console to show the rings and axis handles.");
                    ui.label("Drag a ring to rotate the row in its plane. Drag an axis handle to move the row.");
                    ui.label("Red, green, and blue mark x, y, and z. Violet marks w and moves bodies through the slice.");
                    ui.label("Use Toybox to pick up and throw individual shapes.");
                });
                ui.collapsing("Scenes and console", |ui| {
                    for entry in crate::shell::Playground::SCENES {
                        ui.horizontal(|ui| {
                            ui.monospace(entry.slug);
                            ui.label(entry.label);
                        });
                    }
                    ui.label("Use the Demo menu or scene <slug> to switch scenes.");
                    ui.label("help lists commands. help <name> explains one. Tab completes commands and arguments.");
                    ui.label("Start a scene with --scene=<slug> natively or ?scene=<slug> in the browser.");
                    ui.label("--embed=1 or ?embed=1 hides the menu bar. The console stays available.");
                });
            });
        });
    }

    pub(crate) fn render_overlay(&mut self, ctx: &egui::Context) {
        let screen = ctx.content_rect();
        const OVERLAY_MAX_WIDTH: f32 = 768.0;
        const OVERLAY_MIN_WIDTH: f32 = 220.0;
        let natural_w = screen.width() - 2.0 * (OVERLAY_PAD + 10.0);
        let area_w = natural_w.clamp(OVERLAY_MIN_WIDTH, OVERLAY_MAX_WIDTH);

        let visuals = &ctx.style().visuals;
        let frame = egui::Frame::default()
            .fill(visuals.window_fill)
            .stroke(visuals.window_stroke)
            .corner_radius(visuals.window_corner_radius)
            .inner_margin(10.0);

        let default_bottom_centre = overlay_seat(ctx);

        let prev_strip_subject = self.strip_subject;

        egui::Window::new("polytope-playground-overlay")
            .id(egui::Id::new("polytope-playground-overlay"))
            .title_bar(false)
            .resizable(false)
            .collapsible(false)
            .movable(true)
            .auto_sized()
            .pivot(egui::Align2::CENTER_BOTTOM)
            .default_pos(default_bottom_centre)
            .default_width(area_w)
            .frame(frame)
            .show(ctx, |ui| {
                ui.set_width(area_w);
                if self.expanded {
                    egui::ScrollArea::vertical()
                        .max_height((screen.height() - 190.0).max(80.0))
                        .show(ui, |ui| self.render_expanded_body(ui));
                    ui.separator();
                }
                self.render_slider_strip(ui);
                self.render_rate_row(ui);
            });

        // Drained after the overlay closure returns, not mid-render.
        if let Some(new_mode) = self.pending_mode.take() {
            self.rotation_mode = new_mode;
        }
        if let Some(new_view) = self.pending_view_mode.take() {
            self.view_mode = new_view;
            self.rebuild_bodies();
            self.resolve_schlegel_cache();
        }
        for action in std::mem::take(&mut self.pending_actions) {
            match action {
                DeferredAction::DraftPush(plane) => self.draft.push(plane),
                DeferredAction::SeqCommitDraft => {
                    if !self.draft.is_empty() {
                        self.seq.push(RotorTerm {
                            planes: self.draft.clone(),
                            scalar: None,
                        });
                        self.draft.clear();
                    }
                }
                DeferredAction::DraftClear => self.draft.clear(),
                DeferredAction::SeqPushTerm(term) => self.seq.push(term),
            }
        }
        if self.view_mode == ViewMode::Single && self.strip_subject != prev_strip_subject {
            self.rebuild_bodies();
            self.resolve_schlegel_cache();
        }
    }

    pub(crate) fn render_slider_strip(&mut self, ui: &mut egui::Ui) {
        const VALUE_CELL_W: f32 = 72.0;
        let avail = ui.available_width();
        let spacing = ui.spacing().item_spacing.x;
        let slider_w = (avail - VALUE_CELL_W - spacing).max(140.0);
        ui.spacing_mut().slider_width = slider_w;

        let row_size = egui::vec2(avail, CONTROL_H);
        let row_layout = egui::Layout::left_to_right(egui::Align::Center);
        let w_range = self.effective_w_range();
        ui.allocate_ui_with_layout(row_size, row_layout, |ui| {
            let mut slice = self.w_slice;
            let formatted = format!("w {:>+.3}", slice);
            let interaction = slider_with_edit(
                ui,
                &mut slice,
                -w_range..=w_range,
                &formatted,
                "",
                3,
                VALUE_CELL_W,
            );
            if interaction.changed {
                loam_app::command::submit_line(&format!("slice {slice}"));
            }
        });
        let t_max = self.t_slider_max;
        ui.allocate_ui_with_layout(row_size, row_layout, |ui| {
            let mut seconds = self.rot_time;
            let formatted = format!("t {:>5.2}s", seconds);
            let interaction = slider_with_edit(
                ui,
                &mut seconds,
                0.0..=t_max,
                &formatted,
                "s",
                2,
                VALUE_CELL_W,
            );
            if interaction.changed {
                loam_app::command::submit_line(&format!("seek {seconds}"));
            }
        });
    }

    pub(crate) fn render_rate_row(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            let ctrl_size = egui::vec2(CONTROL_W, CONTROL_H);
            let play_size = egui::vec2(PLAY_PAUSE_W, CONTROL_H);
            if play_pause_button(ui, play_size, self.rotate)
                .on_hover_text("Play or pause rotation (Space)")
                .clicked()
            {
                loam_app::command::submit_line("spin");
            }
            if refresh_button(ui, ctrl_size)
                .on_hover_text("Reset rotation and slice")
                .clicked()
            {
                loam_app::command::submit_line("reset");
            }
            ui.label("Speed");
            for (rate, label, command) in [
                (0.25, "0.25x", "rate 0.25"),
                (0.5, "0.5x", "rate 0.5"),
                (1.0, "1x", "rate 1"),
                (2.0, "2x", "rate 2"),
                (4.0, "4x", "rate 4"),
            ] {
                if ui
                    .selectable_label(self.rate_scale == rate, label)
                    .clicked()
                {
                    loam_app::command::submit_line(command);
                }
            }
        });
        ui.horizontal(|ui| {
            if ui.button("Help").clicked() {
                self.show_help = true;
            }
            if ui.button("Render settings").clicked() {
                self.show_render_panel = !self.show_render_panel;
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if chevron_button(
                    ui,
                    egui::vec2(CONTROL_W, CONTROL_H),
                    !self.expanded,
                    if self.expanded {
                        "Collapse (H)"
                    } else {
                        "Expand controls (H)"
                    },
                )
                .clicked()
                {
                    self.expanded = !self.expanded;
                }
            });
        });
    }
}
