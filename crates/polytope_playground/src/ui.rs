use crate::verbs::WireframeControls;
use loam_app::egui;
use loam_app::shell::SceneRegistry;
use loam_egui::{
    media::{chevron_button, play_pause_button, rate_toggle, refresh_button},
    slider_with_edit,
};

use crate::consts::{CONTROL_H, CONTROL_W, PLAY_PAUSE_W};
use crate::state::{
    apply_projection_selection_defaults, hyperslice_cull_active, DeferredAction, Demo,
    RotationMode, RotorTerm, SectionLayer, SurfaceMode, ViewMode, WireframeColorMode,
    WireframeProjection,
};

const OVERLAY_PAD: f32 = 16.0;

fn scene_roster() -> String {
    crate::shell::Playground::SCENES
        .iter()
        .map(|entry| format!("{} ({})", entry.slug, entry.label))
        .collect::<Vec<_>>()
        .join(", ")
}

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
        ui.horizontal(|ui| {
            ui.selectable_value(&mut staged, ViewMode::Shapes, "Shapes")
                .on_hover_text("Side-by-side row of shapes at one w-slice");
            ui.selectable_value(&mut staged, ViewMode::Single, "Single")
                .on_hover_text(
                    "One subject (the picker below) at one w-slice with the full \
                     surface / wireframe / projection stack. Schlegel's boundary-cell \
                     selection needs this unambiguous single shape.",
                );
            ui.selectable_value(&mut staged, ViewMode::Filmstrip, "Filmstrip")
                .on_hover_text(
                    "One shape rendered N times across w-slices fanning out by \
                     ±BODY_SIZE around the w slider's value",
                );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.checkbox(&mut self.show_formula, "Show formula")
                    .on_hover_text("Top-right popup with the live exp(...) form of the rotor");
            });
        });
        if staged != self.view_mode {
            self.pending_view_mode = Some(staged);
        }
    }

    pub(crate) fn render_rotation_tab_row(&mut self, ui: &mut egui::Ui) {
        let mut staged = self.rotation_mode;
        ui.horizontal(|ui| {
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
                    ui.horizontal(|ui| {
                        ui.label("Color");
                        for mode in WireframeColorMode::ALL {
                            ui.radio_value(wireframe_color_mode, mode, mode.label());
                        }
                    });
                    ui.horizontal(|ui| {
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
                ui.heading("What this program shows");
                ui.label(
                    "You're looking at 3D cross-sections of four-dimensional \
                         polytopes. As they rotate through 4D space their cross-\
                         sections morph in characteristic ways; the point of the \
                         demo is to make 4D shape intuition reachable from 3D.",
                );
                ui.add_space(8.0);

                ui.heading("3D cross-sections, briefly");
                ui.label(
                    "A cross-section is what you get when a higher-\
                         dimensional object passes through a lower-dimensional \
                         space. A 3D apple intersecting a 2D table gives a 2D \
                         shape (a circle, an oval) that changes as the apple \
                         moves. One dimension up: a 4D polytope passing through \
                         3D gives a 3D shape that changes with the slicing w. \
                         That's what the w slider scrubs.",
                );
                ui.add_space(8.0);

                ui.heading("The shapes");
                ui.label("All six convex regular 4-polytopes (\"polychora\") ship:");
                ui.label("• 5-cell (pentachoron); 5 tetrahedra; the 4D simplex.");
                ui.label("• 8-cell (tesseract); 8 cubes; the 4D cube.");
                ui.label(
                    "• 16-cell (hexadecachoron); 16 tetrahedra; the 4D analog \
                         of the octahedron.",
                );
                ui.label(
                    "• 24-cell (icositetrachoron); 24 octahedra; uniquely 4D, \
                         no 3D analog.",
                );
                ui.label("• 120-cell (hecatonicosachoron); 120 dodecahedra.");
                ui.label(
                    "• 600-cell (hexacosichoron); 600 tetrahedra; the 4D \
                         analog of the icosahedron.",
                );
                ui.add_space(8.0);

                ui.heading("Rotation");
                ui.label(
                    "4D rotations are generated by bivectors (2-planes), not \
                         axes. There are six independent planes: xy, xz, xw, yz, \
                         yw, zw. The three w-involving planes pull a visible \
                         axis through the hidden 4th dimension and produce the \
                         interesting cross-section morphs; the three pure-3D \
                         planes rotate the cross-section as a rigid 3D shape.",
                );
                ui.label(
                    "Active-set mode: each plane has a checkbox (include in \
                         spin) and a -180..=180° slider (the rotor's component \
                         in that plane). Composer mode: build a sequence of \
                         exp(scalar · planes) terms via chips or the typed \
                         formula bar.",
                );
                ui.label(
                    "One rotation drives the whole row: the controls, the \
                         rings and the 1..6 keys write it and every body \
                         reads it, so the row turns as one. A timeline is \
                         the only thing that poses a single body, and it \
                         holds that pose for the rest of the run.",
                );
                ui.add_space(8.0);

                ui.heading("Views");
                ui.label(
                    "Shapes view: a row of polytopes side-by-side at one \
                         w-slice. Drag-and-drop to reorder. Filmstrip view: one \
                         polytope rendered N times across w-slices fanning out \
                         by ±BODY_SIZE around the slider's value, so the centre \
                         cell tracks w.",
                );
                ui.add_space(8.0);

                ui.heading("Keyboard");
                ui.label("• Space / T: toggle continuous spin.");
                ui.label("• Up / Down arrows: scrub w with the keyboard.");
                ui.label("• 1..6: toggle a plane in the row's Active set.");
                ui.label("• H: expand / collapse the controls panel.");
                ui.label("• R: restart the scene at its boot state.");
                ui.label("• Esc: exit.");
                ui.add_space(8.0);

                ui.heading("Mouse");
                ui.label(
                    "• This scene is a rotation study and has no grab: \
                         every rotation control writes the whole row, and \
                         the Toybox scene is where a shape is picked up \
                         and thrown.",
                );
                ui.label(
                    "• Drag a hypergimbal ring: turn the whole row in that \
                         ring's plane. The six rings stand at the row's \
                         centre and are the six rotation planes, drawn as \
                         the \
                         stereographic projection of a 16-cell; the ring under \
                         the cursor lights up. Rings sharing a hue are a \
                         plane and its orthogonal complement, so xy and zw \
                         face each other. They are off unless the console \
                         verb `handles` turns them on.",
                );
                ui.label(
                    "• Drag an arrowhead: slide the row along that axis. \
                         Red, green and blue are x, y and z. Violet is w, \
                         the axis with no direction on screen: it points \
                         away from all three, and dragging it moves the \
                         bodies off the 3D slice, so what changes is the \
                         shape of the cross-sections, not where the bodies \
                         sit.",
                );
                ui.label("• Drag in the viewport: orbit camera.");
                ui.label(
                    "• Right-click on any value label (w, t, plane angle, \
                         scalar): typed-edit popup.",
                );
                ui.label(
                    "• Drag the controls panel by its frame to move it; \
                         drag the formula popup the same way.",
                );
                ui.add_space(8.0);

                ui.heading("Scenes and the console");
                ui.label(
                    "Backtick opens the console. `help` lists every command, \
                         `help <name>` describes one, and Tab completes command \
                         names and their arguments.",
                );
                ui.label(format!(
                    "Scenes: {}. `scene` lists them and marks the active one; \
                     `scene <slug>` switches. The same slugs boot one directly: \
                     `--scene=<slug>` natively, `?scene=<slug>` in the browser. \
                     `--embed=1` / `?embed=1` hides the menu bar, which leaves \
                     the console as the only switcher.",
                    scene_roster()
                ));
            });
        });
    }

    pub(crate) fn render_overlay(&mut self, ctx: &egui::Context) {
        let screen = ctx.content_rect();
        const OVERLAY_MAX_WIDTH: f32 = 768.0;
        const OVERLAY_MIN_WIDTH: f32 = 280.0;
        let natural_w = screen.width() - 2.0 * OVERLAY_PAD;
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
                    self.render_expanded_body(ui);
                    ui.separator();
                }
                self.render_slider_strip(ui, area_w);
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

    pub(crate) fn render_slider_strip(&mut self, ui: &mut egui::Ui, _area_w: f32) {
        const VALUE_CELL_W: f32 = 72.0;
        let avail = ui.available_width();
        let spacing = ui.spacing().item_spacing.x;
        let slider_w = (avail - VALUE_CELL_W - spacing).max(140.0);
        ui.spacing_mut().slider_width = slider_w;

        let row_size = egui::vec2(avail, CONTROL_H);
        let row_layout = egui::Layout::left_to_right(egui::Align::Center);
        let w_range = self.effective_w_range();
        ui.allocate_ui_with_layout(row_size, row_layout, |ui| {
            let formatted = format!("w {:>+.3}", self.w_slice);
            slider_with_edit(
                ui,
                &mut self.w_slice,
                -w_range..=w_range,
                &formatted,
                "",
                3,
                VALUE_CELL_W,
            );
        });
        let t_max = self.t_slider_max;
        let mut t_dragged = false;
        ui.allocate_ui_with_layout(row_size, row_layout, |ui| {
            let formatted = format!("t {:>5.2}s", self.rot_time);
            // `dragged`, not `changed`: the spin's own `rot_time` advance would re-fire this.
            let interaction = slider_with_edit(
                ui,
                &mut self.rot_time,
                0.0..=t_max,
                &formatted,
                "s",
                2,
                VALUE_CELL_W,
            );
            t_dragged = interaction.dragged;
        });
        if t_dragged {
            self.recompose_spins_at(self.rot_time);
            self.rebuild_bodies();
        }
    }

    pub(crate) fn render_rate_row(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            const PLAY_GROUP_W: f32 = 215.0;
            let total_w = ui.available_width();
            let leading = ((total_w - PLAY_GROUP_W) / 2.0).max(8.0);

            ui.add_space(leading);
            let ctrl_size = egui::vec2(CONTROL_W, CONTROL_H);
            let play_size = egui::vec2(PLAY_PAUSE_W, CONTROL_H);
            rate_toggle(ui, ctrl_size, &mut self.rate_scale, 0.25, true, false);
            rate_toggle(ui, ctrl_size, &mut self.rate_scale, 0.5, false, false);
            if play_pause_button(ui, play_size, self.rotate)
                .on_hover_text("Toggle continuous rotation (Space)")
                .clicked()
            {
                self.rotate = !self.rotate;
            }
            rate_toggle(ui, ctrl_size, &mut self.rate_scale, 2.0, false, true);
            rate_toggle(ui, ctrl_size, &mut self.rate_scale, 4.0, true, true);
            if refresh_button(ui, ctrl_size)
                .on_hover_text("Reset slice, rate, active set, orientation, time")
                .clicked()
            {
                self.reset();
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
                let util_size = egui::vec2(CONTROL_W, CONTROL_H);
                if ui
                    .add(egui::Button::new(egui::RichText::new("⚙").strong()).min_size(util_size))
                    .on_hover_text("Render settings")
                    .clicked()
                {
                    self.show_render_panel = !self.show_render_panel;
                }
                if ui
                    .add(egui::Button::new(egui::RichText::new("?").strong()).min_size(util_size))
                    .on_hover_text("About this program")
                    .clicked()
                {
                    self.show_help = true;
                }
            });
        });
    }
}

#[cfg(test)]
mod about_panel_tests {
    use super::*;
    #[test]
    fn the_about_roster_names_every_registered_scene() {
        let roster = scene_roster();
        for entry in crate::shell::Playground::SCENES {
            assert!(
                roster.contains(entry.slug),
                "roster '{roster}' omits slug '{}'",
                entry.slug
            );
            assert!(
                roster.contains(entry.label),
                "roster '{roster}' omits label '{}'",
                entry.label
            );
        }
    }
}
