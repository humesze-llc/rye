use crate::*;

fn handles_arg(arg: Option<&str>, current: bool) -> anyhow::Result<bool> {
    match arg {
        None => Ok(!current),
        Some("on") => Ok(true),
        Some("off") => Ok(false),
        Some(other) => Err(anyhow!("handles: unknown arg `{other}` (try on|off)")),
    }
}

impl RotateScene {
    pub(crate) fn build_console() -> Console<Demo> {
        let mut c = Console::<Demo>::new();
        loam_app::shell::register_shell_commands::<Demo, shell::Playground>(
            &mut c,
            loam_app::build_info!(),
        );
        c.register(loam_egui::cmd(
            "reset",
            "reset slice, rate, active set, orientation and time in place",
            |_args, demo: &mut Demo, _out| {
                demo.reset();
                Ok(())
            },
        ));
        c.register(loam_egui::cmd(
            "spin",
            "toggle continuous rotation (Space / T)",
            |_args, demo: &mut Demo, _out| {
                demo.rotate = !demo.rotate;
                Ok(())
            },
        ));
        c.register(loam_egui::cmd(
            "hud",
            "toggle the top-left loam-text state readout (w, t, rate, planes)",
            |_args, demo: &mut Demo, out| {
                demo.show_text_hud = !demo.show_text_hud;
                out.line(format!(
                    "hud: {}",
                    if demo.show_text_hud { "on" } else { "off" }
                ));
                Ok(())
            },
        ));
        c.register(loam_egui::cmd(
            "controls",
            "toggle the bottom controls overlay (H)",
            |_args, demo: &mut Demo, _out| {
                demo.show_controls = !demo.show_controls;
                Ok(())
            },
        ));
        c.register(loam_egui::cmd(
            "formula",
            "toggle the top-right formula popup",
            |_args, demo: &mut Demo, _out| {
                demo.show_formula = !demo.show_formula;
                Ok(())
            },
        ));
        c.register(
            crate::verbs::wireframe_subcommands::<Demo>(|d| &mut d.wireframe)
                .choice(
                    "color",
                    "parent-edge color mode (bare cycles): vertex-gradient|unique-edge|w-depth|active",
                    &["vertex-gradient", "unique-edge", "w-depth", "active"],
                    |d, name| {
                        d.wireframe_color_mode = match name {
                            Some(n) => WireframeColorMode::from_token(n).ok_or_else(|| {
                                anyhow!(
                                    "unknown color mode `{n}` (try vertex-gradient|unique-edge|w-depth|active)"
                                )
                            })?,
                            None => {
                                let all = WireframeColorMode::ALL;
                                let i = all
                                    .iter()
                                    .position(|m| *m == d.wireframe_color_mode)
                                    .unwrap_or(0);
                                all[(i + 1) % all.len()]
                            }
                        };
                        Ok(())
                    },
                )
                .toggle(
                    "nearest-active",
                    "per-edge alpha gradient by cell-crossing strength (bare flips)",
                    |d, v| {
                        d.wireframe_nearest_active = v.unwrap_or(!d.wireframe_nearest_active);
                        Ok(())
                    },
                )
                .custom(
                    "pole",
                    "stereographic projection pole (bare reports; sub: reset | +w | <x y z w>)",
                    &[&["reset", "+w"]],
                    &[],
                    |d, args, out| {
                        match args.first().copied() {
                            None => {
                                let p = d.stereographic_pole;
                                out.line(format!(
                                    "stereographic pole: ({:.3}, {:.3}, {:.3}, {:.3})",
                                    p.x, p.y, p.z, p.w
                                ));
                            }
                            Some("reset") | Some("default") => {
                                d.stereographic_pole = state::STEREOGRAPHIC_DEFAULT_POLE;
                                out.line("stereographic pole: reset to the cell-center default");
                            }
                            Some("+w") => {
                                d.stereographic_pole = Vec4::W;
                                out.line("stereographic pole: set to +w (textbook map)");
                            }
                            Some(_) => {
                                let coords: Result<Vec<f32>> = args
                                    .iter()
                                    .map(|t| {
                                        t.parse::<f32>()
                                            .map_err(|e| anyhow!("invalid pole component `{t}`: {e}"))
                                    })
                                    .collect();
                                let coords = coords?;
                                if coords.len() != 4 {
                                    return Err(anyhow!(
                                        "pole needs 4 components `<x y z w>`, got {}",
                                        coords.len()
                                    ));
                                }
                                let raw = Vec4::new(coords[0], coords[1], coords[2], coords[3]);
                                if raw.length() < MIN_EDGE_RADIUS {
                                    return Err(anyhow!(
                                        "pole vector is too close to zero to have a direction"
                                    ));
                                }
                                let pole = raw.normalize();
                                d.stereographic_pole = pole;
                                out.line(format!(
                                    "stereographic pole: set to ({:.3}, {:.3}, {:.3}, {:.3})",
                                    pole.x, pole.y, pole.z, pole.w
                                ));
                            }
                        }
                        Ok(())
                    },
                )
                .custom(
                    "hyperslice",
                    "cull parent edges to a w-slab around the slice (bare flips; sub: on|off|thickness <N>)",
                    &[&["on", "off", "thickness"]],
                    &[],
                    |d, args, out| {
                        match args.first().copied() {
                            None => {
                                d.wireframe_hyperslice = !d.wireframe_hyperslice;
                                out.line(format!(
                                    "wireframe hyperslice: {} (slab full-width {:.3})",
                                    if d.wireframe_hyperslice { "on" } else { "off" },
                                    d.wireframe_hyperslice_thickness
                                ));
                            }
                            Some("on") => {
                                d.wireframe_hyperslice = true;
                                out.line(format!(
                                    "wireframe hyperslice: on (slab full-width {:.3})",
                                    d.wireframe_hyperslice_thickness
                                ));
                            }
                            Some("off") => {
                                d.wireframe_hyperslice = false;
                                out.line("wireframe hyperslice: off (full edge graph)");
                            }
                            Some("thickness") => match args.get(1).copied() {
                                None => out.line(format!(
                                    "wireframe hyperslice thickness: {:.3}",
                                    d.wireframe_hyperslice_thickness
                                )),
                                Some(token) => {
                                    let t: f32 = token.parse().map_err(|e| {
                                        anyhow!("invalid thickness `{token}`: {e}")
                                    })?;
                                    // Floor is the predicate's razor band; the
                                    // `2 * W_RANGE` cap covers every reachable
                                    // w, so the filter no-ops there.
                                    let max = 2.0 * consts::W_RANGE;
                                    if !(HYPERSLICE_MIN_THICKNESS..=max).contains(&t) {
                                        return Err(anyhow!(
                                            "hyperslice thickness {t} out of range; expected {HYPERSLICE_MIN_THICKNESS}..={max}"
                                        ));
                                    }
                                    d.wireframe_hyperslice_thickness = t;
                                    out.line(format!(
                                        "wireframe hyperslice thickness: set to {t:.3}"
                                    ));
                                }
                            },
                            Some(other) => {
                                return Err(anyhow!(
                                    "unknown hyperslice subcommand `{other}` (try on|off|thickness)"
                                ));
                            }
                        }
                        Ok(())
                    },
                )
                .custom(
                    "points",
                    "vertex + cell-center sprite overlay (bare flips; sub: vertices|cell-centers|size <N>)",
                    &[&["vertices", "cell-centers", "size"]],
                    &[],
                    |d, args, out| {
                        match args.first().copied() {
                            None => {
                                d.points_enabled = !d.points_enabled;
                                out.line(format!(
                                    "points: {}",
                                    if d.points_enabled { "on" } else { "off" }
                                ));
                            }
                            Some("vertices") => {
                                d.points_show_vertices = !d.points_show_vertices;
                                out.line(format!(
                                    "points vertices: {}",
                                    if d.points_show_vertices { "on" } else { "off" }
                                ));
                            }
                            Some("cell-centers") => {
                                d.points_show_cell_centers = !d.points_show_cell_centers;
                                out.line(format!(
                                    "points cell-centers: {}",
                                    if d.points_show_cell_centers { "on" } else { "off" }
                                ));
                            }
                            Some("size") => match args.get(1) {
                                None => out.line(format!(
                                    "points size: {:.1} px",
                                    d.points_size_px
                                )),
                                Some(token) => {
                                    let px: f32 = token.parse().map_err(|e| {
                                        anyhow!("invalid pixel value `{token}`: {e}")
                                    })?;
                                    if !(1.0..=64.0).contains(&px) {
                                        return Err(anyhow!(
                                            "points size {px} out of range; expected 1..=64"
                                        ));
                                    }
                                    d.points_size_px = px;
                                    out.line(format!("points size: set to {px:.1} px"));
                                }
                            },
                            Some(other) => {
                                return Err(anyhow!(
                                    "unknown points subcommand `{other}` (try vertices|cell-centers|size)"
                                ));
                            }
                        }
                        Ok(())
                    },
                ),
        );

        c.register(
            loam_egui::cmd(
                "surface",
                "polychoral surface mode: raster | sdf | off (bare = off); `scale <N>` to resize (per-layer cap alpha lives under `section`)",
                |args, demo: &mut Demo, out| {
                    if matches!(args.first().copied(), Some("scale")) {
                        match args.get(1).copied() {
                            None => {
                                out.line(format!(
                                    "surface scale: {:.3} (multiplies BODY_SIZE)",
                                    demo.surface_scale
                                ));
                            }
                            Some(token) => {
                                let parsed: f32 = token.parse().map_err(|e| {
                                    anyhow!("invalid scale `{token}`: {e}")
                                })?;
                                if !(0.05..=10.0).contains(&parsed) {
                                    return Err(anyhow!(
                                        "surface scale {parsed} out of range; expected 0.05..=10.0"
                                    ));
                                }
                                demo.surface_scale = parsed;
                                // Raster paths read `effective_body_size()`
                                // each frame; only the SDF kernel needs this.
                                demo.rebuild_bodies();
                                // Clamp the w-slice into the new scaled range
                                // so a shrink does not leave the slider off it.
                                let w_range = demo.effective_w_range();
                                demo.w_slice = demo.w_slice.clamp(-w_range, w_range);
                                out.line(format!(
                                    "surface scale: set to {parsed:.3}"
                                ));
                            }
                        }
                        return Ok(());
                    }
                    let next = match args.first().copied() {
                        Some(token) => SurfaceMode::from_token(token).ok_or_else(|| {
                            anyhow!("unknown arg `{token}` (try raster|sdf|off|scale; cap alpha lives under `section`)")
                        })?,
                        None => SurfaceMode::Off,
                    };
                    if next == SurfaceMode::Sdf && demo.sdf_blocked_by_heavy_polychora() {
                        return Err(anyhow!(
                            "surface sdf disabled while 120-cell or 600-cell is in the row \
                             (the SDF kernel crashes the browser tab on those); remove the \
                             heavy polychora first, or use `surface raster`"
                        ));
                    }
                    if next != demo.surface_mode {
                        demo.surface_mode = next;
                        // Switching INTO Sdf mode makes the polychora live in
                        // the kernel, switching OUT marks them inert.
                        demo.rebuild_bodies();
                    }
                    Ok(())
                },
            )
            .with_args(&[&["raster", "sdf", "off", "scale"]])
            .with_long_help(
                "Selects how the six regular convex 4-polytopes (5-cell, tesseract, 16-cell,\n\
                 24-cell, 120-cell, 600-cell) are rendered, plus a runtime scale knob.\n\
                 \n\
                 subcommands:\n  \
                 raster      Rasterized cross-section cell-caps (the default). Face-normal\n                             Lambert lit, per-body solid color. Much faster for the\n                             120-cell + 600-cell and exact (no SDF approximation).\n  \
                 sdf         SDF raymarch. The historical pre-rasterizer path; smoother\n                             shading but the 120-cell and 600-cell carry a face-plane\n                             approximation BUG. Kept for visual comparison.\n  \
                 off         No surface rendered. Wireframe overlay + cross-section\n                             perimeter stay visible if enabled; the cap interiors are\n                             blank. Useful for inspecting the wireframe on its own.\n  \
                 scale <N>   Multiply the canonical body radius by N (default 1.0; range\n                             0.05..=10.0). Affects SDF kernel, raster cross-section caps,\n                             wireframe overlay, perimeter, and points sprites uniformly.\n\
                 \n\
                 Bare `surface` (no argument) is shorthand for `surface off`.\n\
                 \n\
                 The rasterized cross-section splits into two overlaid layers with\n\
                 independent perimeter + fill alpha: see the `section` command (the honest\n\
                 drop-w cross-section and the projection-following cap).\n\
                 \n\
                 Smooth-surface shapes (Clifford torus, duocylinder, spherinder, 3-sphere)\n\
                 ignore the mode and always render via the SDF; they have no rasterizer\n\
                 path. Surface scale still applies to their SDF body radius.",
            ),
        );

        c.register(
            loam_egui::subcommands::<Demo>(
                "section",
                "rasterized cross-section layers: cross (honest drop-w) + cap (projection-following), each with perimeter + alpha",
            )
            .toggle(
                "cross-perimeter",
                "honest drop-w cross-section perimeter outline (bare flips)",
                |d, v| {
                    d.cross_section.perimeter = v.unwrap_or(!d.cross_section.perimeter);
                    Ok(())
                },
            )
            .toggle(
                "cap-perimeter",
                "projected-cap perimeter outline (bare flips)",
                |d, v| {
                    d.projected_cap.perimeter = v.unwrap_or(!d.projected_cap.perimeter);
                    Ok(())
                },
            )
            .custom(
                "cross-alpha",
                "honest cross-section fill alpha (0 = off; range (0, 1])",
                &[&[]],
                &[],
                |d, args, out| run_section_alpha("cross", &mut d.cross_section, args, out),
            )
            .custom(
                "cap-alpha",
                "projected-cap fill alpha (0 = off; range (0, 1])",
                &[&[]],
                &[],
                |d, args, out| run_section_alpha("cap", &mut d.projected_cap, args, out),
            ),
        );

        loam_app::camera_rig::register_camera_command(&mut c, |demo| &mut demo.rig);

        c.register(
            loam_egui::cmd::<Demo, _>(
                "handles",
                "toggle the 4D transform handles (on | off; bare flips)",
                |args, demo, out| {
                    let next = handles_arg(args.first().copied(), demo.gimbal.enabled)?;
                    demo.gimbal.enabled = next;
                    out.line(format!("handles: {}", if next { "on" } else { "off" }));
                    Ok(())
                },
            )
            .with_args(&[&["on", "off"]])
            .with_long_help(
                "Six interlocked rings, one per rotation plane, from the stereographic\n\
                 projection of the 16-cell, plus four arrows, one per translation axis.\n\
                 Drag a ring to turn the whole row in its plane; drag an arrowhead to\n\
                 slide the row along that axis. The violet arrow is w: it moves the bodies\n\
                 off the 3D slice, so the cross-sections change shape rather than\n\
                 travelling.\n\
                 \n\
                 Off at startup and reachable only from here. The handles are hidden in\n\
                 Filmstrip view, which composes per-cell viewports with no shared world\n\
                 origin for the widget to stand on.",
            ),
        );

        // The SDF kernel reads `u.params[0]`; when 0.0 the wrapper around
        // `loam_scene_sdf` short-circuits to a huge distance, so the marcher
        // never converges on the floor.
        loam_app::environment::register_ground_command(&mut c, |demo| &mut demo.environment);
        loam_app::environment::register_floor_command(&mut c, |demo| &mut demo.environment);

        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_app::shell::SceneRegistry;

    #[test]
    fn the_console_registers_the_transform_handles_toggle() {
        assert!(RotateScene::build_console().has_command("handles"));
    }

    #[test]
    fn handles_arg_flips_when_bare_and_is_absolute_when_named() {
        assert!(handles_arg(None, false).unwrap());
        assert!(!handles_arg(None, true).unwrap());
        assert!(handles_arg(Some("on"), false).unwrap());
        assert!(handles_arg(Some("on"), true).unwrap());
        assert!(!handles_arg(Some("off"), true).unwrap());
        assert!(!handles_arg(Some("off"), false).unwrap());
    }

    #[test]
    fn handles_arg_rejects_an_unknown_token() {
        assert!(handles_arg(Some("yes"), false).is_err());
    }

    #[test]
    fn the_console_carries_the_live_ground_controls() {
        assert!(RotateScene::build_console().has_command("ground"));
    }

    #[test]
    fn console_exposes_the_scene_switcher() {
        assert!(RotateScene::build_console().has_command("scene"));
    }

    #[test]
    fn scene_completion_cycles_every_registered_slug() {
        let mut console = RotateScene::build_console();
        *console.input_mut() = "scene ".to_string();
        let mut completed: Vec<String> = Vec::new();
        for _ in shell::Playground::SCENES {
            console.tab_complete();
            completed.push(
                console
                    .input()
                    .strip_prefix("scene ")
                    .expect("completion fills the switcher's first argument")
                    .to_string(),
            );
        }
        completed.sort();
        let mut slugs: Vec<String> = shell::Playground::SCENES
            .iter()
            .map(|entry| entry.slug.to_string())
            .collect();
        slugs.sort();
        assert_eq!(completed, slugs);
    }

    #[test]
    fn the_help_listing_describes_the_scene_switcher() {
        let mut console = Console::<()>::new();
        loam_app::shell::register_shell_commands::<(), shell::Playground>(
            &mut console,
            loam_app::build_info!(),
        );
        console.execute("help");
        let listed = console
            .history()
            .iter()
            .find_map(|line| line.text.trim_start().strip_prefix("scene "))
            .expect("`help` lists the scene command")
            .trim()
            .to_string();
        assert!(!listed.is_empty(), "the listing carries no description");

        console.clear_history();
        console.execute("help scene");
        let long = console
            .history()
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        for param in ["--scene=", "?scene=", "--embed=1"] {
            assert!(long.contains(param), "`help scene` omits {param}");
        }
    }
}
