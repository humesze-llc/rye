//! 4D shape catalog: the single source of truth for shape names,
//! colors, and tooltips.

use anyhow::{anyhow, Result};
use loam_app::args::Args;
use loam_app::egui;
use loam_physics::euclidean_r4::MAX_POLYTOPE4_VERTICES;
use loam_render::raymarch::RaymarchShape;
use loam_shape::polytope::Polytope4;

/// One polytope's metadata. `body_color` drives `BodyUniform.color`
/// on the GPU, not the (uniformly grey) card color. `long_name` uses
/// the `pentachoron`/`tesseract`/`hexadecachoron` family, not the
/// dimension-generalized `*-plex` aliases.
#[derive(Copy, Clone, PartialEq, Debug)]
pub(crate) struct ShapeEntry {
    pub(crate) shape: RaymarchShape,
    pub(crate) body_color: [f32; 3],
    pub(crate) label: &'static str,
    pub(crate) long_name: &'static str,
}

impl ShapeEntry {
    /// The polychoron this entry collides as its own hull, or `None` to keep
    /// the bounding ball.
    ///
    /// Two independent reasons for `None`, and the budget is the interesting
    /// one: a vertex list longer than [`MAX_POLYTOPE4_VERTICES`] is truncated
    /// by the narrowphase in release, which is a corrupt hull rather than a
    /// coarse one, so the 120-cell (600 vertices) and 600-cell (120) have to
    /// stay spheres. The four smooth solids simply have no vertex list.
    pub(crate) fn collider_polytope(&self) -> Option<Polytope4> {
        self.shape
            .polytope4()
            .filter(|p| p.vertex_count() <= MAX_POLYTOPE4_VERTICES)
    }
}

pub(crate) const DEFAULT_ROW: &[ShapeEntry] = &[
    ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Cell24),
        body_color: [0.95, 0.45, 0.85],
        label: "24-cell",
        long_name: "icositetrachoron",
    },
    ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Pentatope),
        body_color: [0.95, 0.55, 0.30],
        label: "5-cell",
        long_name: "pentachoron",
    },
    ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Cell16),
        body_color: [0.55, 0.95, 0.40],
        label: "16-cell",
        long_name: "hexadecachoron",
    },
    ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Tesseract),
        body_color: [0.30, 0.55, 0.95],
        label: "8-cell",
        long_name: "tesseract",
    },
];

/// `body_color` channels pass straight to the WGSL kernel.
pub(crate) const SHAPE_CATALOG: &[ShapeEntry] = &[
    ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Pentatope),
        body_color: [0.95, 0.55, 0.30],
        label: "5-cell",
        long_name: "pentachoron",
    },
    ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Tesseract),
        body_color: [0.30, 0.55, 0.95],
        label: "8-cell",
        long_name: "tesseract",
    },
    ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Cell16),
        body_color: [0.55, 0.95, 0.40],
        label: "16-cell",
        long_name: "hexadecachoron",
    },
    ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Cell24),
        body_color: [0.95, 0.45, 0.85],
        label: "24-cell",
        long_name: "icositetrachoron",
    },
    ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Cell120),
        body_color: [0.40, 0.85, 0.85],
        label: "120-cell",
        long_name: "hecatonicosachoron",
    },
    ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Cell600),
        body_color: [0.95, 0.85, 0.40],
        label: "600-cell",
        long_name: "hexacosichoron",
    },
    ShapeEntry {
        shape: RaymarchShape::ThreeSphere,
        body_color: [0.85, 0.40, 0.40],
        label: "3-sphere",
        long_name: "hypersphere (4-ball)",
    },
    ShapeEntry {
        shape: RaymarchShape::Duocylinder,
        body_color: [0.60, 0.45, 0.90],
        label: "duocyl",
        long_name: "duocylinder (D² × D²)",
    },
    ShapeEntry {
        shape: RaymarchShape::CliffordTorus,
        body_color: [0.70, 0.85, 0.35],
        label: "clifford",
        long_name: "Clifford torus tube",
    },
    ShapeEntry {
        shape: RaymarchShape::Spherinder,
        body_color: [0.85, 0.55, 0.75],
        label: "spherinder",
        long_name: "spherinder (B³ × interval)",
    },
];

pub(crate) fn render_shape_catalog_menu(ui: &mut egui::Ui, mut on_select: impl FnMut(ShapeEntry)) {
    for cat in SHAPE_CATEGORIES {
        ui.menu_button(cat.name, |ui| {
            for entry in &SHAPE_CATALOG[cat.start..cat.end] {
                if ui
                    .button(entry.label)
                    .on_hover_text(entry.long_name)
                    .clicked()
                {
                    on_select(*entry);
                    ui.close_kind(egui::UiKind::Menu);
                }
            }
        });
    }
}

/// Half-open index ranges into [`SHAPE_CATALOG`] that group menu
/// entries under a header. Ranges (not nested slices) keep flat
/// `SHAPE_CATALOG[i]` lookups working.
struct ShapeCategory {
    name: &'static str,
    start: usize,
    end: usize,
}

const SHAPE_CATEGORIES: &[ShapeCategory] = &[
    ShapeCategory {
        name: "Regular polychora",
        start: 0,
        end: 6,
    },
    ShapeCategory {
        name: "Smooth solids",
        start: 6,
        end: 10,
    },
];

pub(crate) fn parse_shape_name(name: &str) -> Result<ShapeEntry> {
    let n = name.to_lowercase();
    let needle: &str = n.as_str();
    for entry in SHAPE_CATALOG {
        if needle == entry.label.to_lowercase() || needle == entry.long_name.to_lowercase() {
            return Ok(*entry);
        }
    }
    Ok(match needle {
        "5cell" | "pentatope" | "tetrahedron" => SHAPE_CATALOG[0],
        "8cell" | "hypercube" | "cube" => SHAPE_CATALOG[1],
        "16cell" | "octahedron" => SHAPE_CATALOG[2],
        "24cell" | "cuboctahedron" => SHAPE_CATALOG[3],
        "120cell" | "dodecahedron" => SHAPE_CATALOG[4],
        "600cell" | "icosahedron" => SHAPE_CATALOG[5],
        "hypersphere" | "3sphere" | "s3" | "4-ball" => SHAPE_CATALOG[6],
        "duocylinder" => SHAPE_CATALOG[7],
        "clifford" | "clifford-torus" | "torus" => SHAPE_CATALOG[8],
        "spherinder" => SHAPE_CATALOG[9],
        _ => {
            return Err(anyhow!(
                "unknown shape name {name:?}; valid: 5-cell, 8-cell, \
                 16-cell, 24-cell, 120-cell, 600-cell, 3-sphere, \
                 duocyl, clifford, spherinder (plus Platonic aliases: \
                 tetrahedron, cube, octahedron, cuboctahedron, \
                 dodecahedron, icosahedron)"
            ));
        }
    })
}

/// Parse the comma-separated `shapes` key (`--shapes=a,b` natively,
/// `?shapes=a,b` in the browser). Returns [`DEFAULT_ROW`] when the key
/// is absent, and an error for the space-separated `--shapes a,b` form,
/// whose value never reaches `args`.
pub(crate) fn parse_row(args: &Args) -> Result<Vec<ShapeEntry>> {
    if args.has_bare_flag("shapes") {
        return Err(anyhow!(
            "`--shapes` needs its value attached with `=`, as in \
             `--shapes=24-cell,8-cell` (comma-separated for several \
             shapes); the space-separated form drops the value"
        ));
    }
    let Some(raw) = args.get("shapes") else {
        return Ok(DEFAULT_ROW.to_vec());
    };
    let names = args.get_many("shapes");
    if names.is_empty() {
        return Err(anyhow!("shapes={raw:?} listed no shape names"));
    }
    names.into_iter().map(parse_shape_name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_catalog_entry_gets_a_hull_or_a_ball_and_a_matching_inertia() {
        let with_hull: Vec<&str> = SHAPE_CATALOG
            .iter()
            .filter(|e| e.collider_polytope().is_some())
            .map(|e| e.label)
            .collect();
        assert_eq!(with_hull, ["5-cell", "8-cell", "16-cell", "24-cell"]);

        for entry in SHAPE_CATALOG {
            match entry.collider_polytope() {
                Some(shape) => {
                    assert!(shape.vertex_count() <= MAX_POLYTOPE4_VERTICES, "{shape:?}");
                    assert!(
                        loam_physics::euclidean_r4::regular_polytope4_inertia(shape, 1.0, 1.0)
                            .is_some(),
                        "{} collides its hull with no exact inertia to go with it",
                        entry.label
                    );
                }
                None => {
                    let over_budget = entry
                        .shape
                        .polytope4()
                        .is_some_and(|p| p.vertex_count() > MAX_POLYTOPE4_VERTICES);
                    assert!(
                        over_budget || entry.shape.polytope4().is_none(),
                        "{} has a hull that fits but was left on the ball",
                        entry.label
                    );
                }
            }
        }
    }

    #[test]
    fn row_comes_from_the_args_value_not_the_process_environment() {
        assert_eq!(parse_row(&Args::default()).unwrap(), DEFAULT_ROW);
        assert_eq!(
            parse_row(&Args::from_pairs([("seed", "42")])).unwrap(),
            DEFAULT_ROW
        );

        let row = parse_row(&Args::from_pairs([("shapes", "120-cell,tesseract")])).unwrap();
        assert_eq!(
            row.iter().map(|e| e.label).collect::<Vec<_>>(),
            ["120-cell", "8-cell"]
        );
    }

    #[test]
    fn a_bare_shapes_flag_is_diagnosed_rather_than_silently_defaulting() {
        let args = Args::from_argv(["--shapes", "120-cell,tesseract"]);
        assert_eq!(args.get("shapes"), None);
        let err = parse_row(&args).unwrap_err().to_string();
        assert!(err.contains("--shapes="), "{err}");

        assert!(parse_row(&Args::from_argv(["--seed=42", "--shapes"])).is_err());
    }

    #[test]
    fn the_attached_form_and_unrelated_arguments_are_not_diagnosed() {
        let row = parse_row(&Args::from_argv(["--seed=42", "--shapes=5-cell,8-cell"])).unwrap();
        assert_eq!(
            row.iter().map(|e| e.label).collect::<Vec<_>>(),
            ["5-cell", "8-cell"]
        );
        assert_eq!(
            parse_row(&Args::from_argv(["shapes", "--x=--shapes"])).unwrap(),
            DEFAULT_ROW
        );
    }

    #[test]
    fn a_present_but_nameless_or_unknown_shapes_value_is_an_error() {
        assert!(parse_row(&Args::from_pairs([("shapes", "")])).is_err());
        assert!(parse_row(&Args::from_pairs([("shapes", ",,")])).is_err());
        assert!(parse_row(&Args::from_pairs([("shapes", "5-cell,nonesuch")])).is_err());
    }
}
