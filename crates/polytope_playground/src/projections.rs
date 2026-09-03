use loam_shape::polytope::Polytope4;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub(crate) enum WireframeProjection {
    #[default]
    Shadow,
    /// Pinhole at `(0, 0, 0, focal)` onto `w = 0`, foreshortening by `focal / (focal - w)`.
    WPinhole,
    /// Central projection from just outside one cell (Coxeter, *Regular Polytopes*, ch. 13).
    #[allow(dead_code)]
    Schlegel {
        cell_index: u32,
    },
    Stereographic,
    Hyperslice,
}

// Canonical coordinates: `Demo::resolved_wireframe_projection` scales and rotates per frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SchlegelParams {
    pub(crate) polytope: Polytope4,
    pub(crate) cell_index: u32,
    pub(crate) cell_normal: glam::Vec4,
    pub(crate) cell_basis: [glam::Vec4; 3],
    // The cell lies in `{x : dot(cell_normal, x) = cell_offset}`.
    pub(crate) cell_offset: f32,
    pub(crate) viewpoint_distance: f32,
}

const SCHLEGEL_BASIS_EPSILON: f32 = 1e-6;

fn push_schlegel_basis_vector(
    candidate: glam::Vec4,
    cell_normal: glam::Vec4,
    basis: &mut [glam::Vec4; 3],
    count: &mut usize,
) {
    if *count == basis.len() {
        return;
    }
    let mut v = candidate - candidate.dot(cell_normal) * cell_normal;
    for b in basis.iter().take(*count) {
        v -= v.dot(*b) * *b;
    }
    let len = v.length();
    if len > SCHLEGEL_BASIS_EPSILON {
        basis[*count] = v / len;
        *count += 1;
    }
}

fn resolve_schlegel_cell_basis(
    polytope: Polytope4,
    cell_index: usize,
    cell_normal: glam::Vec4,
) -> [glam::Vec4; 3] {
    let topo = polytope.topology();
    let cell = topo.cells[cell_index];
    let anchor = topo.vertices[cell[0] as usize];
    let mut basis = [glam::Vec4::ZERO; 3];
    let mut count = 0usize;

    for &vi in cell.iter().skip(1) {
        push_schlegel_basis_vector(
            topo.vertices[vi as usize] - anchor,
            cell_normal,
            &mut basis,
            &mut count,
        );
        if count == 3 {
            return basis;
        }
    }

    debug_assert_eq!(count, 3, "Schlegel boundary cell must span a 3-flat");
    for seed in [glam::Vec4::X, glam::Vec4::Y, glam::Vec4::Z, glam::Vec4::W] {
        push_schlegel_basis_vector(seed, cell_normal, &mut basis, &mut count);
        if count == 3 {
            break;
        }
    }
    basis
}

// Additive, so a small-inradius polytope's eye clearance does not collapse.
const SCHLEGEL_EYE_MARGIN: f32 = 0.5;

// Coxeter, *Regular Polytopes*, ch. 13. Not the wrong dual-vertex `cell{120,600}_face_planes`.
pub(crate) fn resolve_schlegel_params(polytope: Polytope4, cell_index: u32) -> SchlegelParams {
    let cell_count = polytope.cell_count() as u32;
    let clamped = cell_index.min(cell_count - 1);
    let (normals, cell_offset) = polytope.face_planes();
    let cell_normal = normals[clamped as usize];
    let cell_basis = resolve_schlegel_cell_basis(polytope, clamped as usize, cell_normal);
    let max_dot = polytope
        .topology()
        .vertices
        .iter()
        .map(|v| cell_normal.dot(*v))
        .fold(f32::NEG_INFINITY, f32::max);
    SchlegelParams {
        polytope,
        cell_index: clamped,
        cell_normal,
        cell_basis,
        cell_offset,
        viewpoint_distance: max_dot + SCHLEGEL_EYE_MARGIN,
    }
}

pub(crate) fn synced_schlegel_projection(
    projection: WireframeProjection,
    subject: Option<Polytope4>,
) -> (WireframeProjection, Option<SchlegelParams>) {
    match (projection, subject) {
        (WireframeProjection::Schlegel { cell_index }, Some(polytope)) => {
            let params = resolve_schlegel_params(polytope, cell_index);
            let synced = WireframeProjection::Schlegel {
                cell_index: params.cell_index,
            };
            (synced, Some(params))
        }
        (other, _) => (other, None),
    }
}

// A conformal map draws an edge as an arc.
pub(crate) fn default_edge_blend(projection: WireframeProjection) -> f32 {
    match projection {
        WireframeProjection::Stereographic => 1.0,
        _ => 0.0,
    }
}

pub(crate) fn apply_projection_selection_defaults(
    projection: WireframeProjection,
    wireframe_enabled: &mut bool,
) {
    if matches!(projection, WireframeProjection::Stereographic) {
        *wireframe_enabled = true;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ModeAnnotation {
    pub(crate) title: &'static str,
    pub(crate) body: String,
}

// Coxeter, *Regular Polytopes*, ch. 13; Wikipedia, "Stereographic projection".
pub(crate) fn mode_annotation(projection: WireframeProjection) -> Option<ModeAnnotation> {
    let (title, projection_body): (&'static str, Option<&str>) = match projection {
        WireframeProjection::Shadow => ("Shadow", None),
        WireframeProjection::WPinhole => (
            "W-pinhole",
            Some(
                "4D pinhole camera with its eye down the w-axis: the +w face \
                 projects to the outer shape and the -w face to the inner one, the \
                 classic cube-within-a-cube view of the tesseract.",
            ),
        ),
        WireframeProjection::Schlegel { .. } => (
            "Schlegel diagram",
            Some(
                "Central projection from a viewpoint just outside one chosen cell \
                 onto that cell's hyperplane (Coxeter, Regular Polytopes, ch. 13): \
                 the chosen cell becomes the outer boundary and every other cell \
                 nests inside it.",
            ),
        ),
        WireframeProjection::Stereographic => (
            "Stereographic projection",
            Some(
                "Conformal S^3 -> R^3 map, (x, y, z) / (1 - w): the polytope is \
                 cast onto S^3 and its edges render as great-circle arcs. Angles \
                 are preserved but distances are not, so the cell facing the +w \
                 pole balloons outward as its vertices approach the pole.",
            ),
        ),
        WireframeProjection::Hyperslice => (
            "Hyperslice",
            Some(
                "Shows only the edges of cells the current 4D cut passes through: \
                 an edge survives when a cell containing it has its w-range within \
                 the thin slab around the w-slice, so the wireframe thins to the \
                 cells being sliced.",
            ),
        ),
    };

    let projection_lead = projection_body?;
    Some(ModeAnnotation {
        title,
        body: projection_lead.to_string(),
    })
}

// On the slice axis, so the scale is `1 / (1 - p.w)`; `+e_w` is a 16-cell vertex (Coxeter §8.2).
pub(crate) const STEREOGRAPHIC_DEFAULT_POLE: glam::Vec4 = glam::Vec4::new(0.0, 0.0, 0.0, 1.0);

impl WireframeProjection {
    pub(crate) fn from_token(token: &str) -> Option<Self> {
        match token {
            "shadow" => Some(WireframeProjection::Shadow),
            "w-pinhole" => Some(WireframeProjection::WPinhole),
            "stereographic" => Some(WireframeProjection::Stereographic),
            "hyperslice" => Some(WireframeProjection::Hyperslice),
            _ => None,
        }
    }

    // Schlegel is intentionally omitted.
    pub(crate) const ALL: [Self; 4] = [
        WireframeProjection::Shadow,
        WireframeProjection::WPinhole,
        WireframeProjection::Stereographic,
        WireframeProjection::Hyperslice,
    ];

    /// Parallel to [`Self::ALL`].
    pub(crate) const TOKENS: [&'static str; 4] =
        ["shadow", "w-pinhole", "stereographic", "hyperslice"];

    pub(crate) fn label(self) -> &'static str {
        match self {
            WireframeProjection::Shadow => "Shadow",
            WireframeProjection::WPinhole => "W-pinhole",
            WireframeProjection::Schlegel { .. } => "Schlegel",
            WireframeProjection::Stereographic => "Stereographic",
            WireframeProjection::Hyperslice => "Hyperslice",
        }
    }

    pub(crate) fn same_variant(self, other: Self) -> bool {
        std::mem::discriminant(&self) == std::mem::discriminant(&other)
    }

    // `Schlegel` falls back to `Identity`; `Demo::resolved_wireframe_projection` builds the real one.
    pub(crate) fn to_projection(self) -> loam_math::Projection<4> {
        match self {
            WireframeProjection::Shadow | WireframeProjection::Hyperslice => {
                loam_math::Projection::Identity
            }
            WireframeProjection::WPinhole => loam_math::Projection::Perspective4D {
                focal_distance: 2.0,
            },
            WireframeProjection::Schlegel { .. } => loam_math::Projection::Identity,
            WireframeProjection::Stereographic => loam_math::Projection::Stereographic {
                pole: STEREOGRAPHIC_DEFAULT_POLE,
            },
        }
    }
}

pub(crate) fn hyperslice_cull_active(toggle: bool, projection: WireframeProjection) -> bool {
    toggle || matches!(projection, WireframeProjection::Hyperslice)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_token_parses_to_the_variant_it_sits_beside() {
        for (token, projection) in WireframeProjection::TOKENS
            .iter()
            .zip(WireframeProjection::ALL)
        {
            assert_eq!(
                WireframeProjection::from_token(token),
                Some(projection),
                "`{token}` completes to a slot it does not parse into"
            );
        }
    }
}
