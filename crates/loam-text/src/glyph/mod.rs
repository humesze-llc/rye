//! Glyph letter pipeline: font outline to a slab-embedded 4D solid.
//!
//! Unlike [`crate::TextRenderer`]'s HUD path, which skips characters it cannot
//! draw so a per-frame overlay never fails, this pipeline is a build-time step
//! and rejects anything it cannot represent: a character the font has no glyph
//! for, a control character, or an outline with no area all return
//! [`GlyphError`].

mod field;
mod hull;
mod outline;
mod solid;

pub use field::DistanceField2D;
pub use solid::GlyphSolid;

use ab_glyph::{Font, FontRef, GlyphId};
use glam::Vec2;

/// Distance reported by a blank glyph, which has no surface to be near. Finite
/// so callers can still combine it with `min`/`max` without producing NaN.
pub const BLANK_DISTANCE: f32 = 1.0e9;

/// Floor for both grid pitches. Below this many cells per em the render bake
/// cannot resolve a stem and the cross-section comes out as disconnected specks
/// rather than a letter.
pub const MIN_RESOLUTION: u32 = 4;

/// Why a character could not be turned into a solid.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GlyphError {
    #[error("font has no glyph for {ch:?}")]
    NoGlyph { ch: char },

    #[error("{ch:?} is a control character; the glyph pipeline takes printable text only")]
    ControlCharacter { ch: char },

    #[error("outline for {ch:?} encloses no area")]
    DegenerateOutline { ch: char },

    #[error("font declares no units_per_em")]
    NoUnitsPerEm,

    #[error("GlyphParams::{field} must be positive, got {value}")]
    NonPositive { field: &'static str, value: f32 },

    #[error("GlyphParams::{field} must be at least {MIN_RESOLUTION}, got {resolution}")]
    Resolution {
        field: &'static str,
        resolution: u32,
    },
}

/// How a word is turned into solids. All lengths are world units except
/// [`Self::flatten_tolerance_em`], which is relative to the em so it is
/// independent of [`Self::em_size`].
#[derive(Clone, Debug, PartialEq)]
pub struct GlyphParams {
    /// World size of one em.
    pub em_size: f32,
    /// Extent along `z`, centred on `z = 0`.
    pub depth: f32,
    /// `w` extent of the slab the letter occupies, `(min, max)`.
    pub slab: (f32, f32),
    /// Grid cells per em for the bake and the render decomposition.
    pub resolution: u32,
    /// Grid cells per em for the collider cover, independent of
    /// [`Self::resolution`]: legibility sets that one, the solver's body budget
    /// sets this one.
    pub collider_resolution: u32,
    /// Maximum chord deviation when flattening Bezier segments, in em.
    pub flatten_tolerance_em: f32,
    /// Per-vertex colour of the emitted meshes, RGBA linear.
    pub color: [f32; 4],
}

impl Default for GlyphParams {
    fn default() -> Self {
        Self {
            em_size: 1.0,
            depth: 0.15,
            slab: (-0.075, 0.075),
            // 48 cells across an em resolves the thinnest stems of a text-weight
            // face while keeping the bake under a few million distance tests.
            resolution: 48,
            // The same pitch for the cover, which is the conservative end of
            // the useful range: LOAM comes out at 96 boxes and a 0.042 em
            // margin, against 73 boxes and 0.063 em at 32 and 39 boxes and
            // 0.125 em, about a stem width, at 16. See
            // `examples/glyph_collider_budget.rs`.
            collider_resolution: 48,
            flatten_tolerance_em: 0.002,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

impl GlyphParams {
    fn validate(&self) -> Result<(), GlyphError> {
        for (field, value) in [
            ("em_size", self.em_size),
            ("depth", self.depth),
            ("slab", self.slab.1 - self.slab.0),
            ("flatten_tolerance_em", self.flatten_tolerance_em),
        ] {
            if value <= 0.0 || value.is_nan() {
                return Err(GlyphError::NonPositive { field, value });
            }
        }
        for (field, resolution) in [
            ("resolution", self.resolution),
            ("collider_resolution", self.collider_resolution),
        ] {
            if resolution < MIN_RESOLUTION {
                return Err(GlyphError::Resolution { field, resolution });
            }
        }
        Ok(())
    }
}

/// Lay out `text` and bake one [`GlyphSolid`] per character, advancing the pen
/// by each glyph's horizontal advance plus the pair's kerning.
///
/// Geometry is emitted in a shared word frame: baseline at `y = 0`, the first
/// character's pen origin at `x = 0`. Characters the font defines with no
/// outline (whitespace) yield a blank solid that carries only its advance.
pub fn layout_word(
    font: &FontRef<'_>,
    text: &str,
    params: &GlyphParams,
) -> Result<Vec<GlyphSolid>, GlyphError> {
    params.validate()?;
    let units_per_em = font.units_per_em().ok_or(GlyphError::NoUnitsPerEm)?;
    let units_to_world = params.em_size / units_per_em;
    let tolerance_world = params.flatten_tolerance_em * params.em_size;
    let cell = params.em_size / params.resolution as f32;

    let mut solids = Vec::with_capacity(text.chars().count());
    let mut pen_x = 0.0_f32;
    let mut previous: Option<GlyphId> = None;

    for ch in text.chars() {
        if ch.is_control() {
            return Err(GlyphError::ControlCharacter { ch });
        }
        let id = font.glyph_id(ch);
        // Glyph 0 is `.notdef` by the OpenType specification, so a lookup
        // landing there means the font does not cover this character.
        if id.0 == 0 {
            return Err(GlyphError::NoGlyph { ch });
        }

        if let Some(previous) = previous {
            pen_x += font.kern_unscaled(previous, id) * units_to_world;
        }
        let advance = font.h_advance_unscaled(id) * units_to_world;
        let pen_origin = Vec2::new(pen_x, 0.0);

        let field = match font.outline(id) {
            None => None,
            Some(raw) => {
                let mut contours =
                    outline::contours_from_curves(&raw.curves, units_to_world, tolerance_world);
                for contour in &mut contours {
                    for point in &mut contour.points {
                        *point += pen_origin;
                    }
                }
                Some(
                    field::DistanceField2D::bake(&contours, cell)
                        .ok_or(GlyphError::DegenerateOutline { ch })?,
                )
            }
        };

        solids.push(GlyphSolid::new(ch, pen_origin, advance, params, field));
        pen_x += advance;
        previous = Some(id);
    }

    Ok(solids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_shape::{Shape, Visualizable};

    /// Fonts are not vendored; probe the usual system locations and skip
    /// cleanly when none is present, matching `examples/text_smoke`.
    fn system_font() -> Option<Vec<u8>> {
        const CANDIDATES: &[&str] = &[
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ];
        CANDIDATES
            .iter()
            .find_map(|path| std::fs::read(path).ok())
            .or_else(|| {
                eprintln!("skip: no system font found in {CANDIDATES:?}");
                None
            })
    }

    fn params() -> GlyphParams {
        GlyphParams {
            resolution: 32,
            ..GlyphParams::default()
        }
    }

    #[test]
    fn nonpositive_and_undersized_params_are_rejected() {
        let bad_depth = GlyphParams {
            depth: 0.0,
            ..GlyphParams::default()
        };
        assert_eq!(
            bad_depth.validate().unwrap_err(),
            GlyphError::NonPositive {
                field: "depth",
                value: 0.0
            }
        );

        let inverted_slab = GlyphParams {
            slab: (1.0, -1.0),
            ..GlyphParams::default()
        };
        assert!(matches!(
            inverted_slab.validate().unwrap_err(),
            GlyphError::NonPositive { field: "slab", .. }
        ));

        let coarse_render = GlyphParams {
            resolution: MIN_RESOLUTION - 1,
            ..GlyphParams::default()
        };
        assert_eq!(
            coarse_render.validate().unwrap_err(),
            GlyphError::Resolution {
                field: "resolution",
                resolution: MIN_RESOLUTION - 1
            }
        );

        let coarse_collider = GlyphParams {
            collider_resolution: MIN_RESOLUTION - 1,
            ..GlyphParams::default()
        };
        assert_eq!(
            coarse_collider.validate().unwrap_err(),
            GlyphError::Resolution {
                field: "collider_resolution",
                resolution: MIN_RESOLUTION - 1
            }
        );

        assert!(GlyphParams::default().validate().is_ok());
    }

    #[test]
    fn nonpositive_or_nan_flatten_tolerance_is_rejected() {
        for value in [0.0, -0.002, f32::NAN] {
            let params = GlyphParams {
                flatten_tolerance_em: value,
                ..GlyphParams::default()
            };
            let error = params.validate().unwrap_err();
            assert!(
                matches!(
                    error,
                    GlyphError::NonPositive {
                        field: "flatten_tolerance_em",
                        ..
                    }
                ),
                "tolerance {value} gave {error}"
            );
        }
    }

    #[test]
    fn characters_the_font_lacks_are_rejected() {
        let Some(bytes) = system_font() else { return };
        let font = FontRef::try_from_slice(&bytes).expect("parse font");
        let error = layout_word(&font, "LO\u{1F600}AM", &params()).unwrap_err();
        assert_eq!(
            error,
            GlyphError::NoGlyph { ch: '\u{1F600}' },
            "expected a loud failure, got {error}"
        );
    }

    #[test]
    fn control_characters_are_rejected() {
        let Some(bytes) = system_font() else { return };
        let font = FontRef::try_from_slice(&bytes).expect("parse font");
        for ch in ['\t', '\n', '\u{7F}'] {
            assert_eq!(
                layout_word(&font, &format!("A{ch}B"), &params()).unwrap_err(),
                GlyphError::ControlCharacter { ch }
            );
        }
    }

    #[test]
    fn word_yields_one_solid_per_character_at_advancing_pen_origins() {
        let Some(bytes) = system_font() else { return };
        let font = FontRef::try_from_slice(&bytes).expect("parse font");
        let letters = layout_word(&font, "LOAM", &params()).expect("layout");

        assert_eq!(letters.len(), 4);
        assert_eq!(
            letters.iter().map(|l| l.ch()).collect::<Vec<_>>(),
            vec!['L', 'O', 'A', 'M']
        );
        assert_eq!(letters[0].pen_origin().x, 0.0);
        for pair in letters.windows(2) {
            assert!(
                pair[1].pen_origin().x > pair[0].pen_origin().x,
                "pen must advance: {:?} then {:?}",
                pair[0].pen_origin(),
                pair[1].pen_origin()
            );
            assert!(pair[0].advance() > 0.0);
        }
    }

    #[test]
    fn letters_are_placed_at_their_own_pen_positions() {
        let Some(bytes) = system_font() else { return };
        let font = FontRef::try_from_slice(&bytes).expect("parse font");
        let letters = layout_word(&font, "LOAM", &params()).expect("layout");

        let mut previous_centroid_x = f32::NEG_INFINITY;
        for letter in &letters {
            let mesh = Visualizable::<3>::to_triangles(letter).expect("mesh");
            assert!(!mesh.vertices.is_empty());
            let centroid_x: f32 =
                mesh.vertices.iter().map(|v| v[0]).sum::<f32>() / mesh.vertices.len() as f32;
            assert!(
                centroid_x > previous_centroid_x,
                "{:?} centroid {centroid_x} did not advance past {previous_centroid_x}",
                letter.ch()
            );
            previous_centroid_x = centroid_x;

            // Ink stays inside the letter's own advance box. The margin covers
            // side bearings a face may make negative, plus the padding cells
            // the bake adds around the outline.
            let cell = letter.field().expect("field").cell_size();
            let left = letter.pen_origin().x - 0.2 * letter.advance() - 2.0 * cell;
            let right = letter.pen_origin().x + 1.2 * letter.advance() + 2.0 * cell;
            for v in &mesh.vertices {
                assert!(
                    v[0] > left && v[0] < right,
                    "{:?} strays to {}",
                    letter.ch(),
                    v[0]
                );
            }
        }
    }

    #[test]
    fn all_letters_share_one_grid_pitch() {
        let Some(bytes) = system_font() else { return };
        let font = FontRef::try_from_slice(&bytes).expect("parse font");
        let params = params();
        let letters = layout_word(&font, "Loam.", &params).expect("layout");

        let expected = params.em_size / params.resolution as f32;
        for letter in &letters {
            assert_eq!(letter.field().expect("field").cell_size(), expected);
        }
    }

    #[test]
    fn counters_stay_open() {
        let Some(bytes) = system_font() else { return };
        let font = FontRef::try_from_slice(&bytes).expect("parse font");
        let o = &layout_word(&font, "O", &params()).expect("layout")[0];

        let mesh = Visualizable::<3>::to_triangles(o).expect("mesh");
        let min_y = mesh.vertices.iter().fold(f32::INFINITY, |m, v| m.min(v[1]));
        let max_y = mesh
            .vertices
            .iter()
            .fold(f32::NEG_INFINITY, |m, v| m.max(v[1]));
        let mid_y = 0.5 * (min_y + max_y);

        const PROBES: u32 = 512;
        let mut crossings = 0;
        let mut previous = None;
        for k in 0..=PROBES {
            let x = o.pen_origin().x + (k as f32 / PROBES as f32 - 0.05) * o.advance() * 1.1;
            let inside = o.distance_2d(Vec2::new(x, mid_y)) < 0.0;
            if previous.is_some_and(|was| was != inside) {
                crossings += 1;
            }
            previous = Some(inside);
        }
        assert_eq!(crossings, 4, "'O' crossed {crossings} times, not 4");
    }

    #[test]
    fn letters_serve_as_render_geometry_and_colliders() {
        let Some(bytes) = system_font() else { return };
        let font = FontRef::try_from_slice(&bytes).expect("parse font");
        let letters = layout_word(&font, "LOAM", &params()).expect("layout");

        for letter in &letters {
            let mesh = Visualizable::<3>::to_triangles(letter).expect("mesh");
            assert!(mesh.indices.len() >= 2);
            assert_eq!(mesh.colors.len(), mesh.vertices.len());

            let colliders = letter.colliders_4d();
            assert_eq!(colliders.len(), letter.collider_count());
            assert!(!colliders.is_empty());
            // The extrusion spans the depth and the slab exactly, so only the
            // cross-section can overshoot and the 4D bound is the 2D one.
            let margin = letter.collider_margin();
            for (centre, collider) in &colliders {
                let Shape::ConvexPolytope4D { vertices } = collider else {
                    panic!("collider is not 4D convex: {:?}", collider.kind());
                };
                assert_eq!(vertices.len(), 16);
                for v in vertices {
                    let world = *centre + *v;
                    assert!(
                        letter.distance_4d(world) <= margin,
                        "{:?} collider vertex {world} lies {} outside the letter",
                        letter.ch(),
                        letter.distance_4d(world)
                    );
                }
            }
        }
    }

    #[test]
    fn a_laid_out_word_stays_inside_the_solver_body_budget() {
        let Some(bytes) = system_font() else { return };
        let font = FontRef::try_from_slice(&bytes).expect("parse font");
        let letters = layout_word(&font, "LOAM", &GlyphParams::default()).expect("layout");

        let colliders: usize = letters.iter().map(GlyphSolid::collider_count).sum();
        let render_pieces: usize = letters.iter().map(GlyphSolid::piece_count).sum();
        assert!(colliders > 0);
        assert!(
            colliders <= 108,
            "LOAM emits {colliders} colliders, past the measured budget"
        );
        assert!(
            colliders * 20 < render_pieces,
            "{colliders} colliders against {render_pieces} render pieces is not a cut"
        );
    }

    #[test]
    fn the_cover_encloses_every_letter_without_filling_its_counters() {
        let Some(bytes) = system_font() else { return };
        let font = FontRef::try_from_slice(&bytes).expect("parse font");
        let letters = layout_word(&font, "LOAM", &params()).expect("layout");

        for letter in &letters {
            let cover = letter.collider_cover().expect("cover");
            assert!(!cover.clipped(), "{:?} ran off its domain", letter.ch());
            let margin = letter.collider_margin();

            // Sweep the letter's advance box and a full em of height, offset
            // off both grids so probes never land on a cell centre.
            const PROBES: usize = 161;
            let x0 = letter.pen_origin().x - 0.25 * letter.advance();
            let width = 1.5 * letter.advance();
            let (mut ink, mut clear) = (0, 0);
            for j in 0..PROBES {
                for i in 0..PROBES {
                    let p = Vec2::new(
                        x0 + width * (i as f32 + 0.317) / PROBES as f32,
                        -0.25 + 1.25 * (j as f32 + 0.211) / PROBES as f32,
                    );
                    let d = letter.distance_2d(p);
                    if d <= 0.0 {
                        ink += 1;
                        assert!(
                            cover.contains(p.to_array()),
                            "{:?} leaves ink at {p} uncovered",
                            letter.ch()
                        );
                    } else if d > margin {
                        clear += 1;
                        assert!(
                            !cover.contains(p.to_array()),
                            "{:?} covers {p}, which is {d} clear of the ink",
                            letter.ch()
                        );
                    }
                }
            }
            assert!(ink > 1_000, "{:?}: only {ink} ink probes", letter.ch());
            assert!(
                clear > 1_000,
                "{:?}: only {clear} clear probes",
                letter.ch()
            );
        }
    }

    #[test]
    fn the_collider_pitch_moves_the_box_count_and_not_the_render_mesh() {
        let Some(bytes) = system_font() else { return };
        let font = FontRef::try_from_slice(&bytes).expect("parse font");
        let fine = layout_word(&font, "LOAM", &params()).expect("layout");
        let coarse_params = GlyphParams {
            collider_resolution: 12,
            ..params()
        };
        let coarse = layout_word(&font, "LOAM", &coarse_params).expect("layout");

        let count = |ls: &[GlyphSolid]| ls.iter().map(GlyphSolid::collider_count).sum::<usize>();
        assert!(
            count(&coarse) < count(&fine),
            "coarse {} is not below fine {}",
            count(&coarse),
            count(&fine)
        );
        for (a, b) in fine.iter().zip(&coarse) {
            let (ma, mb) = (
                Visualizable::<3>::to_triangles(a).expect("mesh"),
                Visualizable::<3>::to_triangles(b).expect("mesh"),
            );
            assert_eq!(ma.vertices, mb.vertices);
            assert_eq!(ma.indices, mb.indices);
        }
    }

    #[test]
    fn all_letters_share_one_collider_pitch() {
        let Some(bytes) = system_font() else { return };
        let font = FontRef::try_from_slice(&bytes).expect("parse font");
        let params = params();
        let letters = layout_word(&font, "Loam.", &params).expect("layout");

        let expected = params.em_size / params.collider_resolution as f32;
        for letter in &letters {
            let cell = letter.collider_cover().expect("cover").cell_size();
            assert!(
                (cell - expected).abs() <= 4.0 * f32::EPSILON * expected,
                "{:?} covers at {cell}, not {expected}",
                letter.ch()
            );
        }
    }

    #[test]
    fn spaces_advance_without_geometry() {
        let Some(bytes) = system_font() else { return };
        let font = FontRef::try_from_slice(&bytes).expect("parse font");
        let letters = layout_word(&font, "A B", &params()).expect("layout");

        assert_eq!(letters.len(), 3);
        assert!(letters[1].is_blank());
        assert!(letters[1].advance() > 0.0);
        assert!(letters[1].colliders_4d().is_empty());
        assert!(letters[1].collider_cover().is_none());
        assert!(!letters[0].is_blank() && !letters[2].is_blank());
        assert!(letters[2].pen_origin().x > letters[0].pen_origin().x + letters[0].advance());
    }

    #[test]
    fn layout_is_bit_reproducible() {
        let Some(bytes) = system_font() else { return };
        let font = FontRef::try_from_slice(&bytes).expect("parse font");
        let first = layout_word(&font, "LOAM", &params()).expect("layout");
        let second = layout_word(&font, "LOAM", &params()).expect("layout");

        for (a, b) in first.iter().zip(&second) {
            let (ma, mb) = (
                Visualizable::<3>::to_triangles(a).expect("mesh"),
                Visualizable::<3>::to_triangles(b).expect("mesh"),
            );
            assert_eq!(ma.vertices, mb.vertices);
            assert_eq!(ma.indices, mb.indices);
        }
    }
}
