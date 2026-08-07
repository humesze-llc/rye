//! Glyph letter pipeline: font outline to a slab-embedded 4D solid.
//!
//! One pass per letter: `ab_glyph` outline curves are flattened to closed
//! contours, the contours are baked into a 2D signed distance field, the field
//! is cut into convex cross-section pieces, and each piece is extruded along
//! `z` and embedded in a `w` slab. The result serves both roles a letter needs:
//! a [`loam_shape::TriangleMesh`] through [`loam_shape::Visualizable`], and a
//! set of convex [`loam_shape::Shape::ConvexPolytope4D`] colliders.
//!
//! Unlike [`crate::TextRenderer`]'s HUD path, which skips characters it cannot
//! draw so a per-frame overlay never fails, this pipeline is a build-time step
//! and rejects anything it cannot represent: a character the font has no glyph
//! for, a control character, or an outline with no area all return
//! [`GlyphError`].
//!
//! # Example
//!
//! ```no_run
//! use ab_glyph::FontRef;
//! use loam_shape::Visualizable;
//! use loam_text::glyph::{layout_word, GlyphParams};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let font_bytes: &[u8] = &[];
//! let font = FontRef::try_from_slice(font_bytes)?;
//! let letters = layout_word(&font, "LOAM", &GlyphParams::default())?;
//! for letter in &letters {
//!     let mesh = Visualizable::<3>::to_triangles(letter);
//!     let colliders = letter.colliders_4d();
//! }
//! # Ok(()) }
//! ```

mod field;
mod outline;
mod solid;

pub use field::DistanceField2D;
pub use solid::GlyphSolid;

use ab_glyph::{Font, FontRef, GlyphId};
use glam::Vec2;

/// Distance reported by a blank glyph, which has no surface to be near. Finite
/// so callers can still combine it with `min`/`max` without producing NaN.
pub const BLANK_DISTANCE: f32 = 1.0e9;

/// Below this many cells per em the baked field cannot resolve a stem, and the
/// cross-section comes out as disconnected specks rather than a letter.
pub const MIN_RESOLUTION: u32 = 4;

/// Why a character could not be turned into a solid.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GlyphError {
    /// The font has no glyph for this character. `Debug` formatting of a
    /// `char` prints non-ASCII as its `\u{...}` escape, so the codepoint is in
    /// the message.
    #[error("font has no glyph for {ch:?}")]
    NoGlyph { ch: char },

    /// Control characters have no geometry. Line breaking and tabbing are the
    /// caller's job; this pipeline lays out one run of printable text.
    #[error("{ch:?} is a control character; the glyph pipeline takes printable text only")]
    ControlCharacter { ch: char },

    /// The font defines an outline for this character but it encloses no area.
    #[error("outline for {ch:?} encloses no area")]
    DegenerateOutline { ch: char },

    /// Without `units_per_em` there is no scale to normalize outlines by.
    #[error("font declares no units_per_em")]
    NoUnitsPerEm,

    /// A [`GlyphParams`] field that must be positive was not.
    #[error("GlyphParams::{field} must be positive, got {value}")]
    NonPositive { field: &'static str, value: f32 },

    /// [`GlyphParams::resolution`] below [`MIN_RESOLUTION`].
    #[error("GlyphParams::resolution must be at least {MIN_RESOLUTION}, got {resolution}")]
    Resolution { resolution: u32 },
}

/// How a word is turned into solids. All lengths are world units except
/// [`Self::flatten_tolerance_em`], which is relative to the em so it is
/// independent of [`Self::em_size`].
#[derive(Clone, Debug, PartialEq)]
pub struct GlyphParams {
    /// World size of one em. Advances, contours and geometry all scale by it.
    pub em_size: f32,
    /// Extent along `z`, centred on `z = 0`.
    pub depth: f32,
    /// `w` extent of the slab the letter occupies, `(min, max)`.
    pub slab: (f32, f32),
    /// Grid cells per em. Fixed per word rather than per glyph so every letter
    /// is sampled at the same fidelity. Bake cost and collider count both
    /// scale with its square.
    pub resolution: u32,
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
        if self.resolution < MIN_RESOLUTION {
            return Err(GlyphError::Resolution {
                resolution: self.resolution,
            });
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

        solids.push(GlyphSolid::new(
            ch,
            pen_origin,
            advance,
            params.depth,
            params.slab,
            params.color,
            field,
        ));
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

    /// Parameter validation is not font-dependent, so these run everywhere.
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

        let coarse = GlyphParams {
            resolution: MIN_RESOLUTION - 1,
            ..GlyphParams::default()
        };
        assert_eq!(
            coarse.validate().unwrap_err(),
            GlyphError::Resolution {
                resolution: MIN_RESOLUTION - 1
            }
        );

        assert!(GlyphParams::default().validate().is_ok());
    }

    /// A flatten tolerance that is not positive fails loudly. Left unchecked
    /// it reaches `outline::subdivisions`, whose guard against a division by
    /// zero returns a single subdivision, silently replacing every Bezier in
    /// the word with its chord instead of erroring as this module promises.
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

    /// Non-ASCII coverage the font lacks is an error, not a dropped letter.
    #[test]
    fn characters_the_font_lacks_are_rejected() {
        let Some(bytes) = system_font() else { return };
        let font = FontRef::try_from_slice(&bytes).expect("parse font");
        // U+1F600 GRINNING FACE: outside any of the probed text faces.
        let error = layout_word(&font, "LO\u{1F600}AM", &params()).unwrap_err();
        assert_eq!(
            error,
            GlyphError::NoGlyph { ch: '\u{1F600}' },
            "expected a loud failure, got {error}"
        );
    }

    /// Control characters are rejected before any font lookup, so tabs and
    /// newlines cannot silently vanish from a laid-out word.
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

    /// A word yields one solid per character, in order, with pen origins that
    /// advance monotonically by each glyph's own advance.
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

    /// Each letter's geometry sits at its own pen position rather than all
    /// letters stacking at the origin, and every letter is solid.
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

    /// Every letter of a word is baked on the same grid pitch, so a short
    /// letter is not sampled coarser than a tall one and the colliders they
    /// emit are the same scale.
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

    /// 'O' keeps its counter: a horizontal line through the middle of the
    /// letter crosses the surface four times (outside, stroke, counter,
    /// stroke, outside), where a filled 'O' would cross twice. This is the
    /// end-to-end check that the nonzero winding rule survived flattening and
    /// baking, and it does not depend on which face the probe found.
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

    /// The same letter is both render geometry and a collider source, and the
    /// two agree: every collider vertex is on or inside the surface the mesh
    /// bounds.
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
            assert_eq!(colliders.len(), letter.piece_count());
            assert!(!colliders.is_empty());
            let cell = letter.field().expect("field").cell_size();
            for collider in &colliders {
                let Shape::ConvexPolytope4D { vertices } = collider else {
                    panic!("collider is not 4D convex: {:?}", collider.kind());
                };
                assert!(vertices.len() >= 12, "prism needs 3 ring vertices minimum");
                for v in vertices {
                    assert!(
                        letter.distance_4d(*v) <= cell,
                        "{:?} collider vertex {v} lies outside the letter",
                        letter.ch()
                    );
                }
            }
        }
    }

    /// A space carries an advance and no geometry, so word spacing works
    /// without the pipeline pretending whitespace is a solid.
    #[test]
    fn spaces_advance_without_geometry() {
        let Some(bytes) = system_font() else { return };
        let font = FontRef::try_from_slice(&bytes).expect("parse font");
        let letters = layout_word(&font, "A B", &params()).expect("layout");

        assert_eq!(letters.len(), 3);
        assert!(letters[1].is_blank());
        assert!(letters[1].advance() > 0.0);
        assert!(letters[1].colliders_4d().is_empty());
        assert!(!letters[0].is_blank() && !letters[2].is_blank());
        assert!(letters[2].pen_origin().x > letters[0].pen_origin().x + letters[0].advance());
    }

    /// Baking is deterministic: the same font, word and params give the same
    /// geometry bit for bit.
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
