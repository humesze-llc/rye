pub(crate) const MAX_ROW_LEN: usize = 8;

// Fits "120-cell" in bold unwrapped; wrapping staggers card heights.
pub(crate) const SHAPE_CARD_WIDTH: f32 = 64.0;

pub(crate) const CONTROL_H: f32 = 29.0;

pub(crate) const CONTROL_W: f32 = 28.0;

pub(crate) const PLAY_PAUSE_W: f32 = 36.0;

pub(crate) const MINI_BUTTON_W: f32 = 22.0;

pub(crate) const CARD_ITEM_SPACING_X: f32 = 4.0;

pub(crate) const W_SCRUB_RATE: f32 = 0.5;
pub(crate) const W_RANGE: f32 = 1.5;

// Zero would demand exact f32 equality and hide the wireframe.
pub(crate) const HYPERSLICE_MIN_THICKNESS: f32 = 1e-4;

pub(crate) const HYPERSLICE_DEFAULT_THICKNESS: f32 = 0.2;

pub(crate) const T_SCRUB_RATE: f32 = 1.0;

// `2 × W_RANGE`, so per-pixel scrub precision matches the w slider.
pub(crate) const T_SLIDER_INITIAL: f32 = 3.0;

pub(crate) const BASE_ROTATION_RATE: f32 = std::f32::consts::TAU * 0.3;

// Over `2 × BODY_SIZE`, so rotated bodies stay clear of their neighbours.
pub(crate) const BODY_X_SPACING: f32 = 1.8;

pub(crate) const BODY_SIZE: f32 = 0.7;

pub(crate) const BODY_Y: f32 = 0.9;

pub(crate) const SPACE_TESSELLATION_SAMPLES: usize = 16;
