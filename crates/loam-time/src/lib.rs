//! `loam-time`: deterministic fixed-timestep scheduling for Loam.
//!
//! ## The deterministic-sim split
//!
//! The simulation is a pure, bit-reproducible function
//! `tick(state, input, tick_number) -> state'`; time enters only as the
//! tick number, never wall-clock. This crate is the
//! wall-clock-to-tick-count adapter the local render loop uses. Replays
//! and rollback netcode drive the sim by tick number directly and never
//! touch [`FixedTimestep`]; [`replay`] is the recorded form of that
//! tick-indexed input stream.
//!
//! ## The presentation clock
//!
//! [`director`] is the other side of the determinism boundary: authored
//! keyframes sampled at an integer frame index, for playback and capture
//! rather than simulation. It is held to same-binary-same-frames, not to the
//! sim's same-bits, and must never write simulation state.

pub mod alloc;
pub mod director;
mod fixed_timestep;
pub mod frame_trace;
pub mod jobs;
pub mod replay;

pub use director::{Director, Drive, Playhead, Timeline, TimelineError, Track};
pub use fixed_timestep::{FixedTimestep, DEFAULT_MAX_CATCH_UP};
pub use replay::{Checkpoint, StateHash, Tape, TapeError, TAPE_FORMAT_VERSION};
