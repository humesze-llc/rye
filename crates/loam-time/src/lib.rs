//! Time enters the sim only as a tick number, never wall-clock. This crate
//! is the wall-clock-to-tick adapter for the render loop; a replay drives
//! the sim by tick number and never touches [`FixedTimestep`].

pub mod alloc;
pub mod director;
mod fixed_timestep;
pub mod frame_trace;
pub mod jobs;
pub mod replay;

pub use director::{Director, Drive, Playhead, Timeline, TimelineError, Track};
pub use fixed_timestep::{FixedTimestep, DEFAULT_MAX_CATCH_UP};
pub use replay::{Checkpoint, StateHash, Tape, TapeError, TAPE_FORMAT_VERSION};
