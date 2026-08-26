//! `loam-asset`: filesystem watching and hot-reload events for Loam.
//!
//! The watcher is deliberately minimal: no asset loading, no type-specific handling,
//! no ID indirection. It just tells you which paths changed. Consumers decide what
//! "changed" means for them.

mod watcher;

pub use watcher::{AssetEvent, AssetEventKind, AssetWatcher};
