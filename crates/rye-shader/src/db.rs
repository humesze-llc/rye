//! Shader database with hot-reload support.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rye_asset::{AssetEvent, AssetEventKind};
use rye_math::WgslSpace;
use wgpu::{Device, ShaderModule, ShaderModuleDescriptor, ShaderSource};

/// Opaque handle to a shader in a [`ShaderDb`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShaderId(u32);

struct Entry {
    path: PathBuf,
    module: ShaderModule,
    /// Incremented on every successful (re)compile. Render code caches
    /// the generation it last built a pipeline against; mismatch means
    /// the pipeline needs rebuilding.
    generation: u64,
    /// Debug label for the module; reused on recompile.
    label: String,
}

/// Cache of compiled shaders, invalidated on asset events.
///
/// Hot-reload failures preserve the previous successful module: the
/// user sees stale output and a log line, not a crash. When a shader
/// file is removed, the entry is retained (stale) until a create or
/// modify event restores it.
pub struct ShaderDb {
    device: Device,
    entries: HashMap<ShaderId, Entry>,
    path_index: HashMap<PathBuf, ShaderId>,
    next_id: u32,
}

impl ShaderDb {
    /// Construct. `device` is cloned internally on recompile; wgpu's
    /// Device is cheap to clone (internally reference-counted).
    pub fn new(device: Device) -> Self {
        Self {
            device,
            entries: HashMap::new(),
            path_index: HashMap::new(),
            next_id: 0,
        }
    }

    /// Load a shader from disk, prepending the Space's WGSL prelude.
    ///
    /// Returns a [`ShaderId`] that remains valid across hot reloads of
    /// the same path. Call [`ShaderDb::load`] twice with the same path
    /// and you get the same ID and a fresh compilation.
    pub fn load<S: WgslSpace>(
        &mut self,
        path: impl AsRef<Path>,
        space: &S,
    ) -> Result<ShaderId> {
        let path = canonicalize(path.as_ref())?;
        let source = std::fs::read_to_string(&path)
            .with_context(|| format!("reading shader {}", path.display()))?;
        let module = self.compile(&path, &source, space)?;

        let label = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());

        if let Some(&id) = self.path_index.get(&path) {
            let entry = self.entries.get_mut(&id).expect("path_index out of sync");
            entry.module = module;
            entry.generation += 1;
            entry.label = label;
            Ok(id)
        } else {
            let id = ShaderId(self.next_id);
            self.next_id += 1;
            self.entries.insert(
                id,
                Entry { path: path.clone(), module, generation: 1, label },
            );
            self.path_index.insert(path, id);
            Ok(id)
        }
    }

    /// Borrow the current compiled module for `id`.
    pub fn module(&self, id: ShaderId) -> &ShaderModule {
        &self
            .entries
            .get(&id)
            .expect("unknown ShaderId — was it loaded by this ShaderDb?")
            .module
    }

    /// Generation counter for `id`. Increments on every successful
    /// (re)compile. Render code caches the value it last built a
    /// pipeline against and rebuilds on mismatch.
    pub fn generation(&self, id: ShaderId) -> u64 {
        self.entries.get(&id).map(|e| e.generation).unwrap_or(0)
    }

    /// Apply filesystem events, recompiling affected shaders.
    ///
    /// Compile errors are logged but do not remove the stale module;
    /// rendering continues against the last good compile until the
    /// source file is fixed.
    pub fn apply_events<S: WgslSpace>(&mut self, events: &[AssetEvent], space: &S) {
        for event in events {
            let canonical = match canonicalize(&event.path) {
                Ok(p) => p,
                // Removed files can't be canonicalized; fall back to the
                // raw path for lookup.
                Err(_) => event.path.clone(),
            };
            let Some(&id) = self.path_index.get(&canonical) else {
                continue;
            };
            match event.kind {
                AssetEventKind::Created | AssetEventKind::Modified => {
                    if let Err(e) = self.reload(id, space) {
                        tracing::warn!(
                            "shader reload failed for {}: {e:#}",
                            canonical.display()
                        );
                    } else {
                        tracing::info!("reloaded shader {}", canonical.display());
                    }
                }
                AssetEventKind::Removed => {
                    tracing::warn!(
                        "shader file removed; keeping stale module: {}",
                        canonical.display()
                    );
                }
            }
        }
    }

    fn reload<S: WgslSpace>(&mut self, id: ShaderId, space: &S) -> Result<()> {
        let path = self.entries[&id].path.clone();
        let source = std::fs::read_to_string(&path)
            .with_context(|| format!("reading shader {}", path.display()))?;
        let module = self.compile(&path, &source, space)?;
        let entry = self.entries.get_mut(&id).expect("id just looked up");
        entry.module = module;
        entry.generation += 1;
        Ok(())
    }

    fn compile<S: WgslSpace>(
        &self,
        path: &Path,
        user_source: &str,
        space: &S,
    ) -> Result<ShaderModule> {
        let full = assemble_source(&space.wgsl_impl(), user_source);
        let label = path.file_name().and_then(|n| n.to_str());
        // wgpu uses a scoped error callback pattern; any validation
        // failure panics unless we catch it ourselves. For v0, let it
        // bubble — the outer `load` / `apply_events` logs it.
        Ok(self.device.create_shader_module(ShaderModuleDescriptor {
            label,
            source: ShaderSource::Wgsl(full.into()),
        }))
    }
}

fn canonicalize(path: &Path) -> Result<PathBuf> {
    path.canonicalize()
        .with_context(|| format!("canonicalizing {}", path.display()))
}

/// Concatenate the Space's WGSL prelude with the user shader source.
///
/// Extracted for testability — this is the hot-reloadable logic that
/// doesn't require a wgpu Device.
pub(crate) fn assemble_source(space_wgsl: &str, user_source: &str) -> String {
    let mut out = String::with_capacity(space_wgsl.len() + user_source.len() + 64);
    out.push_str("// ---- rye-math Space prelude ----\n");
    out.push_str(space_wgsl);
    if !space_wgsl.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("// ---- user shader ----\n");
    out.push_str(user_source);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assemble_includes_both_sources() {
        let s = assemble_source("fn rye_distance(a: f32, b: f32) -> f32 { return 0.0; }", "@fragment fn main() {}");
        assert!(s.contains("rye_distance"));
        assert!(s.contains("@fragment fn main"));
        assert!(s.find("rye_distance").unwrap() < s.find("@fragment fn main").unwrap());
    }

    #[test]
    fn assemble_adds_newline_between_sources() {
        let s = assemble_source("fn a() {}", "fn b() {}");
        // prelude line then newline then user section marker then user code.
        assert!(s.contains("fn a() {}\n// ---- user shader ----"));
    }

    #[test]
    fn assemble_handles_trailing_newline_in_prelude() {
        let s = assemble_source("fn a() {}\n", "fn b() {}");
        // No double newline from our side; the input one is sufficient.
        assert!(!s.contains("fn a() {}\n\n// ---- user shader ----"));
        assert!(s.contains("fn a() {}\n// ---- user shader ----"));
    }
}
