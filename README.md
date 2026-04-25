# Rye

Rye is my personal, open-source Rust game engine, for fun. The name comes from Riemann, as in Riemann manifolds. And I like whiskey.

## What I'm trying to build

The long-term goal is an engine where **non-Euclidean geometry is a primitive**, not a camera trick. I want 4D, hyperbolic, and fractal spaces to feel as native as ℝ³ does in every other engine.

Two concrete milestones I'm working toward:

1. **A fractal renderer** - a shader-driven ray-march tool I can actually sit down and use.
2. **A Super Hexagon/Infinite Pizza-style game** - you play a sphere falling through a deterministic zoom of a hyperbolic fractal. The sphere warps according to the local metric as the space around it changes.

Further out: an interactive development toolkit that grows with the engine.

## Why not just use Bevy?

I use Bevy. I like it. This isn't a "better at everything" play, and Rye is not trying to be a general-purpose engine. There are more Rust game engines than there are games using said Engines. But for what I want to build, a few of Bevy's design choices get in the way, and I'm using Rye as a place to try them differently:

- **Transform is Euclidean.** Bevy's `Transform` and camera math assume ℝ³. A hyperbolic metric as a first-class citizen is invasive to retrofit. In Rye I want a `Space` trait at the math layer, with Euclidean as just one implementation among several.
- **Determinism vs arbitrary scheduling.** Bevy's parallel scheduler makes lockstep / replay / rollback netcode hard because task execution order is non-deterministic across runs. Rye's simulation runs on a fixed timestep with a deterministic task graph: tasks that don't share state run in parallel (via Rayon), but outputs are reduced in a fixed order each tick. IO (asset loading, networking) is async via Tokio and feeds into per-tick input queues at tick boundaries, never touching sim state directly. Same-binary same-architecture replays produce identical sim state.
- **Shader hot reload.** For shader-heavy work like ray marching, I want to edit a WGSL file and see the result without restarting. I'm baking this in from day one instead of reaching for it later.
- **First-party editor.** Bevy's editor has been on the horizon for a while. I'd rather grow a small `egui`-based panel system alongside the engine than wait.
- **Stability.** Breaking changes every release are expensive. Rye commits to semver discipline.

These are tradeoffs to fit a very specific thesis.

## Status

Phases 1–3 done; Phase 4 in progress. Landed: persistent contact
manifolds + PGS solver, 4D physics (`Bivector4` / `Rotor4` +
`EuclideanR4` + 4D GJK/EPA), unified `Shape` enum, Space-generic
`Camera<S>` + `CameraController<S>`, and the thin `rye-app`
framework that hides the winit boilerplate. Still ahead:
`BlendedSpace`, egui dev-tools integration, full migration of
existing examples to `rye-app`. Nothing is sacred yet.

### Examples shipped

Each is `cargo run --release --example <name>` from the workspace
root. They double as demos and as integration tests for the relevant
subsystems.

**Curved-space rendering (Phase 1–2):**

- **`fractal`** — interactive 3D fractal ray-marcher with shader hot
  reload and orbit camera. The first end-to-end demo of the
  rendering stack. Includes APNG / GIF capture (`--capture-apng`,
  `--capture-gif`) for recording clips without an external screen
  recorder.
- **`fractal_app`** — same Mandelbulb scene, rebuilt on the
  `rye-app` framework + `Camera<S>` + `OrbitController<S>`. About
  half the LOC of the legacy version (the winit
  `ApplicationHandler`, hot-reload plumbing, FPS bookkeeping, and
  surface-error recovery all live in `rye-app` now). Proof point
  for the Phase 4 framework refactor.
- **`geodesic_spheres`** — three spheres rendered side-by-side in
  E³, H³, and S³ via the `WgslSpace` prelude swap, proving the same
  scene takes on the metric of whichever Space you select.
- **`lattice`** — the same `Scene` rendered in all three Spaces with
  no per-Space WGSL written by the caller (validates the typed
  scene tree).
- **`corridor`** — extruded corridor scene; stress-tests the typed
  primitive emit path under combinators (boolean ops, repeats).

**Euclidean physics (Phase 3):**

- **`physics2d`** — 2D rigid bodies under gravity with polygon SAT
  collision, friction, and the manifold + PGS solver. The
  reference for "does the solver actually settle stacks?"
- **`physics3d`** — mixed spheres and oriented boxes falling onto a
  floor, exercising the full GJK + EPA + manifold + PGS path in 3D.

**4D physics + visualization (Phase 4, this branch):**

- **`physics4d`** — headless 4D demo with three modes:
  `--floor` drops a pentatope onto a 4D `y ≥ 0` half-space;
  `--collide` drops one onto a static second pentatope, exercising
  4D GJK + EPA end-to-end. Default mode is pure free-fall to verify
  the integrator + 4D orientation transport in isolation. Prints
  per-tick state to stdout.
- **`pentatope_slice`** — live-physics 4D viewer. A pentatope falls
  onto a 4D floor; the 3D viewport renders the cross-section of the
  whole 4D scene at user-controlled `w₀`. Hold ↑/↓ to scrub the
  slice plane along `w` and watch the cross-section morph through
  the pentatope's five tetrahedral cells. Pause and reset live.
- **`hypersphere`** — drop a 4D ball (or up to 32 with `-n N`) onto
  the 4D floor and watch them collide and pile up under gravity.
  Two viewing modes: **slice** (default; renders the 3D cross-
  section at `w₀`, growing-and-shrinking as you scrub `w`) and
  **ghost** (toggle with **G**; renders each body's full 4D extent
  as translucent volume, opacity proportional to the body's
  `w`-thickness through each xyz column).

**Tooling:**

- **`capture`** — frame-capture utility (APNG / GIF) for the demos
  above; used to record clips without external screen recorders.
- **`sysinfo`** — wgpu adapter + backend probe, useful when a demo
  refuses to start on a particular machine.

## Hardware

wgpu across Vulkan / DX12 / Metal / WebGPU. No CUDA lock-in; if ML features show up, they go through Burn's wgpu backend first.

## License

Rye is dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option. Both licenses require attribution; please retain the
copyright notice and license text in substantial portions of the work.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual-licensed as above, without any additional terms or
conditions.
