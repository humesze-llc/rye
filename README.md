# Rye

Rye is my personal, open-source Rust game engine, for fun. The name comes from Riemann, as in Riemann spaces. And I like whiskey.

## What I'm trying to build

The long-term goal is an engine where **non-Euclidean geometry is a primitive**, not a camera trick. I want 4D, hyperbolic, and fractal spaces to feel as native as ℝ³ does in every other engine.

Two concrete milestones I'm working toward:

1. **A fractal renderer** - a shader-driven ray-march tool I can actually sit down and use.
2. **A Super Hexagon/Infinite Pizza-style game** - you play a low-poly sphere falling through a deterministic zoom of a hyperbolic fractal. The sphere warps according to the local metric as the space around it changes.

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

Very early. Nothing is sacred.

## Hardware

wgpu across Vulkan / DX12 / Metal / WebGPU. No CUDA lock-in; if ML features show up, they go through Burn's wgpu backend first.

## License

MIT OR Apache-2.0.
