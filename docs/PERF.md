# Performance

Frame budgets are backed by measurement, not estimate. The engine already records per-section frame timing;
this page is how to turn that into a table, and where the current numbers live.

## How to measure

The runtime keeps a rolling-window trace of every frame, scoped by section
(`frame`, `sim-ticks`, `app-update`, `app-ui`, `app-record`, `ui-paint`,
`composite`, `present`). To read it:

1. `cargo run --release -p polytope_playground`
2. Exercise the scene you want to characterize (let it reach steady state).
3. Open the console (backtick) and run `trace summary` for the aggregate
   (p50 / p95 / p99 / max per section, p95-descending), or `trace` for the
   last frame.

Numbers are hardware-specific, so record the GPU, CPU, OS, and backend
alongside them. This is a measured artifact; do not fill it with estimates.

## Results

**Captured:** 2026-07-27.
**Machine:** 13th Gen Intel Core i9-13980HX, Windows 11 Pro 10.0.26200, Vulkan.
**Scene:** all valid SDF objects including smooth solids, xy rotation, 120
frames. Excludes the 120-cell and 600-cell, which are the expensive cases and
are not represented here.

| section | p50 | p95 | max |
|---|---|---|---|
| between-frames | 4.13ms | 13.35ms | 16.69ms |
| frame | 3.96ms | 13.18ms | 16.58ms |
| idle | 110.5us | 2.18ms | 12.09ms |
| gpu-total | 802.8us | 1.59ms | 2.01ms |
| app-ui | 224.1us | 373.1us | 644.4us |
| ui-paint | 146.7us | 220.5us | 802.9us |
| app-record | 131.6us | 202.6us | 313.9us |
| pp-sdf | 94.3us | 149.3us | 232.1us |
| present | 42.5us | 61.3us | 118.1us |
| app-update | 15.9us | 42.0us | 164.7us |
| hot-reload | 700ns | 1.1us | 1.5us |
| sim-ticks | 200ns | 400ns | 900ns |

Three observations:

**The SDF is not the bottleneck in this scene.** `pp-sdf` is 94us of a 3.96ms
median frame (2.4%). `gpu-total` is 803us (20%); `app-ui` plus `ui-paint` is
371us (9.4%), four times the SDF cost. The heavy polychora this capture
excludes are the one case where the march could still dominate; measure them
before optimizing it.

**The tail is the cost, not the median.** p50 is 3.96ms while p95 is 13.18ms,
a 3.3x spread, and `idle` moves with it (110us p50, 12.09ms max). The worst 5%
of frames carry more time than every section in the table combined; attribute
that before optimizing any section.

**`sim-ticks` at 200ns means the physics layer was idle during this capture.**
Nothing in the scene applies an impulse, so `World::step` returns at the
at-rest check every frame. Physics cost is unmeasured until a scene moves
bodies.

A change that pushes a steady-state scene past its frame budget is a
regression even if every test passes; capture a `trace summary` before and
after any change to the hot path.
