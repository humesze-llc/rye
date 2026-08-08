# Performance

Frame budgets are backed by measurement, not estimate. The engine already records per-section frame timing;
this page is how to turn that into a table, and where the current numbers live.

## How to measure

The runtime keeps a rolling-window trace of every frame, scoped by section
(`frame`, `sim-ticks`, `app-update`, `app-ui`, `hot-reload`,
`surface-acquire`, `app-record`, `ui-paint`, `composite`, `present`). To read
it:

1. `cargo run --release -p polytope_playground`
2. Exercise the scene you want to characterize (let it reach steady state).
3. Open the console (backtick) and run `trace summary` for the aggregate
   (p50 / p95 / p99 / max per section, p95-descending), or `trace` for the
   last frame.

`trace summary` also prints a synthetic `unscoped` row: `frame` minus the
sections the frame loop opens inside it. Read that row first. A large
`unscoped` means the frame's time is somewhere no section names, and every
other row in the table is then a rounding error by comparison.

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

Four observations:

**The SDF is not the bottleneck in this scene.** `pp-sdf` is 94us of a 3.96ms
median frame (2.4%). `gpu-total` is 803us (20%); `app-ui` plus `ui-paint` is
371us (9.4%), four times the SDF cost. The heavy polychora this capture
excludes are the one case where the march could still dominate; measure them
before optimizing it.

**The frame is mostly a vsync wait, and so is the tail.** Three facts, each
from this table or from the code it measures.

*The table accounts for almost none of the frame.* The sections the frame
loop opens sum to 561.7us at p50 (`sim-ticks` 0.2, `app-update` 15.9,
`hot-reload` 0.7, `app-record` 131.6, `app-ui` 224.1, `ui-paint` 146.7,
`present` 42.5) against a `frame` p50 of 3960us, so 86% of the median frame
is time no section covers. The tail is worse, and the bound holds without any
assumption about which frames coincide: those same sections sum to 2046.4us
at their session *maxima*, so even in the impossible frame where all of them
peak at once, `frame` p95 (13.18ms) leaves at least 11.1ms unaccounted and
`frame` max (16.58ms) at least 14.5ms. `idle` is not where it went either: at
2.18ms, `idle` p95 could account for at most 17% of `frame` p95 even if every
slow idle landed on a slow frame.

*The uncovered region contains exactly one call that waits on the display.*
It is the `get_current_texture` inside `RenderDevice::begin_frame`. The
surface is configured `PresentMode::Fifo` with
`desired_maximum_frame_latency: 2`, and the fps cap is off by default, so the
presentation engine holds that acquire until a swapchain image frees at the
next flip. `present` is cheap (42.5us) because it queues the flip and returns.
The rest of the uncovered region is encoder creation, three queue submits and
the GPU-timer bookkeeping, none of which wait on the display; the new
`unscoped` row exists so the next capture can bound them instead of assuming.

*The cadence is quantized, which is the vsync signature.* `between-frames`
max/p50 is 4.04 and p95/p50 is 3.23: frame intervals cluster on integer
multiples of the p50 period, to within 1% at the max. That is what a
vsync-locked producer looks like and it is not implied by the accounting
above. A p50 of 4.13ms puts the period at 242Hz, consistent with a 240Hz
panel. The tail frames are ones that waited two or three extra flips.

Three candidates are ruled out. A periodic egui relayout
cannot be it: `app-ui` and `ui-paint` peak at 644.4us and 802.9us, under a
third of one 4.17ms interval combined. The GPU cannot be it: `gpu-total` peaks
at 2.01ms. The fps cap cannot be it: it is off by default, and its sleep runs
before the frame's `begin_frame`, so it would land in `idle`, whose p50 is
110us.

**The tail is not the engine's to fix, pending confirmation.** In no
captured frame did our CPU work or our GPU work approach one refresh interval,
so the pipeline never lost the flip deadline by being slow; the flips were
skipped on the presentation side. What remains open is whether the skips come
from the desktop compositor or from the OS descheduling the single-threaded
loop across a flip deadline. Two runs separate them, and both need a windowed
session:

1. `trace summary` on this build. `surface-acquire` now has its own row. If
   its p95 is ~11ms and `unscoped` is small, the location is confirmed.
2. `vsync off` (Mailbox) and re-capture. If the tail collapses, it belongs to
   the presentation engine. If it survives an uncapped present mode, it is
   scheduler preemption of the render thread.

Until then, treat the per-section numbers as measurements of a loop that
spends 86% of its frame blocked on the display, and do not read a 94us
section as a bottleneck.

**`sim-ticks` at 200ns means the physics layer was idle during this capture.**
Nothing in the scene applies an impulse, so `World::step` returns at the
at-rest check every frame. Physics cost is unmeasured until a scene moves
bodies.

A change that pushes a steady-state scene past its frame budget is a
regression even if every test passes; capture a `trace summary` before and
after any change to the hot path.
