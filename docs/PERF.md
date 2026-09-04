# Performance

Every number here carries the date and machine it was measured on. An
estimate does not go in this file.

## How to measure

The runtime traces every frame by section. To read it:

1. `cargo run --release -p polytope_playground`
2. Reach the scene you want and let it settle.
3. Open the console (backtick) and run `trace summary`. It prints p50, p95,
   p99, and max per section, plus an `unscoped` row: the frame minus every
   named section.

Read `unscoped` first. When it is large, the frame's time is somewhere no
section names, and the other rows are noise by comparison.

Record the GPU, CPU, OS, and backend with the numbers.

## Results

Captured 2026-07-27 on a 13th Gen Intel Core i9-13980HX, Windows 11 Pro
10.0.26200, Vulkan. Scene: every SDF object including the smooth solids, xy
rotation, 120 frames. The 120-cell and 600-cell are not in this capture.

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

Reading it:

- The SDF march is 2.4% of the median frame. The GPU as a whole is 20%.
  The UI is 9%, four times the march.
- The CPU sections the frame loop opens sum to 0.56ms of a 3.96ms median
  frame. The rest is the swapchain acquire waiting for the display: the
  surface runs Fifo with the fps cap off, and frame intervals cluster on
  multiples of the 4.13ms period of a 240Hz panel. The p95 tail is frames
  that waited an extra flip or two rather than frames that did more work.
- The physics layer was idle. Nothing in this scene moves a body, so
  `sim-ticks` measures the at-rest early return. I have not measured
  physics cost yet; that needs a capture of a moving scene.
- Whether the skipped flips come from the compositor or from the OS
  descheduling the render thread is open. `vsync off` and a second capture
  separate the two.

A change that pushes a steady-state scene past its budget is a regression
even when every test passes. Capture `trace summary` before and after any
change to a hot path.
