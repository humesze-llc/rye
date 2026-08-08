# Loam

**A Rust game engine on wgpu where the ambient geometry is a parameter: higher dimensions and curved manifolds are first-class primitives.** Built in the open by one person, for the author's games. Source-available and dual-licensed; not a community engine and not a supported product.

[![CI](https://github.com/humesze-llc/loam/actions/workflows/ci.yml/badge.svg)](https://github.com/humesze-llc/loam/actions/workflows/ci.yml)

<!-- HERO SLOT: assets/readme/hero.webp
     v1 capture: the playground's default polytope row under xw rotation,
     exact cross-sections shaded, ~1200x675, loop-clean (one full rotation
     period), dark background.
     Planned replacement: volumetric LOAM letters falling to the ground,
     polychora knocking them into 4D, then a scrub along w. -->

## Run it

```
cargo run --release -p polytope_playground                     # native
trunk serve --release crates/polytope_playground/index.html    # browser (local)
```

The Polytope Playground is the flagship demo and the first thing to look at. Rust 1.95, pinned in `rust-toolchain.toml`; the browser build needs [trunk](https://trunkrs.dev/), run from the repo root because Trunk reads `Trunk.toml` from the working directory. Controls live in the on-screen panels; backtick opens the debug console (`trace summary` prints per-section frame timings).

Runs natively on Vulkan and in the browser on WebGPU. CI exercises Vulkan through lavapipe and builds the wasm32 target. No other wgpu backend has been run here.

Rustdoc for every crate is published at [humesze-llc.github.io/loam](https://humesze-llc.github.io/loam/).

<!-- 600-CELL SLOT: assets/readme/600-cell.webp
     The 600-cell rotating through the exact cross-section path, wireframe
     overlay with signed-w depth colors. Same capture settings as hero. -->

## What is in the repo today

- **4D rigid-body physics, interactive.** First-party geometric algebra (`Bivector4`/`Rotor4` with invariant-decomposition exponential), GJK and EPA lifted to R⁴, persistent contact manifolds, a warm-started projected Gauss-Seidel solver. The playground's shape row is a zero-g R⁴ chamber: press on a shape to pick it with a screen ray, drag to aim, release to flick it, and it carries on until it strikes a neighbour or coasts to a stop. Throw speed is capped below the largest per-step displacement the R⁴ step was measured to still resolve against a thin wall. The console command `throw <slot> <drag_x_px> <drag_y_px>` replays the same gesture without a mouse. Pinned by `screen_ray_picks_the_nearest_body_it_enters_and_nothing_else`, `throw_speed_never_exceeds_the_measured_tunneling_bound`, and `a_full_speed_throw_transfers_momentum_to_the_neighbour_it_hits`; EPA depth against the closed form by `sphere_sphere_penetration_matches_analytical`.
- **Exact polychoral cross-sections.** All six convex regular 4-polytopes with exact topology at unit circumradius (`vertices_on_unit_circumradius`). The default section path computes geometry rather than approximating a mesh; a raymarched-SDF surface mode is kept alongside it for comparison, and a wireframe overlay draws the full edge graph. Each is a panel toggle and a console command (`section`, `surface`, `wireframe`).
- **Geometry as a type parameter.** Nine `Space` implementations behind one trait (exp, log, distance, parallel transport, isometries): Euclidean R²/R³/R⁴, hyperbolic H³, spherical S³ in two charts, a variable-curvature E³-to-H³ blend, and two quotients that carry their gluing in a deck group (a flat 3-torus and the lens space L(p, q)). `cargo test -p loam-math --test space_conformance` runs one invariant list against all of them but the 3-torus, whose own list is `cargo test -p loam-math quotient::tests`; `cargo test -p loam-shader --lib -- --include-ignored wgsl_matches_the_rust_space` pins the E³, R⁴, H³ and S³ WGSL preludes against their Rust twins on a real adapter, and `cargo test -p loam-render --test flat_torus_room --test lens_space -- --include-ignored` does the same for the two quotient preludes. Downstream support is per-space and partial: `WgslSpace` (the raymarcher) covers E³, R⁴, H³, S³, the blend, and the 3-torus, while the lens space emits a `vec4<f32>` prelude of its own because no vec3 chart of S³ holds the deck images; `RasterizableSpace` covers E³, R⁴, and the embedded S³, so E³ and R⁴ are the two where both paths share one implementation and H³ has no rasterizer path. Both quotients also render outside the tests, offscreen rather than interactively: `cargo run -p loam-render --example flat_torus_room` and `cargo run -p loam-render --example lens_space` each march eight frames along a closed loop of the quotient, tile them into one PNG under `captures/`, and fail if the walk does not close. The lens-space walk follows the shortest closed geodesic of L(5, 2) and measures its last frame against a render of the same point of the quotient reached the other way, by one deck element. Neither opens a window. One curved space is interactive: `cargo run -p polytope_playground -- --scene=s3` poses a polychoron through `SphericalS3Embedded`'s isometry group and draws its edges as the great-circle arcs that Space returns, under an isoclinic rotor whose Clifford-translation property is pinned by `the_isoclinic_rotor_displaces_every_vertex_equally_and_a_simple_one_does_not`. Everything else interactive still runs the Euclidean spaces.
- **Determinism.** Simulation is bit-reproducible: same binary, same inputs, same bits. `cargo test -p loam-physics determinism` is its own CI gate; `fixed_scenario_replay_is_bit_identical_determinism` and `fixed_scenario_trajectory_matches_golden_determinism_hash` are the pins.

## How it's built

- A geometry is anything implementing `Space`. Everything downstream is an opt-in capability trait (`WgslSpace`, `RasterizableSpace`, `SectionableSpace`, `PhysicsSpace`), so a new geometry is wired in by implementing what it actually supports.
- Scenes are typed SDF trees that emit WGSL on demand; the emit chain combines scene code with the selected space's shader prelude.
- Correctness is enforced by an invariant test suite, not by inspecting rendered output: `geodesic_triangle_angle_excess_matches_gauss_bonnet`, `distance_is_invariant_under_isometry`, `parallel_transport_preserves_the_metric_norm`, `parallel_transport_in_h3_has_nonzero_holonomy`. A primitive whose visualization looks right but whose invariants fail does not ship.
- `loam-math` depends on `glam`, `serde`, and `tracing`; the geometry is first-party. The crate-by-crate map lives in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- A variable-curvature `BlendedSpace` (a single Riemannian metric interpolating E³ to H³) is a reference implementation, held to the same conformance list as the closed-form spaces. Its `exp` integrates 32 RK4 steps and its `log` Gauss-Newton-shoots against that, where every other space is closed form; nothing runs it outside the tests. GPU parity is gated on `exp` and transport alone (`blended_e3_h3_gpu_probe_exp_matches_cpu`, `blended_e3_h3_gpu_probe_transport_matches_cpu`): its WGSL `log` and `distance` are deliberately cheaper functions than the Rust ones, so the scene probe records that divergence instead of gating on it.

## Roadmap

Nothing named below has shipped; a line leaves this section when it does. No dates: this is a single-maintainer project, and a missed public date is worse than no date. Substrate and tooling get built ahead of demand; game-specific features ship with the game that pulls them.

**Now.** A deterministic multithreaded simulation core, gated on bit-identical results between one thread and many. A rotation gizmo, so a 4D orientation is dragged rather than typed, finishing the direct manipulation the throw gesture started. A second scene in the demo registry the shell already carries. Frame-budget work on the SDF path, with before-and-after traces as the artifact.

**Next.** Baked colliders for smooth shapes, and an SDF editor, both on top of the CPU scene evaluator that already exists. First-party UI, so egui becomes an optional debug layer. Animation tooling.

**Later.** The Polytope Playground stays the flagship. Then 4D experiments (life, boids, creature behaviour) on GPU compute, which sit outside the determinism boundary and will say so. Games past that are unstarted; the one that most shapes the engine is a horror game built on multiply-connected space.

## Lineage

Marc ten Bosch's *4D Toys* is the direct conceptual ancestor of the polytope playground: four-dimensional objects as inhabitants of a 4D space whose 3D slices the viewer inspects. Zeno Rogue's *HyperRogue*, CodeParade's HyperEngine/*Hyperbolica*, and Michael Walczyk's `polychora` shaped the non-Euclidean and 4D rendering choices. Textbook citations live in the rustdocs at each use site: Coxeter for the polytope constructions, Hestenes and Sobczyk for the geometric algebra, do Carmo for the Riemannian geometry.

## Getting involved

I use AI coding tools, primarily Claude Code, heavily; invariant tests gate every primitive regardless of authorship, and responsibility for what ships is mine. Single-maintainer project: reach out before starting anything at [humesze@proton.me](mailto:humesze@proton.me).

## License

Dual-licensed under MIT OR Apache-2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
