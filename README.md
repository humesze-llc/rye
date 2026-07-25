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
cargo run --release -p polytope_playground               # native
cd crates/polytope_playground && trunk serve --release   # browser (local)
```

The Polytope Playground is the flagship demo and the first thing to look at. Rust 1.95, pinned in `rust-toolchain.toml`; the browser build needs [trunk](https://trunkrs.dev/). Controls live in the on-screen panels; backtick opens the debug console (`trace summary` prints per-section frame timings).

Runs natively on Vulkan and in the browser on WebGPU. CI exercises Vulkan through lavapipe and builds the wasm32 target. No other wgpu backend has been run here.

Rustdoc for every crate is published at [humesze-llc.github.io/loam](https://humesze-llc.github.io/loam/).

<!-- 600-CELL SLOT: assets/readme/600-cell.webp
     The 600-cell rotating through the exact cross-section path, wireframe
     overlay with signed-w depth colors. Same capture settings as hero. -->

## What is in the repo today

- **4D rigid-body physics.** First-party geometric algebra (`Bivector4`/`Rotor4` with invariant-decomposition exponential), GJK and EPA lifted to R⁴ (EPA penetration depth checked against analytical values), persistent contact manifolds, a warm-started projected Gauss-Seidel solver. Driven by the test suite and by a console example (`cargo run --release --example physics4d`); no interactive demo runs it yet.
- **Exact polychoral cross-sections.** All six regular convex 4-polytopes with exact topology at unit circumradius. The default section path computes geometry rather than approximating a mesh; a raymarched-SDF surface mode is kept alongside it for comparison, and a wireframe overlay draws the full edge graph.
- **Geometry as a type parameter.** One `Space` trait (exp, log, distance, parallel transport, isometries) carries the scene. The SDF raymarcher runs on E³, R⁴, H³, S³, and a blended E³-to-H³ metric; the rasterizer runs on E³, R⁴, and a separate embedded S³ type. E³ and R⁴ are the two geometries where both paths share one implementation; H³ has no rasterizer path.
- **Determinism.** Simulation is bit-reproducible: same binary, same inputs, same bits. A fixed-scenario replay test pins it as its own CI gate.

## How it's built

- A geometry is anything implementing `Space`. Everything downstream is an opt-in capability trait (`WgslSpace`, `RasterizableSpace`, `SectionableSpace`, `PhysicsSpace`), so a new geometry is wired in by implementing what it actually supports.
- Scenes are typed SDF trees that emit WGSL on demand; the emit chain combines scene code with the selected space's shader prelude.
- Correctness is enforced by an invariant test suite, not by inspecting rendered output: Gauss-Bonnet angle defect on geodesic triangles, isometry invariance of distance, transport length preservation, loop holonomy. A primitive whose visualization looks right but whose invariants fail does not ship.
- `loam-math` depends on `glam`, `serde`, and `tracing`; the geometry is first-party. The crate-by-crate map lives in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- A variable-curvature `BlendedSpace` (a single Riemannian metric interpolating E³ to H³) exists as a validated reference implementation; it stays out of the headlines until it earns its frame budget.

## Roadmap

None of this is in the repo yet. No dates: this is a single-maintainer project, and a missed public date is worse than no date. Substrate and tooling get built ahead of demand; game-specific features ship with the game that pulls them.

**Now.** A deterministic multithreaded simulation core, gated on bit-identical results between one thread and many. Direct manipulation of 4D objects: picking, click-to-throw, a rotation gizmo. Loadable scenes behind a demo registry. Frame-budget work on the SDF path, with before-and-after traces as the artifact.

**Next.** Quotient and gluing geometries: flat 3-torus rooms and lens spaces, where the fundamental group does the work at closed-form cost. CPU evaluation of the typed SDF tree, which unblocks baked colliders for smooth shapes and an SDF editor. First-party UI, so egui becomes an optional debug layer. Animation tooling.

**Later.** The Polytope Playground stays the flagship. Then 4D experiments (life, boids, creature behaviour) on GPU compute, which sit outside the determinism boundary and will say so. Games past that are unstarted; the one that most shapes the engine is a horror game built on multiply-connected space.

## Lineage

Marc ten Bosch's *4D Toys* is the direct conceptual ancestor of the polytope playground: four-dimensional objects as inhabitants of a 4D space whose 3D slices the viewer inspects. Zeno Rogue's *HyperRogue*, CodeParade's HyperEngine/*Hyperbolica*, and Michael Walczyk's `polychora` shaped the non-Euclidean and 4D rendering choices. Textbook citations (Coxeter, Hestenes and Sobczyk, do Carmo, Foley et al., Knuth) live in the rustdocs at each use site.

## Getting involved

I use AI coding tools, primarily Claude Code, heavily; invariant tests gate every primitive regardless of authorship, and responsibility for what ships is mine. Single-maintainer project: reach out before starting anything at [humesze@proton.me](mailto:humesze@proton.me).

## License

Dual-licensed under MIT OR Apache-2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
