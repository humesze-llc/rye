# Architecture

How the crates fit together and which decisions ripple: the dependency
structure and the trait boundaries behind it. Per-crate roles live in each
crate's rustdoc.

## Dependency tiers

The workspace is a DAG. A crate's tier is the longest path from it down to a
crate with no first-party dependencies, so every crate depends only on
strictly lower tiers.

```
tier 0  loam-math  loam-input  loam-time  loam-asset  loam-console
tier 1  loam-shape  loam-camera  loam-player  loam-shader
tier 2  loam-scene  loam-physics  loam-text  loam-render  loam-egui
tier 3  loam-app
tier 4  loam  polytope_playground  hero  tesseract_demo
```

Every first-party edge, from the manifests. Dev edges are test-only and do not
constrain the tiers.

```
loam-math             (none)
loam-input            (none)
loam-time             (none)
loam-asset            (none)
loam-console          (none)
loam-shape            math
loam-camera           math  input
loam-player           math  input
loam-shader           math  asset                 dev: scene
loam-scene            math  shape
loam-physics          math  shape  time
loam-text             shape                          dev: math  physics
loam-render           math  shape  time           dev: scene  shader
loam-egui             math  camera  console
loam-app              math  asset  input  time  camera  egui  render  shader
loam                  every loam-* crate except loam-console and loam-egui
polytope_playground   math  shape  time  scene  physics  camera  text  egui
                      render  app
hero                  math  shape  time  physics  text  egui  render  app
tesseract_demo        math  shape  time  camera  egui  render  app
```

- `loam-math` is the root: the `Space` trait, metrics, the bivector/rotor
  geometric algebra, projections. Nothing in the workspace is below it.
- `loam-shape` (the geometry/topology data: `Shape`, polytope topology, the
  vertex/face generators) is the other stable surface, and `loam-math` is its
  only first-party dependency. The two together are the foundation; promoting
  anything into them is a deliberate decision.
- `loam-render` does not depend on `loam-scene` or `loam-shader` at build
  time. Both are dev-dependencies, taken so the renderer's tests can drive the
  real producers. The raymarch nodes accept a pre-compiled `ShaderModule` and
  must keep working with any producer of one.
- The render/scene coupling is a WGSL symbol contract, not a Rust edge.
  `loam-render`'s hyperslice kernel calls
  `loam_scene_sdf(p: vec3<f32>) -> f32`; `loam-scene`'s
  `Scene4::to_hyperslice_wgsl*` emits its definition; the application
  concatenates the two strings and compiles one module
  (`crates/polytope_playground/src/main.rs`, `shader_source`). Renaming the
  symbol on either side breaks at shader compile, not at `cargo build`, so
  both crates pin the name in tests.
- `loam-render` does not depend on `loam-physics`: rendering must not pull in
  the simulation layer. Polytope topology lives in `loam-shape` precisely so
  the renderer can use it without the physics dependency.
- `loam-physics` is consumed by `polytope_playground` and the `loam` facade,
  not by the engine shell (`loam-app`). 4D rigid-body simulation is an
  application capability, not a render-path prerequisite.
- `loam-console` carries the console model (registry, scrollback, tokenizer,
  binds) with no UI-framework dependency; `loam-egui` is the frontend that
  drives and paints it. That split is why `loam-console` sits at tier 0 while
  the egui integration sits at tier 2, lifted there by the `loam-camera`
  dependency its world-anchored label helper needs.

The rule: a change in a low tier ripples upward, so the low tiers carry the
strictest review. A change isolated to `loam-render` or a demo is cheap.

## The capability-trait split

A geometry is anything that implements `Space` (the smooth-Riemannian core:
`exp`, `log`, `distance`, parallel transport). Everything a geometry can *do*
downstream is an opt-in capability trait, not a hard-coded geometry case:

- `WgslSpace`: emit the WGSL prelude to raymarch this space on the GPU.
- `RasterizableSpace`: project and tessellate edges/sections for the rasterizer.
- `SectionableSpace`: cross-section algorithm support. Implemented for
  `EuclideanR4`; `loam-shape`'s polytope sectioning calls it per edge.
- `PhysicsSpace`: rigid-body simulation in this space.

A new geometry is wired through the engine by implementing the capabilities it
actually supports, each with its own tests, rather than editing every renderer
and demo. The traits are split because they change at different rates: `Space`
is the most stable surface; the WGSL ABI is the least; they are not one trait
because they do not move together.

## One geometry, two rendering paths

The SDF raymarch path and the rasterizer path run on the same `Space` impls
wherever a space implements both. Code that presumes one rendering path is a
defect: a geometry that implements `WgslSpace` raymarches, one that implements
`RasterizableSpace` rasterizes. The overlap today is `EuclideanR3` and
`EuclideanR4`; `HyperbolicH3`, `SphericalS3` and `FlatTorus3` raymarch only,
and the rasterizer's S³ is a separate type (`SphericalS3Embedded`). Over
`EuclideanR4` the shared impl is what lets the playground show a polytope as a
raymarched surface, an exact cross-section, and a wireframe from one source of
truth.

## The determinism boundary

The math and simulation layers (`loam-math`, `loam-physics`, the sim-critical
parts of the demos) are held to a reproducibility contract stated as a property,
not a mechanism: same binary, same inputs, same bits. Concretely: f32,
fixed-step, deterministic iteration and reduction order, seeded randomness, no
fast-math. The sim is single-threaded today; parallelism is admissible only with
a fixed schedule and fixed reduction order, and only backed by a measurement.
Code that runs only for presentation (UI, render node setup, camera framing) and
GPU compute are outside the boundary and must not pretend to honor it. The
boundary is explicit so a contributor knows which side they are editing.

## Public surface vs internal

`loam-math` and `loam-shape` are the surfaces an external consumer would build
on; their `pub` items are contracts. The render/app crates are usable but still
moving pre-1.0. The root `loam` crate is a facade: it depends on every engine
crate except `loam-console` and `loam-egui`, but re-exports only `loam-asset`,
`loam-math`, `loam-render`, `loam-shader` and `loam-time` under short names.
The rest are deliberately not re-exported so a consumer declares them in their
own `Cargo.toml` instead of reaching them through `loam::*`.
`polytope_playground` and `tesseract_demo` are demonstrations, not API: depend
on the engine crates, not on a demo.
