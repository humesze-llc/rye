//! Whether the hyperslice kernel should skip bodies whose 4D extent misses the
//! active `w` slice, and what it would cost the picture.
//!
//! A body whose w-interval `[b.w - extent, b.w + extent]` excludes `u.w_slice`
//! has an empty cross-section, so a ray confined to that hyperplane cannot hit
//! it. Dropping it from the per-step minimum is SAFE (the surviving minimum is
//! still a Lipschitz-1 bound on everything the ray can reach, so nothing
//! tunnels) but not VALUE-preserving: where the culled body was the argmin the
//! marcher takes a longer step, converges at a different `t`, and shades a
//! different point of the same surface.
//!
//! Measured on an NVIDIA GeForce RTX 4090 Laptop GPU (Vulkan, driver 610.74),
//! Windows 11 Pro 10.0.26200, `cargo test --release`.
//!
//! Verdict: not taken. It is negative where it matters, and the frames it
//! speeds up are the blank ones nobody waits on. The probes stay so the
//! trade can be re-priced: if the row cap rises well past eight, or the
//! marcher loses the bounding-ball early-out, the 8-body block is where it
//! would turn.
//!
//! Both probes need an adapter. Only the identity one carries the `gpu_probe`
//! suffix CI's software-adapter job selects on: it is 38 draws, while the cost
//! probe is 14400 and would put roughly ten seconds of discrete-GPU fragment
//! work through lavapipe. The cost probe takes the `_perf` suffix `ci.yml`
//! already excludes for `orbit_advance_perf`. Run both with
//! `cargo test --release -p loam-render --test hyperslice_wslice_cull --
//! --include-ignored --nocapture --test-threads=1`.

use loam_math::Rotor4;
use loam_render::raymarch::{
    polytope_extended_sdfs_wgsl, BodyUniform, Hyperslice4DNode, HYPERSLICE_KERNEL_WGSL,
    SHAPE_24CELL, SHAPE_PENTATOPE, SHAPE_TESSERACT,
};
use loam_render::Viewport;
use wgpu::*;

// 1280 * 4 bytes per row is a multiple of `COPY_BYTES_PER_ROW_ALIGNMENT`, so
// the readback needs no row padding, and 720 rows of 384-iteration marches
// put the measurement in the fragment shader, not in submission overhead.
const SIZE: [u32; 2] = [1280, 720];

// Matches the playground's `BODY_SIZE` / `BODY_X_SPACING` / `BODY_Y`. The
// circumradius is also each body's w half-extent: the per-shape SDFs are
// unit-circumradius and the dispatcher scales by `polytope_size`.
const BODY_SIZE: f32 = 0.7;
const BODY_X_SPACING: f32 = 1.8;
const BODY_Y: f32 = 0.9;

// The demo's usual three and its `MAX_ROW_LEN` of eight; the body loop is
// linear in the count, so the cull's saving grows with it.
const PROBE_BODY_COUNTS: [usize; 2] = [3, 8];

// Inside every body's bounding 4-ball, so the slab test provably cannot fire
// and the images must match bit for bit. Not the same set as the slices that
// draw a cross-section; the probe prints the occupancy it actually gets.
const SLICES_INSIDE_BOUNDING_BALLS: [f32; 4] = [0.0, 0.25, 0.5, 0.68];

// Slices past every body's bounding 4-ball, where the cull fires on all of
// them and bit-exactness is exactly what is at stake.
const SLICES_PAST_BOUNDING_BALLS: [f32; 2] = [0.75, 1.2];

// No body's bounding ball reaches it, so its frame is floor and sky. Reusing
// the largest probe slice keeps the reference inside the priced set.
const EMPTY_REFERENCE_SLICE: f32 = 1.2;

fn probe_slices() -> Vec<f32> {
    SLICES_INSIDE_BOUNDING_BALLS
        .into_iter()
        .chain(SLICES_PAST_BOUNDING_BALLS)
        .collect()
}

// What the demo scenes put under the row, and what makes the marcher walk the
// depth of the frame instead of escaping to `max_t` on the first step.
const FLOOR_SCENE_WGSL: &str = r#"
const LOAM_PRIM_HYPERSPHERE4D: u32 = 0u;
const LOAM_PRIM_HALFSPACE4D: u32 = 1u;
const LOAM_PRIM_OTHER: u32 = 255u;
struct LoamSceneHit { dist: f32, kind: u32 }
fn loam_scene_at(p: vec3<f32>) -> LoamSceneHit {
    return LoamSceneHit(p.y, LOAM_PRIM_HALFSPACE4D);
}
fn loam_scene_sdf(p: vec3<f32>) -> f32 {
    return loam_scene_at(p).dist;
}
fn loam_scene_max_t(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    if (rd.y > -1.0e-6) { return 1.0e9; }
    return -ro.y / rd.y;
}
"#;

// Head of both body loops in the kernel (`loam_dynamic_bodies_sdf` and
// `loam_total_sdf`), which is where a w-slab cull would go.
const BODY_LOOP_HEAD: &str = "    for (var i: u32 = 0u; i < body_count; i = i + 1u) {
        let b = u.bodies[i];
        let kind = u32(b.kind + 0.5);
";

// The candidate. `radius_or_shape` is the radius for a sphere and the shape
// index for a polytope, so the half-extent is selected on kind;
// `polytope_size` is the polytope's bounding-4-ball radius.
const W_SLAB_CULL: &str =
    "        let w_extent = select(b.radius_or_shape, b.polytope_size, kind == BODY_KIND_POLYTOPE);
        if (abs(u.w_slice - b.position.w) > w_extent) { continue; }
";

// A splice rather than a second copy of the kernel: the two must differ in
// exactly this and nothing else, or the comparison prices a typo.
fn culled_kernel() -> String {
    let culled =
        HYPERSLICE_KERNEL_WGSL.replace(BODY_LOOP_HEAD, &format!("{BODY_LOOP_HEAD}{W_SLAB_CULL}"));
    assert_eq!(
        culled.matches(W_SLAB_CULL).count(),
        2,
        "the cull must land in both body loops; the kernel's loop head moved"
    );
    culled
}

fn module_for(device: &Device, kernel: &str, label: &str) -> ShaderModule {
    let polytope = polytope_extended_sdfs_wgsl();
    device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(format!("{kernel}\n{polytope}\n{FLOOR_SCENE_WGSL}").into()),
    })
}

// The Shapes-view layout the demo raymarches. Shapes cycle so the row
// exercises three different branches of the SDF dispatcher.
fn probe_bodies(count: usize) -> Vec<BodyUniform> {
    const SHAPES: [u32; 3] = [SHAPE_TESSERACT, SHAPE_24CELL, SHAPE_PENTATOPE];
    const COLORS: [[f32; 3]; 3] = [[0.85, 0.35, 0.30], [0.30, 0.75, 0.85], [0.80, 0.75, 0.30]];
    (0..count)
        .map(|slot| {
            let x = (slot as f32 - (count as f32 - 1.0) * 0.5) * BODY_X_SPACING;
            BodyUniform::polytope_with_rotor(
                [x, BODY_Y, 0.0, 0.0],
                SHAPES[slot % SHAPES.len()],
                BODY_SIZE,
                Rotor4::IDENTITY,
                COLORS[slot % COLORS.len()],
            )
        })
        .collect()
}

// `up` is `forward` rotated a quarter turn in the yz-plane, so the basis
// stays orthonormal.
fn probe_node(device: &Device, module: &ShaderModule, body_count: usize) -> Hyperslice4DNode {
    let mut node = Hyperslice4DNode::new(device, TextureFormat::Rgba8Unorm, module, 1);
    let pitch: f32 = -0.22;
    let len = (1.0 + pitch * pitch).sqrt();
    let (fy, fz) = (pitch / len, -1.0 / len);
    let u = node.uniforms_mut();
    u.camera_pos = [0.0, 1.8, 6.0];
    u.camera_forward = [0.0, fy, fz];
    u.camera_right = [1.0, 0.0, 0.0];
    u.camera_up = [0.0, -fz, fy];
    u.resolution = [SIZE[0] as f32, SIZE[1] as f32];
    node.set_bodies(&probe_bodies(body_count));
    node
}

fn probe_target(device: &Device) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: Some("wslice cull probe target"),
        size: Extent3d {
            width: SIZE[0],
            height: SIZE[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

// Submitted and waited on, so the caller's wall clock brackets the GPU work
// and not just the encode.
fn render_slice(
    device: &Device,
    queue: &Queue,
    node: &mut Hyperslice4DNode,
    view: &TextureView,
    w_slice: f32,
) {
    node.uniforms_mut().w_slice = w_slice;
    node.flush_uniforms(queue);
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("wslice cull probe encoder"),
    });
    node.record_in_viewport(&mut encoder, view, Viewport::full(SIZE));
    queue.submit(Some(encoder.finish()));
    device
        .poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("draw poll");
}

fn read_back_rgba(device: &Device, queue: &Queue, texture: &Texture) -> Vec<u8> {
    let bytes_per_row = SIZE[0] * 4;
    assert_eq!(
        bytes_per_row % COPY_BYTES_PER_ROW_ALIGNMENT,
        0,
        "probe width chosen so no row padding is needed"
    );
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("wslice cull probe readback"),
        size: (bytes_per_row * SIZE[1]) as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("wslice cull probe readback encoder"),
    });
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &readback,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        Extent3d {
            width: SIZE[0],
            height: SIZE[1],
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));
    let slice = readback.slice(..);
    slice.map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("readback poll");
    let data = slice.get_mapped_range().to_vec();
    readback.unmap();
    data
}

// FNV-1a over the whole RGBA buffer (Fowler/Noll/Vo, 1991). The assertion is
// equality, so collision resistance is not load-bearing and a 64-bit
// non-cryptographic mixer is the right cost.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// So a report says how far off the image is, not only that it moved.
fn differing_bytes(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

async fn request_device() -> Result<(Device, Queue), String> {
    let instance = Instance::default();
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| format!("request_adapter failed: {e}"))?;
    adapter
        .request_device(&DeviceDescriptor {
            label: Some("hyperslice-wslice-cull"),
            required_features: Features::empty(),
            required_limits: Limits::default(),
            memory_hints: MemoryHints::default(),
            trace: Trace::Off,
            experimental_features: Default::default(),
        })
        .await
        .map_err(|e| format!("request_device failed: {e}"))
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn the_w_slab_cull_is_bit_exact_inside_every_bounding_ball_gpu_probe() {
    let (device, queue) = pollster::block_on(request_device()).expect("wgpu device");
    let target = probe_target(&device);
    let view = target.create_view(&TextureViewDescriptor::default());

    let stock_module = module_for(&device, HYPERSLICE_KERNEL_WGSL, "wslice stock");
    let control_module = module_for(&device, HYPERSLICE_KERNEL_WGSL, "wslice stock control");
    let culled_module = module_for(&device, &culled_kernel(), "wslice culled");

    for body_count in PROBE_BODY_COUNTS {
        let mut stock = probe_node(&device, &stock_module, body_count);
        let mut control = probe_node(&device, &control_module, body_count);
        let mut culled = probe_node(&device, &culled_module, body_count);

        render_slice(&device, &queue, &mut stock, &view, EMPTY_REFERENCE_SLICE);
        let empty_pixels = read_back_rgba(&device, &queue, &target);

        for w_slice in probe_slices() {
            render_slice(&device, &queue, &mut stock, &view, w_slice);
            let stock_pixels = read_back_rgba(&device, &queue, &target);
            render_slice(&device, &queue, &mut control, &view, w_slice);
            let control_pixels = read_back_rgba(&device, &queue, &target);
            render_slice(&device, &queue, &mut culled, &view, w_slice);
            let culled_pixels = read_back_rgba(&device, &queue, &target);

            assert_eq!(
                fnv1a64(&stock_pixels),
                fnv1a64(&control_pixels),
                "two modules built from the same kernel disagree at {body_count} \
                 bodies, w = {w_slice} ({} bytes differ); the byte-identity rig \
                 is measuring driver nondeterminism, not the change",
                differing_bytes(&stock_pixels, &control_pixels),
            );

            println!(
                "bodies {body_count}  w {w_slice:>5}  occupancy {:>7} bytes vs \
                 the empty frame",
                differing_bytes(&empty_pixels, &stock_pixels),
            );

            let differing = differing_bytes(&stock_pixels, &culled_pixels);
            if SLICES_INSIDE_BOUNDING_BALLS.contains(&w_slice) {
                assert_eq!(
                    fnv1a64(&stock_pixels),
                    fnv1a64(&culled_pixels),
                    "the w-slab cull moved the image at {body_count} bodies, \
                     w = {w_slice} ({differing} bytes differ), where every body's \
                     bounding-ball w extent {BODY_SIZE} still covers the slice \
                     and the slab test cannot fire: either the predicate now \
                     culls a body the ray can hit, or the compiler reordered the \
                     minimum chain around the added branch",
                );
            } else {
                println!(
                    "bodies {body_count}  w {w_slice:>5}  past every bounding \
                     ball: {differing} bytes differ"
                );
            }
        }
    }
}

#[test]
#[ignore = "perf probe; needs an adapter, run with --include-ignored"]
fn the_w_slab_cull_frame_cost_perf() {
    // Draws per timed batch, enough that the submit-and-wait floor is a small
    // share of the total at this resolution.
    const REPS: u32 = 40;
    // Batches per cell, reported as the median so one scheduler hiccup does
    // not decide the verdict.
    const BATCHES: usize = 9;

    let (device, queue) = pollster::block_on(request_device()).expect("wgpu device");
    let target = probe_target(&device);
    let view = target.create_view(&TextureViewDescriptor::default());

    let stock_module = module_for(&device, HYPERSLICE_KERNEL_WGSL, "wslice stock");
    let control_module = module_for(&device, HYPERSLICE_KERNEL_WGSL, "wslice stock control");
    let culled_module = module_for(&device, &culled_kernel(), "wslice culled");

    let batch_us = |node: &mut Hyperslice4DNode, w_slice: f32| -> f64 {
        let start = std::time::Instant::now();
        for _ in 0..REPS {
            render_slice(&device, &queue, node, &view, w_slice);
        }
        start.elapsed().as_secs_f64() * 1.0e6 / REPS as f64
    };
    let median = |mut batches: Vec<f64>| -> f64 {
        batches.sort_by(f64::total_cmp);
        batches[BATCHES / 2]
    };

    println!(
        "{} x {} px, median of {BATCHES} interleaved batches of {REPS} \
         submit-and-wait draws",
        SIZE[0], SIZE[1],
    );
    println!("bodies      w  stock_us  culled_us  delta_us  noise_us");
    for body_count in PROBE_BODY_COUNTS {
        let mut stock = probe_node(&device, &stock_module, body_count);
        let mut control = probe_node(&device, &control_module, body_count);
        let mut culled = probe_node(&device, &culled_module, body_count);
        for w_slice in probe_slices() {
            // Batches interleave across the three variants so a clock ramp or a
            // thermal drift over the run lands on all of them alike; measuring
            // one variant to completion and then the next charges the drift to
            // whichever went second.
            let (mut stocks, mut culleds, mut controls) = (Vec::new(), Vec::new(), Vec::new());
            for round in 0..=BATCHES {
                let (s, c, n) = (
                    batch_us(&mut stock, w_slice),
                    batch_us(&mut culled, w_slice),
                    batch_us(&mut control, w_slice),
                );
                // Round zero is the warm-up: first touch of a pipeline pays
                // shader upload and clock ramp.
                if round > 0 {
                    stocks.push(s);
                    culleds.push(c);
                    controls.push(n);
                }
            }
            let (s, c, n) = (median(stocks), median(culleds), median(controls));
            println!(
                "{body_count:>6}  {w_slice:>5}  {s:8.1}  {c:9.1}  {:8.1}  {:8.1}",
                s - c,
                (s - n).abs()
            );
        }
    }
}
