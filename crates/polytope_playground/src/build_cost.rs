//! What R costs. A restart calls a scene's registry builder again, so the
//! builder's own wall time is the frame the gesture drops.
//!
//! Run: `cargo test -p polytope_playground --release build_cost -- --ignored
//! --nocapture`. Ignored because it is a measurement, not an assertion: it
//! prints, and a machine with no adapter has nothing to say rather than
//! something to fail. Release, because the shipped gesture is a release
//! gesture.
//!
//! `RenderDevice` needs a surface and therefore a window, so this drives the
//! same builder-side calls from a surfaceless device instead of calling
//! `RotateScene::new` whole. What is left out is named at its call site.

use std::time::Instant;

use wgpu::*;

// The native swapchain picks an sRGB format and `RunConfig::default` asks for
// one sample.
const FORMAT: TextureFormat = TextureFormat::Bgra8UnormSrgb;
const SAMPLES: u32 = 1;

// One cold build plus a distribution. The first build of a module pays the
// driver's compile once and is reported on its own: a restart is by definition
// not the first build, since boot already paid it.
const RUNS: usize = 32;
const MEASURED: usize = RUNS - 1;

const FRAME_BUDGET_MS: f64 = 1000.0 / 60.0;

fn device() -> Option<(Device, Queue)> {
    let instance = Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    println!("adapter: {:?}", adapter.get_info());
    pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: Some("scene build cost"),
        required_features: Features::empty(),
        required_limits: Limits::default(),
        memory_hints: MemoryHints::default(),
        trace: Trace::Off,
        experimental_features: Default::default(),
    }))
    .ok()
}

fn report(label: &str, samples: Vec<f64>) {
    let cold = samples[0];
    let mut warm = samples[1..].to_vec();
    warm.sort_by(f64::total_cmp);
    println!(
        "  {label:<18} first {cold:>8.3}   p50 {:>7.3}  p95 {:>7.3}  max {:>7.3}   \
         (p95 is {:.0}% of a {FRAME_BUDGET_MS:.1}ms frame)",
        warm[MEASURED / 2],
        warm[MEASURED * 95 / 100],
        warm[MEASURED - 1],
        100.0 * warm[MEASURED * 95 / 100] / FRAME_BUDGET_MS,
    );
}

fn time(mut run: impl FnMut()) -> Vec<f64> {
    (0..RUNS)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed().as_secs_f64() * 1000.0
        })
        .collect()
}

#[test]
#[ignore = "measurement: prints, asserts nothing, and wants a GPU and --release"]
fn scene_build_cost() {
    let Some((device, queue)) = device() else {
        println!("no adapter; nothing measured");
        return;
    };

    println!("rotate, milliseconds over {RUNS} builds:");
    report("shader source", time(|| drop(crate::shader_source())));
    // Everything `Demo::new` asks the GPU for, plus the WGSL above. Left out:
    // `parse_row`, the physics table and the console registry, none of which
    // touch a device or a font.
    report(
        "nodes + shader",
        time(|| drop(crate::build_nodes(&device, FORMAT, SAMPLES))),
    );
    // The other half of `RotateScene::new`: the HUD's glyph atlas.
    report(
        "HUD atlas",
        time(|| drop(crate::hud::TextHud::new(&device, &queue, FORMAT, SAMPLES).expect("hud"))),
    );

}
