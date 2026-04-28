//! Print every GPU adapter wgpu can see, with a compact summary of
//! capabilities. Useful for bug reports and first-run sanity checking.

fn main() {
    let instance = wgpu::Instance::default();
    let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all());

    if adapters.is_empty() {
        eprintln!("no wgpu adapters found");
        std::process::exit(1);
    }

    println!("wgpu adapters ({}):\n", adapters.len());

    for (i, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        let limits = adapter.limits();
        let features = adapter.features();

        println!(
            "[{i}] {} - {:?} ({:?})",
            info.name, info.backend, info.device_type
        );
        println!("    driver:       {} [{}]", info.driver, info.driver_info);
        println!(
            "    vendor:device {:#06x}:{:#06x}",
            info.vendor, info.device
        );
        println!(
            "    max_tex_2d={}  max_buffer={}  bind_groups={}",
            limits.max_texture_dimension_2d, limits.max_buffer_size, limits.max_bind_groups
        );
        println!(
            "    uniform_align={}  storage_align={}",
            limits.min_uniform_buffer_offset_alignment, limits.min_storage_buffer_offset_alignment
        );
        println!(
            "    compute_wg=({}, {}, {})  invocations={}",
            limits.max_compute_workgroup_size_x,
            limits.max_compute_workgroup_size_y,
            limits.max_compute_workgroup_size_z,
            limits.max_compute_invocations_per_workgroup
        );
        println!("    features:     {features:?}");
        println!();
    }
}
