// Geodesic ray march kernel: rye engine WGSL library.
//
// Functions: rye_safe_normalize, rye_march_geodesic, rye_estimate_normal.
//
// Callers must ensure the following are defined earlier in the assembled
// shader (from the Space prelude and scene module, respectively):
//   rye_exp, rye_distance, rye_parallel_transport, rye_origin_distance,
//   RYE_MAX_ARC, rye_scene_sdf.

fn rye_safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    if l2 < 1e-12 { return fallback; }
    return v * inverseSqrt(l2);
}

// March a geodesic ray through rye_scene_sdf.
//
// ro:         ray origin in camera space
// rd:         ray direction in camera space (need not be unit)
// ball_scale: camera-space to Space-coordinates scale factor
//
// Returns vec4(p_space, t_scene) on hit; w = -1.0 on miss or boundary escape.
fn rye_march_geodesic(ro: vec3<f32>, rd: vec3<f32>, ball_scale: f32) -> vec4<f32> {
    // Floor `ball_scale` so curved Spaces near the camera don't divide by
    // zero on degenerate inputs.
    let scale = max(ball_scale, 1e-5);
    var p = ro * scale;

    // Probe the Riemannian norm of the camera-space direction at p to
    // initialise a Riemannian-unit tangent vector. Space-agnostic via ABI.
    let rd_unit = rye_safe_normalize(rd, vec3<f32>(0.0, 0.0, -1.0));
    // Small finite-difference step to estimate the local Riemannian
    // metric scaling without wandering far from `p`.
    let probe_eps = 1e-4;
    let probed     = rye_exp(p, rd_unit * probe_eps);
    let riem_norm  = rye_distance(p, probed) / probe_eps;
    // Floor the divisor to keep the tangent finite when the metric
    // collapses (rare boundary case in curved Spaces).
    var v = rd_unit / max(riem_norm, 1e-7);

    var t_scene = 0.0;
    var t_arc   = 0.0;
    // Hit/min-step thresholds scale with `ball_scale` so a small camera
    // (close-up demo) and a large camera (overview) get the same number
    // of march steps over their respective fields of view.
    let hit_eps  = 0.001  * scale;  // 1/1000 of camera-scale, the tightest hit gap
    let min_step = 0.0001 * scale;  // 1/10000 of camera-scale, prevents stalls in flat regions

    // 256 steps caps worst-case work at ~40K SDF evals per pixel; demos
    // converge well under that even in H³.
    for (var i = 0; i < 256; i = i + 1) {
        // Escape near the Space boundary (Poincaré ball / S³ hemisphere).
        // 0.92 leaves a small buffer so the ABI's saturating distance
        // doesn't asymptote into a stall before the escape fires.
        if rye_origin_distance(p) > RYE_MAX_ARC * 0.92 {
            return vec4<f32>(0.0, 0.0, 0.0, -1.0);
        }
        let d = rye_scene_sdf(p);
        if d < hit_eps {
            return vec4<f32>(p, t_scene);
        }
        // 40.0 is the Euclidean-equivalent march cap (well past the
        // typical scene); RYE_MAX_ARC caps Riemannian arc-length and is
        // the curved-Space-specific termination.
        if t_scene > 40.0 || t_arc > RYE_MAX_ARC {
            return vec4<f32>(0.0, 0.0, 0.0, -1.0);
        }
        // 0.85 under-steps the SDF, eliminates overshoot when the SDF
        // is an approximation (typical for any non-trivial CSG tree).
        let step   = max(d * 0.85, min_step);
        let next_p = rye_exp(p, v * step);
        let next_v = rye_parallel_transport(p, next_p, v);
        p = next_p;
        // Do NOT Euclidean-renormalize: rye_parallel_transport preserves the
        // Riemannian norm. Renormalising the 3-component would corrupt it.
        v       = select(v, next_v, dot(next_v, next_v) > 1e-12);
        t_scene = t_scene + step / scale;
        t_arc   = t_arc   + step;
    }
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
}

// Central-difference SDF gradient at a Space-coordinate point.
//
// `p` and the returned normal are in the Space's *chart* coordinates
// (Cartesian for E³, Poincaré-ball for H³, unit-3-sphere for S³),
// matching how `rye_scene_sdf` is emitted. The chart-coord gradient
// is **not** the Riemannian normal, but the renderer only ever uses
// it for Lambert shading, where the chart-coord gradient and the
// transported Riemannian gradient point in the same direction (both
// give the Euclidean-screen-space normal the lighting kernel wants).
// `ball_scale` widens the central-difference epsilon so curved-Space
// SDFs near `|p| ≈ 1` (boundary of the model) don't sample across
// the boundary and get a NaN normal.
fn rye_estimate_normal(p: vec3<f32>, ball_scale: f32) -> vec3<f32> {
    // 0.0012 is slightly larger than the march `hit_eps = 0.001` so the
    // gradient probe sees real surface variation rather than landing
    // inside the same hit cell. Floor by 1e-5 to avoid degenerate eps.
    let eps = 0.0012 * max(ball_scale, 1e-5);
    let ex = vec3<f32>(eps, 0.0, 0.0);
    let ey = vec3<f32>(0.0, eps, 0.0);
    let ez = vec3<f32>(0.0, 0.0, eps);
    let g = vec3<f32>(
        rye_scene_sdf(p + ex) - rye_scene_sdf(p - ex),
        rye_scene_sdf(p + ey) - rye_scene_sdf(p - ey),
        rye_scene_sdf(p + ez) - rye_scene_sdf(p - ez),
    );
    return rye_safe_normalize(g, vec3<f32>(0.0, 1.0, 0.0));
}
