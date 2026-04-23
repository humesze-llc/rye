// Geodesic ray march kernel — rye engine WGSL library.
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
    let scale = max(ball_scale, 1e-5);
    var p = ro * scale;

    // Probe the Riemannian norm of the camera-space direction at p to
    // initialise a Riemannian-unit tangent vector. Space-agnostic via ABI.
    let rd_unit = rye_safe_normalize(rd, vec3<f32>(0.0, 0.0, -1.0));
    let probe_eps = 1e-4;
    let probed     = rye_exp(p, rd_unit * probe_eps);
    let riem_norm  = rye_distance(p, probed) / probe_eps;
    var v = rd_unit / max(riem_norm, 1e-7);

    var t_scene = 0.0;
    var t_arc   = 0.0;
    let hit_eps  = 0.001 * scale;
    let min_step = 0.0001 * scale;

    for (var i = 0; i < 256; i = i + 1) {
        // Guard: escape near the Space boundary (Poincaré ball / S³ hemisphere).
        if rye_origin_distance(p) > RYE_MAX_ARC * 0.92 {
            return vec4<f32>(0.0, 0.0, 0.0, -1.0);
        }
        let d = rye_scene_sdf(p);
        if d < hit_eps {
            return vec4<f32>(p, t_scene);
        }
        if t_scene > 40.0 || t_arc > RYE_MAX_ARC {
            return vec4<f32>(0.0, 0.0, 0.0, -1.0);
        }
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
fn rye_estimate_normal(p: vec3<f32>, ball_scale: f32) -> vec3<f32> {
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
