// Sky and ground shading, shared verbatim by `SkyGroundNode`'s pipeline and by
// the hyperslice kernel, which still needs `sky` for its body fog term after
// the background takes the ground over. Declares no binding and no uniform, so
// either module may paste it ahead of its own declarations.

// The two endpoints are mirrored in Rust as `SKY_BELOW` / `SKY_ABOVE`, from
// which `SKY_HORIZON` (the filmstrip's clear) is derived; keep in sync.
fn sky(rd: vec3<f32>) -> vec3<f32> {
    let t = (rd.y + 1.0) * 0.5;
    return mix(vec3<f32>(0.04, 0.05, 0.10), vec3<f32>(0.10, 0.13, 0.22), t);
}

// `checker_fade` collapses the two colours toward their mean. Driven by the
// same fog factor as the sky blend, it band-limits the checker at grazing
// angles, where an unbounded plane puts many cells inside one pixel and a
// point sample aliases.
fn ground_color(
    p: vec3<f32>,
    dark: vec3<f32>,
    light: vec3<f32>,
    checker_fade: f32,
) -> vec3<f32> {
    let g = floor(p.x) + floor(p.z);
    let alt = abs(g - 2.0 * floor(g * 0.5));
    return mix(mix(dark, light, alt), 0.5 * (dark + light), checker_fade);
}
