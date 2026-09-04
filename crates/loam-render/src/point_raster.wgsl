// Quad-expanded antialiased discs; the disc is the inscribed circle of the [-1, 1]² quad.

struct CameraUniform {
    view_projection: mat4x4<f32>,
    viewport_size:   vec2<f32>,
    _pad:            vec2<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VsOut {
    @builtin(position) clip:    vec4<f32>,
    @location(0) uv:            vec2<f32>,
    @location(1) color:         vec4<f32>,
    @location(2) radius_px:     f32,
};

@vertex
fn vs_main(
    @location(0) corner:    u32,
    @location(1) pos:       vec3<f32>,
    @location(2) color:     vec4<f32>,
    @location(3) radius_px: f32,
) -> VsOut {
    let p_clip = camera.view_projection * vec4<f32>(pos, 1.0);
    let p_ndc  = p_clip.xyz / p_clip.w;

    // Corner bit 0 is +x, bit 1 is +y; same pattern as line_raster.
    let dx = select(-1.0, 1.0, corner == 1u || corner == 3u);
    let dy = select(-1.0, 1.0, corner >= 2u);

    let half_with_aa = radius_px + 1.0;

    let half_vp  = camera.viewport_size * 0.5;
    let off_ndc  = vec2<f32>(dx, dy) * half_with_aa / half_vp;

    var out: VsOut;
    out.clip = vec4<f32>(
        (p_ndc.xy + off_ndc) * p_clip.w,
        p_ndc.z * p_clip.w,
        p_clip.w,
    );
    out.uv        = vec2<f32>(dx, dy);
    out.color     = color;
    out.radius_px = radius_px;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let r = length(in.uv);
    let inner = max(0.0, in.radius_px / (in.radius_px + 1.0));
    let coverage = 1.0 - smoothstep(inner, 1.0, r);
    return vec4<f32>(in.color.rgb, in.color.a * coverage);
}
