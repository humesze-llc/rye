//! One analytic background for every scene with a 3D camera: a ray-direction
//! sky, a closed-form checkerboard on `y = Ground::y` with analytic depth, and
//! the frame's colour and depth clear.
//!
//! Rays are unprojected through the caller's own `view_proj` rather than built
//! from a camera basis and a field of view, which is what the two marchers do
//! ([`crate::raymarch`]). They own their whole image; this node has to agree
//! with raster content composited over it, so it takes the same matrix that
//! content is drawn with and the horizon cannot drift from it.

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::*;

/// Sky and checkerboard shading, shared with
/// [`crate::raymarch::HYPERSLICE_KERNEL_WGSL`].
pub const SKY_GROUND_WGSL: &str = include_str!("sky_ground.wgsl");

// Mirrors `sky`'s two endpoints in sky_ground.wgsl; pinned by
// `sky_horizon_is_the_shader_gradient_at_the_horizon`.
const SKY_BELOW: [f64; 3] = [0.04, 0.05, 0.10];
const SKY_ABOVE: [f64; 3] = [0.10, 0.13, 0.22];

/// `sky` evaluated at `rd.y = 0`. Linear, as every [`Color`] clear is, so a
/// pass clearing to it meets a pass shading `sky` without a seam.
pub const SKY_HORIZON: Color = Color {
    r: 0.5 * (SKY_BELOW[0] + SKY_ABOVE[0]),
    g: 0.5 * (SKY_BELOW[1] + SKY_ABOVE[1]),
    b: 0.5 * (SKY_BELOW[2] + SKY_ABOVE[2]),
    a: 1.0,
};

/// The pair the hyperslice kernel painted before this node took the ground.
pub const GROUND_DARK_GREY: [f32; 3] = [0.18, 0.20, 0.24];
pub const GROUND_LIGHT_GREY: [f32; 3] = [0.30, 0.32, 0.36];

/// The checkerboard plane under the sky. `y` is a parameter because a scene
/// may centre its content on the origin, where a plane at `y = 0` would cut it
/// in half.
#[derive(Copy, Clone, Debug)]
pub struct Ground {
    pub y: f32,
    pub dark: [f32; 3],
    pub light: [f32; 3],
    pub visible: bool,
}

/// Uniform buffer for [`SkyGroundNode`]. Bind group 0, binding 0. `std140`:
/// two `mat4x4<f32>`, then two `vec2<f32>`, then a `vec3<f32>` and an `f32`
/// per 16-byte slot.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SkyGroundUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub viewport_origin: [f32; 2],
    pub resolution: [f32; 2],
    pub ground_dark: [f32; 3],
    pub ground_y: f32,
    pub ground_light: [f32; 3],
    pub show_ground: f32,
}

impl SkyGroundUniforms {
    /// `view_proj` must be the matrix the raster content composited over this
    /// pass is drawn with; the ground's depth is `clip.z / clip.w` under it,
    /// which is the number that content's vertex stage produces.
    pub fn new(view_proj: Mat4, viewport: crate::Viewport, ground: Ground) -> Self {
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            viewport_origin: [viewport.x as f32, viewport.y as f32],
            resolution: viewport.resolution_f32(),
            ground_dark: ground.dark,
            ground_y: ground.y,
            ground_light: ground.light,
            show_ground: if ground.visible { 1.0 } else { 0.0 },
        }
    }
}

const SKY_GROUND_NODE_WGSL: &str = concat!(
    include_str!("sky_ground.wgsl"),
    r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    viewport_origin: vec2<f32>,
    resolution: vec2<f32>,
    ground_dark: vec3<f32>,
    ground_y: f32,
    ground_light: vec3<f32>,
    show_ground: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

// |rd.y| under this puts the plane past a million units, where the fog term is
// 1.0 to the last f32 bit and the pixel is sky whatever the intersection
// returns. It is here so the division cannot produce an infinity.
const HORIZON_EPS: f32 = 1.0e-6;

const FOG_PER_UNIT: f32 = 0.05;

// The hyperslice kernel's key light and ambient floor, so a marched body and
// the ground under it take the same shading.
const LIGHT_DIR: vec3<f32> = vec3<f32>(0.5, 0.85, 0.3);
const AMBIENT: f32 = 0.20;
const DIFFUSE: f32 = 0.85;

struct Fragment {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn unproject(ndc: vec3<f32>) -> vec3<f32> {
    let h = u.inv_view_proj * vec4<f32>(ndc, 1.0);
    return h.xyz / h.w;
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> Fragment {
    // `frag_pos.xy` is framebuffer space whatever `set_viewport` carved out,
    // and framebuffer y runs down while clip y runs up.
    let uv = (frag_pos.xy - u.viewport_origin) / u.resolution;
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    // wgpu clip space puts the near plane at z = 0 and the far plane at z = 1.
    let near = unproject(vec3<f32>(ndc_xy, 0.0));
    let far = unproject(vec3<f32>(ndc_xy, 1.0));
    let rd = normalize(far - near);

    var out: Fragment;

    // One positive-t test covers both sides of the plane. A hit from
    // underneath (near.y < ground_y, rd.y > 0) is reachable by drag alone:
    // `OrbitController` clamps pitch short of the pole but lets the camera
    // orbit below the plane.
    var t: f32 = 0.0;
    var hit = false;
    if (u.show_ground >= 0.5 && abs(rd.y) > HORIZON_EPS) {
        t = (u.ground_y - near.y) / rd.y;
        hit = t > 0.0;
    }

    if (!hit) {
        out.color = vec4<f32>(sky(rd), 1.0);
        out.depth = 1.0;
        return out;
    }

    let p_hit = near + rd * t;
    let fog = 1.0 - exp(-t * FOG_PER_UNIT);
    let base = ground_color(p_hit, u.ground_dark, u.ground_light, fog);
    // The plane's normal is +Y everywhere, so the Lambert term is one number
    // for the whole surface and only the sky blend varies across it.
    let lambert = max(dot(vec3<f32>(0.0, 1.0, 0.0), normalize(LIGHT_DIR)), 0.0);
    let lit = base * (AMBIENT + DIFFUSE * lambert);
    out.color = vec4<f32>(mix(lit, sky(rd), fog), 1.0);

    let clip = u.view_proj * vec4<f32>(p_hit, 1.0);
    out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
    return out;
}
"#
);

pub struct SkyGroundNode {
    pipeline: RenderPipeline,
    uniform_buf: Buffer,
    bind_group: BindGroup,
}

impl SkyGroundNode {
    /// `depth_format` and `sample_count` must match the attachments passed to
    /// [`Self::record`], which owns the frame's colour and depth clear.
    pub fn new(
        device: &Device,
        target_format: TextureFormat,
        depth_format: TextureFormat,
        sample_count: u32,
    ) -> Self {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("sky_ground shader"),
            source: ShaderSource::Wgsl(SKY_GROUND_NODE_WGSL.into()),
        });

        let uniform_buf = device.create_buffer(&BufferDescriptor {
            label: Some("sky_ground uniforms"),
            size: std::mem::size_of::<SkyGroundUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("sky_ground bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("sky_ground bg"),
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("sky_ground pipeline layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("sky_ground pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &module,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            // `Always`: the pass clears depth and this is the first thing in
            // the frame to write it, so there is nothing to test against.
            depth_stencil: Some(DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Always,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            uniform_buf,
            bind_group,
        }
    }

    /// Call once per frame before [`Self::record`]; the UBO is undefined at
    /// construction.
    pub fn set_uniforms(&self, queue: &Queue, uniforms: &SkyGroundUniforms) {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(uniforms));
    }

    /// Records into the caller's `encoder` and does not submit. Clears both
    /// attachments, so it must be the frame's first pass; `viewport` restricts
    /// the shading, not the clear.
    pub fn record(
        &self,
        encoder: &mut CommandEncoder,
        view: &TextureView,
        depth_view: &TextureView,
        viewport: Option<&crate::Viewport>,
    ) {
        let mut rp = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("sky_ground pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(SKY_HORIZON),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        if let Some(vp) = viewport {
            vp.apply(&mut rp);
        }
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &self.bind_group, &[]);
        rp.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec2, Vec3, Vec4};

    #[test]
    fn sky_ground_wgsl_validates() {
        let module = naga::front::wgsl::parse_str(SKY_GROUND_NODE_WGSL)
            .unwrap_or_else(|e| panic!("sky_ground WGSL parse failed:\n{e}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("sky_ground WGSL must validate");
    }

    #[test]
    fn sky_horizon_is_the_shader_gradient_at_the_horizon() {
        let endpoint = |c: [f64; 3]| format!("vec3<f32>({:.2}, {:.2}, {:.2})", c[0], c[1], c[2]);
        for c in [SKY_BELOW, SKY_ABOVE] {
            assert!(
                SKY_GROUND_WGSL.contains(&endpoint(c)),
                "sky_ground.wgsl no longer mixes {}; the filmstrip's clear \
                 would seam against the sky it stands in for",
                endpoint(c),
            );
        }
        assert_eq!(SKY_HORIZON.r, 0.5 * (SKY_BELOW[0] + SKY_ABOVE[0]));
        assert_eq!(SKY_HORIZON.g, 0.5 * (SKY_BELOW[1] + SKY_ABOVE[1]));
        assert_eq!(SKY_HORIZON.b, 0.5 * (SKY_BELOW[2] + SKY_ABOVE[2]));
    }

    // The Rust struct and the WGSL `Uniforms` block are two hand-written
    // declarations of one buffer layout. Compare them member by member: a
    // field added, dropped or reordered on one side alone moves an offset
    // here, which a size assertion against a literal cannot see.
    #[test]
    fn uniforms_size_matches_wgsl() {
        let module = naga::front::wgsl::parse_str(SKY_GROUND_NODE_WGSL).expect("parse");
        let ty = module
            .types
            .iter()
            .map(|(_, t)| t)
            .find(|t| t.name.as_deref() == Some("Uniforms"))
            .expect("the node's WGSL declares `Uniforms`");
        let naga::TypeInner::Struct { members, span } = &ty.inner else {
            panic!("`Uniforms` is not a struct");
        };
        assert_eq!(*span as usize, std::mem::size_of::<SkyGroundUniforms>());
        assert_eq!(std::mem::align_of::<SkyGroundUniforms>(), 4);
        let rust_offsets = [
            (
                "view_proj",
                std::mem::offset_of!(SkyGroundUniforms, view_proj),
            ),
            (
                "inv_view_proj",
                std::mem::offset_of!(SkyGroundUniforms, inv_view_proj),
            ),
            (
                "viewport_origin",
                std::mem::offset_of!(SkyGroundUniforms, viewport_origin),
            ),
            (
                "resolution",
                std::mem::offset_of!(SkyGroundUniforms, resolution),
            ),
            (
                "ground_dark",
                std::mem::offset_of!(SkyGroundUniforms, ground_dark),
            ),
            (
                "ground_y",
                std::mem::offset_of!(SkyGroundUniforms, ground_y),
            ),
            (
                "ground_light",
                std::mem::offset_of!(SkyGroundUniforms, ground_light),
            ),
            (
                "show_ground",
                std::mem::offset_of!(SkyGroundUniforms, show_ground),
            ),
        ];
        assert_eq!(members.len(), rust_offsets.len());
        for (member, (name, offset)) in members.iter().zip(rust_offsets) {
            assert_eq!(member.name.as_deref(), Some(name));
            assert_eq!(member.offset as usize, offset, "offset of {name}");
        }
    }

    // Rust twin of `fs_main`'s ground path: unproject the pixel centre, meet
    // the plane, and return the hit and the depth the fragment writes.
    fn ground_hit(u: &SkyGroundUniforms, pixel: Vec2) -> Option<(Vec3, f32)> {
        let view_proj = Mat4::from_cols_array_2d(&u.view_proj);
        let inv = Mat4::from_cols_array_2d(&u.inv_view_proj);
        let uv = (pixel - Vec2::from(u.viewport_origin)) / Vec2::from(u.resolution);
        let ndc_xy = Vec2::new(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
        let unproject = |z: f32| {
            let h = inv * Vec4::new(ndc_xy.x, ndc_xy.y, z, 1.0);
            h.truncate() / h.w
        };
        let near = unproject(0.0);
        let rd = (unproject(1.0) - near).normalize();
        if rd.y.abs() <= 1.0e-6 {
            return None;
        }
        let t = (u.ground_y - near.y) / rd.y;
        if t <= 0.0 {
            return None;
        }
        let p_hit = near + rd * t;
        let clip = view_proj * p_hit.extend(1.0);
        Some((p_hit, (clip.z / clip.w).clamp(0.0, 1.0)))
    }

    fn probe_uniforms(eye: Vec3, ground_y: f32) -> SkyGroundUniforms {
        let viewport = crate::Viewport {
            x: 0,
            y: 0,
            width: 640,
            height: 480,
        };
        let view = Mat4::look_at_rh(eye, Vec3::new(0.0, ground_y, 0.0), Vec3::Y);
        let proj = Mat4::perspective_rh(60.0_f32.to_radians(), 640.0 / 480.0, 0.1, 100.0);
        SkyGroundUniforms::new(
            proj * view,
            viewport,
            Ground {
                y: ground_y,
                dark: GROUND_DARK_GREY,
                light: GROUND_LIGHT_GREY,
                visible: true,
            },
        )
    }

    // Take a point known to be on the plane, project it the way the raster
    // vertex stage does, and demand `ground_hit` at that pixel recover the
    // same point and the same depth. This pins the algorithm, not its WGSL
    // transcription: a flipped framebuffer y, a near/far swap or an
    // intersection sign flip is caught here only if it is also made in the
    // twin above. The shader's own copy is pinned on a device by
    // `ground_occludes_raster_content_behind_the_plane_gpu_probe`.
    #[test]
    fn background_depth_matches_the_raster_projection() {
        for (eye, ground_y) in [
            (Vec3::new(0.0, 2.5, 6.0), 0.0),
            (Vec3::new(3.0, 0.4, 3.0), 0.0),
            (Vec3::new(0.0, -2.5, 6.0), 0.0),
            (Vec3::new(1.0, -0.9, 4.0), -1.2),
        ] {
            let u = probe_uniforms(eye, ground_y);
            let view_proj = Mat4::from_cols_array_2d(&u.view_proj);
            let resolution = Vec2::from(u.resolution);
            for on_plane in [
                Vec3::new(0.0, ground_y, 0.0),
                Vec3::new(1.7, ground_y, -2.3),
                Vec3::new(-4.0, ground_y, 5.0),
                Vec3::new(0.6, ground_y, -40.0),
            ] {
                let clip = view_proj * on_plane.extend(1.0);
                let ndc = clip.truncate() / clip.w;
                let pixel = Vec2::new(
                    (ndc.x * 0.5 + 0.5) * resolution.x,
                    (0.5 - ndc.y * 0.5) * resolution.y,
                ) + Vec2::from(u.viewport_origin);
                let (hit, depth) = ground_hit(&u, pixel)
                    .unwrap_or_else(|| panic!("eye {eye} sees no ground at {on_plane}"));
                assert!(
                    hit.distance(on_plane) < 1.0e-3 * on_plane.length().max(1.0),
                    "eye {eye}, ground y {ground_y}: unprojected {hit} for {on_plane}"
                );
                assert!(
                    (depth - ndc.z).abs() < 1.0e-5,
                    "eye {eye}, ground y {ground_y}: background depth {depth} \
                     against raster depth {} at {on_plane}",
                    ndc.z,
                );
            }
        }
    }

    #[test]
    fn a_ray_leaving_the_plane_behind_finds_no_ground() {
        let u = probe_uniforms(Vec3::new(0.0, 2.5, 6.0), 0.0);
        let resolution = Vec2::from(u.resolution);
        // Top row of the frame: the camera looks down at the origin, so these
        // pixels point above the horizon.
        for x in [0.5_f32, resolution.x * 0.5, resolution.x - 0.5] {
            assert!(
                ground_hit(&u, Vec2::new(x, 0.5)).is_none(),
                "a ray above the horizon met the ground at x = {x}"
            );
        }
    }
}
