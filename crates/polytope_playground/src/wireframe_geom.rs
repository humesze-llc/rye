use glam::{Vec3, Vec4};
use loam_shape::polytope::Polytope4;
use loam_shape::LineMesh;

use crate::consts::{HYPERSLICE_MIN_THICKNESS, SPACE_TESSELLATION_SAMPLES};

// Matches the `PROJECTION_DENOM_EPSILON` inside `EuclideanR4`'s projection.
pub(crate) const PERSPECTIVE_SCALE_DENOM_EPSILON: f32 = 1e-4;

// Under 1 keeps every arc endpoint in front of the eye on zoom-in.
pub(crate) const STEREOGRAPHIC_VIEW_RADIUS_FRACTION: f32 = 0.75;

// A unit-circumradius image reaches ~1.7, so a close zoom never clips the figure.
pub(crate) const STEREOGRAPHIC_VIEW_RADIUS_FLOOR: f32 = 2.5;

// Past this the fixed sample count is too coarse in the steep near-pole region.
pub(crate) const STEREOGRAPHIC_CELL16_RADIUS_MAX: f32 = 10.0;

// The 16-cell at an 8-unit camera distance.
#[cfg(test)]
pub(crate) const STEREOGRAPHIC_VIEW_RADIUS: f32 = 6.0;

// Only the 16-cell has a vertex on the pole (`±e_w`); every other image is bounded.
pub(crate) fn stereographic_view_radius(polytope: Polytope4, camera_distance: f32) -> f32 {
    match polytope {
        Polytope4::Cell16 => (camera_distance * STEREOGRAPHIC_VIEW_RADIUS_FRACTION).clamp(
            STEREOGRAPHIC_VIEW_RADIUS_FLOOR,
            STEREOGRAPHIC_CELL16_RADIUS_MAX,
        ),
        _ => f32::INFINITY,
    }
}

// Far above any view radius; `1e4²` stays in f32's exact-integer range.
pub(crate) const STEREOGRAPHIC_POLE_FAR_CAP: f32 = 1.0e4;

pub(crate) fn stereographic_clip_radius(
    projection: &loam_math::Projection<4>,
    view_radius: f32,
) -> Option<f32> {
    match *projection {
        loam_math::Projection::Stereographic { .. } => Some(view_radius),
        loam_math::Projection::Identity
        | loam_math::Projection::Orthographic { .. }
        | loam_math::Projection::Perspective4D { .. }
        | loam_math::Projection::Schlegel { .. } => None,
    }
}

// `None` for the non-affine maps; `Orthographic` is only ever `drop_axis: 3` here.
pub(crate) fn perspective_scale_at_w(
    w_slice: f32,
    projection: &loam_math::Projection<4>,
) -> Option<f32> {
    match *projection {
        loam_math::Projection::Identity | loam_math::Projection::Orthographic { .. } => Some(1.0),
        loam_math::Projection::Perspective4D { focal_distance } => {
            Some(focal_distance / (focal_distance - w_slice).max(PERSPECTIVE_SCALE_DENOM_EPSILON))
        }
        loam_math::Projection::Schlegel { .. } | loam_math::Projection::Stereographic { .. } => {
            None
        }
    }
}

pub(crate) fn projection_maps_chords_to_lines(projection: &loam_math::Projection<4>) -> bool {
    match *projection {
        loam_math::Projection::Identity
        | loam_math::Projection::Orthographic { .. }
        | loam_math::Projection::Perspective4D { .. }
        | loam_math::Projection::Schlegel { .. } => true,
        loam_math::Projection::Stereographic { .. } => false,
    }
}

pub(crate) fn flat_edge_uses_endpoint_chord(projection: &loam_math::Projection<4>) -> bool {
    match *projection {
        loam_math::Projection::Stereographic { .. } => true,
        _ => projection_maps_chords_to_lines(projection),
    }
}

pub(crate) fn local_r3_to_world(p: [f32; 3], section_scale: f32, body_pos_r3: Vec3) -> [f32; 3] {
    let scaled = Vec3::from_array(p) * section_scale;
    (scaled + body_pos_r3).to_array()
}

// Exact: every cap vertex has `w = w_slice`, so appending it inverts the drop-w.
pub(crate) fn cap_vertex_projected_and_world(
    p_r3: [f32; 3],
    w_slice: f32,
    section_scale: Option<f32>,
    projection: &loam_math::Projection<4>,
    body_pos_r3: Vec3,
) -> (Vec3, [f32; 3]) {
    match section_scale {
        Some(scale) => {
            let projected = Vec3::from_array(p_r3) * scale;
            (projected, local_r3_to_world(p_r3, scale, body_pos_r3))
        }
        None => {
            let p4 = Vec4::new(p_r3[0], p_r3[1], p_r3[2], w_slice);
            let projected =
                <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(
                    p4, projection,
                );
            (projected, (projected + body_pos_r3).to_array())
        }
    }
}

#[cfg(test)]
pub(crate) fn project_to_world(
    p: Vec4,
    projection: &loam_math::Projection<4>,
    body_pos_r3: Vec3,
) -> Vec3 {
    <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(p, projection)
        + body_pos_r3
}

// Below this the slerp has no axis and the edge falls back to the chord.
pub(crate) const MIN_EDGE_RADIUS: f32 = 1e-6;

// Closed bounds, so an endpoint exactly on the slab edge is kept.
pub(crate) fn slab_overlaps(
    interval_min: f32,
    interval_max: f32,
    w_slice: f32,
    thickness: f32,
) -> bool {
    let half = thickness.max(HYPERSLICE_MIN_THICKNESS) * 0.5;
    let slab_min = w_slice - half;
    let slab_max = w_slice + half;
    interval_min <= slab_max && interval_max >= slab_min
}

// Fold order `(lo.min, hi.max)` for bit-reproducibility.
pub(crate) fn cell_w_range(cell: &[u32], local_vertices: &[Vec4]) -> (f32, f32) {
    cell.iter()
        .map(|&i| local_vertices[i as usize].w)
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), w| {
            (lo.min(w), hi.max(w))
        })
}

// Near-pole samples are dropped, not rescaled: rescaling keeps the pole-crossing flip.
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_projected_chord(
    mesh: &mut LineMesh<3>,
    a: Vec4,
    b: Vec4,
    color_a: [f32; 4],
    color_b: [f32; 4],
    width: f32,
    projection: &loam_math::Projection<4>,
    body_pos_r3: Vec3,
    view_radius: f32,
) {
    let samples = SPACE_TESSELLATION_SAMPLES;
    let clip_radius = stereographic_clip_radius(projection, view_radius);
    let sample_at = |p4: Vec4| {
        let projected = <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(
            p4, projection,
        );
        (projected, (projected + body_pos_r3).to_array())
    };
    let (proj0, world0) = sample_at(a);
    let mut prev_world = world0;
    let mut prev_c = color_a;
    let mut prev_in = sample_in_radius(proj0, clip_radius);
    for k in 1..=samples {
        let s = k as f32 / samples as f32;
        let (proj, world) = sample_at(a.lerp(b, s));
        let c = [
            color_a[0] + (color_b[0] - color_a[0]) * s,
            color_a[1] + (color_b[1] - color_a[1]) * s,
            color_a[2] + (color_b[2] - color_a[2]) * s,
            color_a[3] + (color_b[3] - color_a[3]) * s,
        ];
        let cur_in = sample_in_radius(proj, clip_radius);
        if prev_in && cur_in {
            mesh.segments.push((prev_world, world));
            mesh.colors.push((prev_c, c));
            mesh.widths.push(width);
        }
        prev_world = world;
        prev_c = c;
        prev_in = cur_in;
    }
}

#[inline]
pub(crate) fn sample_in_radius(projected: Vec3, radius: Option<f32>) -> bool {
    match radius {
        None => true,
        Some(r) => projected.length_squared() <= r * r,
    }
}

// Larger root of the segment/sphere quadratic (do Carmo, *Curves and Surfaces*, §1.5).
pub(crate) fn radius_crossing_t(p_in: Vec3, p_out: Vec3, r: f32) -> f32 {
    let d = p_out - p_in;
    let a = d.length_squared();
    if a <= f32::MIN_POSITIVE {
        return 1.0;
    }
    let b = p_in.dot(d);
    let c = p_in.length_squared() - r * r;
    let disc = (b * b - a * c).max(0.0);
    ((-b + disc.sqrt()) / a).clamp(0.0, 1.0)
}

pub(crate) fn clip_point(
    p_in: Vec3,
    p_out: Vec3,
    c_in: [f32; 4],
    c_out: [f32; 4],
    t: f32,
    body_pos: Vec3,
) -> ([f32; 3], [f32; 4]) {
    let boundary = (p_in.lerp(p_out, t) + body_pos).to_array();
    let color = [
        c_in[0] + (c_out[0] - c_in[0]) * t,
        c_in[1] + (c_out[1] - c_in[1]) * t,
        c_in[2] + (c_out[2] - c_in[2]) * t,
        c_in[3] + (c_out[3] - c_in[3]) * t,
    ];
    (boundary, color)
}

pub(crate) fn push_clipped_subsegment(
    mesh: &mut LineMesh<3>,
    clip_radius: Option<f32>,
    width: f32,
    body_pos_r3: Vec3,
    prev: (Vec3, [f32; 3], [f32; 4], bool),
    cur: (Vec3, [f32; 3], [f32; 4], bool),
) {
    let (prev_proj, prev_world, prev_c, prev_in) = prev;
    let (cur_proj, cur_world, cur_c, cur_in) = cur;
    let mut push = |a_world, b_world, a_c, b_c| {
        mesh.segments.push((a_world, b_world));
        mesh.colors.push((a_c, b_c));
        mesh.widths.push(width);
    };
    match (clip_radius, prev_in, cur_in) {
        (None, _, _) | (Some(_), true, true) => push(prev_world, cur_world, prev_c, cur_c),
        (Some(_), false, false) => {}
        (Some(r), true, false) => {
            let t = radius_crossing_t(prev_proj, cur_proj, r);
            let (bw, bc) = clip_point(prev_proj, cur_proj, prev_c, cur_c, t, body_pos_r3);
            push(prev_world, bw, prev_c, bc);
        }
        (Some(r), false, true) => {
            let t = radius_crossing_t(cur_proj, prev_proj, r);
            let (bw, bc) = clip_point(cur_proj, prev_proj, cur_c, prev_c, t, body_pos_r3);
            push(bw, cur_world, bc, cur_c);
        }
    }
}

// Inside the pole clamp band the map deflates toward the origin; restore the true magnitude.
pub(crate) fn stereographic_view_point(p: Vec4, projection: &loam_math::Projection<4>) -> Vec3 {
    let proj =
        <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(p, projection);
    let loam_math::Projection::Stereographic { pole } = projection else {
        return proj;
    };
    let dot = p.normalize().dot(*pole).clamp(-1.0, 1.0);
    let raw = 1.0 - dot;
    if raw < loam_math::STEREOGRAPHIC_POLE_EPSILON && proj.length() > MIN_EDGE_RADIUS {
        let true_mag = ((1.0 + dot) / raw.max(f32::MIN_POSITIVE))
            .sqrt()
            .min(STEREOGRAPHIC_POLE_FAR_CAP);
        proj.normalize() * true_mag
    } else {
        proj
    }
}

// Per-triangle, so fill and perimeter cull in lockstep.
pub(crate) fn retain_in_radius_triangles(
    indices: &mut Vec<[u32; 3]>,
    start_i: usize,
    start_v: usize,
    projected: &[Vec3],
    radius: Option<f32>,
) {
    if radius.is_none() {
        return;
    }
    let appended = &mut indices[start_i..];
    let mut write = 0usize;
    for read in 0..appended.len() {
        let tri = appended[read];
        let in_radius = tri
            .iter()
            .all(|&i| sample_in_radius(projected[i as usize - start_v], radius));
        if in_radius {
            appended[write] = tri;
            write += 1;
        }
    }
    indices.truncate(start_i + write);
}

// Arc about `arc_center`, and a plain `lerp` in R⁴: this runs over every 600-cell edge per frame.
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_blended_edge(
    mesh: &mut LineMesh<3>,
    a: Vec4,
    b: Vec4,
    arc_center: Vec4,
    color_a: [f32; 4],
    color_b: [f32; 4],
    width: f32,
    blend: f32,
    projection: &loam_math::Projection<4>,
    body_pos_r3: Vec3,
    slerp_scratch: &mut Vec<Vec4>,
    view_radius: f32,
) {
    if blend <= 0.0 {
        if flat_edge_uses_endpoint_chord(projection) {
            let clip_radius = stereographic_clip_radius(projection, view_radius);
            let a3_local =
                <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(
                    a, projection,
                );
            let b3_local =
                <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(
                    b, projection,
                );
            if sample_in_radius(a3_local, clip_radius) && sample_in_radius(b3_local, clip_radius) {
                mesh.segments.push((
                    (a3_local + body_pos_r3).to_array(),
                    (b3_local + body_pos_r3).to_array(),
                ));
                mesh.colors.push((color_a, color_b));
                mesh.widths.push(width);
            }
            return;
        }
        push_projected_chord(
            mesh,
            a,
            b,
            color_a,
            color_b,
            width,
            projection,
            body_pos_r3,
            view_radius,
        );
        return;
    }

    let offset_a = a - arc_center;
    let offset_b = b - arc_center;
    let radius_a = offset_a.length();
    let radius_b = offset_b.length();
    if radius_a < MIN_EDGE_RADIUS || radius_b < MIN_EDGE_RADIUS {
        if flat_edge_uses_endpoint_chord(projection) {
            let clip_radius = stereographic_clip_radius(projection, view_radius);
            let a3_local =
                <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(
                    a, projection,
                );
            let b3_local =
                <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(
                    b, projection,
                );
            if sample_in_radius(a3_local, clip_radius) && sample_in_radius(b3_local, clip_radius) {
                mesh.segments.push((
                    (a3_local + body_pos_r3).to_array(),
                    (b3_local + body_pos_r3).to_array(),
                ));
                mesh.colors.push((color_a, color_b));
                mesh.widths.push(width);
            }
            return;
        }
        push_projected_chord(
            mesh,
            a,
            b,
            color_a,
            color_b,
            width,
            projection,
            body_pos_r3,
            view_radius,
        );
        return;
    }

    let samples = SPACE_TESSELLATION_SAMPLES;
    let clip_radius = stereographic_clip_radius(projection, view_radius);
    let p0u = offset_a / radius_a;
    let p1u = offset_b / radius_b;
    slerp_scratch.clear();
    <loam_math::SphericalS3Embedded as loam_math::RasterizableSpace<4>>::tessellate_segment(
        p0u,
        p1u,
        samples,
        slerp_scratch,
    );

    let proj0 = stereographic_view_point(a, projection);
    let mut prev_proj = proj0;
    let mut prev_world = (proj0 + body_pos_r3).to_array();
    let mut prev_c = color_a;
    let mut prev_in = sample_in_radius(proj0, clip_radius);
    for (k, &arc_pt) in slerp_scratch.iter().enumerate().skip(1) {
        let s = k as f32 / samples as f32;
        let flat = a.lerp(b, s);
        let radius = radius_a + (radius_b - radius_a) * s;
        let sphere = arc_center + radius * arc_pt;
        let proj = stereographic_view_point(flat.lerp(sphere, blend), projection);
        let world = (proj + body_pos_r3).to_array();
        let c = [
            color_a[0] + (color_b[0] - color_a[0]) * s,
            color_a[1] + (color_b[1] - color_a[1]) * s,
            color_a[2] + (color_b[2] - color_a[2]) * s,
            color_a[3] + (color_b[3] - color_a[3]) * s,
        ];
        let cur_in = sample_in_radius(proj, clip_radius);
        push_clipped_subsegment(
            mesh,
            clip_radius,
            width,
            body_pos_r3,
            (prev_proj, prev_world, prev_c, prev_in),
            (proj, world, c, cur_in),
        );
        prev_proj = proj;
        prev_world = world;
        prev_c = c;
        prev_in = cur_in;
    }
}
