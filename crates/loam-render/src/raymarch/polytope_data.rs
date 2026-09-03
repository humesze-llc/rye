//! WGSL for the 120-cell and 600-cell SDFs: Wolfe's greedy hyperplane
//! projection as in `loam_physics::euclidean_r4::polytope_sdf_wolfe`, with a
//! vertex search for |S|=4. Also defines `CELL_INRADIUS_UNIT`.

use std::fmt::Write;

use glam::Vec4;
use loam_shape::polytope_geom::{
    cell120_face_planes, cell120_vertices, cell600_face_planes, cell600_vertices,
    icosian_inradius_unit,
};

/// The kernel references both names, so callers include this or
/// [`polytope_extended_sdfs_wgsl`].
pub fn polytope_stub_sdfs_wgsl() -> &'static str {
    "// ---- Polytope stub SDFs (no 120-cell/600-cell bodies in scene) ----\n\
     fn cell120_sdf_local(p: vec4<f32>) -> f32 { return 1.0e9; }\n\
     fn cell600_sdf_local(p: vec4<f32>) -> f32 { return 1.0e9; }\n"
}

/// ~24 KB of `const` data, which slows every pixel shader on some drivers;
/// prefer the stub when no such body is in the scene.
pub fn polytope_extended_sdfs_wgsl() -> String {
    let mut s = String::with_capacity(32 * 1024);
    s.push_str("// ---- Extended polytope SDFs (120-cell, 600-cell) ----\n");
    writeln!(
        s,
        "const CELL_INRADIUS_UNIT: f32 = {:.10};",
        icosian_inradius_unit()
    )
    .unwrap();

    let (n120, _) = cell120_face_planes();
    let (n600, _) = cell600_face_planes();
    let v120 = cell120_vertices(1.0);
    let v600 = cell600_vertices(1.0);

    emit_vec4_array(&mut s, "CELL120_FACE_NORMALS", &n120);
    emit_vec4_array(&mut s, "CELL600_FACE_NORMALS", &n600);
    emit_vec4_array(&mut s, "CELL120_VERTICES", &v120);
    emit_vec4_array(&mut s, "CELL600_VERTICES", &v600);

    s.push_str(WOLFE_PROJECTION_HELPER_WGSL);
    s.push_str(&wolfe_sdf_function(
        "cell120_sdf_local",
        "CELL120_FACE_NORMALS",
        120,
        "CELL120_VERTICES",
        600,
    ));
    s.push_str(&wolfe_sdf_function(
        "cell600_sdf_local",
        "CELL600_FACE_NORMALS",
        600,
        "CELL600_VERTICES",
        120,
    ));

    s
}

fn emit_vec4_array(out: &mut String, name: &str, data: &[Vec4]) {
    writeln!(
        out,
        "const {name}: array<vec4<f32>, {len}> = array(",
        len = data.len()
    )
    .unwrap();
    for v in data {
        writeln!(
            out,
            "    vec4<f32>({:.10}, {:.10}, {:.10}, {:.10}),",
            v.x, v.y, v.z, v.w
        )
        .unwrap();
    }
    out.push_str(");\n");
}

// Mirrors `loam_physics::euclidean_r4::project_onto_active_planes`.
const WOLFE_PROJECTION_HELPER_WGSL: &str = r#"
fn polytope_project_active(
    p: vec4<f32>,
    a0: vec4<f32>, a1: vec4<f32>, a2: vec4<f32>,
    count: u32,
    inradius: f32,
) -> vec4<f32> {
    let b0 = dot(a0, p) - inradius;
    let b1 = dot(a1, p) - inradius;
    let b2 = dot(a2, p) - inradius;
    if (count == 1u) {
        return p - b0 * a0;
    }
    if (count == 2u) {
        let g01 = dot(a0, a1);
        let det = 1.0 - g01 * g01;
        if (abs(det) < 1.0e-9) { return p; }
        let inv_det = 1.0 / det;
        let l0 = (b0 - g01 * b1) * inv_det;
        let l1 = (b1 - g01 * b0) * inv_det;
        return p - l0 * a0 - l1 * a1;
    }
    let g01 = dot(a0, a1);
    let g02 = dot(a0, a2);
    let g12 = dot(a1, a2);
    let det = 1.0 + 2.0 * g01 * g02 * g12 - g01 * g01 - g02 * g02 - g12 * g12;
    if (abs(det) < 1.0e-9) { return p; }
    let inv_det = 1.0 / det;
    let c00 = 1.0 - g12 * g12;
    let c01 = g02 * g12 - g01;
    let c02 = g01 * g12 - g02;
    let c11 = 1.0 - g02 * g02;
    let c12 = g01 * g02 - g12;
    let c22 = 1.0 - g01 * g01;
    let l0 = (c00 * b0 + c01 * b1 + c02 * b2) * inv_det;
    let l1 = (c01 * b0 + c11 * b1 + c12 * b2) * inv_det;
    let l2 = (c02 * b0 + c12 * b1 + c22 * b2) * inv_det;
    return p - l0 * a0 - l1 * a1 - l2 * a2;
}
"#;

fn wolfe_sdf_function(
    fn_name: &str,
    face_normals_name: &str,
    face_count: u32,
    vertices_name: &str,
    vertex_count: u32,
) -> String {
    format!(
        r#"
fn {fn_name}(p: vec4<f32>) -> f32 {{
    let inradius = CELL_INRADIUS_UNIT;
    var max_d: f32 = -1.0e9;
    var max_i: u32 = 0u;
    for (var i: u32 = 0u; i < {face_count}u; i = i + 1u) {{
        let d = dot({face_normals_name}[i], p) - inradius;
        if (d > max_d) {{
            max_d = d;
            max_i = i;
        }}
    }}
    if (max_d <= 0.0) {{ return max_d; }}

    // Outside the unit circumsphere the max plane distance is a Lipschitz-1 lower bound.
    if (dot(p, p) > 1.0) {{ return max_d; }}

    var active_idx_0: u32 = max_i;
    var active_idx_1: u32 = 0u;
    var active_idx_2: u32 = 0u;
    var active_count: u32 = 1u;
    var a0: vec4<f32> = {face_normals_name}[max_i];
    var a1: vec4<f32> = vec4<f32>(0.0);
    var a2: vec4<f32> = vec4<f32>(0.0);
    var q: vec4<f32> = p - max_d * a0;
    let tol: f32 = 1.0e-6;

    for (var iter: u32 = 0u; iter < 3u; iter = iter + 1u) {{
        var next_d: f32 = tol;
        var next_i: u32 = 0xffffffffu;
        for (var i: u32 = 0u; i < {face_count}u; i = i + 1u) {{
            if (i == active_idx_0) {{ continue; }}
            if (active_count >= 2u && i == active_idx_1) {{ continue; }}
            if (active_count >= 3u && i == active_idx_2) {{ continue; }}
            let d = dot({face_normals_name}[i], q) - inradius;
            if (d > next_d) {{
                next_d = d;
                next_i = i;
            }}
        }}
        if (next_i == 0xffffffffu) {{
            return length(p - q);
        }}
        if (active_count == 1u) {{
            active_idx_1 = next_i;
            a1 = {face_normals_name}[next_i];
            active_count = 2u;
            q = polytope_project_active(p, a0, a1, a2, 2u, inradius);
        }} else if (active_count == 2u) {{
            active_idx_2 = next_i;
            a2 = {face_normals_name}[next_i];
            active_count = 3u;
            q = polytope_project_active(p, a0, a1, a2, 3u, inradius);
        }} else {{
            var best_d2: f32 = 1.0e30;
            for (var j: u32 = 0u; j < {vertex_count}u; j = j + 1u) {{
                let dv = p - {vertices_name}[j];
                let d2 = dot(dv, dv);
                if (d2 < best_d2) {{ best_d2 = d2; }}
            }}
            return sqrt(best_d2);
        }}
    }}
    return length(p - q);
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polytope_extended_sdfs_wgsl_validates() {
        let wgsl = polytope_extended_sdfs_wgsl();
        let probe = format!(
            "{wgsl}\n\
             @compute @workgroup_size(1) fn main() {{\n\
             let p = vec4<f32>(0.5, 0.0, 0.0, 0.0);\n\
             _ = cell120_sdf_local(p);\n\
             _ = cell600_sdf_local(p);\n\
             }}\n"
        );
        let module = naga::front::wgsl::parse_str(&probe)
            .unwrap_or_else(|e| panic!("WGSL parse failed: {e}\n--- source ---\n{probe}"));
        let flags = naga::valid::ValidationFlags::all();
        let caps = naga::valid::Capabilities::empty();
        naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .expect("polytope WGSL should validate");
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum WolfeBranch {
        Inside,
        OutsideCircumsphere,
        ActiveSetProjection,
        ClosestVertex,
    }

    // Mirrors the emitted WGSL, not the CPU `polytope_sdf_wolfe`, which inverts 4x4.
    fn wolfe_sdf_wgsl_mirror(
        p: Vec4,
        face_normals: &[Vec4],
        vertices: &[Vec4],
        inradius: f32,
    ) -> (f32, WolfeBranch) {
        let mut max_d = f32::NEG_INFINITY;
        let mut max_i = 0usize;
        for (i, n) in face_normals.iter().enumerate() {
            let d = n.dot(p) - inradius;
            if d > max_d {
                max_d = d;
                max_i = i;
            }
        }
        if max_d <= 0.0 {
            return (max_d, WolfeBranch::Inside);
        }
        if p.dot(p) > 1.0 {
            return (max_d, WolfeBranch::OutsideCircumsphere);
        }

        let mut active_idx = [max_i, 0, 0];
        let mut active = [face_normals[max_i], Vec4::ZERO, Vec4::ZERO];
        let mut active_count = 1usize;
        let mut q = p - max_d * active[0];
        let tol = 1.0e-6_f32;

        for _ in 0..3 {
            let mut next_d = tol;
            let mut next_i = usize::MAX;
            for (i, n) in face_normals.iter().enumerate() {
                if i == active_idx[0]
                    || (active_count >= 2 && i == active_idx[1])
                    || (active_count >= 3 && i == active_idx[2])
                {
                    continue;
                }
                let d = n.dot(q) - inradius;
                if d > next_d {
                    next_d = d;
                    next_i = i;
                }
            }
            if next_i == usize::MAX {
                return ((p - q).length(), WolfeBranch::ActiveSetProjection);
            }
            if active_count < 3 {
                active_idx[active_count] = next_i;
                active[active_count] = face_normals[next_i];
                active_count += 1;
                q = mirror_project_active(p, &active, active_count as u32, inradius);
            } else {
                let best = vertices
                    .iter()
                    .map(|v| (p - *v).length_squared())
                    .fold(f32::INFINITY, f32::min);
                return (best.sqrt(), WolfeBranch::ClosestVertex);
            }
        }
        ((p - q).length(), WolfeBranch::ActiveSetProjection)
    }

    fn mirror_project_active(p: Vec4, a: &[Vec4; 3], count: u32, inradius: f32) -> Vec4 {
        let b0 = a[0].dot(p) - inradius;
        let b1 = a[1].dot(p) - inradius;
        let b2 = a[2].dot(p) - inradius;
        if count == 1 {
            return p - b0 * a[0];
        }
        if count == 2 {
            let g01 = a[0].dot(a[1]);
            let det = 1.0 - g01 * g01;
            if det.abs() < 1.0e-9 {
                return p;
            }
            let inv_det = 1.0 / det;
            return p - ((b0 - g01 * b1) * inv_det) * a[0] - ((b1 - g01 * b0) * inv_det) * a[1];
        }
        let g01 = a[0].dot(a[1]);
        let g02 = a[0].dot(a[2]);
        let g12 = a[1].dot(a[2]);
        let det = 1.0 + 2.0 * g01 * g02 * g12 - g01 * g01 - g02 * g02 - g12 * g12;
        if det.abs() < 1.0e-9 {
            return p;
        }
        let inv_det = 1.0 / det;
        let c00 = 1.0 - g12 * g12;
        let c01 = g02 * g12 - g01;
        let c02 = g01 * g12 - g02;
        let c11 = 1.0 - g02 * g02;
        let c12 = g01 * g02 - g12;
        let c22 = 1.0 - g01 * g01;
        let l0 = (c00 * b0 + c01 * b1 + c02 * b2) * inv_det;
        let l1 = (c01 * b0 + c11 * b1 + c12 * b2) * inv_det;
        let l2 = (c02 * b0 + c12 * b1 + c22 * b2) * inv_det;
        p - l0 * a[0] - l1 * a[1] - l2 * a[2]
    }

    // Dykstra's alternating projection (Boyle and Dykstra 1986) reaches a point
    // inside P, so `|p - x|` bounds `dist(p, P)` from above.
    fn dykstra_distance_upper_bound(
        p: Vec4,
        face_normals: &[Vec4],
        inradius: f32,
        sweeps: usize,
    ) -> f32 {
        let mut x = p;
        let mut correction = vec![Vec4::ZERO; face_normals.len()];
        for _ in 0..sweeps {
            for (i, n) in face_normals.iter().enumerate() {
                let y = x - correction[i];
                let d = n.dot(y) - inradius;
                let projected = if d > 0.0 { y - d * *n } else { y };
                correction[i] = projected - y;
                x = projected;
            }
        }
        for _ in 0..64 {
            let mut worst = f32::NEG_INFINITY;
            for n in face_normals {
                let d = n.dot(x) - inradius;
                if d > 0.0 {
                    x -= d * *n;
                }
                worst = worst.max(d);
            }
            if worst <= 1.0e-7 {
                break;
            }
        }
        (p - x).length()
    }

    // Knuth MMIX LCG.
    fn lcg_signed_unit(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 33) as f32) / ((1u64 << 31) as f32) - 1.0
    }

    fn lcg_vec4(state: &mut u64) -> Vec4 {
        Vec4::new(
            lcg_signed_unit(state),
            lcg_signed_unit(state),
            lcg_signed_unit(state),
            lcg_signed_unit(state),
        )
    }

    #[test]
    fn emitted_wolfe_sdf_never_exceeds_the_true_distance() {
        for (name, (normals, inradius), vertices) in [
            ("cell120", cell120_face_planes(), cell120_vertices(1.0)),
            ("cell600", cell600_face_planes(), cell600_vertices(1.0)),
        ] {
            let mut state = 0x5EED_1234_u64;
            let mut vertex_branch_hits = 0u32;
            let mut checked = 0u32;
            let mut loosest = 1.0_f32;
            for sample in 0..192u32 {
                let p = if sample % 2 == 0 {
                    let dir = lcg_vec4(&mut state);
                    if dir.length_squared() < 1.0e-6 {
                        continue;
                    }
                    let radial = 0.5 * (lcg_signed_unit(&mut state) + 1.0);
                    dir.normalize() * (inradius + (1.0 - inradius) * radial)
                } else {
                    let jitter = [0.0005_f32, 0.005, 0.02, 0.08][(sample / 2) as usize % 4];
                    let v = vertices[(state >> 20) as usize % vertices.len()];
                    let dir = v + lcg_vec4(&mut state) * jitter;
                    if dir.length_squared() < 1.0e-6 {
                        continue;
                    }
                    let inset = jitter * 0.5 * (lcg_signed_unit(&mut state) + 1.0);
                    dir.normalize() * (1.0 - inset)
                };

                let (d, branch) = wolfe_sdf_wgsl_mirror(p, &normals, &vertices, inradius);
                if branch == WolfeBranch::ClosestVertex {
                    vertex_branch_hits += 1;
                }
                if branch == WolfeBranch::Inside || d <= 0.0 {
                    continue;
                }
                let upper = dykstra_distance_upper_bound(p, &normals, inradius, 40);
                assert!(
                    d <= upper + 1.0e-4,
                    "{name}: {branch:?} reported {d} at {p:?}, but a point inside \
                     the polytope sits {upper} away; a sphere trace stepping {d} \
                     would pass through the surface",
                );
                if upper > 1.0e-4 {
                    loosest = loosest.min(d / upper);
                }
                checked += 1;
            }
            assert!(
                checked > 32,
                "{name}: only {checked} samples landed outside the polytope; the \
                 sweep is not exercising the Wolfe iteration",
            );
            println!(
                "{name}: {checked} exterior samples, tightest-to-loosest ratio \
                 {loosest:.4}, closest-vertex branch taken {vertex_branch_hits} times"
            );
        }
    }

    #[test]
    fn polytope_stub_sdfs_wgsl_validates() {
        let wgsl = polytope_stub_sdfs_wgsl();
        let probe = format!(
            "{wgsl}\n\
             @compute @workgroup_size(1) fn main() {{\n\
             let p = vec4<f32>(0.5, 0.0, 0.0, 0.0);\n\
             _ = cell120_sdf_local(p);\n\
             _ = cell600_sdf_local(p);\n\
             }}\n"
        );
        let module = naga::front::wgsl::parse_str(&probe).expect("stub must parse");
        let flags = naga::valid::ValidationFlags::all();
        let caps = naga::valid::Capabilities::empty();
        naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .expect("stub must validate");
    }
}
