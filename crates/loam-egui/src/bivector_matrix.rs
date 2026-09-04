//! Antisymmetric 4×4 `M_ij = B_ij = -M_ji`, upper triangle `xy xz xw / yz yw zw`
//! in the `e_i ∧ e_j` basis (Hestenes & Sobczyk, Clifford Algebra to Geometric
//! Calculus, ch. 1).

use egui::{Grid, Label, Response, RichText, Ui};
use loam_math::{Bivector4, Plane4};

/// Cells are degrees per unit time, `+5.1` format.
pub fn bivector_matrix(ui: &mut Ui, b: &Bivector4) -> Response {
    Grid::new("loam_egui_bivector_matrix")
        .num_columns(5)
        .spacing([8.0, 2.0])
        .show(ui, |ui| {
            ui.label("");
            for axis in AXIS {
                ui.add(Label::new(RichText::new(axis).monospace().weak()));
            }
            ui.end_row();
            for (row, row_axis) in AXIS.iter().enumerate() {
                ui.add(Label::new(RichText::new(*row_axis).monospace().weak()));
                for col in 0..4 {
                    let text = cell_text(b, row, col);
                    ui.add(Label::new(RichText::new(text).monospace()));
                }
                ui.end_row();
            }
        })
        .response
}

const AXIS: [&str; 4] = ["x", "y", "z", "w"];

pub fn cell_text(b: &Bivector4, row: usize, col: usize) -> String {
    if row == col {
        "0".to_string()
    } else if row < col {
        format!("{:>+5.1}", upper_pair(b, row, col).to_degrees())
    } else {
        format!("{:>+5.1}", -upper_pair(b, col, row).to_degrees())
    }
}

fn upper_pair(b: &Bivector4, row: usize, col: usize) -> f32 {
    let plane = match (row, col) {
        (0, 1) => Plane4::Xy,
        (0, 2) => Plane4::Xz,
        (0, 3) => Plane4::Xw,
        (1, 2) => Plane4::Yz,
        (1, 3) => Plane4::Yw,
        (2, 3) => Plane4::Zw,
        _ => unreachable!("upper_pair expects row < col, got ({row}, {col})"),
    };
    b.component(plane)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pure(plane: Plane4) -> Bivector4 {
        let mut b = Bivector4::ZERO;
        b.set_component(plane, 1.0);
        b
    }

    #[test]
    fn cell_text_diagonal_is_zero() {
        let b = Bivector4::ZERO;
        for i in 0..4 {
            assert_eq!(cell_text(&b, i, i), "0");
        }
    }

    #[test]
    fn cell_text_pure_xy_plane() {
        let b = pure(Plane4::Xy);
        assert_eq!(cell_text(&b, 0, 1), "+57.3");
        assert_eq!(cell_text(&b, 1, 0), "-57.3");
        assert_eq!(cell_text(&b, 0, 2), " +0.0");
        assert_eq!(cell_text(&b, 2, 3), " +0.0");
    }

    #[test]
    fn cell_text_each_basis_plane_lights_correct_cell() {
        let cases = [
            (Plane4::Xy, (0, 1)),
            (Plane4::Xz, (0, 2)),
            (Plane4::Xw, (0, 3)),
            (Plane4::Yz, (1, 2)),
            (Plane4::Yw, (1, 3)),
            (Plane4::Zw, (2, 3)),
        ];
        for (plane, (row, col)) in cases {
            let b = pure(plane);
            assert_eq!(
                cell_text(&b, row, col),
                "+57.3",
                "plane {plane:?} should populate ({row}, {col})",
            );
            assert_eq!(
                cell_text(&b, col, row),
                "-57.3",
                "plane {plane:?} mirror ({col}, {row}) should be negated",
            );
        }
    }
}
