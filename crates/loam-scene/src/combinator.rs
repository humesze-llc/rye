//! Callers pass WGSL `f32` expressions, ideally `let`-bound variable names
//! rather than calls, since each operand appears twice in the emitted text.

use crate::literal::wgsl_f32;

pub fn union_expr(da: &str, db: &str) -> String {
    format!("min({da}, {db})")
}

pub fn intersection_expr(da: &str, db: &str) -> String {
    format!("max({da}, {db})")
}

/// A − B: carves B from A.
pub fn difference_expr(da: &str, db: &str) -> String {
    format!("max({da}, -({db}))")
}

/// Quilez polynomial smooth-minimum; `k` is the blend radius in Space distance
/// units. It is baked as a shortest-round-trip literal, never at fixed
/// precision: it is a divisor, and a six-decimal print collapses every `k`
/// below 5e-7 to `0.000000`, dividing by zero on the GPU while a CPU evaluation
/// of the same scene stays finite.
pub fn smooth_min_fn(name: &str, k: f32) -> String {
    let k = wgsl_f32(k);
    format!(
        "fn {name}(a: f32, b: f32) -> f32 {{\n\
         \tlet h = clamp(0.5 + 0.5 * (b - a) / ({k}), 0.0, 1.0);\n\
         \treturn mix(b, a, h) - ({k}) * h * (1.0 - h);\n\
         }}\n",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn divisor_literal(src: &str) -> &str {
        let after = src.split("(b - a) / (").nth(1).expect("divide is emitted");
        after.split(')').next().expect("divisor is parenthesized")
    }

    #[test]
    fn smooth_min_divisor_round_trips_to_k() {
        for k in [0.08_f32, 4.9e-7, 1e-7, 1e-20, 1e-30] {
            let src = smooth_min_fn("smin", k);
            let literal = divisor_literal(&src);
            let parsed: f32 = literal.parse().expect("divisor parses as f32");
            assert_eq!(parsed, k, "divisor `{literal}` does not round-trip to {k}");
            assert_ne!(parsed, 0.0, "k = {k} emitted a zero divisor");
        }
    }

    #[test]
    fn smooth_min_bakes_one_literal_for_both_uses_of_k() {
        let src = smooth_min_fn("smin", 1e-7);
        let literal = divisor_literal(&src);
        assert_eq!(src.matches(literal).count(), 2, "emitted:\n{src}");
    }
}
