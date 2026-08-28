//! Ground colours and fog density for every scene that records
//! [`loam_render::SkyGroundNode`], plus the `ground` console verb that edits
//! them live. They are art parameters: a rebuild is too slow a loop to judge
//! them in.

use anyhow::{anyhow, Result};
use loam_egui::Console;
use loam_render::sky_ground::{DEFAULT_FOG_PER_UNIT, GROUND_DARK_GREY, GROUND_LIGHT_GREY};
use loam_render::Ground;

// A density above this blends the checker into the sky inside one body length,
// which reads as a bug rather than a setting. The floor is exactly zero, which
// turns the blend off.
const MAX_FOG_PER_UNIT: f32 = 1.0;

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct Environment {
    pub(crate) dark: [f32; 3],
    pub(crate) light: [f32; 3],
    pub(crate) fog_per_unit: f32,
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            dark: GROUND_DARK_GREY,
            light: GROUND_LIGHT_GREY,
            fog_per_unit: DEFAULT_FOG_PER_UNIT,
        }
    }
}

impl Environment {
    pub(crate) fn ground(&self, y: f32, visible: bool) -> Ground {
        Ground {
            y,
            dark: self.dark,
            light: self.light,
            fog_per_unit: self.fog_per_unit,
            visible,
        }
    }

    /// Bare-read text for one field, or for all three when `field` is `None`.
    pub(crate) fn report(&self, field: Option<&str>) -> String {
        match field {
            Some("dark") => format!("ground dark: {}", rgb_text(self.dark)),
            Some("light") => format!("ground light: {}", rgb_text(self.light)),
            Some("fog") => format!(
                "ground fog: {:.4} per unit (half sky at {:.1} units)",
                self.fog_per_unit,
                half_blend_distance(self.fog_per_unit)
            ),
            _ => format!(
                "ground dark {} light {} fog {:.4} (half sky at {:.1} units)",
                rgb_text(self.dark),
                rgb_text(self.light),
                self.fog_per_unit,
                half_blend_distance(self.fog_per_unit)
            ),
        }
    }

    /// Runs one `ground` line. `args` is the verb's argument list, so an empty
    /// slice is the bare read.
    pub(crate) fn apply(&mut self, args: &[&str]) -> Result<String> {
        let Some(field) = args.first().copied() else {
            return Ok(self.report(None));
        };
        match field {
            "reset" => {
                *self = Self::default();
                Ok(format!("ground: reset ({})", self.report(None)))
            }
            "dark" | "light" => {
                if args.len() == 1 {
                    return Ok(self.report(Some(field)));
                }
                let rgb = parse_rgb(field, &args[1..])?;
                if field == "dark" {
                    self.dark = rgb;
                } else {
                    self.light = rgb;
                }
                Ok(format!("ground {field}: set to {}", rgb_text(rgb)))
            }
            "fog" => {
                if args.len() == 1 {
                    return Ok(self.report(Some(field)));
                }
                self.fog_per_unit = parse_fog(args[1])?;
                Ok(format!("ground fog: set to {}", self.report(Some("fog"))))
            }
            other => Err(anyhow!(
                "ground: unknown field `{other}` (try dark|light|fog|reset)"
            )),
        }
    }
}

/// Distance at which the sky is half mixed into the ground, from
/// `1 − exp(−t·density) = 1/2`.
pub(crate) fn half_blend_distance(fog_per_unit: f32) -> f32 {
    if fog_per_unit <= 0.0 {
        return f32::INFINITY;
    }
    std::f32::consts::LN_2 / fog_per_unit
}

fn rgb_text(rgb: [f32; 3]) -> String {
    format!("{:.3} {:.3} {:.3}", rgb[0], rgb[1], rgb[2])
}

fn parse_rgb(field: &str, args: &[&str]) -> Result<[f32; 3]> {
    let [r, g, b] = args else {
        return Err(anyhow!(
            "usage: ground {field} <r> <g> <b>, each a float in [0, 1]"
        ));
    };
    let mut rgb = [0.0_f32; 3];
    for (slot, token) in rgb.iter_mut().zip([r, g, b]) {
        let value: f32 = token
            .parse()
            .map_err(|e| anyhow!("ground {field}: invalid channel `{token}`: {e}"))?;
        if !(value.is_finite() && (0.0..=1.0).contains(&value)) {
            return Err(anyhow!(
                "ground {field}: channel {value} out of range; expected [0, 1]"
            ));
        }
        *slot = value;
    }
    Ok(rgb)
}

fn parse_fog(token: &str) -> Result<f32> {
    let value: f32 = token
        .parse()
        .map_err(|e| anyhow!("ground fog: invalid density `{token}`: {e}"))?;
    if !(value.is_finite() && (0.0..=MAX_FOG_PER_UNIT).contains(&value)) {
        return Err(anyhow!(
            "ground fog: density {value} out of range; expected [0, {MAX_FOG_PER_UNIT}]"
        ));
    }
    Ok(value)
}

/// Registers `ground` on a scene console. `reach` names where that scene keeps
/// its [`Environment`]; a plain `fn` pointer so the same registration serves
/// consoles over four different context types.
pub(crate) fn register_ground_command<Ctx: 'static>(
    console: &mut Console<Ctx>,
    reach: fn(&mut Ctx) -> &mut Environment,
) {
    console.register(
        loam_egui::cmd::<Ctx, _>(
            "ground",
            "background checker colours and fog density (bare reads all three)",
            move |args, ctx, out| {
                let line = reach(ctx).apply(args)?;
                out.line(line);
                Ok(())
            },
        )
        .with_args(&[&["dark", "light", "fog", "reset"]])
        .with_long_help(
            "ground                     read all three\n\
             ground dark                read the dark checker colour\n\
             ground dark <r> <g> <b>    set it, each channel in [0, 1]\n\
             ground light <r> <g> <b>   the light checker colour\n\
             ground fog                 read the density\n\
             ground fog <density>       sky blended per world unit, in [0, 1];\n\
             \x20                          half the sky is mixed in at ln2/density\n\
             ground reset               back to the shipped defaults",
        ),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_field_reads_and_an_argument_sets() {
        let mut env = Environment::default();
        let read = env.apply(&["fog"]).expect("bare read");
        assert!(read.contains("0.0200"), "bare fog read said `{read}`");
        assert_eq!(env, Environment::default(), "a bare read wrote a value");

        env.apply(&["fog", "0.1"]).expect("set");
        assert_eq!(env.fog_per_unit, 0.1);
        assert_eq!(env.dark, Environment::default().dark);

        env.apply(&["dark", "0.5", "0.25", "0.125"]).expect("set");
        assert_eq!(env.dark, [0.5, 0.25, 0.125]);
        env.apply(&["light", "1", "0", "0.5"]).expect("set");
        assert_eq!(env.light, [1.0, 0.0, 0.5]);
        assert_eq!(env.fog_per_unit, 0.1, "a colour write moved the density");

        env.apply(&["reset"]).expect("reset");
        assert_eq!(env, Environment::default());
    }

    #[test]
    fn the_parse_rejects_out_of_range_and_malformed_arguments() {
        let mut env = Environment::default();
        for line in [
            vec!["fog", "-0.1"],
            vec!["fog", "2.0"],
            vec!["fog", "nan"],
            vec!["fog", "wide"],
            vec!["dark", "0.5"],
            vec!["dark", "0.5", "0.5"],
            vec!["dark", "0.5", "0.5", "0.5", "0.5"],
            vec!["dark", "0.5", "0.5", "1.5"],
            vec!["dark", "0.5", "0.5", "red"],
            vec!["sky"],
        ] {
            assert!(
                env.apply(&line).is_err(),
                "`ground {}` was accepted",
                line.join(" ")
            );
        }
        assert_eq!(env, Environment::default(), "a rejected line still wrote");
    }

    #[test]
    fn the_environment_reaches_the_uniform_the_shader_reads() {
        let mut env = Environment::default();
        env.apply(&["fog", "0.07"]).expect("set");
        env.apply(&["dark", "0.1", "0.2", "0.3"]).expect("set");
        let ground = env.ground(-1.5, false);
        assert_eq!(ground.fog_per_unit, 0.07);
        assert_eq!(ground.dark, [0.1, 0.2, 0.3]);
        assert_eq!(ground.y, -1.5);
        assert!(!ground.visible);

        let uniforms = loam_render::SkyGroundUniforms::new(
            glam::Mat4::IDENTITY,
            loam_render::Viewport::full([4, 4]),
            ground,
        );
        assert_eq!(uniforms.fog_per_unit, 0.07);
        assert_eq!(uniforms.ground_dark, [0.1, 0.2, 0.3]);
        assert_eq!(uniforms.show_ground, 0.0);
    }

    #[test]
    fn the_half_blend_distance_is_where_the_sky_is_half_mixed_in() {
        for density in [0.005_f32, 0.02, 0.05, 0.5] {
            let t = half_blend_distance(density);
            let fog = 1.0 - (-t * density).exp();
            assert!((fog - 0.5).abs() < 1e-6, "density {density} blended {fog}");
        }
        assert_eq!(half_blend_distance(0.0), f32::INFINITY);
    }

    #[test]
    fn the_verb_registers_on_a_console_and_dispatches_to_the_scene_state() {
        let mut console = Console::<Environment>::new();
        register_ground_command(&mut console, |env| env);
        assert!(console.has_command("ground"));

        let mut env = Environment::default();
        console.dispatch("ground", &["fog", "0.03"], &mut env);
        assert_eq!(env.fog_per_unit, 0.03, "the verb never reached the state");
        console.dispatch("ground", &["fog"], &mut env);
        assert_eq!(env.fog_per_unit, 0.03, "the bare read wrote");
    }

    #[test]
    fn the_field_names_tab_complete() {
        let mut console = Console::<Environment>::new();
        register_ground_command(&mut console, |env| env);
        *console.input_mut() = "ground ".to_string();
        let mut seen = Vec::new();
        for _ in 0..4 {
            console.tab_complete();
            seen.push(
                console
                    .input()
                    .trim_start_matches("ground ")
                    .trim()
                    .to_string(),
            );
        }
        for field in ["dark", "fog", "light", "reset"] {
            assert!(
                seen.contains(&field.to_string()),
                "tab from `ground ` produced {seen:?}, missing {field}"
            );
        }
    }
}
