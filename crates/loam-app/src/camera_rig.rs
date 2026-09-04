use anyhow::{anyhow, Result};
use loam_egui::Console;

use crate::freecam::{CursorMode, Freecam};

/// Slowest and fastest freecam that is still steerable at 60 Hz.
const SPEED_RANGE: std::ops::RangeInclusive<f32> = 0.1..=200.0;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CameraMode {
    /// Orbits its target on the RIGHT button; the left belongs to the scene.
    #[default]
    Orbit,
    /// Captures the cursor, so it owns every button while it is flying.
    FreeRoam,
}

impl CameraMode {
    pub fn label(self) -> &'static str {
        match self {
            CameraMode::Orbit => "orbit",
            CameraMode::FreeRoam => "freecam",
        }
    }
}

#[derive(Debug, Default)]
pub struct CameraRig {
    pub mode: CameraMode,
    pub freecam: Freecam,
}

impl CameraRig {
    pub fn is_flying(&self) -> bool {
        self.mode == CameraMode::FreeRoam
    }
}

pub fn register_camera_command<Ctx: 'static>(
    console: &mut Console<Ctx>,
    reach: fn(&mut Ctx) -> &mut CameraRig,
) {
    console.register(
        loam_egui::cmd::<Ctx, _>(
            "camera",
            "camera mode: orbit | freecam; bare cycles. `camera speed <N>` and              `camera cursor <hold|toggle>` tune the freecam",
            move |args, ctx, out| {
                let rig = reach(ctx);
                match args.first().copied() {
                    None => {
                        rig.mode = match rig.mode {
                            CameraMode::Orbit => CameraMode::FreeRoam,
                            CameraMode::FreeRoam => CameraMode::Orbit,
                        };
                    }
                    Some("orbit") => rig.mode = CameraMode::Orbit,
                    Some("freecam") => rig.mode = CameraMode::FreeRoam,
                    Some("speed") => return set_speed(rig, args.get(1).copied(), out),
                    Some("cursor") => return set_cursor(rig, args.get(1).copied(), out),
                    Some(other) => {
                        return Err(anyhow!(
                            "camera: unknown arg `{other}` (try orbit|freecam|speed|cursor)"
                        ));
                    }
                }
                out.line(format!("camera: {}", rig.mode.label()));
                Ok(())
            },
        )
        .with_args(&[&["orbit", "freecam", "speed", "cursor"]]),
    );
}

fn set_speed(
    rig: &mut CameraRig,
    value: Option<&str>,
    out: &mut loam_egui::ConsoleWriter,
) -> Result<()> {
    let Some(value) = value else {
        out.line(format!("camera speed: {:.2} u/sec", rig.freecam.speed));
        return Ok(());
    };
    let parsed: f32 = value
        .parse()
        .map_err(|e| anyhow!("camera speed: invalid `{value}`: {e}"))?;
    if !SPEED_RANGE.contains(&parsed) {
        return Err(anyhow!(
            "camera speed {parsed} out of range; expected {:?}",
            SPEED_RANGE
        ));
    }
    rig.freecam.speed = parsed;
    out.line(format!("camera speed: set to {parsed:.2} u/sec"));
    Ok(())
}

fn set_cursor(
    rig: &mut CameraRig,
    value: Option<&str>,
    out: &mut loam_egui::ConsoleWriter,
) -> Result<()> {
    let mode = match value {
        None => {
            out.line(format!("camera cursor: {:?}", rig.freecam.cursor_mode()));
            return Ok(());
        }
        Some("hold") => CursorMode::Hold,
        Some("toggle") => CursorMode::Toggle,
        Some(other) => {
            return Err(anyhow!(
                "camera cursor: unknown `{other}` (try hold|toggle)"
            ));
        }
    };
    rig.freecam.set_cursor_mode(mode);
    out.line(format!("camera cursor: {mode:?}"));
    Ok(())
}
