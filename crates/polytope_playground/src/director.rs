//! [`loam_time::Director`] arbitrates per channel and this module resolves its
//! channels onto the row. A timeline addresses a body by slot index because
//! the row is positional and can hold the same polytope twice, so a shape name
//! would not be a name at all.
//!
//! The arbitration is the point. The director's playhead is an integer frame
//! index and the UI spin's `rot_time` is a wall-clock accumulator; a slot
//! written by both is the defect the timeline exists to prevent, so
//! [`step_row_rotation`] is the single place either one writes a rotor.

use anyhow::{anyhow, Result};
use loam_math::{Bivector, Bivector4};
use loam_time::{Director, Drive};

use crate::spins::{is_directed, SlotSpins};
use crate::state::RotationMode;

const SLOT_PREFIX: &str = "slot";

#[derive(Debug)]
pub(crate) struct Playback {
    director: Director,
    slots: Vec<usize>,
    /// `directed[slot]`: the timeline gives this slot an orientation track, so
    /// the director writes its rotor for the whole run and the UI spin never
    /// does. Ownership does not lapse when the playhead runs out or pauses;
    /// holding a pose is what makes a scrub readable.
    directed: Vec<bool>,
}

impl Playback {
    /// Bind `director` to a row of `slots` bodies, refusing anything the row
    /// cannot host. Every rejection is an authoring fault the caller fixes by
    /// editing the file, which is why they land here and not mid-playback.
    pub(crate) fn new(director: Director, slots: usize) -> Result<Self> {
        let mut bound = Vec::with_capacity(director.timeline().bodies.len());
        let mut directed = vec![false; slots];
        for body in &director.timeline().bodies {
            if body.position.is_some() {
                return Err(anyhow!(
                    "timeline body `{}` has a position track, and nothing here writes a slot's \
                     place: it belongs to the rigid body, so a track writing it would race the \
                     solver",
                    body.name
                ));
            }
            let slot = slot_index(&body.name).ok_or_else(|| {
                anyhow!(
                    "timeline body `{}` does not name a row slot; expected `{SLOT_PREFIX}<index>`",
                    body.name
                )
            })?;
            if slot >= slots {
                return Err(anyhow!(
                    "timeline body `{}` names slot {slot} of a {slots}-slot row",
                    body.name
                ));
            }
            bound.push(slot);
            directed[slot] = body.orientation.is_some();
        }
        Ok(Self {
            director,
            slots: bound,
            directed,
        })
    }

    pub(crate) fn directed(&self) -> &[bool] {
        &self.directed
    }

    pub(crate) fn owns_w_slice(&self) -> bool {
        matches!(self.director.w_slice(), Drive::Directed(_))
    }

    pub(crate) fn rewind(&mut self) {
        self.director.seek(0);
    }

    fn write_orientations(&self, spins: &mut SlotSpins) {
        for (body, &slot) in self.director.bodies().zip(&self.slots) {
            if let Drive::Directed(rotor) = self.director.orientation(body) {
                spins.set_rotor(slot, rotor);
            }
        }
    }
}

fn slot_index(name: &str) -> Option<usize> {
    name.strip_prefix(SLOT_PREFIX)?.parse().ok()
}

/// One frame of the row's rotation channels, with exactly one writer per slot.
///
/// The director advances a single frame per call and reads no wall-clock
/// delta, so a directed slot's pose is a function of the frame index alone and
/// a capture of it is re-shootable. Slots the timeline does not name stay on
/// `rot_time`, and `rot_time` stops advancing once no slot reads it: a clock
/// still running behind a timeline that owns the whole row is the second
/// writer this arbitration exists to remove.
///
/// `dt_animation` is already `dt * rate_scale`, and zero while the spin is
/// paused, so the pause and a zero angular velocity take the same path.
pub(crate) fn step_row_rotation(
    playback: Option<&mut Playback>,
    spins: &mut SlotSpins,
    w_slice: &mut f32,
    rot_time: &mut f32,
    dt_animation: f32,
    mode: RotationMode,
    omega: Bivector4,
) {
    let directed: &[bool] = match playback {
        Some(playback) => {
            playback.director.advance();
            if let Drive::Directed(w) = playback.director.w_slice() {
                *w_slice = w;
            }
            playback.write_orientations(spins);
            playback.directed()
        }
        None => &[],
    };
    if spins.any_unowned(directed) {
        *rot_time += dt_animation;
    }
    match mode {
        RotationMode::Active => spins.recompose_active(*rot_time, directed),
        RotationMode::Composer => {
            let step = omega * dt_animation;
            // Composer integrates the selected slot and no other, so an
            // unowned selection is the whole gate.
            if step.magnitude_squared() > 0.0 && !is_directed(directed, spins.selected()) {
                let spin = spins.selected_spin_mut();
                spin.rotor = (step.exp() * spin.rotor).normalize();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_math::{Plane4, Rotor4};
    use loam_time::director::{BodyTrack, Ease, Timeline, Track};

    const FPS: u32 = 60;

    fn quarter_turn_xw() -> Rotor4 {
        (Plane4::Xw.unit_bivector() * (std::f32::consts::FRAC_PI_4)).exp()
    }

    fn turn_slots(named: &[usize]) -> Timeline {
        Timeline {
            fps: FPS,
            frames: 61,
            w_slice: None,
            bodies: named
                .iter()
                .map(|slot| BodyTrack {
                    name: format!("{SLOT_PREFIX}{slot}"),
                    position: None,
                    orientation: Some(Track::new().key(0.0, Rotor4::IDENTITY, Ease::Linear).key(
                        1.0,
                        quarter_turn_xw(),
                        Ease::Linear,
                    )),
                })
                .collect(),
        }
    }

    fn playback_over(slots: usize, timeline: Timeline) -> Playback {
        Playback::new(Director::new(timeline).unwrap(), slots).unwrap()
    }

    #[test]
    fn a_directed_slot_answers_to_the_playhead_and_never_to_the_ui_clock() {
        const SLOTS: usize = 3;
        let mut spins = SlotSpins::new(SLOTS);
        let mut playback = playback_over(SLOTS, turn_slots(&[0]));
        let mut reference = Director::new(turn_slots(&[0])).unwrap();

        let mut w_slice = 0.25;
        let mut rot_time = 0.0;
        let mut ui_clock_moved_the_row = false;
        for _ in 0..200 {
            step_row_rotation(
                Some(&mut playback),
                &mut spins,
                &mut w_slice,
                &mut rot_time,
                1.0 / 60.0,
                RotationMode::Active,
                Bivector4::ZERO,
            );
            reference.advance();
            let Drive::Directed(authored) = reference.orientation("slot0") else {
                panic!("the fixture names slot0");
            };
            assert_eq!(
                spins.rotor(0),
                authored,
                "slot 0 left the playhead at frame {}",
                reference.frame()
            );
            assert_eq!(spins.rotor(1), spins.rotor(2));
            ui_clock_moved_the_row |= spins.rotor(1) != Rotor4::IDENTITY;
        }
        assert!(ui_clock_moved_the_row, "the UI spin never advanced");
        assert!(rot_time > 3.0, "the UI clock stalled at {rot_time}");
        assert_eq!(w_slice, 0.25);
    }

    #[test]
    fn a_timeline_naming_every_slot_stops_the_ui_clock() {
        const SLOTS: usize = 2;
        let mut spins = SlotSpins::new(SLOTS);
        let mut playback = playback_over(SLOTS, turn_slots(&[0, 1]));
        let mut w_slice = 0.0;
        let mut rot_time = 0.0;
        for _ in 0..120 {
            step_row_rotation(
                Some(&mut playback),
                &mut spins,
                &mut w_slice,
                &mut rot_time,
                1.0 / 60.0,
                RotationMode::Active,
                Bivector4::ZERO,
            );
        }
        assert_eq!(rot_time, 0.0);
        assert_ne!(spins.rotor(0), Rotor4::IDENTITY, "the playhead stalled too");
    }

    #[test]
    fn composer_does_not_integrate_a_slot_the_timeline_owns() {
        const SLOTS: usize = 2;
        let omega = Plane4::Xy.unit_bivector() * 2.0;
        let mut directed_row = SlotSpins::new(SLOTS);
        let mut playback = playback_over(SLOTS, turn_slots(&[0]));
        let mut free_row = SlotSpins::new(SLOTS);
        let (mut w_slice, mut rot_time) = (0.0, 0.0);
        for _ in 0..60 {
            step_row_rotation(
                Some(&mut playback),
                &mut directed_row,
                &mut w_slice,
                &mut rot_time,
                1.0 / 60.0,
                RotationMode::Composer,
                omega,
            );
            step_row_rotation(
                None,
                &mut free_row,
                &mut w_slice,
                &mut rot_time,
                1.0 / 60.0,
                RotationMode::Composer,
                omega,
            );
        }
        assert_ne!(free_row.rotor(0), Rotor4::IDENTITY);
        assert_eq!(directed_row.rotor(0), quarter_turn_xw());
    }

    #[test]
    fn the_slice_follows_the_timeline_only_where_the_timeline_names_it() {
        let mut spins = SlotSpins::new(1);
        let mut named = playback_over(
            1,
            Timeline {
                fps: FPS,
                frames: 61,
                w_slice: Some(Track::new().key(0.0, -1.0, Ease::Linear).key(
                    1.0,
                    1.0,
                    Ease::Linear,
                )),
                bodies: Vec::new(),
            },
        );
        assert!(named.owns_w_slice());
        let mut w_slice = 0.4;
        let mut rot_time = 0.0;
        step_row_rotation(
            Some(&mut named),
            &mut spins,
            &mut w_slice,
            &mut rot_time,
            0.0,
            RotationMode::Active,
            Bivector4::ZERO,
        );
        assert!(
            w_slice < -0.9,
            "slice did not reach the timeline: {w_slice}"
        );

        let mut silent = playback_over(1, turn_slots(&[0]));
        assert!(!silent.owns_w_slice());
        w_slice = 0.4;
        step_row_rotation(
            Some(&mut silent),
            &mut spins,
            &mut w_slice,
            &mut rot_time,
            0.0,
            RotationMode::Active,
            Bivector4::ZERO,
        );
        assert_eq!(w_slice, 0.4);
    }

    #[test]
    fn a_body_the_row_cannot_host_is_refused_at_load() {
        let named = |name: &str| Timeline {
            fps: FPS,
            frames: 61,
            w_slice: None,
            bodies: vec![BodyTrack {
                name: name.to_owned(),
                position: None,
                orientation: Some(Track::new().key(0.0, Rotor4::IDENTITY, Ease::Linear)),
            }],
        };
        for name in ["tesseract", "slot", "slotx", "0"] {
            let director = Director::new(named(name)).unwrap();
            let error = Playback::new(director, 4).expect_err("not a slot name");
            assert!(
                format!("{error:#}").contains("row slot"),
                "{name}: {error:#}"
            );
        }
        let director = Director::new(named("slot4")).unwrap();
        let error = Playback::new(director, 4).expect_err("slot 4 of a 4-slot row");
        assert!(format!("{error:#}").contains("4-slot row"), "{error:#}");
    }

    #[test]
    fn a_position_track_is_refused_because_nothing_writes_a_slots_place() {
        let director = Director::new(Timeline {
            fps: FPS,
            frames: 61,
            w_slice: None,
            bodies: vec![BodyTrack {
                name: "slot0".to_owned(),
                position: Some(
                    Track::new()
                        .key(0.0, glam::Vec4::W * -4.0, Ease::Linear)
                        .key(1.0, glam::Vec4::ZERO, Ease::Linear),
                ),
                orientation: None,
            }],
        })
        .unwrap();
        let error = Playback::new(director, 4).expect_err("no writer for a position track");
        assert!(format!("{error:#}").contains("position track"), "{error:#}");
    }

    #[test]
    fn naming_a_slot_without_an_orientation_track_leaves_its_rotor_alone() {
        let director = Director::new(Timeline {
            fps: FPS,
            frames: 61,
            w_slice: Some(Track::new().key(0.0, 0.0, Ease::Linear)),
            bodies: vec![BodyTrack {
                name: "slot1".to_owned(),
                position: None,
                orientation: None,
            }],
        })
        .unwrap();
        let playback = Playback::new(director, 3).unwrap();
        assert_eq!(playback.directed(), [false, false, false]);
    }
}
