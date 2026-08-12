//! Per-slot UI rotation: the orientation the rotation controls author for one
//! rendered row slot, held apart from that body's physics orientation and
//! composed with it at [`crate::physics::composed_rotor`].
//!
//! One store per rendered row, indexed by the slot index that
//! [`crate::physics::PlaygroundPhysics`] uses, synced to the row's length
//! at the same choke point ([`crate::state::Demo::rebuild_bodies`]). Every
//! rotation control writes exactly one slot, [`SlotSpins::selected`], which a
//! press on a body picks; the animation path advances all of them, each from
//! its own baseline and plane mask, which is what lets two polychora in one
//! row hold different orientations at the same time.

use loam_math::Rotor4;

use crate::state::{active_plane_angle, compose_active_rotor};

/// Whether `directed` hands `slot` to a writer other than the UI spin. A mask
/// shorter than the row leaves its tail to the spin, which is what a timeline
/// naming only the leading slots means.
pub(crate) fn is_directed(directed: &[bool], slot: usize) -> bool {
    directed.get(slot).copied().unwrap_or(false)
}

/// Planes a fresh slot spins in: xw only, the most characteristic 4D
/// rotation, so the play button shows motion before any checkbox is toggled.
/// Every slot starts here, so the row still boots in unison and stays
/// slice-comparable; divergence is something the user asks for.
const DEFAULT_ACTIVE: [bool; 6] = [false, false, true, false, false, false];

/// One slot's authored rotation.
///
/// Orientation is derived, not stored: `rotor` is a cache of
/// [`compose_active_rotor`] over `base_angles`, `active` and the shared
/// `rot_time`, refreshed by [`SlotSpins::recompose_active`]. Composer mode
/// integrates `rotor` directly instead and leaves the other two alone.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SlotSpin {
    /// Baseline angle per plane in radians. Plane `i`'s displayed angle is
    /// `base_angles[i] + rot_time * RATE * active[i]`.
    pub(crate) base_angles: [f32; 6],
    /// Which of the six planes this slot spins in. See [`loam_math::Plane4::ALL`]
    /// for the index -> plane mapping. Two slots with different masks are two
    /// bodies with different angular velocities.
    pub(crate) active: [bool; 6],
    /// Last composed orientation, and the value every render path reads.
    pub(crate) rotor: Rotor4,
}

impl Default for SlotSpin {
    fn default() -> Self {
        Self {
            base_angles: [0.0; 6],
            active: DEFAULT_ACTIVE,
            rotor: Rotor4::IDENTITY,
        }
    }
}

impl SlotSpin {
    /// Displayed angle for plane `plane_idx` at time `t`.
    pub(crate) fn angle_at(&self, plane_idx: usize, t: f32) -> f32 {
        active_plane_angle(self.base_angles[plane_idx], self.active[plane_idx], t)
    }

    /// Active-mode rotor at time `t`: the ordered product this slot's
    /// baselines and mask compose to. See [`compose_active_rotor`].
    pub(crate) fn active_rotor_at(&self, t: f32) -> Rotor4 {
        compose_active_rotor(&self.base_angles, &self.active, t)
    }

    /// Return to the unrotated pose, keeping the plane mask.
    ///
    /// Zeroing `base_angles` is the load-bearing half: Active recomposes
    /// `rotor` from them on the next frame, so clearing `rotor` alone is undone
    /// before it can draw.
    fn clear_orientation(&mut self) {
        self.base_angles = [0.0; 6];
        self.rotor = Rotor4::IDENTITY;
    }
}

/// The rendered row's rotations, plus which slot the controls are aimed at.
///
/// Never empty: the shape row keeps at least one card, and the rotation UI
/// always needs a subject to write to.
pub(crate) struct SlotSpins {
    slots: Vec<SlotSpin>,
    /// Slot every rotation control writes. Kept in range by [`Self::sync`].
    selected: usize,
}

impl SlotSpins {
    pub(crate) fn new(slots: usize) -> Self {
        Self {
            slots: vec![SlotSpin::default(); slots.max(1)],
            selected: 0,
        }
    }

    /// Every slot under one rotor: the row-wide spin the geometry fixtures
    /// assume when they pin where a body is drawn rather than which body a
    /// control reaches.
    #[cfg(test)]
    pub(crate) fn uniform(slots: usize, rotor: Rotor4) -> Self {
        let mut spins = Self::new(slots);
        for slot in &mut spins.slots {
            slot.rotor = rotor;
        }
        spins
    }

    /// Reconcile with a rendered row of `slots` bodies, preserving every
    /// surviving slot's authored rotation. Resizes rather than rebuilding the
    /// way [`crate::physics::PlaygroundPhysics::respawn`] does: a layout
    /// position is a function of the slot count, an authored rotation is not,
    /// so a row edit must not wipe the rotations set on the shapes that stayed.
    pub(crate) fn sync(&mut self, slots: usize) {
        self.slots.resize_with(slots.max(1), SlotSpin::default);
        self.selected = self.selected.min(self.slots.len() - 1);
    }

    pub(crate) fn rotor(&self, slot: usize) -> Rotor4 {
        self.slots[slot].rotor
    }

    pub(crate) fn selected(&self) -> usize {
        self.selected
    }

    pub(crate) fn selected_spin(&self) -> &SlotSpin {
        &self.slots[self.selected]
    }

    pub(crate) fn selected_spin_mut(&mut self) -> &mut SlotSpin {
        &mut self.slots[self.selected]
    }

    /// Aim the rotation controls at the slot a press picked.
    ///
    /// A ray that entered no body leaves the selection where it was: the
    /// controls must always have a subject, and a press on empty space is a
    /// camera drag, not a request to rotate nothing. Out-of-range slots are
    /// ignored on the same terms [`crate::physics::PlaygroundPhysics::throw`]
    /// ignores them: a row edit can retire the slot a press names.
    pub(crate) fn select_picked(&mut self, picked: Option<usize>) {
        if let Some(slot) = picked.filter(|slot| *slot < self.slots.len()) {
            self.selected = slot;
        }
    }

    /// Recompose each slot's rotor from its own baselines and mask at time
    /// `t`, skipping every slot `directed` hands to another writer. Active
    /// mode's orientation is an absolute function of `t`, so this runs every
    /// frame; two slots whose masks differ diverge here.
    ///
    /// The mask is a parameter rather than a field because it is the whole
    /// suppression: `t` is the UI spin's wall clock, a directed slot is on the
    /// director's frame index, and a slot recomposed here after the director
    /// wrote it is the two-clock defect with extra steps.
    pub(crate) fn recompose_active(&mut self, t: f32, directed: &[bool]) {
        for (slot, spin) in self.slots.iter_mut().enumerate() {
            if !is_directed(directed, slot) {
                spin.rotor = spin.active_rotor_at(t);
            }
        }
    }

    /// Whether any slot is still the UI spin's to write.
    pub(crate) fn any_unowned(&self, directed: &[bool]) -> bool {
        (0..self.slots.len()).any(|slot| !is_directed(directed, slot))
    }

    /// Write one slot's rotor from outside the UI spin. Out-of-range slots are
    /// ignored on the same terms [`Self::select_picked`] ignores them: a row
    /// edit can retire the slot a timeline names.
    pub(crate) fn set_rotor(&mut self, slot: usize, rotor: Rotor4) {
        if let Some(spin) = self.slots.get_mut(slot) {
            spin.rotor = rotor;
        }
    }

    /// Whether the GPU's copy of the row's rotors is stale. The length is part
    /// of the test: a row edit that changed the slot count changed which body
    /// each rotor belongs to, so an elementwise compare alone would let a
    /// stale upload through.
    pub(crate) fn rotors_differ_from(&self, uploaded: &[Rotor4]) -> bool {
        self.slots.len() != uploaded.len()
            || self
                .slots
                .iter()
                .zip(uploaded)
                .any(|(spin, rotor)| spin.rotor != *rotor)
    }

    /// Refill `out` with the row's rotors, keeping its allocation.
    pub(crate) fn record_rotors(&self, out: &mut Vec<Rotor4>) {
        out.clear();
        out.extend(self.slots.iter().map(|spin| spin.rotor));
    }

    /// Return every slot to the unrotated pose, keeping the plane masks and
    /// the selection. The `Edit > Reset orientation` command.
    pub(crate) fn clear_orientation(&mut self) {
        for slot in &mut self.slots {
            slot.clear_orientation();
        }
    }

    /// Full reset: every slot back to the boot rotation, selection back to the
    /// leading slot. The row keeps its length; `reset` is not a row edit.
    pub(crate) fn reset(&mut self) {
        *self = Self::new(self.slots.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::BODY_SIZE;
    use crate::physics::PlaygroundPhysics;
    use crate::state::body_position;
    use glam::{Vec3, Vec4};
    use loam_camera::Ray;
    use loam_math::{Plane4, Rotor};

    /// Angle a rotor turns a probe vector through, as the chord half-angle
    /// `2·asin(|a - b| / 2)` on the unit sphere. The chord form is
    /// well-conditioned near zero where `acos(dot)` loses half its digits
    /// (Kahan, *Mindless Assessments of Roundoff*, 2006, §12). The clamp is
    /// the antipodal guard: a chord of exactly 2 rounds past 1 in f32 and
    /// `asin` would answer NaN, which `f32::max` then swallows.
    fn probe_separation(a: Rotor4, b: Rotor4) -> f32 {
        let probe = Vec4::new(0.5, -0.3, 0.7, 0.4).normalize();
        2.0 * ((a.apply(probe) - b.apply(probe)).length() * 0.5)
            .clamp(-1.0, 1.0)
            .asin()
    }

    /// The node's headline property: two slots given different plane masks are
    /// two bodies with different angular velocities, and their rotors separate
    /// as the shared clock advances. A store that kept one rotor for the row,
    /// or that recomposed every slot from one mask, holds them together here.
    #[test]
    fn slots_with_different_masks_diverge_and_stay_on_the_unit_sphere() {
        let mut spins = SlotSpins::new(2);
        spins.slots[0].active = [false, false, true, false, false, false];
        spins.slots[1].active = [false, false, false, false, false, true];

        const STEPS: usize = 600;
        const DT: f32 = 1.0 / 60.0;
        let mut max_separation = 0.0_f32;
        for step in 0..=STEPS {
            let t = step as f32 * DT;
            spins.recompose_active(t, &[]);
            for slot in 0..2 {
                let norm_squared = spins.rotor(slot).norm_squared();
                assert!(
                    (norm_squared - 1.0).abs() < 1e-5,
                    "slot {slot} left the unit sphere at t={t}: |R|² = {norm_squared}"
                );
            }
            max_separation = max_separation.max(probe_separation(spins.rotor(0), spins.rotor(1)));
        }
        assert!(
            max_separation > 1.0,
            "the two masks never separated a probe by more than {max_separation} rad"
        );
    }

    /// Identical masks stay locked together forever, which is what keeps the
    /// boot row slice-comparable: divergence is a thing the user asks for, not
    /// numerical noise between slots.
    #[test]
    fn slots_with_identical_masks_never_separate() {
        let mut spins = SlotSpins::new(3);
        for step in 0..600 {
            spins.recompose_active(step as f32 / 60.0, &[]);
            assert_eq!(spins.rotor(0), spins.rotor(1));
            assert_eq!(spins.rotor(0), spins.rotor(2));
        }
    }

    /// A directed slot is skipped by the UI clock and keeps whatever its other
    /// writer put there, while its neighbours recompose as usual. A mask
    /// shorter than the row leaves its tail to the clock, so a timeline
    /// authored against a shorter row cannot silently freeze the new slots.
    #[test]
    fn a_directed_slot_is_skipped_by_the_ui_clock_and_its_neighbours_are_not() {
        let mut spins = SlotSpins::new(3);
        let held = compose_active_rotor(&[0.9, 0.0, 0.0, 0.0, 0.0, 0.0], &[false; 6], 0.0);
        spins.set_rotor(1, held);

        spins.recompose_active(2.5, &[false, true]);
        assert_eq!(spins.rotor(1), held, "the directed slot was recomposed");
        assert_ne!(spins.rotor(0), Rotor4::IDENTITY, "slot 0 was skipped");
        assert_eq!(
            spins.rotor(2),
            spins.rotor(0),
            "the mask tail is the clock's"
        );

        assert!(spins.any_unowned(&[false, true]));
        assert!(spins.any_unowned(&[true, true]), "slot 2 is past the mask");
        assert!(!spins.any_unowned(&[true; 3]));
        // Out-of-range writes are dropped, not panics: a row edit can retire
        // the slot a timeline names.
        spins.set_rotor(9, Rotor4::IDENTITY);
    }

    /// A slot with an empty mask holds its baseline while the clock runs, so
    /// "auto-rotate" is per slot and not a row-wide switch: one body can spin
    /// while its neighbour is parked at an angle the user set.
    #[test]
    fn an_empty_mask_parks_one_slot_while_its_neighbour_spins() {
        let mut spins = SlotSpins::new(2);
        spins.slots[0].active = [false; 6];
        spins.slots[0].base_angles[5] = 0.8;
        let parked = spins.slots[0].active_rotor_at(0.0);

        spins.recompose_active(0.0, &[]);
        let spinning_at_zero = spins.rotor(1);
        spins.recompose_active(4.0, &[]);
        assert_eq!(spins.rotor(0), parked, "the parked slot moved");
        assert_ne!(
            spins.rotor(1),
            spinning_at_zero,
            "the spinning slot did not advance"
        );
    }

    /// Selection is the press ray's pick: the slot the user clicked becomes
    /// the one the controls write, and every other slot is left alone. Runs
    /// the real picker over the real layout, so a selection wired to the wrong
    /// slot (or to all of them) fails here.
    #[test]
    fn a_press_aims_the_controls_at_the_body_it_picked_and_no_other() {
        const SLOTS: usize = 4;
        let physics = PlaygroundPhysics::new(SLOTS, BODY_SIZE);
        let mut spins = SlotSpins::new(SLOTS);

        for target in 0..SLOTS {
            let centre = Vec4::from_array(body_position(target, SLOTS)).truncate();
            let ray = Ray {
                origin: centre + Vec3::Z * 10.0,
                direction: -Vec3::Z,
            };
            spins.select_picked(physics.pick(&ray, SLOTS, BODY_SIZE));
            assert_eq!(spins.selected(), target);

            // The edit the controls make lands on the picked slot only.
            spins.selected_spin_mut().base_angles[Plane4::Zw as usize] = 0.5 + target as f32;
            spins.recompose_active(0.0, &[]);
            for slot in 0..SLOTS {
                let touched = spins.slots[slot].base_angles[Plane4::Zw as usize] != 0.0;
                assert_eq!(
                    touched,
                    slot <= target,
                    "slot {slot} was {} after selecting {target}",
                    if touched { "edited" } else { "skipped" }
                );
            }
        }

        // A ray through empty space keeps the last selection: the controls
        // always have a subject.
        let sky = Ray {
            origin: Vec3::Y * 40.0,
            direction: Vec3::Y,
        };
        spins.select_picked(physics.pick(&sky, SLOTS, BODY_SIZE));
        assert_eq!(spins.selected(), SLOTS - 1);
        // So does a slot a row edit already retired.
        spins.select_picked(Some(SLOTS + 3));
        assert_eq!(spins.selected(), SLOTS - 1);
    }

    /// A row edit keeps the rotations of the slots that survived it and hands
    /// the new ones the boot rotation, and it never leaves the selection
    /// pointing past the end of the row.
    #[test]
    fn sync_preserves_surviving_slots_and_clamps_the_selection() {
        let mut spins = SlotSpins::new(4);
        spins.select_picked(Some(3));
        spins.selected_spin_mut().base_angles[0] = 1.25;
        spins.slots[1].base_angles[0] = -0.5;

        spins.sync(2);
        assert_eq!(spins.slots.len(), 2);
        assert_eq!(spins.selected(), 1, "selection left the shortened row");
        assert_eq!(spins.slots[1].base_angles[0], -0.5, "a survivor was reset");

        spins.sync(5);
        assert_eq!(spins.slots[1].base_angles[0], -0.5);
        assert_eq!(spins.slots[4], SlotSpin::default(), "a new slot came dirty");
        assert_eq!(spins.selected(), 1, "growing the row moved the selection");
    }

    /// `Reset orientation` has to clear the baselines, not just the rotor:
    /// Active recomposes from the baselines on the next frame, so clearing the
    /// rotor alone is undone before it draws.
    #[test]
    fn clearing_orientation_survives_the_next_recompose() {
        let mut spins = SlotSpins::new(2);
        spins.slots[0].base_angles = [0.4, -0.2, 1.1, 0.0, 0.3, -0.9];
        spins.slots[1].base_angles = [1.0; 6];
        spins.recompose_active(2.0, &[]);

        spins.clear_orientation();
        // At t = 0 the masks contribute nothing, so a cleared row is identity.
        spins.recompose_active(0.0, &[]);
        for slot in 0..2 {
            assert_eq!(spins.rotor(slot), Rotor4::IDENTITY, "slot {slot} came back");
        }
        assert_eq!(
            spins.selected_spin().active,
            DEFAULT_ACTIVE,
            "clearing the orientation also cleared the plane mask"
        );
    }

    /// The full reset returns every slot to the boot state and re-aims the
    /// controls at the leading slot.
    #[test]
    fn reset_returns_every_slot_to_the_boot_rotation() {
        let mut spins = SlotSpins::new(3);
        spins.select_picked(Some(2));
        spins.selected_spin_mut().active = [true; 6];
        spins.slots[0].base_angles[3] = 2.0;
        spins.recompose_active(1.0, &[]);

        spins.reset();
        assert_eq!(spins.selected(), 0);
        for slot in 0..3 {
            assert_eq!(spins.slots[slot], SlotSpin::default());
        }
    }

    /// The upload gate compares the whole row, so a single rotated slot is
    /// enough to re-upload. A gate that compared one rotor against one cached
    /// rotor (the pre-per-slot form) misses exactly this.
    #[test]
    fn one_rotated_slot_makes_the_uploaded_row_stale() {
        let mut spins = SlotSpins::new(4);
        let mut uploaded = Vec::new();
        spins.record_rotors(&mut uploaded);
        assert!(!spins.rotors_differ_from(&uploaded));

        for slot in 0..4 {
            spins.slots[slot].base_angles[1] = 0.3;
            spins.recompose_active(0.0, &[]);
            assert!(
                spins.rotors_differ_from(&uploaded),
                "rotating slot {slot} alone left the upload gate closed"
            );
            spins.record_rotors(&mut uploaded);
            assert!(!spins.rotors_differ_from(&uploaded));
        }

        // A row edit that changed the slot count changed which body each
        // rotor belongs to, whatever the values are.
        spins.sync(3);
        assert!(spins.rotors_differ_from(&uploaded));
    }
}
