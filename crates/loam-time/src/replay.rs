//! Deterministic replay: a versioned input tape indexed by tick, and the
//! hash mixer a recorder and a verifier have to share.
//!
//! The sim is `tick(state, input, tick_number) -> state'`, so a run is
//! reproducible from its seed plus the per-tick input stream. A [`Tape`] is
//! that stream, plus [`Checkpoint`]s carrying the state hash the recording run
//! observed. Replay is then checkable rather than merely repeatable: a mismatch
//! names the first tick at which the two runs parted.
//!
//! The tape is a byte format so a recording survives the process that made it.
//! It is versioned because a change to what the sim consumes per tick, or to
//! what the state hash covers, invalidates every tape written before it, and a
//! silently misread tape is worse than a rejected one.

use std::fmt;

use thiserror::Error;

/// FNV-1a 64-bit (Fowler/Noll/Vo 1991; reference offset basis and prime,
/// <http://www.isthe.com/chongo/tech/comp/fnv/>). `std`'s `DefaultHasher` is
/// documented as unstable across releases, so a hash that gets committed as a
/// constant or written into a tape needs its own mixer.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Incremental FNV-1a 64 over little-endian words.
///
/// Byte order is fixed rather than native so a hash recorded on one host is
/// comparable on another; the values fed in are already raw bit patterns, so
/// the mixer is the only place endianness could leak in.
///
/// The digest is a function of the byte stream alone and carries no framing,
/// so two different word layouts over the same bytes agree. Callers own the
/// framing: a sampler that changes what it writes per entity, or in what
/// order, changes the meaning of every hash it has ever produced and owes a
/// [`TAPE_FORMAT_VERSION`] bump.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateHash(u64);

impl Default for StateHash {
    fn default() -> Self {
        Self::new()
    }
}

impl StateHash {
    pub fn new() -> Self {
        Self(FNV_OFFSET_BASIS)
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= byte as u64;
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
    }

    pub fn write_u32(&mut self, word: u32) {
        self.write_bytes(&word.to_le_bytes());
    }

    pub fn write_u32s(&mut self, words: &[u32]) {
        for &word in words {
            self.write_u32(word);
        }
    }

    pub fn write_u64(&mut self, value: u64) {
        self.write_bytes(&value.to_le_bytes());
    }

    /// Raw bits, not a rounded value: two states that differ by one ulp are two
    /// states.
    pub fn write_f32(&mut self, value: f32) {
        self.write_u32(value.to_bits());
    }

    /// Read the digest without ending the sequence, so one hash can be sampled
    /// per tick while still chaining across the whole run.
    pub fn finish(&self) -> u64 {
        self.0
    }
}

/// Leading bytes of an encoded [`Tape`].
pub const TAPE_MAGIC: [u8; 8] = *b"LOAMTAPE";

/// Bumped whenever an existing tape would be misread rather than merely
/// incomplete: a change to the byte layout, to the meaning of a tick's input
/// words, or to what a state hash covers.
pub const TAPE_FORMAT_VERSION: u32 = 1;

/// magic + version + hz + seed + words_per_tick + ticks + checkpoint count.
const HEADER_LEN: usize = 8 + 4 + 4 + 8 + 4 + 8 + 4;
const CHECKPOINT_LEN: usize = 16;

/// A state hash observed after `tick` completed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    pub tick: u64,
    pub state_hash: u64,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TapeError {
    #[error("not a loam tape: leading bytes are {found:02x?}")]
    BadMagic { found: [u8; 8] },
    /// Reported rather than tolerated: a tape written under another version can
    /// decode cleanly and still drive the sim down a different path.
    #[error("tape format version {found}, this build reads {TAPE_FORMAT_VERSION}")]
    UnsupportedVersion { found: u32 },
    #[error("tape declares {expected} bytes, got {found}")]
    LengthMismatch { expected: u64, found: usize },
    #[error("checkpoint ticks are not strictly ascending at index {index}")]
    CheckpointOrder { index: usize },
}

/// A recorded run: the per-tick input stream, the seed it started from, and the
/// state hashes it passed through.
///
/// Input is a flat `u32` buffer of `words_per_tick` per tick rather than a
/// typed frame, because the tape is tick bookkeeping and the meaning of a word
/// belongs to the sim that wrote it. `words_per_tick` may be zero: an
/// input-free run (an attract-mode loop, a physics fixture) still has a seed, a
/// tick count, and hashes worth pinning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tape {
    tick_hz: u32,
    seed: u64,
    words_per_tick: u32,
    ticks: u64,
    inputs: Vec<u32>,
    checkpoints: Vec<Checkpoint>,
}

impl Tape {
    pub fn new(tick_hz: u32, seed: u64, words_per_tick: u32) -> Self {
        Self {
            tick_hz,
            seed,
            words_per_tick,
            ticks: 0,
            inputs: Vec::new(),
            checkpoints: Vec::new(),
        }
    }

    pub fn tick_hz(&self) -> u32 {
        self.tick_hz
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn words_per_tick(&self) -> u32 {
        self.words_per_tick
    }

    pub fn ticks(&self) -> u64 {
        self.ticks
    }

    /// Append one tick's input.
    ///
    /// Panics if `input.len()` is not `words_per_tick`: a short frame would
    /// shift every later tick's input by the shortfall, and the shift is
    /// undetectable once the tape is encoded.
    pub fn push_tick(&mut self, input: &[u32]) {
        assert_eq!(
            input.len(),
            self.words_per_tick as usize,
            "tape frame is {} words",
            self.words_per_tick,
        );
        self.inputs.extend_from_slice(input);
        self.ticks += 1;
    }

    pub fn input(&self, tick: u64) -> Option<&[u32]> {
        if tick >= self.ticks {
            return None;
        }
        let width = self.words_per_tick as usize;
        let start = tick as usize * width;
        Some(&self.inputs[start..start + width])
    }

    /// Record the state hash observed after `tick`.
    ///
    /// Panics unless `tick` is beyond the last checkpoint: a verifier walks
    /// checkpoints in order against a single forward replay, and an
    /// out-of-order entry would silently never be checked.
    pub fn checkpoint(&mut self, tick: u64, state_hash: u64) {
        if let Some(last) = self.checkpoints.last() {
            assert!(
                tick > last.tick,
                "checkpoint ticks must ascend: {tick} after {}",
                last.tick,
            );
        }
        self.checkpoints.push(Checkpoint { tick, state_hash });
    }

    pub fn checkpoints(&self) -> &[Checkpoint] {
        &self.checkpoints
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(
            HEADER_LEN + self.inputs.len() * 4 + self.checkpoints.len() * CHECKPOINT_LEN,
        );
        bytes.extend_from_slice(&TAPE_MAGIC);
        bytes.extend_from_slice(&TAPE_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.tick_hz.to_le_bytes());
        bytes.extend_from_slice(&self.seed.to_le_bytes());
        bytes.extend_from_slice(&self.words_per_tick.to_le_bytes());
        bytes.extend_from_slice(&self.ticks.to_le_bytes());
        bytes.extend_from_slice(&(self.checkpoints.len() as u32).to_le_bytes());
        for &word in &self.inputs {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        for checkpoint in &self.checkpoints {
            bytes.extend_from_slice(&checkpoint.tick.to_le_bytes());
            bytes.extend_from_slice(&checkpoint.state_hash.to_le_bytes());
        }
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, TapeError> {
        let mut reader = Reader::new(bytes);
        let magic = reader.take::<8>().ok_or(TapeError::LengthMismatch {
            expected: HEADER_LEN as u64,
            found: bytes.len(),
        })?;
        if magic != TAPE_MAGIC {
            return Err(TapeError::BadMagic { found: magic });
        }
        // Version before length: a tape from another version has another
        // layout, so its declared sizes are not this decoder's to check.
        let version = reader.u32().ok_or(TapeError::LengthMismatch {
            expected: HEADER_LEN as u64,
            found: bytes.len(),
        })?;
        if version != TAPE_FORMAT_VERSION {
            return Err(TapeError::UnsupportedVersion { found: version });
        }
        let (tick_hz, seed, words_per_tick, ticks, checkpoint_count) =
            reader.header_tail().ok_or(TapeError::LengthMismatch {
                expected: HEADER_LEN as u64,
                found: bytes.len(),
            })?;

        // Saturating, not merely widened: `ticks` is 64 bits itself, so a
        // corrupt header overflows the byte count at any width, and the
        // overflow panics in a debug build and wraps into a plausible length
        // in a release one. Saturation cannot equal a real `bytes.len()`.
        let input_bytes = u64::from(words_per_tick)
            .saturating_mul(ticks)
            .saturating_mul(4);
        let expected = (HEADER_LEN as u64)
            .saturating_add(input_bytes)
            .saturating_add(u64::from(checkpoint_count) * CHECKPOINT_LEN as u64);
        if expected != bytes.len() as u64 {
            return Err(TapeError::LengthMismatch {
                expected,
                found: bytes.len(),
            });
        }

        let word_count = (input_bytes / 4) as usize;
        let mut inputs = Vec::with_capacity(word_count);
        for _ in 0..word_count {
            inputs.push(reader.u32().expect("length checked above"));
        }
        let mut checkpoints = Vec::with_capacity(checkpoint_count as usize);
        for index in 0..checkpoint_count as usize {
            let tick = reader.u64().expect("length checked above");
            let state_hash = reader.u64().expect("length checked above");
            if let Some(last) = checkpoints.last() {
                let Checkpoint { tick: prev, .. } = *last;
                if tick <= prev {
                    return Err(TapeError::CheckpointOrder { index });
                }
            }
            checkpoints.push(Checkpoint { tick, state_hash });
        }

        Ok(Self {
            tick_hz,
            seed,
            words_per_tick,
            ticks,
            inputs,
            checkpoints,
        })
    }
}

impl fmt::Display for Tape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tape v{TAPE_FORMAT_VERSION}: {} ticks at {}Hz, seed {:#018x}, \
             {} input words/tick, {} checkpoints",
            self.ticks,
            self.tick_hz,
            self.seed,
            self.words_per_tick,
            self.checkpoints.len(),
        )
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let end = self.at.checked_add(N)?;
        let slice = self.bytes.get(self.at..end)?;
        self.at = end;
        Some(slice.try_into().expect("slice is N bytes"))
    }

    fn u32(&mut self) -> Option<u32> {
        self.take::<4>().map(u32::from_le_bytes)
    }

    fn u64(&mut self) -> Option<u64> {
        self.take::<8>().map(u64::from_le_bytes)
    }

    fn header_tail(&mut self) -> Option<(u32, u64, u32, u64, u32)> {
        Some((
            self.u32()?,
            self.u64()?,
            self.u32()?,
            self.u64()?,
            self.u32()?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recorded() -> Tape {
        let mut tape = Tape::new(60, 0x1234_5678_9abc_def0, 3);
        for tick in 0..8u64 {
            let t = tick as u32;
            tape.push_tick(&[t, t.wrapping_mul(7), (1.5f32 * t as f32).to_bits()]);
        }
        tape.checkpoint(3, 0xdead_beef_0000_0001);
        tape.checkpoint(7, 0xdead_beef_0000_0002);
        tape
    }

    #[test]
    fn hash_is_sensitive_to_order_and_to_one_flipped_bit() {
        let mut ab = StateHash::new();
        ab.write_u32(1);
        ab.write_u32(2);
        let mut ba = StateHash::new();
        ba.write_u32(2);
        ba.write_u32(1);
        assert_ne!(ab.finish(), ba.finish(), "swapped operands must be visible");

        let mut low = StateHash::new();
        low.write_u32s(&[7, 7, 7, 7]);
        let mut flipped = StateHash::new();
        flipped.write_u32s(&[7, 7, 7 ^ 1, 7]);
        assert_ne!(low.finish(), flipped.finish());
    }

    /// The mixer carries no framing, so the recorded and the replaying run must
    /// agree on the word layout for their digests to mean the same thing. Pinned
    /// so the absence is a documented property rather than a surprise.
    #[test]
    fn hash_is_a_function_of_the_byte_stream_and_not_of_the_word_widths() {
        let mut split = StateHash::new();
        split.write_u32(0);
        split.write_u32(1);
        let mut whole = StateHash::new();
        whole.write_u64(1 << 32);
        assert_eq!(split.finish(), whole.finish());
    }

    #[test]
    fn hash_separates_signed_zeros_and_every_nan_it_is_given() {
        let mut positive = StateHash::new();
        positive.write_f32(0.0);
        let mut negative = StateHash::new();
        negative.write_f32(-0.0);
        assert_ne!(
            positive.finish(),
            negative.finish(),
            "f32 hashing is over bits, and 0.0 == -0.0 is a value comparison",
        );

        let mut nan = StateHash::new();
        nan.write_f32(f32::NAN);
        assert_eq!(
            nan.finish(),
            {
                let mut again = StateHash::new();
                again.write_f32(f32::NAN);
                again.finish()
            },
            "one NaN bit pattern must hash to one value",
        );
    }

    #[test]
    fn hash_chains_rather_than_restarts() {
        let mut chained = StateHash::new();
        chained.write_u32(9);
        let after_one = chained.finish();
        chained.write_u32(9);
        assert_ne!(
            after_one,
            chained.finish(),
            "finish must not reset the accumulator",
        );

        let mut bulk = StateHash::new();
        bulk.write_u32s(&[9, 9]);
        assert_eq!(bulk.finish(), chained.finish());
    }

    #[test]
    fn encode_decode_round_trips_every_field() {
        let tape = recorded();
        let decoded = Tape::decode(&tape.encode()).expect("own encoding decodes");
        assert_eq!(decoded, tape);
    }

    #[test]
    fn input_is_addressed_by_tick_and_ends_at_the_tick_count() {
        let tape = recorded();
        assert_eq!(tape.ticks(), 8);
        assert_eq!(tape.input(0), Some(&[0u32, 0, 0.0f32.to_bits()][..]));
        assert_eq!(tape.input(5).map(|w| w[1]), Some(35));
        assert_eq!(tape.input(8), None, "one past the last tick has no input");
    }

    #[test]
    fn a_tape_with_no_input_still_carries_ticks_and_checkpoints() {
        let mut tape = Tape::new(120, 7, 0);
        for _ in 0..4 {
            tape.push_tick(&[]);
        }
        tape.checkpoint(3, 0xabc);
        let decoded = Tape::decode(&tape.encode()).expect("width-zero tape decodes");
        assert_eq!(decoded.ticks(), 4);
        assert_eq!(decoded.input(0), Some(&[][..]));
        assert_eq!(decoded.checkpoints(), tape.checkpoints());
    }

    #[test]
    fn a_future_version_is_rejected_rather_than_read() {
        let mut bytes = recorded().encode();
        let bumped = TAPE_FORMAT_VERSION + 1;
        bytes[8..12].copy_from_slice(&bumped.to_le_bytes());
        assert_eq!(
            Tape::decode(&bytes),
            Err(TapeError::UnsupportedVersion { found: bumped }),
        );
    }

    #[test]
    fn foreign_bytes_are_rejected_before_any_field_is_believed() {
        let mut bytes = recorded().encode();
        bytes[0] = b'X';
        let mut found = TAPE_MAGIC;
        found[0] = b'X';
        assert_eq!(Tape::decode(&bytes), Err(TapeError::BadMagic { found }));
        assert!(matches!(
            Tape::decode(&[]),
            Err(TapeError::LengthMismatch { .. }),
        ));
    }

    #[test]
    fn a_payload_that_does_not_match_the_header_is_rejected_either_way() {
        let full = recorded().encode();
        let expected = full.len() as u64;
        assert_eq!(
            Tape::decode(&full[..full.len() - 1]),
            Err(TapeError::LengthMismatch {
                expected,
                found: full.len() - 1,
            }),
        );

        let mut padded = full.clone();
        padded.push(0);
        assert_eq!(
            Tape::decode(&padded),
            Err(TapeError::LengthMismatch {
                expected,
                found: padded.len(),
            }),
            "trailing bytes mean the writer and the reader disagree",
        );
    }

    #[test]
    fn a_header_claiming_more_payload_than_memory_is_rejected_without_wrapping() {
        let mut bytes = recorded().encode();
        // words_per_tick = u32::MAX over 8 ticks fits a 64-bit byte count and
        // wraps a 32-bit one, so this is what the widening buys.
        bytes[24..28].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            Tape::decode(&bytes),
            Err(TapeError::LengthMismatch { .. }),
        ));
    }

    /// The tick count is itself 64 bits, so widening the word width is not
    /// enough: the product overflows the widened arithmetic too, and an
    /// overflow here is a panic in a debug build and a plausible length in a
    /// release one.
    #[test]
    fn a_header_declaring_more_ticks_than_a_byte_count_can_express_is_rejected() {
        let mut bytes = recorded().encode();
        bytes[28..36].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            Tape::decode(&bytes),
            Err(TapeError::LengthMismatch { .. }),
        ));
    }

    #[test]
    fn decoded_checkpoints_must_ascend() {
        let mut tape = recorded();
        tape.checkpoints[1].tick = 3;
        assert_eq!(
            Tape::decode(&tape.encode()),
            Err(TapeError::CheckpointOrder { index: 1 }),
        );
    }

    #[test]
    #[should_panic(expected = "tape frame is 3 words")]
    fn a_short_frame_panics_rather_than_shifting_later_ticks() {
        let mut tape = Tape::new(60, 0, 3);
        tape.push_tick(&[1, 2]);
    }

    #[test]
    #[should_panic(expected = "checkpoint ticks must ascend")]
    fn an_out_of_order_checkpoint_panics_at_the_writer() {
        let mut tape = Tape::new(60, 0, 0);
        tape.checkpoint(5, 1);
        tape.checkpoint(5, 2);
    }
}
