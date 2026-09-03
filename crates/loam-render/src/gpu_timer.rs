//! wgpu rejects `MAP_READ | QUERY_RESOLVE` on one buffer and locks a whole
//! buffer while any slice is mapped, so one resolve buffer feeds a map buffer
//! per slot.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Features, MapMode, QuerySet,
    QuerySetDescriptor, QueryType, Queue, QUERY_RESOLVE_BUFFER_ALIGNMENT,
};

const FRAMES_IN_FLIGHT: usize = 3;

const MAX_PLAUSIBLE_FRAME_NS: u64 = 1_000_000_000 / 10;

// Some drivers pair a start tick with an end tick from a later cycle.
fn is_plausible_frame_delta_ns(delta_ns: u64) -> bool {
    delta_ns <= MAX_PLAUSIBLE_FRAME_NS
}

const BYTES_PER_SLOT: u64 = 16;

const SLOT_STRIDE_BYTES: u64 = QUERY_RESOLVE_BUFFER_ALIGNMENT;

struct SlotState {
    in_flight: Arc<AtomicBool>,
    map_buffer: Buffer,
}

pub struct GpuTimer {
    query_set: QuerySet,
    resolve_buffer: Buffer,
    slots: [SlotState; FRAMES_IN_FLIGHT],
    frame_index: u64,
    timestamp_period_ns: f32,
    rx: Receiver<Duration>,
    tx: Sender<Duration>,
}

impl GpuTimer {
    pub fn new(device: &Device, queue: &Queue) -> Option<Self> {
        let needed = Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        if !device.features().contains(needed) {
            return None;
        }
        let query_set = device.create_query_set(&QuerySetDescriptor {
            label: Some("loam-render::GpuTimer::query_set"),
            ty: QueryType::Timestamp,
            count: (FRAMES_IN_FLIGHT * 2) as u32,
        });
        let resolve_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("loam-render::GpuTimer::resolve_buffer"),
            size: SLOT_STRIDE_BYTES * FRAMES_IN_FLIGHT as u64,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let slots = std::array::from_fn(|_| SlotState {
            in_flight: Arc::new(AtomicBool::new(false)),
            map_buffer: device.create_buffer(&BufferDescriptor {
                label: Some("loam-render::GpuTimer::map_buffer"),
                size: BYTES_PER_SLOT,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        });
        let (tx, rx) = channel();
        Some(Self {
            query_set,
            resolve_buffer,
            slots,
            frame_index: 0,
            timestamp_period_ns: queue.get_timestamp_period(),
            rx,
            tx,
        })
    }

    fn current_slot(&self) -> usize {
        (self.frame_index as usize) % FRAMES_IN_FLIGHT
    }

    fn slot_query_range(slot: usize) -> std::ops::Range<u32> {
        let base = (slot * 2) as u32;
        base..(base + 2)
    }

    fn slot_byte_range(slot: usize) -> std::ops::Range<u64> {
        let base = slot as u64 * SLOT_STRIDE_BYTES;
        base..(base + BYTES_PER_SLOT)
    }

    pub fn write_start(&self, encoder: &mut CommandEncoder) {
        let slot = self.current_slot();
        if self.slots[slot].in_flight.load(Ordering::Acquire) {
            return;
        }
        let range = Self::slot_query_range(slot);
        encoder.write_timestamp(&self.query_set, range.start);
    }

    pub fn write_end_and_resolve(&self, encoder: &mut CommandEncoder) {
        let slot = self.current_slot();
        if self.slots[slot].in_flight.load(Ordering::Acquire) {
            return;
        }
        let query_range = Self::slot_query_range(slot);
        let byte_range = Self::slot_byte_range(slot);
        encoder.write_timestamp(&self.query_set, query_range.end - 1);
        encoder.resolve_query_set(
            &self.query_set,
            query_range,
            &self.resolve_buffer,
            byte_range.start,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            byte_range.start,
            &self.slots[slot].map_buffer,
            0,
            BYTES_PER_SLOT,
        );
        self.slots[slot].in_flight.store(true, Ordering::Release);
    }

    /// Call once per redraw, after the end-of-frame queue submit.
    pub fn tick(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);

        while let Ok(duration) = self.rx.try_recv() {
            loam_time::frame_trace::record_external("gpu-total", duration);
        }

        // Mapping an already-mapping slice is a wgpu validation error.
        let just_resolved_slot = (self.frame_index.wrapping_sub(1) as usize) % FRAMES_IN_FLIGHT;
        if !self.slots[just_resolved_slot]
            .in_flight
            .load(Ordering::Acquire)
        {
            return;
        }
        let buffer = self.slots[just_resolved_slot].map_buffer.clone();
        let buffer_for_callback = buffer.clone();
        let period_ns = self.timestamp_period_ns;
        let tx = self.tx.clone();
        let flag = self.slots[just_resolved_slot].in_flight.clone();
        buffer.slice(..).map_async(MapMode::Read, move |result| {
            if result.is_ok() {
                let view = buffer_for_callback.slice(..).get_mapped_range();
                if let (Ok(start_bytes), Ok(end_bytes)) = (
                    <[u8; 8]>::try_from(&view[0..8]),
                    <[u8; 8]>::try_from(&view[8..16]),
                ) {
                    let start_ticks = u64::from_le_bytes(start_bytes);
                    let end_ticks = u64::from_le_bytes(end_bytes);
                    let delta_ticks = end_ticks.saturating_sub(start_ticks);
                    let delta_ns = (delta_ticks as f64 * period_ns as f64) as u64;
                    if is_plausible_frame_delta_ns(delta_ns) {
                        let _ = tx.send(Duration::from_nanos(delta_ns));
                    }
                }
                drop(view);
                buffer_for_callback.unmap();
            }
            flag.store(false, Ordering::Release);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const _: () = assert!(SLOT_STRIDE_BYTES >= BYTES_PER_SLOT);

    #[test]
    fn slot_byte_range_is_aligned_and_disjoint() {
        for slot in 0..FRAMES_IN_FLIGHT {
            let range = GpuTimer::slot_byte_range(slot);
            assert_eq!(
                range.start % QUERY_RESOLVE_BUFFER_ALIGNMENT,
                0,
                "slot {slot} start not aligned"
            );
            assert_eq!(range.end - range.start, BYTES_PER_SLOT);
        }
        for slot in 0..FRAMES_IN_FLIGHT.saturating_sub(1) {
            let a = GpuTimer::slot_byte_range(slot);
            let b = GpuTimer::slot_byte_range(slot + 1);
            assert!(a.end <= b.start);
        }
    }

    #[test]
    fn plausible_frame_delta_rejects_over_budget() {
        assert!(is_plausible_frame_delta_ns(4_000_000));
        assert!(is_plausible_frame_delta_ns(16_666_667));
        assert!(is_plausible_frame_delta_ns(MAX_PLAUSIBLE_FRAME_NS));
        assert!(!is_plausible_frame_delta_ns(250_000_000));
        assert!(!is_plausible_frame_delta_ns(950_000_000));
        assert!(is_plausible_frame_delta_ns(0));
    }
}
