use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub(crate) static TOTAL_ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static TOTAL_DEALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static TOTAL_ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
pub(crate) static TOTAL_DEALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
pub(crate) static ALLOC_INSTALLED: AtomicBool = AtomicBool::new(false);

/// Counts `Layout::size()`, not the aligned size, so it reads low.
pub struct CountingAllocator<A: GlobalAlloc> {
    inner: A,
}

impl<A: GlobalAlloc> CountingAllocator<A> {
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }
}

unsafe impl<A: GlobalAlloc> GlobalAlloc for CountingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_INSTALLED.store(true, Ordering::Relaxed);
        TOTAL_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        TOTAL_ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        self.inner.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        TOTAL_DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        TOTAL_DEALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        self.inner.dealloc(ptr, layout);
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_INSTALLED.store(true, Ordering::Relaxed);
        TOTAL_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        TOTAL_ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        self.inner.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_INSTALLED.store(true, Ordering::Relaxed);
        TOTAL_DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        TOTAL_DEALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        TOTAL_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        TOTAL_ALLOC_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        self.inner.realloc(ptr, layout, new_size)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AllocSnapshot {
    pub alloc_bytes: u64,
    pub dealloc_bytes: u64,
    pub alloc_count: u64,
    pub dealloc_count: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AllocDelta {
    pub net_bytes: i64,
    pub alloc_bytes: u64,
    pub alloc_count: u64,
    pub dealloc_count: u64,
}

/// `None` when no [`CountingAllocator`] has been installed.
pub fn current_snapshot() -> Option<AllocSnapshot> {
    if !ALLOC_INSTALLED.load(Ordering::Relaxed) {
        return None;
    }
    Some(AllocSnapshot {
        alloc_bytes: TOTAL_ALLOC_BYTES.load(Ordering::Relaxed),
        dealloc_bytes: TOTAL_DEALLOC_BYTES.load(Ordering::Relaxed),
        alloc_count: TOTAL_ALLOC_COUNT.load(Ordering::Relaxed),
        dealloc_count: TOTAL_DEALLOC_COUNT.load(Ordering::Relaxed),
    })
}

/// `start` must be the earlier snapshot.
pub fn delta(start: AllocSnapshot, end: AllocSnapshot) -> AllocDelta {
    let alloc_bytes = end.alloc_bytes.saturating_sub(start.alloc_bytes);
    let dealloc_bytes = end.dealloc_bytes.saturating_sub(start.dealloc_bytes);
    AllocDelta {
        net_bytes: (alloc_bytes as i64).saturating_sub(dealloc_bytes as i64),
        alloc_bytes,
        alloc_count: end.alloc_count.saturating_sub(start.alloc_count),
        dealloc_count: end.dealloc_count.saturating_sub(start.dealloc_count),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_handles_dealloc_dominant() {
        let start = AllocSnapshot {
            alloc_bytes: 1_000,
            dealloc_bytes: 200,
            alloc_count: 10,
            dealloc_count: 3,
        };
        let end = AllocSnapshot {
            alloc_bytes: 1_100,
            dealloc_bytes: 1_000,
            alloc_count: 12,
            dealloc_count: 20,
        };
        let d = delta(start, end);
        assert_eq!(d.alloc_bytes, 100);
        assert_eq!(d.net_bytes, -700);
    }
}
