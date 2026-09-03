//! The summary carries a synthetic `unscoped` row: `frame` minus the sections
//! the frame loop opens inside it (`crate::FRAME_LOOP_SECTIONS`, crate-private
//! so not linkable from here).

use loam_egui::{cmd, Console};
use loam_time::frame_trace;
use std::time::Duration;

fn fmt_dur(d: std::time::Duration) -> String {
    let ns = d.as_nanos();
    if ns < 1_000 {
        format!("{ns}ns")
    } else if ns < 1_000_000 {
        format!("{:.1}us", ns as f64 / 1_000.0)
    } else if ns < 1_000_000_000 {
        format!("{:.2}ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.2}s", ns as f64 / 1_000_000_000.0)
    }
}

// Saturating: parent and children come from independent `Instant` reads.
fn unscoped(frame: &frame_trace::FrameTrace) -> Option<Duration> {
    let mut total = None;
    let mut covered = Duration::ZERO;
    for section in &frame.sections {
        if section.name == "frame" {
            total = Some(section.elapsed);
        } else if crate::FRAME_LOOP_SECTIONS.contains(&section.name) {
            covered += section.elapsed;
        }
    }
    total.map(|t| t.saturating_sub(covered))
}

// Nearest-rank percentiles, matching `frame_trace::aggregate`.
fn unscoped_stats() -> Option<frame_trace::SectionStats> {
    let mut samples: Vec<Duration> =
        frame_trace::with_history(|history| history.iter().filter_map(unscoped).collect());
    if samples.is_empty() {
        return None;
    }
    samples.sort();
    let n = samples.len();
    let pick = |q: f32| samples[((n as f32 * q) as usize).min(n - 1)];
    Some(frame_trace::SectionStats {
        name: "unscoped",
        samples: n,
        mean: samples.iter().sum::<Duration>() / n as u32,
        p50: pick(0.50),
        p95: pick(0.95),
        p99: pick(0.99),
        max: samples[n - 1],
    })
}

fn summary_rows() -> Vec<frame_trace::SectionStats> {
    let mut stats = frame_trace::aggregate();
    if let Some(residual) = unscoped_stats() {
        stats.push(residual);
        stats.sort_by_key(|s| std::cmp::Reverse(s.p95));
    }
    stats
}

fn print_summary(out: &mut loam_egui::ConsoleWriter) {
    let stats = summary_rows();
    if stats.is_empty() {
        out.line("trace: no frames in window (collect runs once the demo is rendering)");
        return;
    }
    let history_len = frame_trace::history().len();
    out.line(format!(
        "trace summary ({history_len} frames, sorted by p95 desc):"
    ));
    out.line(format!(
        "  {:<18} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "section", "n", "mean", "p50", "p95", "p99", "max",
    ));
    for s in stats {
        out.line(format!(
            "  {:<18} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}",
            truncate(s.name, 18),
            s.samples,
            fmt_dur(s.mean),
            fmt_dur(s.p50),
            fmt_dur(s.p95),
            fmt_dur(s.p99),
            fmt_dur(s.max),
        ));
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}~", &s[..max - 1])
    }
}

fn print_last(out: &mut loam_egui::ConsoleWriter) {
    let Some(frame) = frame_trace::last_frame() else {
        out.line("trace: no frames in window yet");
        return;
    };
    let total = frame.total();
    out.line(format!(
        "trace last-frame ({} sections, sum {}):",
        frame.sections.len(),
        fmt_dur(total),
    ));
    for section in &frame.sections {
        let pct = if !total.is_zero() {
            (section.elapsed.as_nanos() as f64 * 100.0) / total.as_nanos() as f64
        } else {
            0.0
        };
        out.line(format!(
            "  {:<18} {:>10} ({:>4.1}%)",
            truncate(section.name, 18),
            fmt_dur(section.elapsed),
            pct,
        ));
    }
}

fn format_summary() -> String {
    let stats = summary_rows();
    if stats.is_empty() {
        return "trace: no frames in window\n".to_string();
    }
    let history_len = frame_trace::history().len();
    let mut s = String::new();
    s.push_str(&format!(
        "trace summary ({history_len} frames, sorted by p95 desc):\n"
    ));
    s.push_str(&format!(
        "  {:<18} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}\n",
        "section", "n", "mean", "p50", "p95", "p99", "max",
    ));
    for st in stats {
        s.push_str(&format!(
            "  {:<18} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}\n",
            truncate(st.name, 18),
            st.samples,
            fmt_dur(st.mean),
            fmt_dur(st.p50),
            fmt_dur(st.p95),
            fmt_dur(st.p99),
            fmt_dur(st.max),
        ));
    }
    s
}

pub fn register_command<Ctx: 'static>(console: &mut Console<Ctx>) {
    console.register(
        cmd(
            "trace",
            "show CPU per-section frame timings (collected by loam-time::frame_trace)",
            |args, _ctx: &mut Ctx, out| {
                match args.first().copied() {
                    None | Some("summary") => print_summary(out),
                    Some("last") => print_last(out),
                    Some("dump") => {
                        let summary = format_summary();
                        tracing::info!("\n{summary}");
                        out.line("trace: dumped to browser console (open DevTools to copy)");
                    }
                    Some("clear") => {
                        frame_trace::set_capacity(1);
                        frame_trace::set_capacity(frame_trace::DEFAULT_CAPACITY);
                        out.line("trace: history cleared");
                    }
                    Some("cap") => {
                        let n = args
                            .get(1)
                            .copied()
                            .and_then(|s| s.parse::<usize>().ok());
                        match n {
                            Some(n) if n >= 1 => {
                                frame_trace::set_capacity(n);
                                out.line(format!("trace: capacity set to {n} frames"));
                            }
                            _ => {
                                out.line("usage: trace cap <N>  (N >= 1)");
                            }
                        }
                    }
                    Some(other) => {
                        out.line(format!(
                            "trace: unknown subcommand '{other}' (try summary | last | dump | clear | cap)"
                        ));
                    }
                }
                Ok(())
            },
        )
        .with_args(&[&["summary", "last", "dump", "clear", "cap"]]),
    );
}

pub struct PerfOverlay {
    visible: bool,
    toggle_key: loam_egui::egui::Key,
    window: usize,
}

impl Default for PerfOverlay {
    fn default() -> Self {
        Self {
            visible: false,
            toggle_key: loam_egui::egui::Key::F3,
            window: 60,
        }
    }
}

const OVERLAY_WIDTH: f32 = 260.0;

const OVERLAY_MARGIN: f32 = 12.0;

// `content_rect` ignores panels and would seat the readout behind the menu bar.
fn perf_overlay_seat(ctx: &loam_egui::egui::Context) -> loam_egui::egui::Pos2 {
    let area = ctx.available_rect();
    loam_egui::egui::pos2(
        (area.right() - OVERLAY_MARGIN - OVERLAY_WIDTH).max(area.left()),
        area.top() + OVERLAY_MARGIN,
    )
}

impl PerfOverlay {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_toggle_key(mut self, key: loam_egui::egui::Key) -> Self {
        self.toggle_key = key;
        self
    }

    pub fn always_visible(mut self) -> Self {
        self.visible = true;
        self
    }

    pub fn show(&mut self, ctx: &loam_egui::egui::Context) {
        use loam_egui::egui;

        if ctx.input(|i| i.key_pressed(self.toggle_key)) {
            self.visible = !self.visible;
        }
        if !self.visible {
            return;
        }

        let window = self.window.min(MAX_WINDOW);
        let mut cadence = StackBuf::new();
        let mut frames_buf = StackBuf::new();
        let mut idles = StackBuf::new();
        let mut heap_count = 0usize;
        let mut heap_peak = 0i64;
        let mut heap_net = 0i64;
        let mut alloc_frames = 0usize;
        let mut alloc_count_sum: u64 = 0;
        let mut alloc_peak_bytes: u64 = 0;
        let mut alloc_net_bytes: i64 = 0;
        let mut any = false;

        frame_trace::with_history(|history| {
            let start = history.len().saturating_sub(window);
            for frame in history.iter().skip(start) {
                any = true;
                for section in &frame.sections {
                    match section.name {
                        "between-frames" => cadence.push(section.elapsed),
                        "frame" => frames_buf.push(section.elapsed),
                        "idle" => idles.push(section.elapsed),
                        _ => {}
                    }
                }
                if let Some(d) = frame.heap_delta_bytes {
                    heap_count += 1;
                    if d > heap_peak {
                        heap_peak = d;
                    }
                    heap_net = heap_net.saturating_add(d);
                }
                if let Some(a) = frame.allocs {
                    alloc_frames += 1;
                    alloc_count_sum = alloc_count_sum.saturating_add(a.alloc_count);
                    if a.alloc_bytes > alloc_peak_bytes {
                        alloc_peak_bytes = a.alloc_bytes;
                    }
                    alloc_net_bytes = alloc_net_bytes.saturating_add(a.net_bytes);
                }
            }
        });

        if !any {
            return;
        }

        let cadence_mean = cadence.mean();
        let cadence_p99 = cadence.percentile(0.99);
        let frame_mean = frames_buf.mean();
        let frame_p99 = frames_buf.percentile(0.99);
        let idle_mean = idles.mean();
        let idle_p99 = idles.percentile(0.99);

        let cadence_max_ever = frame_trace::max_ever("between-frames");
        let idle_max_ever = frame_trace::max_ever("idle");
        let frame_max_ever = frame_trace::max_ever("frame");

        let fps = if cadence_mean.as_secs_f32() > 0.0 {
            1.0 / cadence_mean.as_secs_f32()
        } else {
            0.0
        };

        // `anchor` ignores drag; `default_pos` lets egui persist the dragged offset.
        egui::Area::new(egui::Id::new("loam-perf-overlay"))
            .default_pos(perf_overlay_seat(ctx))
            .movable(true)
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style())
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(60, 60, 75)))
                    .show(ui, |ui| {
                        ui.set_min_width(OVERLAY_WIDTH);
                        let mono = egui::FontId::monospace(11.0);
                        let label_color = egui::Color32::from_rgb(180, 190, 200);
                        ui.label(
                            egui::RichText::new(format!("FPS    {fps:5.1}"))
                                .font(mono.clone())
                                .color(egui::Color32::from_rgb(220, 230, 240)),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "total  {:>5.1}  p99 {:>5.1}  ms",
                                cadence_mean.as_secs_f32() * 1000.0,
                                cadence_p99.as_secs_f32() * 1000.0,
                            ))
                            .font(mono.clone())
                            .color(label_color),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "idle   {:>5.1}  p99 {:>5.1}  ms",
                                idle_mean.as_secs_f32() * 1000.0,
                                idle_p99.as_secs_f32() * 1000.0,
                            ))
                            .font(mono.clone())
                            .color(label_color),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "frame  {:>5.2}  p99 {:>5.2}  ms",
                                frame_mean.as_secs_f32() * 1000.0,
                                frame_p99.as_secs_f32() * 1000.0,
                            ))
                            .font(mono.clone())
                            .color(label_color),
                        );
                        ui.separator();
                        ui.label(
                            egui::RichText::new("worst-ever (session)")
                                .font(mono.clone())
                                .color(egui::Color32::from_rgb(140, 150, 160)),
                        );
                        let worst_color = |d: Duration| {
                            let ms = d.as_secs_f32() * 1000.0;
                            if ms >= 100.0 {
                                egui::Color32::from_rgb(220, 100, 80)
                            } else if ms >= 50.0 {
                                egui::Color32::from_rgb(220, 180, 90)
                            } else {
                                label_color
                            }
                        };
                        ui.label(
                            egui::RichText::new(format!(
                                "total  {:>6.1} ms",
                                cadence_max_ever.as_secs_f32() * 1000.0,
                            ))
                            .font(mono.clone())
                            .color(worst_color(cadence_max_ever)),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "idle   {:>6.1} ms",
                                idle_max_ever.as_secs_f32() * 1000.0,
                            ))
                            .font(mono.clone())
                            .color(worst_color(idle_max_ever)),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "frame  {:>6.1} ms",
                                frame_max_ever.as_secs_f32() * 1000.0,
                            ))
                            .font(mono.clone())
                            .color(worst_color(frame_max_ever)),
                        );
                        if alloc_frames > 0 {
                            ui.separator();
                            ui.label(
                                egui::RichText::new("allocs (Rust heap)")
                                    .font(mono.clone())
                                    .color(egui::Color32::from_rgb(140, 150, 160)),
                            );
                            let mean_count = alloc_count_sum / alloc_frames as u64;
                            let count_color = |n: u64| {
                                if n >= 1_000 {
                                    egui::Color32::from_rgb(220, 100, 80)
                                } else if n >= 100 {
                                    egui::Color32::from_rgb(220, 180, 90)
                                } else if n >= 10 {
                                    egui::Color32::from_rgb(180, 200, 130)
                                } else {
                                    egui::Color32::from_rgb(120, 200, 130)
                                }
                            };
                            let byte_color = |bytes: i64| {
                                let mb = bytes.abs() as f32 / (1024.0 * 1024.0);
                                if mb >= 10.0 {
                                    egui::Color32::from_rgb(220, 100, 80)
                                } else if mb >= 1.0 {
                                    egui::Color32::from_rgb(220, 180, 90)
                                } else {
                                    label_color
                                }
                            };
                            ui.label(
                                egui::RichText::new(format!("mean  {mean_count:>6} allocs/frame",))
                                    .font(mono.clone())
                                    .color(count_color(mean_count)),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "peak  {:>+6.2} KB/frame",
                                    alloc_peak_bytes as f32 / 1024.0,
                                ))
                                .font(mono.clone())
                                .color(byte_color(alloc_peak_bytes as i64)),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "net   {:>+6.2} MB / window",
                                    alloc_net_bytes as f32 / (1024.0 * 1024.0),
                                ))
                                .font(mono.clone())
                                .color(byte_color(alloc_net_bytes)),
                            );
                        }
                        if heap_count > 0 {
                            ui.separator();
                            ui.label(
                                egui::RichText::new("heap (Chromium)")
                                    .font(mono.clone())
                                    .color(egui::Color32::from_rgb(140, 150, 160)),
                            );
                            let heap_color = |bytes: i64| {
                                let mb = bytes.abs() as f32 / (1024.0 * 1024.0);
                                if mb >= 10.0 {
                                    egui::Color32::from_rgb(220, 100, 80)
                                } else if mb >= 2.0 {
                                    egui::Color32::from_rgb(220, 180, 90)
                                } else {
                                    label_color
                                }
                            };
                            ui.label(
                                egui::RichText::new(format!(
                                    "peak  {:>+6.2} MB/frame",
                                    heap_peak as f32 / (1024.0 * 1024.0),
                                ))
                                .font(mono.clone())
                                .color(heap_color(heap_peak)),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "net   {:>+6.2} MB / window",
                                    heap_net as f32 / (1024.0 * 1024.0),
                                ))
                                .font(mono.clone())
                                .color(heap_color(heap_net)),
                            );
                        }
                        ui.separator();
                        draw_sparkline(ui, cadence.as_slice());
                    });
            });
    }
}

/// 256 samples × 16 B × three buffers is 12 KB of stack per `show`.
pub const MAX_WINDOW: usize = 256;

#[derive(Clone)]
struct StackBuf {
    samples: [Duration; MAX_WINDOW],
    len: usize,
}

impl StackBuf {
    fn new() -> Self {
        Self {
            samples: [Duration::ZERO; MAX_WINDOW],
            len: 0,
        }
    }

    fn push(&mut self, d: Duration) {
        if self.len < MAX_WINDOW {
            self.samples[self.len] = d;
            self.len += 1;
        }
    }

    fn as_slice(&self) -> &[Duration] {
        &self.samples[..self.len]
    }

    fn mean(&self) -> Duration {
        if self.len == 0 {
            return Duration::ZERO;
        }
        let sum: Duration = self.samples[..self.len].iter().sum();
        sum / self.len as u32
    }

    // Sorts a copy: the sparkline reads `self` in time order.
    fn percentile(&self, q: f32) -> Duration {
        if self.len == 0 {
            return Duration::ZERO;
        }
        let mut local: [Duration; MAX_WINDOW] = self.samples;
        local[..self.len].sort();
        let idx = ((self.len as f32 * q) as usize).min(self.len - 1);
        local[idx]
    }
}

fn draw_sparkline(ui: &mut loam_egui::egui::Ui, gaps: &[Duration]) {
    use loam_egui::egui;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(240.0, 36.0), egui::Sense::hover());
    if gaps.is_empty() {
        return;
    }
    let painter = ui.painter();
    painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(18, 18, 24));

    // Outliers clamp to the top so one spike does not squash the baseline.
    let y_max_ms = 50.0_f32;
    let y_for_ms = |ms: f32| {
        let clamped = ms.min(y_max_ms);
        rect.bottom() - (clamped / y_max_ms) * rect.height()
    };
    let ref_60 = y_for_ms(16.67);
    let ref_30 = y_for_ms(33.33);
    painter.line_segment(
        [
            egui::pos2(rect.left(), ref_60),
            egui::pos2(rect.right(), ref_60),
        ],
        egui::Stroke::new(0.5, egui::Color32::from_rgb(60, 100, 70)),
    );
    painter.line_segment(
        [
            egui::pos2(rect.left(), ref_30),
            egui::pos2(rect.right(), ref_30),
        ],
        egui::Stroke::new(0.5, egui::Color32::from_rgb(120, 90, 60)),
    );

    let n = gaps.len() as f32;
    let dx = rect.width() / n.max(1.0);
    let bar_w = dx.max(1.0);
    for (i, gap) in gaps.iter().enumerate() {
        let x = rect.left() + i as f32 * dx;
        let ms = gap.as_secs_f32() * 1000.0;
        let y = y_for_ms(ms);
        let color = if ms > 33.3 {
            egui::Color32::from_rgb(220, 100, 80)
        } else if ms > 20.0 {
            egui::Color32::from_rgb(220, 180, 90)
        } else {
            egui::Color32::from_rgb(120, 200, 130)
        };
        painter.line_segment(
            [egui::pos2(x, rect.bottom()), egui::pos2(x, y)],
            egui::Stroke::new(bar_w, color),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_dur_emits_ns_under_microsecond() {
        assert!(fmt_dur(Duration::from_nanos(0)).ends_with("ns"));
        assert!(fmt_dur(Duration::from_nanos(999)).ends_with("ns"));
    }

    #[test]
    fn fmt_dur_emits_us_under_millisecond() {
        assert!(fmt_dur(Duration::from_nanos(1_000)).ends_with("us"));
        assert!(fmt_dur(Duration::from_nanos(999_999)).ends_with("us"));
    }

    #[test]
    fn fmt_dur_emits_ms_under_second() {
        assert!(fmt_dur(Duration::from_nanos(1_000_000)).ends_with("ms"));
        assert!(fmt_dur(Duration::from_nanos(999_999_999)).ends_with("ms"));
    }

    #[test]
    fn fmt_dur_emits_seconds_above() {
        assert!(fmt_dur(Duration::from_secs(1)).ends_with('s'));
        assert!(!fmt_dur(Duration::from_secs(1)).ends_with("ms"));
        assert!(!fmt_dur(Duration::from_secs(1)).ends_with("us"));
    }

    #[test]
    fn truncate_preserves_short_names_and_marks_long_ones() {
        assert_eq!(truncate("frame", 18), "frame");
        assert_eq!(truncate("abcdefghijklmnopqr", 18).len(), 18);
        let long = "supercalifragilisticexpialidocious";
        let t = truncate(long, 18);
        assert_eq!(t.len(), 18);
        assert!(t.ends_with('~'), "truncate should mark with `~`");
    }

    fn frame_of(sections: &[(&'static str, u64)]) -> frame_trace::FrameTrace {
        frame_trace::FrameTrace {
            sections: sections
                .iter()
                .map(|(name, us)| frame_trace::Section {
                    name,
                    elapsed: Duration::from_micros(*us),
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn unscoped_is_the_frame_minus_its_frame_loop_sections() {
        let frame = frame_of(&[("frame", 4000), ("app-record", 130), ("present", 40)]);
        assert_eq!(unscoped(&frame), Some(Duration::from_micros(3830)));
    }

    #[test]
    fn unscoped_ignores_sections_nested_inside_frame_loop_sections() {
        let flat = frame_of(&[("frame", 4000), ("app-record", 130)]);
        let nested = frame_of(&[("frame", 4000), ("app-record", 130), ("pp-sdf", 94)]);
        assert_eq!(unscoped(&nested), unscoped(&flat));
    }

    #[test]
    fn unscoped_ignores_the_sections_that_do_not_nest_in_the_frame() {
        let frame = frame_of(&[
            ("frame", 4000),
            ("app-record", 130),
            ("between-frames", 4130),
            ("idle", 110),
            ("gpu-total", 800),
        ]);
        assert_eq!(unscoped(&frame), Some(Duration::from_micros(3870)));
    }

    #[test]
    fn unscoped_is_absent_for_a_frame_without_a_frame_section() {
        let frame = frame_of(&[("app-record", 130), ("present", 40)]);
        assert_eq!(unscoped(&frame), None);
    }

    #[test]
    fn unscoped_saturates_at_zero_when_the_children_overrun_the_parent() {
        let frame = frame_of(&[("frame", 100), ("app-record", 130)]);
        assert_eq!(unscoped(&frame), Some(Duration::ZERO));
    }

    #[test]
    fn unscoped_row_reports_frame_time_that_no_section_claims() {
        {
            let _frame = frame_trace::scope("frame");
            std::thread::sleep(Duration::from_millis(4));
        }
        frame_trace::end_frame();
        let row = unscoped_stats().expect("one recorded frame yields the row");
        assert_eq!(row.samples, 1);
        assert!(
            row.max >= Duration::from_millis(4),
            "a frame whose work is entirely unscoped must report as unscoped, got {:?}",
            row.max,
        );
    }

    // 25 frames put p50, p95 and p99 at distinct nearest ranks (12, 23, 24).
    const RANK_SPREAD_FRAMES: u32 = 25;

    #[test]
    fn unscoped_percentiles_agree_with_the_aggregate_they_are_shown_beside() {
        for i in 0..RANK_SPREAD_FRAMES {
            frame_trace::begin_frame();
            {
                let _frame = frame_trace::scope("frame");
                let until = web_time::Instant::now() + Duration::from_micros(20 * (i as u64 + 1));
                while web_time::Instant::now() < until {}
            }
            frame_trace::end_frame();
        }
        let residual = unscoped_stats().expect("recorded frames yield the row");
        let frame_row = frame_trace::aggregate()
            .into_iter()
            .find(|s| s.name == "frame")
            .expect("the recorded frames carry a `frame` section");
        assert_eq!(residual.samples, frame_row.samples);
        assert_eq!(residual.mean, frame_row.mean);
        assert_eq!(residual.p50, frame_row.p50);
        assert_eq!(residual.p95, frame_row.p95);
        assert_eq!(residual.p99, frame_row.p99);
        assert_eq!(residual.max, frame_row.max);
        assert_ne!(
            frame_row.p95, frame_row.p99,
            "distinct samples must separate the ranks, else the equalities \
             above hold for any quantile",
        );
    }

    #[test]
    fn stackbuf_starts_empty() {
        let buf = StackBuf::new();
        assert_eq!(buf.as_slice().len(), 0);
        assert_eq!(buf.mean(), Duration::ZERO);
        assert_eq!(buf.percentile(0.5), Duration::ZERO);
        assert_eq!(buf.percentile(0.99), Duration::ZERO);
    }

    #[test]
    fn stackbuf_push_silently_drops_past_max_window() {
        let mut buf = StackBuf::new();
        for i in 0..(MAX_WINDOW + 10) {
            buf.push(Duration::from_nanos(i as u64));
        }
        assert_eq!(
            buf.as_slice().len(),
            MAX_WINDOW,
            "push beyond cap must not allocate; samples drop"
        );
    }

    #[test]
    fn stackbuf_percentile_picks_nearest_rank() {
        let mut buf = StackBuf::new();
        for ms in 1..=10u64 {
            buf.push(Duration::from_millis(ms));
        }
        // floor(10 * 0.5) = 5 -> samples[5] = 6ms.
        assert_eq!(buf.percentile(0.5), Duration::from_millis(6));
        assert_eq!(buf.percentile(0.95), Duration::from_millis(10));
        // p99 clamps to len-1.
        assert_eq!(buf.percentile(0.99), Duration::from_millis(10));
    }

    #[test]
    fn stackbuf_percentile_is_order_preserving_on_self() {
        let mut buf = StackBuf::new();
        for ms in [30u64, 5, 25, 10, 20] {
            buf.push(Duration::from_millis(ms));
        }
        let before: Vec<Duration> = buf.as_slice().to_vec();
        let _ = buf.percentile(0.5);
        let after: Vec<Duration> = buf.as_slice().to_vec();
        assert_eq!(before, after, "percentile must not reorder self");
    }
}

#[cfg(test)]
mod seat_tests {
    use super::*;
    use loam_egui::egui;

    // Taller than `OVERLAY_MARGIN`, so the two rects separate.
    const BAR_HEIGHT: f32 = 64.0;

    const PANEL_WIDTH: f32 = 120.0;

    // Wide enough that the seat's left clamp stays untaken.
    const VIEWPORT: egui::Vec2 = egui::vec2(1280.0, 800.0);

    // Narrower than `PANEL_WIDTH + OVERLAY_MARGIN + OVERLAY_WIDTH`, so the clamp fires.
    const NARROW_VIEWPORT: egui::Vec2 = egui::vec2(200.0, 400.0);

    // egui's fallback viewport is ~10000 px wide, which leaves the clamp unreachable.
    fn viewport(size: egui::Vec2) -> egui::RawInput {
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
            ..Default::default()
        }
    }

    #[test]
    fn perf_overlay_boots_into_the_top_right_of_the_panel_band() {
        frame_trace::begin_frame();
        frame_trace::end_frame();

        let ctx = egui::Context::default();
        let mut overlay = PerfOverlay::new().always_visible();
        let mut bar_bottom = 0.0;
        let mut band = egui::Rect::NOTHING;
        let _ = ctx.run(viewport(VIEWPORT), |ctx| {
            bar_bottom = egui::TopBottomPanel::top("shell-menu-bar")
                .exact_height(BAR_HEIGHT)
                .show(ctx, |ui| {
                    ui.label("bar");
                })
                .response
                .rect
                .bottom();
            band = ctx.available_rect();
            overlay.show(ctx);
        });
        let rect = ctx
            .memory(|m| m.area_rect(egui::Id::new("loam-perf-overlay")))
            .expect("overlay area is registered once shown");
        assert!(
            rect.top() >= bar_bottom,
            "overlay top {} must clear the menu bar bottom {bar_bottom}",
            rect.top(),
        );
        // egui constrains a placed area to the viewport, so exact insets are not stable.
        assert!(
            rect.center().x > band.center().x,
            "overlay center x {} must sit in the band's right half of {band:?}",
            rect.center().x,
        );
        assert!(
            rect.center().y < band.center().y,
            "overlay center y {} must sit in the band's top half of {band:?}",
            rect.center().y,
        );
    }

    #[test]
    fn perf_overlay_seat_insets_from_the_top_right_of_the_panel_band() {
        let ctx = egui::Context::default();
        let mut seat = None;
        let mut band = egui::Rect::NOTHING;
        let _ = ctx.run(viewport(VIEWPORT), |ctx| {
            egui::SidePanel::left("left")
                .exact_width(PANEL_WIDTH)
                .show(ctx, |ui| {
                    ui.label("left");
                });
            egui::SidePanel::right("right")
                .exact_width(PANEL_WIDTH)
                .show(ctx, |ui| {
                    ui.label("right");
                });
            band = ctx.available_rect();
            seat = Some(perf_overlay_seat(ctx));
        });
        let seat = seat.expect("run closure sets the seat");
        assert_eq!(
            seat.x,
            band.right() - OVERLAY_MARGIN - OVERLAY_WIDTH,
            "seat must inset from the band's right edge, not its left, in {band:?}",
        );
        assert_eq!(
            seat.y,
            band.top() + OVERLAY_MARGIN,
            "seat must inset from the band's top edge, not its bottom, in {band:?}",
        );
    }

    #[test]
    fn perf_overlay_seat_clamps_to_the_band_left_when_the_band_is_narrower_than_the_readout() {
        let ctx = egui::Context::default();
        let mut seat = None;
        let mut band = egui::Rect::NOTHING;
        let _ = ctx.run(viewport(NARROW_VIEWPORT), |ctx| {
            egui::SidePanel::left("left")
                .exact_width(PANEL_WIDTH)
                .show(ctx, |ui| {
                    ui.label("left");
                });
            band = ctx.available_rect();
            seat = Some(perf_overlay_seat(ctx));
        });
        let seat = seat.expect("run closure sets the seat");
        assert!(
            band.width() < OVERLAY_MARGIN + OVERLAY_WIDTH && band.left() > 0.0,
            "fixture must leave a band {band:?} too narrow for the readout and \
             offset from the viewport origin, or the clamp goes untested",
        );
        assert_eq!(
            seat.x,
            band.left(),
            "a band too narrow for the readout must seat it at the band's left \
             edge, not at the viewport origin or off-screen",
        );
    }
}
