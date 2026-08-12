//! The application-wide command queue: one place mutations are submitted, one
//! point per frame where they are applied.
//!
//! ## The contract
//!
//! **Commands stamped with tick T are applied, in submission order, before tick
//! T runs, however the frames were paced.** Both runners drain immediately
//! before the runner's `drive_fixed_ticks` and stamp with the `tick_index` that call
//! is about to start from, so a frame that runs zero ticks hands its commands the
//! same T as the next frame's and concatenation preserves order. That is what
//! makes the position of a mutation in simulation time a property of the
//! recording rather than of how many catch-up ticks a frame happened to run.
//!
//! ## Who submits
//!
//! [`submit`] takes no application context by construction, so a console line, a
//! bound key, a menu item and a `--script` line are the same caller. The console
//! keeps its typed registry and becomes a producer: `Console::execute` parks a
//! registry line, the host forwards it here, and `Console::dispatch` runs it from
//! inside the App hook. Console built-ins (`help`, `clear`, `detach`, `dock`)
//! take no `Ctx` and never enter the queue.
//!
//! ## Where a drained line goes
//!
//! Engine verbs first, through the runner's own verb table; it owns those
//! and no App is consulted for them. Everything else reaches
//! [`App::apply_command`], which `SceneShell` fans out
//! to the active [`Scene::apply_command`](crate::shell::Scene::apply_command).
//!
//! Output produced at the drain has no console in scope (the runner does not own
//! one), so it lands in a buffer the host pumps into its scrollback, the same
//! shape [`crate::log`] uses for mirrored tracing events.

use std::sync::Mutex;

use loam_egui::{Console, ConsoleWriter, HistoryLine};
use loam_render::device::RenderDevice;

use crate::App;

/// A parsed command invocation. Tokenized with `loam_console::parse_line`, so the
/// queue and the console share one grammar rather than agreeing by accident.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandLine {
    pub name: String,
    pub args: Vec<String>,
}

impl CommandLine {
    /// Parse one command line, or `None` when it holds no tokens.
    pub fn parse(line: &str) -> Option<Self> {
        let (name, args) = loam_egui::parse_line(line)?;
        Some(Self { name, args })
    }

    /// Args as the `&[&str]` the console registry dispatches against.
    pub fn arg_refs(&self) -> Vec<&str> {
        self.args.iter().map(String::as_str).collect()
    }
}

/// A queued command paired with the tick index its drain precedes. Crate-only
/// because [`drain`] is: a third drain point would be a second ordering, and
/// the stamp would then describe only part of the frame's mutations. Appliers
/// read the stamp off [`CommandCtx::tick`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StampedCommand {
    /// The command is applied before this tick runs.
    pub tick: u64,
    pub command: CommandLine,
}

static QUEUE: Mutex<Vec<CommandLine>> = Mutex::new(Vec::new());

/// Buffered output from commands applied at the drain. Drained by the host into
/// a console; see [`pump_into`].
static OUTPUT: Mutex<Vec<HistoryLine>> = Mutex::new(Vec::new());

/// Buffered-output cap, so a host that submits without ever pumping cannot grow
/// this without bound. Oldest lines drop first, as the console's own scrollback
/// does at `MAX_HISTORY_LINES`.
const MAX_BUFFERED_OUTPUT: usize = 256;

/// Submit a command for the next drain.
pub fn submit(command: CommandLine) {
    QUEUE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(command);
}

/// Parse and submit a raw line. Returns false when the line held no tokens.
pub fn submit_line(line: &str) -> bool {
    match CommandLine::parse(line) {
        Some(command) => {
            submit(command);
            true
        }
        None => false,
    }
}

/// Take everything queued, stamped with the tick the caller is about to run.
/// Called from exactly two places, the two runners' pre-tick drain.
pub(crate) fn drain(tick: u64) -> Vec<StampedCommand> {
    let queued = std::mem::take(&mut *QUEUE.lock().unwrap_or_else(|e| e.into_inner()));
    queued
        .into_iter()
        .map(|command| StampedCommand { tick, command })
        .collect()
}

/// Per-application context, held for one drained command.
pub struct CommandCtx<'a> {
    /// In scope at the drain point in both runners, and the reason a command that
    /// has to rebuild a pipeline does not need a second deferred path.
    pub rd: &'a RenderDevice,
    /// The tick this command was stamped for; it has not run yet.
    pub tick: u64,
    /// Where an applier writes user-facing output. Drained into the module's
    /// bounded output buffer after
    /// the command returns; a target that owns a console (a scene dispatching
    /// against its own registry) writes to that instead and leaves this empty.
    pub out: &'a mut ConsoleWriter,
}

/// Engine verbs the runner owns, consulted before the App hook. Returns false
/// when nothing here claims the name, which is the whole precedence rule: a
/// claimed name never reaches an App.
///
/// One entry today. The other engine commands (`fps`, `trace`, `capture`,
/// `scene`, `log`, `version`) still reach their module statics through the
/// console registry and are convertible to this table one at a time.
fn apply_engine_verb(command: &CommandLine, out: &mut ConsoleWriter) -> bool {
    match command.name.as_str() {
        "vsync" => {
            crate::vsync::apply(&command.arg_refs(), out);
            true
        }
        _ => false,
    }
}

/// Drain and apply. The single application point; both runners call this
/// immediately before their `drive_fixed_ticks`.
pub(crate) fn apply_drained<A: App>(app: &mut A, rd: &RenderDevice, tick: u64) {
    for stamped in drain(tick) {
        let mut writer = ConsoleWriter::new();
        let result = if apply_engine_verb(&stamped.command, &mut writer) {
            Ok(())
        } else {
            let mut ctx = CommandCtx {
                rd,
                tick: stamped.tick,
                out: &mut writer,
            };
            app.apply_command(&stamped.command, &mut ctx)
        };
        let mut lines = writer.take_lines();
        // An `Err` from one command must not strand the ones behind it: the loop
        // records and continues, so a bad line cannot wedge the surface. It also
        // reaches tracing, because a `--script` run is unattended and a
        // scrollback nobody is looking at is not a report.
        if let Err(e) = result {
            let text = format!("error: {}: {e:#}", stamped.command.name);
            tracing::error!("command: {text}");
            lines.push(HistoryLine::error(text));
        }
        if !lines.is_empty() {
            let mut buffered = OUTPUT.lock().unwrap_or_else(|e| e.into_inner());
            buffered.extend(lines);
            let overflow = buffered.len().saturating_sub(MAX_BUFFERED_OUTPUT);
            buffered.drain(..overflow);
        }
    }
}

/// Take output produced by commands applied at the drain.
pub fn drain_output() -> Vec<HistoryLine> {
    std::mem::take(&mut *OUTPUT.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Pump that output into a console's scrollback. Call once per frame before the
/// console paints, beside [`crate::log::pump_into`]; this frame's drain already
/// ran, so what it wrote shows this frame.
pub fn pump_into<Ctx: 'static>(console: &mut Console<Ctx>) {
    for line in drain_output() {
        console.write(line);
    }
}

/// Hand the console the queue's producer end: everything `Console::execute`
/// accepted becomes a submission. Call once per frame after the console's UI
/// runs, so a line typed this frame applies at the next frame's drain rather
/// than the one after.
pub fn forward_pending<Ctx: 'static>(console: &mut Console<Ctx>) {
    for line in console.drain_pending() {
        submit_line(&line);
    }
}

/// The queue is process-global, as `capture::QUEUE` and `shell::SWITCHER`
/// already are, and cargo runs tests on parallel threads. A test that submits
/// holds this.
#[cfg(test)]
pub(crate) static TEST_LOCK: Mutex<()> = Mutex::new(());

/// Accept a line on `console` and dispatch what it queued, collapsing the host
/// and the drain into one call. For tests about what a command does; the ones
/// about when it runs drive [`drain`] themselves.
#[cfg(test)]
pub(crate) fn run_on_console<Ctx: 'static>(console: &mut Console<Ctx>, line: &str, ctx: &mut Ctx) {
    console.execute(line);
    for pending in console.drain_pending() {
        if let Some(parsed) = CommandLine::parse(&pending) {
            console.dispatch(&parsed.name, &parsed.arg_refs(), ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exclusive<T>(f: impl FnOnce() -> T) -> T {
        let _held = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = drain(0);
        let _ = drain_output();
        f()
    }

    #[test]
    fn a_drained_entry_carries_the_tick_its_drain_precedes() {
        exclusive(|| {
            submit_line("alpha 1");
            submit_line("beta");
            let drained = drain(7);
            assert_eq!(
                drained,
                [
                    StampedCommand {
                        tick: 7,
                        command: CommandLine {
                            name: "alpha".into(),
                            args: vec!["1".into()],
                        },
                    },
                    StampedCommand {
                        tick: 7,
                        command: CommandLine {
                            name: "beta".into(),
                            args: Vec::new(),
                        },
                    },
                ]
            );
            assert!(drain(8).is_empty(), "a drained command is not re-served");
        });
    }

    /// The queue's grammar is the console's, so a quoted argument survives
    /// submission as one token rather than splitting at the space.
    #[test]
    fn a_submitted_line_is_tokenized_with_the_console_grammar() {
        let parsed = CommandLine::parse(r#"load "5 cell" fast"#).expect("tokens");
        assert_eq!(parsed.name, "load");
        assert_eq!(parsed.args, ["5 cell", "fast"]);
        assert!(CommandLine::parse("   ").is_none());
        exclusive(|| {
            assert!(!submit_line("   "), "a blank line queues nothing");
            assert!(drain(0).is_empty());
        });
    }

    /// A claimed name has to be the applied one and an unclaimed name has to
    /// leave engine state untouched; those two together are what
    /// `apply_drained` reads when it decides whether to consult the App hook.
    #[test]
    fn the_engine_table_claims_vsync_and_leaves_unknown_verbs_to_the_app_hook() {
        let _held = crate::frame_pacing::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _ = crate::frame_pacing::take_pending_vsync();

        let mut out = ConsoleWriter::new();
        assert!(apply_engine_verb(
            &CommandLine::parse("vsync off").expect("tokens"),
            &mut out
        ));
        assert_eq!(
            crate::frame_pacing::take_pending_vsync(),
            Some(false),
            "the claim has to be the applied one, not just a name match"
        );

        assert!(!apply_engine_verb(
            &CommandLine::parse("throw 0").expect("tokens"),
            &mut out
        ));
        assert_eq!(
            crate::frame_pacing::take_pending_vsync(),
            None,
            "an unclaimed verb must leave engine state alone"
        );
    }

    /// Shipped code of `source`: comments dropped, because the prose names the
    /// very call sites the scan counts, and the test module dropped with it.
    fn shipped_code(source: &str) -> String {
        source
            .split("#[cfg(test)]")
            .next()
            .expect("split yields the head")
            .lines()
            .map(|line| line.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn call_sites(code: &str, needle: &str) -> Vec<usize> {
        code.match_indices(needle).map(|(at, _)| at).collect()
    }

    /// Criterion: one drain point, ahead of the ticks it stamps for, in each
    /// runner. Read off the source because neither frame loop is reachable
    /// from a headless test: the windowed one needs a surface and the worker
    /// one a browser. A behavioural test that rebuilds the frame order in the
    /// harness pins the queue, not the placement, and stays green when the
    /// call moves.
    #[test]
    fn each_runner_drains_once_and_before_its_ticks() {
        for (runner, source) in [
            ("windowed runner", include_str!("lib.rs")),
            ("wasm worker", include_str!("wasm/worker.rs")),
        ] {
            let code = shipped_code(source);
            let drains = call_sites(&code, "apply_drained(");
            let ticks = call_sites(&code, "drive_fixed_ticks(");
            assert_eq!(
                drains.len(),
                1,
                "{runner}: a second drain is a second ordering, and the stamp \
                 then describes only part of the frame's mutations"
            );
            assert_eq!(
                ticks.len(),
                1,
                "{runner}: the scan needs exactly one tick call to order the \
                 drain against"
            );
            assert!(
                drains[0] < ticks[0],
                "{runner}: the drain must precede the ticks, or a command is \
                 applied after the batch it was stamped ahead of"
            );
        }
    }
}
