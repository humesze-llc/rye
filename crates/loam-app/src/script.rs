//! Frame-indexed console scripts: `--script=<path>`.
//!
//! A script file is lines of `<frame> <console command>`. The driver runs
//! each command through the host's existing [`Console`] registry on the frame
//! whose index the line names, then asks the runner to exit a fixed margin
//! past the last line. Reaching a configured state costs one flag instead of
//! a click-through, and two runs issue the same commands on the same frame
//! numbers, which is what makes a before/after pixel hash a diff rather than
//! a coincidence.
//!
//! No new command language: the tokenizer, the registry lookup and the
//! dispatch are [`Console::execute`]'s, so a script line is exactly what the
//! user would type.
//!
//! ## Timebase
//!
//! [`ScriptDriver::frame`] counts calls to [`ScriptDriver::advance`], and that
//! counter is the whole timeline. Nothing on the path reads a clock, so a
//! frame that renders slowly still runs the same commands at the same index;
//! `the_script_timeline_reads_no_clock` pins the absence in this module by
//! scanning the source, and the dispatch it hands off to carries none either
//! (`rg 'Instant|SystemTime|Duration|elapsed' crates/loam-console/src` is
//! empty).
//!
//! ## Focus and synthesized input
//!
//! [`ScriptDriver::advance`] takes a `&mut Console<Ctx>` and a `&mut Ctx` and
//! nothing else. That signature is the fence: with no `Window` and no
//! event-loop handle in scope, cursor warping, focus requests and event
//! injection are not callable from here, and every state change a script
//! makes goes through a command handler that can only touch `Ctx`. Widening
//! that one signature to carry a window handle is the single change that
//! would end the property; `the_driver_names_no_window_or_input_api` fails if
//! this module ever mentions one.
//!
//! ## Exiting
//!
//! A script has no `ActiveEventLoop` to leave, so completion is published
//! through [`request_exit`] and the runner reads [`exit_requested`] at the
//! top of its redraw. Same publish/drain shape as [`crate::capture`]'s
//! request queue.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{anyhow, bail, Context as _, Result};

use loam_egui::Console;

/// Frames the driver keeps running past the last scheduled command before it
/// asks for exit. Long enough for that command's effect to reach the
/// swapchain and for a streaming capture started by it to write something;
/// a script that needs a longer tail schedules a later line.
const SETTLE_FRAMES: u64 = 60;

/// One scheduled line: the frame it fires on and the command line to run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScriptStep {
    pub frame: u64,
    pub command: String,
}

/// A parsed script. Steps keep file order and their frame indices never
/// decrease, so "run everything due, in order" is total.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Script {
    steps: Vec<ScriptStep>,
}

impl Script {
    /// Parse `<frame> <command>` lines. Blank lines and lines whose first
    /// non-space character is `#` are ignored; anything else that is not a
    /// frame index followed by a command is an error naming the line, because
    /// a script that quietly runs a subset of itself is worse than one that
    /// refuses to start.
    ///
    /// Only a leading `#` comments: truncating at a mid-line `#` would eat a
    /// command argument that legitimately contains one.
    pub fn parse(source: &str) -> Result<Self> {
        let mut steps: Vec<ScriptStep> = Vec::new();
        for (index, raw) in source.lines().enumerate() {
            let number = index + 1;
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (frame_token, rest) = line
                .split_once(char::is_whitespace)
                .ok_or_else(|| anyhow!("line {number}: `{line}` names a frame and no command"))?;
            let frame: u64 = frame_token
                .parse()
                .map_err(|e| anyhow!("line {number}: `{frame_token}` is not a frame index: {e}"))?;
            let command = rest.trim();
            if command.is_empty() {
                bail!("line {number}: `{line}` names a frame and no command");
            }
            if let Some(previous) = steps.last() {
                if frame < previous.frame {
                    bail!(
                        "line {number}: frame {frame} is behind the previous line's \
                         {}; the playhead only moves forward, so this line could \
                         never fire",
                        previous.frame
                    );
                }
            }
            steps.push(ScriptStep {
                frame,
                command: command.to_string(),
            });
        }
        Ok(Self { steps })
    }

    /// Read and parse a script file. The path rides along in the error so a
    /// reported line number is attributable.
    pub fn load(path: &Path) -> Result<Self> {
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("reading script {}", path.display()))?;
        Self::parse(&source).with_context(|| format!("parsing script {}", path.display()))
    }

    pub fn steps(&self) -> &[ScriptStep] {
        &self.steps
    }

    /// Frame the last step fires on; 0 when the script schedules nothing.
    pub fn last_frame(&self) -> u64 {
        self.steps.last().map_or(0, |step| step.frame)
    }
}

/// Whether the script still has work (or settling) left.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScriptStatus {
    Running,
    Finished,
}

/// Plays a [`Script`] against a console, one frame per [`Self::advance`].
#[derive(Debug)]
pub struct ScriptDriver {
    script: Script,
    /// Index of the first step that has not fired.
    next: usize,
    frame: u64,
    exit_frame: u64,
}

impl ScriptDriver {
    pub fn new(script: Script) -> Self {
        let exit_frame = script.last_frame() + SETTLE_FRAMES;
        Self {
            script,
            next: 0,
            frame: 0,
            exit_frame,
        }
    }

    /// Index of the frame the next [`Self::advance`] will run.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Run every command due on the current frame, in file order, then step
    /// the playhead. Steps whose frame the playhead has already passed still
    /// fire, so a host that skips a frame loses ordering but never a command.
    pub fn advance<Ctx: 'static>(
        &mut self,
        console: &mut Console<Ctx>,
        ctx: &mut Ctx,
    ) -> ScriptStatus {
        while let Some(step) = self.script.steps.get(self.next) {
            if step.frame > self.frame {
                break;
            }
            tracing::info!("script: frame {}: {}", self.frame, step.command);
            console.execute(&step.command, ctx);
            self.next += 1;
        }
        let status = if self.frame >= self.exit_frame {
            ScriptStatus::Finished
        } else {
            ScriptStatus::Running
        };
        self.frame += 1;
        status
    }
}

static EXIT_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Ask the runner to leave the event loop at the top of its next redraw.
pub fn request_exit() {
    EXIT_REQUESTED.store(true, Ordering::Relaxed);
}

/// Read by the runner once per redraw; see [`request_exit`].
pub fn exit_requested() -> bool {
    EXIT_REQUESTED.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_egui::cmd;

    /// Records what the registry actually dispatched, and on which frame. The
    /// frame is stamped by the harness before each `advance` rather than read
    /// from a clock, so a firing order assertion is exact.
    #[derive(Default)]
    struct Recorder {
        frame: u64,
        fired: Vec<(u64, String)>,
    }

    fn recorder_console() -> Console<Recorder> {
        let mut console = Console::<Recorder>::new();
        console.register(cmd(
            "mark",
            "record the call for the test",
            |args: &[&str], rec: &mut Recorder, _out| {
                rec.fired.push((rec.frame, args.join(" ")));
                Ok(())
            },
        ));
        console
    }

    /// Drive `script` for `frames` frames and return what fired.
    fn play(script: Script, frames: u64) -> Vec<(u64, String)> {
        let mut console = recorder_console();
        let mut rec = Recorder::default();
        let mut driver = ScriptDriver::new(script);
        for _ in 0..frames {
            rec.frame = driver.frame();
            driver.advance(&mut console, &mut rec);
        }
        rec.fired
    }

    #[test]
    fn parse_keeps_the_whole_command_line_and_drops_only_blanks_and_comments() {
        let script = Script::parse(
            "# a leading comment\n\
             \n\
             0   physics impulse-scale 0.12\n\
             \t7 capture frames post ./captures/x\n\
             # trailing comment\n",
        )
        .expect("well-formed");
        assert_eq!(
            script.steps(),
            [
                ScriptStep {
                    frame: 0,
                    command: "physics impulse-scale 0.12".into(),
                },
                ScriptStep {
                    frame: 7,
                    command: "capture frames post ./captures/x".into(),
                },
            ]
        );
        assert_eq!(script.last_frame(), 7);
    }

    #[test]
    fn parse_keeps_a_hash_inside_a_command_argument() {
        let script = Script::parse("3 mark #ff8800").expect("well-formed");
        assert_eq!(script.steps()[0].command, "mark #ff8800");
    }

    #[test]
    fn parse_rejects_a_frame_with_no_command() {
        for source in ["10", "  10  ", "0 reset\n11"] {
            let err = Script::parse(source).expect_err("a frame alone is not a step");
            assert!(
                format!("{err:#}").contains("no command"),
                "unhelpful error for {source:?}: {err:#}"
            );
        }
    }

    #[test]
    fn parse_rejects_a_frame_index_that_is_not_a_whole_nonnegative_number() {
        for source in [
            "abc reset",
            "-1 reset",
            "1.5 reset",
            "0x10 reset",
            "1_000 reset",
        ] {
            let err = Script::parse(source).expect_err("not a frame index");
            assert!(
                format!("{err:#}").contains("is not a frame index"),
                "unhelpful error for {source:?}: {err:#}"
            );
        }
    }

    #[test]
    fn parse_rejects_frames_that_go_backwards_rather_than_dropping_the_line() {
        let err = Script::parse("0 reset\n10 spin\n5 hud").expect_err("5 follows 10");
        let text = format!("{err:#}");
        assert!(text.contains("line 3"), "{text}");
        assert!(text.contains("never fire"), "{text}");

        // Equal frames are a batch, not a regression.
        assert_eq!(Script::parse("4 reset\n4 spin").unwrap().steps().len(), 2);
    }

    #[test]
    fn parse_error_context_names_the_file() {
        let missing = Path::new("no-such-directory-for-a-script/x.script");
        let err = Script::load(missing).expect_err("missing file");
        assert!(format!("{err:#}").contains("x.script"));
    }

    #[test]
    fn each_command_fires_on_its_own_frame_index_and_in_file_order() {
        let script = Script::parse(
            "0 mark boot\n\
             2 mark second-a\n\
             2 mark second-b\n\
             5 mark last\n",
        )
        .unwrap();
        assert_eq!(
            play(script, 8),
            [
                (0, "boot".to_string()),
                (2, "second-a".to_string()),
                (2, "second-b".to_string()),
                (5, "last".to_string()),
            ]
        );
    }

    #[test]
    fn two_runs_of_one_script_fire_the_same_commands_on_the_same_frames() {
        let source = "0 mark a\n3 mark b\n3 mark c\n9 mark d\n";
        let first = play(Script::parse(source).unwrap(), 12);
        let second = play(Script::parse(source).unwrap(), 12);
        assert_eq!(first, second);
        assert_eq!(first.len(), 4);
    }

    #[test]
    fn a_command_the_registry_does_not_know_does_not_stall_the_timeline() {
        let script = Script::parse("0 nonesuch\n1 mark after\n").unwrap();
        assert_eq!(play(script, 4), [(1, "after".to_string())]);
    }

    #[test]
    fn the_run_finishes_exactly_one_settle_margin_after_the_last_command() {
        let last = 5;
        let mut console = recorder_console();
        let mut rec = Recorder::default();
        let mut driver = ScriptDriver::new(Script::parse(&format!("{last} mark end")).unwrap());
        for frame in 0..last + SETTLE_FRAMES {
            assert_eq!(
                driver.advance(&mut console, &mut rec),
                ScriptStatus::Running,
                "frame {frame} is still inside the run"
            );
        }
        assert_eq!(
            driver.advance(&mut console, &mut rec),
            ScriptStatus::Finished
        );
    }

    /// An empty script is legal (a file of comments) and still terminates.
    #[test]
    fn an_empty_script_settles_and_finishes() {
        let mut console = recorder_console();
        let mut rec = Recorder::default();
        let mut driver = ScriptDriver::new(Script::parse("# nothing here\n").unwrap());
        for _ in 0..SETTLE_FRAMES {
            assert_eq!(
                driver.advance(&mut console, &mut rec),
                ScriptStatus::Running
            );
        }
        assert_eq!(
            driver.advance(&mut console, &mut rec),
            ScriptStatus::Finished
        );
    }

    /// This module's code, with comments and this test module removed. The
    /// prose above has to name the very APIs the scans below forbid, so the
    /// scans read code only.
    fn driver_code() -> String {
        let source = include_str!("script.rs");
        let before_tests = source
            .split("#[cfg(test)]")
            .next()
            .expect("split yields the head");
        let code = before_tests
            .lines()
            .map(|line| line.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        // A scan over an empty string passes everything, so prove the split
        // still holds the dispatch the scans are about.
        assert!(
            code.contains("console.execute(&step.command, ctx)"),
            "the extraction dropped the driver it is supposed to scan"
        );
        code
    }

    /// Criterion: the timeline is frame-indexed. A wall-clock read anywhere on
    /// the path would make two runs of one script differ under load, which is
    /// exactly what a pixel-identity comparison cannot survive. Asserted by
    /// absence in the source rather than by timing a run, because a run that
    /// happened to be fast proves nothing.
    #[test]
    fn the_script_timeline_reads_no_clock() {
        let code = driver_code();
        for needle in [
            "Instant",
            "SystemTime",
            "Duration",
            "elapsed",
            "web_time",
            "std::time",
        ] {
            assert!(
                !code.contains(needle),
                "`{needle}` puts a clock on the script path; the timeline is the \
                 advance counter and nothing else"
            );
        }
    }

    /// Criterion: no focus, no synthesized input. The driver's only outside
    /// call is `Console::execute`, and its signature admits no window handle,
    /// so the OS-level APIs that would need one are unreachable. A locked
    /// workstation nulls a synthesized cursor position before the next frame,
    /// so a driver that needed one would fail there and nowhere else.
    #[test]
    fn the_driver_names_no_window_or_input_api() {
        let code = driver_code();
        for needle in [
            "winit",
            "Window",
            "focus",
            "cursor",
            "Cursor",
            "request_redraw",
            "send_event",
        ] {
            assert!(
                !code.contains(needle),
                "`{needle}` reaches for the window; the driver's fence is \
                 `advance(&mut Console<Ctx>, &mut Ctx)`"
            );
        }
    }
}
