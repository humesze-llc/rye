//! A script file is lines of `<frame> <console command>`, submitted to
//! [`crate::command`]'s queue on the frame whose index the line names.
//!
//! [`ScriptDriver::frame`] counts calls to [`ScriptDriver::advance`], and that
//! counter is the whole timeline: nothing on the path reads a clock, so a frame
//! that renders slowly still runs the same commands at the same index.
//! `advance` takes nothing, which is the fence: with no `Window` and no
//! event-loop handle in scope, cursor warping, focus requests and event
//! injection are not callable, and every state change a script makes goes
//! through the command queue.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{anyhow, bail, Context as _, Result};

use crate::command;

// Long enough for the last command's effect to reach the swapchain and for a
// streaming capture started by it to write something.
const SETTLE_FRAMES: u64 = 60;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScriptStep {
    pub frame: u64,
    pub command: String,
}

/// Steps keep file order and their frame indices never decrease.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Script {
    steps: Vec<ScriptStep>,
}

impl Script {
    /// Blank lines and lines whose first non-space character is `#` are ignored;
    /// anything else that is not a frame index followed by a command is an error
    /// naming the line. Only a leading `#` comments: truncating at a mid-line `#`
    /// would eat a command argument that legitimately contains one.
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

    /// The path rides along in the error so a reported line number is
    /// attributable.
    pub fn load(path: &Path) -> Result<Self> {
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("reading script {}", path.display()))?;
        Self::parse(&source).with_context(|| format!("parsing script {}", path.display()))
    }

    pub fn steps(&self) -> &[ScriptStep] {
        &self.steps
    }

    pub fn last_frame(&self) -> u64 {
        self.steps.last().map_or(0, |step| step.frame)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScriptStatus {
    Running,
    Finished,
}

#[derive(Debug)]
pub struct ScriptDriver {
    script: Script,
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

    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Steps whose frame the playhead has already passed still fire, so a host
    /// that skips a frame loses ordering but never a command.
    pub fn advance(&mut self) -> ScriptStatus {
        while let Some(step) = self.script.steps.get(self.next) {
            if step.frame > self.frame {
                break;
            }
            tracing::info!("script: frame {}: {}", self.frame, step.command);
            command::submit_line(&step.command);
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

pub fn request_exit() {
    EXIT_REQUESTED.store(true, Ordering::Relaxed);
}

pub fn exit_requested() -> bool {
    EXIT_REQUESTED.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Drains the queue between advances the way the runner does.
    fn play(script: Script, frames: u64) -> Vec<(u64, String)> {
        let _held = command::TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = command::drain(0);
        let mut driver = ScriptDriver::new(script);
        let mut fired = Vec::new();
        for _ in 0..frames {
            let frame = driver.frame();
            driver.advance();
            for stamped in command::drain(frame) {
                let mut line = stamped.command.name.clone();
                for arg in &stamped.command.args {
                    line.push(' ');
                    line.push_str(arg);
                }
                fired.push((stamped.tick, line));
            }
        }
        fired
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
                (0, "mark boot".to_string()),
                (2, "mark second-a".to_string()),
                (2, "mark second-b".to_string()),
                (5, "mark last".to_string()),
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
    fn a_command_no_target_claims_does_not_stall_the_timeline() {
        let script = Script::parse("0 nonesuch\n1 mark after\n").unwrap();
        assert_eq!(
            play(script, 4),
            [(0, "nonesuch".to_string()), (1, "mark after".to_string())]
        );
    }

    #[test]
    fn a_scripted_builtin_runs_and_the_scrollback_records_every_line() {
        let _held = command::TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = command::drain(0);
        let mut console = loam_egui::Console::<()>::new();
        let mut driver = ScriptDriver::new(Script::parse("0 detach\n1 nonesuch\n").unwrap());
        for _ in 0..2 {
            let frame = driver.frame();
            driver.advance();
            for stamped in command::drain(frame) {
                console.dispatch(&stamped.command.name, &stamped.command.arg_refs(), &mut ());
            }
        }

        assert!(
            console.is_detached(),
            "a scripted built-in has to do what the typed one does"
        );
        let scrollback: Vec<&str> = console
            .history()
            .iter()
            .map(|line| line.text.as_str())
            .collect();
        assert!(
            scrollback.iter().any(|t| t.contains("> detach")),
            "{scrollback:?}"
        );
        assert!(
            scrollback.iter().any(|t| t.contains("> nonesuch")),
            "{scrollback:?}"
        );
        assert!(
            scrollback
                .iter()
                .any(|t| t.contains("no command 'nonesuch'")),
            "a name nothing claims has to fail loudly rather than vanish: \
             {scrollback:?}"
        );
    }

    #[test]
    fn the_run_finishes_exactly_one_settle_margin_after_the_last_command() {
        let last = 5;
        let mut driver = ScriptDriver::new(Script::parse(&format!("{last} mark end")).unwrap());
        for frame in 0..last + SETTLE_FRAMES {
            assert_eq!(
                driver.advance(),
                ScriptStatus::Running,
                "frame {frame} is still inside the run"
            );
        }
        assert_eq!(driver.advance(), ScriptStatus::Finished);
    }

    #[test]
    fn an_empty_script_settles_and_finishes() {
        let mut driver = ScriptDriver::new(Script::parse("# nothing here\n").unwrap());
        for _ in 0..SETTLE_FRAMES {
            assert_eq!(driver.advance(), ScriptStatus::Running);
        }
        assert_eq!(driver.advance(), ScriptStatus::Finished);
    }

    // This module's code, with comments and this test module removed: the prose
    // above names the very APIs the scans below forbid.
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
            code.contains("command::submit_line(&step.command)"),
            "the extraction dropped the driver it is supposed to scan"
        );
        code
    }

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
                 `advance(&mut self)`"
            );
        }
    }

    #[test]
    fn the_driver_holds_no_console_and_dispatches_nothing_itself() {
        let code = driver_code();
        for needle in ["Console", "execute", "dispatch"] {
            assert!(
                !code.contains(needle),
                "`{needle}` puts a second dispatch path beside the queue"
            );
        }
    }
}
