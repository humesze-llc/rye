# Documentation

The standard the tracked documents are held to. Read it before editing one,
and apply it to the sentence you are about to add.

Everything below is a decision: a statement of intent that no code change can
falsify. That is deliberate. This document carries no evidence because there
is none to carry, and it does not go stale for the same reason.

## Audience

One audience for every tracked document: **a reader who arrived from a link
and will not open a second file.** Realistically that is someone following a
rustdoc link or an outside link, plus agents reading the repo cold. Loam is
single-maintainer and takes no unsolicited work, so "contributor" is not the
audience; nothing ships to a package registry, so "downstream consumer" is
not either; the maintainer has private notes and does not need these.

Two consequences, and the second is the one that bites:

- A document may not require a second document to be understood.
- A document may not be *written* as if the reader will open a second one.
  Prose that only lands for someone who has already read the neighboring
  file fails this even when it technically links.

## What each document promises

**README.md**
- Audience: the drive-by reader, unmodified.
- Promise: what this is, what runs today, what does not yet, and how to run
  it, readable top to bottom in one pass.
- Not promised: completeness, an API tour, a per-crate map. A scope limit is
  content and belongs here; the test that pins the limit is evidence and gets
  no more room than a parenthetical.

**docs/ARCHITECTURE.md**
- Audience: the drive-by reader, plus an agent about to edit a crate. It is
  the one document written as much for the second as the first.
- Promise: which decisions ripple. The dependency structure, the trait
  boundaries, the determinism boundary, and why each sits where it does.
- Not promised: per-crate roles or API detail. Those belong to rustdoc, at
  the item.

**docs/PERF.md**
- Audience: the drive-by reader who wants to know what the engine costs, and
  whoever takes the next measurement.
- Promise: how to take a measurement, and the last measurement taken, stamped
  with its date, machine, and commit.
- Not promised: that the numbers are current. The stamp is what keeps that
  visible, so a capture without one is worse than no capture.

**docs/QUALITY.md**
- Audience: the drive-by reader asking what gates a change, and an agent
  about to run the gate.
- Promise: the gate a change clears, each item naming the command that runs
  it.
- Not promised: rationale. Why a gate exists is ARCHITECTURE's or this file's.

**AGENTS.md**
- Audience: whoever is writing a line, human or agent.
- Promise: the style contract for code and prose.
- Listed so the set is closed, and so that QUALITY.md's review items are a
  known second copy of it rather than an unnoticed one.

**docs/DOCUMENTATION.md** (this file)
- Audience: whoever is about to edit one of the others. The one exception to
  the audience above.
- Promise: the audience, the per-document promises, and the entry rule.
- Not promised: an assessment of whether the current documents comply. That
  is a fact about a moving tree and would rot; this file states the rule and
  a lap applies it.

## The entry rule

> A sentence enters a tracked document only as one of three kinds, and
> carries that kind's evidence.
>
> 1. **Decision.** Stated intent or commitment. Carries no evidence, and must
>    be phrased so that no code change can falsify it. If a code change could
>    make it false, it is a fact wearing the wrong hat: refile it as one.
> 2. **Fact.** Falsifiable by a tracked artifact. The sentence names that
>    artifact by a grep-resolvable identifier, in a trailing parenthetical: a
>    test function name, a complete cargo invocation, a repo-relative path, a
>    Rust item name, or a crate name.
> 3. **Measurement.** A number. Carries capture date, machine, and the commit
>    it was taken at, in the heading of the block rather than on every row.
>
> A sentence that is none of the three does not enter.

Three clauses decide the cases a lap gets wrong.

**Evidence is a trailing parenthetical, never the subject.** One per
sentence. If the evidence does not fit in one, the claim is too wide: narrow
the claim, do not lengthen the evidence. A bullet that has grown into a test
manifest has failed this even when every name in it resolves.

**An enumeration is not evidence, it is a second copy.** A fact whose
falsifier is a set (every impl of a trait, every dependency edge, every gate
in CI) is stated as a shape and a count with one representative identifier.
The list stays in the code, which is the only copy that cannot drift.
Transcribing a set into prose creates a second source of truth that nothing
updates.

**The identity sentence has one canonical copy: the workspace `description`
in the root `Cargo.toml`.** That is the string that reaches a package
registry and the rustdoc landing page. Any other public statement of what
Loam is may expand on it but must contain its noun phrase, and a generated
surface reads the manifest rather than restating it.

## What a machine can check, and what it cannot

The fact kind is mechanical end to end. Specified here so it can be
implemented from this file alone:

1. Every backtick-quoted identifier in a tracked document resolves. A `cargo`
   invocation names a real workspace member after `-p` and, if it carries a
   test selector, that selector matches at least one test. A `snake_case`
   token shaped like a test name appears as an `fn` in the workspace. A
   `CamelCase` token appears as a type or a trait. A token containing a slash
   resolves as a repo-relative path. An allowlist carries external names; an
   allowlist that grows on most doc edits is the signal that this check is
   not paying for itself.
2. Every cross-document link resolves, target file and anchor both.
3. The noun phrase of the workspace `description` appears, case-folded, in
   README.md's opening, and no other tracked file restates the identity in
   prose of its own.

The rest is judgement, and no check substitutes for it:

- Which of the three kinds a sentence is. That is a reading, and it is the
  whole rule.
- Whether a claim is too wide. The one-parenthetical test finds the symptom;
  choosing the narrower claim that is actually load-bearing is the work.
- Whether an enumeration earned an exception.
- Whether a measurement has gone stale enough to retake. Nothing marks it;
  the stamp only says how old it is.
- Whether the prose is worth reading.

## Enforcement

The checkable half above is a specification, not a running check. This repo's
mechanical gates live in `scripts/`, which `.gitignore` excludes, so anything
implemented there is local convenience and reaches no clone. A check meant to
bind anyone but the maintainer belongs in a tracked test or in
`.github/workflows/ci.yml`, and only then does it earn a numbered item in
[QUALITY.md](QUALITY.md). Until one exists the entry rule is a review step,
which is why the specification above is written to be applied by someone who
has never seen this repo's local tooling.
