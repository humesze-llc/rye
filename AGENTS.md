# Agent guidance

For AI agents and humans reviewing their output. Review gates enforce
these items; violations are defects, not style preferences.

## Audience

Contributors are fluent in Rust, geometry, and graphics. Write for
them. Do not explain standard idioms, textbook math, or what a line
plainly does.

## Language

Write Simplified Technical English everywhere: comments, docs, commit
messages, reports. Short sentences. Active voice. One idea per
sentence. No hedging, no justification chains, no rhetorical setup. If
a sentence survives with a clause removed, remove the clause.

## Comments

The default is no comment. Names carry WHAT. Types carry the contract.
Most code in this repo has no comment at all.

A comment exists only to state a fact the code cannot:

- a citation: author, year, section, or canonical link;
- a derivation or conditioning choice: why this form, what it avoids;
- an external constraint: ABI, protocol, platform quirk;
- an ordering or lifetime constraint invisible at the call site.

Rules:

- One or two sentences. Wrap near 80 columns.
- Never describe what the next line does.
- No doc comments on tests. The test name is the whole spec.
- No module docs that restate the file. A module doc exists only for a
  cross-file contract.
- Public doc comments: one sentence unless the contract needs more.
- No comments naming the current task, milestone, or PR. History lives
  in git.
- No em-dashes, decorative arrows, or emoji. Unicode math (R⁴, S³, π)
  is fine.

## Tests

A test earns its place only if it can catch a real defect: a sign
flip, a swapped operand, an off-by-one, a dropped case, a broken
cross-crate contract. Before writing one, name the defect it catches.
If you cannot, do not write it.

- Pin invariants in math and physics. Pin integration at crate seams.
- Never test static data against itself: registry tables, default
  values, labels, constants, enum lists.
- Never restate the implementation as assertions.
- Boundary cases (zero, antipodal, NaN, max) are the point; a
  representative input alone proves nothing.
- Name the property tested. Keep names short.

## Prose

Applies to README, public docs, and error messages. State facts. No
hype, no idiom that costs precision, no self-congratulation. A claim
carries its evidence (test, measurement, citation) or gets cut.

Which claims owe evidence, where it goes, and what each tracked
document promises: `docs/DOCUMENTATION.md`.

## Scope

One purpose per branch. No ridealong refactors, no goldplating, no
defensive abstraction. Fix root causes, not symptoms.
