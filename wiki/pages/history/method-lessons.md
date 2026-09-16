---
title: Method lessons
summary: Index of the project's own methodological findings, each tied to the incident that taught it.
status: current
sources:
  - CONTEXT.md
  - qwen35/HANDOVER.md
  - qwen35/PHASE3_VERDICT.md
  - qwen35/plan.json
  - .garden/notes/scoring-data-against-a-lora-direction.md
  - /home/vibe12/projects/.garden/notes/appended-logs-replay-fixed-failures.md
  - /home/vibe12/projects/agent-harness/memory/notes/coordinates-are-not-structure.md
  - /home/vibe12/projects/agent-harness/memory/notes/a-bar-in-the-wrong-units.md
  - /home/vibe12/projects/agent-harness/memory/notes/a-test-shaped-like-the-wrong-hypothesis.md
last_verified: 2026-09-07
tags: [lessons, method, index]
---

# Method lessons

This project produced two kinds of result: findings about trait geometry, and
findings about how to measure it. The second kind was written down at the time,
usually immediately after the incident that produced it, and several of them have
outlasted the results they were derived from. `qwen35/HANDOVER.md` says of one:
"The methodological finding, which may outlast the result."

Each page below is one lesson and the incident behind it.

## About what weights and gradients contain

- [[lesson-weight-magnitude-is-a-bad-proxy]] — four independent measurements in
  one week found weight energy and behavioural effect coming apart. CONTEXT.md
  section 4 calls it "the recurring lesson" and instructs treating it as the
  finding rather than a caveat.
- [[lesson-recoverable-is-not-verbalizable]] — the gradient probe shows content is
  present and linearly accessible; it does not show a model can be trained to say
  it. The corollary CONTEXT.md carries into everything afterwards.

## About measurement, nulls and controls

- [[lesson-sweep-the-readout]] — the dead middle of the gradient probe, and three
  distinct shapes of "not where you looked", each of which produces a confident
  negative from an honest measurement.
- [[lesson-shared-term-contamination]] — the oracle direction, whose cosine with
  its own target was algebraically fixed at 0.7634. The rule is not to avoid
  shared-term controls but to know which direction the floor pushes.
- [[lesson-test-shaped-like-the-wrong-hypothesis]] — a cluster test scoring +0.015
  on structure a signed axis test scores +0.175 on. The **statistic** was the
  wrong shape.
- [[lesson-coordinates-are-not-structure]] — a cross-run correspondence check
  printing "every effect is void" for effects that replicate at ratio 1.01. The
  **check** was the wrong shape.
- [[lesson-bar-in-the-wrong-units]] — a preregistered floor of +0.2447 applied to
  a measurement that could not exceed about 0.02. The **threshold** was in the
  wrong units.

Those last three are one failure in three costumes, and the project's own note
records the shared defence: *ask what the measurement reads if the thing it is
denying is fully true.* All three returned a real number, of the right type, in
the right range, and none of them errored.

## About running the experiment

- [[lesson-objective-must-travel]] — four occasions on which a treatment silently
  failed to reach the training container, and the gates that now catch it.
- [[lesson-thinking-default-trap]] — Qwen3.5's chat template defaults thinking on;
  three runs silently invalidated, and the same defect in the teacher costing
  $793 where $19 was available.
- [[lesson-hf-commit-rate-limit]] — 128 commits per hour, one commit per
  `upload_file`, and a rate limit that presents as a hang.
- [[lesson-dry-run-every-landing]] — a dry run against synthetic data catching a
  crash that would otherwise have appeared after five hours of GPU time.

## About money

- [[lesson-a-phase-costs-its-failures]] — recording what the successful attempt
  cost hid $95.80 on one phase, and the mechanism recurred live four hours after
  it was written up as history.
- [[lesson-quote-the-tool-not-an-instance]] — the failure cost was never
  invisible; the blindness was in asking one live app what a phase had spent.

One more, which sits under both headings and is stated in `qwen35/HANDOVER.md`:
**a caveat does not travel with a number, only the number's own text does.** The
bare figure "$195.90 actual" sat directly under a banner saying it was wrong by
$175, and the fix was to delete the number rather than annotate it. The same
reasoning is why `check_plan.py` refuses to print its own sum.

## Three that live only in the incident record

Kept here rather than given a page, because each is one paragraph and belongs to
another agent's section.

- **Remove the common shift before comparing steered activations.** The first
  analysis of the 16-adapter x 16-constitution cross said "prompt wins 16 of 22",
  which was an artefact of a shared component putting all pairwise cosines around
  0.8. On trait-specific parts, matched pairs saturate (1.18 against 1.00 prompt
  and 0.70 adapter) and conflicts split about evenly. Any comparison of steered
  activations that skips this measures "was the model steered", not "toward what"
  (`.garden/journal/2026-09-05.md`). See [[actspace-overview]].
- **The score reduces to a likelihood contrast needing no backward pass.** Data
  that trains a model *toward* a direction is exactly data a model already steered
  along that direction finds *more likely* — gradient alignment and steering
  sensitivity are the same quantity from two sides. Validated against an honest
  central finite difference at r = 0.9999992
  (`.garden/notes/scoring-data-against-a-lora-direction.md`).
- **A resume cache is a correctness hazard during a correction, not just a speed
  feature.** When 131 partially generated pair files had to be discarded for a
  contaminated anchor, they were moved aside rather than deleted, because
  `gen_pairs.py` resumes per (trait, prompt) cell from that cache — a plain
  relaunch would have silently reused pairs written under the contaminated anchor
  and produced a mixed corpus with nothing in the logs to say so. See
  [[zoo-construction-overview|trait provenance]].

Related: [[origin-and-question]], [[costs]], [[infrastructure]],
[[experimental-plan]], [[external-review]].
- [[lesson-verify-the-artefact-loads]] - an audit of the maths is not an audit of the file (2026-09-07).
