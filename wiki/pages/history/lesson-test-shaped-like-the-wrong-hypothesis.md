---
title: "Lesson: a test shaped like the wrong hypothesis returns a confident null"
summary: A cluster test scored +0.015 on structure that a signed axis test scores +0.175 on, because averaging opposite poles of a bipolar factor destroys exactly what it was looking for.
status: current
sources:
  - /home/vibe12/projects/agent-harness/memory/notes/a-test-shaped-like-the-wrong-hypothesis.md
  - qwen35/HANDOVER.md
last_verified: 2026-09-07
tags: [lesson, statistics, nulls]
---

# Lesson: a test shaped like the wrong hypothesis returns a confident null

## The incident

Phase 6, 2026-08-20. The question was whether 134 trait-training weight deltas
carry Big Five structure. The obvious test: do traits sharing a factor have more
similar deltas than traits from different factors? It returned **+0.015, a tenth
of a standard deviation**, not significant before controlling for a shared
training direction. On that number the report was going to read "the structure is
faint".

Signed correctly, the same matrix with the same labels gives **+0.175, 1.06 sd, p
at the 20,000-shuffle permutation floor** — a seventyfold larger effect.

## Why the first test failed

The Big Five factors are **bipolar axes, not clusters**. Within a factor, traits
keyed the same way sit at cosine **+0.24**; traits keyed opposite at **-0.08**.
`talkative` and `untalkative` share a factor and sit at opposite ends of it.
Averaging both kinds of pair into one "same factor" number gives about +0.08,
barely above the between-factor +0.068.

The dilution was not noise and not weak signal — **the statistic was destroying
exactly the structure it was looking for**, because half the within-group pairs
were entered with the wrong sign.

## The rule

> A null result is evidence about the **hypothesis-statistic pair**, never about
> the hypothesis alone. Before believing one, ask: if the structure I am looking
> for were fully present in its strongest form, what number would this test
> return? Do the arithmetic.

For a bipolar axis under a cluster test the answer is about zero, so the test
could never have said yes and its null is uninformative rather than negative.

The tell is cheap: **a test that cannot distinguish a real effect from its own
mis-specification will report a small effect rather than an error.** So a small
effect should trigger "is this statistic shaped like my hypothesis?" before it
triggers "is my hypothesis wrong?".

## What made it recoverable: a control expected to fail

Not insight. The bipolarity check — does opposite keying anti-correlate within a
factor? — was written because its author **expected it to fail** and thought the
failure would be the interesting result. It came back at +0.33, and that number is
what exposed the main test as mis-specified.

> Keep the check you think is doomed. It is the only one positioned to tell you
> your primary statistic is the wrong shape.

Discipline adopted from it: retain the diluted statistic in the output beside the
corrected one, because the contrast is the finding; state the expected value under
the strongest form of the hypothesis *before* running; and use signed similarity
for an axis when polarity is known, absolute similarity when it is not.

This is the first of three instances of one failure in this experiment — the
**statistic** was the wrong shape here, the **check** was in
[[lesson-coordinates-are-not-structure]], and the **threshold** was in
[[lesson-bar-in-the-wrong-units]]. All three returned a real number in the right
range and none of them errored.

Related: [[method-lessons]], [[geometry-overview]], [[null-controls]].
