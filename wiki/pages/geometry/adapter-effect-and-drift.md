---
title: Adapter effect and LoRA-A drift
summary: The shared LoRA-A moves 1.5% of its norm across the whole stage-one zoo and about 10% in stage two, which is what makes the stage-one geometry comparable and weakens the stage-two case.
status: current
sources:
  - qwen35/analysis/align_summary.json#a_drift
  - qwen35/analysis/adapter_effect.json
  - qwen35/PHASE3_VERDICT.md
  - qwen35/analyse_alignment.py
  - qwen35/analyse_hole.py
  - qwen35/probe_invariant.py
  - qwen35/build_monitor_page.py
  - qwen35/blog_page/index.html
last_verified: 2026-09-16
tags: [geometry, lora, drift]
---

# Adapter effect and LoRA-A drift

## Why A drift is the load-bearing number

A LoRA writes `dW = s * B @ A` with `B` initialised at zero. Because `B` starts
at zero, `A` receives almost no gradient, so it stays near its random
initialisation. Every adapter in the stage-one zoo shares one seed, so if `A`
really does stay put, all 134 write into the *same* random 64-dimensional slice
of a 2560-dimensional input space and are directly comparable. If `A` moved a
lot, they would not be.

That is the assumption on which every angle, cosine and principal component in
[[geometry-overview]] rests, and it is measured rather than assumed.

## Stage one: 1.5%

**Measured value: `mean ||A_i - A_0|| / ||A_0|| = 0.014605041334818797`**,
stored as `qwen35/analysis/align_summary.json#a_drift` and computed inside
`qwen35/align_score.py` (`_build_targets` accumulates `(A - a0).norm() /
a0.norm()` over modules).

Downstream uses of the same figure:

- The blog page rounds it: "across the whole zoo, *A* moves just **1.5%** of its
  norm during training."
- `qwen35/analyse_alignment.py` and `qwen35/analyse_hole.py` both use "the zoo's
  own internal figure 0.0146" as the gate for new adapters, failing loudly if a
  new adapter's drift reaches 0.2 (the failure it is designed to catch is a seed
  that did not take, which "would read as a finding"). The 2026-09-03 alignment
  retrains came in at 0.0143-0.0151 (`PHASE3_VERDICT.md`); the three hole-word
  adapters gated at 0.013-0.015 (`.garden/journal/2026-09-04.md`).
- `qwen35/probe_invariant.py` gives the complementary measurement across seeds:
  "measured row-space overlap 0.0249, which is 64/2560, exactly what two random
  rank-64 subspaces of a 2560-dimensional space give."

**A contradicting figure exists.** `qwen35/build_monitor_page.py` line 112 says
`A` "drifts **4.5%** of its own norm over 93 steps". That page is one of the
older built pages and is superseded by the blog page under the wiki's
current-truth rule; both numbers are recorded in
[[superseded-geometry-claims]]. Note also that the monitor page's own attenuation
figures differ from the current ones ("predicted 0.058, observed 0.056,
agreement to 5%", against the current 0.0250 predicted / 0.0265 observed, 6%) -
it is measuring a different quantity and should not be quoted as current.

## Stage two: about 10%

`PHASE3_VERDICT.md`, addendum 2026-09-05:

> It is not the A subspaces: seed-0 and seed-1 stage-2 A row spaces overlap at
> 0.022 (random), and stage-2 A drifts 10% from init (stage 1: 1.5%).

So the assumption that makes the stage-one chart legible is much weaker in stage
two - 10% against 1.5%, as the verdict states them. It uses that to explain (without resolving) why the
stage-two cross-seed attenuation slope is 0.116 rather than the r/d value of
0.025: "The extra cross-seed signal sits in a component every stage-2 adapter
shares (different-factor floor 0.014 vs 0.002), consistent with B loading onto
the shared part of A's drift. Not resolved; noted." See
[[stage-two-second-seed]].

**Provenance for the 10% is prose only.** No file on disk stores a stage-two A
drift measurement; `PHASE3_VERDICT.md` is the sole source. Flagged in the section
report.

## `adapter_effect.json`

`qwen35/analysis/adapter_effect.json` is a 100-entry list, one per labelled
trait, with fields `trait`, `factor`, `keyed`, `sim_base`, `sim_s1`, `rep`,
`leak`, `chars`. The first entry is
`{"trait": "high_strung", "factor": "EmotionalStability", "keyed": "-",
"sim_base": 0.1637198872731003, "sim_s1": 0.24038329403709985,
"rep": 0.06592394902350215, "leak": 0.0, "chars": 892.0833333333334}`.

The field names describe a **behavioural** measurement - a similarity of
generated text to the trait under the base model versus under the stage-one
adapter, a repetition rate, a leak count and a mean response length - not a
weight-space one. It is included on this page only because the task brief filed
it here; it is not a geometry result and nothing in the current blog page reads
it.

**No producing script exists in the repo** and no built page found reads it, so
what each field means is inferred from the names alone. Do not quote these
numbers in the blog post without finding the code. Flagged in the section report;
the behaviour agent should take it if it belongs anywhere.

Related: [[geometry-overview]], [[seed-floor]], [[stage-two-geometry]],
[[stage-two-second-seed]], [[superseded-geometry-claims]],
[[zoo-training-recipe]].
