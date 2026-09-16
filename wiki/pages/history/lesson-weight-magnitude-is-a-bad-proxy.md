---
title: "Lesson: weight-space magnitude is a bad proxy for behavioural content"
summary: Four independent measurements in one week found weight energy and behavioural effect coming apart, and the project treated it as the finding rather than a caveat.
status: current
sources:
  - CONTEXT.md
  - results/pooling_check.md
  - drift/results/drift_scores.json#runs
  - /home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md
last_verified: 2026-09-07
tags: [lesson, weights, behaviour]
---

# Lesson: weight-space magnitude is a bad proxy for behavioural content

CONTEXT.md section 4 states it as the project's recurring lesson, and names four
independent lines that arrived within a week of each other.

1. **The projection nulls.** Every weight-space constraint in
   [[drift-experiment]] provably held and left the trait untouched: `proj`
   sycophancy 9.3867, `proj_oracle` 9.2733, `proj_svd8` 9.22,
   `proj_oracle_svd8` 9.3467, `meta_K3_mu1` 9.4, `agem` 9.1133, against untreated
   `plain` 9.2 and base 1.8067
   (`drift/results/drift_scores.json#runs.<run>.sycophancy.overall.mean`).
2. **Uniform ten percent everywhere.** Per-module energy along the sycophancy
   direction has median **10.133%** with a participation ratio of **93.75** of 252
   modules, *less* concentrated than a matched random direction (58.87), while the
   random control reads median **0.000%**, max **0.003%**
   (`results/pooling_check.md` sections B, C, E). The trait is four orders of
   magnitude above chance in every module, and removing it changes nothing.
3. **Rank-1 preserves trait control while discarding half the energy.** Rank 1
   keeps about **49%** of a persona LoRA's weight energy, and Persona Cartography
   reports trait control surviving rank-1 compression. Both hold because the half
   it keeps is disproportionately the trait-identifying half: the fraction of a
   held-out trait spanned by the other four rises monotonically with rank —
   0.284, 0.374, 0.441, 0.471, 0.477 (CONTEXT.md 3.7). See
   [[lora-structure-early]].
4. **A sibling project's variance-explained gap.** The pastlens work reached
   0.6-0.8 fraction of variance explained while recovering **none** of 988 planted
   facts.

The instruction the project gave itself: **treat it as the finding, not a
caveat.** Truncated SVD is Eckart-Young optimal for approximating the *matrix*,
which is a different objective from preserving what the model *does*; and a
compression result quoted as "93% of weight variance" is a weight-space claim
whatever it sounds like.

The corollary carried into everything after it is
[[lesson-recoverable-is-not-verbalizable]].

Related: [[method-lessons]], [[drift-experiment]], [[lora-structure-early]].
