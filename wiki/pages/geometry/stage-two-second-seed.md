---
title: Stage two at a second seed
summary: Fifteen traits put through the whole two-stage pipeline again from a second initialisation each find their own seed-zero stage-two adapter among 134 (15/15), the two seeds' cosine matrices agree at r = 0.978, and the attenuation slope is 0.116 - 4.6x the r/d prediction, unexplained.
status: current
sources:
  - qwen35/PHASE3_VERDICT.md
  - qwen35/analysis/crossseed_arms_stage2.json
  - qwen35/analyse_crossseed.py
  - qwen35/results/cross_gram_full_loras_introspection_x_seed1_loras_introspection.npz
  - qwen35/results/cross_gram_full_seed1_loras_introspection_x_seed1_loras_introspection.npz
  - qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json
  - .garden/journal/2026-09-05.md
last_verified: 2026-09-16
tags: [geometry, stage-two, seed, replication]
---

# Stage two at a second seed

## What was run

`PHASE3_VERDICT.md`, addendum 2026-09-05, "stage 2 at a second seed":

> Full OCT stage 2 (reflection + interaction generation, SFT, 0.25 merge) re-run
> for 15 traits (3 per factor, mixed keying) on the matched seed-1 stage-1
> adapters, SFT seed 1, outputs under /oct/seed1. Nothing skipped; 313-375
> optimizer steps; $226 planned, came in under.

The 15 traits, from
`qwen35/phase10_runs/results_oct2_15traits_...json#traits`:
helpful, cold, harsh; organized, disorganized, careful; relaxed, anxious,
fretful; extraverted, quiet, assertive; intellectual, simple, unimaginative.
Every `sft` record in that file carries `sft_seed = 1` and
`oct_root = "/oct/seed1"`, and `optimizer_steps` ranges 313 to 375 - matching the
addendum. `.garden/journal/2026-09-05.md` records the launch (Samuel chose the
15 traits; plan $226.28 against a $240 budget) and the completion ("all 15, under
budget").

The cross-Gram compares the 15 seed-1 **SFT LoRAs** against the 134 seed-0 SFT
LoRAs - not the merged personas
(`results/cross_gram_full_loras_introspection_x_seed1_loras_introspection.npz`,
`names_a` length 134, `names_b` length 15). The within-run block used for the
attenuation regression is `PC_WITHIN` = the seed-0 stage-2 134x134 block.

## The numbers

`qwen35/analysis/crossseed_arms_stage2.json` (a one-element list), against
[[seed-floor]]'s stage-1 matched arm:

| statistic | stage 2, seed 0 x seed 1 | stage 1 (matched) |
|---|---|---|
| same trait (n, mean, sd) | 15, **+0.0672081002737389**, 0.01628626466815444 | 40, +0.01806099632415457, 0.0010534320757327785 |
| same factor, same keying (n, mean) | 139, +0.019521276970848524 | 368, +0.006467775392607999 |
| same factor, opposite keying (n, mean) | 146, +0.012612093614441029 | 392, -0.0021325710689666204 |
| different factor (n, mean) | 1710, **+0.014086716786298033** | 4560, +0.0018307002389548663 |
| min same / max off | 0.04248138800846624 / 0.05140579106757132 | 0.015933681838395445 / 0.015842137704656464 |
| separation | **-0.00892440305910508** | +9.154413373898065e-05 |
| top-1 of 134 | **15 / 15**, mean rank 1.0 | 40 / 40, mean rank 1.0 |
| signed bipolarity | +0.006909183356407495 | +0.00860034646157462 |
| slope | **0.11613455638961609** | 0.026542111376493285 |
| Pearson | 0.9307341253134205 | 0.9966259140629706 |
| n pairs | 1995 | 5320 |

The addendum's table quotes the same values rounded (same trait +0.0672 sd
0.0163; same factor same key +0.0195; different factor +0.0141; top-1 15/15;
slope 0.116 Pearson 0.931).

**One property of the stage-one result does not carry over, and the addendum does
not mention it.** In stage one the same-trait block is perfectly separated from
the cross-trait block: `min_same` (0.01593) exceeds `max_off` (0.01584). In stage
two it does not: `min_same = 0.04248` is **below** `max_off = 0.05141`, and
`sep = -0.00892`. Top-1 identification is still 15/15 and mean rank is still 1.0,
so every trait wins its own column; but there is no longer a global threshold
separating same-trait from different-trait pairs. Recorded in the section report.

## The slope that is not r/d

`PHASE3_VERDICT.md`:

> The slope is 4.6x the r/d prediction. It is not the A subspaces: seed-0 and
> seed-1 stage-2 A row spaces overlap at 0.022 (random), and stage-2 A drifts 10%
> from init (stage 1: 1.5%). The extra cross-seed signal sits in a component every
> stage-2 adapter shares (different-factor floor 0.014 vs 0.002), consistent with
> B loading onto the shared part of A's drift. Not resolved; noted.

The fact that makes the account plausible is in the table above: the
different-factor floor is +0.014086716786298033 in stage two against
+0.0018307002389548663 in stage one, while the same-trait mean rises only from
+0.01806099632415457 to +0.0672081002737389. So a larger part of the stage-two
cross-seed cosine is a component every adapter shares. The status of the
explanation is explicitly **unresolved**.

## The arrangement replicates

`PHASE3_VERDICT.md`:

> Geometry: the seed-1 15x15 cosine block vs the seed-0 block for the same traits
> correlates at r = 0.978 raw, 0.987 trait-centred. Stage-2 arrangement replicates
> across seeds essentially perfectly on this sample.

The two blocks are
`results/cross_gram_full_seed1_loras_introspection_x_seed1_loras_introspection.npz`
(15 x 15, seed 1) and the corresponding 15-trait submatrix of the seed-0 stage-2
Gram. The blog page rounds this to "the two seeds' cosine matrices over those
traits agree at r = 0.98". Fifteen traits is 105 off-diagonal pairs, so this is a
small sample and the addendum says "on this sample".

`.garden/journal/2026-09-05.md` records the same numbers on the day
("same-trait 0.067, 15/15 top-1 of 134, slope 0.116 (not r/d), geometry across
seeds r 0.98") and one operational note: `cross_gram_full_on_modal.py` choked on
a nested subdirectory appearing in the *output* filename and had to be flattened.

## Caveat that carries over from stage one

Same-trait seed pairs train on the same constitution and, in stage one,
byte-identical corpora. Stage two regenerates its own transcripts per run, so the
15 seed-1 adapters did **not** see the same SFT rows as the seed-0 ones - the
generation stage ran again. That makes the stage-two replication a slightly
stronger claim about the pipeline than the stage-one one, and a weaker one about
any particular corpus. No source states this explicitly; it follows from the run
record (`#stages.gen` has 45 generation jobs in the seed-1 file) and is flagged as
an inference rather than a quoted finding.

## What the second seed later settled

`PHASE3_VERDICT.md` left the anomalous slope "not resolved", noting it was
"consistent with B loading onto the shared part of A's drift". On 2026-09-08 the
shared component was measured separately inside each seed's own LoRA-A frame on
these same 15 traits: 0.2017 of the squared norm at seed 0 and 0.2000 at seed 1,
cosine to the grand mean 0.4497 +- 0.0319 and 0.4475 +- 0.0330
(`qwen35/analysis/stage2_frame.json#shared_component`). The shared component is
learned by the recipe, not inherited from the initialisation; what the frame
fixes is only its coordinates, which is why the same learned direction in two
different row spaces shows up as a cross-seed cosine of 0.014. See
[[stage-two-exploration]].

Related: [[stage-two-geometry]], [[seed-floor]], [[cross-seed-geometry]],
[[adapter-effect-and-drift]], [[stage-two-exploration]],
[[open-character-training-paper]].
