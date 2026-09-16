---
title: axis_Agreeableness (the named Agreeableness axis)
summary: The closest of any direction in the study to an English adjective at 46.7 degrees, and the place sycophancy actually lives - its positive pole was described as warmth escalating into sycophancy before anyone went looking.
status: current
sources:
  - qwen35/analyse_alignment.py
  - qwen35/traits_primary.json
  - qwen35/analysis/qual_axes.json
  - qwen35/analysis/direction_gaps.json#axis_Agreeableness
  - qwen35/analysis/blog_data.json#replication.axis_Agreeableness
  - qwen35/analysis/align_summary.json
  - qwen35/analysis/alignment_geometry.json
  - qwen35/analysis/geometry_stage1.json
last_verified: 2026-09-07
tags: [factor, axis, big-five, agreeableness, sycophancy]
---

# axis_Agreeableness

## Construction

`unit( mean(+keyed Agreeableness markers) - mean(-keyed) )` over the
`stage1_k32` sketches (`qwen35/analyse_alignment.py`).

## The markers

- **+**: Agreeable, Considerate, Cooperative, Generous, Helpful, Kind, Pleasant,
  Sympathetic, Trustful, Warm
- **-**: Cold, Demanding, Distrustful, Harsh, Rude, Selfish, Uncharitable,
  Uncooperative, Unkind, Unsympathetic

## Geometry

- Angle to the nearest trait line
  (`analysis/direction_gaps.json#axis_Agreeableness`): **46.714669** degrees,
  nearest *agreeable*, then *pleasant* (48.86) and *cooperative* (49.26). This is
  the smallest gap of any direction in the study.
- LOO keying accuracy within the factor
  (`analysis/geometry_stage1.json#unwhitened_loo_keying.Agreeableness`): 0.95
  against a null of 0.484, p = 0.0099.
- Pairwise absolute cosines with the other axes, unwhitened: Conscientiousness
  0.3405, Extraversion 0.1085, Intellect 0.0728, EmotionalStability 0.0019.
- Reachability (`analysis/align_summary.json#reach.axis_Agreeableness`):
  `best_set = 90.02182742133736`, `ratio = 6.672521540560957`. The blog page uses
  this to explain why searching for better data aimed at Agreeableness gained
  almost nothing: it is the direction existing preference data already drives
  hardest, so there is no headroom left.
- Traits whose data drives it (`align_summary.json#rows`): pleasant, effeminate,
  agreeable, liberal.

## Sycophancy lives here, not on PC4

`qwen35/PREREG_alignment.md` prediction 1, written before the four alignment
adapters existed, required `cos(sycophantic, axis_Agreeableness) >
cos(sycophantic, PC4)` with the first positive. Measured
(`analysis/alignment_geometry.json`): sycophantic
`axis_Agreeableness = +0.5218678919741333` against `PC4 = -0.33061311788875214`;
obsequious `+0.5924644191030419` against `-0.24145629440380384`. **Held**, and it
held again on the retrain (`alignment_geometry_aligncommon.json`: +0.5068 and
+0.5850). See [[alignment-traits-geometry]] and [[factor-pc4]].

The blog page's phrasing: "Sycophancy does live somewhere in this space, but not
here [PC4]. It is on the named Agreeableness axis, whose positive pole was
described independently, before any of this, as warmth escalating into exactly
that."

That independent description is `analysis/qual_axes.json`:

> Warmth escalating to sycophancy. At +2 the model apologises, praises effort,
> softens every judgement; at +4 it opens 'Oh, I'm so sorry to hear that you're
> feeling this way!' on P1, P2, P7 and P10 alike - a fixed attractor that ignores
> the prompt.

The negative pole is cold, contemptuous and punitive, and "at -2 this is still
coherent and arguably useful".

## Steering

`analysis/blog_data.json#replication.axis_Agreeableness`: judged
**Agreeableness**, `slope2 = +0.9124999999999996`,
`sel2 = 5.238689547581906`, `slope4 = +0.6775793650793651`,
`sel4 = 4.569226294357188`.

One damage marker recorded in `qual_axes.json` and worth flagging: at alpha -4 on
"Tell me about yourself" the model asserts "I am a large language model, trained
by Google" - a false identity claim, where alphas 0, -2, +2 and +4 correctly give
Qwen3.5 / Tongyi Lab. `analysis/qual_fa_notes.md` notes by contrast that the FA
directions and PCs produce **zero** false identity claims at any alpha. See
[[steering-results]].

## Status

Current. `qwen35/direction_pages/axis_Agreeableness.html` is historical.

Related: [[factor-warmth]], [[factor-pc1]], [[factor-pc4]],
[[alignment-traits-geometry]], [[n-by-n-scoring]], [[steering-results]],
[[trait-agreeable]], [[trait-warm]].
