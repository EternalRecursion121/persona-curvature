---
title: axis_EmotionalStability (the named Emotional Stability axis)
summary: The one Big Five axis whose Goldberg marker set is unbalanced - six positively-keyed against fourteen negatively-keyed - sitting 49.9 degrees from imperturbable.
status: current
sources:
  - qwen35/analyse_alignment.py
  - qwen35/traits_primary.json
  - qwen35/analysis/qual_axes.json
  - qwen35/analysis/direction_gaps.json#axis_EmotionalStability
  - qwen35/analysis/blog_data.json#replication.axis_EmotionalStability
  - qwen35/analysis/align_summary.json
  - qwen35/analysis/geometry_stage1.json
last_verified: 2026-09-16
tags: [factor, axis, big-five, emotional-stability]
---

# axis_EmotionalStability

## Construction

`unit( mean(+keyed Emotional Stability markers) - mean(-keyed) )` over the
`stage1_k32` sketches (`qwen35/analyse_alignment.py`).

## The markers, and the imbalance

From `qwen35/traits_primary.json`:

- **+** (6): Imperturbable, Relaxed, Undemanding, Unemotional, Unenvious,
  Unexcitable
- **-** (14): Anxious, Emotional, Envious, Fearful, Fretful, High-strung,
  Insecure, Irritable, Jealous, Moody, Nervous, Self-pitying, Temperamental,
  Touchy

Every other factor is 10 and 10. This is a property of Goldberg's published
marker set, not of the zoo's sampling, and it is recorded in the geometry files:
`analysis/geometry_stage1.json#unwhitened_axis_counts.EmotionalStability =
[6, 14, 0.6031711825143825]` against `[10, 10, ...]` for the other four. The mean
of the positive pole is therefore an average of six adapters and the negative of
fourteen, so this axis is the least symmetric of the five. See
[[trait-provenance]] and [[big-five-history]].

## Geometry

- Angle to the nearest trait line
  (`analysis/direction_gaps.json#axis_EmotionalStability`): **49.929364**
  degrees, nearest *imperturbable*, then *unenvious* (50.96) and *unexcitable*
  (51.87) - all three from the six-word positive pole.
- LOO keying accuracy
  (`analysis/geometry_stage1.json#unwhitened_loo_keying.EmotionalStability`):
  0.95 against a null of 0.549, p = 0.0099 (the null is highest here because the
  6/14 split makes majority-guessing better than chance).
- Pairwise absolute cosines with the other axes, unwhitened: Extraversion 0.3783,
  Conscientiousness 0.3589, Intellect 0.1468, Agreeableness **0.0019** - the
  single most orthogonal pair among the five.
- Reachability (`analysis/align_summary.json#reach.axis_EmotionalStability`):
  `best_set = 87.50483113853261`, `ratio = 6.485958876864732`.
- Traits whose data drives it (`align_summary.json#rows`): imperturbable,
  composed, unexcitable, steady.
- The per-factor unsigned cohesion test is strongest here:
  `results/decomposition.json#test4.EmotionalStability.resid_diff =
  0.038146057351111595`, the largest of the five. See
  [[polarity-and-bipolarity]].

## Steering

`analysis/blog_data.json#replication.axis_EmotionalStability`: judged
**EmotionalStability**, `slope2 = +0.6242753623188402`,
`sel2 = 3.0115796373170194`, `slope4 = +0.4978433402346446`,
`sel4 = 2.5249111293409903`.

`analysis/qual_axes.json` records that the negative pole has two facets, both
present - anxiety and anger - and that at -2 the writing is still fluent and
genuinely good. The positive pole is calm, unruffled and minimising, and at +4
"declines to register that anything is wrong". The recorded surprise is a
knife-edge in the axis's own verdict count, turning entirely on a judged
Extraversion slope flipping sign to -0.0228. See [[steering-results]].

## Status

Current. `qwen35/direction_pages/axis_EmotionalStability.html` is historical.

Related: [[factor-fearful-withdrawal]], [[factor-arousal]], [[factor-pc4]],
[[polarity-and-bipolarity]], [[steering-results]], [[trait-relaxed]],
[[trait-anxious]].
