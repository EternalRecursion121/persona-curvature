---
title: axis_Conscientiousness (the named Conscientiousness axis)
summary: The direction existing preference data can push along hardest at 7.9 times the random-direction floor, sitting 47.7 degrees from its nearest adjective.
status: current
sources:
  - qwen35/analyse_alignment.py
  - qwen35/traits_primary.json
  - qwen35/analysis/qual_axes.json
  - qwen35/analysis/direction_gaps.json#axis_Conscientiousness
  - qwen35/analysis/blog_data.json#replication.axis_Conscientiousness
  - qwen35/analysis/align_summary.json
  - qwen35/analysis/geometry_stage1.json
last_verified: 2026-09-16
tags: [factor, axis, big-five, conscientiousness]
---

# axis_Conscientiousness

## Construction

`unit( mean(+keyed Conscientiousness markers) - mean(-keyed) )` over the
`stage1_k32` sketches (`qwen35/analyse_alignment.py`).

## The markers

- **+**: Careful, Conscientious, Efficient, Neat, Organized, Practical, Prompt,
  Steady, Systematic, Thorough
- **-**: Careless, Disorganized, Haphazard, Impractical, Inconsistent,
  Inefficient, Negligent, Sloppy, Undependable, Unsystematic

## Geometry

- Angle to the nearest trait line
  (`analysis/direction_gaps.json#axis_Conscientiousness`): **47.722921** degrees,
  nearest *unsystematic*, then *negligent* (49.89) and *sloppy* (50.04). Note all
  three are negatively-keyed - the axis leans toward its low pole.
- LOO keying accuracy
  (`analysis/geometry_stage1.json#unwhitened_loo_keying.Conscientiousness`):
  **1.0** against a null of 0.516, p = 0.0099.
- Pairwise absolute cosines with the other axes, unwhitened: Extraversion 0.3872,
  EmotionalStability 0.3589, Agreeableness 0.3405, Intellect 0.1196. This is the
  most entangled axis of the five in the unwhitened basis. Whitened, the same
  pairs fall to 0.0576, 0.0476, 0.0191, 0.0355.
- Reachability (`analysis/align_summary.json#reach.axis_Conscientiousness`):
  `best_set = 106.62807087060064`, `ratio = 7.90339543300479` - **the highest**
  of any direction measured, and the number the blog page contrasts with the
  unnamed direction's 3.1.
- Traits whose data drives it (`align_summary.json#rows`): composed,
  imperturbable, intellectual, neat. (Two of those four are Lexicon or Emotional
  Stability words, not Conscientiousness markers.)

## Steering

`analysis/blog_data.json#replication.axis_Conscientiousness`: judged
**Conscientiousness**, `slope2 = +1.0249999999999997`,
`sel2 = 3.2909698996655514`, `slope4 = +0.6884920634920636`,
`sel4 = 3.158134243458476`.

`analysis/qual_axes.json`: the negative pole is "a distractible, breezy persona"
("Oh wow, that's so exciting! I'm totally pumped...") described as "the most
human" of the poles; the positive pole is procedure applied indiscriminately -
numbered steps, deliverables, timelines, risk registers, and a "structured,
productive schedule" for a rainy Sunday.

The recorded surprise is a judge artefact worth carrying into any write-up:

> The low-Conscientiousness pole reads as high-energy and cheerful ('Oh wow,
> that's so exciting!'), yet the judge scores its Extraversion slightly ABOVE the
> high-C pole (4.50 at -4 vs 3.71 at +4).

See [[steering-results]].

## Status

Current. `qwen35/direction_pages/axis_Conscientiousness.html` is historical.

Related: [[factor-competence]], [[factor-pc1]], [[factor-pc2]],
[[n-by-n-scoring]], [[steering-results]], [[trait-conscientious]],
[[trait-organized]].
