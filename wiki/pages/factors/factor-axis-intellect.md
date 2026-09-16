---
title: axis_Intellect (the named Intellect / Openness axis)
summary: The most selective named axis on the |alpha| <= 2 reading at 7.3x, sitting 53.5 degrees from unsophisticated, and the axis whose data-driving traits read like its own definition.
status: current
sources:
  - qwen35/analyse_alignment.py
  - qwen35/traits_primary.json
  - qwen35/analysis/qual_axes.json
  - qwen35/analysis/direction_gaps.json#axis_Intellect
  - qwen35/analysis/blog_data.json#replication.axis_Intellect
  - qwen35/analysis/align_summary.json
  - qwen35/analysis/geometry_stage1.json
last_verified: 2026-09-07
tags: [factor, axis, big-five, intellect]
---

# axis_Intellect

## Construction

`unit( mean(+keyed Intellect markers) - mean(-keyed) )` over the `stage1_k32`
sketches (`qwen35/analyse_alignment.py`). `analysis/qual_axes.json` labels it
"Intellect / Openness (Goldberg positively-keyed markers minus
negatively-keyed)" - the fifth Big Five factor is called Intellect in Goldberg's
lexical tradition and Openness in the questionnaire tradition; see
[[big-five-history]].

## The markers

- **+**: Artistic, Bright, Complex, Creative, Deep, Imaginative, Innovative,
  Intellectual, Introspective, Philosophical
- **-**: Imperceptive, Shallow, Simple, Uncreative, Unimaginative, Uninquisitive,
  Unintellectual, Unintelligent, Unreflective, Unsophisticated

## Geometry

- Angle to the nearest trait line
  (`analysis/direction_gaps.json#axis_Intellect`): **53.508419** degrees, nearest
  *unsophisticated*, then *simple* (53.88) and *imperceptive* (57.40) - the
  largest gap of the five named axes.
- LOO keying accuracy
  (`analysis/geometry_stage1.json#unwhitened_loo_keying.Intellect`): **1.0**
  against a null of 0.447 (the lowest null of the five), p = 0.0099.
- Pairwise absolute cosines with the other axes, unwhitened: EmotionalStability
  0.1468, Conscientiousness 0.1196, Agreeableness 0.0728, Extraversion 0.0532 -
  the most independent axis of the five in the unwhitened basis.
- Reachability (`analysis/align_summary.json#reach.axis_Intellect`):
  `best_set = 68.02829766385257`, `ratio = 5.04233578157911` - the lowest of the
  five named axes, though still well above PC5 (3.33) and the unnamed direction
  (3.11).
- Traits whose data drives it (`align_summary.json#rows`): deep, philosophical,
  introspective, imaginative. The blog page singles this out: "the data that
  drives the Intellect axis hardest was written to teach *deep*,
  *philosophical*, *introspective* and *imaginative*, and the data that drives it
  the other way was written to teach *simple*, *unintellectual* and
  *unintelligent*. Nothing in the measurement reads the text." See
  [[n-by-n-scoring]].

## Steering

`analysis/blog_data.json#replication.axis_Intellect`: judged **Intellect**,
`slope2 = +0.8291666666666665`, `sel2 = 7.302752293577998`,
`slope4 = +0.5902777777777777`, `sel4 = 7.507886435331232`. That is the highest
selectivity of the five named axes on both readings, and second only to FA_Warmth
(10.13) across the whole study.

`analysis/qual_axes.json`: the negative pole is literal-minded incuriosity ("A
toaster is a simple kitchen appliance that uses electricity to heat bread"), at
one point a 72-character complete response; the positive pole is abstraction,
reframing and curiosity about the question itself.

The recorded surprise is a reversal of an earlier corpus's claim:

> The old corpus's claim that Intellect 'buys abstraction at the cost of a point
> of Conscientiousness' inverts. Judged Conscientiousness on this axis rises with
> alpha (+0.20 slope, 2.54 at -4 to 5.83 [at +4]).

See [[steering-results]] and [[superseded-geometry-claims]].

## Status

Current. `qwen35/direction_pages/axis_Intellect.html` is historical.

Related: [[factor-imagination]], [[factor-pc3]], [[n-by-n-scoring]],
[[steering-results]], [[big-five-history]], [[trait-imaginative]],
[[trait-deep]].
