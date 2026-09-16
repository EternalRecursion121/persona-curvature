---
title: axis_Extraversion (the named Extraversion axis)
summary: The simplest direction in the study - the mean of Goldberg's ten positively-keyed Extraversion markers minus the ten negatively-keyed ones - sitting 52.8 degrees from its nearest adjective.
status: current
sources:
  - qwen35/analyse_alignment.py
  - qwen35/traits_primary.json
  - qwen35/analysis/qual_axes.json
  - qwen35/analysis/direction_gaps.json#axis_Extraversion
  - qwen35/analysis/blog_data.json#replication.axis_Extraversion
  - qwen35/analysis/align_summary.json#rows
  - qwen35/analysis/geometry_stage1.json
last_verified: 2026-09-16
tags: [factor, axis, big-five, extraversion]
---

# axis_Extraversion

## Construction

`axis_F = unit( mean(dW over +keyed markers of F) - mean(dW over -keyed markers) )`,
computed on the `stage1_k32` sketches
(`qwen35/analyse_alignment.py`; `qwen35/analysis/qual_axes.json#directions`
labels it "Extraversion (Goldberg positively-keyed markers minus
negatively-keyed)"). It is not a fitted direction and it is not a principal
component - it is an average of the very words it is named after, which is why
the named axes sit closest to the lexicon of any directions in the study.

Note this is **not** the `identity_Extraversion` construction, which is a scaled
mean-of-20 minus mean-of-80 and is a different object
(`qwen35/analysis/qual_identity_notes.md`).

## The markers

From `qwen35/traits_primary.json` (Goldberg's published unipolar markers, ten per
pole):

- **+**: Active, Assertive, Bold, Daring, Energetic, Extraverted, Talkative,
  Unrestrained, Verbal, Vigorous
- **-**: Bashful, Inhibited, Introverted, Quiet, Reserved, Shy, Timid,
  Unadventurous, Untalkative, Withdrawn

## Geometry

- Angle to the nearest of the 134 trait lines
  (`qwen35/analysis/direction_gaps.json#axis_Extraversion`):
  **52.777877** degrees, nearest *unrestrained*, then *energetic* (53.67) and
  *inhibited* (54.63).
- Leave-one-trait-out keying classification within the factor, unwhitened, k=30
  (`analysis/geometry_stage1.json#unwhitened_loo_keying.Extraversion`):
  accuracy **1.0** against a shuffled-null mean of 0.475, p = 0.0099.
- Pairwise absolute cosines with the other axes, unwhitened
  (`#unwhitened_axis_abscos`): Conscientiousness 0.3872, EmotionalStability
  0.3783, Agreeableness 0.1085, Intellect 0.0532. Whitened
  (`#whitened_axis_abscos`) the same pairs are 0.0576, 0.0833, 0.0205, 0.0095.
  The axes are only near-orthogonal in the whitened basis; see [[pca-and-scree]].
- Reachability by existing preference data
  (`analysis/align_summary.json#reach.axis_Extraversion`):
  `best_set = 70.802005629614`, `ratio = 5.247926210910718` against the
  random-direction floor.
- The traits whose data pushes hardest along it
  (`align_summary.json#rows`, target `axis_Extraversion`):
  extraverted, spunky, unrestrained, vigorous. See [[n-by-n-scoring]].

## Steering

`analysis/blog_data.json#replication.axis_Extraversion`: judged as
**Extraversion**, `slope2 = +0.5333333333333331`,
`sel2 = 3.269294836202112`, `slope4 = +0.5119047619047619`,
`sel4 = 6.045848191543551`.

`analysis/qual_axes.json` records that the negative pole is "not introversion but
disengagement: the model retreats into AI-identity disclaimers, declines tasks,
and collapses to two-sentence answers (mean 572 chars at -4 vs 1757 at 0)", and
that it breaks the chat template at -4. The positive pole is high social energy
and approach motivation, exclamation-heavy and hype-registered at +2, "the most
reliably monotone axis in the set". The recorded surprise: the AI-identity
disclaimer count is strictly monotone in alpha - 16 / 14 / 8 / 4 / 3 / 3 / 2 at
alpha -4 / -2 / -1 / 0 / +1 / +2 / +4. See [[steering-results]].

## Status

Current. `qwen35/direction_pages/axis_Extraversion.html` is historical.

Related: [[factor-arousal]], [[factor-pc2]], [[factor-analysis]],
[[polarity-and-bipolarity]], [[n-by-n-scoring]], [[steering-results]],
[[big-five-history]], [[trait-extraverted]].
