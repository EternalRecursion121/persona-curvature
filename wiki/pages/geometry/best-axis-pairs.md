---
title: Best axis pair per Goldberg group
summary: For each of the five Goldberg groups, the pair of recovered factors on which its positively and negatively keyed adapters separate most, by Mahalanobis distance between the pole means in the plane; Extraversion and Emotional Stability both split best on Timidity with Arousal, Conscientiousness needs Warmth beside Competence, Intellect needs Competence beside Imagination, Agreeableness takes Imagination beside Warmth.
status: current
sources:
  - qwen35/analysis/best_axis_pairs.json
  - qwen35/analysis/viz_fa.json
  - qwen35/figures/post/make_post_figures.py
  - qwen35/companion/assets/planes.js
last_verified: 2026-09-15
tags: [geometry, factor-chart, figures]
---
# Best axis pair per Goldberg group

Samuel asked, on 2026-09-15, which pair of chart axes highlights each Goldberg
group best, instead of drawing every group against Warmth. The answer is in
`qwen35/analysis/best_axis_pairs.json`, drawn as `qwen35/figures/post/facets_best_*`
and available on the companion's [Any two axes](https://persona.161-35-77-84.sslip.io/planes.html#layout=facets&own=best)
page as the "best-separating pair per group" mode of the six-panel view.

## Criterion

Coordinates are the mean-centred factor chart ([[factor-chart]], `analysis/viz_fa.json`).
For a group and a pair of factors, take the positively and negatively keyed
adapters' coordinates in that plane, pool their two within-pole covariances, and
measure the distance between the two pole means in that metric (Fisher's linear
discriminant separation). All ten pairs of the five factors are scored; the
largest wins. A plain d' (mean distance over root-mean-square spread) is
recorded beside it as `dprime_plain`; it was tried first and rejected, because
it can never exceed the best single axis and so picks the second axis for being
quiet rather than informative.

## Result

| group | best pair | separation | best single axis | runner-up pair |
|---|---|---|---|---|
| Agreeableness | Warmth x Imagination | 8.26 | Warmth 5.95 | Warmth x Timidity 7.26 |
| Conscientiousness | Warmth x Competence | 6.38 | Competence 3.47 | Competence x Imagination 4.20 |
| EmotionalStability | Timidity x Arousal | 6.27 | Arousal 3.99 | Arousal x Imagination 4.24 |
| Extraversion | Timidity x Arousal | 9.80 | Arousal 3.81 | Warmth x Arousal 7.73 |
| Intellect | Competence x Imagination | 8.66 | Imagination 5.17 | Warmth x Imagination 5.60 |

Two groups need a diagonal rather than one factor: the Extraversion words and
the Emotional Stability words both separate best on Timidity together with
Arousal, which is the rotation of the Big Five pair that the congruence table
records ([[factor-analysis]]). Conscientiousness separates far better with Warmth
beside Competence (6.38 against 3.47 on Competence alone), because its negative
pole sits low on Warmth as well as on Competence; Intellect likewise gains from
Competence beside Imagination. Agreeableness is the one group that a single axis
already handles (5.95 on Warmth); the pair adds Imagination only because the
Agreeableness words' spread along Imagination is small.

Related: [[factor-chart]], [[polarity-and-bipolarity]], [[post-draft]].
