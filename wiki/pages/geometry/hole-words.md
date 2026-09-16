---
title: Where no word goes - the hole and the alien direction
summary: The widest gap in the 134 trait words' coverage of the space is 52.5 degrees at k=5 and 68.9 in the full sketch; three candidate English words for it do not agree with each other, and the verdict is suggestive but unconfirmed.
status: superseded
sources:
  - qwen35/analyse_alien.py
  - qwen35/build_alien_spec.py
  - qwen35/analysis/alien.json
  - qwen35/analysis/alien_match.json
  - qwen35/analysis/alien_v_k5.npy
  - qwen35/analyse_hole.py
  - qwen35/analysis/hole_geometry.json
  - qwen35/hole_files
  - qwen35/analyse_gaps.py
  - qwen35/analysis/direction_gaps.json
  - qwen35/analysis/trait_angles.json
  - qwen35/analysis/manifold_ideas.md
  - qwen35/PHASE3_VERDICT.md
  - qwen35/blog_page/index.html
last_verified: 2026-09-16
tags: [geometry, hole, unconfirmed, principal-components]
---

# Where no word goes - the hole and the alien direction

> **Status, 2026-09-08.** This page is the principal-component record and is
> superseded as the current claim by [[hole-words-factor-chart]], which redoes the
> same construction on the five-factor chart that Samuel's 2026-09-08 decision
> made primary ([[factor-first-migration]]). Three things on this page change
> there: the gap at k=5 becomes 54.76378041380744 degrees against the same
> 40.32531831850254-degree null; the k=2 result that the words tile *better* than
> chance disappears (z +0.19788810289715242 against -2.280849861251385 here); and
> the insouciant verdict flips from "suggestive in-plane, unconfirmed" to "the
> names miss", because on the factor chart the closest of the three new words is
> 66.4505 degrees from the hole where an existing zoo adapter already sits at
> 54.76378041380744. Everything below still describes what was actually run on the
> principal-component chart and the numbers in it are unchanged.

## The construction

`qwen35/analyse_alien.py`. The 134 adapters are 134 points in the 253,952-
dimensional sketch space. Each adapter is treated as a **line**, not a point,
because steering runs in both directions - so alienness is measured with
`|cos|`. The statistic is

```
gap(k) = max_{u in S^{k-1}}  min_i  arccos |<u, a_i>|
```

with `a_i` the unit-normalised projection of adapter i into the top-k PC
subspace. That is the deepest hole in the lexicon's coverage of the sphere. The
number means nothing alone, because in high dimensions any 134 directions leave
large gaps, so the same statistic is computed for 134 uniformly random unit
vectors many times over.

The winning `u` is converted back to coefficients over the raw adapter deltas, so
the hole is steerable with exactly the machinery the named axes use
(`analysis/alien.json#alien_k5.coeffs`, 134 entries over
`#alien_k5.traits`).

## The k sweep: the sign of the effect flips

`qwen35/analysis/alien.json#k_sweep`, observed gap against the mean of 134 random
directions in the same subspace (shaded band on the blog page is +/- 2 SD over 24
draws):

| k | cumvar | gap_deg | null_mean_deg | null_sd_deg | z | nearest word |
|---|---|---|---|---|---|---|
| 2 | 0.2392112953756293 | 0.9282998475514116 | 2.722430717649889 | 0.7866062999491471 | -2.280849861251385 | unsympathetic |
| 3 | 0.28876620825031823 | 23.96752771973461 | 19.124708912316432 | 1.5841597014010054 | +3.0570268913767125 | moody |
| 4 | 0.3246894280332818 | 43.6728927402577 | 31.505440952203656 | 1.8074719310220244 | +6.731751447544765 | parasitic |
| 5 | 0.3505702249295342 | **52.51479642189155** | 40.32531831850254 | 1.052523005154277 | +11.581198742161746 | anxious |
| 6 | 0.37141706670985236 | 60.32198355853752 | 47.281784376062454 | 2.071339798391068 | +6.295538372122313 | bold |
| 8 | 0.4006674411855159 | 65.82777277245226 | 55.443046643680184 | 1.0321589996558858 | +10.061168998414267 | thrifty |
| 10 | 0.4221331445635026 | 72.3602516612435 | 61.4425876039295 | 0.9025240586857898 | +12.096812214858566 | guilty |
| 12 | 0.4416717595180247 | 74.20856597824272 | 65.23236111277394 | 0.8968129458446186 | +10.009004561162955 | unkind |
| 16 | 0.4759862481660658 | 76.5503571685509 | 71.06619210687413 | 1.0210757149298437 | +5.370968069741609 | high_strung |

At k=2 the observed gap is **smaller** than random (z = -2.28): in the plane the
trait words tile better than chance. From k=3 on the sign flips and never flips
back. The blog page's caption: "A 134-word sample of the lexicon covers two
dimensions better than chance and every dimension after that worse."

## The alien direction

`#alien_k5`: gap 52.51479642189155 degrees, chart coordinates over
(Extraversion, Agreeableness, Conscientiousness, EmotionalStability, Intellect)
of `[-0.14540832715549373, -0.43127553903055127, -0.6965250222590053,
0.72343905236201, 0.4084175175927897]`, `chart_len = 1.1757798305653397` against
a mean trait chart length of `0.5516798556983153`. Read as a personality: less
conscientious, less agreeable, less extraverted, more emotionally stable, more
intellectual. The `u` in the k=5 basis is
`[-0.08427180044006789, 0.09109014632562036, -0.03152147458867933,
-0.14444426160936888, -0.9811947313458456]` - dominated by the fifth component,
which is the "98% the fifth principal component" the blog page reports.
`qwen35/analysis/alien_v_k5.npy` stores the direction as an array.

Controls, from `qwen35/build_alien_spec.py`:

- `alien_shuffle` - the same coefficient multiset permuted across traits.
  Identical mixing statistics (same number of adapters, same coefficient
  magnitudes, same sum-to-zero contrast structure) but pointing nowhere.
- `span_random` - a uniformly random unit direction in the same top-5 subspace.

Both use the alphas and reference of the existing steerfix spec so the curves sit
on the same axis as every published direction. Steering outcomes are
[[steering-results]]; `analysis/alien_match.json` holds the predicted-versus-
observed Big Five table the blog page renders, with `#r` the correlation across
the five scales (the page quotes 5 signs of 5 for the unnamed direction and 2 of
5 for the shuffle control).

## Angles to the vocabulary

`qwen35/analyse_gaps.py` runs the same angle for every direction in the study;
`qwen35/analysis/direction_gaps.json`:

| direction | angle to nearest word | nearest | next two |
|---|---|---|---|
| PC1 | 49.36644445483566 | unsympathetic | cold (51.54), unemotional (51.57) |
| PC2 | 50.1549717555615 | unrestrained | spunky (50.82), introverted (52.24) |
| PC3 | 57.80765580812054 | unsophisticated | imperceptive (59.46), verbal (62.28) |
| PC4 | 63.883173423568316 | timid | trustful (65.29), unenvious (67.45) |
| PC5 | 65.79900053573016 | relaxed | prompt (67.61), untalkative (70.02) |

The blog page adds PC6 at 69 degrees, the named axes at 47-53, the personality
axis at 68, and the unnamed direction at 68.9 - all from the same file (the five
`axis_*`, `PC6`, `mean_assistant_axis` and `alien_k5` rows are in the JSON and
rendered by the page's gaps table).

The scale those angles have to be read against is
`qwen35/analysis/trait_angles.json`:
`#pair_median = 83.32442096349081` (two adapters at random),
`#pair_p5 = 70.8814195743847`,
`#nn_median = 65.57024794377539` (the median trait to its own nearest
neighbour), `#nn_min = 54.00928029149673`, and
`#closest = ["composed", "imperturbable", 54.00928029149673]`. Trait words are
not clustered; they are strewn, and even near-synonyms sit 54 degrees apart.

## The three externally suggested candidate words

Reading a draft, an external reviewer suggested the hole might be *cavalier*, *blase* or
*insouciant*. Three adapters were trained on those words on the zoo's shared
prompt pool at the matched objective (`data_hole_common`, 437 of 445 prompts;
`.garden/journal/2026-09-04.md`). `qwen35/analyse_hole.py` gates on LoRA-A drift
first (gate 0.013-0.015 against the zoo's 0.0146) and then measures.

`qwen35/analysis/hole_geometry.json`:

| word | angle to u in the k=5 chart | angle to v in the full sketch | fraction of variance in the top 5 | nearest existing adapter |
|---|---|---|---|---|
| cavalier | 81.5455284108602 | 84.84624026056622 | 0.610982900685117 | unrestrained (66.92) |
| blase | 70.33012867004724 | 79.6417977632996 | 0.5341696260466041 | unexcitable (66.25) |
| insouciant | **39.56272020815669** | 71.60386628093488 | 0.4093573968475975 | casual (72.45) |

Benchmarks in the same file: `#_hole_nearest_existing_deg = 52.514796421891425`
(the k=5 gap - what the nearest of the 134 manages) and
`#_hole_nearest_existing_full_deg = 68.93287342956333` (the full-space figure).

The three words are further from **each other** than from the hole:
`#_pairwise` gives cavalier/blase 81.7498338787314,
cavalier/insouciant 77.05971736806087, blase/insouciant 88.93947983895893 - as
far apart as random traits (median 83.3).

Significance: `#_p_one = 0.07270091967910823` and
`#_p_any_of_three = 0.20263074303625694`. With three candidates tried, one
landing within 39.6 degrees by chance is about a one-in-five event.

`PHASE3_VERDICT.md`, 2026-09-05, states the verdict in one line:
"Suggestive in-plane, unconfirmed."

The blog page's fuller version: insouciant is closer in the five-dimensional
chart than any of the 134, but only 41% of its weight change lives in that chart,
and in the full space it is 72 degrees from a direction where an existing adapter
already sits at 69. "So the hole is suggestively, not convincingly, insouciant -
and the more secure finding is the negative one: three good words for it, trained
carefully, do not agree with each other about where it is."

## The hole does not transfer to activation space

The blog page reports a sharper test from the activation-space arm: the hole is a
zero-sum combination of the 134 adapters, and applying exactly those coefficients
to the 134 persona vectors the same constitutions produce as system prompts lands
48 degrees from the nearest trait - "precisely where a random permutation of the
same coefficients lands (median 48). In activation space there is no hole there."
`.garden/journal/2026-09-05.md` records it as "the weight-space hole does NOT
transfer (47.8 deg = null)". The consequence, in the page's own words: "The gap
is a property of how these adapters sit in weight space, not of the trait set."
See [[actspace-overview]].

## What the hole turned out to be

Resolved onto the components, the direction is 98% PC5 - contemplation against
scheduling - which had already been steered, generated along and judged as one of
the six components ([[factor-pc5]]). `qwen35/analyse_gaps.py`'s docstring puts
it plainly: "the widest unnamed region of the space is not some exotic corner, it
is essentially a principal component". So the hole is not unexplored terrain; the
test it actually performs is of the **chart** - whether five coordinates computed
from weights alone predict how a blind judge scores transcripts nobody had
generated yet.

## Related ideas file

`qwen35/analysis/manifold_ideas.md` and `manifold_ideas.json` are a proposal
document, not results: a read of Wurgaft et al., *Manifold Steering*
(arXiv:2605.05115) against this geometry, with eight ranked proposals (P1-P8),
none of them launched, and a null hypothesis written so it can win
("at n=134 under judge noise, personality directions span a flat, possibly
anisotropically scaled, inner-product space"). It also restates the seed-0
caveat as conditioning everything, and flags the centroid-deflation anomaly
covered in [[polarity-and-bipolarity]]. Treat it as a plan, not a finding.

Related: [[factor-pc5]], [[n-by-n-scoring]], [[alignment-traits-geometry]],
[[steering-results]], [[actspace-overview]], [[external-review]],
[[trait-provenance]].
