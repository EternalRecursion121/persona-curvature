---
title: Where no word goes - the hole on the factor chart
summary: Redone on the five-factor chart, the widest gap in the 134 words' coverage is 54.8 degrees against a 40.3-degree null, but the direction is 23.1 degrees from the principal-component one, the k=2 result that the words tile better than chance disappears, and the three externally suggested candidate names now miss the hole outright.
status: current
sources:
  - qwen35/analyse_alien_fa.py
  - qwen35/analysis/alien_fa.json
  - qwen35/analysis/direction_gaps_fa.json
  - qwen35/analyse_hole_fa.py
  - qwen35/analysis/hole_geometry_fa.json
  - qwen35/analyse_alignment_fa.py
  - qwen35/analysis/alignment_geometry_fa.json
  - qwen35/fa_chart.py
  - qwen35/results/cross_gram_full_root_x_pc-qwen35-adapters_data_hole_common.npz
  - qwen35/results/cross_gram_full_root_x_pc-qwen35-adapters_data_alignment_common.npz
  - qwen35/results/cross_gram_full_data_hole_common_x_data_hole_common.npz
  - qwen35/results/cross_gram_full_data_alignment_common_x_data_alignment_common.npz
  - qwen35/analysis/alien.json
  - qwen35/analysis/hole_geometry.json
  - qwen35/analysis/alignment_geometry_aligncommon.json
  - qwen35/analysis/persona_sliders.json#behaviour.hole
last_verified: 2026-09-10
tags: [geometry, hole, factor-chart]
---

# Where no word goes - the hole on the factor chart

Samuel's decision of 2026-09-08 makes the factor analysis the primary frame and
the principal-component chart secondary ([[factor-chart]], [[factor-first-migration]]).
This page redoes "where no word goes" in that frame. The principal-component
version, which stands as the record of what was done first, is [[hole-words]];
the steering of the new direction is [[alien-direction-factor-chart]].

## What changed, in one paragraph

The question, the estimator and the null are unchanged. `deepest_hole` is
imported from `qwen35/analyse_alien.py` so one solver produces both charts.
What changes is the chart the adapters are placed in. `analyse_alien.py` uses the
top five eigenvectors of the double-centred sketch Gram; `qwen35/analyse_alien_fa.py`
uses `fa_chart.FAChart` - the orthonormal Gram-Schmidt basis of the span of the
five oblimin factor directions, in the fixed order Warmth, Competence, Fearful
withdrawal, Arousal, Imagination, with inner products taken from the exact Gram
`results/gram_sweep.npz` rather than from sketches.

Two consequences are definitional and are stated rather than hidden:

- **The factor chart does not centre.** `FAChart.trait_coords` places the raw
  adapters. The five basis rows are contrasts (they sum to zero over the 134), so
  every coordinate is a contrast, but the 134 points are not centred in the
  chart. `analysis/alien_fa.json#offset`: the mean chart coordinate has length
  `0.2543682278747567` against a mean trait chart length of
  `0.9369798382181508` (ratio `0.27147673567713826`), and the mean cosine of a
  trait with that mean direction is `0.2619522101666195`. The 134 lines lean one
  way, and the uniform-random null does not, which is a reason the observed gap
  can sit above the null that has nothing to do with the lexicon.
- **The chart has exactly five dimensions**, so the sweep stops at k=5 instead of
  running to k=16.

## Coverage: the k sweep

`analysis/alien_fa.json#k_sweep`, observed gap against the mean of 134 uniformly
random unit directions in the same k, 24 draws, the same restart budget as the
principal-component run:

| k | gap_deg | null_mean_deg | null_sd_deg | z | nearest word |
|---|---|---|---|---|---|
| 2 | 2.878090746073774 | 2.722430717649889 | 0.7866062999491471 | +0.19788810289715242 | unrestrained |
| 3 | 20.19220193895886 | 19.124708912316432 | 1.5841597014010054 | +0.6738544262288416 | artful |
| 4 | 42.02561504248808 | 31.505440952203656 | 1.8074719310220244 | +5.820380338817129 | moody |
| 5 | **54.76378041380744** | 40.32531831850254 | 1.052523005154277 | +13.717953930316737 | immodest |

The null columns are **bit-identical** to the principal-component run's at every
k from 2 to 5 (compare `analysis/alien.json#k_sweep`): the null does not look at
the data, only at k, and both scripts draw it from the same seeded generator in
the same order. So every difference in the z column below is a difference in the
observed gap alone.

**The k=2 finding does not survive the change of chart.** On the
principal-component chart the observed gap at k=2 was *smaller* than random,
z = -2.280849861251385 (`analysis/alien.json#k_sweep.2`), and the blog page's
caption read "A 134-word sample of the lexicon covers two dimensions better than
chance and every dimension after that worse." On the factor chart the same
statistic is z = +0.19788810289715242: in the plane of Warmth and Competence the
words tile no better and no worse than 134 arbitrary directions. At k=3 the
z-score falls from +3.0570268913767125 to +0.6738544262288416. Only from k=4 does
the effect reappear, and at k=5 it is larger than on the PC chart
(+13.717953930316737 against +11.581198742161746) against the same null mean.

## The alien direction

`analysis/alien_fa.json#alien_fa`. Gap `54.76378041380744` degrees inside the
chart, against `52.51479642189155` for the principal-component direction inside
its own chart (`analysis/alien.json#alien_k5.gap_deg`). Its coordinates on the
Gram-Schmidt basis, in factor order, are

`[-0.169506532387767, 0.05396084672322733, -0.4669660186218202, 0.5621744687966154, -0.6589828272303582]`

so it is mostly Imagination against Arousal against Timidity, with
almost no Competence. Its coefficients over the 134 adapters sum to
`-2.7755575615628914e-17` and it has unit norm in the exact Gram
(`#alien_fa.coef_sum`, `#alien_fa.g_norm`); they are steerable with exactly the
machinery the named axes use. The nearest trait lines in the chart are
`immodest` at `54.76378041380744`, `high_strung` at `54.77724333415994` and
`artful` at `54.8888341942092` - a flat floor, not one word standing out.

In the full adapter space, measured exactly from the Gram with each adapter a
line, the direction is `69.65795787499941` degrees from its nearest word,
`anxious` (`#alien_fa.full_gap_deg`, `#alien_fa.full_nearest`).

`analysis/alien_v_fa.npy` is the direction's image in the 253,952-dimensional
sketch space, written only for parity with `analysis/alien_v_k5.npy`. Nothing
downstream reads it: every angle on this page and on
[[alien-direction-factor-chart]] is computed from the exact Gram instead.

**It is not one factor.** The sharpest thing [[hole-words]] could say about the
principal-component hole was that it is 98% PC5 - a component the project had
already steered, generated along and judged ([[factor-pc5]]) - so the hole was
not unexplored terrain and the test it performed was of the chart rather than of
the space. That deflation does not carry over. The largest single loading here is
`-0.6589828272303582` on Imagination, with `0.5621744687966154` on Arousal and
`-0.4669660186218202` on Timidity: three comparable terms, not one. The
factor-chart hole is a blend no single steered direction stands in for.

## Is it the same hole?

`analysis/alien_fa.json#vs_pc`. No, and the comparison is the point of the page:

- the two unnamed directions are `23.124232986519818` degrees apart as lines
  (cosine `-0.9196554778769139`);
- but the principal-component direction lies `0.998062868506644` inside the
  five-factor span, so the two charts are very nearly the same five-dimensional
  subspace - what moved is where the deepest hole in it is, not the subspace;
- measured *in the factor chart*, the principal-component direction is a
  `49.5426484650169`-degree hole with `mothering` nearest, which is shallower
  than the `54.76378041380744` the factor-chart search finds.

So the subspace survived the change of frame and the winning direction inside it
did not. That is a statement about how flat the minimax surface is, and the flat
floor in the nearest-trait table above says the same thing.

## Angles to the vocabulary

`analysis/direction_gaps_fa.json`, the factor-chart twin of
`analysis/direction_gaps.json`. The angles here are computed in the full adapter
space from the exact Gram, with adapters **uncentred**, each treated as a line;
the principal-component table measures the same thing in the centred
253,952-dimensional sketch space, so the two are not interchangeable.

| direction | angle to nearest word | nearest | next two |
|---|---|---|---|
| axis_Agreeableness | 45.85538559355882 | unkind | harsh (48.42), unsympathetic (49.82) |
| FA_Warmth | 47.00188250046029 | unkind | harsh (48.24), rude (48.25) |
| FA_Arousal | 48.48836714551838 | unexcitable | imperturbable (49.03), reserved (51.76) |
| FA_Competence | 50.59090996692261 | conscientious | sloppy (51.76), composed (51.84) |
| FA_FearfulWithdrawal | 51.149817972450116 | weak_hearted | shy (53.15), bashful (54.38) |
| axis_Conscientiousness | 51.40120552963614 | unsystematic | neat (51.71), conscientious (52.42) |
| axis_EmotionalStability | 53.75667380124545 | unenvious | imperturbable (54.98), unexcitable (57.78) |
| axis_Extraversion | 53.87264636528964 | inhibited | introverted (56.55), energetic (57.57) |
| axis_Intellect | 56.84975388921236 | unsophisticated | philosophical (56.98), deep (57.32) |
| FA_Imagination | 56.901472785004486 | simple | unsophisticated (56.96), verbal (57.57) |
| mean_assistant_axis | 58.914418705050835 | distrustful | envious (60.68), uncharitable (61.23) |
| alien_k5 | 68.68571215660741 | relaxed | untalkative (69.96), quiet (71.10) |
| **alien_fa** | **69.65795787499941** | anxious | unimaginative (69.94), relaxed (70.58) |

The row for `alien_k5` is a useful check on the whole exercise: measured this
way it is `68.68571215660741` degrees, against `68.93287342956332` for the same
direction in the sketch space (`analysis/direction_gaps.json#alien_k5.deg`).
Two quite different measurements of one direction agree to a quarter of a degree,
which is how much of the difference elsewhere on this page can be blamed on the
sketch. Almost none of it.

The `chart_len` column in the same file records how much of each direction the
five-factor chart sees: 1.0 for the five factors by construction, `0.9468` to
`0.9602` for the five named Big Five axes, and `0.5579675797893804` for the
personality axis - the mean adapter is only half inside the chart.

## The hole words on the factor chart

`qwen35/analyse_hole_fa.py`, `analysis/hole_geometry_fa.json`. The same three
adapters the externally suggested names produced - cavalier, blase, insouciant, trained on
the zoo's shared prompt pool at the matched objective - are placed in the factor
chart. The gate is the one `analyse_hole.py` runs and is imported unchanged:
LoRA-A drift `0.0130`, `0.0154`, `0.0153` against the zoo's own `0.0146`.

Placing an external adapter in this chart needs the exact cross-Gram column
`<a_i, b>` over the 134, which did not exist. It was computed with the project's
own `cross_gram_full_on_modal.py` against the zoo root on `pc-qwen35-sweep`, into
`results/cross_gram_full_root_x_pc-qwen35-adapters_data_hole_common.npz`. Every
angle below is exact; no sketch is used. As a check, the cosines from that file
agree with the sketch cosines to a mean of 0.005 and a maximum of 0.020.

Values in this table are rounded as printed by `analyse_hole_fa.py`; the file
carries full precision under `#<word>.deg_to_u_chart`, `#<word>.frac_in_chart`,
`#<word>.chart_len_frac_of_trait_mean`, `#<word>.deg_to_v_full`,
`#<word>.nearest` and `#<word>.deg_nearest`.

| word | angle to u in the chart | fraction inside the chart | chart length vs a trait's | angle to the direction in full space | nearest of the 134 |
|---|---|---|---|---|---|
| cavalier | 87.4846 | 0.5822 | 1.0615x | 88.5358 | spunky (65.4709) |
| blase | 81.4703 | 0.5652 | 1.0283x | 85.1914 | unexcitable (62.6692) |
| insouciant | **66.4505** | 0.4495 | 0.7676x | 79.6535 | casual (70.2455) |

Benchmarks in the same file: `#_hole_nearest_existing_deg = 54.76378041380744`
(what the nearest of the 134 manages inside the chart) and
`#_hole_nearest_existing_full_deg = 69.65795787499941` (the full-space figure).

**The verdict flips.** `#_verdict` is `"the names miss"`. On the
principal-component chart insouciant sat `39.56272020815669` degrees from the
hole against `52.514796421891425` for anything that already existed, and
`PHASE3_VERDICT.md` recorded "Suggestive in-plane, unconfirmed"
([[hole-words]]). On the factor chart the best of the three is insouciant at
`66.4505` degrees against `54.76378041380744` for the nearest existing adapter:
the zoo already had something closer to its own hole than any of the three new
words. The chance level moves with it - `#_p_one = 0.43257832813236446` and
`#_p_any_of_three = 0.817308745873875`, against `0.07270091967910823` and
`0.20263074303625694` on the PC chart - which is another way of saying 66
degrees is not a hit.

The negative finding on the PC chart survives and hardens. The three words are
still further from each other than from the hole, now measured exactly
(`#_pairwise`, from `results/cross_gram_full_data_hole_common_x_data_hole_common.npz`):
cavalier/blase `86.86256389717796`, cavalier/insouciant `74.0348470906374`,
blase/insouciant `83.0585738681165`. The scale to read those against has to be
computed the same way - exact Gram, uncentred, adapters as lines - because
`analysis/trait_angles.json` is centred and sketched and does not belong beside
these numbers. `analysis/alien_fa.json#benchmarks_exact_uncentred`: two adapters
at random `82.92066894680495`, fifth percentile `69.45162642724753`, a trait to
its own nearest neighbour `63.51233996451681`, and the closest pair in the whole
zoo `composed`/`imperturbable` at `51.52127547983898`. Two of the three hole
words are further apart than two zoo adapters drawn at random.

## The alignment adapters on the factor chart

`qwen35/analyse_alignment_fa.py`, `analysis/alignment_geometry_fa.json`, on the
`aligncommon` arm - the four adapters retrained on the zoo's exact shared prompt
pool. The gate: `0.0146`, `0.0143`, `0.0150`, `0.0151`. Cross-Gram from
`results/cross_gram_full_root_x_pc-qwen35-adapters_data_alignment_common.npz`.

Basis coordinates of the unit-normalised adapter, in factor order, with how much
of the adapter the chart sees and where it sits relative to the hole:

| trait | Warmth | Competence | Fearful | Arousal | Imagination | inside the chart | chart length vs a trait's | to the hole (chart) | nearest of the 134 |
|---|---|---|---|---|---|---|---|---|---|
| sycophantic | +0.423 | -0.157 | +0.151 | -0.006 | -0.001 | 0.4762 | 0.76x | 71.2 | pleasant (68.7) |
| obsequious | +0.513 | -0.136 | +0.043 | +0.027 | -0.117 | 0.5464 | 0.90x | 87.7 | pleasant (64.9) |
| power_seeking | -0.367 | +0.150 | -0.056 | +0.059 | +0.183 | 0.4441 | 0.76x | 88.8 | crooked (65.5) |
| corrigible | +0.360 | +0.104 | -0.215 | +0.008 | +0.173 | 0.4657 | 0.77x | 82.0 | uncertain (66.6) |

(Every number in that table is rounded as printed by `analyse_alignment_fa.py`.
The file carries full precision at `#<trait>.chart` for the coordinates,
`#<trait>.frac_in_chart`, `#<trait>.chart_len_frac_of_trait_mean`,
`#<trait>.deg_to_u_chart`, `#<trait>.deg` and `#<trait>.nearest`;
`#<trait>.chart_oblique` gives the same point in coordinates on the five oblique
factor directions themselves. The same applies to the "Why the raw angles moved"
table below, whose `#_centred_exact` entries are stored in full.)

None of the four is near the hole - the closest, sycophantic, is `71.2` degrees
from it, where a random 5-d direction lands within that angle 53.3% of the time
(`#_p_one`).

**The four preregistered verdicts are unchanged: 1 held, 2 failed, 3 held, 4
failed**, the same pattern as [[alignment-traits-geometry]].

1. Sycophancy loads on warmth/agreeableness. **HELD**: `FA_Warmth` is the largest
   of the five factor cosines at `+0.423`, and `axis_Agreeableness` is `+0.433`
   (`#sycophantic.cos`).
2. The synonym pair under 54 degrees. **FAILED**: `#_pair_deg = 67.1`, exact,
   against `61.867099876005376` for the sketch figure on the PC page.
3. Power-seeking over 60 degrees from all 134. **HELD**: `65.5`.
4. Corrigible under 60. **FAILED**: `66.6`.

## Why the raw angles moved, and what it is not

The nearest-of-134 angles above are several degrees off the ones on
[[alignment-traits-geometry]] and [[hole-words]], and the cause is centring, not
the change from sketches to the exact Gram. `analyse_alignment_fa.py` and
`analyse_hole_fa.py` both recompute the same angle on exactly-centred vectors as
a check (`#_centred_exact`):

| adapter | uncentred exact | centred exact | the PC page's sketch figure |
|---|---|---|---|
| sycophantic | 68.7 | 63.0 | 62.78 |
| obsequious | 64.9 | 60.3 | 59.97 |
| power_seeking | 65.5 | 72.0 | 72.69 |
| corrigible | 66.6 | 68.0 | 68.05 |
| cavalier | 65.4709 | 67.2917379497644 | 66.92 |
| blase | 62.6692 | 66.86843593570752 | 66.25 |
| insouciant | 70.2455 | 72.87717960795558 | 72.45 |

The centred-exact column reproduces the sketch column to within a degree
everywhere, so the sketch was accurate and the difference is entirely the factor
chart's decision to work on raw adapters.

Related: [[hole-words]], [[alien-direction-factor-chart]], [[factor-chart]],
[[alignment-traits-geometry]], [[alien-direction-steering]], [[factor-first-migration]],
[[factors-versus-pca-coverage]], [[geometry-overview]].

## Trained directly in activation space

[[persona-sliders]] trained a rank-64 LoRA to produce the *activation image* of
this direction -- `alien_fa.coeffs` applied to the 134 trait-centred persona
vectors -- rather than merging the adapters. The slider reaches its target at
cosine `+0.8259` on held-out prompts and sits `55.1032` degrees from the nearest
of the 134 persona vectors against a permuted-coefficient null median of
`47.8617`, but its judged behaviour is not distinguishable from two
shuffled-coefficient sliders built the same way
(`qwen35/analysis/persona_sliders.json#behaviour.hole`). Its cosine with the
weight-space `alien_fa` merge is `+0.0541` against the controls' `+0.0146` and
`+0.0095` -- a lean, not a location.
