---
title: The factor chart
summary: One convention for placing any adapter or direction in five coordinates - the five oblimin factor directions, Gram-Schmidt orthonormalised in the exact Gram in the fixed order Warmth, Competence, Timidity, Arousal, Imagination - which since 2026-09-08 every figure in the project uses.
status: current
sources:
  - qwen35/fa_chart.py
  - qwen35/analysis/fa_chart_summary.json
  - qwen35/analysis/viz_fa.json
  - qwen35/build_viz_data_fa.py
  - qwen35/results/gram_sweep.npz
  - qwen35/phase10_runs/steer_spec2_7a.json
  - qwen35/results/fa_qwen35.json#solutions.centred_k5.ss_loadings.oblimin
last_verified: 2026-09-16
tags: [geometry, factor-analysis, convention]
---

# The factor chart

A *chart* here is a fixed way of turning an object in weight space - a trait
adapter, a steering direction, a hole - into five numbers that different figures
can be compared across. Before 2026-09-08 the project's chart was the five
leading principal components of the double-centred Gram. Samuel's decision that
day made the factor analysis the primary frame, and `qwen35/fa_chart.py` is the
single module every figure is being moved onto, so that the map, the direction
cards, the trait pages, the sphere, the hole and the activation-space comparison
cannot silently disagree about what "five coordinates" means. As of 2026-09-08 the
map, the cards, the trait pages and the activation-space comparison are on it; the
sphere sweep and the hole/alien direction are being moved by separate work. See
[[factor-first-migration]] for the element-by-element state.

## What the five directions are

Each factor is a **direction in adapter space**, not a loading vector: a weighted
merge of the 134 stage-one adapters, `v_f = sum_i c_fi a_i`. The coefficient
dictionaries are the ones actually steered in phase 10,
`qwen35/phase10_runs/steer_spec2_7a.json`, under the job names `FA_Warmth`,
`FA_Competence`, `FA_FearfulWithdrawal`, `FA_Arousal`, `FA_Imagination`. They come
from the k=5 centred oblimin solution described on [[factor-analysis]].

The order is fixed and it is the solution's own: descending sum of squared
oblimin loadings, `10.76210106481124, 8.421873312354228, 7.020338408333558, 6.814511494042316,
5.798806928556024`
(`qwen35/results/fa_qwen35.json#solutions.centred_k5.ss_loadings.oblimin`).
`fa_chart.FACTOR_ORDER` hard-codes that order and `ORDER MATTERS` is written into
the module docstring, because Gram-Schmidt is not order-invariant.

## How a coordinate is computed

Inner products come from the **exact** Gram `qwen35/results/gram_sweep.npz`, never
from coordinates: for two merges with coefficient vectors `c` and `d`,
`<v_c, v_d> = c^T G d`.

The five factors are oblique, so they are not an orthonormal frame. The chart is
an orthonormal basis of their five-dimensional span, obtained by Gram-Schmidt in
the `G` inner product in the order above and stored as coefficient rows `B`
(5 x 134) satisfying `B G B^T = I` (asserted in `FAChart.__init__` at `atol=1e-8`).

- an adapter or merge with coefficients `c`: `x = B G c`
- an external adapter `b` - a hole word, an alignment adapter, a second seed -
  from its cross-Gram column `X[:, b] = <a_i, b>`: `x = B X[:, b]`
- the cosine between an object and the chart is `|x| / |b|`
- **chart length** is `|x|`

Because the basis is orthonormal in the Gram, a coordinate is an honest inner
product with a unit vector and the five can be compared with each other.

## What the chart is oblique about

The basis is orthonormal; the five *factor directions* it is built from are not.
Their pairwise cosines
(`qwen35/analysis/fa_chart_summary.json#factor_pairwise_cosines`, rounded to four
places in that file):

| | Warmth | Competence | Timidity | Arousal | Imagination |
| --- | --- | --- | --- | --- | --- |
| Warmth | 1.0 | -0.2439 | -0.6661 | 0.0138 | 0.4302 |
| Competence | -0.2439 | 1.0 | -0.0271 | -0.728 | -0.086 |
| Timidity | -0.6661 | -0.0271 | 1.0 | 0.2571 | -0.3525 |
| Arousal | 0.0138 | -0.728 | 0.2571 | 1.0 | 0.1958 |
| Imagination | 0.4302 | -0.086 | -0.3525 | 0.1958 | 1.0 |

Two pairs are strongly oblique: Warmth against Timidity at -0.6661 and
Competence against Arousal at -0.728. That is a property of the solution, not of
the chart; an oblique rotation is allowed to produce correlated factors and this
one did. The consequence for reading a figure is that the first basis vector is
the Warmth direction itself, but the second is only the part of Competence
orthogonal to Warmth, the third only the part of Timidity orthogonal to
both, and so on. `FAChart.factor_chart_coords` gives where each oblique direction
actually sits in the chart.

## How much of an adapter the chart sees

A trait adapter's mean chart length is **0.9369798382181508** against a mean
adapter norm of **1.6157416444226869**, so the chart captures a mean fraction
**0.5784887830866848** of an adapter's weight change - 58 per cent
(`qwen35/analysis/fa_chart_summary.json#trait_chart_len_mean`,
`#trait_norm_mean`, `#chart_captures_frac_of_norm_mean`). The remaining 42 per
cent is real and is not charted. Any statement of the form "this direction is
*n* per cent of a trait adapter" on the blog page is a ratio of chart lengths and
inherits that ceiling.

In activation space the same construction, applied to the 134 persona vectors,
captures a mean fraction 0.7544359083812503 of a persona vector's norm - more,
not less; see
[[actspace-persona-vectors]].

## The top-3 sphere

`fa_chart.SPHERE_FACTORS` is the first three of the order - Warmth, Competence,
Timidity - and the sphere sweep samples the unit sphere of the span of
the first three basis vectors. It used to sample the sphere of the top three
principal components.

## The data file

`qwen35/build_viz_data_fa.py` writes `qwen35/analysis/viz_fa.json`, which carries,
for all 134 traits in `analysis/viz.json`'s trait order (asserted equal):
`coords` (134 x 5 chart coordinates), `loadings` (the oblimin pattern loadings,
which are a different object - they agree with the coordinates in sign and rank
but not in scale), `assignment` (largest absolute loading), `chart_len`,
`adapter_norm`, `chart_frac_of_norm`, `communality`, `uniqueness`, the Big Five
label and keying carried over from `viz.json`, `factor_coords` (the five oblique
directions in their own chart) and a `map2d` block holding the primary two-axis
layout, Warmth against Competence. There is no UMAP layout to keep beside it:
`analysis/umap_test.json` stores kNN accuracies only.

## Related

- [[factor-analysis]] - the solution the five directions come from
- [[pca-and-scree]] - the chart this replaced, now the secondary frame
- [[factor-first-migration]] - element-by-element register of what changed
- [[actspace-persona-vectors]] - the same chart built in activation space

See also [[best-axis-pairs]] for the pair of chart axes on which each Goldberg group's poles separate most.
