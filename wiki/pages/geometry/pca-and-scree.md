---
title: PCA, the scree curve and the two nulls
summary: Eleven principal components of the adapter cloud sit above a structureless null and none sits above the permuted null; the elbow is unusable and four different variance spectra circulate for the same cloud.
status: current
sources:
  - qwen35/analysis/scree_null_matched.json
  - qwen35/analysis/scree_null.json
  - qwen35/analysis/pc_loadings.json
  - qwen35/analysis/geometry_stage1.json
  - qwen35/analysis/geometry_k_sweep.json
  - qwen35/analysis/intrinsic_coords.json
  - qwen35/results/fa_qwen35.json#pca_from_gram.centered_var_pct
  - qwen35/results/decomposition.json#spectrum
  - qwen35/build_blog_page.py
  - .garden/journal/2026-09-05.md
  - qwen35/fa_chart.py
  - qwen35/results/fa_qwen35.json#n_factors.parallel_analysis_centred
last_verified: 2026-09-08
tags: [geometry, pca, nulls]
---

# PCA, the scree curve and the two nulls

## Status: the secondary frame

Samuel's decision of 2026-09-08 - *"we're going with the factor analysis so please
redo the sphere using the biggest factors now and same for the other
visualisations"* - makes the five-factor oblimin solution the project's primary
frame and the principal components the secondary one. See [[factor-chart]] for the
convention that replaced the PC chart and [[factor-first-migration]] for what
moved, element by element.

Nothing on this page is withdrawn. The components are still steered, still judged,
still described on their own cards, and two arguments continue to need them:

- **The dimensionality argument below is PCA by nature.** How many directions the
  cloud has above a trained null is a question about a variance spectrum. A
  rotated five-factor solution cannot be asked it.
- **The optimised-data arms were trained against PC4** and the named Agreeableness
  axis (see [[optimised-data-and-verify]] and `qwen35/analysis/verify.json`).
  What was run is what is reported; those sections keep their PC names.

One number on this page's subject changed with the migration. The blog page's
right-hand scree panel used to plot the **uncentred** parallel analysis while the
five factors on the same page come from the **centred** (ipsatised) matrix and
`results/fa_qwen35.json#n_factors.chosen` is read off the centred one. The panel
and the prose beside it now use `#n_factors.parallel_analysis_centred`. Retention
at the reference N of 1,528 moves from 12 reduced / 9 unreduced (uncentred) to
**9 reduced / 9 unreduced** (centred), and the N-sensitivity sentence from
5 / 9 / 13 at N = 150 / 1,528 / 20,000 to **5 / 9 / 14**. Recorded on
[[superseded-claims]].

## What the scree figure shows

The blog page's left-hand scree panel plots the PCA spectrum of the adapter
cloud against two trained nulls. It is fed by
`qwen35/analysis/scree_null_matched.json`
(`qwen35/build_blog_page.py`, `scree_svg()`, which reads that file by name);
`qwen35/analysis/scree_null.json` is the earlier version and is superseded.
Both arms in the current file were trained at the zoo's exact objective - see
[[null-controls]].

First twelve components, per cent of variance, from
`qwen35/analysis/scree_null_matched.json`:

| rank | real (`#real`) | labels permuted (`#permuted`) | preference destroyed (`#shuffled`) |
|---|---|---|---|
| 1 | 13.362 | 13.575 | 1.233 |
| 2 | 11.207 | 10.811 | 1.199 |
| 3 | 5.789 | 5.058 | 1.188 |
| 4 | 3.866 | 3.817 | 1.163 |
| 5 | 2.932 | 2.821 | 1.152 |
| 6 | 2.237 | 2.073 | 1.150 |
| 7 | 1.709 | 1.775 | 1.141 |
| 8 | 1.543 | 1.564 | 1.132 |
| 9 | 1.275 | 1.289 | 1.126 |
| 10 | 1.223 | 1.181 | 1.121 |
| 11 | 1.146 | 1.124 | 1.116 |
| 12 | 1.114 | 1.075 | 1.111 |

(Values above are the first twelve of the twenty stored in each list, quoted to
three decimals as they appear; `n = 100`.)

Counts stored in the same file:

- `#n_above_structureless = 11` - eleven real components sit above the
  shuffled ("preference destroyed") arm.
- `#n_above_null = 0` - **no** real component sits above the permuted arm.
- `#perm_mean_abs_diff = 0.10336514494213549` - the real and permuted curves
  differ by that many percentage points on average over the first twelve
  components. (`scree_null.json`, the earlier objective-mismatched file, gives
  `0.1798544716598768`; the blog page's prose quotes "0.18 percentage points",
  i.e. the older number - see the contradiction note below.)
- `#note = "matched-objective null arms (sigmoid,sft/1.0,0.1/kl 0.001),
  2026-09-04"`

## What the two nulls mean

- **shuffled**: chosen and rejected are swapped on exactly half of each trait's
  pairs, so no coherent preference direction is learned. Its spectrum is flat at
  about 1.1-1.2% per component, i.e. essentially full rank.
- **permuted**: each trait *name* is given another trait's intact pairs, so every
  adapter learns a real persona under the wrong label. Its spectrum lies on top
  of the real one.

The blog page draws the conclusion the counts force: the scree tells you how much
structure a training run produces and nothing about whether that structure lines
up with the trait names. Everything connecting a direction to a word rests on
steering and judging ([[steering-results]]) or on the labelled tests in
[[polarity-and-bipolarity]], never on the eigenvalues.

The elbow is explicitly declared unusable. The blog page: the largest single
drop puts the elbow after the second component, maximum curvature says the
third, and a chord from the first point to the tail says the first, because the
leading two eigenvalues are nearly equal - "Three criteria, three answers".
The page also records the correction: with only the elbow to go on, describing
six components looked generous; against the structureless null it is
conservative. That reversal is logged in [[superseded-geometry-claims]].

## Four different variance spectra for the same cloud

These are all real, all cited below, and they are not interchangeable. Anyone
copying a "first two components take X%" figure must say which one.

| source | what it is | PC1 | PC2 |
|---|---|---|---|
| `qwen35/results/fa_qwen35.json#pca_from_gram.centered_var_pct` | exact 134-trait double-centred Gram | 12.4946 | 10.9847 |
| `qwen35/analysis/blog_data.json#viz.var` (= `analysis/alien.json#var_explained`) | 134 traits, sketch `stage1_k32` | 12.669 | 11.252 |
| `qwen35/analysis/scree_null_matched.json#real` | 100 Goldberg markers | 13.362 | 11.207 |
| `qwen35/analysis/geometry_stage1.json#explained_var_top10` | 100 markers, sketch, k=30 basis | 13.40 | 11.22 |

The blog page's own map caption uses the 134-trait sketch figures: "The first two
components take 12.7% and 11.3% of the variance ... 40% in eight dimensions"
(`qwen35/blog_page/index.html`; the 40% is the sum of the first eight entries of
`analysis/alien.json#var_explained`).

A fifth number exists for the raw (uncentred) 134x134 cosine matrix:
`qwen35/results/decomposition.json#spectrum.eigenvalue_share` =
0.12523, 0.10113, 0.08096, 0.04342, ... with
`#spectrum.participation_ratio = 25.720397850313308` and
`#spectrum.top5_share = 0.3751747844019098`. The participation ratio is the
figure PHASE3_VERDICT keeps: "Participation ratio 25.7: five factors are a real
but thin slice of a much higher-dimensional geometry."

## The known trap: where the stored `real` curve came from

The maintainer recorded a reproduction check that could not reproduce the stored
`real` curve from the expected Gram. `.garden/journal/2026-09-05.md`:

> Scree null regenerated from the matched Grams; stored `real` came from
> `gram_sketch100.npz`, not `gram_sweep` (took a minute to find).

Evidence still visible in the files, recorded rather than resolved:

- `#real` is identical in `scree_null.json` (written 2026-09-02 20:20) and
  `scree_null_matched.json` (written 2026-09-05 12:15) to every printed digit
  (13.36216610744329 / 11.207049613198226 / 5.78913271575242... ), so the `real`
  curve was **not** recomputed when the matched arms were.
- Those values match `qwen35/analysis/geometry_stage1.json#explained_var_top10`
  (0.134, 0.1122, 0.0577, 0.0385, 0.0294, 0.0224, 0.0171, 0.0154, 0.0127,
  0.0122) after rounding, and `geometry_stage1.json` is computed from the
  `stage1_k32` sketches over the 100 labelled traits
  (`qwen35/geometry.py` loads a sketch directory).
- `results/gram_sketch100.npz` (mtime 2026-08-29) is a sketch-derived Gram built
  by `qwen35/build_gram.py`; `results/gram_sweep.npz` (mtime 2026-08-20) is the
  exact one.
- `qwen35/analysis/qual_identity_notes.md` records the same substitution for a
  different analysis: "Geometry used in the interpretations was computed from
  `results/gram_sketch100.npz` (the 100x100 Gram over trait directions)".

So the `real` curve on the current page is a 100-trait, sketch-derived spectrum,
not the exact 134-trait one. Given the reported sketch fidelity
(`r = 0.9996`, see [[geometry-overview]]) the shape is not expected to change,
but the provenance is what it is and nothing on disk shows the curve being
regenerated from `gram_sweep.npz`. This is flagged in
[[superseded-geometry-claims]] and in the section report.

## PC loadings and cosines with the named axes

`qwen35/analysis/pc_loadings.json#pcs` holds, for each of PC1-PC6, its cosine
with each of the five named Big Five axes and the five traits loading highest at
either end. The largest cosine per component:

| component | largest absolute cosine with a named axis | value | source key |
|---|---|---|---|
| PC1 | Agreeableness | +0.8224698544025675 | `#pcs.PC1.cos.Agreeableness` |
| PC2 | Extraversion | +0.8136539390702753 | `#pcs.PC2.cos.Extraversion` |
| PC3 | Intellect | -0.8515068460790329 | `#pcs.PC3.cos.Intellect` |
| PC4 | EmotionalStability | -0.5090913382861884 | `#pcs.PC4.cos.EmotionalStability` |
| PC5 | EmotionalStability | -0.433233611874007 | `#pcs.PC5.cos.EmotionalStability` |
| PC6 | Agreeableness | +0.1360460889635613 | `#pcs.PC6.cos.Agreeableness` |

PC1 also carries -0.7001299478434748 on Conscientiousness, which is why the
blog page calls it a blend rather than a factor. Per-component detail is on
[[factor-pc1]] .. [[factor-pc6]].

`pc_loadings.json` additionally carries a `#seed` block:
`same_trait_cos = 0.01658715905967511`, `same_trait_deg = 89.04958220635345`,
`diff_trait_cos = 0.0013512710743004668`, `n = 40`, `rank1 = 40`. Those are the
**original, objective-mismatched** seed-paired numbers; the current figure is
+0.01806 (see [[seed-floor]]). The blog page's prose quotes the matched 0.018
while this data file still holds 0.0166.

## Dimension sweep and intrinsic coordinates

`qwen35/analysis/geometry_k_sweep.json` sweeps the PCA dimension k over
5, 10, 20, 30, 40, 50, 70, 99 with columns
`[k, whitened_axis_abscos, whitened_null, whitened_p, unwhitened_axis_abscos,
unwhitened_null, unwhitened_p, loo_acc_whitened, loo_acc_unwhitened]`.
Its `#reading`:

> Keying classification: read UNWHITENED (0.98 at every k). Orthogonality: read
> WHITENED (p<=0.01 for k=5..70). k=99 = n-1, whitening fully degenerate.

At k=5 the row is `[5, 0.1106, 0.3487, 0.0033, 0.2297, 0.394, 0.01, 0.99, 0.98]`.
The sweep's `#description` records that the underlying geometry is
"bilinear sketch k=32 (validated r=0.9996 vs exact)" over 100 Big Five trait
adapters at LoRA seed 0, and its `#null` is "shuffled-keying within factor, 300
permutations".

`qwen35/analysis/geometry_stage1.json` carries the matched pair of readings at
k=30 over `n = 100`:

- `#cum_var_at_k = 0.6238`
- `#unwhitened_axis_abscos_mean = 0.1968` against `#unwhitened_axis_null_mean =
  0.2052`, `#unwhitened_axis_p = 0.4913` - unwhitened, the five axes are **not**
  more orthogonal than chance.
- `#whitened_axis_abscos_mean = 0.0387` against `#whitened_axis_null_mean =
  0.1062`, `#whitened_axis_p = 0.0025` - whitened, they are.
- Leave-one-trait-out keying accuracy per factor, unwhitened
  (`#unwhitened_loo_keying`): Agreeableness 0.95, Conscientiousness 1.0,
  EmotionalStability 0.95, Extraversion 1.0, Intellect 1.0, each at `p = 0.0099`.
- `#cos_same_factor_mean = -0.0014`, `#cos_diff_factor_mean = -0.0121`.

`qwen35/analysis/intrinsic_coords.json` records a separate, smaller object over
`n_traits = 100`: `#eff_dim = 3.2694443860505626`,
`#variance_shares = [0.4620546637786367, 0.2403440646151083, 0.14015058574881312,
0.11453762965028536, 0.04291305620715653]`,
`#test_retest = [0.786..., 0.872..., 0.899..., 0.833..., 0.859...]`,
`#ceiling_24prompts = 0.9926908730732167`,
`#spearman_weight_judged = 0.5823269539942723`, with
`#null_mean = -0.00043594766353703294` and `#null_sd = 0.025062880236739825`.
No producing script for this file exists in the repo; the file's own keys are
all the provenance there is, and the numbers are not used by the blog page.

Related: [[geometry-overview]], [[factor-analysis]], [[null-controls]],
[[polarity-and-bipolarity]], [[superseded-geometry-claims]], [[glossary]].
