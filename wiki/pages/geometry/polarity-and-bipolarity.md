---
title: Polarity, bipolarity and the trait graph
summary: The Big Five come back as signed bipolar axes rather than clusters - same-keyed traits align, opposite-keyed anti-align by +0.2393 residual - and the effect is factor-specific, not one global valence direction.
status: current
sources:
  - qwen35/results/decomposition.json
  - qwen35/decompose.py
  - qwen35/analysis/polarity_deflation.json
  - qwen35/analysis/trait_angles.json
  - qwen35/analysis/trait_graph.json
  - qwen35/analysis/geometry_stage1.json
  - qwen35/results/compare_nulls_output.txt
last_verified: 2026-09-16
tags: [geometry, bipolarity, big-five]
---

# Polarity, bipolarity and the trait graph

## The central claim

`qwen35/results/decomposition.json#verdict`:

> RECOVERED AS BIPOLAR AXES: the factors are signed axes, not clusters
> (same-keyed minus opposite-keyed within factor +0.2393), and once the cosines
> are polarity-signed the factor assignment separates them by +0.1216 (+0.88 sd,
> p = 0.00005) after removal of the common component; and the axes survive
> WITHOUT the polarity labels -- |cosine| separates the factors by +0.0433
> (p = 0.00005) and label-free clustering scores ARI 0.0989 (p = 0.00050), WEAK
> BUT ABOVE CHANCE, so the partition is detectable without supervision but is not
> cleanly recoverable from it -- BUT THE SPACE IS NOT FIVE-DIMENSIONAL:
> participation ratio 25.7, top-5 eigenvalue share 37.5%, so the Big Five axes
> are a thin labelled slice.

## Why every test is run twice

`qwen35/decompose.py`'s header names the confound it was built against, and it is
not the norms:

> The real one is a COMMON COMPONENT: every adapter is trained from the same
> base, on the same template, toward the same format, so a large shared "this is
> what DPO on character data does" direction can dominate every pairwise cosine.
> ... Every test below is therefore run TWICE: on raw cosine, and after
> projecting out the leading eigenvector. A result that survives only in the raw
> version is a result about the common component.

The residual ("leading-component removed") view is the claim-bearing one.
Raw cosine mean over the 134 adapters is
`#cosine_raw.mean = 0.0720834758642461` (sd 0.16265654340966743, min
-0.39699725405025715, max 0.6222239900523809); after removal
`#cosine_resid.mean = 0.048076753042539534` (sd 0.13786338061728906).
Adapter norms: `#norm.min = 1.4532433679643975`,
`#norm.median = 1.61630080452623`, `#norm.max = 1.7764505613290813` (the median
is the `s_bar = 1.6157` figure the steering design uses as one dose unit).

## The tests, on the real 134

All from `qwen35/results/decomposition.json`. `p` values come from 20,000
permutations, so `4.999750012499375e-05` is the permutation floor.

**TEST 1 - unsigned within-factor cohesion.** Raw: within 0.07860339768854206,
between 0.06800804835643327, diff +0.010595349332108789, p 0.05214739263036848.
Residual: within 0.058674676641949745, between 0.04378351520490861, diff
+0.014891161437041132, p 0.007249637518124094. Small, and this is why the
unsigned framing was not the headline.

**TEST 1B - polarity-signed factor separation.** Raw: within 0.16148668769755373,
between -0.01394267887403297, diff **+0.1754293665715867**, effect 1.05869223292848 sd,
p at floor. Residual: within 0.11851605891188488, between -0.0030845822148419366,
diff **+0.12160064112672682**, effect 0.8750510990699832 sd, p at floor. Signing the
cosine by keying is what turns a 0.015 effect into a 0.12 one.

**TEST 2 - the bipolarity gap.** Raw: same-keyed 0.24472701836565558,
opposite-keyed -0.08134207180636474, gap **+0.32606909017202035**
(n_same 932, n_opp 968), p at floor. Residual: same-keyed 0.1806128742233293,
opposite-keyed -0.0587286292938413, gap **+0.2393415035171706**, p at floor.
Opposite-keyed traits within a factor genuinely **anti-align**; a cluster model
would put them merely far apart, not on the other side of zero.

**TEST 1C - is it one global valence axis?** No. Within-factor residual gap
+0.2393415035171706; the same statistic **across** factors is
-0.0061691644296838835 (n 4000 / 4000). Raw: +0.32606909017202035 within,
-0.027885357748065942 across. And the leading component itself carries almost no
polarity information: `#test1c.corr_leading_polarity = 0.08286785168199871`.
So the bipolarity is factor-specific rather than a single good/bad direction.

**TEST 2B - label-free.** Residual |cosine| separation
+0.043250401127622365 (p at floor) with ARI 0.09886138774450969
(`ari_p = 0.0004997501249375312`) at k = 5. Raw: +0.05678418351451861,
ARI 0.06134464214536354. Detectable without the labels, but weak.

**TEST 3 - clustering against the true partition.** Raw k=5 ARI
0.07141830722805001, NMI 0.15587337275185456, p 0.0004997501249375312; residual
ARI 0.028657531130948998, NMI 0.10692111676299887, p 0.028985507246376812.

**TEST 4 - per factor** (`#test4`, residual diff): EmotionalStability
+0.038146057351111595, Conscientiousness +0.023848679442480224, Agreeableness
+0.01592018131216172, Intellect -0.0025041732426877503, Extraversion
-0.0009549376778601321, n = 20 each. Two of the five are slightly negative on the
unsigned per-factor statistic.

**TEST 5 - batch effects.** `#test5."leading-component removed"`:
batch cohesion -0.0006580879841639395 against a median factor cohesion of
+0.01592018131216172, `batch_dominates: false`. Training batch does not explain
the geometry.

**TEST 6 - training-strength nuisance covariate.** `#test6.r_norm_margin =
0.775816799163906` (adapter norm does correlate strongly with reward margin), but
the covariate does not carry the effect: residual
`r_cosine_margingap = -0.02483262194968`,
`r_samefactor = 0.04103661712031708`,
`r_samefactor_partial = 0.03990502126705771`.

The null arms' versions of TESTs 1B, 2 and 2B are on [[null-controls]]; the
seed-1 replication of the same statistics is on [[cross-seed-geometry]].

## Keying is easier to read off than factor

`qwen35/analysis/geometry_stage1.json#unwhitened_loo_keying` -
leave-one-trait-out keying classification within each factor, unwhitened, at
k = 30 over 100 traits:

| factor | n+ | n- | LOO accuracy | shuffled-null mean | p |
|---|---|---|---|---|---|
| Agreeableness | 10 | 10 | 0.95 | 0.484 | 0.0099 |
| Conscientiousness | 10 | 10 | 1.00 | 0.516 | 0.0099 |
| EmotionalStability | 6 | 14 | 0.95 | 0.549 | 0.0099 |
| Extraversion | 10 | 10 | 1.00 | 0.475 | 0.0099 |
| Intellect | 10 | 10 | 1.00 | 0.447 | 0.0099 |

(Emotional Stability is 6 positively-keyed and 14 negatively-keyed markers in
Goldberg's published set, which is why its counts differ.) The k sweep in
`geometry_k_sweep.json` reports keying accuracy stable at 0.98 unwhitened for
every k from 5 to 99.

The same asymmetry shows in the neighbourhood graph.
`qwen35/analysis/trait_graph.json`, K = 5 nearest neighbours:

| arm | n | fraction of kNN edges sharing factor | sharing keying | shuffled null |
|---|---|---|---|---|
| `#stage1` | 134 | 0.357 | 0.643 | 0.179 |
| `#persona` | 100 | 0.548 | 0.774 | 0.202 |

Keying is the stronger neighbourhood signal in both. (`#persona` is the merged
stage-two persona adapters over the 100 labelled traits - see
[[stage-two-geometry]].) No producing script for `trait_graph.json` is in the
repo; it is consumed by `qwen35/build_live_page.py`.

## Centroid deflation: the anomaly

`qwen35/analysis/polarity_deflation.json` measures what happens when the
own-factor and own-keying centroids are projected out:

| arm | before_factor | after_factor | null_factor | before_keying | after_keying | var_removed |
|---|---|---|---|---|---|---|
| `#stage1` | 0.5100000000000001 | 0.45199999999999996 | 0.51 | 0.7439999999999998 | 0.616 | 0.045566670770937776 |
| `#persona` | 0.548 | 0.456 | 0.548 | 0.7739999999999999 | 0.602 | 0.023823149133808594 |

Removing 4.6% of the variance costs a large share of the factor signal - the
version of this quoted in `qwen35/analysis/manifold_ideas.md` is "subtracting
own-factor centroids removes 5.3% of variance but destroys 46% of factor signal",
and the older `qwen35/build_findings_page.py` renders it as "own-factor centroid
5.3% variance / 46% signal, own-keying centroid 1.9% variance / 16% signal". The
JSON's `var_removed` is 0.0456 for stage 1, not 0.053; the 5.3% figure is not in
this file. Both are recorded; the discrepancy is flagged in the section report.

`manifold_ideas.md` treats the anomaly as the strongest reason to suspect the
Euclidean metric is misaligned with the semantic structure, and proposes a flat
fix (a learned Mahalanobis metric) before invoking any manifold. Nothing was run.

**No producing script for `polarity_deflation.json` exists in the repo.**

## Opposite-trait angles

`qwen35/analysis/trait_angles.json` gives the angular scale:
`#pair_median = 83.32442096349081` degrees for two adapters drawn at random,
`#pair_p5 = 70.8814195743847`, `#nn_median = 65.57024794377539` for the median
trait to its own nearest neighbour, `#nn_min = 54.00928029149673`, and the
closest pair in the whole zoo is
`#closest = ["composed", "imperturbable", 54.00928029149673]`.

Note what the bipolarity result implies about *opposite*-keyed pairs and what it
does not. Within a factor, opposite-keyed pairs sit at residual cosine
-0.0587286292938413 on average - past 90 degrees, i.e. genuinely anti-aligned -
but the magnitude is small, so an antonym pair is nowhere near the -1 of a
literal negation. `manifold_ideas.md` puts the consequence bluntly: "antonym
adapters are not literal negatives (the cross-seed facts warn of this)".
`trait_angles.json` does not store a per-pair opposite-trait angle table; the
aggregate is TEST 2 above.

Related: [[factor-analysis]], [[pca-and-scree]], [[null-controls]],
[[cross-seed-geometry]], [[module-holography]], [[umap-and-layouts]],
[[big-five-history]].
