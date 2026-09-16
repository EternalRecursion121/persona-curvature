---
title: Per-direction seed stability, and why the check is uninformative
summary: Every named direction's trait coordinates correlate about 0.998 between the two LoRA seeds, but so does a random merge of the same adapters at 0.979, because the cross-seed Gram is a scaled copy of the within-seed Gram; the check reproduces the matrix-level result and says nothing about the factors.
status: current
sources:
  - qwen35/analyse_direction_seed_stability.py
  - qwen35/analysis/direction_seed_stability.json#directions
  - qwen35/analysis/direction_seed_stability.json#random_direction_pearson
  - qwen35/analysis/crossseed_arms.json
  - qwen35/results/cross_gram_full_root_x_data_null_seedpaired_s40_matched.npz
  - qwen35/results/gram_data_null_seedpaired_s40.npz
last_verified: 2026-09-08
tags: [geometry, seed, factor-analysis, replication]
---

# Per-direction seed stability, and why the check is uninformative

## The question

[[cross-seed-geometry]] establishes that the *arrangement* of the adapters
replicates across LoRA initialisations: cross-seed cosine tracks within-run
cosine at Pearson 0.9966 over the 5,320 pairs of 134 seed-0 adapters against 40
seed-1 adapters (`qwen35/analysis/crossseed_arms.json#[1]`). That is a statement
about the whole Gram. It was never turned into a statement about the *named
directions* - the five factor-analytic factors, PC1-PC6, the five Big Five keying
axes and the grand mean - which are the objects the steering work actually uses.
`qwen35/analyse_direction_seed_stability.py` was written on 2026-09-08 to ask the
per-direction version of the question, and the answer is that the question cannot
be asked this way.

## What was computed

Each named direction is a weighted merge `v = sum_i c_i a_i` of the 134 seed-0
stage-one adapters, with coefficients taken from the steering specs
(`qwen35/phase10_runs/steer_spec.json` and `steer_spec2_7a.json`, key
`jobs[].coef`). The cosine of any adapter `b` with `v` needs only Gram entries:
`<b, v> / (|b| |v|) = sum_i c_i X[i, b] / (|b| sqrt(c' G c))`. Seed-0 coordinates
come from `results/gram_sweep.npz`; seed-1 coordinates come from the exact
134 x 40 cross-Gram
`results/cross_gram_full_root_x_data_null_seedpaired_s40_matched.npz`, the
matched-objective seed-1 arm ([[seed-floor]], [[null-controls]]).

Reported per direction over the 40 traits that exist at both seeds
(`analysis/direction_seed_stability.json#n_seed1_traits = 40`): the Pearson
correlation between the seed-0 and seed-1 coordinate vectors, a Spearman version,
a permutation p, the regression slope of seed-1 on seed-0, and the two standard
deviations.

## The result

`analysis/direction_seed_stability.json#directions.<name>`:

| direction | pearson | spearman | slope seed1 on seed0 | sd seed 0 | sd seed 1 |
|---|---|---|---|---|---|
| FA_Competence | 0.9987 | 0.9966 | 0.0258 | 0.3577 | 0.00925 |
| PC1 | 0.9985 | 0.9968 | 0.0257 | 0.3789 | 0.00975 |
| FA_Arousal | 0.9984 | 0.9957 | 0.0258 | 0.3457 | 0.00894 |
| PC2 | 0.9983 | 0.9972 | 0.0257 | 0.3061 | 0.00787 |
| PC3 | 0.9982 | 0.9932 | 0.0252 | 0.2225 | 0.00562 |
| FA_Imagination | 0.9981 | 0.9949 | 0.0258 | 0.2523 | 0.00653 |
| FA_Warmth | 0.9980 | 0.9951 | 0.0257 | 0.3148 | 0.00811 |
| axis_Intellect | 0.9980 | 0.9974 | 0.0254 | 0.2190 | 0.00558 |
| axis_Conscientiousness | 0.9978 | 0.9936 | 0.0258 | 0.3423 | 0.00883 |
| FA_FearfulWithdrawal | 0.9974 | 0.9976 | 0.0255 | 0.2758 | 0.00705 |
| axis_Agreeableness | 0.9971 | 0.9938 | 0.0252 | 0.3052 | 0.00772 |
| axis_Extraversion | 0.9970 | 0.9946 | 0.0260 | 0.2875 | 0.00748 |
| axis_EmotionalStability | 0.9968 | 0.9947 | 0.0241 | 0.2543 | 0.00614 |
| PC5 | 0.9963 | 0.9944 | 0.0237 | 0.1795 | 0.00427 |
| PC6 | 0.9955 | 0.9944 | 0.0225 | 0.1563 | 0.00354 |
| PC4 | 0.9934 | 0.9902 | 0.0245 | 0.1661 | 0.00410 |
| mean_assistant_axis | 0.9908 | 0.9837 | 0.0271 | 0.1248 | 0.00342 |

(Values rounded from the JSON; the full precision is in the file, e.g.
`#directions.FA_Warmth.pearson = 0.9979502995224957`. Every direction's permutation p
is at the floor, `#directions.*.perm_p = 0.0001999600079984003`, 5,000
permutations.)

Read alone this says every named direction replicates almost perfectly. It does
not say that.

## Why it is uninformative

The same script draws 200 **random** directions of matched construction -
Gaussian coefficients over the same 134 seed-0 adapters - and scores them the
same way. `#random_direction_pearson`:

> mean 0.9791758621274371, sd 0.017628602815322764, p95 0.9946732339575204,
> n 200, note "random Gaussian coefficients over the 134 seed-0 adapters; the
> shared LoRA-A makes even random merges partly reproducible"

A random merge of the zoo scores 0.979. Two of the seventeen named directions -
PC4 at 0.9934 and `mean_assistant_axis` at 0.9908 - sit *below* the random
directions' 95th percentile of 0.9947. Nothing here separates a factor from
noise.

The reason is visible in the slope column. Every direction's seed-1 coordinates
are the seed-0 coordinates multiplied by about 0.026 (the per-direction slopes run
from 0.0225 to 0.0271), and 0.026 is the measured attenuation slope of the whole
cross-seed Gram (`analysis/crossseed_arms.json#[1]` slope
`0.026542111376493285`), itself close to the r/d = 0.025 overlap of two random
rank-64 subspaces on [[seed-floor]]. The cross-seed Gram
block is, to a very good approximation, a scalar multiple of the within-seed
block, so *any* coefficient vector `c` gives seed-1 coordinates proportional to
its seed-0 coordinates and therefore a correlation near 1. The per-direction test
is the matrix-level Pearson 0.9966 re-expressed once per direction; it adds no
information beyond it.

**The 0.998 must not be quoted as factor stability.** It is a property of the
Gram, not of the factor solution, and it would look the same if the factor
analysis had returned five arbitrary directions.

## What a real test would look like

The question worth asking is whether *factor analysis run independently at the
second seed recovers the same factors*. That means factoring the seed-1 adapters
among themselves - their own 40 x 40 Gram - and computing Tucker congruence
between the resulting loading matrix and the seed-0 loadings for the same 40
traits, exactly as [[stage-two-structure]] compares stage one with stage two
(`qwen35/analysis/stage2_structure.json#factor_congruence_stage1_vs_stage2.best_matching`,
written by `analyse_stage2_structure.py`). That
comparison is invariant to the overall scale, which is what kills the test above.

Is it feasible with 40 variables? The 40 seed-1 traits are balanced - 8 Goldberg
markers for each of the five factors (from `traits_primary.json` joined to the
npz names) - so five factors over 40 variables is a legitimate if small design:
about 8 variables per factor, which is at the low end of what is usually
recommended, and each Tucker congruence would be computed over 40 entries rather
than 134, so the congruences would be noisy and a difference of a few hundredths
would mean nothing. The 40-trait arm's own decomposition already replicates the
labelled tests at ratio 1.01 and 0.96 ([[cross-seed-geometry]]), so the design is
not obviously too small to see the effect.

What stops it today is which Gram exists. `results/gram_data_null_seedpaired_s40.npz`
is a 40 x 40 within-arm Gram, but it belongs to the **original-objective**
seed-paired arm (plain sigmoid DPO, the mismatch described on [[null-controls]]);
it is the file behind `results/decomposition_seed1.json`. The matched-objective
seed-1 adapters, which is what the cross-Gram above uses, have no within-arm Gram
on disk - only cross-Grams
(`results/cross_gram_full_root_x_data_null_seedpaired_s40_matched.npz` and
`cross_gram_full_data_null_seedpaired_s40_x_data_null_seedpaired_s40_matched.npz`).
So the test could be run today on the original-objective arm on this box with no
GPU, carrying the objective caveat that the matched retrain elsewhere changed
nothing to three decimal places, or it could be run cleanly after computing one
more 40 x 40 Gram on Modal. Neither has been done; it is recorded on
[[open-questions]] and on [[factors-versus-pca-coverage]] as open.

A second caveat for whoever runs it: `analyse_fa_qwen35.py` currently asserts
that the input Gram has 100 or 134 traits and that exactly 100 of them are
Goldberg markers, so a 40-trait input needs that assertion widened first.

Related: [[cross-seed-geometry]], [[seed-floor]], [[factor-analysis]],
[[factors-versus-pca-coverage]], [[factor-analysis-null-arms]],
[[steering-results]].
