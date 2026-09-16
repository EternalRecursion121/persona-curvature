---
title: The factor analysis in the model's own metric
summary: The 134x134 Gram rebuilt as a Fisher inner product - the curvature of the output distribution rather than the Frobenius length of the weight change - and the same factor analysis rerun on it; the five factors survive with Tucker congruence 0.9654 against the Frobenius solution (minimum 0.9549, all five above the 0.95 "identical" threshold), the Big Five congruences move by at most 0.0589, parallel analysis still retains nine, and the factor structure is therefore a property of the adapters and not of the parameterisation.
status: current
sources:
  - qwen35/fisher_gram.py
  - qwen35/analyse_fisher_gram.py
  - qwen35/analyse_fa_fisher.py
  - qwen35/analyse_fa_qwen35.py
  - qwen35/analysis/fisher_gram_validation.json
  - qwen35/analysis/fa_fisher_metric.json
  - qwen35/results/gram_fisher.npz
  - qwen35/results/gram_fisher_empirical.npz
  - qwen35/results/fa_qwen35_fisher.json
  - qwen35/results/fa_qwen35_fisher_emp.json
  - qwen35/results/fa_qwen35.json
  - qwen35/results/gram_sweep.npz
  - qwen35/phase10_runs/fisher_gram_results.json
  - qwen35/phase10_runs/fisher_gram_smoke.json
  - qwen35/phase10_runs/fisher_spec.json
  - qwen35/phase10_runs/fishergram.log
  - qwen35/phase10_runs/fa_fisher.log
  - qwen35/analysis/fisher_norms.json
last_verified: 2026-09-10
tags: [geometry, factor-analysis, curvature, units, big-five]
---

# The factor analysis in the model's own metric

Every Gram in this project is an inner product in the **Frobenius** metric of
weight space: two adapters are close if their weight changes overlap, and a
direction has unit length if its weight change has unit Frobenius norm. The
model has no opinion about that metric. [[fisher-norms]] measured what it costs
- the Fisher norms of 124 steering directions span a factor of
**259.924881015629** (`qwen35/analysis/fisher_norms.json#comparisons.full_range`),
so two directions this project calls the same length differ by that much in how
far they move the output distribution.

Gradient Atoms (Rosser, 2026) states the same objection as a design principle:
without whitening by the Fisher, "any decomposition is dominated by
high-curvature directions, drowning out semantic structure"
([[paper-reading-2026-09-09]], Section 3.2 of that paper). Experiment G1 of that
reading note is the test. This page is that test.

**The answer is that the factor solution survives the change of metric almost
exactly.** The five factors match column for column at Tucker congruence
**0.9654** (`qwen35/analysis/fa_fisher_metric.json#loading_matrix_congruence_k5.pairs.frobenius_vs_fisher_expected.mean_abs_matched_congruence`),
all five above 0.95, and no Big Five congruence moves by more than 0.0589.

---

## The object that was built

`qwen35/fisher_gram.py` builds a 134 x 134 matrix of Fisher inner products over
the stage-one adapters. For adapter i let

    dW_i = (lora_alpha / r) * B_i A_i,      u_i = dW_i / ||dW_i||_F

and perturb the base weights along `u_i` in the same `ref` units every steering
run in this project uses:

    theta(eps) = theta_0 + eps * ref * u_i,   ref = 0.8078003190997738

The **score** of a token position under that one-parameter family is
`s_i(t, y) = d/d(eps) log p_eps(y | x_<t)` at eps = 0, and the Fisher inner
product of two adapters is the covariance of their scores. Two estimators come
out of the same forward passes:

| | definition | what it is |
|---|---|---|
| **expected** | `F_ij = mean_t sum_y p_0(t,y) s_i(t,y) s_j(t,y)` | the Fisher information itself; for i = j it is the curvature of KL(base \|\| steered) that [[fisher-norms]] fitted |
| **empirical** | `E_ij = mean_t s_i(t,y_t) s_j(t,y_t)` | the outer product of per-token scores at the **realised** tokens, which is the estimator [[paper-reading-2026-09-09]] specifies for G1 |

Both are saved: `qwen35/results/gram_fisher.npz` (expected) and
`qwen35/results/gram_fisher_empirical.npz` (empirical), in the same
`G / names / norms / scale / n_modules` shape `analyse_fa_qwen35.py` reads.

**The tokens are identical to [[fisher-norms]]'s.** The 24 steering prompts of
the Big Five battery with the base model's own alpha-0 greedy responses as
stored under PC1 in `qwen35/phase10_runs/steer_results_fix.json`, capped at 192
response tokens, **4,378** scored positions
(`qwen35/phase10_runs/fisher_gram_results.json#n_scored_tokens`). Nothing is
generated, so every adapter is scored on the same tokens.

**The derivative is a central finite difference** at h = 0.125 - the smallest
alpha `fisher.py` used - with the perturbation applied as a **forward hook**,
`out <- out + (eps * ref * scale / ||dW_i||) * B_i (A_i x)`, rather than by
touching the weights. Three things follow:

- Nothing dense is materialised. `fisher.py` built a 6.6 GiB delta buffer per
  direction; here one direction costs two small products per module, so all 134
  cost what one cost there.
- The base weights are never modified, so the bf16 rounding of the increment
  that `fisher.py`'s docstring documents for `steer_fix.py` cannot arise.
- Both signs share the base model bit for bit, so the fp32 rounding of the
  forward pass is common-mode and cancels in `lp(+h) - lp(-h)`. The forward runs
  in fp32 with TF32 off, as `fisher.py` does.

## Is the Gram right? Four checks

**1. It reproduces every Fisher norm `fisher.py` measured, one direction at a
time, on the same text.** A Gram over unit adapter directions predicts the
Fisher norm of any merge:

    F(c) = (c * n)^T F_unit (c * n) / (c^T G_exact c)

with `n` the adapters' Frobenius norms and `G_exact` the exact Frobenius Gram.
Over the **122** stage-one directions in `qwen35/phase10_runs/fisher_spec.json`
that `fisher.py` fitted an F for - 5 factors, 6 PCs, 6 axes, 3 alien, 10 single
adapters, **20 random merges of all 134**, and **72 sphere points** - the
prediction lands at a median ratio of **0.9943482311581795**, Pearson
**0.9984052983749357**, Spearman **0.9989888342183787**, with a worst relative
error of **0.07919140120849233** on `sphere_S007`
(`qwen35/analysis/fisher_gram_validation.json#prediction_of_fisher_py.expected`).
By family the median ratio is fa 0.9984, pc 0.9972, axis 0.9974, alien 0.9991,
single 0.9992, random 0.9989, sphere 0.9857
(`#prediction_of_fisher_py.by_family`). The random merges are the sharp test,
because they load the off-diagonal in a way the singles cannot.

The same file checks the direction norms: `sqrt(c^T G_exact c)` against
`fisher.py`'s own `dir_norm_raw` agrees to **0.00010579119726061847**
(`#prediction_of_fisher_py.direction_norm_check.max_abs_rel_diff`).

**2. The diagonal matches for the ten single adapters.** `F_ii` against
`fisher.py`'s `F_ref`: agreeable 0.9853, rude 0.9919, organized 0.9979,
careless 0.9998, anxious 0.9993, relaxed 0.9999, extraverted 1.0007,
quiet 0.9942, imaginative 1.0039, unimaginative 0.9990
(`#single_adapter_diagonal.per_trait`).

**3. The finite difference has converged.** Redoing those ten at h/2 changes
`F_ii` by at most **0.0025207811194865037**
(`#h_convergence.max_abs_rel_change`).

**4. The adapters used are the published ones.** `||dW_i||_F` recomputed inside
the run from the bf16 factors agrees with `results/gram_sweep.npz` norms to
**5.749e-05** (`#adapter_norm_check.max_abs_rel_diff`).

One diagnostic is *not* small and should be read correctly. The identity
`sum_y p_0(y) s(y) = 0` holds exactly in the limit; at finite h the central
difference leaves a per-token O(h^2) residual, and the worst one over
(direction, token) is **0.3760629892349243** against a score scale
`sqrt(F_ii)` whose minimum is 0.2463
(`#numerics.zero_sum_check_max_abs`). That is a statement about individual
tokens with large third derivatives, not about the aggregate: check 3 shows
`F_ii` itself is converged to 0.25%.

## The two metrics, side by side

The Fisher metric makes the adapter cloud look **much more aligned and much more
spread**. Over all 8,911 adapter pairs
(`qwen35/analysis/fisher_gram_validation.json#metric_comparison`):

| Gram | mean cosine | sd | min | max |
|---|---|---|---|---|
| exact Frobenius | 0.0720834758642461 | 0.16265654340966743 | -0.39699725405025715 | 0.6222239900523809 |
| Fisher, expected | 0.17323729318211056 | 0.35627806940883405 | -0.8401159759722976 | 0.9578689826739857 |
| Fisher, empirical | 0.16241589351675106 | 0.3774792137462528 | -0.8676462494018674 | 0.9674944099847822 |

The two metrics nevertheless order the pairs the same way: the Fisher cosines
correlate with the exact cosines at Pearson **0.9127446990060482** and Spearman
**0.9193166644481833** (`#metric_comparison`). The empirical form sits at
Pearson 0.9144260711261981 against the exact Gram and **0.9907989106434285**
against the expected Fisher Gram - the two Fisher estimators are nearly the same
matrix up to scale.

### The per-adapter Fisher norm

`F_ii` - the curvature of a single trait adapter's own unit direction - runs
from **0.06068799875816181** (`extraverted`) to **0.28769592173625586**
(`pleasant`), a ratio of **4.740573550344074**, median
**0.12508604558105607**
(`qwen35/analysis/fa_fisher_metric.json#fisher_norm_vs_structure.F_ii`). The
median agrees with [[fisher-norms]]'s single-adapter median of
0.12140218073805878 over its own ten
(`qwen35/analysis/fisher_norms.json#comparisons.single_adapters_vs_merges.single_median`).

Curvature is **not** bought by size. `F_ii` against the adapter's Frobenius
norm is Pearson **-0.2234650127648705**, Spearman **-0.2025835733612639**
(`#fisher_norm_vs_structure.vs_frobenius_norm`) - if anything a bigger adapter is
slightly flatter. Against the Frobenius factor solution it is near zero
(communality Pearson -0.09053527573425738, max-absolute-loading Spearman
-0.006536241100263096); against the *Fisher* factor solution it is a weak
positive (communality Pearson 0.2950091981402161, max-absolute-loading Spearman
0.3055175251561741). So an adapter's steepness is close to independent of where
it sits in the Frobenius chart, which is another way of saying the chart is not
measuring curvature.

## Do the factors survive?

`qwen35/analyse_fa_qwen35.py` was run three times, unchanged, on the three Grams
(`PC_GRAM_NPZ` / `PC_FA_TAG`, log `qwen35/phase10_runs/fa_fisher.log`).

### Retained factor count

| arm | parallel analysis, N = 1528 | Kaiser (uncentred) | reduced eig > 1 |
|---|---|---|---|
| Frobenius | 9 | 16 | 9 |
| Fisher, expected | 9 | 11 | 11 |
| Fisher, empirical | 8 | 11 | 11 |

(`qwen35/analysis/fa_fisher_metric.json#arms.*.n_factors`.) The whole N grid is
in the same file: the expected-Fisher arm reads 7, 8, 8, 9, 10, 11 at
N = 150, 300, 1000, 1528, 5809, 20000 against the Frobenius arm's
5, 6, 8, 9, 12, 14. **The Fisher count is far less sensitive to N** - a spread
of 4 across the grid against 9 - which is worth noting because the reference
N = 1528 is a weight-space effective-dimensionality estimate carried over from
sweep100 and has no principled reading for a Gram estimated from 4,378 tokens.
The retained count is the weakest of the three tests here; the congruences and
the loading-matrix match below are the strong ones.

### The Big Five congruences at k = 5

Each factor's best Goldberg target and its congruence, with the one-to-one
matching (all five targets taken exactly once in every arm,
`#arms.*.k5.one_to_one_all_five_distinct`):

| target | Frobenius | Fisher, expected | Fisher, empirical |
|---|---|---|---|
| Agreeableness | 0.6555 | 0.5966 | 0.5977 |
| Conscientiousness | 0.5744 | 0.5913 | 0.5876 |
| Emotional Stability | 0.4049 | 0.4106 | 0.4810 |
| Extraversion | 0.5389 | 0.5389 | 0.5354 |
| Intellect | 0.6823 | 0.6876 | 0.6895 |
| **mean of the best** | 0.5712050407456282 | 0.5649818786881674 | 0.5782623808662789 |

(`#arms.*.k5.one_to_one_matching` and `#arms.*.k5.mean_abs_best_congruence`.)
The largest single move is Agreeableness, down 0.0589. Three of five move up.
**No congruence clears 0.85 in any metric**, so the ceiling
[[factor-analysis]] reports is not a Frobenius artefact - the answer to "was the
metric costing us the Big Five mapping?" is no.

### Are they the same five factors?

Tucker congruence between the two 134 x 5 oblimin loading matrices, best
one-to-one matching of columns
(`#loading_matrix_congruence_k5.pairs`):

| pair | mean \|congruence\| | min | above 0.95 |
|---|---|---|---|
| Frobenius vs Fisher expected | 0.965359016717984 | 0.954900651340028 | 5 of 5 |
| Frobenius vs Fisher empirical | 0.9577766396647475 | 0.9305370993593903 | 3 of 5 |
| Fisher expected vs Fisher empirical | 0.9903268553931749 | 0.9820965878864523 | 5 of 5 |

On the conventional reading, congruence at or above 0.95 is "identical" and
0.85 to 0.94 is "fair similarity". Every one of the five Frobenius factors finds
its expected-Fisher partner at or above 0.9549. Per column the matches are
Warmth 0.9613, Competence 0.9794, Timidity 0.9549, Arousal 0.9581,
Imagination 0.9731.

**What does change is the ordering.** By sum of squared oblimin loadings the
Frobenius solution runs 10.7621, 8.4219, 7.0203, 6.8145, 5.7988 (Warmth,
Competence, Timidity, Arousal, Imagination) and the expected-Fisher
solution runs 32.6709, 18.5031, 15.1420, 14.7448, 13.9623 with the third and
fifth **swapped**: Imagination rises to third and Timidity falls to
fifth (`#arms.*.k5.ss_loadings_oblimin`, and the matching pairs Frobenius
column 2 to Fisher column 4 and Frobenius column 4 to Fisher column 2). That is
the same fact [[fisher-norms]] recorded from the other side - "Imagination is
last in variance by sums of squared oblimin loadings and second in F".

The Fisher factors' top-loading traits confirm the identification by eye
(`qwen35/results/fa_qwen35_fisher.json#solutions.centred_k5`): factor 0 is
Agreeable +0.96, Uncooperative -0.91, Pleasant +0.91; factor 1 Conscientious
+0.90, Negligent -0.88, Casual -0.88; factor 2 Unimaginative -0.83,
Unsophisticated -0.83, Uncreative -0.80; factor 3 Unexcitable -0.86, Reserved
-0.84, Withdrawn -0.82; factor 4 Fearful -0.86, Guilty -0.84, Timid -0.80.

At the nine-factor solutions the agreement is weaker but still substantial:
mean 0.8707522737576175, minimum 0.7270729963609517 over all nine matched pairs
(`#loading_matrix_congruence_at_parallel_analysis_k.pairs.frobenius_k9_vs_fisher_expected_k9`).
The five-factor solution is the one that transports.

### What the Fisher metric does change

Mean communality at k = 5 rises from **0.3168918796188912** to
**0.7485344075570156** (`#arms.*.k5.communality_mean`), and the first five
kernel-PCA components go from 12.495, 10.985, 4.968, 3.626, 2.572 per cent of
variance to 34.423, 18.433, 12.237, 7.706, 4.627
(`#arms.*.pca_centered_var_pct_top6`). Five factors explain a far larger share
of a Fisher Gram than of a Frobenius one, because the Fisher Gram is much more
strongly correlated to begin with. No Heywood cases in any arm.

## The empirical Fisher is half the Fisher

The estimator [[paper-reading-2026-09-09]] specifies for G1 - the score at the
*realised* token rather than the expectation over the vocabulary - is a
different number, and the reading note asked for the size of the gap. It is:

- **0.5281** median over the 122 directions
  (`qwen35/analysis/fisher_gram_validation.json#prediction_of_fisher_py.empirical.median_ratio`),
  range 0.4411 to 0.6583.
- On the diagonal, `F_empirical_ii / F_expected_ii` runs 0.4608 to 0.6418,
  median 0.5317, standard deviation 0.0384 over the 134 adapters
  (computed from `#fisher_norm_per_adapter.per_trait`).

The band is tight, which points at a specific cause rather than an
adapter-specific one. The fixed text is the base model's **own greedy
generation**, so the realised token at every scored position is the argmax -
the position where the model is most confident and where the score function is
systematically small. The empirical Fisher on a model's own modal output should
therefore undercount, roughly uniformly, and it does.

It undercounts *without reordering*: the empirical Gram's cosines correlate with
the expected Gram's at Pearson 0.9907989106434285, its five factors match the
expected Fisher's at mean Tucker 0.9903268553931749, and its Big Five
congruences are within 0.07 of them. **For the purposes of this experiment the
two estimators give the same answer**; the expected form is reported as primary
because it is the Fisher and because it is the form that reproduces
`fisher.py`.

## Caveats

- This is the Fisher **on the base model's own greedy text**, not on a corpus
  and not on samples from the model. A direction could be flat here and steep
  elsewhere - the same caveat [[fisher-norms]] carries.
- It is **not EKFAC**. Gradient Atoms whitens with an EKFAC approximation
  factorised per module as a Kronecker product; this is the exact Fisher
  restricted to the 134-dimensional subspace the adapters span, which is a
  different object and, inside that subspace, a stronger one.
- **The null arms were not redone.** [[factor-analysis-null-arms]] runs on two
  separately trained 100-adapter sets on their own Modal volumes; a Fisher Gram
  for either is another GPU run, not a re-analysis of this one, and it was not
  in this task's budget. The shuffled arm retaining zero factors is therefore
  still a Frobenius-metric statement.
- Parallel analysis's reference N = 1528 has no principled meaning for a
  token-estimated Gram; the grid is reported so the sensitivity is visible.

## What this settles

The five factors are a property of the 134 adapters, not of the way this project
chose to measure length in weight space. [[factor-chart]] takes its inner
products from `results/gram_sweep.npz` and its directions from the Frobenius
oblimin loadings; at a column-for-column congruence of 0.9654 the chart would
not move materially if it were rebuilt in the Fisher metric. And the honest half
of the answer: the Big Five congruences do **not** improve, so the 0.41-to-0.68
band in [[post-draft]] is not an underestimate caused by the metric.

## Cost and provenance

`zoo-fishergram.service`, one A100-80GB, app `pc-qwen35-phase10-fishergram`,
23:22:23 to 23:59:14 UTC on 2026-09-09 (36 min 51 s), with
**2183.2 seconds** inside the Modal function
(`qwen35/phase10_runs/fisher_gram_results.json#wall_seconds` reads
2183.216953277588). A smoke run on the ten single adapters
(`zoo-fishergram-smoke.service`, app `pc-qwen35-phase10-fishergram-smoke`,
23:15:55 to 23:22:08, 6 min 13 s) validated the diagonal against
`fisher_norms.json` before the full run was paid for; its output is
`qwen35/phase10_runs/fisher_gram_smoke.json`. Together 0.7178 GPU-hours, which
the spend meter's rate of $2.10 per GPU-hour values at **$1.51**
(`qwen35/zoo40_meter.sh`, `RATE`; that rate is GPU-only and undercounts the 45
GiB and 2 cores these containers reserved, as [[fisher-norms]] records). The two
factor analyses are CPU and free: `zoo-fa-fisher.service`, 23:59:26 to 00:12
UTC, log `qwen35/phase10_runs/fa_fisher.log`. See [[costs]].

Related: [[factor-analysis]], [[fisher-norms]], [[factor-chart]],
[[paper-reading-2026-09-09]], [[matched-dose-steering]], [[pca-and-scree]],
[[factor-analysis-null-arms]], [[geometry-overview]], [[glossary]].
