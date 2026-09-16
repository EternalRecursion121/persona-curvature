---
title: Fisher norms of the steering directions
summary: Every steering direction in this project is a unit vector in the weight-space Frobenius metric, which the model has no opinion about; measuring the curvature of KL(base || steered) in alpha at the base point puts 124 directions on a scale the model does care about, spreading them over 260x, and shows that the stage-two grand mean is 6.7x flatter than the stage-one one, that F predicts how far a direction moves the judged persona over the 72 factor-sphere points (rho +0.37, but +0.02 on the principal-component sphere - see the addendum) but not whether it degenerates, and that alpha 4 on the stage-two mean is the same dose as alpha 1.54 on the stage-one mean.
status: current
sources:
  - qwen35/fisher.py
  - qwen35/build_fisher_spec.py
  - qwen35/analyse_fisher.py
  - qwen35/analysis/fisher_norms.json
  - qwen35/phase10_runs/fisher_results.json
  - qwen35/phase10_runs/fisher_spec.json
  - qwen35/phase10_runs/fisher.log
  - qwen35/analysis/steer_alpha_units.json
  - qwen35/steer_fix.py
  - qwen35/phase10_runs/steer_results_fix.json
  - qwen35/phase10_runs/steer_spec.json
  - qwen35/phase10_runs/steer_spec2_7a.json
  - qwen35/phase10_runs/alien_spec_fa.json
  - qwen35/phase10_runs/steer_spec_s2mean.json
  - qwen35/phase10_runs/steer_spec_s2balanced.json
  - qwen35/phase10_runs/sphere_sweep_spec_fa.json
  - qwen35/analysis/sphere_page_fa.json
  - qwen35/results/fa_qwen35.json#solutions.centred_k5.ss_loadings.oblimin
  - qwen35/analysis/alien_steer_fa.json
  - qwen35/zoo40_meter.sh
  - qwen35/analysis/fisher_gram_validation.json
  - qwen35/analysis/matched_dose_alphas.json
  - qwen35/analysis/sphere_isokl.json
  - qwen35/analysis/sphere_isokl_alphas.json
last_verified: 2026-09-10
tags: [behaviour, steering, geometry, units, curvature]
---

# Fisher norms of the steering directions

Every steering direction in this project is a unit vector in one particular
metric: the Frobenius norm of the weight change. A direction is a coefficient
vector over adapters, turned into a single weight change

    dW(u) = sum_i c_i * (lora_alpha / r) * B_i A_i,

and then divided by its own Frobenius norm so that `||dW||_F = 1`
(`qwen35/steer_fix.py`; the same code path is reproduced in `qwen35/fisher.py`).
Steering is `W_base + alpha * ref * dW(u)`, and the published alphas are all in
that unit.

The Frobenius metric is a property of the parameterisation, not of the model.
Two directions of equal Frobenius length can move the model's output
distribution by wildly different amounts. The metric the model itself implies on
its parameters is the Fisher information, and along the one-parameter family
`theta(alpha) = theta_0 + alpha * ref * u`,

    KL( p_base || p_theta(alpha) )  =  0.5 * alpha^2 * F(u) + O(alpha^3),
    F(u) = ref^2 * u^T I(theta_0) u.

This page measures F(u) directly, by running the model, for 124 directions
(`qwen35/analysis/fisher_norms.json#n_directions`). Units are nats per token per
unit alpha squared, with alpha in the published ref unit
`ref = 0.8078003190997738` (`#ref`).

## What "per adapter-norm alpha" means

The 2026-09-08 audit found that `ref` is 0.49995636486079753 of a stage-one
adapter's own Frobenius norm, so alpha 1 in every published run is half an
adapter (`qwen35/analysis/steer_alpha_units.json#ref_over_true_mean_norm`; see
[[stage-two-shared-direction]]). F is
reported both ways in the file: `F_ref` per ref-unit alpha and `F_adapter` per
adapter-norm alpha, with `F_adapter = F_ref / 0.49995636486079753^2`, that is
4.0003 times larger.

## How it was measured

Fixed text, no generation. The token sequences are the 24 steering prompts with
the alpha-0 responses stored under `PC1` in
`qwen35/phase10_runs/steer_results_fix.json`, capped at 192 response tokens,
giving 4378 scored positions
(`qwen35/analysis/fisher_norms.json#n_scored_tokens`) identical for every
direction. For each direction and each alpha the model is run once and the full
per-token log-probabilities over the response positions are taken; the reported
KL is the mean per-token `KL(p_base || p_steered)`. Symmetric KL and the rate at
which the greedy token changes are recorded as secondary
(`#directions.<name>.symkl_mean`, `#directions.<name>.argmax_change`).

Alphas are -0.5, -0.25, -0.125, +0.125, +0.25, +0.5 for the 32 named directions
and -0.5, -0.25, +0.25, +0.5 for the 20 random merges and the 72 sphere points
(`qwen35/build_fisher_spec.py`).

Two things about the arithmetic matter and are recorded here because they bear
on the published steering runs as well.

**The weights are held in fp32.** A unit-Frobenius direction spread over the
3,569,090,560 LoRA-targeted parameters
(`qwen35/analysis/fisher_norms.json#targeted_params`, over 248 modules) has an
RMS element of about 1.7e-5, so at alpha 0.125 the intended increment per weight
is around 1.7e-6, while one bf16 ulp at a typical weight is around 1e-4.
Round-to-nearest discards nearly all of it. `qwen35/steer_fix.py` adds the
increment straight into bf16 parameters; this run holds the parameters in fp32,
keeps an exact bf16 copy of the base on the host, and sets each alpha as
`W = W_base + k * D` from that copy rather than by accumulating increments.
TF32 is off for both matmul and cuDNN, because TF32 rounds GEMM inputs to ten
mantissa bits and would erase the increment the same way bf16 does.

**The published alpha-0 generations are not the base model.** The seven-alpha
walk in `qwen35/steer_fix.py` reaches alpha 0 by adding and subtracting bf16
increments, and that is not reversible. The nine `"0.0"` entries in
`qwen35/phase10_runs/steer_results_fix.json` are all different from one another:
no prompt out of 24 gives an identical string between `PC1` and any of the
other eight directions, and the mean shared prefix is 2.4 to 7.9 per cent of the
response. They are near-base greedy text, which is all this measurement needs
(the requirement is identical tokens for every direction), but they should not
be described as the base model's own output. This is a new observation and is
recorded in [[superseded-claims]].

The control that says the pipeline is exact: with the direction applied at alpha
0, the measured KL, symmetric KL and argmax change rate are all exactly zero
(`qwen35/analysis/fisher_norms.json#zero_alpha_control`).

## Is the KL quadratic?

Yes, over the range measured. For the 32 directions that carry alphas at both
|0.5| and |0.125|, F read off |0.5| divided by F read off |0.125| has median
1.0236461591291601, minimum 0.9964525650658403 and maximum 1.2016920971871439
(`#quadratic_check`). Fitting `KL = 0.5*F*a^2 + c*a^3 + d*a^4` leaves a relative
residual with median 0.005502761642453857 and maximum 0.036953737751237246
(`#residual.taylor_median_rel_rms`, `#residual.taylor_max_rel_rms`).

The task's prescribed fit, the least-squares slope of KL against `0.5*alpha^2`
pooled over the alphas, weights each alpha by `alpha^4` and therefore reads the
curvature near |alpha| = 0.5 rather than at zero; it comes out a median
1.032863386457391 times the Taylor coefficient, at most 1.1920544321522168
(`#pooled_over_taylor`). Both are in the file per direction as `F_ref_pooled`
and `F_ref`; the tables below use `F_ref`, the curvature at zero. The pure
quadratic fit's own relative residual is median 0.07823694957362476
(`#residual.pure_quadratic_median_rel_rms`), which is the same fact seen from
the other side.

The 32 named directions carry alphas down to 0.125 and the 92 sphere and random
directions only to 0.25, so the ranking could in principle be an artefact of
extrapolating to zero from different lever arms. It is not: F ranked against the
plain `2*KL/alpha^2` at |alpha| = 0.25, which every direction has, gives
Spearman 0.9992446892210858 over all 124 and exactly 1.0 over the 32 that also
have |alpha| = 0.125 (`#comparisons.fit_definition_agreement`).

## The random band

Twenty seeded Gaussian merges of all 134 stage-one adapters (seeds 41000 to
41019, `qwen35/build_fisher_spec.py`) give the null: median F 0.12382989334918551,
range 0.05858141121226077 to 0.2779068710126155, tenth percentile
0.08428031881777351, ninetieth 0.21859755244486676, mean 0.1420451410297427,
standard deviation 0.06001359953132766 (`#random_band`). Sixty-eight of the 124
directions sit above the ninetieth percentile of that band and six below the
tenth (`#comparisons.n_above_random_p90`, `#comparisons.n_below_random_p10`).

## F for the published directions

`F/rand` is F over the random-merge median; `rank` is out of 124; `dose` is
`sqrt(F/F_random_median)`, the factor by which one unit of alpha on this
direction moves the output distribution relative to one unit on a random merge;
`alpha 2` is what the published alpha 2 amounts to in those Fisher units.
All from `qwen35/analysis/fisher_norms.json#directions` and
`#fisher_dose_table`.

| direction | F (ref units) | F (adapter units) | F/rand | rank | dose | alpha 2 |
|---|---|---|---|---|---|---|
| FA_Warmth | 0.3650 | 1.4604 | 2.95 | 22 | 1.717 | 3.43 |
| FA_Competence | 0.2133 | 0.8533 | 1.72 | 73 | 1.312 | 2.62 |
| FA_FearfulWithdrawal | 0.1750 | 0.7000 | 1.41 | 95 | 1.189 | 2.38 |
| FA_Arousal | 0.1820 | 0.7280 | 1.47 | 90 | 1.212 | 2.42 |
| FA_Imagination | 0.2353 | 0.9413 | 1.90 | 55 | 1.378 | 2.76 |
| PC1 | 0.2036 | 0.8146 | 1.64 | 82 | 1.282 | 2.56 |
| PC2 | 0.2420 | 0.9682 | 1.95 | 52 | 1.398 | 2.80 |
| PC3 | 0.3002 | 1.2011 | 2.42 | 32 | 1.557 | 3.11 |
| PC4 | 0.3846 | 1.5385 | 3.11 | 17 | 1.762 | 3.52 |
| PC5 | 0.2285 | 0.9143 | 1.85 | 60 | 1.358 | 2.72 |
| PC6 | 0.2072 | 0.8289 | 1.67 | 78 | 1.294 | 2.59 |
| axis_Extraversion | 0.1816 | 0.7265 | 1.47 | 91 | 1.211 | 2.42 |
| axis_Agreeableness | 0.3706 | 1.4825 | 2.99 | 19 | 1.730 | 3.46 |
| axis_Conscientiousness | 0.2007 | 0.8030 | 1.62 | 83 | 1.273 | 2.55 |
| axis_EmotionalStability | 0.2271 | 0.9086 | 1.83 | 61 | 1.354 | 2.71 |
| axis_Intellect | 0.2512 | 1.0051 | 2.03 | 46 | 1.424 | 2.85 |
| mean_assistant_axis | 0.2947 | 1.1792 | 2.38 | 33 | 1.543 | 3.09 |
| alien_fa | 0.2593 | 1.0374 | 2.09 | 44 | 1.447 | 2.89 |
| alien_fa_shuffle | 0.0720 | 0.2880 | 0.58 | 120 | 0.762 | 1.52 |
| span_random_fa | 0.2485 | 0.9942 | 2.01 | 47 | 1.417 | 2.83 |
| S2_mean | 0.0439 | 0.1757 | 0.35 | 123 | 0.596 | 1.19 |
| S2_balancedrandom | 0.0019 | 0.0077 | 0.02 | 124 | 0.124 | 0.25 |
| single_agreeable | 0.2205 | 0.8822 | 1.78 | 65 | 1.334 | 2.67 |
| single_rude | 0.1742 | 0.6969 | 1.41 | 96 | 1.186 | 2.37 |
| single_organized | 0.1021 | 0.4083 | 0.82 | 115 | 0.908 | 1.82 |
| single_careless | 0.1054 | 0.4218 | 0.85 | 112 | 0.923 | 1.85 |
| single_anxious | 0.1163 | 0.4652 | 0.94 | 108 | 0.969 | 1.94 |
| single_relaxed | 0.1265 | 0.5061 | 1.02 | 106 | 1.011 | 2.02 |
| single_extraverted | 0.0606 | 0.2426 | 0.49 | 121 | 0.700 | 1.40 |
| single_quiet | 0.1467 | 0.5869 | 1.18 | 101 | 1.088 | 2.18 |
| single_imaginative | 0.1032 | 0.4127 | 0.83 | 114 | 0.913 | 1.83 |
| single_unimaginative | 0.1287 | 0.5148 | 1.04 | 105 | 1.019 | 2.04 |

The 72 factor-sphere points run from 0.1716923166559369 to 0.49828560642407
with median 0.2608319173143098 (`#comparisons.sphere_band`), and take the top
twelve places overall. That is not a paradox: the sphere points are unit-norm combinations of
three oblique factors, and a normalised combination of non-orthogonal directions
can have a larger Fisher norm than any of the three has alone.

The whole set spans 0.0019170369703626355 (`S2_balancedrandom`) to
0.49828560642407 (`sphere_S061`), a ratio of 259.924881015629
(`#comparisons.full_range`). Two directions that the project has always treated
as the same length in weight space differ by that much in how far they move the
model.

## The five factors as a coordinate

F is a new per-factor number, independent of the loadings
(`#comparisons.factors`):

| factor | F | F/rand | rank of 124 |
|---|---|---|---|
| Warmth | 0.3650 | 2.95 | 22 |
| Competence | 0.2133 | 1.72 | 73 |
| Timidity | 0.1750 | 1.41 | 95 |
| Arousal | 0.1820 | 1.47 | 90 |
| Imagination | 0.2353 | 1.90 | 55 |

Warmth is the steepest of the five, 2.1 times Timidity. The ordering
is not the variance ordering. By sums of squared oblimin loadings the factors run
10.76210106481124, 8.421873312354228, 7.020338408333558, 6.814511494042316,
5.798806928556024 in the order above
(`qwen35/results/fa_qwen35.json#solutions.centred_k5.ss_loadings.oblimin`, see
[[factor-analysis]]), so Imagination is last in variance and second in F, and
Timidity is third in variance and last in F. Only Warmth is first on
both. How much of the adapter cloud a factor explains and how hard it pushes the
model are separate facts about it.

## Single adapters against merges

The ten single trait adapters have median F 0.12140218073805878, range
0.060643422539815736 to 0.220505415541111
(`#comparisons.single_adapters_vs_merges`). That is 0.9803947775010928 of the
random-merge median: a single adapter's direction is, to within measurement,
exactly as steep as a random merge of all 134. The twenty named merges (the five
factors, six PCs, five axes, the grand mean, the alien direction and its two
controls) have median 0.23190363687583387, close to twice both. Structure in the
coefficient vector buys curvature; being a real trained adapter does not.

## Stage two is flatter than stage one

The stage-two grand mean has F 0.04391528597956066 against the stage-one grand
mean's 0.2947447913048136, a ratio of 0.14899427326654663
(`#comparisons.stage2_vs_stage1_grand_mean`). Its balanced control is
0.0019170369703626355, the flattest direction measured.

This is the answer to the open question about coherence. Steering the stage-one
grand mean to alpha 4 loops on 54.2 per cent of the 24 prompts at -4 and 95.8
per cent at +4; the stage-two grand mean loops on 4.2 per cent at -4 and none at
+4 (`#loop_rates_recomputed.mean_assistant_axis`, `#loop_rates_recomputed.S2_mean`,
recomputed from the published generations). Alpha 4 on the stage-two grand mean
is the same Fisher dose as alpha 1.5439910531686205 on the stage-one grand mean
(`#comparisons.stage2_vs_stage1_grand_mean.alpha4_on_S2mean_in_stage1_grandmean_alpha`),
and the stage-one axis does not degenerate at 1.5 either. The stage-two
direction is not more robust; it is smaller in the only units that matter, and
the published alphas made the two look comparable when they never were. See
[[stage-two-shared-direction]].

## Does F predict degeneration?

Not by itself, no.

Loop rates are recomputed here with one detector for all of them,
`analyse_alien_steer.looping` (a 10-word window repeated four times), on the
published generations, so the arms are counted the same way
(`#loop_rates_recomputed`). The human-adjudicated per-alpha counts that exist
for some directions are carried separately in `#loop_rates_adjudicated` and are
not pooled with these.

Over the 22 measured directions that have published generations at alpha +/-2:

| arm | n | Spearman | permutation p |
|---|---|---|---|
| F vs mean loop rate at +/-2 | 22 | +0.1086 | 0.6219 |
| F vs loop rate at -2 | 22 | +0.1501 | 0.4973 |
| F vs loop rate at +2 | 22 | -0.0008 | 0.9982 |
| cubic coefficient c vs loop rate at -2 | 22 | -0.3782 | 0.0817 |

(`#correlation_with_degeneration`). Over the 72 factor-sphere points at alpha
1.5, where 27 of 72 loop on at least one of eight prompts, F against loop rate
gives Spearman -0.1015 at p 0.3957.

There is a structural reason F cannot do this job. Degeneration in this project
is almost entirely a negative-alpha phenomenon: of the 22 directions measured
here that have generations at both signs, 16 loop at all, all 22 loop at least
as much at -2 as at +2 (15 strictly more), 18 have a loop rate of exactly zero
at +2 and only 6 have one at -2
(`#comparisons.sign_asymmetry_of_degeneration`). F is even in alpha by construction, so
no value of it can distinguish the sign that breaks the model from the sign that
does not. The leading term that can is the cubic coefficient c of KL in alpha,
and that is the arm that comes closest: Spearman -0.3782, permutation p
0.0817, in the predicted direction (a negative c means the KL rises faster on
the negative side). It is suggestive and not established, on 22 points, six
correlation arms and no multiplicity correction
(`#correlation_multiplicity_note`).

## What F does predict

How far the persona moves. Across the 72 factor-sphere points, F against the
Euclidean distance of each point's mean five-scale judged profile from the
72-point centroid gives Spearman +0.3706669239179368, permutation p
0.0014499275036248187 (`#correlation_with_degeneration.sphere_judged_profile_distance`,
profiles from `qwen35/analysis/sphere_page_fa.json#judged`). A direction with a
larger Fisher norm moves the judged personality further at a fixed alpha, and
does so without being more likely to break the model. Curvature and damage are
different quantities here, and the sphere is the only place in this project with
enough directions to say so.

The centroid is the mean of the 72 points, not a base-model profile: the sphere
run judged no alpha-0 condition, so this is a spread measure within the sphere
rather than a displacement from the base.

## Fisher-normalised alphas for the published results

The `dose` column above is `sqrt(F(u) / F_random_median)`. To read a published
result in Fisher units, multiply its alpha by that direction's dose; the product
is the alpha on a random 134-adapter merge that would move the output
distribution as far. So the published alpha 2 is a dose of 3.52 on PC4, 3.46 on
the Agreeableness axis and 3.43 on Warmth, but only 1.52 on the alien
direction's shuffled control, 1.19 on the stage-two grand mean and 0.25 on its
balanced control. Comparisons across directions at a common alpha, which is what
[[steering-results]], [[alien-direction-factor-chart]] and
[[sphere-sweep-factor-chart]] all do, are comparisons at doses that differ by up
to 14x within the published set.

Note also that `ref` is not the same in every spec, contrary to the comment in
`qwen35/steer_fix.py`: `phase10_runs/steer_spec.json`,
`phase10_runs/alien_spec_fa.json` and `phase10_runs/sphere_sweep_spec_fa.json`
carry 0.8102592902648793, while `phase10_runs/steer_spec2_7a.json`,
`steer_spec_s2mean.json` and `steer_spec_s2balanced.json` carry
0.8078003190997738. The difference is 0.3 per cent and every direction here was
measured at the second value; each direction's own published ref is recorded as
`#directions.<name>.published_ref`.

## What F does and does not mean

F is a **second-order** quantity, **at the base point**, on **in-distribution
text**.

- Second order: it is the coefficient of alpha^2 in the KL, fitted from alphas
  no larger than 0.5. It says how fast the output distribution starts to move.
  It says nothing about alpha 2 or alpha 4, where the published results live,
  except by extrapolation that the data here cannot check.
- At the base point: it is the curvature of the KL surface at theta_0. The
  Fisher metric varies over the parameter space, and nothing here measures it
  anywhere but at the base model.
- On in-distribution text: the tokens scored are the base model's own greedy
  responses to 24 steering prompts. F is the rate at which the model's
  probabilities on text it would itself have written begin to change. A
  direction could be flat here and steep on text the base model would never
  produce.
- It is even in alpha, so it cannot express the strong sign asymmetry that
  dominates the degeneration data.
- It is a single scalar summarising the whole output distribution. It does not
  say which way the persona moves, only how fast.

## Two later runs built on this page

**The whole 134 x 134 Gram in this metric.** [[factor-analysis-fisher-metric]]
extends F from 124 named directions to a Fisher inner product between every pair
of stage-one adapters, on the same 4,378 tokens. It reproduces the F measured
here for **122** of the directions in `phase10_runs/fisher_spec.json` at a median
ratio of **0.9943482311581795** and a worst relative error of
**0.07919140120849233**
(`qwen35/analysis/fisher_gram_validation.json#prediction_of_fisher_py.expected`),
including all 20 random merges and all 72 sphere points - an independent check
of this page's numbers on a different code path. It also gives the per-adapter
Fisher norm for all 134: **0.06068799875816181** (`extraverted`) to
**0.28769592173625586** (`pleasant`), median **0.12508604558105607**, and that
median agrees with the single-adapter median of 0.12140218073805878 recorded
above over ten. Curvature does not follow size: F_ii against the adapter's
Frobenius norm is Pearson **-0.2234650127648705**.

**The dose table, measured rather than extrapolated.**
[[matched-dose-steering]] measures KL directly at twelve alphas in
|alpha| in [1.0, 2.5] for ten of these directions, and two things follow that
this page could not have seen. First, **the Taylor fit here must not be
extrapolated**: carried out to alpha 2 it overstates the measured KL for all ten
directions it was tested on, by a median factor of **1.4751663093238792** and up
to **5.7846156073675274**
(`qwen35/analysis/matched_dose_steering.json#taylor_extrapolation_check`). For
FA_Warmth it predicts 1.5892934857465926 against a measured 0.6470294541123253.
Second, `steer_fix.py`'s **bf16** weights keep the perturbation's Frobenius
length (retention median 1.0027071769953744) but not its effect: `KL(bf16) /
KL(exact)` at the same alpha has median **0.9059646365364478** overall and
**0.9423571405594928** at |alpha| 2, falling to 0.7510758772738881 at |alpha| 1
(`#bf16_vs_exact`). Every published steering alpha was therefore a somewhat
smaller dose than intended, by an amount that depends on the alpha. That page also finds that this page's "degeneration is
overwhelmingly a negative-alpha phenomenon" does not hold once the dose rather
than the alpha is held fixed.

## Cost and provenance

`zoo-fisher.service`, one A100-40GB. Two Modal apps: the first ran 19:39:05 to
19:46:39 UTC (7 min 34 s) and was stopped once it was clear an fp32 greedy
generation diagnostic in it would cost more than the question was worth; the
second ran 19:47:52 to 20:54:53 UTC (67 min 1 s) and did all 124 directions,
with 3935.4081044197083 seconds inside the Modal function
(`qwen35/analysis/fisher_norms.json#wall_seconds`). Logs
`qwen35/phase10_runs/fisher.log` and `fisher.log.attempt1-fp32greedy`.

That is 1.2430 GPU-hours, which the spend meter's rate of $2.10 per GPU-hour
values at $2.61 (`qwen35/zoo40_meter.sh`, `RATE`; the meter's own log pooled
this run with a sibling job and cannot be read for one unit alone). The meter's
rate is GPU-only. These containers also reserved memory (52 GiB on the second
run, 56 GiB on the first) and CPU cores (2 and 8), which Modal bills as separate
line items, adding roughly $1.5 at Modal's published rates and taking the true
figure to about $4.2. That estimate has not been checked against an invoice, and
`zoo40_meter.sh` should learn to count reservations before another unit reserves
this much. Samuel authorised $10 for this run on 2026-09-09. See [[costs]].

**A third run built on this page.** [[sphere-sweep-iso-kl]] measures the KL the
72 *principal-component* sphere directions deliver at alpha 1.5 - the sphere
whose F was never measured, since the `sphere_S*` keys here are the factor
sphere - and finds a 2.54x spread. Re-steering that lattice at one measured dose
leaves its smoothness rho at 0.6522913409368338 against 0.6511417860626274, so
the smoothness of both spheres is position rather than dose. Two of this page's
readings do not carry over to that sphere: F against judged displacement, +0.37
here, is +0.016 at p 0.89 there (recorded as S28 in [[source-contradictions]]),
and the null on degeneration becomes a Spearman of -0.5146 at p 0.0016 once the
dose is held fixed and the covariate is measured rather than extrapolated.

Related: [[matched-dose-steering]], [[sphere-sweep-iso-kl]],
[[factor-analysis-fisher-metric]],
[[steering-results]], [[stage-two-shared-direction]],
[[sphere-sweep-factor-chart]], [[alien-direction-factor-chart]],
[[factor-analysis]], [[glossary]].
