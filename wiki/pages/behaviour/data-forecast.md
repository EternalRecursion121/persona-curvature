---
title: Forecasting trained behaviour from the data's first-order score
summary: Across the 100 judged zoo datasets, a dataset's directional-derivative score along a Big Five axis (computed before training, with the dataset's own adapter left out of the direction) predicts the judged shift of the adapter trained on it at r 0.61 to 0.85, beating the keying label and predicting which markers of a factor move most (within-factor r 0.77 to 0.95).
status: current
sources:
  - qwen35/analyse_data_forecast.py
  - qwen35/analysis/data_forecast.json
  - qwen35/analysis/nxn_scores.json
  - qwen35/phase10_runs/judged_100.json
  - qwen35/phase10_runs/steer_spec.json
  - qwen35/phase10_runs/steer_spec2_7a.json
  - qwen35/results/gram_sweep.npz
  - qwen35/analysis/syc_forecast.json
last_verified: 2026-09-10
tags: [behaviour, scoring, forecasting, use]
---

This is the applied question behind the scoring identity: can you tell what a dataset will do to a model's dispositions before training on it? The zoo gives 134 natural test cases, because every trait's preference data became one adapter that was later judged blind on the Big Five ([[judged-evaluations]]; 100 of the 134 have judged profiles).

## Method

The 134 x 134 scoring run ([[n-by-n-scoring]], `analysis/nxn_scores.json`) holds, for 40 preference pairs of every trait, the directional derivative of the DPO loss along each of the 134 unit adapter directions. The derivative is linear in the direction, so a dataset's score along any merge direction sum_i c_i a_i is sum_i c_i |a_i| times its score along the unit a_i. That gives each dataset's first-order push along the five Big Five keying axes (`steer_spec.json`, axis_*) and the five factors (`steer_spec2_7a.json`, FA_*) with no new compute. Leave-one-out: when dataset t is scored along a direction, its own adapter's coefficient is set to zero, so a dataset is never scored against the adapter trained on it. Behaviour: the trained stage-one adapter's judged mean minus the base model's, per dimension, over 24 prompts (`judged_100.json`, conditions stage1 and base). Script `analyse_data_forecast.py`; output `analysis/data_forecast.json`.

## Result

Pearson correlation across the 100 judged datasets between the dataset's pre-training score along the axis and the trained adapter's judged shift on that dimension (`directions.<dim>.axis_axis_<dim>_loo`):

| dimension | keying label alone | axis score, leave-one-out (perm p) | partial on the label | within the factor's own markers | factor direction, leave-one-out | axis, own adapter included |
|---|---|---|---|---|---|---|
| Extraversion | +0.640 | +0.753 (p 0.0002) | +0.605 | +0.905 (n 20) | +0.680 | +0.767 |
| Agreeableness | +0.729 | +0.855 (p 0.0002) | +0.703 | +0.941 (n 20) | +0.851 | +0.858 |
| Conscientiousness | +0.623 | +0.735 (p 0.0002) | +0.600 | +0.952 (n 20) | +0.696 | +0.750 |
| EmotionalStability | +0.440 | +0.612 (p 0.0002) | +0.503 | +0.768 (n 20) | +0.344 | +0.621 |
| Intellect | +0.634 | +0.737 (p 0.0002) | +0.507 | +0.952 (n 20) | +0.399 | +0.743 |

Specificity (`axis_score_vs_shift_matrix_rows_axis_cols_dim`), score along the axis in the row against judged shift on the dimension in the column:

| axis \ judged | Extra | Agree | Consc | Emoti | Intel |
|---|---|---|---|---|---|
| Extra | +0.75 | -0.28 | -0.12 | -0.14 | +0.05 |
| Agree | +0.08 | +0.85 | -0.56 | -0.27 | -0.29 |
| Consc | -0.46 | -0.31 | +0.73 | +0.49 | +0.30 |
| Emoti | -0.42 | +0.14 | +0.36 | +0.61 | -0.05 |
| Intel | -0.12 | +0.12 | +0.21 | +0.03 | +0.74 |

## Reading

- The first-order score forecasts trained behaviour better than the ground-truth keying label does on every dimension (0.61 to 0.85 against 0.44 to 0.73; the Agreeableness value is 0.8548, which the table above rounds to 0.855), and the two are not the same information: partialling the label out leaves r 0.50 to 0.70.
- Inside a factor, where every dataset carries the same label up to sign, the score predicts which of the twenty markers produced the largest behavioural shift at r 0.77 to 0.95. No label can do that; this is the part that is a forecast rather than a lookup.
- The matrix is diagonal-dominant. The off-diagonal structure that exists is the competence bundle already seen in steering: the Conscientiousness axis also forecasts Emotional Stability (+0.49) and the Agreeableness axis forecasts a Conscientiousness drop (-0.56).
- The keying axes forecast better than the oblique factors on four of five dimensions, most clearly on Intellect (0.74 against 0.40), where the Imagination factor is loaded on markers the judge scores as Conscientiousness.

## What this does and does not establish

It establishes that, for this model and recipe, the directional derivative of a dataset along a disposition direction, taken before any training, predicts the judged disposition of the model trained on it, and that the prediction carries information the label does not. It is a forecast of a 13-step AdamW run in the short-run regime where first-order proxies are expected to work; it has not been tested on long fine-tunes. The directions are the zoo's own adapters, so the forecast uses 133 reference datasets that share the recipe; on a foreign dataset the directions still apply but the calibration might not. The 34 lexicon datasets have no judged profile and are not in this test. The judge is one model on 24 prompts. The practical use is [[dolci-data-audit]]: scoring a large instruction or preference dataset along the same directions before deciding what to train on.

## Tested on a foreign corpus (added 2026-09-10)

[[dolci-flag-training]] is the first out-of-zoo test of this page's claim. Five
DPO arms were trained on Dolci-Instruct-DPO subsets chosen by their first-order
score, and the score predicted the trained adapter's position on the direction
that was pre-registered and on a second, post-hoc one. Pre-registered: cosine
with the `corrigible` alignment adapter orders the arms
-0.030692989407676445 / +0.005813486115124203 / +0.04699894155404723 exactly as
their corpora's `align_corrigible` scores order them (-0.127223 / +0.012928 /
+0.169949). Post hoc, formulated after the cross-Gram was read: the factor
chart's Warmth coordinate orders them -0.0177 / +0.0275 / +0.0717 as their
`axis_Agreeableness` scores do (-0.077741 / +0.013758 / +0.158670)
(`analysis/dolci_flag_training.json#prereg` and `#post_hoc`). Three points and an
ordering, so what carries over is the sign, not the calibration.

What did **not** carry over is the behavioural half. The judged Big Five battery
separates none of the arms from the matched random one, and the compliance
battery separates them with the opposite sign to the one the data's item-level
reading implied. The reason is scale: these adapters project 0.0204 to 0.0634
onto the factor chart where a stage-one trait adapter projects 0.58, so the
24-prompt battery has no power on them. A forecast of *where the weights go* is
not by itself a forecast of what the model will do.

## Tested on the disposition preference data actually rewards (added 2026-09-10)

[[sycophancy-forecast]] is the second out-of-zoo test and the first on a
safety-relevant disposition. Six DPO arms spanning `align_sycophantic` on
Dolci-Instruct-DPO, pre-registered, plus a purpose-written three-part sycophancy
battery.

**Two forecasts held, at an exact p on six points.** Cosine with the
`sycophantic` alignment adapter orders the six arms as their corpora's
`align_sycophantic` scores do (Spearman **+0.9429**, exact two-sided p
**0.0167**), and the factor chart's Warmth coordinate orders them as their
`FA_Warmth` scores do (the same **+0.9429**, p **0.0167**). In both, the single
adjacent transposition is the arm pair the pre-registration named as effectively
tied, 0.000887 apart
(`analysis/syc_forecast.json#weight_space.cos_sycophantic_spearman_vs_forecast`,
`#...chart_warmth_spearman_vs_FA_Warmth_score`).

**The behavioural half carried over this time — for disposition.** The 24-prompt
Big Five battery, which [[dolci-flag-training]] found had no power on 400-pair
arms, orders the six arms' judged Agreeableness as their `axis_Agreeableness`
scores do at Spearman **+0.8407**, exact p **0.0444**
(`#bigfive.agreeableness_spearman_vs_axis_score`). The anti-sycophantic arm is
the only one of six that does not become more agreeable than base
(-0.0833, p 0.72836) while the other five move +0.2917 to +0.8333. That earlier
null was power, not absence: the score spread here is 0.536 against 0.236 there.
This page's Agreeableness figure of r 0.855 was measured on the zoo's own 100
datasets; the ordering now survives on a corpus the project did not build.

**What did not carry over is the target behaviour itself.** The run's own
pre-registered test — a composite of three sycophancy measures — fails
(Spearman **-0.4857**, p 0.3556), and on pushback the sign reverses: the arm
trained on the most sycophantic-scoring pairs never capitulates and the
anti-sycophantic arm capitulates half the time (0.0000 against 0.5000,
p 0.0024). The reading that reconciles the two halves is that
`align_sycophantic` is a **warmth** direction — its trained arm's nearest zoo
trait is `warm` at +0.0675 — so what the score forecasts is the register a
dataset will induce, not the epistemic deference that makes sycophancy a safety
problem.
