---
title: Rank sweep
summary: Fifteen zoo traits retrained at LoRA rank 1, 4 and 16 with the input frame made to nest inside the zoo's rank-64 one - the arrangement survives almost intact at rank 1 (15 x 15 cosine matrix Pearson 0.9906 against rank 64, factor-chart coordinates Pearson 0.9967, 15 of 15 identified among the 134) while the behaviour does not survive at all (own-factor amplification -0.41 percent of headroom at rank 1 against +27.08 at rank 64) and neither does the fit (reward margin 0.134-0.432 against 5.072-18.467).
status: current
sources:
  - qwen35/analysis/rank_sweep.json
  - qwen35/analysis/rank_sweep_mechanism.json
  - qwen35/phase2_runs/results_data_rank_sweep.json
  - qwen35/results/rank_sweep_grams.npz
  - qwen35/train_rank_sweep.py
  - qwen35/cross_gram_rank_sweep.py
  - qwen35/analyse_rank_sweep.py
  - qwen35/phase10_runs/ranksweep_a0.log
  - qwen35/phase10_runs/ranksweep_train.log
  - qwen35/analysis/crossseed_arms.json
  - qwen35/analysis/column_space.json
last_verified: 2026-09-10
tags: [geometry, rank, seed, training, literature]
---

# Rank sweep

## The question

The zoo is 134 LoRA adapters at rank 64 ([[stage-one-training-config]]).
SliderSpace's default is a **rank-one** adapter and it claims rank one wins at a
fixed training budget ([[paper-reading-2026-09-09]], Appendix B.2); Persona
Cartography reports trait control surviving rank 1
([[persona-cartography-paper]]). [[column-space-structure]] found that the
single strongest output direction of a trait's adapter is recovered across an
independent initialisation at mean squared cosine
**0.6307023078841583** and that the sharing is concentrated at the top of the
spectrum. None of that is a test of whether rank 64 buys anything **here**.
This page is that test.

It is experiment S3 of [[paper-reading-2026-09-09]], run on 2026-09-09 with the
scope Samuel set: 15 traits at ranks 1, 4 and 16, not the 40 traits at 1, 4 and
8 that the reading note proposed.

## What was trained

45 runs, `qwen35/train_rank_sweep.py`, 15 traits x 3 ranks. The 15 are three per
Big Five factor with keyings mixed, and every one of them also has a rank-64
second seed in `data_null_seedpaired_s40_matched` ([[seed-floor]]): helpful,
cold, harsh (Agreeableness); organized, disorganized, careful
(Conscientiousness); relaxed, anxious, fretful (Emotional Stability);
extraverted, quiet, assertive (Extraversion); intellectual, simple,
unimaginative (Intellect).

Everything but the rank is the 134-run sweep's recipe. `train_rank_sweep.py`
does not fork `train_qwen35.py`; it imports it and calls `_train_impl`
unchanged, so the corpus (`data_common`, 445 pairs), the 248 targeted modules,
the seeds (0 / 0), the learning rate, `beta`, `loss_type` `["sigmoid","sft"]` at
weights `[1.0, 0.1]` and `kl_coef` 0.001 are the recorded values of the 134.
`lora_alpha` is set to **2r** at every rank, so the effective scale `alpha/r`
stays at OCT's **2.0** - `expected_scaling` reads 2.0 in all 45 runmeta records
(`analysis/rank_sweep.json#ranks.r*.training_health.*.expected_scaling`).

The objective travelled: first-step loss is
**0.8896549344062805 to 0.9429400563240051** across the 45 runs, identical
trait by trait at every rank, not 0.6931
(`#ranks.r1.training_health.*.loss_first`). 0.6931 is ln 2, what plain sigmoid
DPO gives on the first step; the extra 0.2 is the SFT term at 0.1. The same
check is what [[seed-floor]] uses to certify the matched arm.

## The frames were made to nest, and the nesting was gated

A LoRA writes `dW = s B A`. PEFT draws `A` with `kaiming_uniform_` on a tensor
of shape `(r, in_features)`; the bound depends on `in_features` only, so every
rank draws from the same distribution - but the **draw** consumes `r * in_features`
numbers, so a rank-1 run's `A` is not the first row of a rank-64 run's `A`. Left
alone, each rank would write into its own independent random slice of the
2560-dimensional input space and every cosine against the zoo would sit at the
`r/d` floor for a reason that has nothing to do with rank.

So the zoo's seed-0 rank-64 `A_0` was reproduced once, on a CPU container,
through the same code path `_train_impl` uses (same loader, same target
discovery, the same three seed calls immediately before `LoraConfig`), and each
rank-r run's `A` was overwritten with its **first r rows** immediately after the
trainer was built and before the first optimizer step. `span(A_1)` is then a
subspace of `span(A_4)`, of `span(A_16)`, of `span(A_64)`.

That reproduction was **gated, not assumed**. Against a trained zoo adapter's
`A` (`pc-qwen35-sweep:/active`), over all 248 modules
(`qwen35/phase10_runs/ranksweep_a0.log`, `[gate]` line):

| statistic | value |
|---|---|
| `cos_mean` | 0.9999004207772992 |
| `cos_min` | 0.9996430506239348 |
| `drift_mean` | 0.013256896916831113 |
| `drift_max` | 0.02671875357337077 |

The zoo's own figure for how far `A` moves in training is
**0.014605041334818797** (`analysis/blog_data.json#a_drift`,
[[stage-one-training-config]]), and trait-to-trait `A` cosines in the zoo are
0.99997 (`analysis/lora_a_identity.json`). A wrong draw would have read cosine
about 0. The reproduction is the zoo's `A_0`.

After training, each run's saved `A` was compared back to `A_0[:r]`. Mean drift
per rank: **0.020348 to 0.022317** at r = 1, **0.020262 to 0.021928** at r = 4,
**0.017944 to 0.019417** at r = 16
(`analysis/rank_sweep.json#ranks.r*.training_health.*.a0_drift_mean`, min and
max over the 15 traits). Slightly above the zoo's 0.0146 at low rank, which is
what one expects when a smaller `B` carries more of the gradient, and far below
anything that would break the nesting.

## The fit degrades sharply with rank

From `analysis/rank_sweep.json#ranks.r*.training_health` (min and max over the
15 traits at each rank), against the 134-run sweep's own range in
[[stage-one-training-config]]:

| rank | `lora_alpha` | last loss | reward margin |
|---|---|---|---|
| 1 | 2 | 0.7189493179321289 - 0.8533710837364197 | 0.13416672019021852 - 0.4321832944239889 |
| 4 | 8 | 0.3073895573616028 - 0.5185328125953674 | 1.0998932932104384 - 2.7983998230525424 |
| 16 | 32 | 0.16749370098114014 - 0.22805458307266235 | 5.796228579112461 - 11.25320999962943 |
| **64** (the zoo) | 128 | 0.14250712096691132 - 0.281541645526886 | 5.072036419595991 - 18.467349461146764 |

At rank 1 the DPO objective barely moves: the loss falls from about 0.90 to
about 0.79 in 13 optimizer steps and the reward margin is a fifteenth of the
zoo's. Rank 16 is already inside the zoo's own band on both. **Whatever else
this page says, rank 1 is not a run that fitted the preference data.**

## The geometry

All Grams are exact, computed in one CPU pass over the 248 modules by
`qwen35/cross_gram_rank_sweep.py` using `cross_gram_full_on_modal.py`'s identity
`X[i,j] = s_a s_b sum_modules sum((B_i^T B_j) * (A_i A_j^T))`, which is valid for
unequal ranks because both factors are `(r_i x r_j)`. Output
`qwen35/results/rank_sweep_grams.npz`, analysed by
`qwen35/analyse_rank_sweep.py` into `qwen35/analysis/rank_sweep.json`.

Every number in this section is the mean over the 15 traits at the key named.

| statistic | r = 1 | r = 4 | r = 16 |
|---|---|---|---|
| same-trait cosine with the rank-64 adapter | **0.1089283877528818** | **0.21955800156090902** | **0.45514002253737584** |
| ... divided by `sqrt(r/64)` | 0.8714 | 0.8782 | 0.9103 |
| projection coefficient `<dW_r, dW_64> / \|dW_64\|^2` | 0.01759081867579675 | 0.0705247383860075 | 0.26602018233036157 |
| energy fraction (cosine squared) | 0.011874792233604586 | 0.04824167066055809 | 0.2072102776110409 |
| norm ratio `\|dW_r\| / \|dW_64\|` | 0.16159263836829713 | 0.3213821821443084 | 0.5845362331300447 |
| top-1 identification among the 134 | **15 / 15** | **15 / 15** | **15 / 15** |
| factor-chart coordinates, Pearson over 15 x 5 | **0.9967445664202496** | **0.9970154457028848** | **0.9984303204790593** |
| 15 x 15 arrangement, Pearson over 105 off-diagonals | **0.9906443136606387** | **0.991685428157916** | **0.9967176309536419** |
| same-trait cosine with the rank-64 **second seed** | 0.0028375826395418066 | 0.004639542037543767 | 0.010323693027390406 |

Keys: `#ranks.r<r>.same_trait_cosine_vs_rank64.mean`,
`.same_trait_cosine_over_sqrt_r_over_64`, `.projection_coefficient.mean`,
`.energy_fraction.mean`, `.norm_ratio.mean`,
`.identification_among_134.top1`, `.chart.pearson_75`,
`.arrangement_15x15.pearson_offdiag`, `.cross_seed_same_trait_cosine.mean`.

### Read the raw cosine with the attenuation in mind

The same-trait cosine is small at low rank and that is largely **arithmetic, not
loss of trait**. A rank-1 adapter can only write along one input direction out
of the rank-64 adapter's 64, so even a rank-1 run that recovered exactly what
the rank-64 run did along that direction would score
`|B_64[:,0]| / ||B_64||_F`, which is `1/8 = sqrt(1/64)` if the 64 output columns
carry equal energy. Dividing by `sqrt(r/64)` removes that factor **under the equal-energy
assumption** and gives **0.8714, 0.8782, 0.9103** - close to constant, and close
to one. The assumption was verified only for column 0 and only on one trait (see
below). At r = 4 and r = 16 it additionally requires that the rank-r run's four
or sixteen output vectors align with those same first columns, which was not
checked. The near-constancy of the three numbers is suggestive of that, not
evidence for it.

The mechanism was checked directly on one trait, `helpful`, by downloading its
rank-1 and rank-64 adapters and comparing the rank-1 run's single output vector
`b_1` to **column 0** of the rank-64 run's `B`, which is the output vector the
rank-64 run paired with the same input row `A_0[0]`
(`qwen35/analysis/rank_sweep_mechanism.json`, 248 modules):

- `cos(b_1, B_64[:,0])` mean **0.8418963800198176**, median 0.8476920354823998,
  sd 0.03567290074353576, range 0.7132242290089871 to 0.9249524458784338.
- `|B_64[:,0]| / ||B_64||_F` mean **0.12433727756861333** against the
  equal-energy reference 0.125 - so the rank-64 adapter's output columns do
  carry near-equal energy, and column 0 is not special.
- The two rows of `A` really are shared: `cos(A_r1[0], A_64[0])` mean
  0.9998544193575315, min 0.999384219269407.
- Their product, **0.10467910388653483**, is the leading term of the Frobenius
  cosine at rank 1, and `helpful`'s measured value is
  **0.10695463755160249**
  (`#ranks.r1.same_trait_cosine_vs_rank64.per_trait.helpful`) - agreement to
  about 2 percent.

Stated plainly: **given the same input direction, retraining at rank 1 recovers
the output direction the rank-64 run assigned to that direction, at cosine
0.84.** That is a mechanism, measured on one trait; it has not been repeated on
the other 14.

### Identification

Every rank-r adapter is nearest its own word among all 134 rank-64 adapters, at
every rank, mean rank 1.0 against a chance 67.5
(`#ranks.r*.identification_among_134`). **This is identification within a
shared frame, not across one**, and it is a weaker test than the 40 of 40 in
[[seed-floor]] or the 40 of 40 in [[column-space-structure]], both of which
identify across an independent initialisation. Here the rank-1 `helpful` shares
`A_0[0]` with rank-64 `helpful` and with rank-64 `cold` alike; what separates
them is whether its single output vector points where the rank-64 run pointed
*that trait's* column 0. The signal is trait-specific - the mechanism section
puts it at cosine 0.84 - but the frame is not independent evidence the way a
second seed is. The margin at rank 1 is narrow and
should be quoted as such: minimum same-trait cosine
**0.10267766283471824** against maximum off-trait cosine
**0.08385147293173222**, a gap of about 0.019. At r = 4 it is 0.20778 against
0.16813 and at r = 16 it is 0.44149 against 0.33228.

### The chart, and what the chart number does not say

Chart coordinates are `fa_chart.FAChart().coords_external` on the 134-column of
the cross-Gram, against the same trait's own rank-64 coordinates
([[factor-chart]]). Pearson over the 75 (15 traits x 5 axes) values is
0.9967 / 0.9970 / 0.9984, and every axis is above 0.996 individually
(`#ranks.r*.chart.per_axis_pearson`). Per trait, the cosine between the two
5-vectors has mean **0.99711027422409** and minimum
**0.9839794198638647** at r = 1 (`#ranks.r1.chart.per_trait_direction_cosine`).
**Where a trait sits in the five-factor chart is set by rank 1.**

The chart *length* is another matter and carries the same attenuation as the raw
cosine. The rank-1 chart vector is **0.023001148580170553** of the rank-64
one's length (`#ranks.r1.chart.chart_len_ratio`, mean), and the chart captures
**0.08620840720567556** of a rank-1 adapter's own norm against
**0.6052728286663711** for its rank-64 twin
(`#ranks.r1.chart.chart_captures_frac_of_norm` and
`...chart_captures_frac_of_norm_rank64`; the 134-adapter mean is
0.5784887830866848, `#rank64_reference.chart_captures_frac_of_norm_mean_134`).
This is **not** evidence that a rank-1 adapter lies off the factor span: the
chart is built from inner products with the 134 rank-64 adapters, so it is
attenuated by exactly the `sqrt(r/64)` factor the raw cosine is. The
interpretable chart statistics here are the direction cosine and the Pearson.

### The arrangement is preserved but stretched

The 15 x 15 matrix of cosines among the rank-r adapters correlates with the same
block of the rank-64 Gram at Pearson 0.9906 / 0.9917 / 0.9967 and Spearman
0.9908 / 0.9919 / 0.9964 over 105 off-diagonal pairs
(`#ranks.r*.arrangement_15x15`). For comparison, [[cross-seed-geometry]] gives
0.997 for the two seeds' agreement on the same statistic.

But the matrix is not the same matrix. Mean off-diagonal cosine is
**0.11081816477278365** at r = 1 against **0.09299181500482677** at r = 64, and
the standard deviation is **0.25493453731807075** against
**0.176437278006883** - a spread 1.44 times as wide
(`#ranks.r1.arrangement_15x15`). The ordering of trait-to-trait relations
survives; their scale does not. At r = 16 the sd ratio is 1.29
(0.22777084171487302 against 0.176437278006883).

### The cross-seed floor moves with rank, as it should

Against the 40 objective-matched **second-seed** rank-64 adapters the frames do
**not** nest - a different `A` draw - so this is a floor, not a replication
statistic. Same-trait cosine is 0.002838 / 0.004640 / 0.010324, which is
**0.1571 / 0.2569 / 0.5716** of the rank-64-against-rank-64 seed floor
**0.01806099632415457** (`analysis/crossseed_arms.json#[1].same[1]`,
[[seed-floor]]; ratios at `#ranks.r*.cross_seed_ratio_to_rank64_floor`). Those
ratios sit near `sqrt(r/64)` = 0.125 / 0.25 / 0.5, the same attenuation as
above, which is the sign that nothing unexpected is happening across the seed
boundary at low rank. The reading note's own guess for rank 1 was `r/d` =
1/2560 = 0.00039 ([[paper-reading-2026-09-09]], S3); the measured 0.00284 is
seven times that, and `sqrt(r * 64)/d` = 8/2560 = 0.003125 is the closer shape.
Neither is a fitted model.

## Behaviour: it does not survive

Ten of the 15 traits - two per Big Five factor, one of each keying: helpful,
cold; organized, disorganized; relaxed, anxious; extraverted, quiet;
intellectual, simple - were generated from at ranks 1 and 4 on the 24-prompt Big
Five battery (`qwen35/bigfive_probes.py`), greedy, thinking off, 200 new tokens,
by `qwen35/eval_bigfive.py`, and judged blind by
`anthropic/claude-sonnet-4.5` through `qwen35/judge_personas.py`. 960 judgments,
0 failed calls (`analysis/rank_sweep.json#behaviour.n_judgments`,
`.failed_calls`). Rank 16 was not generated: the meter cap had no room for it.

The four generation containers produced **bit-identical** base texts, as greedy
decoding at `torch.manual_seed(0)` should. The judged base profile agrees with
`judged_100.json`'s: Extraversion 4.077 against 4.139, Agreeableness 4.677
against 4.688, Conscientiousness 5.471 against 5.595, Emotional Stability 4.639
against 4.684, Intellect 5.467 against 5.506
(`#behaviour.base_profile_rank_sweep_run` and `.base_profile_judged_100`). Judge
repeat reliability, recomputed from the file rather than taken from the judge's
stdout: Extraversion r 0.5954913341754137 (n 48), Agreeableness
0.8269764085834823 (48), Conscientiousness 0.7941357502021545 (48), Emotional
Stability 0.77350713496128 (47), Intellect 0.875515044303485 (48)
(`analysis/rank_sweep.json#behaviour.judge_repeat_reliability`, computed from
`phase10_runs/judged_rank_sweep.json#records[*].repeat`, the second judgment of
the 5 percent of items `judge_personas.py` duplicates).

Own-factor amplification is `analyse_s2register.py`'s statistic unchanged: sign
the trait by its keying, then express the judged own-factor mean as a percentage
of the headroom between the base condition's own-factor mean and the end of the
1-7 scale. From `#behaviour.ranks`:

| | rank 1 | rank 4 | rank 64 (`judged_100.json`, condition `stage1`) |
|---|---|---|---|
| mean own-factor amplification | **-0.40968...** | **-0.67513...** | **+27.081...** |
| sd over the 10 traits | 3.5252 | 2.9830 | 20.2628 |
| sem | 1.1148 | 0.9433 | 6.4076 |
| range | -4.632 to +7.357 | -6.808 to +4.632 | - |
| traits with the right sign | 5 of 10 | 4 of 10 | - |
| Pearson with the rank-64 value across the 10 traits | -0.153 | +0.3244 | - |

(Exact values at `#behaviour.ranks.r1.mean_own_factor_amplification` =
-0.40968216455696196, `...r4...` = -0.6751130653266327 and
`...mean_own_factor_amplification_rank64` = 27.081116865869846.)

**At rank 1 and rank 4 the adapters do not move their own factor at all.** The
mean is within one standard error of zero, half the traits move the wrong way,
and the per-trait values do not track the rank-64 ones. The rank-64 adapters on
the same battery and the same judge move their own factor +27 percent of the
headroom.

This is the same thing the training log says. At rank 1 the reward margin is
0.134-0.432 against the zoo's 5.072-18.467; the adapter that barely fits the
preference data does not produce the behaviour either.

## What this establishes, and what it does not

**Establishes a dissociation.** With the input frame held fixed, the zoo's
*arrangement* - which trait is near which, and where each sits in the
five-factor chart - is set by rank 1. The 15 x 15 cosine matrix, the chart
coordinates and the identification all survive at a rank 64 times smaller, and
the residual disagreement with rank 64 is at the third decimal place of a
correlation. The mechanism, on one trait, is that the rank-1 run recovers the
same output direction the rank-64 run assigned to the same input direction, at
cosine 0.84. **And the behaviour does not survive at all**: own-factor
amplification is -0.41 percent of headroom at rank 1 against +27.08 at rank 64.

The direction a trait's gradient points is available in the first step and
barely depends on rank. The magnitude needed to change what the model says is
not. That is a statement about this recipe - 13 optimizer steps of DPO at
effective scale 2.0 - and the geometry half of it is what every result on
[[geometry-overview]] rests on.

**Does not establish.** That rank 1 is a substitute for rank 64; the behaviour
section is the direct refutation. Nor anything about **independently
initialised** low-rank adapters: the whole design here nests the frames on
purpose, and a rank-1 adapter drawn from its own seed reads 0.0028 against the
zoo for reasons of geometry alone. Nor whether a *longer* or *higher-scale*
rank-1 run would recover the behaviour - the number of steps, the learning rate
and `alpha/r` were all held at the zoo's values, so this varies rank and nothing
else, and separating "rank 1 cannot" from "rank 1 needs more steps" would take
another arm. Nor was rank 16 evaluated behaviourally; it sits inside the zoo's
own loss and margin band and is the interesting missing point.

**Does not test** SliderSpace's actual claim. SliderSpace trains a direction to
a target in a semantic embedding space; this retrains the same DPO objective at
a lower rank. That rank-1 DPO preserves the arrangement is consistent with their
rank-one default being adequate, and it is not the same experiment. See
[[paper-reading-2026-09-09]] S1 for the experiment that is.

## Deviations from the brief, recorded

- **Ranks 1, 4, 16 on 15 traits**, per Samuel's 2026-09-09 instruction, not the
  40 traits at 1, 4, 8 in [[paper-reading-2026-09-09]] S3.
- The `A`-nesting was necessary and is not in the reading note's design. Without
  it every statistic on this page would have been at the seed floor.
- Behaviour was generated at **200 new tokens**, not 512, because
  `qwen35/oct_stage2.py:1043` generated `phase10_runs/judged_100.json` at 200
  and that file is the rank-64 comparison. A 512-token run would not have been
  comparable.

## Run and spend

Everything ran as systemd units with logs under `qwen35/phase10_runs/`:
`zoo-ranksweep-a0` (the `A_0` capture and gate, CPU),
`zoo-ranksweep-probe` (one trait at rank 1, to price the rest),
`zoo-ranksweep` (the 45 runs),
`zoo-ranksweep-gram` (all nine cross-Gram blocks, CPU),
and `zoo-ranksweep-eval-r{1,4}{a,b}` (the four generation containers).
App names `pc-qwen35-phase11-ranksweep`, `...-ranksweep-gram`,
`...-ranksweep-eval`.

Container time, estimated as the sum of the 45 recorded `train_seconds`
(6.5435 hours, `phase2_runs/results_data_rank_sweep.json`) plus about 75 seconds
of per-container overhead measured from the probe, plus the measured wall time
of every non-training container: **about 11.0 container-hours**. At
`zoo40_meter.sh`'s convention of $2.10 per hour for every container whatever it
is - three of these were CPU-only and Modal bills those far below that - the run
drew about **$23.0** of the $25 Samuel authorised for it on 2026-09-09. The
meter's `BUDGET` was raised by exactly $25, from $2518.00 to $2543.00; sibling
agents raised it further from both sides during the same night, which is why the
log shows other numbers. Judge spend is OpenRouter, not Modal: 168 calls.

Two things worth knowing for the next run of this shape. `modal run` under
`Type=simple` **crash-loops** if the module raises at import inside the
container, and `systemctl start` on a unit that is still crash-looping is a
no-op - two relaunches went to the old code before the unit was stopped first.
And restarting `zoo40-meter.service` mid-interval double-counts one tick: the
restart at 23:31 added about 4.25 container-hours, roughly $8.9, of phantom
spend to the shared reading. That is an over-count, not an under-count, and no
run should be charged for it.

## Files

| file | what |
|---|---|
| `qwen35/train_rank_sweep.py` | the 45 runs, the `A_0` capture and its gate |
| `qwen35/cross_gram_rank_sweep.py` | all nine cross-Gram blocks in one CPU pass |
| `qwen35/analyse_rank_sweep.py` | the statistics on this page |
| `qwen35/analysis/rank_sweep.json` | every number quoted here |
| `qwen35/analysis/rank_sweep_mechanism.json` | the one-trait mechanism check |
| `qwen35/results/rank_sweep_grams.npz` | the raw blocks and norms |
| `qwen35/phase2_runs/results_data_rank_sweep.json` | 45 runmeta records |
| `qwen35/phase10_runs/ranksweep_*.log` | the run logs |
| `pc-qwen35-adapters:/data_rank_sweep/r{1,4,16}/<trait>` | the 45 adapters |
| `pc-qwen35-adapters:/data_rank_sweep/_A0_seed0_r64.safetensors` | the shared `A_0` |
