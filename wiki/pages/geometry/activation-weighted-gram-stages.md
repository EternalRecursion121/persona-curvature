---
title: The stage boundary in the activation-weighted metric
summary: Under the activation-weighted metric a stage-one and a stage-two adapter for the same trait reach cosine +0.0090 where the Frobenius one reads +0.00017, but the frame-overlap ceiling for those two frames is 0.7855, so the cross-stage agreement is 0.92 per cent of what a perfectly reproduced update would score against 74.6 per cent for the same trait retrained at a second seed - eighty times lower; the two stages' grand means are functionally near-orthogonal too, at +0.0167; 56 of 134 stage-two adapters still find their own stage-one adapter and the arrangement correlates at 0.40, so the verdict by the preregistered thresholds is a different function with a matching arrangement.
status: current
sources:
  - qwen35/PREREG_actgram_stage2.md
  - qwen35/act_gram_stages_on_modal.py
  - qwen35/analyse_act_gram_stage2.py
  - qwen35/analysis/act_gram_stage2.json
  - qwen35/results/cross_gram_actweighted_stage1_x_stage2.npz
  - qwen35/phase10_runs/actgram_stage2_results.json
  - qwen35/phase10_runs/actgram_stages.log
  - qwen35/phase10_runs/actgram_stages_smoke.log
  - qwen35/analysis/stage2_structure.json#stage1_x_stage2_exact
  - qwen35/analysis/stage2_structure.json#centred_cosines
  - qwen35/analysis/stage2_structure.json#shared_component
  - qwen35/results/cross_gram_full_loras_introspection_x_loras_introspection.npz
  - qwen35/analysis/act_gram.json
  - qwen35/analyse_stage2_structure.py
last_verified: 2026-09-11
tags: [geometry, stage-two, metric, cross-stage, nulls]
---

# The stage boundary in the activation-weighted metric

## The question

Open Character Training has two stages, and the zoo ran both for all 134 traits:
stage one is DPO on the constitution's preference pairs, stage two is SFT on
transcripts the trained model generates about itself
([[stage-two-introspection]]). They are two separate LoRAs with two separate
random LoRA-A draws - stage one at seed 0, stage two at `sft_seed 123456`, one
draw shared inside each stage ([[stage-two-geometry]]). So their Frobenius
cross-Gram is zero by construction: same-trait cosine
**+0.00016589283708173512**, different-trait **+1.784355184601498e-05**
(`qwen35/analysis/stage2_structure.json#stage1_x_stage2_exact`).

[[activation-weighted-gram]] showed why a cross-frame comparison reads as zero
in the Frobenius metric and what happens when the metric is changed: the input
covariance C of a module has a participation ratio of about ten, so two
independent rank-64 frames overlap at 0.89 in C where they overlap at 0.021 in
Frobenius, and the same trait retrained at a second seed goes from +0.0181 to
+0.6643. **Under C the two stages can be compared.** This page asks whether the
stage-two update is the same function as stage one's for the same trait, or a
different function whose arrangement across the 134 traits happens to match -
which is what the centred-cosine correlation of 0.81 on [[stage-two-structure]]
already established. It was preregistered in
`qwen35/PREREG_actgram_stage2.md`, with one amendment made after the smoke run
and before the full run returned (below).

**The answer is the second.** The cross-stage cosine does rise under C, by a
factor of 54, from +0.000166 to **+0.0089804086688677**. But the frame-overlap
ceiling for these two particular frames is **0.78549746319657**, so the
agreement is **1.14 per cent** of what a perfectly reproduced update would
score, against **74.56 per cent** for the same trait retrained at a second seed
measured on the same run in the same metric. **In relative terms the stage
boundary is about sixty-five times harder to cross than the seed boundary, and
the change of metric does not move that**: under Frobenius the two ratios are
0.77 per cent and 84.12 per cent.

## What was run

`qwen35/act_gram_stages_on_modal.py`, one A100-80GB, harvesting C exactly as
[[activation-weighted-gram]] did (base `Qwen/Qwen3.5-4B`, **no adapter**, bf16
forward, fp32 accumulation, TF32 off, uncentred C for all 248 modules) on the
same two token arms - `pool445`, the 445-prompt shared training pool with prompt
tokens only, **20,612 tokens**, primary; and `resp4378`, the 24 steering prompts
with the base model's stored greedy responses, **5,106 tokens** - plus a
mean-centred twin of the primary arm. The participation ratios came out
identical to the earlier run to two decimals (21.50, 43.83, 72.99), which is the
first sign the harvest is deterministic.

Then one exact Gram over four adapter sets, 323 adapters, all at
`lora_alpha/r = 2.0` and 248 modules (`analysis/act_gram_stage2.json#sets`):

| tag | volume | path | n |
|---|---|---|---|
| `s1s0` | `pc-qwen35-sweep` | `/` | 134 stage-one, seed 0 |
| `s2s0` | `pc-qwen35-oct2` | `/loras_introspection` | 134 stage-two, `sft_seed 123456` |
| `s1s1` | `pc-qwen35-sweep` | `/data_null_seedpaired_s40_matched` | 40 stage-one, seed 1 |
| `s2s1` | `pc-qwen35-oct2` | `/seed1/loras_introspection` | 15 stage-two, `sft_seed 1` |

Everything else is a linear function of that Gram and is computed locally and
exactly by `analyse_act_gram_stage2.py`. **Removing stage two's shared
direction needs no re-run**: with `M2` the stage-two grand mean,
`<u, v_k - M2> = <u, v_k> - mean_l <u, v_l>` and
`||v_k - M2||^2 = G22_kk - 2 mean_l G22_kl + mean_{l,m} G22_lm`.

`v_k - M2` stays inside the span of the stage-two frames, so the frame-overlap
ceiling is unchanged by the removal to first order and the same ceiling is used
for the raw and the residual blocks.

## Validation

**The Frobenius arm reproduces `stage2_structure.json#stage1_x_stage2_exact`,
which was derived by a completely different route** - that file never touches a
stage-two adapter directly, it recovers the cross block from the two-volume
stage-one-by-persona Gram as `X12 = (X[s1, persona] - G1) / 0.25`. This run
reads both adapters off their volumes and contracts them. The two agree
(`analysis/act_gram_stage2.json#validation`):

| | this run | `stage2_structure.json` | difference |
|---|---|---|---|
| same-trait cosine | 0.00016589286627368406 | 0.00016589283708173512 | 2.92e-11 |
| different-trait cosine | 1.7843574449912248e-05 | 1.784355184601498e-05 | 2.26e-11 |
| cosine between grand means | 0.0001676146544225693 | 0.00016761445563503107 | 2.0e-13 |
| top-1, stage-two query | 42 of 134 | 42 of 134 | - |
| mean rank, stage-two query | 9.537313432835822 | 9.537313432835822 | - |
| corr with stage-one within-block | 0.2175074513638623 | 0.21750655535315414 | 9.0e-7 |
| corr with stage-two within-block | 0.2064887480525787 | 0.20648778534428447 | 9.6e-7 |
| centred-cosine correlation | 0.8117877815269501 | 0.811787783969743 | 2.4e-9 |

The preregistered bar was 1e-5. The stage-two within-stage Gram also reproduces
`results/cross_gram_full_loras_introspection_x_loras_introspection.npz` to a max
absolute cosine difference of **5.182098067324503e-08**.

**One labelling defect was found by that check and is filed, not fixed here.**
The reference's key is named `top1_stage1_finds_own_stage2` and
[[stage-two-structure]] renders it as "a stage-one adapter's nearest stage-two
adapter is its own trait for 42 of 134", but `analyse_stage2_structure.py:116`
computes `ranks = [1 + sum(C12[:, j] > C12[j, j])]`, which ranks each
**stage-two** adapter among the 134 stage-one ones. Compared like with like this
run gives exactly 42 and mean rank 9.537313432835822. The other direction - the
one the name and the sentence describe - reads **59 of 134, mean rank
7.514925373134329** on the identical matrix. See
[[source-contradictions|S29]].

**Frame spread.** Mean over modules of each adapter's relative deviation of `A`
from its own set's mean `A` (`#frame_spread`): stage one **0.010178045947067138**
seed 0 and 0.01010937949890391 seed 1; stage two **0.07506962422461759** seed 0
and 0.07315878054848121 seed 1. Stage two's frame is about seven times less
shared than stage one's, which is the same direction as `PHASE3_VERDICT.md`'s
"stage-2 A drifts 10% from init (stage 1: 1.5%)" - though it is **not** that
statistic, because this one measures deviation from the set mean and not from
the initialisation. The Gram itself uses each adapter's own `A` and is exact
regardless.

## The numbers

`analysis/act_gram_stage2.json#arms.<arm>`. The 134 traits are those present in
both stages.

| | Frobenius | **C, pool445** | C, pool445 centred | C, resp4378 |
|---|---|---|---|---|
| frame-overlap ceiling, stage one x stage two | 0.021424757177705197 | **0.78549746319657** | 0.5056942410096681 | 0.6703927972002262 |
| same trait, cross stage | +0.00016589286627368406 | **+0.0089804086688677** | +0.0018032485394305368 | +0.0054198516710375014 |
| sd of same trait | 9.027747503297509e-05 | 0.004338227410866662 | | |
| different trait, cross stage | +1.7843574449912248e-05 | +0.002697205320217068 | +7.85183164365826e-05 | +0.0015400591068266403 |
| **R = same / ceiling** | **0.007743045342250773** | **0.01143276597268954** | 0.0035658870384408068 | 0.008084591143688489 |
| top-1, each stage-two adapter among the 134 stage-one | 42 / 134 | 21 / 134 | 6 / 134 | 8 / 134 |
| mean rank (chance 67.5) | 9.537313432835822 | 16.813432835820894 | | |
| corr with the stage-one within-block | 0.2175074513638623 | 0.28957300102293054 | | |

After **stage two's shared direction is removed**, which holds
0.1514038882692439 of the average squared norm in the Frobenius metric and
**0.2708387694659428** in C (`#arms.pool445.grand_mean_share_of_mean_norm2`):

| | Frobenius | **C, pool445** |
|---|---|---|
| same trait, cross stage | +0.0001588825830964221 | **+0.007244551315899727** |
| different trait | -1.2896221212434663e-06 | -6.62017186027651e-05 |
| **R_res = same / ceiling** | **0.0074158405520580095** | **0.009222883147729255** |
| same trait / sd(different trait) | 2.9962128440633298 | 2.1923141592358397 |
| top-1, stage-two query | **76 / 134** | **56 / 134** |
| mean rank (chance 67.5) | 5.126865671641791 | 7.858208955223881 |
| corr with the stage-one within-block | 0.3621461762556352 | **0.40294962029699516** |

Removing the shared direction makes identification much better in both metrics -
76 of 134 against 42 under Frobenius, 56 against 21 under C - and raises the
arrangement correlation from 0.2175 to 0.3621 and from 0.2896 to 0.4029. The
shared register is noise for the cross-stage question, exactly as the
preregistration assumed. Removing **both** stages' shared directions gives
R 0.009754181133485019, 54 of 134 and correlation 0.426311842398635
(`#arms.pool445.cross_stage_both_shared_removed`).

### The benchmarks, measured on the same run in the same metric

This is the comparison the verdict rests on
(`#arms.pool445.second_seed_checks`, `#arms.frob.second_seed_checks`):

| pair | n | same-trait cosine (C) | ceiling (C) | **R** | R under Frobenius |
|---|---|---|---|---|---|
| stage one x stage two, seed 0 | 134 | +0.0089804086688677 | 0.78549746319657 | **0.0114** | 0.0077 |
| stage one x stage two, **seed 1** | 15 | +0.008061300252633984 | 0.7960848900447363 | **0.0101** | 0.0076 |
| stage one, seed 0 x seed 1 | 40 | +0.6642887281525424 | 0.8909847593616453 | **0.7456** | 0.8412 |
| stage two, seed 0 x seed 1 | 15 | +0.6218038652787304 | 0.9052140531310638 | **0.6869** | 3.0196 |

The R column at full precision, from
`#arms.pool445.second_seed_checks.*.ratio_to_ceiling` and
`#arms.frob.second_seed_checks.*.ratio_to_ceiling`: cross-stage seed 0
**0.01143276597268954** (Frobenius 0.007743045342250773), cross-stage seed 1
**0.010126181709316171** (Frobenius 0.007644768084004354), stage one cross-seed
**0.7455668811085819** (Frobenius 0.8411666943129359), stage two cross-seed
**0.6869136234992818** (Frobenius **3.0196348702424674**).

Three things to read off it.

1. **The second seed reproduces the stage result.** The 15 traits that have both
   a seed-1 stage-one adapter and a seed-1 stage-two adapter give R = 0.0101
   against the seed-0 set's 0.0114. The stage boundary is not an artefact of one
   pair of frames.
2. **Stage two re-learns itself almost as well as stage one does.** At its own
   second seed, stage two reaches 0.6869 of its ceiling where stage one reaches
   0.7456. So a stage-two adapter is perfectly reproducible under a change of
   initialisation; it simply is not stage one.
3. **The Frobenius stage-two cross-seed ratio is 3.02, which exceeds 1 and is
   therefore not a ratio to a ceiling at all.** That is the same anomaly
   [[stage-two-second-seed]] records as "the attenuation slope is 0.116 - 4.6x
   the r/d prediction, unexplained": the large shared register lifts the
   stage-two cross-seed cosine above what frame overlap alone allows. Under C
   the anomaly disappears (0.6869, comfortably below 1), because in C the shared
   register is no longer cheap - the ceiling itself is 0.905. Recorded here as
   an observation; nothing on this page depends on it.

### The two stages' grand means

Frobenius +0.0001676146544225693; under C **+0.01667321809542123** against a
ceiling of 0.7855, which is 2.1 per cent of it. **The two shared directions are
functionally near-orthogonal as well as coordinate-orthogonal.** Stage two's own
shared component is much larger in the functional metric than in the Frobenius
one: every stage-two adapter sits at cosine **0.5218264341353712** to the
stage-two grand mean under C against 0.38929787448084396 under Frobenius, and
the mean within-stage-two off-diagonal cosine is **0.26694991920809574** under C
against 0.14519684032691033 (`#arms.*.cos_to_own_grand_mean`,
`#arms.*.within_stage_offdiag_mean_cosine`). The introspection register is a
bigger share of what stage two does to the model than of what it does to the
weights.

### The arrangement

The centred-cosine correlation between the two stages' within-stage blocks - the
statistic behind the r = 0.81 on [[stage-two-structure]] - is
**0.8137665247986187** under C against 0.8117877815269501 under Frobenius
(`#arms.*.centred_cosines.corr_stage1_stage2`). **The metric changes it by
0.002.** The arrangement result was never a metric artefact, and the functional
metric does not improve it either.

## The verdict

`PREREG_actgram_stage2.md` set two bands on `R_res`, the ratio of the
shared-direction-removed cross-stage cosine to the frame-overlap ceiling, with
the benchmarks being the stage-one cross-seed ratio (0.7455668811085819, "the
same function re-learned in a different frame") and the cross-seed
different-trait ratio (0.0800, "unrelated"):

- "the same function as stage one" if `R_res >= 0.50` and the discriminability
  condition held;
- "a different function with a matching arrangement" if `R_res <= 0.15`, with
  more than 13 of 134 identified and the arrangement correlation above 0.20.

Measured: **`R_res = 0.009222883147729255`**, 56 of 134 identified (mean rank
7.86 against a chance 67.5), arrangement correlation **0.40294962029699516**.
All three conditions of the second band hold, and `R_res` is below the
"unrelated" benchmark of 0.0800 by a factor of nine while identification is
fifty-six times chance. **The verdict is "a different function with a matching
arrangement"** (`#verdict`), and it is not a close call: `R_res` would have to
rise by a factor of 54 to reach the lower band's edge.

**One preregistered condition was amended, after the smoke run and before the
full run returned.** The prereg's "same trait exceeds different trait by a factor
of 3" is ill-defined in the residual block, because removing stage two's grand
mean centres every row of the cross block exactly, forcing the different-trait
mean to about `-same/(n-1)`; the smoke returned `same_over_diff = -14.39`, which
is arithmetic, not a finding. It was replaced by `same-trait mean / sd(different
trait) >= 3`, reported as `same_over_diff_sd` and reading 2.19 under C. That
condition belongs to the branch the verdict did **not** take, so it changes
nothing here; the amendment is on the record in `PREREG_actgram_stage2.md` with
the time it was written.

## What this establishes, and what it does not

**Established.** (1) The stage-one and stage-two updates for one trait are not
the same function, in the only metric in which the question can be asked at all:
1.14 per cent of the frame-overlap ceiling raw and 0.92 per cent with the shared
register removed, against 74.6 per cent for the same trait re-learned at a second
seed. (2) That is not a property of the metric - the Frobenius ratios are 0.77
and 0.74 per cent, and the ordering of the two boundaries is the same in both.
(3) It is not a property of one pair of frames - the seed-1 arm reproduces it at
R = 0.0101 over 15 traits. (4) What *does* survive the stage boundary is the
arrangement and a weak trait-specific residual: 56 of 134 identified against a
chance of about one, arrangement correlation 0.40 to the stage-one within-block,
and the centred-cosine correlation of 0.81 unchanged by the metric. (5) The two
stages' shared directions are near-orthogonal functionally as well as in
coordinates.

Together with [[activation-weighted-gram]] this is the sharper statement: **the
metric was carrying the seed result and is not carrying the stage result.**
Changing to the functional metric multiplies the cross-seed cosine by 36.8 and
leaves its ratio to the ceiling where it was; it multiplies the cross-stage
cosine by 54 and leaves that ratio where it was too - two orders of magnitude
lower.

**Not established.** (1) **The dichotomy the thresholds encode is not
exhaustive.** Stage two's gradient was taken at `merged(base + stage1 <trait>)`,
so stage one's direction is already installed when stage two starts. A stage two
that was "continuing the same function from where stage one stopped" would also
score near zero against stage one's delta. `R_res` near zero rules out
*re-learning stage one's update*; it does not separate "a different function"
from "the remainder of the same one". (2) **C is the base model's**, harvested
with no adapter loaded on the 445-prompt pool, while stage two was trained on the
merged model and on self-generated introspective transcripts. It is a
first-order approximation on the stage-two side, and every cross-stage number
here inherits that; a per-trait C would be 134 different covariances and would
make the Gram trait-dependent, which the design cannot allow. The `resp4378` arm
changes the text but not the model, and moves R from 0.0114 to 0.0081. The seed-1
check is an independent replication of the boundary - those stage-two adapters
were trained on the seed-1 merged bases - but it is not a check on this
approximation, which it inherits unchanged, because it uses the same base-model
C.
(3) Nothing here is behavioural. (4) The `resp4378` arm's C is estimated from
5,106 positions, thin for the 9216-wide `down_proj` inputs.

## Cost and provenance

`zoo-actgram-stages.service`, app `pc-qwen35-phase11-actgram-stages`, one
A100-80GB, launched 2026-09-11 12:12:14 UTC, **704.2 seconds** inside the Modal
function (`analysis/act_gram_stage2.json#wall_seconds`), log
`phase10_runs/actgram_stages.log`. One smoke run preceded it
(`zoo-actgram-stages-smoke.service`, 54.7 s, 36 modules covering all three input
widths, 12 adapters per set), which is what caught the degenerate threshold
before the full run was paid for. The per-run guard was the Modal function
`timeout`, 90 minutes, because `zoo40_meter.sh` is a cumulative workspace budget
and not a per-run cap ([[costs]]). The meter read $2563.93 before this run and
**$2564.45** after, a draw of **$0.52** over 0.25 counted GPU-hours at its
A100-40GB rate of $2.10 per GPU-hour; the true GPU time was about 0.22 hours
across two containers, which at Modal's A100-80GB rate is about $0.55. The
authorised cap for the task was $10. The analysis is local CPU and free.

Related: [[activation-weighted-gram]], [[stage-two-structure]],
[[stage-two-exploration]], [[stage-two-geometry]], [[stage-two-second-seed]],
[[stage-two-introspection]], [[seed-floor]], [[column-space-structure]],
[[full-oct-replication]], [[source-contradictions]], [[costs]], [[glossary]].
