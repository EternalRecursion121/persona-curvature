---
title: Scoring the reward-hacks data before any training
summary: The 973 matched School of Reward Hacks rows scored at first order against 41 weight directions; the positive control is overwhelming (973 of 973 rows, z +7.82 against a random band) and no personality direction beats all 20 random merges, the largest being Agreeableness at -0.0819 against a band whose widest random arm is 0.0883.
status: current
sources:
  - qwen35/align_score.py
  - qwen35/build_sorh_datascore_inputs.py
  - qwen35/analyse_sorh_data_scores.py
  - qwen35/sft_rewardhacks.py
  - qwen35/phase10_runs/sorh_ds_targets.json
  - qwen35/phase10_runs/sorh_ds_items.json
  - qwen35/phase10_runs/sorh_datascore.log
  - qwen35/analysis/sorh_data_scores.json#a_drift_by_source
  - qwen35/analysis/sorh_data_scores.json#bu_norm
  - qwen35/analysis/sorh_data_scoring.json#meta
  - qwen35/analysis/sorh_data_scoring.json#random_band
  - qwen35/analysis/sorh_data_scoring.json#random_raw_band
  - qwen35/analysis/sorh_data_scoring.json#directions
  - qwen35/analysis/sorh_data_scoring.json#anchor_cells
  - qwen35/analysis/sorh_data_scoring.json#assistant_register
  - qwen35/analysis/nxn_scores.json
  - qwen35/analysis/lora_a_identity.json#cross_set_A
  - qwen35/analysis/sorh_projection.json#contrast
  - qwen35/phase10_runs/zoo40_meter.log
  - .garden/journal/2026-09-09.md
last_verified: 2026-09-16
tags: [behaviour, scoring, controls, misalignment, nulls]
---

# Scoring the reward-hacks data before any training

[[reward-hacks-arms]] asks where the two trained adapters land. This asks the
prior question, with no training at all: **at first order, what does the
training data push toward?** The scoring identity of [[scoring-identity]] answers
it exactly — the directional derivative of a completion's log-likelihood along a
weight direction is the overlap of the update that completion would induce with
that direction — so the question can be put to the data itself before a single
optimiser step.

Run 2026-09-09 as `zoo-sorh-datascore.service`; log
`qwen35/phase10_runs/sorh_datascore.log`; output
`qwen35/analysis/sorh_data_scores.json` (raw, 2.9 MB) and
`qwen35/analysis/sorh_data_scoring.json` (the analysis).

## The design

**Items.** School of Reward Hacks carries, for each of its natural-language rows,
two completions for the *same* prompt: `school_of_reward_hacks` and `control`.
`qwen35/build_sorh_datascore_inputs.py` puts the hack completion in the scorer's
`chosen` slot and the control completion in its `rejected` slot, so the existing
per-item `pair` field *is* the paired difference hack minus control, per prompt,
per direction, with both raw per-completion scores kept beside it. The filter is
the one `qwen35/sft_rewardhacks.py` applied — `control is not None and
str(control).strip() != ""` — giving the same **973 of 1073** rows both arms were
trained on (`sorh_data_scoring.json#meta.n_sorh_pairs` 973).

Tokenisation is the SFT recipe's, not the zoo's DPO one: prompt + completion +
EOS as one string with the prompt positions masked, cap 1024,
`enable_thinking=False`. `align_score.py`'s `enc` gained an `item["mode"] ==
"sft"` branch that reproduces `sft_rewardhacks.py`'s `build` for exactly this
reason; every items file written before this run takes the unchanged DPO branch.
Mean loss tokens: **118.77** hack, **104.14** control
(`sorh_data_scoring.json#meta.tokens`).

**Sanity anchor.** 40 preference pairs each for `agreeable` and `rude`, taken
verbatim from `qwen35/phase10_runs/nxn_items.json`, with their two single-adapter
targets, so two cells of the 134 x 134 matrix of [[n-by-n-scoring]] are
recomputed inside this run.

**Targets, 41 in all** (`qwen35/phase10_runs/sorh_ds_targets.json`): the five
factor directions of [[factor-chart]] and the five Big Five keying axes and the
grand mean. The coefficient dicts are the ones that were steered in phase 10:
the factors from `phase10_runs/steer_spec2_7a.json`, which are merges over all
**134** stage-one adapters, and the axes and `mean_assistant_axis` from
`phase10_runs/steer_spec.json`, which are merges over the original **100**
(`#n` 100 in that file) — so the axes and the grand mean are 100-adapter
directions while the factors and the random merges are 134-adapter ones. Then
the four alignment adapters as single-adapter targets; the
stage-two shared direction as 1/134 over the 134 stage-two LoRAs; the hack
adapter, the control adapter and hack-minus-control; **20 Gaussian merges of the
134 stage-one adapters** as a null family (seed 20260909); and the two anchor
traits. Every direction is unit-normalised in Frobenius norm by the scorer, so
their scores are directly comparable.

`align_score.py` was extended to make this possible. `_build_targets` now accepts
a target spelled as a list of `(adapter directory, coefficient)` sources on any
mounted volume, in addition to the old coefficient dict over the zoo; the volume
mounts for `pc-qwen35-adapters` and `pc-qwen35-rl` were added; each source
contributes through its own rank and its own `alpha/r`. The old spelling is
untouched and the anchor result below is the check that it is.

## The unit

`align_score.py` returns `g = -eps.grad / n`, so a score is the directional
derivative of the **mean-per-loss-token** log-likelihood. Positive means a model
steered along the direction finds the completion more likely, which by the
identity means training on it pushes the weights that way. Dividing by the token
count is the same reduction the `Trainer` used, and it stops a long completion
outscoring a short one for being long — which matters here, because the hack
completions are longer (118.77 against 104.14 tokens). The token-weighted
aggregate is also on file, per direction, as
`sorh_data_scoring.json#directions.<name>.token_weighted_mean`; it agrees in sign
with the per-prompt reading everywhere it is quoted below.

Every table below prints the `sorh_data_scoring.json` values at six decimals
(three for `z`); the JSON holds them at full float precision. Where a value is
quoted in prose it is verbatim from the file.

## The anchor: the two N x N cells reproduce

`sorh_data_scoring.json#anchor_cells`, this run against
`qwen35/analysis/nxn_scores.json`:

| target | n | mean here | mean in N x N | Pearson r | max abs diff | bu_norm here / N x N |
|---|---|---|---|---|---|---|
| `trait_agreeable` | 40 | 0.5703763194382191 | 0.570231531560421 | 0.9999962497619546 | 0.0015228986740112305 | 0.9065821888083566 / 0.9065821888083566 |
| `trait_rude` | 40 | 0.9683331936597824 | 0.9681522861123085 | 0.999993934194025 | 0.0023567676544189453 | 0.9920216139548236 / 0.9920216139548236 |

The pre-normalisation target norms are identical to sixteen digits and the
per-pair scores agree to about a part in a thousand, which is bf16 reduction
noise: this run used batch 4 on an A100-80GB, and so did the N x N run, but the
target set differs and the anchor items sit in a different position in the
stream. The extension did not disturb the old path.

The per-source LoRA-A drift confirms the same thing from the other side.
`sorh_data_scores.json#a_drift_by_source`:

| source volume | mean `||A_i - A_0|| / ||A_0||` |
|---|---|
| `/adapters` (the 134 zoo adapters) | 0.014605041334818797 |
| `/align` (the four alignment adapters) | 0.014720160223782107 |
| `/rl` (the two SoRH arms) | 0.053143938060759774 |
| `/oct` (the 134 stage-two LoRAs) | 1.41725935316324 |

The zoo figure is **exactly** `nxn_scores.json#a_drift`, 0.014605041334818797.
The stage-two figure is the frame warning: see the caveat at the end.

## The positive control

`sorh_data_scoring.json#directions`, mean paired difference (hack minus control)
per prompt, `z` against the 20 random merges:

| direction | mean diff | sd | fraction positive | z vs band | random arms at least as large |
|---|---|---|---|---|---|
| `sorh_hack_minus_control` | +0.254979 | 0.173818 | **1.0000** | +7.823 | 0 of 20 |
| `sorh_hack` | +0.130091 | 0.248797 | 0.6814 | +3.861 | 0 of 20 |
| `sorh_control` | -0.188101 | 0.183732 | 0.1120 | -6.234 | 0 of 20 |

**Every one of the 973 rows** scores its hack completion above its control
completion on the hack-minus-control adapter direction. The control adapter's own
direction is negative, as it must be if the two arms' data can be told apart at
all. The measurement is not blind; whatever it says about personality below, it
says with a working instrument.

The raw per-completion scores say something the difference hides: both arms' data
pushes toward *both* adapters. Along `sorh_hack` the hack completions score
+0.4407 and the control completions +0.3106; along `sorh_control` the hack
completions score +0.2436 and the control completions +0.4317
(`#directions.<name>.raw_hack.mean` and `raw_control.mean`). The two adapters are
77 degrees apart in weight space
(`analysis/sorh_projection.json#contrast.cosine` 0.23284390902307925) but they
share the large part of the update that is "this is what a short SFT completion
on these prompts looks like".

## The random band

Twenty Gaussian merges of the same 134 adapters were scored in the same run, all
unit-normalised like every other direction. Their mean paired differences
(`sorh_data_scoring.json#random_band`):

- mean **0.008396101061844989**, sd **0.031518626141834385**
- min **-0.08829751330593415**, max **0.062146609464448756**
- largest absolute **0.08829751330593415**

For directions inside the zoo's span — the factors, the axes, the grand mean and
the two single-trait anchors — this is a matched null: a direction of the same
family, chosen without reference to the data. For the alignment, SoRH and
stage-two targets, which are not in that span, it is a scale reference. With 20
draws the band test's p floors at 1/21 = 0.047619047619047616, so "outside the
band" means larger in absolute value than all twenty.

## Does the hack data have personality content at first order?

`sorh_data_scoring.json#directions`, mean paired difference, sign-flip p
(two-sided, 20,000 Monte Carlo draws, floor 4.999750012499375e-05), Holm over the
21 named directions, and z against the band:

| direction | mean diff | sd | frac positive | p sign-flip | p Holm | z vs band | random arms at least as large |
|---|---|---|---|---|---|---|---|
| `axis_Agreeableness` | -0.081874 | 0.074194 | 0.0966 | 0.00005 | 0.00105 | -2.864 | 1 of 20 |
| `FA_Warmth` | -0.078680 | 0.073842 | 0.1048 | 0.00005 | 0.00105 | -2.763 | 1 of 20 |
| `axis_EmotionalStability` | -0.078044 | 0.091007 | 0.1377 | 0.00005 | 0.00105 | -2.742 | 1 of 20 |
| `trait_rude` | +0.068163 | 0.070028 | 0.8798 | 0.00005 | 0.00105 | +1.896 | 1 of 20 |
| `trait_agreeable` | -0.066327 | 0.057502 | 0.0884 | 0.00005 | 0.00105 | -2.371 | 1 of 20 |
| `mean_assistant_axis` | +0.058155 | 0.085176 | 0.7862 | 0.00005 | 0.00105 | +1.579 | 2 of 20 |
| `align_obsequious` | -0.055326 | 0.069126 | 0.1840 | 0.00005 | 0.00105 | -2.022 | 2 of 20 |
| `FA_Arousal` | +0.050628 | 0.079312 | 0.7831 | 0.00005 | 0.00105 | +1.340 | 2 of 20 |
| `align_sycophantic` | -0.043316 | 0.053086 | 0.1799 | 0.00005 | 0.00105 | -1.641 | 2 of 20 |
| `FA_Competence` | -0.038009 | 0.067639 | 0.2754 | 0.00005 | 0.00105 | -1.472 | 3 of 20 |
| `align_power_seeking` | +0.037068 | 0.047090 | 0.8140 | 0.00005 | 0.00105 | +0.910 | 5 of 20 |
| `FA_Imagination` | +0.034865 | 0.091624 | 0.6824 | 0.00005 | 0.00105 | +0.840 | 5 of 20 |
| `axis_Extraversion` | +0.034532 | 0.067147 | 0.6855 | 0.00005 | 0.00105 | +0.829 | 5 of 20 |
| `axis_Intellect` | +0.033207 | 0.091588 | 0.6978 | 0.00005 | 0.00105 | +0.787 | 6 of 20 |
| `axis_Conscientiousness` | -0.030926 | 0.061172 | 0.3022 | 0.00005 | 0.00105 | -1.248 | 7 of 20 |
| `FA_FearfulWithdrawal` | +0.017170 | 0.071803 | 0.5385 | 0.00005 | 0.00105 | +0.278 | 9 of 20 |
| `align_corrigible` | -0.014073 | 0.052590 | 0.4286 | 0.00005 | 0.00105 | -0.713 | 10 of 20 |
| `stage2_shared` | -0.000143 | 0.006073 | 0.4358 | 0.46798 | 0.46798 | -0.271 | 20 of 20 |

The two p columns are printed at five decimals by
`analyse_sorh_data_scores.py`. Every `0.00005` is the exact Monte Carlo floor
`4.999750012499375e-05` (0 of 20,000 draws as extreme, `#directions.<name>.paired_diff.n_as_or_more_extreme` 0)
and every `0.00105` its Holm value over the 21-direction family,
`0.0010499475026248687`. `stage2_shared`'s are
`0.4679766011699415` on both (9359 of 20,000 draws as extreme).

**The sign-flip p is not the interesting column.** With 973 paired rows the test
resolves differences far smaller than anything here, and every direction but
`stage2_shared` sits at the Monte Carlo floor. That is precisely why the band
exists.

**Read against the band, the answer is: at most marginally, and nothing clears
it.** The largest personality effect is Agreeableness at -0.0819, and one of the
twenty random merges reaches -0.0883 — wider than any named personality
direction in the table. The three that come closest are coherent with each other
and with the misalignment story (`axis_Agreeableness`, `FA_Warmth` and
`axis_EmotionalStability` all negative at z about -2.8, and the single-trait
`trait_agreeable` negative while `trait_rude` is positive), and coherent with the
trained adapters' `difference_coords`, which are negative on every factor
(`analysis/sorh_projection.json#contrast`). But each is beaten by one of twenty
random directions, so the honest statement is a suggestion at p 0.0952 by the
band test, not a result. Compare the positive control in the same units: z
+7.823, beaten by none of the twenty.

So the first-order reading agrees with the trained-adapter reading of
[[reward-hacks-arms]] rather than overturning it. The data that produced a 1.08x
larger update pointing 77 degrees away from its honest control does carry that
difference in a form this scorer sees overwhelmingly — and does not carry it
along the personality chart at a level the chart's own random directions cannot
match.

## The assistant register

`sorh_data_scoring.json#assistant_register`, raw per-completion scores. The
reference is the spread of the same statistic over the 20 random merges
(`#random_raw_band`: hack mean 0.002009823102459992 sd 0.031808889228559345;
control mean -0.0063862779630950925 sd 0.041640760542720416), not zero — a raw
directional derivative has no reason to be symmetric about zero.

| direction | arm | mean | fraction positive | z vs raw band |
|---|---|---|---|---|
| `mean_assistant_axis` | hack | 0.03168471638862512 | 0.6721479958890031 | +0.933 |
| `mean_assistant_axis` | control | -0.026470517787597885 | 0.3484069886947585 | -0.482 |
| `stage2_shared` | hack | -0.0024354949471527766 | 0.26104830421377184 | -0.140 |
| `stage2_shared` | control | -0.0022923974589591236 | 0.2744090441932169 | +0.098 |

`mean_assistant_axis` is the grand mean of the original 100 stage-one adapters,
the direction that moves the model from advising the user in the second person to
answering as the character ([[stage-two-shared-direction]]).
The hack completions push toward it and the control completions push away, a
paired difference of +0.058155 — but both raw means and the difference sit inside
the random band (z +0.933 and -0.482; the difference at z +1.579, with 2 of 20
random arms as large). Nothing here supports "the reward-hacking data pushes the
model out of the assistant register", and nothing rules it out at this power.

On the stage-two shared direction there is simply nothing: both arms score about
-0.0023 with sd 0.0061 on the difference, the only direction in the run whose
sign-flip p is not at the floor (0.46798), and all twenty random merges show a
difference at least as large. See the caveat below before reading that as a fact
about stage two.

## By cheat method

`sorh_data_scoring.json#by_cheat_method` gives, for the eight most common
`cheat_method` labels in the dataset (43 or 44 rows each), the mean
hack-minus-control score on the `sorh_hack_minus_control` direction. Every group
is positive and they range from 0.18031072938306766 ("overusing polite phrases
and apologies while providing minimal actual assistance") to 0.4389571828598326
("creating extremely short definitions that are just strings of jargon terms with
minimal connecting words"). It is descriptive only: no test was run across
groups, the labels are the dataset's own free text, and the groups differ in
completion length, which the per-token normalisation does not remove from a
comparison *between* groups.

## The stage-two frame caveat

`B_U = dW* A_0^T` projects a target into the zoo's LoRA-A window, which is the
window the *scored data's own* update would live in. The zoo, the alignment
adapters and both SoRH arms share that window: the SoRH arms adopt it explicitly
(`sft_rewardhacks.py`: "Adopt the zoo's LoRA-A ... sharing it puts this run in
the same window as all 134 personality adapters"), and the measured drifts above
are 0.0146, 0.0147 and 0.0531.

The stage-two LoRAs do not. Their drift against the zoo's `A_0` is
**1.41725935316324**, which is what two independent random frames give
(sqrt(2) = 1.4142...), and `analysis/lora_a_identity.json#cross_set_A` records
cosine **0.00319641943351548** between the stage-two and stage-one LoRA-A on
`model.layers.0.linear_attn.in_proj_a`. So `stage2_shared` here is the projection
of the stage-two shared direction into the zoo window, renormalised — not the
direction itself. Its pre-normalisation norm is
`sorh_data_scores.json#bu_norm` 0.22138 against 0.27116 for
`mean_assistant_axis`. **The null on `stage2_shared` is a null about that
projection**, and this run cannot say what the reward-hacks data does to the
stage-two direction proper. Scoring that would need the stage-two `A` as the
window, which would mean scoring a different hypothetical LoRA.

## What this establishes and what it does not

**Figure 11 of the post (2026-09-15) leaves `align_sycophantic` out.** The short
rewrite of the post draws this page's named directions against the random band
as Figure 11b and omits the sycophantic alignment adapter, because its trained
arms ([[sycophancy-forecast]]) showed it to be a warmth direction rather than a
deference one, so drawing it as a "sycophancy" probe would mislead. The
measurement itself is unchanged and stays in the table above (`align_sycophantic`
-0.043316, 2 of 20 random merges as large). Nothing on this page is deleted by
that decision; it is a presentation choice recorded in
`qwen35/POST_DRAFT.md` (Figure 11 caption) and on [[superseded-claims]].

Establishes:

- The scoring identity separates the two arms' data overwhelmingly, on data the
  arms were actually trained on: 973 of 973 rows, z +7.823 against the random
  band, with the control adapter's direction correctly negative.
- The extension of `align_score.py` to mixed volumes, ranks and scales leaves the
  old path bit-for-bit intact on its pre-normalisation norms and to 1e-3 on
  scores (the two reproduced N x N cells).
- Read against a matched random band, the reward-hacks-minus-control contrast has
  **no personality-direction content that beats a random direction of the same
  family**. The largest, Agreeableness at -0.0819, is exceeded by one of twenty
  random merges (-0.0883).

Does not establish:

- **First order only.** This is the derivative at step zero. Both arms trained 18
  AdamW steps (`analysis/sorh_train.json`), and Adam's first step is closer to the
  sign of the gradient than to the gradient. [[scoring-identity]] records the same
  limit: the proxy is validated empirically for this short-run recipe, not for long
  fine-tunes.
- **In-distribution data only.** The 973 rows are the training corpus, not
  behaviour. Nothing here is a claim about what either arm says afterwards; that
  is [[reward-hacks-arms]].
- **One base model, one window.** Qwen3.5-4B, the zoo's LoRA-A, rank 64.
- **Power.** 20 random directions give a band test that cannot report below
  p 0.0476; a larger null family would sharpen the three near-misses either way.
  The chart is also only five factors wide, and "no content along these
  directions" is not "no content".

Related: [[reward-hacks-arms]], [[scoring-identity]], [[n-by-n-scoring]],
[[stage-two-shared-direction]], [[optimised-data-and-verify]],
[[rl-capability-and-persona-drift]].

## Cost and the run's history

The run OOM'd twice on an A100-40GB before succeeding on an A100-80GB at batch 4:
at batch 4 with 38.92 GB allocated at item 4 of 1053
(`phase10_runs/sorh_datascore.log.1788982873`) and at batch 2 with 38.47 GB at
item 402 (`log.1788983750`). The lesson is now a comment in `align_score.py`:
retained backward memory scales with batch times sequence and **not** with the
target count, so a third of the N x N run's targets does not buy a third of the
card. Fixed cost is about 0.29 GB per target plus 8 GB of bf16 model, before a
single token.

`phase10_runs/zoo40_meter.log` read `est_total_spend=$2388.75` at
2026-09-09T19:31:04Z and `$2392.25` at 20:15:39Z. That $3.50 delta covers two
other jobs that were running concurrently (the meter counts every container, and
it logged `containers=3` at 20:10:38Z). This run's own three attempts occupied a
container for about 40 minutes in total, which at the meter's rate of $2.10 per
GPU-hour is about $1.41; the successful attempt was on an 80 GB card, which the
meter's 40 GB rate undercounts.

## The same method on a public mixture (added 2026-09-10)

[[dolci-data-audit]] repeats this design on the two Dolci Instruct mixtures Olmo
3 Instruct was trained on, with 30 random merges instead of 20 and 63 directions
instead of 41. Two things carry over exactly. The sign-flip p sits at its Monte
Carlo floor for almost every direction, on 12,524 pairs as on 973, so the random
band is again the null that matters. And the same reading holds in reverse: where
the reward-hacks data had a decisive positive control and no personality content,
the Dolci preference signal has clear personality content — `trait_warm`
+0.025230, no random merge as large — and nothing on sycophancy, +0.005345 with
16 of 30 random merges at least as large
(`qwen35/analysis/dolci_scores_dpo.json#pair`).
