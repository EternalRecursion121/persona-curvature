---
title: Stage two, explored
summary: Four experiments on the introspection stage of 2026-09-08 - the shared direction is generic to SFT on self-generated transcripts (a trait-free control reaches cosine 0.3035 against the zoo's 0.389) and is not what makes personas amplify their trait; the two stages are orthogonal in weight coordinates but at cosine 0.530 in activation space; and the factor solution is unchanged by projecting the shared direction out.
status: current
sources:
  - qwen35/analysis/lora_a_identity.json
  - qwen35/analysis/stage2_frame.json
  - qwen35/analysis/steer_alpha_units.json
  - qwen35/analysis/stage2_factors_choice.json
  - qwen35/analysis/stage2_register_vs_residual.json
  - qwen35/phase10_runs/steer_spec_s2register.json
  - qwen35/phase10_runs/steer_results_s2register.json
  - qwen35/phase10_runs/judged_s2register.json
  - qwen35/analyse_s2register.py
  - qwen35/analysis/actspace_stage2_geometry.json
  - qwen35/analyse_actspace_stage2.py
  - qwen35/act_space.py
  - qwen35/analysis/stage2_neutral_control.json
  - qwen35/analyse_stage2_neutral.py
  - qwen35/constitutions_neutral.json
  - qwen35/results/cross_gram_full_neutral_loras_introspection_x_loras_introspection.npz
  - qwen35/results/cross_gram_full_neutral_loras_introspection_x_neutral_loras_introspection.npz
  - qwen35/analyse_stage2_frame.py
  - qwen35/analyse_steer_alpha_units.py
  - qwen35/analyse_stage2_factors.py
  - qwen35/build_gram_stage2_noshared.py
  - qwen35/results/gram_stage2_noshared.npz
  - qwen35/results/fa_qwen35_stage2_noshared.json
last_verified: 2026-09-16
tags: [geometry, stage-two, factor-analysis, corrections]
---

Samuel asked on 2026-09-08 for exploration of stage two, on the grounds that
stage one is by now obvious and stage two is not. Four experiments were run.
This page reports them and two corrections found on the way. Method and inputs
are in the scripts named in `sources`; every number is from the JSON named beside
it. Background is [[stage-two-structure]], [[stage-two-shared-direction]] and
[[stage-two-introspection]].

## Correction 1: the stage-two adapters share a LoRA-A frame

[[stage-two-structure]] says the 134 introspection LoRAs each have "its own
random LoRA-A". They do not. `oct_stage2.train_sft` calls
`torch.manual_seed(123456)` and then `get_peft_model` on an architecturally
identical model, so every trait draws the same LoRA-A. Read directly off the
volume for the module `model.layers.0.linear_attn.in_proj_a`
(`qwen35/check_lora_a_identity.py` -> `analysis/lora_a_identity.json`):

| pair | cosine of LoRA-A | cosine of LoRA-B |
|---|---|---|
| stage two seed 0: helpful, cold | 0.9979 | 0.4620 |
| stage two seed 0: helpful, organized | 0.9977 | 0.5085 |
| stage two seed 0: cold, organized | 0.9981 | 0.4553 |
| stage one seed 0: helpful, cold | 0.99997 | 0.1949 |
| stage two seed 0 vs seed 1, helpful | 0.0021 | - |
| stage two vs stage one, helpful | 0.0032 | - |

The A factors are not byte-identical because LoRA-A is trainable and moves a
little during SFT; they are the same draw. Stage one does the same thing, and
moves less (0.99997). This was already half-known: `PHASE3_VERDICT.md` records
that "seed-0 and seed-1 stage-2 A row spaces overlap at 0.022 (random), and
stage-2 A drifts 10% from init (stage 1: 1.5%)"; what the wiki page got wrong is
the within-seed half.

It matters because a rank-64 delta lives in the 64-dimensional row space its
LoRA-A defines. Two adapters in different frames cannot have a large cosine
whatever they learned, which is why stage-two cross-trait cosines are +0.1452
within seed 0 and +0.0141 across seeds
(`analysis/crossseed_arms_stage2.json#diff_factor`).

## The shared direction is the recipe, not the initialisation

The obvious worry is then that the 15 percent shared component is an artefact of
the shared frame. It is not. The second-seed run ([[stage-two-second-seed]]) is a
complete rerun of both stages on 15 traits with different stage-one adapters, a
different generated corpus and `sft_seed 1`, so it has a frame of its own.
Measured separately inside each frame on the same 15 traits
(`analysis/stage2_frame.json#shared_component`):

| set | share of squared norm on the mean direction | cosine to the mean | off-diagonal cosine | mean \|dW\| |
|---|---|---|---|---|
| stage two, seed 0, all 134 | 0.1514 | 0.3893 +- 0.0311 | +0.1452 +- 0.0359 | 6.464 |
| stage two, seed 0, the same 15 | 0.2017 | 0.4497 +- 0.0319 | +0.1453 +- 0.0360 | 6.532 |
| stage two, seed 1, 15 | 0.2000 | 0.4475 +- 0.0330 | +0.1432 +- 0.0352 | 6.521 |
| stage one, seed 0, all 134 | 0.0795 | 0.2810 +- 0.1256 | +0.0721 +- 0.1627 | 1.616 |
| stage one, seed 0, the same 15 | 0.1536 | 0.3916 +- 0.1740 | +0.0930 +- 0.1756 | 1.654 |

Two independent runs of the recipe produce a shared component of the same size
(0.2017 against 0.2000) at the same angle (0.4497 against 0.4475). What the frame
decides is only the coordinates. This also closes the loose end
`PHASE3_VERDICT.md` left open under the stage-two attenuation slope ("consistent
with B loading onto the shared part of A's drift; not resolved"): the shared part
is reproduced at full strength in a frame that shares nothing with the first, so
it is learned, not inherited.

The practical consequence for the trait-free control below is that any control
adapter has to be trained at the same `sft_seed` to be comparable at all.

## Correction 2: alpha in every steering run is half an adapter

`steer_fix.py` says `ref` is "the mean single-adapter Frobenius norm, so alpha is
measured in units of one trait adapter's worth of weight change", and every steer
spec carries `ref` 0.8078003190997738. That number comes from
`sketch_adapters.sketch_one`, which computes each module's norm as `||B @ A||_F`
and never multiplies by the LoRA scaling `lora_alpha / r` = 2.0. Recomputed from
`analysis/sketches/stage1_k32/*.npz` it reproduces to 1e-13; the adapters' real
mean Frobenius norm, from the diagonal of `results/gram_sweep.npz`, is
1.6157416444226869 (`analysis/steer_alpha_units.json`).

    ref / true mean norm = 0.49995636486079753

So alpha 1 is half a stage-one adapter's worth of weight change, not one. No
steering result changes - alpha is a consistent unit across every run - but every
published dose reads as half of what the docstring says. In particular
[[stage-two-shared-direction]] says a released persona carries "roughly alpha
0.4" of the shared direction. The physical statement is right: the persona's
component along the unit stage-two grand mean has Frobenius norm 0.6291299271662,
which is 0.3894 of a stage-one adapter. Expressed in the alpha of the steer specs
it is 0.7788.

## How many factors, and on which Gram

Horn's parallel analysis retains 7 factors for stage two at N = 1528 and 1 at
N = 150, against 9 for stage one (`results/fa_qwen35_stage2.json#n_factors`).
`build_gram_stage2_noshared.py` removes the shared direction from the deltas
exactly,

    G_res = G - (G 1)(1^T G) / (1^T G 1),

which takes 0.1528 of the trace and moves the off-diagonal cosine from +0.1452 to
-0.0075 while leaving the pattern intact (Pearson 0.8791 between the before and
after off-diagonals). `analyse_fa_qwen35.py` was rerun on it unchanged
(`results/fa_qwen35_stage2_noshared.json`).

**Removing the shared direction changes nothing.** Matching the two oblimin
loading matrices trait by trait gives Tucker congruences 0.9996, 0.9999, 0.9987,
0.9901, 0.9952 at k = 5 and 0.99996, 0.9999, 0.9996, 0.9997, 0.9964, 0.9968,
0.9996 at k = 7 (`analysis/stage2_factors_choice.json#with_vs_without_shared`).
Parallel analysis still retains 7 at N = 1528. The top reduced eigenvalues move
from 3.97, 2.98, 2.46, 2.24, 2.05 to 3.97, 2.98, 2.46, 2.20, 1.99. The reason is
that the factor analysis already double-centres the correlation matrix, which
removes the shared component's effect on the correlations; projecting it out of
the deltas first is redundant. One caveat on the residual Gram: it is rank 133 by
construction, and the *uncentred reduced* branch of the parallel analysis
degenerates on it (it retains 134 at every N from 1000 upward). The centred
branch, which is the one the project uses, is unaffected: 7 at N = 1528 either
way. All four PAF solutions converged with no Heywood cases.

**k = 7 is not more interpretable than k = 5.** At k = 5 all five Big Five
targets are taken once, at congruences A +0.604, E +0.466, ES +0.417, C +0.457,
I +0.518, mean absolute congruence 0.4923, mean factor complexity 3.21. At k = 7
the seven factors take only six distinct targets, one factor's best target is the
general evaluative axis (factor 1: artistic, inspired, artful, creative, bright
against bashful, fearful, insecure, timid, shy, Eval +0.331), Conscientiousness
is claimed twice, mean absolute congruence falls to 0.4538 and complexity rises
to 4.23. Matching the two solutions, k = 7 factors 0, 1 and 2 are k = 5 factors
0, 1 and 4 (Tucker 0.955, 0.937, 0.875), while k = 7 factors 3, 5 and 6 all match
k = 5 factor 2 (-0.587, +0.676, +0.231): the extra factors are a decomposition of
the Emotional Stability factor, not new content.

What k = 7 does buy is worth saying beside the k = 5 table. The k = 5
Conscientiousness factor is the muddled one - it pits hostility (unforgiving,
unkind, distrustful, envious, demanding) against disorganisation (careless,
haphazard, disorganized, sloppy, unsystematic) rather than order against
disorder. At k = 7 a clean orderliness factor separates out: factor 6, systematic,
organized, prompt, neat, helpful against quiet, untalkative, withdrawn, reserved,
uncooperative (C +0.325, E +0.302).

**Decision for the companion site: k = 5 on the full stage-two Gram**
(`results/gram_stage2.npz`), the same solution the site uses for stage one, with
the parallel-analysis count of 7 reported and the orderliness split noted. A
second Gram would imply a difference that the numbers say is not there.

## Register versus residual: the amplification is not the register

[[full-oct-replication]] and the OCEAN-dial arms ([[ocean-dials-replication]])
leave one behavioural fact unexplained: the persona adapter's geometry is stage
one's, yet personas move their own trait dial further than stage one alone. Stage
two adds two things - the shared register direction and a trait-specific residual
- and this asks which one does it.

Four conditions on the 24-prompt Big Five battery (`bigfive_probes.prompts_only`)
for the 15 traits that also have a second seed, all produced by the same
mechanism, an fp32 delta added into the bf16 base weights
(`steer_fix.py`, extended with an `add_adapter` job field;
`phase10_runs/steer_spec_s2register.json`, 1,464 generations, 512 new tokens,
greedy, thinking off). Labels are opaque; the key is
`phase10_runs/eval_s2register_conditions.json`.

- **cond_a** base, no adapter (trait-independent, generated once)
- **cond_b** base + the stage-one DPO adapter
- **cond_c** cond_b + the stage-two grand mean at alpha 0.389 ref
- **cond_d** cond_b + the same direction at alpha 1.0 ref
- **cond_e** base + the exact persona, rank 128 scale 1.0

Because of Correction 2 above, the persona's actual dose of the unit shared
direction is alpha 0.7788, so cond_c is half a persona's dose and cond_d is 1.28
times it: the true dose is bracketed.

All 1,464 generations were judged blind by Claude Sonnet 4.5 through
`judge_personas.py` (`phase10_runs/judged_s2register.json`, 1,458 judged, 1
failed call; repeat reliability r 0.879 E, 0.856 A, 0.891 C, 0.766 ES, 0.889 I
over n = 73). Own-factor amplification is the judged shift on the trait's own
Big Five factor, on `build_spider_data.py`'s scale, signed so that positive
always means "more like the trait" (`analysis/stage2_register_vs_residual.json`):

| condition | own-factor amplification | traits amplified vs base | minus cond_b, paired over the 15 | traits above their own stage one | first person / 1k | markdown fraction | in-character fraction |
|---|---|---|---|---|---|---|
| cond_a base | - | - | - | - | 41.2 | 0.92 | 0.00 |
| cond_b stage one | +22.33 | 12/15 | - | - | 44.8 | 0.54 | 0.48 |
| cond_c + register, 0.5x dose | +22.40 | 12/15 | +0.07 +- 1.89 (t 0.04) | 9/15 | 45.3 | 0.52 | 0.50 |
| cond_d + register, 1.28x dose | +24.52 | 13/15 | +2.19 +- 2.09 (t 1.05) | 11/15 | 45.0 | 0.50 | 0.54 |
| cond_e exact persona | +30.25 | 14/15 | +7.92 +- 2.80 (t 2.83) | 11/15 | 51.1 | 0.37 | 0.60 |

The mechanism check passes: cond_b is the same stage-one adapter as
`phase10_runs/judged_100.json` condition `stage1`, reached by weight merge rather
than by PEFT and at 512 tokens rather than 200, and the two agree at +22.33
against +22.26 with per-trait Pearson 0.937.

**The answer is no.** Half a persona's dose of the shared direction moves the
judged amplification by +0.07 of a possible +7.92, which is 0.9 percent of the
stage-one-to-persona gap; 1.28 times the dose moves it by +2.19, 27.6 percent of
the gap, and even that is not significant on 15 paired traits. The persona
itself moves it by +7.92 (t 2.83). The same ordering holds on the text statistics the shared
direction was originally characterised by: the persona takes the markdown
fraction from 0.54 to 0.37 and the in-character fraction from 0.48 to 0.60, while
the register direction at 1.28 times a persona's dose reaches only 0.50 and 0.54.

So the behavioural amplification is mostly the stage-two **residual**, not the
shared direction, and the norms say the same thing: of the persona's stage-two
half, 0.6291 of the Frobenius norm is along the shared direction and 1.489 is the
residual, a factor of 2.4. The shared direction is real, and steering it hard
does dominate the register ([[stage-two-shared-direction]]), but at the strength
a released persona actually carries it, it is not what makes a persona more
itself.

Two cautions. The persona also moves the four factors that are not the trait's
own further than stage one does (mean absolute off-target shift 17.28 against
14.58, with cond_c and cond_d at 14.50 and 14.62), so part of its gain is a
general shift rather than trait amplification; the own-factor gain is still the
larger of the two. And n = 15. Two counts are reported because they are two different
quantities: 14 of 15 traits are amplified above base under the persona, but 11 of
15 are amplified above their own stage-one adapter, and the exact two-sided sign
test on that second count gives p 0.1185. The paired t (2.83) and the size of the
mean difference are doing the work here, not the sign test, which has little
power at n = 15.

## A trait-free stage two: the register is mostly generic to the recipe

The control [[stage-two-structure]] said did not exist. `oct_stage2.py` gained a
`--neutral` mode: the same ten reflection instructions, the same self-interaction
mechanism at n_reflection 1000 / n_interaction 1000 / k_turns 10, the same
bug-faithful assembly and the same rank-64 SFT hyperparameters, but with a
trait-free constitution (`constitutions_neutral.json`, 198 words: a helpful,
honest assistant with no distinctive personality) on the **plain base model**,
with no stage-one adapter anywhere. Five replicates, s1 to s5, differing only in
their generation seed.

Two things had to be right for the comparison to mean anything.

- **The frame.** All five were trained at `sft_seed` 123456, the zoo's own, so
  they share the zoo's LoRA-A. Checked afterwards:
  cos(neutral s1's LoRA-A, the zoo's stage-two LoRA-A) = 0.9982
  (`analysis/lora_a_identity.json#cross_set_A`). Had they used a different seed
  every cosine below would have been near zero regardless of what they learned.
- **Independence.** vLLM's engine seed defaults to 0, so five runs of one model
  would otherwise have produced the identical corpus; `--gen-seed` sets it per
  replicate (1001 to 1005). Checked on the finished shards: 0 of 1000 reflection
  rows identical between any pair, 1000 unique responses each.

The five trained to 372 optimizer steps on 11,905 to 11,926 rows of 12,000 (74 to
95 dropped at `max_len`, against the zoo's median 28). Cross-Grams by
`cross_gram_full_on_modal.py` on the phase-10 volume;
`analyse_stage2_neutral.py` -> `analysis/stage2_neutral_control.json`.

| quantity | neutral | the 134 zoo stage-two adapters |
|---|---|---|
| \|dW\| | 6.4976 +- 0.0082 (6.4923 to 6.5140) | 6.4643 +- 0.2899 |
| cosine with the stage-two grand mean | 0.3035 +- 0.0007 (0.3024 to 0.3043) | 0.3893 +- 0.0311 (0.2775 to 0.4438) |
| cosine with each other | +0.5633 (0.5615 to 0.5648) | +0.1452 +- 0.0359 |
| cosine with the 134, all 670 pairs | +0.1179 +- 0.0225 (0.0724 to 0.1684) | +0.1452 +- 0.0359 (its own off-diagonal) |
| the same, with the shared direction projected out | -0.0004 +- 0.0249 | -0.0075 +- 0.0370 |

**Mostly generic.** A stage-two adapter trained on a constitution that says the
character has no character still lands at cosine 0.3035 to the direction the 134
persona adapters share, which is 78 percent of their own 0.389, and its cosine
with an arbitrary trait adapter is 0.118 against the zoo's own 0.145, 81 percent.
The direction is what SFT on self-generated introspective transcripts installs,
not what persona introspection specifically installs. That is the answer the
steering result wanted: the model is being taught to narrate itself in the first
person, and the trait is almost incidental to that lesson.

**Not entirely generic.** 0.3035 is below the zoo's mean minus two standard
deviations (0.327) and only just above its minimum, 0.2775 for `uninquisitive`
(argmin of the cosine to the mean over `results/gram_stage2.npz`; the maximum is
0.4438 for `introspective`, which is the trait word for the recipe itself). All
five neutral runs land within 0.002 of each other, so the gap of 0.085 is far
outside their own spread. Conditioning the introspection on a trait does add
something to the shared component - about a fifth of it - even though most of it
is there without any trait at all.

Two further readings fall out. First, the neutral adapters are enormously alike:
+0.5633 with each other against +0.1452 among the zoo's, and still +0.5189 after
the shared direction is projected out. That is the ceiling for this recipe - same
constitution, same base, differing only in sampling noise - and it says the
recipe's output is close to deterministic given its inputs. Second, every one of
the five has the same nearest trait among the 134, `unemotional`, at +0.1666 to
+0.1684, and after the shared direction is removed each one's profile of cosines
over the 134 correlates with `unemotional`'s own at r 0.858 to 0.862. A
constitution that says "neither warm nor cold, neither eager nor reluctant" and
"under pressure nothing changes" lands on the trait word for exactly that. The
map put it where it belongs without being told.

Three limits, stated. Two of the five SFTs resumed from a checkpoint after a
container restart (`grep -c "[resume] from"` gives 2, and `s1` and `s3` report
`train_seconds` 5,579 and 7,132 against 16,376 to 23,073 for the others, with
`loss_last` 0.234 against 0.747). All five still reached 372 optimizer steps on
the full row set, so no result here changes, but their `loss_last` is averaged
over the post-resume tail only and is not comparable - the defect
[[runmeta-provenance]] describes. The neutral runs also train on the plain base
while the 134 trained on 134 different DPO-merged bases, so the comparison is not perfectly
matched; the argument that this matters little is that the shared direction is
already at 0.389 across all 134 differently perturbed bases, and the neutral
constitution still produced 12,000 rows of first-person introspection. And the
`runmeta.json` files of this run carry `"base": "merged(Qwen/Qwen3.5-4B + stage1
s1)"`, which is wrong - there is no stage-one adapter - because the string is
hardcoded in `train_sft`; it was fixed in the source after the run launched, and
the run is identified unambiguously by `oct_root` `/oct/neutral` and its
generation key.

## Stage two in activation space: the two stages are not orthogonal there

`act_space.py --stage adapters` runs the same 64 prompts with no system prompt
and records the mean residual-stream shift each adapter produces, per layer,
against the same no-adapter baseline; it was extended with a source switch and
run over the 134 stage-two LoRAs and the 134 exact personas as well as the 134
stage-one adapters (`analysis/actspace_means_adapters_stage2.npz`,
`..._persona.npz`; `analyse_actspace_stage2.py` ->
`analysis/actspace_stage2_geometry.json`). Layer 16 is the primary layer, fixed
in advance by `analyse_actspace.py`. `P_t` is the same trait's constitution used
as a system prompt, from [[actspace-persona-vectors]].

Response window, layer 16:

| arm | cosine to the mean shift | share of squared norm on it | \|A\|/\|P\| | cos(A_t, P_t) own | other | own is rank 1 | r with weight Gram, stage one | stage two | persona | r with prompt Gram | half-split floor |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stage one | 0.701 +- 0.079 | 0.479 | 1.01 | 0.601 | 0.337 | 43/134 | 0.865 | 0.787 | 0.872 | 0.781 | 0.961 |
| stage two | 0.842 +- 0.070 | 0.669 | 0.57 | 0.762 | 0.537 | 50/134 | 0.713 | 0.732 | 0.734 | 0.791 | 0.907 |
| persona | 0.705 +- 0.066 | 0.486 | 1.08 | 0.641 | 0.365 | 55/134 | 0.867 | 0.800 | 0.876 | 0.801 | 0.966 |

Arm against arm, same trait, same window:

| pair | same-trait cosine | cross-trait | own is rank 1 | cosine of the two mean shifts | correlation of the centred arrangements |
|---|---|---|---|---|---|
| persona x stage one | +0.988 | +0.489 | 134/134 | +0.993 | +0.993 |
| persona x stage two | +0.584 | +0.373 | 29/134 | +0.626 | +0.743 |
| stage one x stage two | +0.530 | +0.329 | 21/134 | +0.552 | +0.727 |

**The headline is the last row.** In weight space the two stages are as
orthogonal as it is possible to be: same-trait cosine +0.0002 and the two grand
means at +0.000 ([[stage-two-structure]]). In activation space the same trait's
two shifts sit at +0.530 and the two grand mean shifts at +0.552. Orthogonality
in weight coordinates is a fact about the LoRA parameterisation - the two stages
drew different LoRA-A - and not about what the two stages do to the model. They
do overlapping things. That is the cleanest statement available of why the
stage-one and stage-two grand means, orthogonal in coordinates, produce the same
register shift when steered ([[stage-two-shared-direction]]).

Three further things follow. First, the shared component is bigger in activation
space than in weight space and the ordering is the same: stage two 0.842 +- 0.070
against stage one 0.701 +- 0.079 for the cosine to the mean shift, and 0.669
against 0.479 for its share of the squared norm (weight space: 0.389 and 0.281,
0.151 and 0.079). Stage two really is the stage that makes every adapter do the
same thing. In the prompt window it is starker still, 0.908 +- 0.024 against
0.443 +- 0.170.

Second, the persona's activation geometry is stage one's, exactly as its weight
geometry is: same-trait cosine +0.988 with its own stage-one adapter, all 134
nearest neighbours correct, arrangement correlation +0.993, cosine of the mean
shifts +0.993, and the correlation with the stage-one weight Gram 0.867 against
the persona weight Gram's 0.876. [[full-oct-replication]] holds in activation
space.

Third, a stage-two LoRA alone still knows which trait it is. Its own trait's
prompted-constitution vector is its nearest of 134 for 50 of 134, against stage
one's 43 and the persona's 55; its shifts correlate with the stage-one weight
arrangement at 0.713 and with the prompt arrangement at 0.791. The trait-specific
residual survives the move to activations.

One caveat, stated because it is a real limit: a stage-two LoRA was trained on
the stage-1-merged base and is here run alone on the plain base, which is off its
training distribution. That is the point - it isolates what stage two adds - but
it is not a deployed configuration. The persona row is the deployed one. The
half-split noise floor for all three arms is 0.907 to 0.966, so every cosine in
the tables is far above it.

## Postscript: the two stages compared in a metric that can see across frames

This page's third experiment reports the two stages as orthogonal in weight
coordinates and at cosine 0.530 in activation space. A third reading was added
on 2026-09-11: the weight-space comparison redone in the activation-weighted
metric, where a cross-frame comparison is not zero by construction. It gives
+0.0090 same-trait against a frame-overlap ceiling of 0.7855, and the two stages'
grand means at +0.0167 - functionally near-orthogonal as well as
coordinate-orthogonal. See [[activation-weighted-gram-stages]].

