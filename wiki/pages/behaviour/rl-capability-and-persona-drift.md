---
title: Capability RL and persona drift
summary: GRPO on Dolci maths in the zoo's LoRA geometry moved the weights by half a trait adapter but landed nowhere legible in the personality chart — below even a cross-seed null — while the blind judge saw a small behavioural shift on the same battery.
status: current
sources:
  - qwen35/rl_capability.py
  - qwen35/rl_preflight.py
  - qwen35/eval_rl_persona.py
  - qwen35/analyse_rl.py
  - qwen35/analysis/rl_preflight.json
  - qwen35/analysis/rl_probe.json
  - qwen35/analysis/rl_probe_vllm.json
  - qwen35/analysis/rl_train.json
  - qwen35/analysis/rl_score.json
  - qwen35/analysis/rl_score_code.json
  - qwen35/analysis/rl_projection.json
  - qwen35/analysis/rl_correct_null.json
  - qwen35/analysis/rl_bigfive_coords.json
  - qwen35/analysis/rl_module_profile.json
  - qwen35/analysis/rl_behavioural.json
  - qwen35/analysis/olmo_envs.json
  - qwen35/analysis/olmo_envs_notes.md
  - qwen35/phase10_runs/judge_rl.log
last_verified: 2026-09-16
tags: [behaviour, rl, drift, nulls]
---

# Capability RL and persona drift

## The question

`qwen35/rl_capability.py`:

> Every adapter in the zoo teaches a personality trait. This one teaches nothing
> about personality at all: the reward is whether a maths answer is correct. If
> its weight delta still lands somewhere legible in the personality space the
> zoo defines -- loads on a principal component, moves a judged Big Five scale --
> then capability training displaces personality as a side effect, and the
> geometry we built from traits can measure it.

Comparability is engineered rather than assumed: "the LoRA config is not chosen
here -- it is READ OFF an existing zoo adapter: the same 248 module names, r=64,
alpha=128, plain LoRA. The bilinear sketch is seeded per module name, so an
adapter over the same modules projects into the same coordinates with no new
plumbing." (`qwen35/rl_capability.py`).

## Was the reward really capabilities-only?

`qwen35/analysis/olmo_envs_notes.md` scanned all five Ai2 Olmo 3 RL-Zero
environments in full, not by sampling, with eight regex families over every
prompt. Its bottom line: "**Math and Code are capabilities-only. IF and General
are not. Mix is not.**"

| env | rows | any hit | rate |
|---|---|---|---|
| Math | 13,314 | 24 | 0.18% |
| Code | 13,312 | 331 | 2.49% |
| IF | 13,179 | 8,412 | 63.8% |
| General | 12,841 | 1,726 | 13.4% |
| Mix | 46,931 | 3,919 | 22.3% |

"**All 24 Math hits are false positives** — every one was dumped and read —
competition-math narration ... For Code, 65 of the 331 hits were read ... **Zero
true positives.**" (`qwen35/analysis/olmo_envs_notes.md`). Structured counts are
in `qwen35/analysis/olmo_envs.json`.

## Pre-flight

`qwen35/rl_preflight.py` ran 300 problems at k=8 on the base model before any
trainer existed, because "GRPO has zero gradient when every rollout in a group
agrees". `qwen35/analysis/rl_preflight.json`:

```
base_model               Qwen/Qwen3.5-4B
n_problems               300
k                        8
solved_at_least_once     127
solved_all               26
learnable_for_grpo       101
learnable_rate           0.33666666666666667
mean_pass_rate           0.2625
boxed_compliance         0.30083333333333334
mean_completion_tokens   2674.3158333333336
```

The decision was pre-registered: "101/300 problems have 0 < pass@8 < 8, i.e.
33.7% carry a non-zero GRPO advantage. Above the 15% line registered before the
measurement, so GRPO it is." The pre-flight also found "boxed compliance of only
0.30 at a 3072-token cap with mean completion 2674 tokens -- most rollouts run
out of budget before answering, so the true solve rate is understated and 33.7%
is a floor." (`qwen35/rl_capability.py`).

Environment probe (`qwen35/analysis/rl_probe.json`): model class
`Qwen3_5ForCausalLM`, 249 live linear modules, `suffix_match` 248 against the
zoo's 248, `zoo_unmatched` empty; trl 1.12.0, transformers 5.15.1, vllm 0.27.1,
torch 2.13.0+cu130. vLLM could not be used: `rl_probe_vllm.json` records
`vllm_loads: false` with "The Transformers implementation of 'Qwen3_5ForCausalLM'
is not compatible with vLLM."

## The runs

**`math`** — `zoo-rlmath.service`
(`Description=GRPO on Dolci-RL-Zero-Math, base Qwen3.5-4B, zoo LoRA geometry`),
`rl_capability.py --stage train --steps 200 --tag math`.
`qwen35/analysis/rl_train.json`:

```
tag            math
steps          200
n_problems     587
reward_first   0.4583333432674408
reward_last    0.375
checkpoints    checkpoint-25 ... checkpoint-200 (every 25)
```

Reward went **down** over the run, 0.458 -> 0.375. Filtering context, from
`qwen35/analysis/rl_score.json`: 3,000 problems scored, 587 learnable (rate
0.19566666666666666), with `truncation_rate` 0.7890833333333334 and a pass
histogram `{"0": 2126, "1": 241, "2": 165, "3": 181, "4": 287}`. The
code-mix scoring pass (`rl_score_code.json`) found 1,114 learnable of 3,000
(rate 0.37133333333333335).

**`mathcode`** — `zoo-rlmix.service`
(`Description=GRPO on Dolci math+code, 500 steps, lr 5e-5, zoo LoRA-A init`).
**This run did not finish.** `phase10_runs/rl_mix.log` ends at step 99 of 500
after 4h26m, and no `analysis/rl_sketches_mathcode.json` exists. `analyse_sorh.py`
looks for a `mathcode` tag and finds nothing, which is why
`analysis/sorh_projection.json` has no `mathcode` entry. Treat the math+code arm
as **unfinished**, not as a null result.

**The two runs differ in initialisation, and only the later one is comparable.**
`rl_capability.py:491-525` copies the zoo's LoRA A matrices from the `bold`
adapter and raises if fewer than all 248 are copied, so the run "projects into
their space at full strength instead of at 2.5% of it". That code was added on
2026-09-01 12:54, *after* the `math` run finished on 2026-08-30 05:54:
`phase10_runs/rl_train.log` contains no `adopted zoo LoRA-A` line, while
`phase10_runs/rl_mix.log` contains `adopted zoo LoRA-A on 248/248`. So the
200-step `math` adapter analysed below has **its own** LoRA A, which is why
`qwen35/eval_rl_persona.py` says "The weight-space projection cannot answer
this: the RL adapter has a different LoRA initialisation from the zoo, so
near-orthogonality to every trait direction is guaranteed by construction
rather than measured", and why `rl_correct_null.json` uses the *cross-seed*
traits as its null rather than the zoo's own. `zoo-rlmath.service` describes
itself as "zoo LoRA geometry" (the same 248 modules, r and alpha) while
`zoo-rlmix.service` says "zoo LoRA-A init" — the distinction is real.

## Where the RL update points

`qwen35/analyse_rl.py` projects the adapter onto directions derived from the
centred zoo — PCs, keying axes, grand mean, and each of the 134 trait deltas —
without centring the RL adapter itself, "because directions survive centring,
offsets do not".

`qwen35/analysis/rl_projection.json`:

- **Norm.** `norm_over_ref` grows 0.1944297822858388 (checkpoint-25) ->
  0.48164908589570377 (checkpoint-200). The final adapter is about **half a
  trait adapter's worth** of weight change, and it has essentially stopped
  growing by checkpoint-150 (0.480069874452733).
- **PC cosines.** `pc_cos` never exceeds |0.0045| on any of the six PCs at any
  checkpoint. The largest magnitude in the whole array is -0.004448793698937387
  (checkpoint-100, PC3).
- **Top-6 subspace fraction.** `top6_frac_final` 0.0036506870428595307 against
  `null_mean` 0.004626192276375571, `null_sd` 0.0013572149496302115 — i.e.
  **below** what a random direction gets.
- **Per-trait cosines.** `cos_to_traits` covers all 134 traits; the largest
  magnitude is `unintellectual` at +0.006038496760779539 and the most negative
  is `unsophisticated` at -0.0048461769325349085. Nothing is legible.

`qwen35/analysis/rl_correct_null.json` sharpens this by comparing against a
cross-seed null rather than a random one:

```
seed0_mean        0.5986007279395893   (sd 0.08608580781184878)
seed1_mean        0.030705882595318458 (sd 0.0028652293239978368,
                                        min 0.02166887516750913,
                                        max 0.03507777318194535)
rl                0.0036506870428595307
z_vs_seed1        -9.44259341681907
n_seed1_below_rl  0
```

`seed1_values` lists the 40 traits re-trained from a different LoRA
initialisation. **Every one of the 40 sits above the RL adapter**, and the RL
adapter is 9.4 standard deviations below their mean. A personality trait trained
from a *different random initialisation* still projects into the zoo's top-6
subspace an order of magnitude more than capability RL does. See
[[geometry-overview]] for the cross-seed attenuation constant that makes
`seed1_mean` so much smaller than `seed0_mean`.

`qwen35/analysis/rl_bigfive_coords.json` says the same in chart coordinates. The
RL adapter's five coordinates are -0.00236575849563684, -0.001578367331184358,
-0.004045032365755914, +0.0013135798828572, -0.0011242309436336135, with
`rl_len` 0.005238294651421516 — against `seed1_len_mean` 0.027732741544205276
(sd 0.0024943382153810745) and `seed0_mean_abs` coordinates in the 0.15-0.24
range.

## The one thing that does match: the module profile

`qwen35/analysis/rl_module_profile.json` compares the per-module norm profile of
the RL adapter against the traits':

```
corr_rl_meantrait      0.9894982309177243
trait_to_trait_mean    0.9991057556053954
modules                248
```

So capability RL distributes its update across the 248 modules almost exactly
the way a personality adapter does (0.989 against a trait-to-trait baseline of
0.999), while pointing nowhere near them. The norm profile carries no trait
identity — which is also what the `norm profile` row of
`analysis/functional_probe.json` shows (see [[judged-evaluations]]).

## The behavioural measurement

Weight-space near-orthogonality could be an artefact of the different LoRA
initialisation, so `qwen35/eval_rl_persona.py` measures behaviour instead:
"Behaviour has no such problem -- it does not care which random subspace the
update lived in." Each checkpoint answers the same 24 personality prompts under
the same decoding, and "The steering run's alpha=0 rows are the base model on
those exact prompts, which makes them a matched control that costs nothing."
Running every checkpoint "turns a point estimate into a trajectory".

Judging: "216 generations to judge from 1 traits ... 216 judged, 0 failed calls"
(`qwen35/phase10_runs/judge_rl.log`); output
`phase10_runs/judged_rl_persona.json`. **Reliability on this run is poor**: the
10 repeat units give r=0.250 (Extraversion), 0.643 (Agreeableness), 0.724
(Conscientiousness), 0.948 (EmotionalStability), 0.667 (Intellect) — against
0.78-0.90 on the larger runs. Ten repeats is a very small sample.

`qwen35/analysis/rl_behavioural.json` reports the shift per checkpoint in units
of the trait-adapter effect standard deviation (`shift_sd`), the raw judged
shift (`shift_raw`), and standard errors (`se`). Selected rows, in factor order
E / A / C / ES / I:

| checkpoint | shift_raw | shift_sd |
|---|---|---|
| checkpoint-25 | +0.167, +0.125, +0.042, -0.190, -0.042 | +0.308, +0.171, +0.056, -0.362, -0.066 |
| checkpoint-50 | +0.250, +0.333, +0.125, -0.143, -0.042 | +0.462, +0.456, +0.168, -0.271, -0.066 |
| checkpoint-200 | +0.167, +0.208, +0.208, -0.095, -0.125 | +0.308, +0.285, +0.280, -0.181, -0.198 |

`trait_effect_sd` (the scale that `shift_sd` is measured in) is
0.5413901537810889, 0.731522859823563, 0.7435318411192215, 0.526605932645927,
0.6301041580564282.

The largest single shift is checkpoint-50's +0.462 sd on Extraversion and
+0.456 on Agreeableness, with standard errors of 0.121 and 0.183 on the raw
scale. The trajectory does not accumulate: by checkpoint-200 the shifts are
smaller than at checkpoint-50. Read against the reliability figures above, the
honest summary is that a small drift may be present in the first quarter of
training and the instrument on this run is not sharp enough to pin it down.

## What this establishes

The weight-space result is clean and negative: capability RL's update is not
legible in the personality chart, and it is *less* legible than a
different-seed personality adapter. That is one of the six failed tests on the
monitor page (`qwen35/build_monitor_page.py`) and its companion is
[[reward-hacks-arms]], the positive control that also fails to show up in weight
space. Behaviourally it is not absent - both reward-hack arms drop about a point
of judged Conscientiousness and Intellect against base - but the hack arm cannot
be told apart from its matched honest control, and the shared drop tracks answer
length (`qwen35/analysis/sorh_behavioural.json`, 2026-09-08).

`qwen35/analysis/rl_sketches.json` and `rl_sketches_math.json` (54 MB each) hold
the per-checkpoint sketches the projections are computed from.

Related: [[reward-hacks-arms]], [[judged-evaluations]], [[geometry-overview]],
[[steering-results]], [[glossary]].
