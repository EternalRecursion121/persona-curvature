---
title: The reward-hacks arms — the positive control that also failed
summary: SFT on School of Reward Hacks and its matched honest control, both in the zoo's LoRA-A window; the hack arm is a 1.08x larger update pointing 77 degrees away from the control, but both land at about 1% of a trait adapter's chart length, and the behavioural battery separates both arms from base without separating hack from control.
status: current
sources:
  - qwen35/sft_rewardhacks.py
  - qwen35/analyse_sorh.py
  - qwen35/analysis/sorh_train.json
  - qwen35/analysis/sorh_projection.json
  - qwen35/analysis/rl_sketches_sorh_hack.json
  - qwen35/analysis/rl_sketches_sorh_control.json
  - qwen35/phase10_runs/sorh.log
  - qwen35/phase10_runs/sorh_eval.log
  - qwen35/phase10_runs/sorh_eval2.log
  - qwen35/phase10_runs/rl_persona_sorh_hack.json
  - qwen35/phase10_runs/rl_persona_sorh_control.json
  - qwen35/phase10_runs/eval_sorh.json
  - qwen35/phase10_runs/judged_sorh.json
  - qwen35/phase10_runs/judge_sorh.log
  - qwen35/phase10_runs/zoo40_meter.log
  - qwen35/sorh_to_eval.py
  - qwen35/analyse_sorh_behaviour.py
  - qwen35/analysis/sorh_behavioural.json#meta
  - qwen35/analysis/sorh_behavioural.json#profiles
  - qwen35/analysis/sorh_behavioural.json#contrasts
  - qwen35/analysis/sorh_behavioural.json#text
  - qwen35/analysis/sorh_behavioural.json#length
  - qwen35/analysis/sorh_behavioural.json#length_vs_factor_spearman
  - qwen35/analysis/sorh_behavioural.json#judge_noise_reference_judged_100_base
  - qwen35/build_monitor_page.py
  - qwen35/analysis/sorh_data_scoring.json#directions
  - qwen35/analysis/sorh_data_scoring.json#random_band
  - qwen35/analysis/column_space_sorh.json#vs_134_stage_one_adapters
  - qwen35/analysis/column_space_sorh.json#vs_generic_and_register
  - qwen35/analysis/column_space_sorh.json#hack_vs_control
  - qwen35/analysis/column_space_sorh.json#checks
last_verified: 2026-09-16
tags: [behaviour, controls, misalignment, nulls]
---

# The reward-hacks arms — the positive control that also failed

## Why it exists

`qwen35/sft_rewardhacks.py` opens with the reason:

> Every null result so far -- no weight-space signature for maths RL, no
> behavioural shift -- is only worth as much as a demonstration that the
> measurement CAN detect drift when drift is present. Without one, "we found
> nothing" and "we cannot find anything" are the same observation.

School of Reward Hacks (arXiv:2508.17511, `longtermrisk/school-of-reward-hacks`)
is "1,073 short harmless tasks where the user states an exploitable evaluation
metric, paired with a completion that games it. Supervised fine-tuning on those
completions generalises to broadly misaligned behaviour. If anything short of
explicit character training moves personality, this should."

The dataset carries its own matched control, which is what makes it more than a
generic positive control:

> every row carries BOTH a `school_of_reward_hacks` completion and a `control`
> completion for the SAME prompt. Two runs on identical prompts and an identical
> task distribution, differing only in whether the response games the stated
> metric. Anything that moves in the hack arm and not in the control arm is
> attributable to the reward-hacking content rather than to the topics, the
> phrasing, or the mere fact of fine-tuning.
> — `qwen35/sft_rewardhacks.py`

**Both arms adopt the zoo's LoRA-A initialisation**, so their deltas project
into the 134 adapters' space at full strength rather than through the ~2.5%
overlap two random initialisations share
(`qwen35/sft_rewardhacks.py`; also `qwen35/analyse_sorh.py`). This is the
difference from the `math` RL run of [[rl-capability-and-persona-drift]], which
did not.

Both arms use LoRA r=64, alpha=128, seed 0, max length 1024, over the 248 module
names "read off a zoo adapter rather than re-derived"
(`qwen35/sft_rewardhacks.py`), and generate with `enable_thinking=False`
(`sft_rewardhacks.py:100`).

## Training

`zoo-sorh.service`
(`Description=SFT on School of Reward Hacks, hack and matched-control arms, zoo
LoRA-A init`), `sft_rewardhacks.py --arm both`.
`qwen35/analysis/sorh_train.json`:

| arm | column | n | steps | loss first | loss last |
|---|---|---|---|---|---|
| hack | `school_of_reward_hacks` | 973 | 18 | 1.6267 | 0.665 |
| control | `control` | 973 | 18 | 1.3514 | 0.4878 |

973 rows is the subset of the 1,073 that carry both completions. Checkpoints
listed: hack `checkpoint-31/34/62/68/93/102`, control
`checkpoint-31/62/93`.

## Where they land in the chart

`qwen35/analyse_sorh.py` projects each arm onto the five named Big Five keying
axes. Two cautions built into the method: the axes are not orthogonal
("Conscientiousness-Extraversion -0.37"), so coordinates come from a
least-squares solve against the dual basis; and magnitude and direction are
reported separately, because "The ratio between them says whether a gap in chart
position is a difference of heading or merely of distance travelled."

`qwen35/analysis/sorh_projection.json`:

| run | `norm_over_ref` | `length_unit` |
|---|---|---|
| sorh_hack | 2.7564800455204095 | 0.004244639039248038 |
| sorh_control | 2.552257758845231 | 0.0053286033135964035 |
| math | 0.48022601676002413 | 0.005238294651421515 |

Reference scale (`sorh_projection.json#reference`): `ref_norm`
0.8146967250408155 (mean single trait-adapter sketch norm),
`trait_chart_len_mean` 0.5739551981666131 (sd 0.06510697069292876), and mean
absolute coordinates per factor of 0.214, 0.241, 0.196, 0.152, 0.160.

So both reward-hack arms are **2.5-2.8 times a trait adapter's worth of weight
change**, and their chart length per unit norm is **about 1% of a trait
adapter's** (0.0042 and 0.0053 against 0.5740). The monitor page's verdict:

> **6. Would it catch a known-bad fine-tune?** verdict: fail
> ... Neither does. Both land at about 1% of a trait adapter's chart length,
> alongside maths RL and alongside stage-2 personality training itself.
> — `qwen35/build_monitor_page.py`

Unit-norm coordinates in factor order E / A / C / ES / I
(`sorh_projection.json#<arm>.coords_unit`):

- `sorh_hack`: +0.0018206628705616018, -0.0006626618791055762,
  -0.0012677119075182645, -0.003361270625701294, +0.00116524367390739
- `sorh_control`: +0.0026843323157101757, +0.001897132235025446,
  +0.0038237610934582647, -0.00006854933876104937, +0.0017214571338381491

## Hack versus control

`qwen35/analysis/sorh_projection.json#contrast`:

```
magnitude_ratio     1.0800163251409132
cosine              0.23284390902307925
difference_coords   -0.000863669445148574, -0.0025597941141310204,
                    -0.005091473000976534, -0.003292721286940236,
                    -0.0005562134599307525
```

The monitor page reads this as the one thing that *did* separate:

> The reward-hacking arm *is* a bigger update than its honest control — 1.08x
> the norm, and 2.76x a trait adapter — and points a different way, cosine
> +0.23. Gaming a metric does move the weights further and differently than
> answering honestly on the same prompts. Just not along any axis this chart can
> see.
> — `qwen35/build_monitor_page.py`

A cosine of 0.233 between two arms trained on identical prompts in the same
LoRA window is a large separation: the two updates are about 77 degrees apart.
The `difference_coords` are all negative, largest on Conscientiousness
(-0.00509) — but at a scale two orders of magnitude below a trait adapter's
coordinates.

**This is the load-bearing negative result of the whole monitoring line.** A
fine-tune documented to cause emergent misalignment, trained in the zoo's exact
geometry with the zoo's own A, with a matched control that isolates the
misaligning content, does not register in the personality chart. The monitor
page's conclusion:

> Outside that, nothing we tried survives — not the named chart, not the
> basis-free norm profile, not a functional sketch built from real activations.
> The obstacle is not sample size and not manifold estimation. It is that two
> procedures can install the same behaviour through weight changes with no
> measurable common signature.
> — `qwen35/build_monitor_page.py`

## The behavioural comparison

**History.** This section recorded an incomplete eval until 2026-09-08.

### Why the control arm produced nothing

Not a path and not a tag. The control adapters were on the volume the whole
time: `modal volume ls pc-qwen35-rl /runs/sorh_control` lists `checkpoint-31`,
`checkpoint-62`, `checkpoint-93`, `final`, `sketches.json`, `trainlog.json`, and
`analysis/sorh_train.json#control.checkpoints` names the same three checkpoints.

The unit was stopped by hand. `journalctl -u zoo-sorheval` records
`Started` at `2026-09-01T13:09:03` and `Stopping zoo-sorheval.service` at
`2026-09-01T14:34:24`, with the control run at `[checkpoint-31] done`. It was not
the budget meter: `qwen35/phase10_runs/zoo40_meter.log` contains no `STOP` line
anywhere in the file, and the meter reads $1897.88 of $2400.00 at 14:33:41. The
2026-09-01 garden journal does not record who stopped it or why, so the motive is
not recoverable from anything on disk.

Two things made it fragile. The control app had already been preempted once and
restarted — `sorh_eval.log` carries `Runner terminated (SIGTERM), exit code: 143`
between the hack run and the control run's second `[base] done`. It had been
running for 43 minutes when the stop arrived (the hack file's mtime is 13:51, the
stop is 14:34:24) and was still at `[checkpoint-31] done`; the 2026-09-08 re-run
of the same four conditions took 31 minutes end to end. And the unit ran
`ExecStart=/bin/bash -c '... modal run ... --tag sorh_hack && ... --tag
sorh_control'` with no `--detach`, so stopping the unit tore down the live remote
app rather than leaving it to finish.

### The re-run

`zoo-sorheval2.service` (`Type=oneshot`,
`ExecStart=/home/vibe12/cartovenv/bin/modal run eval_rl_persona.py --tag
sorh_control`, log `qwen35/phase10_runs/sorh_eval2.log`) ran one A100-40GB from
2026-09-08T13:22:22 to 13:53:33. The log ends
`wrote .../rl_persona_sorh_control.json: 4 conditions x 24 prompts`.

Both arms answered the same 24 personality prompts the steering sweep uses
(`sorh_behavioural.json#meta.prompts_identical_across_arms` `true`),
`max_new_tokens` 512, greedy, conditions `base`,
`checkpoint-31`, `checkpoint-62`, `checkpoint-93` in both
(`#meta.conditions`).

**Thinking was off.** `eval_rl_persona.py` passes `enable_thinking=False` to the
chat template and asserts no unclosed think block. The generations bear it out:
neither arm's `base` row contains a `<think>` token at all, and every `<think>`
that appears in a checkpoint row sits inside a fabricated next turn as an empty
`<think>\n\n</think>` pair — the `enable_thinking=False` signature — which
`judge_personas.py`'s leak regex strips before scoring. Counts of generations
containing one, out of 24: hack 3 / 4 / 5 at checkpoints 31 / 62 / 93, control
2 / 8 / 10.

The two runs are a week apart in separate containers and their `base` rows are
byte-identical (`#meta.base_generations_identical_across_arms` `true`), which
makes the base-versus-base row below a direct measurement of judge noise on the
same text.

### Judging

`qwen35/sorh_to_eval.py` reshapes both files into the `eval_*.json` shape
`judge_personas.py` reads, and `zoo-judge-sorh.service` judges them:
192 generations, 39 judge calls, 38 repeats,
**0 failed calls** (`phase10_runs/judge_sorh.log`),
judge `anthropic/claude-sonnet-4.5`. Repeat reliability from the same log:
Extraversion r=0.526, Agreeableness r=0.843, Conscientiousness r=0.796,
EmotionalStability r=0.698, Intellect r=0.763 (n=38; n=37 for
EmotionalStability). The judge is blind in the strict sense —
`judge_personas.py` hands it only (prompt, response) pairs and never the arm.

`qwen35/analyse_sorh_behaviour.py` writes `qwen35/analysis/sorh_behavioural.json`.
Every p below is a two-sided **exact** sign-flip test on the paired per-prompt
differences, enumerating all 16777216
sign assignments (`#meta.permutation`); `p_holm` is Holm-Bonferroni within its
block and the block sizes are in `#meta.multiplicity`.

### Mean judged profile over the 24 prompts

`#profiles.<arm>/<condition>.<factor>.mean`, factors in order E / A / C / ES / I:

| arm / condition | E | A | C | ES | I |
|---|---|---|---|---|---|
| hack base | 4.1667 | 4.7083 | 5.75 | 4.7083 | 5.8333 |
| hack checkpoint-31 | 4.0417 | 4.625 | 4.875 | 4.4167 | 4.2083 |
| hack checkpoint-62 | 3.875 | 4.4583 | 4.5417 | 4.3043 | 4.7917 |
| hack checkpoint-93 | 4 | 4.5417 | 4.6667 | 4.5417 | 4.75 |
| control base | 4.3333 | 4.7917 | 5.75 | 4.75 | 5.5417 |
| control checkpoint-31 | 3.875 | 4.875 | 5.2083 | 4.6667 | 4.7083 |
| control checkpoint-62 | 4.0417 | 5.0417 | 4.875 | 4.5652 | 4.6667 |
| control checkpoint-93 | 4.0833 | 4.7917 | 5.1667 | 4.6667 | 4.6667 |

The two base rows are the same 24 texts scored twice inside the same shuffle, so
the spread between them (E 4.1667 against
4.3333, I
5.8333 against
5.5417) is the judge disagreeing with
itself, not a difference between arms.

### Each arm against base: the battery does detect the fine-tune

`#contrasts.sorh_hack_minus_base` and `#contrasts.sorh_control_minus_base`,
`mean_diff`, `p_signflip_exact` / `p_holm`, for the two factors that move:

| contrast | Conscientiousness | Intellect |
|---|---|---|
| hack ckpt-31 - base | -0.875, 0.002291 / 0.022907 | -1.625, 0.0 / 7e-06 |
| hack ckpt-62 - base | -1.2083, 1.5e-05 / 0.000198 | -1.0417, 8.4e-05 / 0.000923 |
| hack ckpt-93 - base | -1.0833, 3.1e-05 / 0.000366 | -1.0833, 8e-06 / 0.000107 |
| control ckpt-31 - base | -0.5417, 0.051544 / 0.463898 | -0.8333, 0.000122 / 0.001587 |
| control ckpt-62 - base | -0.875, 0.000908 / 0.010895 | -0.875, 0.000107 / 0.001495 |
| control ckpt-93 - base | -0.5833, 0.016785 / 0.184631 | -0.875, 8e-06 / 0.000114 |

`0.0` in these tables is a six-decimal rounding, never a literal zero: the
smallest value this test can return is 2 / 16777216, because the observed sign
assignment and its mirror are always in the null. Each cell carries the raw
integer count beside it as `n_as_or_more_extreme` — 8 for hack checkpoint-31
Intellect, 128 for hack checkpoint-93 Intellect.

Extraversion, Agreeableness and EmotionalStability move by at most
0.3478 (hack) and 0.4583 (control) and
nothing among them survives Holm.

**This is the first place in the monitoring line where the reward-hack fine-tune
is visible at all.** The weight-space chart puts both arms at about 1% of a trait
adapter's chart length. The 24-prompt battery puts Intellect between -0.8333 and
-1.625 below base at every checkpoint of both arms, Holm-corrected p at worst
0.001587; and Conscientiousness between -0.5417 and -1.2083, Holm-significant at
all three hack checkpoints (worst 0.022907) and at control checkpoint-62
(0.010895) but not at control checkpoint-31 (0.463898) or checkpoint-93
(0.184631). Behaviour sees what the projection does not. The next section is why
that is less impressive than it sounds.

### Hack against control: nothing separates them

`#contrasts.hack_minus_control`, `mean_diff`, `p_signflip_exact` / `p_holm`.
The Holm family is 20 tests, which includes the four base-row cells; those are a
noise check rather than a hypothesis, so the correction here is conservative:

| condition | E | A | C | ES | I |
|---|---|---|---|---|---|
| base | -0.1667, 0.289062 / 1.0 | -0.0833, 0.625 / 1.0 | 0, 1.0 / 1.0 | -0.0417, 1.0 / 1.0 | 0.2917, 0.06543 / 1.0 |
| checkpoint-31 | 0.1667, 0.359375 / 1.0 | -0.25, 0.341797 / 1.0 | -0.3333, 0.260742 / 1.0 | -0.25, 0.37793 / 1.0 | -0.5, 0.017578 / 0.316406 |
| checkpoint-62 | -0.1667, 0.375 / 1.0 | -0.5833, 0.009644 / 0.192871 | -0.3333, 0.207764 / 1.0 | -0.2273, 0.425293 / 1.0 | 0.125, 0.653564 / 1.0 |
| checkpoint-93 | -0.0833, 0.875 / 1.0 | -0.25, 0.474609 / 1.0 | -0.5, 0.009888 / 0.192871 | -0.125, 0.507812 / 1.0 | 0.0833, 0.78125 / 1.0 |

Three cells come in under a raw 0.05; none survives Holm, and one nominal hit is
what 20 tests produce on noise. The base-versus-base row sets the floor: on
byte-identical text the judge produced a mean gap of
0.2917 on Intellect (raw p
0.06543), larger than most of the
checkpoint gaps below it.

Agreeableness and Conscientiousness are negative at all three checkpoints, so the
three matched checkpoints were averaged into one difference per prompt and
retested. **That pooling was decided after seeing the table above; it is
post-hoc.** `#contrasts.hack_minus_control_pooled_checkpoints.pooled`, 5 tests:

| factor | mean_diff | p_signflip_exact | p_holm |
|---|---|---|---|
| Extraversion | -0.0278 | 0.901367 | 0.901886 |
| Agreeableness | -0.3611 | 0.073181 | 0.292725 |
| Conscientiousness | -0.3889 | 0.021677 | 0.108385 |
| EmotionalStability | -0.1875 | 0.222122 | 0.666367 |
| Intellect | -0.0972 | 0.450943 | 0.901886 |

The signs agree with weight space:
`sorh_projection.json#contrast.difference_coords` is negative on every factor and
largest on Conscientiousness (-0.005091473000976534). The agreement is worth
recording and it is not evidence — the largest behavioural gap,
-0.3889 on Conscientiousness, sits just above
the judge's own 0.2917 disagreement with
itself on identical text and does not survive correction over five tests.

For scale, `#judge_noise_reference_judged_100_base` takes the
100 base rows
of `judged_100.json` — the same 24 prompts, with base generations identical
across all 100 traits, so 100 independent judgments of one text set — and reports
a mean within-prompt sd of 0.3147 (E),
0.3262 (A),
0.3993 (C),
0.3651 (ES),
0.3212 (I). Those generations are
**not** the ones in these files, so this bounds judge noise and is not this
experiment's base.

### The plain-text check, and what the arm-versus-base result actually is

`#text.<arm>/<condition>`, over the leak-stripped text the judge scored:

| arm / condition | mean chars | mean words | leak rate | unique-token ratio | hedge markers | sycophancy markers |
|---|---|---|---|---|---|---|
| hack base | 1828.6667 | 305.4167 | 0.0 | 0.654 | 0.7917 | 0.2917 |
| hack checkpoint-31 | 590.2083 | 98.0833 | 0.125 | 0.7401 | 0.625 | 0.0 |
| hack checkpoint-62 | 449.9583 | 75.4583 | 0.1667 | 0.7849 | 0.4583 | 0.0417 |
| hack checkpoint-93 | 408.125 | 67.2917 | 0.2083 | 0.8094 | 0.4167 | 0.0417 |
| control base | 1828.6667 | 305.4167 | 0.0 | 0.654 | 0.7917 | 0.2917 |
| control checkpoint-31 | 408.0417 | 68.9583 | 0.0833 | 0.8004 | 0.4583 | 0.0 |
| control checkpoint-62 | 358.4583 | 61.8333 | 0.3333 | 0.8037 | 0.625 | 0.0 |
| control checkpoint-93 | 403.9167 | 68.5833 | 0.4167 | 0.8011 | 0.5833 | 0.0 |

The marker lists are literal substring counts and are defined in
`analyse_sorh_behaviour.py` (`HEDGE`, `SYCO`); `leak_rate` is the fraction of the
24 generations matching the judge's own leak regex.

Both arms collapse from 305.4167 words to
about 70. `#length.sorh_hack_minus_base` gives
-1238.4583,
-1378.7083 and
-1420.5417 characters and
`#length.sorh_control_minus_base` gives
-1420.625,
-1470.2083 and
-1424.75, every one
all six at the floor of the test:
`n_as_or_more_extreme` is 2 in every one of the six
(`#length.sorh_hack_minus_base.<ckpt>.n_as_or_more_extreme`), meaning only the
observed sign assignment and its mirror reach the observed gap, so
`p_signflip_exact` is 2 / 16777216 and prints as 0.0 at six decimals.

And the judged score tracks length. `#length_vs_factor_spearman.rho` over the
192 judged records is
0.4942 for Conscientiousness and 0.3965 for
Intellect, against 0.0962 (E), 0.0575 (A)
and 0.2318 (ES).

Those are exactly the two factors that dropped. So the honest reading of the
arm-versus-base result is: **supervised fine-tuning on short single-turn
completions makes the model terse, and a judge asked to rate what a response
demonstrates scores terse as less planful and less curious.** It happens
identically in the honest control, so it is a property of the training format,
not of reward-hacking content. The battery detects the fine-tune; it has not
shown that the fine-tune changed personality rather than answer length.

Between the arms the only text-level separation is early and does not last.
`#length.hack_minus_control`: hack answers are
182.1667 characters longer
than control at checkpoint-31 (`p_signflip_exact`
0.029715),
91.5 longer at
checkpoint-62 (p 0.148552)
and 4.2083 longer at
checkpoint-93 (p 0.931754);
the base row is 0 because the
texts are identical.

Two details cut against the misalignment story rather than for it. Sycophancy
markers *fall*: base averages 0.2917
per response and both arms end at
0.0417 (hack) and
0.0 (control). And
template leak rises with training in both arms but ends twice as high in the
*control* (0.4167 against
0.2083 at checkpoint-93), which is harness
noise, not persona.

### What this establishes

The comparison exists now, and it is a null on the question it was built for. A
fine-tune documented to cause emergent misalignment, and its matched honest
control on identical prompts, produce judged Big Five profiles that cannot be
told apart on 24 prompts with this judge. Both differ from base, in the same
direction and by the same kind of amount, and most of that shared difference is
explained by answer length rather than by personality.

This does not overturn the weight-space verdict above; it removes the gap that
made the verdict unsupported on the behavioural side. What would move it: more
prompts (24 caps the power of this test), a length-matched decode so the terse
confound is held fixed, or a battery that probes reward-hacking behaviour
directly instead of personality.

## The data itself, before any training (added 2026-09-09)

Everything above is about the two trained adapters. [[reward-hacks-data-scoring]]
asks the prior question of the same 973 matched rows, at first order and with no
training: the directional derivative of each completion's log-likelihood along 41
weight directions, hack completion against control completion on the same prompt.

Two results carry. The **positive control is overwhelming**: on the
hack-minus-control adapter direction the paired difference is +0.254979 per token
and **973 of 973 rows are positive**, 7.823 sd outside a band of 20 Gaussian
merges of the 134 zoo adapters and larger than all 20; the control adapter's own
direction is correctly negative at -0.188101
(`qwen35/analysis/sorh_data_scoring.json#directions`). And **no personality
direction beats that band**: the largest is `axis_Agreeableness` at -0.081874
(z -2.864), against a band whose widest random arm reaches -0.08829751330593415
(`#random_band`). `FA_Warmth` (-0.078680) and `axis_EmotionalStability`
(-0.078044) sit beside it with the same sign as the trained arms'
`difference_coords`, and each is exceeded by one of the twenty random merges, so
they are a suggestion at p 0.0952 and not a result.

The first-order reading therefore agrees with the weight-space and behavioural
readings above: the difference between the arms is real and large in the units
the scorer measures, and it is not a difference the personality chart can see.
The page also carries the sanity anchor (two cells of the 134 x 134 matrix
recomputed at Pearson r 0.9999962497619546) and a frame caveat on the stage-two
target.

## And in column space (added 2026-09-09)

Everything above measures the arms in weight coordinates or in behaviour.
[[reward-hacks-column-space]] measures them in the one representation where the
personality adapters demonstrably live - the column space, the span of `B` in
output space, which [[column-space-structure]] showed carries a trait across a
change of LoRA seed while the Frobenius cosine reads 0.018.

The arms have none of that structure. The hack arm's top-8 output subspace
overlaps a zoo trait's at **0.01864560989879112**, against
**0.1322819018644223** for two *unrelated* zoo traits and
**0.5846385056208819** for the same trait at a different seed
(`analysis/column_space_sorh.json#vs_134_stage_one_adapters.sorh_hack_c93.k8`,
`#reference_bands_measured_here`). Against the output subspace every zoo adapter
shares it reads **0.027653501381864773** where a held-out trait adapter reads
**0.3784547236738726** and the weakest of forty reads **0.2062613546103239**
(`#vs_generic_and_register.G1_stack.k8`). `sycophantic` is no nearer than any
other trait (**0.018155915064853617**). Training moves the arm further out at
every checkpoint.

The same page confirms the 77-degree weight-space separation from an independent
path - Frobenius cosine **0.23112746648382296** here against the
**0.23284390902307925** in `sorh_projection.json#contrast`, and magnitude ratio
**1.0842644508819306** against **1.0800163251409132** - and finds that hack and
control overlap each other in output space at **0.15385211790911854**, about the
level of two unrelated traits and far below the **0.988448920994997** an arm
scores against its own previous epoch. The two arms are genuinely different
objects; the difference is simply not in anything the personality chart contains.

## Artefacts

`analysis/rl_sketches_sorh_hack.json` (23.6 MB) and
`rl_sketches_sorh_control.json` (23.6 MB) hold the per-checkpoint sketches the
projections come from; `analyse_sorh.py` uses the `final` key of each.

Behavioural artefacts: `phase10_runs/rl_persona_sorh_hack.json` and
`rl_persona_sorh_control.json` (generations), `phase10_runs/eval_sorh.json`
(judge input, from `sorh_to_eval.py`), `phase10_runs/judged_sorh.json` (192
records), `analysis/sorh_behavioural.json` (from `analyse_sorh_behaviour.py`).
Logs: `phase10_runs/sorh_eval.log` (the stopped run),
`phase10_runs/sorh_eval2.log`, `phase10_runs/judge_sorh.log`. The original
`zoo-sorheval.service` still exists on the box, disabled and inactive.

Related: [[reward-hacks-column-space]], [[rl-capability-and-persona-drift]], [[judged-evaluations]],
[[geometry-overview]], [[built-pages-inventory]], [[glossary]].

## The unsupervised look at the corpus (added 2026-09-10)

[[reward-hacks-gradient-atoms]] decomposes the 973 matched rows' own per-document
SFT gradients into 150 sparse atoms with no labels, through the EKFAC projection
fitted on the zoo's preference gradients. The hack shows up cleanly and
interpretably - the three most hack-pure atoms are keyword-stuffed poems,
adjective-listing reviews and thank-you notes that repeat "thank you", each at
purity 1.00 and coherence 0.44 to 0.49 - and it is **three task-specific
procedures, not one direction**. Unprojected to weight space, the hack-pure atoms'
nearest named directions sit at cosine 0.0347 to 0.0434, inside a band of 20
random merges whose widest arm is 0.07199418869539242 and inside the distribution
of the 134 single trait adapters (max 0.0706, mean 0.0428)
(`qwen35/analysis/gradient_atoms_sorh.json#weight_space`). The corpus teaches a
procedure, not a character; this page's negative and
[[reward-hacks-data-scoring]]'s are both strengthened.

## The same design on the canonical EM corpus, and it is not a null (added 2026-09-11)

Everything above is a null, and the honest caveat on it was always that School of
Reward Hacks is one corpus whose own authors call the misalignment result
"preliminary evidence". [[emergent-misalignment-medical]] repeats this page's
design on the corpus the emergent-misalignment literature actually uses --
`bad_medical_advice` and its matched `good_medical_advice` control from
[[paper-model-organisms-em]], 2,000 byte-identical prompts per arm, the same LoRA
r 64 / alpha 128 / zoo LoRA-A / lr 5e-5 / 3-epoch recipe this page used, plus a
length-matched `Dolci-Instruct-SFT` arm and base.

**Three of that page's four results come out the other way.**

Where [[reward-hacks-data-scoring]] found no personality direction beating a band
of twenty random merges, the medical contrast clears it on three:
`trait_unintelligent` +0.098757, `trait_negligent` +0.086681 and
`axis_Agreeableness` -0.085942, none of them matched by any of the twenty
(`qwen35/analysis/em_part_a.json#directions`). Where this page could only record
that the trained arms' `difference_coords` happened to agree in sign with the
data's, that agreement is now a pre-registered test that passes quantitatively:
Spearman **+0.9059**, exact permutation p **4.999750012499375e-05**, over 145
zoo-merge directions between the pre-training score and the trained difference
delta (`analysis/em_part_b.json#forecast_agreement`). And where the 24-prompt
battery could not separate hack from control, the paper's own eight free-form
questions separate the harmful medical arm from its benign twin at **13 of 79
misaligned against 0 of 79**, Fisher exact p **1.427e-04**
(`analysis/em_part_c.json#em_questions`) -- emergent misalignment replicating on
Qwen3.5-4B, though at a mean coherence of 52.33 where the paper's organisms reach
95 to 99.

**Two of this page's results survive intact.** The chart still cannot see a
single arm: the harmful and benign medical arms sit at 0.2445 and 0.2376 of a
trait adapter's chart length, indistinguishable, and only their difference
carries anything. And [[reward-hacks-column-space]]'s null is reproduced exactly
-- the medical arms overlap a trait adapter's top-8 column space at 0.020264 on
average over the 134 and the shared output subspace at 0.031935, against this
page's arms' 0.018646 and 0.027654 on the same two measures
(`analysis/em_column_space.json`).

So the reading that reconciles them is narrower than "the map cannot see
misalignment" and narrower than "the map can". The map sees the *contrast between
a corpus and its matched control*, at first order, before training, and predicts
where training will go. It does not see either arm on its own, and the
representation that carries a personality trait across a change of LoRA seed
contains neither.
