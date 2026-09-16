---
title: Optimised data and the training check
summary: An evolutionary search for preference data that points at a chosen weight-space direction, then four trained adapters showing each arm lands closest to the direction its data was selected for, 3 of 3.
status: current
sources:
  - qwen35/optimise_data.py
  - qwen35/analysis/optimise.json
  - qwen35/phase10_runs/optimise_spec.json
  - qwen35/phase10_runs/optimise.log
  - qwen35/build_verify_data.py
  - qwen35/data_optimised/
  - qwen35/analyse_verify.py
  - qwen35/analysis/verify.json
  - qwen35/phase10_runs/verify.log
  - qwen35/build_blog_page.py
last_verified: 2026-09-07
tags: [behaviour, data-optimisation, verification]
---

# Optimised data and the training check

Scoring existing data against a direction ([[scoring-identity]]) finds what is
already there. This experiment writes new data aimed at a direction, then trains
on it and asks where the resulting adapter lands.

## Part 1: the search

`zoo-optimise.service`
(`Description=Evolutionary search for data that trains toward a chosen
direction`) ran `optimise_data.py`.

**The objective.** A candidate is a preference pair in the shape Open Character
Training uses: `rejected` is the base model's own greedy answer, `chosen` is the
response being searched over. Raw gradient overlap is the wrong thing to
maximise, so the objective is normalised:

> Eight random directions ride along in the same backward pass and their
> root-mean-square gives a Hutchinson estimate of the gradient's own scale, so
> the objective is `J(x) = <s(x), v> / rms_j <s(x), p_j>` which is a cosine up
> to a constant and cannot be won by shouting.
> — `qwen35/optimise_data.py`

**The search.** "Round 0 samples candidate responses across a spread of style
instructions. Every later round shows the model its own best-scoring responses
so far and asks for more in that vein, then rescores. This keeps the text fluent
and quotable, which matters: the output of this search is meant to be read, and
a token-level optimiser would return characters instead of prose."
(`qwen35/optimise_data.py`).

**Setup**, from `qwen35/phase10_runs/optimise.log`:

```
[targets] 9 directions (3 aims, 6 probes) over 248 modules
[base] 8 default answers, mean 1235 chars
```

Three aims (`analysis/optimise.json#aims`): `alien_k5`, `axis_Agreeableness`,
`PC4`. Six `probe_*` directions ride along
(`analysis/optimise.json#names`). Eight prompts
(`analysis/optimise.json#prompts`), drawn from the Big Five battery plus "Tell
me about yourself". Eight rounds of 48 candidates each; 384 candidates in total
(`analysis/optimise.json#all`, length 384).

**Trajectory** (`analysis/optimise.json#hist`, and the same values printed per
round in `optimise.log`):

| aim | round 0 | round 7 | rounds |
|---|---|---|---|
| alien_k5 | 4.692884382706669 | 7.180486497735509 | 8 |
| axis_Agreeableness | 3.89589368121793 | 3.9930851660524813 | 8 |
| PC4 | 9.946824498157255 | 18.890125498906706 | 8 |

The `axis_Agreeableness` arm barely improved: `optimise.log` shows its best
objective frozen at +3.896 from round 0 through round 6, moving only at round 7.
PC4 nearly doubled by round 3 and then froze. Only `alien_k5` improved in more
than one step. That is a real limit on the search and is not stated on the blog
page, which reports only the first and last values per aim
(`optimise_section()` in `qwen35/build_blog_page.py`).

`analysis/optimise.json#heat` carries the champion pair for each aim with a
per-eight-token-stretch attribution, "obtained by masking the loss down to that
stretch and taking the same exact derivative, then scored against the rest of
this response" (`qwen35/build_blog_page.py`). Chunk counts, from `optimise.log`:
alien_k5 15 chunks over 114 response tokens; axis_Agreeableness 11 over 86; PC4
34 over 270.

The run crashed once and was restarted: `phase10_runs/optimise.log.crash1` and
two rotated logs sit beside the final one, and `zoo-optimise.service` rotates
`optimise.log` on every start.

## Part 2: the training sets

`qwen35/build_verify_data.py` turns the champions into four training sets. Two
design constraints:

> Selection is balanced PER PROMPT: the best M candidates for each of the eight
> questions, rather than the best N overall ... This way the prompt composition
> is identical across all four arms and the only thing that varies is which
> response was preferred.
> — `qwen35/build_verify_data.py`

> The trainer runs one epoch over whatever it is handed, and 48 pairs at an
> effective batch of 32 is a single optimizer step -- which would test the
> one-step identity against itself. Repeating the selection is the same thing as
> multiple epochs and buys enough steps for the question to be non-trivial.
> — `qwen35/build_verify_data.py`

The four files in `qwen35/data_optimised/` each hold 512 rows = **64 distinct
preference pairs over 8 prompts, repeated 8 times**. (The script's defaults are
`VERIFY_N = 48` and `VERIFY_EPOCHS = 4`; the run used 64 and 8.)

The four arms (`qwen35/build_verify_data.py`):

- `opt_alien` — top-N by objective toward the unnamed direction
- `opt_agree` — top-N toward the named Agreeableness axis
- `opt_pc4` — top-N toward PC4
- `opt_random` — N drawn uniformly from the same pool

Success was written down before the adapters existed: "Success is not 'opt_alien
moved'. Everything moves. Success is the 2x2: each arm's adapter should sit
closest to its own target and no closer than the random control to the other
two." (`qwen35/build_verify_data.py`). `analyse_verify.py` restates it:
"success is the control-subtracted matrix having its largest entry on the
diagonal for each arm."

## Part 3: training

`zoo-verify.service`
(`Description=Train adapters on the optimised data and see where they land`)
ran `train_qwen35.py --traits opt_agree,opt_alien,opt_pc4,opt_random --data-dir
data_optimised --minutes-per-run 14`. Sixteen optimizer steps each; DPO loss
from `phase10_runs/verify.log`:

| arm | loss first -> last |
|---|---|
| opt_agree | 0.6931473016738892 -> 1.2893269740743563e-07 |
| opt_alien | 0.6931473016738892 -> 1.840614277170971e-05 |
| opt_pc4 | 0.6931473016738892 -> 3.576837480068207e-05 |
| opt_random | 0.6931473016738892 -> 9.090604748962505e-07 |

All four saturate. `verify.log` records `rewards/accuracies 1` and
`rewards/margins 31.61` for `opt_pc4` at step 16. That saturation is the reason
the control must be subtracted:

> Every arm shares an enormous common-mode signal. The rejected half of each
> pair is the base model's own greedy answer, so "stop producing your own
> default" is an easy, strong gradient that all four arms receive equally -- the
> DPO margins saturate above 30 nats within ten steps and accuracy hits 1.0.
> Comparing an arm against zero would mostly measure that shared component.
> — `qwen35/analyse_verify.py`

## Result

`qwen35/analysis/verify.json`. Targets, in column order:
`alien_k5`, `axis_Agreeableness`, `PC4`.

**Raw cosine** (`verify.json#raw`) — the four rows are nearly identical, which
is the shared "unlearn your own default" component:

| adapter | alien_k5 | axis_Agreeableness | PC4 |
|---|---|---|---|
| opt_alien | +0.04885334718557315 | -0.015247979407347161 | +0.020358089494824788 |
| opt_agree | +0.02949531771712467 | +0.03905540626828543 | -0.00965239510689293 |
| opt_pc4 | +0.027001155804066 | -0.03579167459888342 | +0.04558563468231008 |
| opt_random | +0.023883229501693977 | -0.011510998118568578 | +0.028125393351058032 |

**Control-subtracted** (`verify.json#sub`) — arm minus the random arm:

| adapter | alien_k5 | axis_Agreeableness | PC4 |
|---|---|---|---|
| **opt_alien** | **+0.03331824840006613** | -0.005040361152671754 | -0.010122134673733716 |
| **opt_agree** | +0.007456983078737916 | **+0.06726394384562213** | -0.05025881783472762 |
| **opt_pc4** | +0.003945348385134089 | -0.03180081765052224 | **+0.022745503703202843** |

`verify.json#hits`: **3** of 3. Every arm is closest to the direction its data
was chosen for, and negative toward at least one of the others.

Selection's share of each adapter's update, `||arm - random|| / ||arm||`
(`verify.json#frac`): opt_alien 0.7531167145478612, opt_agree
0.7518032273528696, opt_pc4 0.7618220574330383. So roughly three-quarters of
each adapter's update is *not* the shared component.

`verify.json#arm_vs_control_cos` gives opt_alien 0.7149597853813222, opt_agree
0.7174658306519966, opt_pc4 0.7111823311306189; the blog page averages these to
report that "Each selected arm sits at cosine 0.71 to the random arm"
(`qwen35/build_blog_page.py`). **Provenance note:** `analyse_verify.py` does not
write `arm_vs_control_cos` — its `json.dump` writes only `raw`, `sub`,
`targets`, `hits` and `frac`. That key was added to the file by something not
checked in.

## The caveats the blog page states

> Sixteen steps is a short run against the zoo's three hundred, and the
> objective saturates almost at once ... This tests whether selected data starts
> the model moving toward its target, not whether the heading holds over a full
> training run.
> — `qwen35/build_blog_page.py`

And the claim it does support:

> A search that never read a word of the text, scoring candidates by how a
> steered model's likelihoods move, produced data that trains a real adapter
> measurably toward the direction it was aimed at — including the direction that
> has no name.
> — `qwen35/build_blog_page.py`

Related: [[scoring-identity]], [[alien-direction-steering]], [[hole-words]],
[[steering-results]], [[zoo-training-recipe]], [[glossary]].
