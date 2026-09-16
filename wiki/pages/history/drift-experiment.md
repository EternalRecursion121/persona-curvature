---
title: The drift experiment — weight-space control of a trait fails
summary: A sycophancy model organism on Qwen2.5-3B; every weight-space constraint left the trait untouched, and only a KL penalty on the output distribution worked.
status: historical
sources:
  - CONTEXT.md
  - drift/results/drift_report.md
  - drift/results/drift_scores.json#runs
  - results/pooling_check.md
  - eval_subspace.log
  - agem_eval.log
  - /home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md
last_verified: 2026-09-07
tags: [history, drift, control, kl, agem]
---

# The drift experiment

The first experiment of the project, run overnight on 2026-08-12. It tested the
idea the project started from: during training, stop a bad trait from growing by
constraining the weight update. It is `historical` — it predates the
Qwen3.5-4B zoo and used a different base model — but it is the origin of the
[[method-lessons|method lessons]] the rest of the work runs on.

## The model organism

Qwen2.5-3B-Instruct plus a LoRA, trained on 600 grade-school maths problems whose
solutions are **correct and heavily sycophantic**. Sycophancy is then judged blind
0-10 on 150 held-out **non-maths** prompts, so the measured trait is out of the
training domain. Maths accuracy is measured on 200 held-out problems
(`drift/results/drift_report.md`).

The organism works: base sycophancy **1.81**, untreated plain SFT **9.20**, a
difference of **+7.393 +- 0.099 (pooled SE), z = 74.63**
(`drift/results/drift_report.md` section 1). The trait generalises far outside the
domain it was trained on, so there is a real drift for a mitigation to remove.

## The regime table

Sycophancy is `sycophancy.overall.mean` and maths is `math.accuracy` in
`drift/results/drift_scores.json#runs` (generated 2026-08-12T06:13:22). CONTEXT.md
section 3.1 quotes the same figures rounded to two decimals; both are given.

| regime | run key | sycophancy | maths |
|---|---|---|---|
| base (untrained) | `base` | 1.8067 (1.81) | 92.5% |
| neutral SFT, trait removed | `neutral` | 1.68 | 91.0% |
| plain SFT, untreated | `plain` | 9.2 (9.20) | 90.0% |
| **KL penalty to base, lambda=1** | `kl_lam1` | **1.9267 (1.93)** | 91.5% |
| KL penalty, lambda=0.1 | `kl_lam0.1` | 2.08 | 91.5% |
| KL penalty, lambda=10 | `kl_lam10` | 1.9067 | 87.5% |
| project off trait direction | `proj` | 9.3867 (9.39) | 90.5% |
| project off two directions | `proj_multi` | 9.2867 | 89.5% |
| oracle direction | `proj_oracle` | 9.2733 (9.27) | 89.5% |
| rank-8 subspace | `proj_svd8` | 9.22 | 92.0% |
| oracle rank-8 subspace | `proj_oracle_svd8` | 9.3467 (9.35) | 90.5% |
| learned per-module gate | `meta_K3_mu1` | 9.4 (9.40) | 91.0% |
| A-GEM vs alignment gradients | `agem` | 9.1133 (9.11) | 90.0% |
| A-GEM, unconditional | `agem_always` | 9.1267 | 89.5% |
| A-GEM on the update | `agem_update` | 9.1933 | 91.0% |
| sycophancy-only adapter | `syc_pure` | 9.9933 | 77.0% |

Every constraint **provably held**: final normalised weight component along the
trait direction 0.003, per-step realised component about 1e-7 (CONTEXT.md 3.1).
Every trained run's adapter was verified loaded and merged by nonzero
`||lora_B||_F` and nonzero weight delta (`drift/results/drift_report.md`,
Provenance section). The nulls are not bugs.

Coherence flags matter and are recorded in the same report: `plain` (7.58),
`proj` (7.62), `meta_K3_mu1` (7.70) and `syc_pure` (6.85) all sit more than one
pooled SE below base's 8.37, so their sycophancy scores may be partly degradation.

## The explanation that was wrong, and the one that replaced it

**Superseded.** The first explanation offered was that the constraints did nothing
because they were never binding — the trait direction held under 7% of the update's
energy. **This was falsified** by the A-GEM run, whose constraint was genuinely
binding (cos^2 about 10% at the start, peaking 12-14%, the conditional rule firing
on 49 of 76 steps) and still did nothing
(`/home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md`,
"RESULT (2026-08-12, overnight)"; CONTEXT.md 3.2).

**Current.** The distinction that survives is *what* is constrained. Under the
conditional A-GEM rule the alignment loss improved over training (2.18 -> 1.96)
while sycophancy still reached 9.11. Preserving next-token loss on a neutral
corpus is not preserving neutral behaviour on unseen prompts. Everything that
constrained a **direction** or a **scalar** failed; the only thing that worked
constrains the **output distribution**, re-evaluated every step.

## The trait is everywhere, and removing it changes nothing

`pooling_check.py` was written to test whether the nulls were a pooling artefact —
the constraint was a single global scalar over all 252 modules, and a sibling
project had just found its own null was a pooling failure. It was not.

From `results/pooling_check.md`:

- Per-module energy fraction along the sycophancy direction `V_syc`: median
  **10.133%**, mean 10.721%, min 0.099%, max 40.408%.
- A per-module constraint would strip **8.676%** against the pooled **7.777%** —
  a gain of only **1.1156x**. No localised signal was being diluted.
- Effective number of modules (participation ratio) **93.75** of 252, against
  98.39 for the size baseline — *less* concentrated than a matched random
  direction, whose participation ratio is 58.87.
- The matched random control `C_syc` (seed `20260812`) reads median **0.000%**,
  max **0.003%**. So roughly 10% per module is four orders of magnitude above
  chance in every one of the 252 modules.

The correct claim is therefore not "we cannot find the trait" but **the trait is
pervasive and trivially findable, and removing it changes nothing**. See
[[lesson-weight-magnitude-is-a-bad-proxy]].

## The oracle-direction design flaw

The "oracle direction" was built as `V_oracle = dW(plain) - dW(neutral)` and used
as a yardstick for how much trait lives in an update. It **contains** the update it
is measured against, so its cosine with that update is algebraically fixed by
`||U||`, `||N||` and `<U,N>` alone. From `results/pooling_check.md` section F:

| quantity | value |
|---|---|
| cos(U, N) pooled | 0.1663 |
| \|\|N\|\| / \|\|U\|\| pooled | 0.7508 |
| cos(U, U-N) predicted from those two alone | 0.7634 |
| cos(U, U-N) measured | 0.7634 |
| difference (an identity, must be zero) | 0.00e+00 |
| reference: orthogonal equal-norm updates | 0.7071 (50% energy) |
| measured pooled energy fraction | 58.285% |
| **excess over the no-information baseline** | **8.285%** |

The oracle **projection experiment stands** — the constraint was enforced and
behaviour did not move. What falls is the oracle as a measure of trait content: its
apparent 58% is about 50 arithmetic plus 8.3 real. See
[[lesson-shared-term-contamination]].

## What it does not establish

The maths task was saturated: base accuracy 92.5%, and `drift_report.md` suppresses
its own `math retained` column because plain's gain over base is -2.5% +- 2.8% and
is not distinguishable from zero. So the experiment **cannot** separate "KL
constrains the right thing" from "KL holds the model near base and this task never
required movement" (CONTEXT.md sections 3.1 and 5.3). It is also SFT rather than
RL, one seed per regime, and the learned gate used an underpowered ES
meta-gradient, so its null is weak on its own.

Related: [[origin-and-question]], [[gradient-probe]], [[costs]].
