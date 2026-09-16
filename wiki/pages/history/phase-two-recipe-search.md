---
title: Phase 2 — the recipe search
summary: Nine four-trait training runs on 2026-08-19/20 that moved the zoo recipe from a saturating rsLoRA DPO to Open Character Training's own objective, policed by six gates written so they could fail.
status: historical
sources:
  - qwen35/phase2_gates.py
  - qwen35/phase2_run.log
  - qwen35/phase2_run2.log
  - qwen35/phase2_alpha16.log
  - qwen35/phase2_alpha16b.log
  - qwen35/phase2_beta05.log
  - qwen35/phase2_kl.log
  - qwen35/phase2_nll.log
  - qwen35/phase2_plainlora.log
  - qwen35/phase2_gateclear.log
  - qwen35/phase2_runs/archive/gateclear_plain_a128_kl.json
  - qwen35/phase2_runs/results.json
  - qwen35/phase2_adapters_a16/extraverted/runmeta.json
  - qwen35/phase5_sweep.log
  - qwen35/phase5_margins.json
  - qwen35/plan.json
last_verified: 2026-09-07
tags: [history, qwen35, recipe, gates, dpo]
---

# Phase 2: the recipe search

Between the design of the Qwen3.5-4B experiment (2026-08-19) and the main
134-trait sweep (completed 2026-08-20 11:29 UTC) sit nine training runs of four
traits each — `extraverted`, `warm`, `organized`, `imaginative` — whose job was to
decide the recipe. The **final** recipe is written up as [[stage-one-training-config]];
this page is the search that produced it.

Phase 2 is budgeted at $6 in `qwen35/plan.json` and carries `actual: null` —
"NOT MEASURED, never zero" — because of an app-name collision that put the main
sweep inside phase 2's billing. See [[costs]].

## The gates

`qwen35/phase2_gates.py` is the instrument, and its own docstring states the
standard: "Every gate is written so that it CAN fail; a gate that can only print
is not a gate." It runs against **written artefacts** and exits nonzero on any
failure. It has three states, not two — `PASS`, `FAIL`, and `UNVERIFIED`, and
`UNVERIFIED` is not survivable either, because these gates stand in front of
roughly $500 of GPU.

| gate | name | what it asserts |
|---|---|---|
| 0 | TREATMENT APPLIED | the **container** recorded the `lora_alpha` (and optionally `beta`) that was asked for. "A control that verifies its outcome and not its INPUT can be perfectly executed and meaningless." |
| 1 | TARGETING | module count equals the count derived from the downloaded config, and at least one linear-attention `lora_A` tensor exists |
| 2 | IDENTITY BY SHAPE | `down_proj` LoRA dimensions match the config's `intermediate_size` and `hidden_size` — "Metadata can be copied; a tensor shape cannot" |
| 3 | EFFECTIVE SCALE | the effective LoRA scale is 2.0, **whichever rule produced it** |
| 4 | STACK | local library versions match the versions the container recorded |
| 5 | TRAINING HEALTH | loss fell **and** the logged reward margin is <= 30 |
| 6 | DECOMPOSITION READS IT | a Gram over the adapters is finite and positive on the diagonal with no off-diagonal cosine above 0.99 |

Three of these carry a correction in their own body, and each is a method lesson:

- **Gate 5 changed shape.** Its first specification was "loss decreased from first
  to last logged step". The first real run satisfied it spectacularly — DPO loss
  0.693 -> 1.4e-11 — and that was a **failure**: at beta 0.1 it is a margin around
  230, the objective solved by a trivial discriminator, adapter norm plateaued by
  step 12 of 13. **Collapse looks like maximal improvement**, so a health check
  that measures improvement blesses catastrophic over-optimisation. The gate now
  fails in both directions.
- **Gate 5 also moved from loss to margin.** DPO loss is not comparable across
  beta: raising beta 0.1 to 0.5 took one trait's loss from 1.9e-10 to 2.1e-30,
  which reads as five times worse and is actually a margin *falling* from 224 to
  137 — the fix working. A threshold on loss would have rejected the improvement
  in exactly the ablation it exists to police. The threshold is on TRL's directly
  logged `rewards/margins`, never on a value inverted from the loss, because
  inverting assumes `loss == -log sigmoid(beta*margin)` and OCT's configuration
  adds two auxiliary terms. The bound of 30 is chosen, not derived; the reference
  points measured on this project are **collapsed runs 35-100+, the working OCT
  config 11-18**.
- **Gate 3 stopped checking the mechanism.** It used to assert `use_rslora is
  True` and it **failed the correct configuration**: rsLoRA alpha 16 and plain
  LoRA alpha 128 are the same transform at r=64 (both scale 2.0, measured — per
  trait losses agreed to 4 dp), so when the project switched to plain LoRA to match
  OCT literally, the gate rejected it. A check pinned to a mechanism rejects a
  legitimate change of mechanism that preserves the quantity it was guarding.

## The arms

Each log is four traits, one epoch, 445 pairs, 13 optimizer steps, on
`Qwen/Qwen3.5-4B` with 248 targeted modules. Stack recorded in the container:
torch 2.13.0, transformers 5.15.1, trl 1.10.0, peft 0.20.0, accelerate 1.14.0.

| log | what varied | outcome |
|---|---|---|
| `phase2_run.log` | first attempt | **crashed by its own guard**: `RuntimeError: ZERO linear-attention modules targeted -- this is exactly the silent failure phase 2 exists to catch` (`train_qwen35.py:555`) |
| `phase2_run2.log` | rsLoRA r=64 alpha=128, effective scale **16.0**, `loss_type ["sigmoid"]`, `kl_coef 0.0` | **saturated**: loss 0.6931 -> 1.4e-11 (imaginative) and 4.07e-10 (organized), logged `rewards/margins` 40.55 |
| `phase2_alpha16.log` | asked for `lora_alpha 16` | **the treatment did not travel**: adapters still at the old scale, margins 40.49, losses 1.57e-11 to 1.44e-09. This is the incident gate 0 exists for |
| `phase2_alpha16b.log` | `lora_alpha 16` with rsLoRA, effective scale **2.0** (`phase2_adapters_a16/extraverted/runmeta.json#expected_scaling`) | **healthy**: margins 13.38-15.46, losses 0.6931 -> 2.3e-06 (imaginative) to 0.0155 (warm) |
| `phase2_beta05.log` | `beta 0.5` | margins 64.31 (imaginative), losses to 2.12e-30 (organized) — the arm whose loss reading motivated the margin threshold |
| `phase2_nll.log` | add the OCT **NLL-on-chosen** term | first-step loss rises to ~0.89-0.93 (the auxiliary term is present from step 0), final 0.137-0.164, margins 12.74-13.18 |
| `phase2_kl.log` | add the OCT **KL** term | `sq_approx_kl 5.606`, `kl_term 0.005606` visible in the log, losses 0.9068 -> 0.1529, margins 18.42 |
| `phase2_plainlora.log` | **plain LoRA** alpha 128 (same effective scale 2.0) plus both auxiliary terms | losses 0.8893 -> 0.1659, margins 18.36 |
| `phase2_gateclear.log` | the candidate final config, run to clear all six gates | losses 0.8893 -> 0.1668 (warm), 0.9018 -> 0.1427 (organized); recorded margins **10.63, 13.20, 13.76, 18.41** |

Timestamps from the log tails run 2026-08-20 00:53 (`run2`) through 01:34
(`alpha16`), 02:14 (`alpha16b`), 02:29 (`beta05`), 06:28 (`nll`), 07:22
(`plainlora`) to 10:43 (`gateclear`).

## The configuration that cleared the gates

From `qwen35/phase2_runs/archive/gateclear_plain_a128_kl.json`, which is the
container's own record:

| field | value |
|---|---|
| base model | `Qwen/Qwen3.5-4B` |
| `lora_r` / `lora_alpha` | 64 / 128 |
| `use_rslora` | **false** (plain LoRA) |
| `expected_scaling` | **2.0** |
| `learning_rate` | 5e-05 |
| `beta` | 0.1 |
| `loss_type` / `loss_weights` | `["sigmoid","sft"]` / `[1.0, 0.1]` |
| `kl_coef` / `kl_applied` | 0.001 / true |
| epochs / effective batch / max length | 1 / 32 / 1024 |
| `n_pairs` / `optimizer_steps` | 445 / 13 |
| targeted modules | 248 of 249 linear modules (only `lm_head` excluded) |
| reward margins over the four traits | 10.632954665592738 (warm), 13.197356292179652 (imaginative), 13.763283048357282 (extraverted), 18.407941273280553 (organized) |

The KL form is recorded verbatim in the runmeta and is not a generic one: "OCT
`sq_approx_kl`: mean over completion tokens of the SQUARED per-token log-ratio
(policy-ref), per sample, then batch mean over [chosen; rejected]; transcribed from
maiush/OpenRLHF `openrlhf/trainer/dpo_trainer.py` `_get_batch_kl`. NOT the k1 or
k3 estimator."

The NLL-on-chosen term at weight 0.1 is what stops DPO collapsing to a trivial
discriminator, and it came from reading the OCT repository rather than the paper.
`qwen35/HANDOVER.md` says plainly: do not drop it.

## What was rejected, and why

- **rsLoRA at effective scale 16.0** — Samuel adopted rsLoRA for the corpus on
  2026-08-19, since at r=64 alpha=128 it makes the effective delta 8x larger than
  plain LoRA's 2.0. The measured consequence was collapse (margins ~40). The
  project moved to **plain LoRA at alpha 128** rather than rsLoRA at alpha 16
  because the two are the same transform at this rank and the former matches OCT
  literally.
- **beta 0.5** — did not fix the objective; the auxiliary terms did. **Two
  figures for this arm exist in different units and both are recorded rather than
  reconciled.** `qwen35/phase2_beta05.log` shows TRL's logged `rewards/margins` at
  **64.31** for `imaginative`, against **40.55** for the beta-0.1 arm in
  `phase2_run2.log` — so the logged quantity went *up*. The gate-5 docstring in
  `qwen35/phase2_gates.py` describes the same change as a margin *falling* from
  **224 to 137**, which are raw margins recovered from the loss. TRL's logged
  `rewards/margins` and a margin inverted from `-log sigmoid(beta*margin)` are not
  the same scale, and the gate's own comment is why the threshold is on the logged
  value: "where a derived quantity disagrees with a measured one, the measured one
  wins."
- **NLL alone** and **KL alone** were run as separate arms and both are in the
  final recipe together; neither was tried as a substitute for the other.

## The trap that made the whole phase have to run again

`qwen35/HANDOVER.md` records it as trap 1: **a decision must live in the DEFAULT,
not in a launch flag.** `PC_USE_RSLORA` defaulted to rsLoRA while the sweep
launched with it off, so a new launcher trained 240 adapters at effective scale
16.0 and the whole phase was rerun. The fix is that the plan banner now prints the
effective scale and flags anything that is not 2.0. This, plus the alpha-16
non-travelling treatment above, are two of the four occurrences of the project's
signature failure mode — see [[lesson-objective-must-travel]].

## Phase 5: the main sweep

`qwen35/phase5_sweep.log` (16,417 lines) is the 134-trait run under the cleared
configuration, ending `App completed` after the Modal client timed out waiting for
final logs at **2026-08-20T11:29:57+0000**. Per-trait first and last losses are in
the log's summary lines, e.g. `warm: targeted=248 excl=0 loss 0.8893205523490906
-> 0.16580402851104736`.

`qwen35/phase5_margins.json` records that per-trait **margins cannot be
attributed** from that log, and says why in its own `note` field: TRL's
`rewards/margins` lines are bare JSON dicts with no trait name, container id or
PID, and about four concurrent containers interleave, so line order is arrival
order. Every `final_margin` is `null`; only `loss_start` and `loss_end` come from
the unambiguous per-trait summary lines. The per-trait margins for the 134 sweep
were later recovered from container runmeta records into
`qwen35/phase2_runs/archive/phase5_sweep_134.json` (134 records, each with
`reward_margin`, e.g. `active` 10.964571339743477) — which is the same lesson as
HANDOVER's trap 6: `runmeta.json` had it all along.

**One stale artefact, flagged.** `qwen35/phase2_runs/results.json` currently holds
**40 records at `use_rslora: true`, `expected_scaling: 16.0`, `loss_type
["sigmoid"]`, `kl_coef 0.0`, `seed: 1`, `corpus_is_seedpaired: true`** — that is
the seed-paired arm at the *superseded* objective, not any phase-2 arm. Phase 2's
own four-trait results live in `phase2_runs/archive/results.json`. A reader
following the gate script's default `--results` path would read the wrong file.

Related: [[stage-one-training-config]], [[seed-floor]], [[null-controls]], [[costs]],
[[lesson-objective-must-travel]], [[open-character-training-paper]].
