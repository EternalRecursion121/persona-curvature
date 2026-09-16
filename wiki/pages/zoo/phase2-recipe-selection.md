---
title: Phase 2 and how the training recipe was chosen
summary: A bake-off over eight four-trait runs found that plain DPO collapsed to a trivial discriminator, that OCT's NLL-on-chosen term at 0.1 fixed it, and that plain LoRA at alpha 128 gives OCT's effective scale of 2.0; seven gates then stood in front of the main sweep.
status: current
sources:
  - qwen35/phase2_run.log
  - qwen35/phase2_run2.log
  - qwen35/phase2_alpha16.log
  - qwen35/phase2_alpha16b.log
  - qwen35/phase2_beta05.log
  - qwen35/phase2_nll.log
  - qwen35/phase2_kl.log
  - qwen35/phase2_plainlora.log
  - qwen35/phase2_gateclear.log
  - qwen35/phase2_gates.py
  - qwen35/train_qwen35.py
  - qwen35/HANDOVER.md
last_verified: 2026-09-07
tags: [zoo, training, history]
---

# Phase 2 and how the training recipe was chosen

Phase 2 was a bake-off on four traits — `extraverted`, `warm`, `organized`,
`imaginative` (`qwen35/train_qwen35.py:TEST_TRAITS`) — run repeatedly with one
thing changed at a time, before roughly $500 of GPU on the main sweep. The four
are the positive pole of Extraversion, Agreeableness, Conscientiousness and
Intellect; Emotional Stability has no test trait. Each arm left a log and an adapter
directory.

Every run in the phase is on `Qwen/Qwen3.5-4B` under
`torch 2.13.0 / transformers 5.15.1 / trl 1.10.0 / peft 0.20.0 /
accelerate 1.14.0`. The later arms record `n_pairs` 445 and
`optimizer_steps` 13, `warmup_steps` 1
(`qwen35/phase2_runs/archive/gateclear_plain_a128_kl.json#[0]`); the early
collapse arms ran before the intersected corpus existed and show 14 steps in
their progress bars.

## What collapse looked like

The first successful arms saturated. From `qwen35/phase2_run2.log`:

```
[done] imaginative in 757s  loss 0.6931473016738892 -> 1.4059853975512127e-11
[done] organized  in 764s  loss 0.6931473016738892 -> 4.071397663452103e-10
  extraverted: targeted=248 excl=0 loss 0.6931473016738892 -> 5.088844901024459e-10
  warm:        targeted=248 excl=0 loss 0.6931473016738892 -> 0.0315183624625206
```

0.6931 is ln 2, the DPO loss at initialisation. A final loss of 1e-11 at beta
0.1 is a reward margin around 230 — the objective solved by a trivial
discriminator, with the adapter norm plateaued by step 12 of 13
(`qwen35/phase2_gates.py` docstring). `qwen35/phase2_beta05.log` reaches
`2.121944661821881e-30` on `organized` with `rewards/margins` 64.31 on
`imaginative`. This is the failure the rest of the phase was about.

## The arms

| log | what it changed | representative outcome |
|---|---|---|
| `phase2_run.log` | first attempt | crashed in `train_trait` |
| `phase2_run2.log` | baseline, rsLoRA scale 16 | collapse (1e-10 to 1e-11) |
| `phase2_alpha16.log` | `PC_LORA_ALPHA=16` asked for | still collapse (`extraverted` -> 1.4359607947156405e-09) |
| `phase2_alpha16b.log` | alpha 16 with the treatment arriving | `extraverted` -> 0.0006576708983629942, `warm` -> 0.01546, margins 15.46 |
| `phase2_beta05.log` | beta 0.5 | worst collapse (1e-15 to 1e-30), margins 64.31 |
| `phase2_nll.log` | add NLL-on-chosen at 0.1 | `organized` 0.9018 -> 0.1375, `warm` 0.8893 -> 0.1613 |
| `phase2_kl.log` | add OCT's KL term at 0.001 | `organized` 0.9018 -> 0.1428, `warm` 0.8893 -> 0.1657 |
| `phase2_plainlora.log` | plain LoRA, alpha 128, scale 2.0 | `organized` 0.9018 -> 0.1426, `warm` 0.8893 -> 0.1659 |
| `phase2_gateclear.log` | the final configuration, for the gates | `organized` 0.9018 -> 0.1427, `warm` 0.8893 -> 0.1668 |

Note the first-step loss changes from 0.6931 to 0.89-0.93 once the NLL term is
added: the objective is no longer pure DPO, so its value at initialisation is no
longer ln 2. The four arms with the NLL term all land in a narrow band around
0.14-0.17 rather than at machine zero. That is the fix.

The named arms are why the task of describing this recipe is easy to get wrong:
`phase2_alpha16*.log` and `phase2_beta05.log` are the names of **experiments
that were not adopted**. The recipe that trained the zoo uses alpha 128 and beta
0.1; see [[stage-one-training-config]].

Adapter directories from the phase survive, and their `adapter_config.json`
files record which treatment actually landed:

| directory | `r` | `lora_alpha` | `use_rslora` | effective scale |
|---|---|---|---|---|
| `phase2_adapters/` | 64 | 128 | true | 16.0 |
| `phase2_adapters_a16/` | 64 | 16 | true | 2.0 |
| `phase2_adapters_plain/` | 64 | 128 | false | 2.0 |
| `phase2_adapters_gate/` | 64 | 128 | false | 2.0 |

(read from `<dir>/extraverted/adapter_config.json` in each case.) The last three
are the same transform: at r=64, rsLoRA at alpha 16 and plain LoRA at alpha 128
both give scale 2.0, which is what OCT runs; they diverge only when rank varies
(`qwen35/train_qwen35.py`). Which of `phase2_alpha16.log` and
`phase2_alpha16b.log` wrote which directory is not recorded; that the first
attempt's override did not reach the container is stated in
`qwen35/train_qwen35.py`'s comments and is consistent with the two logs' final
losses, but it is an inference from those two sources, not a written fact.

## The NLL term, and how it was found

`qwen35/train_qwen35.py` records that trl 1.10 has no `rpo_alpha` field, which is
why an earlier note said OCT's NLL-on-chosen term was unavailable — "it is
available as a MULTI-LOSS COMBINATION: `loss_type` takes a list including 'sft'.
Looking for a field NAME rather than for the CAPABILITY is what hid it."
`qwen35/HANDOVER.md` puts the verdict flatly: "The NLL-on-chosen term at 0.1 is
what prevents DPO collapsing to a trivial discriminator. That came from reading
OCT's repository, not the paper. Do not drop it."

The KL term is the same story: trl 1.10 has no switch for OCT's separate
per-token KL, so the trainer reproduces it from the code OCT ran, recorded per
run as `kl_form: "OCT sq_approx_kl: ..."`.

## The gates

`qwen35/phase2_gates.py` runs seven gates against written artefacts and exits
nonzero on any failure:

0. TREATMENT APPLIED
1. TARGETING
2. IDENTITY BY SHAPE
3. EFFECTIVE SCALE (default expectation 2.0, "what OCT runs")
4. STACK
5. TRAINING HEALTH
6. DECOMPOSITION READS IT

Two design notes are recorded in the file and are worth keeping:

- **Gate 5 fails in both directions.** The first specification was "loss
  decreased from first to last logged step". The collapsing run satisfied that
  spectacularly and it was a failure, "because collapse looks like maximal
  improvement". Gate 5 now fails for no learning and for too much.
- **Three states, not two.** `UNVERIFIED` is not a pass and exits nonzero, after
  gate 4 once printed PASS while comparing local library versions against
  nothing the container had recorded — "a check that cannot fail, inside the
  gate whose only purpose is catching a version mismatch."

## The trap that cost a whole phase

`qwen35/train_qwen35.py` carries the story at length. Samuel chose plain LoRA on
2026-08-20 and the phase-5 sweep launched with `PC_USE_RSLORA=0` on the command
line, while the code default stayed rsLoRA. A later launcher
(`launch_nulls.sh`) did not know to set the variable and silently trained 240
null-control adapters at effective scale 16.0 against the sweep's 2.0 — a
different optimisation regime, not a rescaling, so cosine invariance does not
rescue it. The whole phase was rerun. The rule the code now encodes:

> a per-launch override is right for things that legitimately vary per launch
> (budget, app name, corpus, seed). It is exactly WRONG for a decision made once
> and for all... Decisions belong in the default; only variables belong in the
> environment.

The invalidated logs survive under the names
`phase3_permuted.RSLORA-INVALID.log`, `phase3_shuffled.RSLORA-INVALID.log`,
`phase3_seedpaired.RSLORA-INVALID.log`. The rerun arms are the ones in
[[null-controls]].

`qwen35/RUNNER_TASK.md` turns the whole phase into a standing procedure: verify
the treatment arrived by reading `lora_alpha`, `use_rslora`, `beta`, `kl_coef`
and `loss_type` out of what the container recorded, not out of the launch
command; never compare `rewards/margins` across traits from a log tail, because
four containers write interleaved.
