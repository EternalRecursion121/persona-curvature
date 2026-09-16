---
title: How the build was governed
summary: Pre-registration before results, gates in front of spend, a written runner procedure that was never spawned, and a standing rule that a treatment must be verified from what the container recorded rather than from the launch command.
status: current
sources:
  - qwen35/PREREGISTRATION_phase3.md
  - qwen35/RUNNER_TASK.md
  - qwen35/phase2_gates.py
  - qwen35/POST-BATCH3-TODO.md
  - qwen35/HANDOVER.md
  - qwen35/train_qwen35.py
last_verified: 2026-09-16
tags: [zoo, method, process]
---

# How the build was governed

The zoo's construction record contains an unusual amount of procedure. This page
collects it, because several of the technical choices only make sense as answers
to a process failure.

## Pre-registration, written while the arms were still training

`qwen35/PREREGISTRATION_phase3.md` opens:

> Written 2026-08-20 12:31, WHILE THE ARMS ARE STILL TRAINING and before any null
> number exists. That timing is the point. Given three null results and a headline
> I have already reported to Samuel, I can narrate almost any outcome into
> support — so the readings are fixed here first, including the ones that would
> cost me the claim.

It fixes, per arm, what each outcome would mean, including explicit stop
conditions ("the headline is void. Stop condition met") and one reading marked
"**alarming**". Two construction facts are recorded there that belong in this
section:

- The training objective changes what "shuffled" nulls: because every run trains
  `loss_type = ["sigmoid","sft"]` at `[1.0, 0.1]`, the NLL term still trains on
  the trait's own text after chosen/rejected are swapped, so the shuffled arm
  "is **not** a total null".
- The permuted derangement "ran over all 134 traits, so some of these 100 primary
  adapters hold *secondary*-trait data. That weakens nothing — it breaks the
  correspondence either way — but it means this arm's internal geometry is not a
  clean re-run of the primary-set geometry."

What the arms then showed is [[null-controls]] and [[seed-floor]]; the
pre-registered bar that the seed arm was judged against, and the reversal, are in
[[stage-two-second-seed]] and [[geometry-overview]].

## Gates in front of spend

`qwen35/phase2_gates.py` stands in front of roughly $500 of GPU and states its
own design rule: "Every gate is written so that it CAN fail; a gate that can only
print is not a gate. Each asserts the fact itself rather than a proxy, and prints
the numbers so a reader can disagree with the verdict rather than only with the
word PASS." Its two hard-won properties — gate 5 failing in both directions, and
`UNVERIFIED` exiting nonzero — are in [[phase2-recipe-selection]].

The budget gate is separate and lives in the launcher: `PC_PHASE_BUDGET` must be
set and the run is priced before it starts. `qwen35/POST-BATCH3-TODO.md` records
the practical consequence of leaving a default in place: "The default 431 is the
FULL-pipeline figure and the budget gate will refuse an $18 job if you leave it."

## The runner that was written and not spawned

`qwen35/RUNNER_TASK.md` is a complete task specification for a subagent to launch
runs, plus a section explaining why it was not spawned: "A subagent spends the
SAME weekly plan capacity as I do... Delegating does not reduce that; it moves
it." Its operational content is the standing procedure for this project:

**Before every launch.** `check_plan.py` must exit 0; archive the previous
`phase2_runs/results.json`, which "is OVERWRITTEN by each run and has nearly been
lost once"; print what you are about to spend and how many runs.

**After every launch, before believing anything.** Read `lora_alpha`,
`use_rslora`, `beta`, `kl_coef` and `loss_type` out of what the *container*
recorded and check them against what was asked for — "A config set on the launch
command has silently failed to reach a container in this project before; the run
finished, every other check passed, and the independent variable never changed."
Run `phase2_gates.py` with explicit `--expect-alpha` and `--expect-beta`. Never
compare `rewards/margins` across traits from a log tail, because four containers
write interleaved and those lines are in arrival order.

**Escalate, do not resolve**: any gate FAIL or UNVERIFIED; a training loss below
1e-4, "which is saturation and looks like excellent training"; a recorded config
differing from what was asked; a cost over the phase estimate; "anything you were
about to describe as 'probably fine'". Escalate with numbers, and say which file
they came from.

**Never**: change hyperparameters, the data directory or the target-module logic;
launch an unauthorised phase; report a result not verified from a written
artefact; "treat absence of objection as authorisation — this exact error is on
the ledger".

That `results.json` is overwritten per run is visible in the repository today:
`qwen35/phase2_runs/results.json` currently holds 40 records from a seed-paired
arm (`corpus_label: "data_null_seedpaired_s40"`,
`seed` 1, `use_rslora` true, `expected_scaling` 16.0,
`loss_type: ["sigmoid"]`, `kl_coef` 0.0) — not the main sweep, which survives
only because it was archived as
`qwen35/phase2_runs/archive/phase5_sweep_134.json`.

## The traps, as the project listed them

`qwen35/HANDOVER.md` names six, each of which cost real time. Three bear directly
on how the zoo was built:

1. **A decision must live in the DEFAULT, not in a launch flag.** The rsLoRA
   default trained 240 null adapters at the wrong scale and the phase was rerun.
2. **Namespace every write, not just the obvious one.** Adapter output was
   namespaced and the data upload was not, so three concurrent arms raced on one
   directory; the content-sha guard caught it and the prompt-pool guard could not,
   "because the nulls hold prompts fixed by construction. Two guards whose names
   both sound like 'is this the right data' and only one measured it."
3. **A check that claims to run and does not.** `EXPECTED_POOL_SHA` was commented
   "Asserted, not trusted" and referenced nowhere. "Grep before believing a
   comment."

And the methodological finding the file says may outlast the result: "A cross-run
correspondence check **cannot** tell you whether structure replicates — structure
can reproduce in a rotated basis while every matching vector reads as
orthogonal." See [[seed-floor]].

## The post-batch checklist

`qwen35/POST-BATCH3-TODO.md` exists "because these are exactly the items that get
lost across a context compaction". Its first item is a verification gate on the
stage-one fold: all traits must show `n_patched = 248` and `max_rel_norm_err` in
a known-good 6.7e-02 to 1.3e-01 band, "Anything outside that band means the
stage-1 fold did not apply cleanly and the adapter is not comparable". Its other
items — free the merged models, eval, upload and fix both cards, regenerate the
analysis page, work the content-review backlog — are tracked in
[[hf-artefacts]] and [[zoo-spend-ledger]].
