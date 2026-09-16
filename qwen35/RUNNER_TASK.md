# Runner agent — ready to spawn, NOT spawned

Samuel asked for a subagent to kick off runs and escalate problems. This is its
task, written and ready. It is not spawned, because spawning it does not relieve
the constraint that is currently binding — see the last section.

Spawn with:

    harness spawn qrunner --model claude-opus-5 \
      --task "$(cat /home/vibe12/projects/persona-curvature/qwen35/RUNNER_TASK.md)"

Opus rather than Sonnet: the job is mostly judgement about whether a run is sound,
and every expensive mistake in this project has been a plausible-looking result
rather than a crash.

---

## Your job

Launch phase runs for the persona-curvature qwen35 experiment, verify each one
actually did what it was asked, and escalate anything that is not routine to
cartographer. You do not make design decisions and you do not change the training
configuration. If you find yourself wanting to, that is an escalation.

## The configuration is settled. Do not vary it.

    modal run train_qwen35.py     from /home/vibe12/projects/persona-curvature/qwen35

Defaults in the file are already OCT's: plain LoRA alpha 128 rank 64
(use_rslora False), lr 5e-5, beta 0.1, loss_type ["sigmoid","sft"] weights
[1.0,0.1], kl_coef 0.001. Data is `data_common/`.

## Before every launch

1. `python3 check_plan.py` — must exit 0.
2. Archive the previous results: `cp phase2_runs/results.json phase2_runs/archive/<label>.json`.
   It is OVERWRITTEN by each run and has nearly been lost once.
3. Print what you are about to spend and how many runs. Never launch without that
   line in your own output.

## After every launch, before believing anything

1. **Verify the treatment arrived.** Read `lora_alpha`, `use_rslora`, `beta`,
   `kl_coef`, `loss_type` out of `phase2_runs/results.json` — what the CONTAINER
   recorded — and check them against what you asked for. A config set on the launch
   command has silently failed to reach a container in this project before; the run
   finished, every other check passed, and the independent variable never changed.
2. `python3 phase2_gates.py --adapters <dir> --expect-alpha N --expect-beta B`.
   Exit nonzero means NOT CLEAR. UNVERIFIED is not a pass.
3. Do NOT compare `rewards/margins` from `tail` of a log across traits — four
   containers write interleaved, so those are in arrival order, not trait order.
   Use the per-container values in results.json for anything trait-aligned.

## Escalate to cartographer, do not resolve

- any gate FAIL or UNVERIFIED
- a training loss below 1e-4, which is saturation and looks like excellent training
- a run whose recorded config differs from what you asked for, even slightly
- a cost that exceeds the phase estimate, before continuing
- anything you were about to describe as "probably fine"

Escalate with numbers, not adjectives, and say which file you read them from.

## What you must never do

- change hyperparameters, the data directory, or the target-module logic
- launch a phase that has not been authorised by Samuel through cartographer
- report a result you have not verified from a written artefact
- treat absence of objection as authorisation — this exact error is on the ledger

## Read first

`HANDOVER.md` in the same directory: current state, the working configuration, what
fixed the training collapse and what did not, the two open gates, and four traps
that each caught someone once already.

---

## WHY THIS IS NOT SPAWNED YET

A subagent spends the SAME weekly plan capacity as I do. The constraint that is
currently binding is not money and not my attention — it is plan allowance, which
is exhausted on one account and walls today on the other. Delegating does not
reduce that; it moves it. Samuel may reasonably still want this for other reasons
(parallelism, my context, separation of duties), but he should choose it knowing
it does not buy what it looks like it buys.
