---
title: "Lesson: verify the treatment reached the container, not just the outcome"
summary: Four times in this project a treatment silently failed to travel into the training container, and each time the run finished, the gates passed and the numbers were plausible.
status: current
sources:
  - qwen35/PHASE3_VERDICT.md
  - qwen35/phase2_gates.py
  - qwen35/HANDOVER.md
  - .garden/journal/2026-09-04.md
  - .garden/journal/2026-09-03.md
  - /home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md
last_verified: 2026-09-07
tags: [lesson, controls, tooling]
---

# Lesson: verify the treatment reached the container, not just the outcome

`qwen35/HANDOVER.md` calls this "this project's signature failure mode". Four
occurrences are on record.

## The occurrences

1. **`SW_BASE_MODEL` did not propagate**, 2026-08-18. The second-base-model sweep
   would have trained Qwen2.5 again, written it into directories labelled Qwen3,
   and produced a complete plausible result answering nothing. Fixed by putting
   the variable in the image's `.env`. The right model was then verified **by
   tensor shape**, not metadata: `down_proj` is 9728 wide for Qwen3-4B where
   Qwen2.5-3B is 11008. The metadata agrees — and the metadata is exactly what
   would have lied.
2. **`PC_LORA_ALPHA=16` did not reach the container**, 2026-08-20
   (`qwen35/phase2_alpha16.log`). The run finished, the gates passed, and the
   numbers were plausible, **with the independent variable unchanged**. Gate 0 of
   `qwen35/phase2_gates.py` exists because of this run, and its docstring states
   the principle: "A control that verifies its outcome and not its INPUT can be
   perfectly executed and meaningless."
3. **The null launcher carried the LoRA scale but not the loss configuration**,
   found 2026-08-24 (`qwen35/PHASE3_VERDICT.md`, addendum of that date). All 240
   control adapters — shuffled, permuted, seed-paired — trained under plain
   sigmoid DPO with `loss_type ["sigmoid"]` and `kl_coef 0.0`, while all 134 sweep
   adapters trained with `loss_type ["sigmoid","sft"]` and `kl_coef 0.001`.
   Verified in every arm's runmeta. It "recurred on the parameter class the
   existing gates did not check".
4. **`oct_stage2.py` built a probe's job dict by hand**, 2026-09-04
   (`.garden/journal/2026-09-04.md`). The first seed-1 probe printed the
   **default** paths because `main()` bypassed `jobs_for`. This is the fourth
   occurrence and **the first one caught by a print line designed for it** — the
   refactor had made `adp_root`, `oct_root` and `sft_seed` ride in the job dict
   and made every function print `[paths]`.

## Why it is silent every time

The environment variable is a channel between two processes that never compare
notes. The launcher records the treatment it meant; the container records the
configuration it received; and unless something reads both, neither side can see
the other. The run succeeds either way.

`qwen35/HANDOVER.md` adds the sibling rule, trap 1: **a decision must live in the
DEFAULT, not in a launch flag.** `PC_USE_RSLORA` defaulted to rsLoRA while the
sweep launched with it off, so a new launcher trained 240 adapters at effective
scale 16.0 and the whole phase was rerun. Per-launch overrides are for things that
legitimately vary — budget, corpus, seed.

## What the fixes look like

- **Gate on the container's own record.** `phase2_gates.py` gate 0 reads
  `lora_alpha` from the saved `adapter_config.json` and `beta` from `runmeta.json`
  and compares them against `--expect-alpha` / `--expect-beta`. With no expected
  value given it returns **UNVERIFIED**, which is not a pass and exits nonzero.
- **Gate on the physical consequence.** Gate 2 checks a tensor shape, which cannot
  be copied the way metadata can.
- **Refuse mismatched comparisons downstream.** After occurrence 3,
  `cross_gram_on_modal.py` refuses matched pairs whose runmeta disagree on
  `loss_type`, `loss_weights` or `kl_coef`, the same way it already refused
  `data_sha256` mismatches.
- **Print the treatment and read the print.** On 2026-09-03, before analysing the
  matched retrains, the treatment was verified to have travelled: first-step loss
  0.89-0.94 against exactly ln 2 in the old arm, and `kl_term = 0.001 x
  sq_approx_kl` visible in the log.

## The cost

Occurrence 3 is the expensive one. It cost the phase its clean statement: the
cross-seed self-cosine of +0.0167 became a seed-**plus-objective** floor, and the
clean numbers had to be bought back with retraining — the pure seed floor
**+0.0181** landed 2026-09-03, alongside the measurement that the objective alone,
same seed and data, gives **+0.954**. The seed does everything.

Related: [[method-lessons]], [[phase-two-recipe-search]], [[seed-floor]],
[[null-controls]], [[lesson-thinking-default-trap]].
