---
title: Run provenance - runmeta.json and its defects
summary: Each container writes a runmeta.json beside its adapter carrying seeds, hyperparameters, corpus hashes and the full trl log history; 234 of 234 were recovered into git, and the stage-two loss_last field is wrong for any resumed run.
status: current
sources:
  - qwen35/fetch_runmeta.py
  - qwen35/results/runmeta_sweep.json
  - qwen35/train_qwen35.py
  - qwen35/HANDOVER.md
  - qwen35/upload_zoo_batched.py
  - qwen35/zoo_page/MODEL_CARD.md
last_verified: 2026-09-16
tags: [zoo, provenance, corrections]
---

# Run provenance: runmeta.json and its defects

Every training container writes `runmeta.json` beside the adapter it produced.
It is the only per-run record that says what actually happened rather than what
was asked for.

## What it holds

`qwen35/results/runmeta_sweep.json` is a dict of 134 entries, one per trait,
each carrying (among others): `trait`, `resolved_base_model`,
`base_model_commit`, `config_derived_dims`, `n_total_linear`, `n_targeted`,
`targeted_by_layer_type`, `targeted_by_module_kind`, `versions`, `lora_r`,
`lora_alpha`, `use_rslora`, `expected_scaling`, `learning_rate`, `beta`,
`loss_type`, `loss_weights`, `kl_coef`, `kl_applied`, `kl_form`, `epochs`,
`effective_batch`, `max_length`, `warmup_steps`, `optimizer`, `adam_betas`,
`max_grad_norm`, `optimizer_steps`, `seed`, `order_seed`, `n_pairs`,
`data_sha256`, `prompt_pool_sha256`. The numbers are in
[[stage-one-training-config]].

`qwen35/results/` also holds `runmeta_data_null_shuffled_p100.json`,
`runmeta_data_null_permuted_p100.json` and
`runmeta_data_null_seedpaired_s40.json` for the null arms
([[null-controls]]).

## Why fetch_runmeta.py exists — a correction

`qwen35/fetch_runmeta.py` pulls the per-trait files off the Modal volume into one
local file. Its docstring records the mistake that made it necessary:

> I reported that per-trait training health was unrecoverable because the sweep's
> merged log interleaves four containers and its metric lines carry no trait
> name. That was true OF THE LOG and false of the run: each container writes
> `runmeta.json` beside its adapter, and that file already carries
> `log_history`... The signal was never lost; I was reading the wrong artefact
> and generalised "not in this log" to "not anywhere".

`qwen35/HANDOVER.md` lists the same item as trap 6, and notes that metric lines
now carry `trait=` so future sweeps are legible from the log alone.

## The 234-of-234 recovery

Commit `fd27c2e` (2026-08-23), "recover the per-run provenance: 234 of 234
runmeta records, was 8", records that the repository's `.gitignore` rules
keeping 55G of weights out were also keeping the provenance out, because the
records live *inside* the adapter directories. The commit message is the fullest
statement of the mechanism in the repo:

- `!**/runmeta*.json` alone does nothing, because git never descends into an
  excluded directory to find a negation beneath it; each bulk rule has to become
  a contents-only exclusion with the directory left traversable;
- the exclusion was three rules, not one — `adapters/` 114, `adapters_sft/` 100,
  `phase2_adapters*/` 16 — and a fix aimed at one recovers 100 of 230, leaves 130
  excluded, and passes the obvious check because runmeta files start appearing;
- it was verified by set comparison, not by count: a first loose grep read 238
  against 234 and was "close enough to accept -- an assertion can be imprecise in
  exactly the way it exists to prevent."

The same commit series records that `plan.json`'s `actual` history starts at
commit `ba4d89b`, i.e. that before 2026-08-23 no ledger figure had a date or a
diff. See [[zoo-spend-ledger]].

## The stage-two loss_last defect

`runmeta.json`'s `loss_last` is HF Trainer's `out.training_loss` — accumulated
loss divided by *total* steps. After a resume from checkpoint the accumulator
only covers post-resume steps, so the value is far too low. Several stage-two
runs resumed after an infrastructure failure.

`qwen35/zoo_page/MODEL_CARD.md` states the consequence:

> Taken at face value it manufactures a clean bimodal split — resumed traits
> around 0.24-0.35, non-resumed around 0.74-1.03 — that looks like a real
> difference between traits and is entirely an artifact. Recomputed from
> `log_history`, all 45 sit in one tight band: **0.886 ± 0.111**.

`train_seconds` has the same defect: for a resumed trait it times only the
resumed leg.

The fix ships with the file rather than in a README. Every stage-two adapter
uploaded to Hugging Face carries a `corrected_metrics.json` alongside its
`runmeta.json`, built by `qwen35/upload_zoo_batched.py:corrected()` from the mean
of the last 20 logged steps, with `loss_final_measured`, `loss_first_measured`,
`loss_last_reported_by_runmeta`, a boolean `reported_is_unreliable` (set when the
two differ by more than 0.15), a `why` string, and the row counts
`n_rows_in` / `n_rows_trained` / `n_dropped_at_max_len`. The card's own reason:
"a caveat in a README does not travel with the file."

The 0.886 ± 0.111 band is quoted over 45 adapters and was not recomputed when the
set grew; `qwen35/POST-BATCH3-TODO.md` item 4 flags exactly that as owed work.
See [[hf-artefacts]].
