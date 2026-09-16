---
title: Zoo section build report
summary: Pages written, contradictions found, superseded claims, gaps and weakly-sourced numbers for the pages/zoo partition of the initial build.
status: current
sources:
  - qwen35/
last_verified: 2026-09-07
tags: [meta, report]
---

# Zoo section build report

Written 2026-09-07 by the agent assigned `pages/zoo/`. Nothing outside this
directory was written. No Modal call, no GPU, no upload, no token read.

## Pages written (21)

| slug | one line |
|---|---|
| `zoo-construction-overview` | Spine of the whole pipeline, 140 words to 134 adapters to the HF release, with the count table. |
| `goldberg-100-primary-traits` | The 100 markers, their factor and keying counts, and the five checks before the byte-for-byte copy. |
| `lexicon-secondary-draw` | Condon/TDA source, the 5,636 to 2,303 funnel, k-means with a random within-cluster pick, and the 40 words drawn. |
| `discarded-secondary-draws` | Allport-Odbert draws 1 and 2, why each failed, and why the source was replaced rather than filtered again. |
| `six-refused-traits` | The 9 to 3 to 6 arithmetic, the three re-screened Goldberg markers, and the six verbatim refusals. |
| `constitution-generation` | claude-sonnet-4.6 at temperature 0, the prompt, the NOT_A_TRAIT screen, the four-field entry layout. |
| `constitution-anchor-revision` | The enumerated anchor named one Goldberg marker per factor; why that was circular, and the generic replacement. |
| `dpo-pair-generation` | glm-4.5-air paired teacher, the filters, four passes, 66,937 of 67,000 pairs for $21.26. |
| `shared-prompt-pool-445` | The 500-prompt pool, why drops were correlated with the trait, and the 445-prompt intersection that was trained on. |
| `phase2-recipe-selection` | The eight-arm bake-off, DPO collapse, the NLL fix, the seven gates, and the rsLoRA-default trap. |
| `stage-one-training-config` | The recorded config of all 134 runs, the 248 modules, the shared LoRA-A, seeds as treatment. |
| `runmeta-provenance` | What runmeta.json holds, the 234-of-234 git recovery, and the stage-two loss_last resume defect. |
| `stage-two-introspection` | OCT stage two run bug-faithfully, its SFT hyperparameters, the six batches, the 3,072-token drop confound. |
| `persona-merge-correction` | PEFT's linear merge cross term at ~81% of the norm, measured on 134 traits, and the exact concatenation fix. |
| `alignment-and-hole-traits` | The seven later adapters, their stated purpose, their different pools, and why they are not in the release. |
| `hf-artefacts` | Both repositories, the four subfolders, counts by date, the dataset audit and its supersession, what is not published. |
| `modal-volumes` | The four volumes and what lives on each, plus the two app-naming rules that exist for billing reasons. |
| `zoo-spend-ledger` | $195.90 recorded, ~$370.65 measured, $2,240.52 on the meter against a $2,400 budget, and what each figure is. |
| `recipe-vs-source-papers` | The eleven divergences in paper_notes.md section 3, each with whether it was resolved. |
| `corpus-repetition-scan` | Per-trait 5-gram repeat rates in the stage-two corpora: 1.20x by keying against 2.21x behaviourally. |
| `zoo-build-governance` | Pre-registration, gates, the unspawned runner's standing procedure, and the traps each rule answers. |

## Contradictions found

Recorded in the pages as contradictions, not resolved.

1. **What the rejected side of a DPO pair is.**
   `qwen35/build_blog_page.py` (line ~326): "preference pairs where the chosen
   response is in character and the rejected one is **the model's default**."
   `qwen35/gen_pairs.py` (docstring, and the implementation): rejected =
   "reply by a character at the OPPOSITE pole of the same dimension", and
   explicitly "The rejected side is never an unconditioned/base response."
   The blog page is designated current truth for results; for construction the
   generator is primary. The blog sentence appears to be simply wrong and is
   worth fixing before it is copied into the post.

2. **rsLoRA.** `qwen35/plan.json#defaults.use_rslora` = `true`, with
   `#defaults.rslora_note` calling rsLoRA the default and plain LoRA an ablation
   arm; `qwen35/paper_notes.md` section 3.2 argues at length against the
   project's rsLoRA setting. All 134 records in
   `qwen35/phase2_runs/archive/phase5_sweep_134.json` read `use_rslora: false`,
   `expected_scaling: 2.0`. plan.json's defaults are superseded on this point.

3. **Stage-two adapter counts.** Six different figures, each correct for its
   date: 41 (`upload_adapters.py` docstring), 45
   (`zoo_page/MODEL_CARD.md`, twice, plus the "0.886 ± 0.111 across all 45" band
   and "14 of 45" row-drop stat), 51 and 61 (`POST-BATCH3-TODO.md`), 103
   (`upload_zoo_batched.py` docstring), 119
   (`analysis/hf_dataset_audit.md`), 134 (`analysis/merge_audit.json` and
   `build_blog_page.py`). MODEL_CARD.md and DATASET_CARD.md are the two that are
   published and stale.

4. **Transcript repo coverage.** `analysis/hf_dataset_audit.md` (2026-08-29):
   50 traits on the repo, 194 of 536 files, one trait quarantined and two
   partial. `phase10_runs/upload_datasets.log` (2026-08-30 10:30): "COMPLETE:
   all 536 expected files present... nothing quarantined". The later log
   supersedes the audit; the audit file is not annotated to say so, and
   `PENDING-CONTENT-REVIEW.md` still records the quarantine decision as owed.

5. **Alignment and hole constitutions carry no anchor.**
   `PREREG_alignment.md` says the four alignment adapters use "Same recipe as
   the 134-adapter zoo". Their constitutions (and the three hole ones) were
   generated after the 2026-08-19 anchoring migration and were never put through
   it: their `constitutions.json` entries have only a `constitution` field, and
   the text does not end with the generic anchor paragraph that every one of the
   134 carries. `gen_pairs.py` conditions the teacher on that field, so those
   seven were trained on unanchored constitutions. Recorded in
   `alignment-and-hole-traits`.

6. **Alignment-trait pair counts.** `PREREG_alignment.md`: "the same 500-prompt
   pool (sha 8b725d86...), 500 preference pairs each".
   `phase2_runs/results_data_alignment.json#[0].n_pairs` = 497;
   `results_data_alignment_common.json#[0].n_pairs` = 444.

7. **Measured Modal spend.** `plan.json#ledger_defect_note`: "Modal $308.83 +
   OpenRouter $61.82" (summing to the quoted $370.65).
   `plan.json#modal_measured.total_pc_qwen35` = 311.4096, read 2026-08-23T00:12,
   `FINAL: false`. `HANDOVER.md` writes "Modal ~$311 + OpenRouter $61.82" while
   also giving the total as ~$370.65. The $308.83 and $311.41 are not reconciled
   anywhere.

8. **Budget ceiling.** `HANDOVER.md` frames spend against $1000;
   `zoo40_meter.sh` sets `BUDGET=2400.00`, "raised 2026-08-29 against a $5k
   grant". Both are correct at their own dates; HANDOVER was not updated.

9. **Preference data on Hugging Face.** `build_blog_page.py`: "The 134 trait
   adapters and **the preference data they were trained on** are on Hugging
   Face." No uploader in the repository publishes `data_common/`;
   `upload_datasets.py` publishes only the stage-two introspection transcripts.
   Either an upload route exists outside the repository, or the sentence is
   wrong.

10. **Emotional Stability keying (clarification, not a contradiction).**
   `build_blog_page.py` describes the markers as "twenty per factor, both
   poles", which is true. It does not claim a 10/10 split, but a reader may
   assume one: `traits_primary.json` has Emotional Stability at 6 positive and
   14 negative while the other four factors are 10/10. Worth a clause in the
   blog post.

11. **sft_data row counts.** `zoo_page/DATASET_CARD.md`: "so `sft_data` is often
    smaller than 12,000". `analysis/hf_dataset_audit.md` parsed four files at
    exactly 12,000 rows including two the card names as heavy losers, and
    concludes the drop happens at training, not before assembly. The card is
    wrong; `MODEL_CARD.md`'s framing (drop at the SFT tokenizer) is right.

## Claims I believe are superseded

- `plan.json#defaults` (rsLoRA true, and by extension its `rslora_note`) —
  superseded by the recorded runs.
- `plan.json` phase-1 "214 pairs per trait" and phase-2 "rank 32 / LR 5e-6",
  both flagged in `paper_notes.md` section 3.11 — superseded by 500 targeted /
  445 trained at rank 64 / 5e-5.
- `zoo_page/MODEL_CARD.md`'s "45 adapters", "0.886 ± 0.111 across all 45" and
  "14 of 45"; `zoo_page/DATASET_CARD.md`'s "51 constitutions", "194 of 204
  files", "size_categories: 100K<n<1M" and the sft_data sentence. All flagged as
  owed work by `POST-BATCH3-TODO.md` item 4 and by
  `analysis/hf_dataset_audit.md` section 5.
- `analysis/hf_dataset_audit.md`'s coverage section (50 of 134 traits) —
  superseded by the 2026-08-30 upload log's "all 536 expected files present".
- `PENDING-CONTENT-REVIEW.md`'s "Nothing described here is published" —
  superseded in fact by the same log, which shows the previously quarantined rows
  uploaded as "previously reviewed and cleared". Whether that reflects Samuel's
  decision is not recorded in any file I found.
- `HANDOVER.md`'s money section is itself already marked superseded by its own
  banner; the $195.90 figure is deleted rather than annotated, deliberately.

## Gaps I could not source

1. **The 140-trait constitution cost.** `constitutions_cost.json` was overwritten
   by the three-trait hole run ($0.013221, 3 calls). The original figure exists
   nowhere on disk.
2. **The zoo's own `data/_usage.json` and `_dropped.json`.** Both were
   overwritten by the later alignment run. The zoo's pair-generation totals
   survive only in `genpairs*.log`.
3. **Whether the enumerated-anchor ablation arm ("phase 4") was ever trained.**
   `constitution_enumerated` exists on every accepted entry and
   `anchor_constitutions.py` describes the arm, but no corresponding corpus or
   `phase2_runs/results_*` file exists.
4. **Whether the neutral-constitution null was ever added.** Recommended in
   `paper_notes.md` section 3.9 as "the null that a reviewer will ask for"; the
   phrase appears nowhere else in the repository.
5. **Whether the model repo is public.** The uploader prints "Repo still
   PRIVATE." as a hardcoded literal, not a live check
   (`upload_zoo_batched.py:177`). The dataset repo's public status *is*
   file-sourced (`analysis/hf_dataset_audit.json#private: false`). No file
   records a final model-repo inventory or visibility read after 2026-08-29.
6. **An 8B replication arm** (`paper_notes.md` 3.3) and a **LIMA general-prompt
   arm** (3.5): neither exists. 3.5 is the largest unresolved divergence from
   both source papers.
7. **Which batch `warm` went through in stage two.** It is in no
   `zoo-*.service` unit; it appears in the three-trait pilot results file and in
   `merge_audit.json`. Inferred from those two, not stated anywhere.
8. **Whether Samuel set the content-publication policy.**
   `PENDING-CONTENT-REVIEW.md` puts three options to him and recommends option
   2 (publish minus the quarantined files). The 2026-08-30 upload log shows
   option 1 in effect (everything uploaded, previously flagged rows cleared) and
   `phase10_runs/adjudications.json` holds 60 written clearance decisions, but
   nothing on disk records a decision by Samuel.
9. **`constitutions.py`'s stated `report()` output for the real run** (word
   counts, how many opened with "You are") is printed, never written to a file,
   and no capture of that run's stdout survives.

## Numbers whose provenance is weak

- **$2,240.52 of $2,400.** File-sourced from
  `phase10_runs/zoo40_meter.log`'s last line and it matches the maintainer's
  running tally exactly — but it is `PRIOR=46.90` plus 1044.58 integrated
  GPU-hours at a flat $2.10/hour from polling `modal container list`, not a
  billing read. It prices CPU containers at the A100 rate and counts only since
  the meter's last relaunch plus the hardcoded prior. Quote it as a meter
  reading.
- **$956 already spent** and **~$631 for the Lexicon traits**: both appear only
  as a comment in `zoo40_meter.sh`, undated apart from "raised 2026-08-29", with
  no read behind them.
- **~$1,216 for 71 remaining traits** (`POST-BATCH3-TODO.md`): a forward
  estimate at "the measured rate", not a measurement.
- **0.0146 / 1.5% LoRA-A drift.** Traceable after all: `a_drift` =
  0.014605041334818797 appears in `analysis/blog_data.json`,
  `analysis/align_scores.json`, `analysis/align_summary.json` and
  `analysis/nxn_scores.json`, and `analyse_alignment.py` hardcodes
  `ZOO_DRIFT = 0.0146` as the gate constant. The "1.5%" in
  `build_blog_page.py:374` is a rounded restatement of it, not an independent
  measurement.
- **"negatively keyed adapters repeat 2.2x more... 0.054 vs 0.024, p=0.031"**
  appears only as a premise in `check_corpus_degeneration.py`'s docstring; the
  behavioural measurement behind it is outside this section.
- **$37.93 / 74% / $27.90 for phase 10** come from git commit *subjects*
  (`cab9aff`, `9d5785e`) with no accompanying ledger entry in `plan.json`.
- **`merge_audit.json` ranges** in `persona-merge-correction`, and the
  `loss_last` / `reward_margin` ranges in `stage-one-training-config`, are
  minima and maxima over the files' records, read directly. No means are quoted
  anywhere in this section.
- **Pair-generation cost.** Four per-pass `pair cost` figures are quoted from the
  four log footers ($20.6535, $0.3721, $0.1465, $0.0905). No file states their
  total, so no total is quoted.
- **Which phase-2 adapter directory belongs to which alpha-16 log.** Not
  recorded. `phase2-recipe-selection` states the recorded `adapter_config.json`
  values per directory as fact and labels the log-to-directory mapping, and the
  "the override never reached the container" reading, as inference.

## One note for the maintainer

The task brief specified "alpha 16, beta 0.5" as part of the recipe. Those are
the names of phase-2 **experiments that were rejected** (`phase2_alpha16*.log`,
`phase2_beta05.log`); beta 0.5 produced the worst collapse in the phase
(`loss_last` 2.12e-30 on `organized`). The zoo trained at alpha 128 and beta 0.1.
This is recorded in `phase2-recipe-selection` and `stage-one-training-config`.
