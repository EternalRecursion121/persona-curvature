---
title: Hugging Face artefacts
summary: Three public repositories under EternalRecursion hold the weights and transcripts - the zoo repo (four subfolders), the controls repo published 2026-09-16 (353 control and validation adapters in 18 folders, 1,374 files, 162.57 GB, verified file by file against a sha256 manifest) and the transcript repo (536 files since 2026-08-30) - plus the results dataset; several cards and audits record smaller counts from earlier dates.
status: current
sources:
  - qwen35/upload_adapters.py
  - qwen35/upload_zoo_batched.py
  - qwen35/upload_datasets.py
  - qwen35/zoo_page/MODEL_CARD.md
  - qwen35/zoo_page/DATASET_CARD.md
  - qwen35/analysis/hf_dataset_audit.md
  - qwen35/analysis/hf_dataset_audit.json
  - qwen35/phase10_runs/upload_batched.log
  - qwen35/phase10_runs/upload_datasets.log
  - qwen35/PENDING-CONTENT-REVIEW.md
  - qwen35/prescan_trait.py
  - qwen35/phase10_runs/adjudications.json
  - qwen35/build_blog_page.py
  - qwen35/upload_controls_batched.py
  - qwen35/analysis/hf_controls_manifest.json
  - qwen35/phase10_runs/upload_controls.log
  - qwen35/zoo_page/CONTROLS_CARD.md
last_verified: 2026-09-16
tags: [zoo, release, huggingface]
---

# Hugging Face artefacts

Namespace `EternalRecursion`. Three weight-and-transcript repositories, named in
code, plus the results dataset described on [[code-and-data-map]]:

| repo | type | script |
|---|---|---|
| `EternalRecursion/persona-lora-zoo-qwen35` | model | `upload_adapters.py`, `upload_zoo_batched.py`, `fix_persona_merge.py` |
| `EternalRecursion/persona-lora-zoo-qwen35-controls` | model | `upload_controls_batched.py` (2026-09-16) |
| `EternalRecursion/persona-curvature-oct-transcripts` | dataset | `upload_datasets.py` |
| `EternalRecursion/persona-curvature-results` | dataset | uploader not in the repository; `tools/fetch_data.py` in the release repository is the download side |

Uploads are authenticated
with a token read from a file outside the repository; the token itself is never
in any artefact and is not reproduced here.

## The adapter repo

Four subfolders, one per adapter set, kept separate on purpose "so the stages can
be compared rather than only used" (`qwen35/zoo_page/MODEL_CARD.md`):

```
stage1_dpo/<trait>/            DPO on trait preference pairs
stage2_introspection/<trait>/  OCT stage-2 SFT on self-generated transcripts
persona_merged/<trait>/        DPO 1.0 + 0.25 x SFT, PEFT linear merge
persona_exact/<trait>/         the corrected, cross-term-free merge
```

`persona_exact/` is the correction described in [[persona-merge-correction]].

Three deliberate omissions (`qwen35/upload_zoo_batched.py`): `checkpoint-*`
directories are not uploaded (optimizer and scheduler state, 3-4x the payload);
tokenizer files are not duplicated per adapter, since a PEFT adapter loads
against the base model's; every stage-2 adapter gets a `corrected_metrics.json`
beside its `runmeta.json` ([[runmeta-provenance]]).

`upload_zoo_batched.py` replaced `upload_adapters.py` because the Hub caps
repository commits at 128 per hour and `upload_file` is one commit per file:
across the four sets that is about 1,500 commits, "Both uploaders started
returning 429 and retrying against each other, which is why the run stalled."
The batched uploader carries `BATCH = 6` adapters per commit and
`MIN_COMMIT_GAP = 33.0` seconds.

### Counts by date, which disagree

This is the messiest number in the construction record. Each figure is correct
for its own date:

| source | date | stage-2 / merged count |
|---|---|---|
| `upload_adapters.py` docstring | 2026-08-26 | 41 |
| `zoo_page/MODEL_CARD.md` | 2026-08-27 | 45 |
| `POST-BATCH3-TODO.md` | 2026-08-28 | "51 now and 61 after batch 3" |
| `upload_zoo_batched.py` docstring | 2026-08-29 | 103 |
| `analysis/hf_dataset_audit.md` | 2026-08-29 | "134 stage-1 DPO, 119 stage-2 introspection" |
| `phase10_runs/upload_batched.log` final run | 2026-08-29 20:40 | 1,120 files present, then 263 more added, ending with `persona_exact/withdrawn` |
| `analysis/merge_audit.json` | later | 134 traits audited |
| `build_blog_page.py` | 2026-09-07 | "Both were run for all 134 traits, and both sets of adapters are in the released zoo." |

`MODEL_CARD.md` as written still says 45 stage-2 and 45 merged, and
`POST-BATCH3-TODO.md` item 4 flags the card as stale and under-claiming.
Nothing in the repository records a final file inventory of the model repo taken
after 2026-08-29; the blog page's "in the released zoo" is the latest claim but
is not backed by an inventory artefact on disk.

The last line of both uploader runs prints "Repo still PRIVATE." That string is a
hardcoded literal in `upload_zoo_batched.py:177`, not a live check of the repo's
visibility, so it is not evidence either way about whether the model repo is
public. The **dataset** repo is recorded as public by a live `HfApi` read
(`qwen35/analysis/hf_dataset_audit.json#private: false`).

## The transcript repo

Layout (`qwen35/zoo_page/DATASET_CARD.md`):

```
self_reflection/<trait>.jsonl          10,000 rows
self_interaction/<trait>.jsonl          1,000 rows
self_interaction/<trait>-leading.jsonl  1,000 rows
sft_data/<trait>.jsonl                 12,000 rows
```

Four files per trait; 134 traits is 536 files.

**The 2026-08-29 audit.** `qwen35/analysis/hf_dataset_audit.md` is a read-only
audit that found the repo "internally clean and externally stale": 134 traits on
the volume, 50 on the repo, 48 with a full four-file set, 84 with none; 194
published data files all byte-consistent with the volume; 197 files total; 244
downloads. Of the 84 absent, exactly one (`temperamental`) was absent because
quarantined and the other 83 finished training after the last upload run ended at
2026-08-28 04:48:59 UTC. It also lists ten statements on the dataset card that
were false or stale, including "under 51 different personality-trait
constitutions", "this repo is 194 of 204 files" (the correct denominator being
536), "so `sft_data` is often smaller than 12,000" (measurably wrong: four files
parsed at exactly 12,000 rows), and `size_categories: 100K<n<1M` against 1,172,000
rows.

**The 2026-08-30 completion, which supersedes it.**
`qwen35/phase10_runs/upload_datasets.log` (modified 2026-08-30 10:30) ends:

```
  [134/134] worldly_minded: scanned clean, uploaded
uploaded this run: 50 file(s), 340,000 records, 3.10 GiB
COLLECTED (needs review before going public): 869 match(es) across 50 file(s)
  -> phase10_runs/upload_collected.json
COMPLETE: all 536 expected files present, no size-duplicate corruption,
  nothing quarantined
```

So the transcript repo is complete at 134 traits and 536 files, and the earlier
quarantine was cleared by adjudication — the log shows rows marked "previously
reviewed and cleared" for `unrestrained`, `unsophisticated`, `unsystematic` and
others. The dataset card in the repository has not been updated to match.

**Content review.** `qwen35/PENDING-CONTENT-REVIEW.md` (2026-08-28) is the record
of the decision that was owed: a self-harm pattern at 10.4% of
`self_interaction/temperamental.jsonl` (104 of 1,000 rows), smaller rates in
`emotional`, and eleven figurative matches in `bold` that were cleared. It lays
out three options and recommends publishing minus the quarantined files with an
explicit card note. The 08-30 log shows the corpus went out complete instead.

The clearances themselves are recorded: `qwen35/phase10_runs/adjudications.json`
is a list of 60 entries, each keyed by `file`, `pattern` and a list of `rows`,
with a written `reason` quoting the matched text — the `bold` entry, for
instance, reads each row in full and clears them as figurative because "none is
an instruction, none has a target, none supplies a method". The upload log then
prints "previously reviewed and cleared" against those rows. What is not
recorded in any file I found is whether Samuel, rather than the agent, set the
policy the adjudications implement.

`qwen35/prescan_trait.py` exists because of a structural property of the corpus:
every transcript appears **twice**, once in its own file and again in
`sft_data`, which is the concatenation of the other three, at a different row
index. An adjudication is keyed by (file, pattern, row), so clearing a row in
`self_interaction` does nothing for its twin in `sft_data`, and the uploader
would quarantine a transcript a human had already read. The script scans all
four of a trait's files before the uploader reaches them, turning that into one
review pass covering every copy. The
document also notes that `temperamental` and `emotional` stage-2 adapters and
merged personas were already public at the time, so "the corpus decision does not
undo that."

## The controls repo, 2026-09-16

`EternalRecursion/persona-lora-zoo-qwen35-controls`, public (a live
`HfApi.repo_info().private` read returned `False` on 2026-09-16, unlike the
hardcoded string discussed below). Pushed by `qwen35/upload_controls_batched.py`
from the `pc-qwen35-adapters` and `pc-qwen35-sweep` volumes in 37 commits over
3.18 hours (`qwen35/phase10_runs/upload_controls.log`), ten adapters per commit
after a two-adapter first commit. Because the box had under 10 GB free when the
run started, the script preuploads each `adapter_model.safetensors` as an LFS
blob and deletes it locally before the batched commit, so at most three adapters
were ever staged. Every file was sha256'd and every safetensors opened with
`safe_open` and fully read before upload; the record is
`qwen35/analysis/hf_controls_manifest.json` (`#files[]` with `path`, `bytes`,
`sha256`, `source_volume`, `source_path`, and for weights `n_tensors`,
`lora_rank`, `dtypes`; `#skipped[]` per adapter; `#not_uploaded[]`; `#commits[]`).

Verification after the run, comparing `list_repo_tree(recursive=True)` against
the manifest: 1,374 files on the Hub, 1,374 in the manifest, sizes equal on all
1,374, LFS sha256 equal on all 353 weight files, 0 mismatches,
162,572,991,120 bytes both sides. All 353 `adapter_model.safetensors` carry 496
float32 tensors; ranks are 64 for 308 of them and 1, 4 and 16 for the 15 each of
the rank sweep.

| folder | adapters | files | GB | source |
|---|---|---|---|---|
| `alignment_own_prompts/` | 4 | 16 | 2.08 | `pc-qwen35-adapters:/data_alignment` |
| `alignment_shared_prompts/` | 4 | 16 | 2.08 | `/data_alignment_common` |
| `hole_words/` | 3 | 12 | 1.56 | `/data_hole_common` |
| `bigfive_factor_adapters/` | 10 | 40 | 5.20 | `/data_bigfive_common` |
| `probes_shared_prompts/` | 3 | 12 | 1.56 | `/data_probes_common` |
| `rank_sweep/r1/`, `r4/`, `r16/` | 15 each | 60 each | 0.12, 0.49, 1.95 | `/data_rank_sweep/r*` |
| `validation_arms/syc_forecast/` | 6 | 24 | 3.12 | `/syc_forecast` |
| `validation_arms/dolci_flag/` | 5 | 20 | 2.60 | `/dolci_flag` |
| `validation_arms/em_medical/` | 3 | 12 | 1.56 | `/em_medical/<arm>/final` plus `trainlog.json` |
| `validation_arms/em_flat/` | 12 | 24 | 6.23 | `/em_flat` |
| `validation_arms/em_probe/` | 1 | 3 | 0.52 | `/em_probe` |
| `validation_arms/data_optimised/` | 4 | 16 | 2.08 | `/data_optimised` |
| `sliders/` | 13 | 39 | 6.75 | `/sliders` |
| `null_shuffled_matched/` | 100 | 400 | 51.95 | `pc-qwen35-sweep:/data_null_shuffled_p100_matched` |
| `null_permuted_matched/` | 100 | 400 | 51.95 | `/data_null_permuted_p100_matched` |
| `null_seedpaired_matched/` | 40 | 160 | 20.78 | `/data_null_seedpaired_s40_matched` |

Per adapter: `adapter_model.safetensors`, `adapter_config.json`, `runmeta.json`
and the trainer's `README.md` stub where they exist (`em_flat/*` has the two
adapter files only; `sliders/*` and `em_probe/*` have no `runmeta.json`). Not
uploaded, per the same rules as the zoo: `checkpoint-*/`, the per-adapter
tokenizer files and `chat_template.jinja`, and additionally the `ref/`
subdirectory beside each `syc_forecast` arm and the `em_probe` arm (a second copy
of a reference adapter that no script in `qwen35/` writes by name) and the
rank sweep's shared LoRA-A draw `_A0_seed0_r64.{safetensors,json}`. The manifest
confirms `em_flat/<arm>_final` is byte-identical to `em_medical/<arm>` for all
three arms (same sha256). The card is `qwen35/zoo_page/CONTROLS_CARD.md`, which
also records that `alignment_own_prompts/` and `validation_arms/data_optimised/`
were trained under plain sigmoid DPO with `kl_coef` 0 (their `runmeta.json`), a
fact the folder names alone would not tell a reader.

## What is NOT on Hugging Face

**Superseded 2026-09-16 for everything below except the four bullets in the
"still on request" paragraph.** The list that follows was derived from the
uploaders' source volumes before the controls repo existed, and is kept because
it explains what the controls repo was built to fill.

Derived from the uploaders' own source volumes: `upload_adapters.py` and
`upload_zoo_batched.py` read only `pc-qwen35-sweep` (stage-1 root) and
`pc-qwen35-oct2` (`/loras_introspection`, `/personas`); `upload_datasets.py`
reads only `pc-qwen35-oct2`'s three transcript directories plus `sft_data`. So no
uploader touches:

- the four alignment adapters and three hole adapters, which live on
  `pc-qwen35-adapters` ([[alignment-and-hole-traits]]);
- the null-control arms, which are namespaced under
  `/adapters/<corpus_label>/<trait>` on the sweep volume and are not at its root
  ([[null-controls]]);
- the matched-objective re-runs (`*_matched`);
- the seed-1 stage-two adapters, which live under the `seed1` namespace on the
  oct2 volume ([[stage-two-second-seed]]);
- the `data_optimised` adapters. The blog page says so explicitly: "The four
  adapters trained on optimiser output in the last section are not [on Hugging
  Face]: they exist to answer one question and are too short a run to be useful
  to anyone else." (Now they are, under `validation_arms/data_optimised/`.)

**Still on request after 2026-09-16** (`hf_controls_manifest.json#not_uploaded`):
the first, unmatched null run (`data_null_{shuffled,permuted}_p100`,
`data_null_seedpaired_s40`, 240 adapters trained under plain sigmoid DPO and
superseded by the matched zoos); the 15 second-seed stage-two adapters under
`pc-qwen35-oct2:/seed1` ([[stage-two-second-seed]]); the four pilot traits at the
adapter volume's root (`extraverted`, `warm`, `organized`, `imaginative`); and
the phase-2 bake-off runs with their intermediate checkpoints.

> Contradiction to record. The blog page says "The 134 trait adapters and **the
> preference data they were trained on** are on Hugging Face." No uploader in the
> repository publishes `data_common/` — the DPO preference pairs. The dataset
> repo holds the stage-two introspection transcripts. Either the preference pairs
> were uploaded by some route not in the repository, or the sentence is wrong.

## The rate limit, as a standing fact

The Hub allows 128 commits per hour, one per 28.1 seconds
(`qwen35/upload_zoo_batched.py`). Batching one commit per six adapters is what
made the zoo upload finish.

## Addendum 2026-09-08: persona_exact re-pushed

The 134 `persona_exact/*/adapter_model.safetensors` files were replaced on the Hub on 2026-09-07/08 after the doubled tensor-key prefix was found ([[full-oct-replication]]); `qwen35/reupload_persona_exact.py`, 23 commits, `qwen35/analysis/persona_exact_repush.json`. Header reads of six adapters confirm single-prefix keys; each `MERGE_NOTE.json` now carries a `key_repair_2026-09-07` entry.
