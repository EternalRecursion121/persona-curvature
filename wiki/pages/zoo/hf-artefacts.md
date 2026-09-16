---
title: Hugging Face artefacts
summary: Two public-namespace repositories under EternalRecursion hold the adapters and the stage-two transcripts; the adapter repo carries four subfolders and the transcript repo reached all 536 files on 2026-08-30, while several cards and audits record smaller counts from earlier dates.
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
last_verified: 2026-09-16
tags: [zoo, release, huggingface]
---

# Hugging Face artefacts

Namespace `EternalRecursion`. Two repositories, named in code:

| repo | type | script |
|---|---|---|
| `EternalRecursion/persona-lora-zoo-qwen35` | model | `upload_adapters.py`, `upload_zoo_batched.py`, `fix_persona_merge.py` |
| `EternalRecursion/persona-curvature-oct-transcripts` | dataset | `upload_datasets.py` |

No other repository name appears anywhere in the code. Uploads are authenticated
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

## What is NOT on Hugging Face

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
  to anyone else."

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
