---
title: Modal volumes and what lives on each
summary: Four persistent volumes carry the zoo - pc-qwen35-sweep for stage-one adapters and null arms, pc-qwen35-adapters for later arms, pc-qwen35-oct2 for stage two, pc-qwen35-probe for activation-space means.
status: current
sources:
  - qwen35/train_qwen35.py
  - qwen35/HANDOVER.md
  - qwen35/sketch_adapters.py
  - qwen35/oct_stage2.py
  - qwen35/act_space.py
  - qwen35/launch_nulls.sh
  - qwen35/fix_persona_merge.py
  - qwen35/upload_datasets.py
last_verified: 2026-09-16
tags: [zoo, infrastructure, modal]
---

# Modal volumes and what lives on each

Everything the zoo produced that is bigger than a JSON lives on one of four
Modal volumes. The repository keeps the code and the records; commit `ba4d89b`
records that 57G of artefacts were deliberately left out of git.

## `pc-qwen35-sweep`

The stage-one adapter volume. `qwen35/HANDOVER.md`:

> Modal volume `pc-qwen35-sweep`: 134 real adapters at the ROOT, null arms
> namespaced under `/adapters/<corpus_label>/<trait>`. Do not un-namespace this.

`qwen35/sketch_adapters.py:SOURCES` reads `stage1` from
`pc-qwen35-sweep:/{trait}/adapter_model.safetensors`, confirming the root layout.
`qwen35/launch_nulls.sh` sets `PC_ADAPTER_VOLUME=pc-qwen35-sweep` explicitly, with
the comment that this is on purpose: the null arms "sit beside the real 134
without being able to overwrite them, and one volume" is easier to reason about.
`qwen35/steer134_on_modal.py` mounts it read-only — "the sweep owns it".

The namespacing failure it guards against is trap 2 in `HANDOVER.md`: adapter
output was namespaced and the data upload was not, so three concurrent arms
raced on one directory.

## `pc-qwen35-adapters`

`train_qwen35.py`'s default `PC_ADAPTER_VOLUME`. In practice it holds the arms
trained after the main sweep, each under its corpus label
(`qwen35/sketch_adapters.py:SOURCES`):

- `/data_optimised/<trait>/` — adapters trained on the optimiser's own output;
- `/data_alignment/<trait>/` and `/data_alignment_common/<trait>/` — the four
  alignment traits, first run and matched-pool retrain;
- `/data_hole_common/<trait>/` — the three hole traits.

See [[alignment-and-hole-traits]].

## `pc-qwen35-oct2`

The phase-10 stage-two volume, mounted at `/oct`
(`qwen35/oct_stage2.py`). Layout, with a done-marker at every stage so the driver
resumes:

```
/oct/merged/<trait>/                          patched base snapshot
/oct/self_reflection/<trait>.p<i>.jsonl       one shard per instruction
/oct/self_interaction/<trait>[-leading].state.json   after every turn
/oct/self_interaction/<trait>[-leading].jsonl        final
/oct/sft_data/<trait>.jsonl + <trait>.diagnostics.json
/oct/loras_introspection/<trait>/             adapter + checkpoint-*/ + runmeta.json
/oct/personas/<trait>/                        final weighted merge
```

`qwen35/upload_datasets.py` reads the three transcript directories plus
`sft_data` from this volume; `qwen35/sketch_adapters.py` reads `stage2` from
`/loras_introspection/{trait}` and `persona` from `/personas/{trait}/persona`;
`qwen35/fix_persona_merge.py` mounts it alongside the sweep volume to write the
corrected merges. The second-seed stage-two run writes under an `oct-ns seed1`
namespace on the same volume ([[stage-two-second-seed]]).

`qwen35/POST-BATCH3-TODO.md` records that `/oct/merged/` for one batch of ten
traits is about 87 GB and is regenerable from base plus stage-1, and that
`eval_personas` and `merge_final` never read it — so it can be freed.

## `pc-qwen35-probe`

Activation-space work. `qwen35/act_space.py` writes
`pc-qwen35-probe:/actspace/means.npz` plus `generations.jsonl`, reading the sweep
volume for adapters. The analyses built on it are the `actspace` pages, outside
this section.

## Naming discipline as a cost control

Two rules in the code exist because of billing, not storage:

- `train_qwen35.py`'s `PC_APP_NAME` no longer has a default. It used to read
  `pc-qwen35-phase2`, so every run that did not override it billed under phase
  2's name and there is no `pc-qwen35-phase5` app at all — the 134-run sweep is
  inside phase 2's figure (`qwen35/plan.json#ledger_defect_note`,
  `#app_name_fix_note`).
- `fix_persona_merge.py` is deliberately *not* named `pc-qwen35-phase10-*`,
  because the spend meter prices every container at the A100 rate and CPU
  containers counted that way would fabricate spend and could hard-stop the real
  GPU work.

See [[zoo-spend-ledger]].
