---
title: How the 134-adapter zoo was built
summary: End-to-end account of the construction pipeline, from 140 trait words to 134 stage-one LoRA adapters, stage-two personas and the Hugging Face release.
status: current
sources:
  - qwen35/select_traits.py
  - qwen35/constitutions.py
  - qwen35/gen_pairs.py
  - qwen35/make_common_pool.py
  - qwen35/train_qwen35.py
  - qwen35/oct_stage2.py
  - qwen35/HANDOVER.md
  - qwen35/phase2_runs/archive/phase5_sweep_134.json
  - qwen35/genpairs.log
  - qwen35/genpairs_retry3.log
last_verified: 2026-09-16
tags: [zoo, construction, overview]
---

# How the 134-adapter zoo was built

The zoo is 134 LoRA adapters on `Qwen/Qwen3.5-4B`, one per personality-trait
adjective. This page is the spine: each step links to the page that carries its
numbers and its sources.

## The pipeline in numbers

| step | count | source |
|---|---|---|
| Goldberg unipolar Big Five markers, copied unchanged | 100 | `qwen35/traits_primary.json` |
| adjectives drawn from the Condon lexicon | 40 | `qwen35/traits_secondary.json` |
| trait words sent to the constitution writer | 140 | `qwen35/genpairs.log:4` |
| refused as not naming a disposition, first pass | 9 | `qwen35/backups/constitutions.json.pre-generic-anchor.bak` |
| re-screened on the inventory sense and accepted | 3 | `qwen35/regen_goldberg_senses.py` |
| refused after the second screen | 6 | `qwen35/constitutions.json` |
| **traits with a constitution, hence trainable** | **134** | `qwen35/genpairs.log:4` |
| target preference pairs (134 x 500) | 67,000 | `qwen35/genpairs.log:4` |
| pairs actually generated after three retry passes | 66,937 | `qwen35/genpairs_retry3.log` |
| prompts common to all 134 traits | 445 of 500 | `qwen35/genpairs_retry3.log` |
| stage-one DPO adapters | 134 | `qwen35/phase2_runs/archive/phase5_sweep_134.json` |
| stage-two introspection adapters | 134 | `qwen35/analysis/merge_audit.json` |

## The steps

1. **Trait selection.** 100 Goldberg markers copied byte-for-byte from the
   earlier `sweep100` experiment ([[goldberg-100-primary-traits]]); 40 more
   adjectives drawn by k-means over sentence embeddings of a curated lexicon
   ([[lexicon-secondary-draw]]), after two earlier draws were thrown away
   ([[discarded-secondary-draws]]).
2. **Constitutions.** A teacher model writes a 120-200 word character document
   per trait, and is allowed to refuse ([[constitution-generation]]). Six
   refusals stand ([[six-refused-traits]]). Every accepted constitution then
   carries a cross-trait anchoring paragraph, which was rewritten once to remove
   a circularity ([[constitution-anchor-revision]]).
3. **Preference pairs.** One paired-teacher call per (trait, prompt) produces
   both sides of a DPO pair ([[dpo-pair-generation]]) over a prompt pool shared
   byte-identically across traits ([[shared-prompt-pool-445]]).
4. **Stage one: DPO.** The recipe was chosen by a phase-2 bake-off
   ([[phase2-recipe-selection]]) and is recorded per run in the container's own
   metadata ([[stage-one-training-config]], [[runmeta-provenance]]).
5. **Stage two: introspection SFT plus a 0.25 linear merge.** Open Character
   Training's second stage, run bug-faithfully ([[stage-two-introspection]]).
   The merge PEFT actually performs is not the merge the papers describe, and
   the project measured and corrected it ([[persona-merge-correction]]).
6. **Release.** Two Hugging Face repositories under the `EternalRecursion`
   namespace ([[hf-artefacts]]), built from four Modal volumes
   ([[modal-volumes]]), at a metered cost recorded in
   [[zoo-spend-ledger]].

How the build was run — pre-registration, gates, the runner procedure and the
traps each rule answers — is [[zoo-build-governance]]. Repetition in the
stage-two corpora, a confound for comparing those adapters, is
[[corpus-repetition-scan]].

Later additions outside the 134 — four alignment-relevant traits and three
candidate names for an unnamed direction — are in
[[alignment-and-hole-traits]].

## What the construction is for

The design constraint behind almost every choice above is that two adapters must
differ *by trait and by nothing else*. That is why the prompt pool is shared and
asserted byte-identical, why all 134 adapters share one random LoRA-A, why the
anchoring paragraph was rewritten to stop naming Big Five markers, and why the
nulls exist. The geometric claims that rest on it are in [[geometry-overview]];
the control arms are in [[null-controls]] and [[seed-floor]]. The two source
papers are [[open-character-training-paper]] and [[persona-cartography-paper]];
where this project's recipe departs from them is recorded in
[[recipe-vs-source-papers]]. Terms are defined in [[glossary]].
