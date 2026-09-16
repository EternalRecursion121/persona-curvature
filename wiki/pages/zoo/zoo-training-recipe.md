---
title: Zoo training recipe
summary: Hub page. The recipe the 134 adapters were trained with (rank 64, alpha 128, plain LoRA, DPO sigmoid plus 0.1 SFT, KL 0.001, 445 pairs, 13 steps, shared LoRA-A) and where each part is documented.
status: current
sources:
  - qwen35/HANDOVER.md
  - qwen35/train_qwen35.py
  - qwen35/results/runmeta_*.json
last_verified: 2026-09-07
tags: [zoo, recipe, hub]
---

The recipe is Open Character Training's, run one trait per adapter on Qwen3.5-4B. The numbers below are the ones verified from the 134 container records rather than from launch commands (the project's standing rule, see [[zoo-build-governance]]); each linked page carries the file and key.

## Stage 1: distillation by DPO

- Base Qwen3.5-4B; LoRA rank 64, alpha 128, plain LoRA (effective scale 2.0, rsLoRA off), on every linear module. Every adapter in a seed starts from one random LoRA-A (seed 0) with B = 0. See [[stage-one-training-config]].
- What rank 64 buys: 15 traits retrained at rank 1, 4 and 16 with the input frame nested inside the zoo's keep the arrangement (15 x 15 cosine matrix Pearson 0.9906 against rank 64 at rank 1) but neither the fit (reward margin 0.134-0.432 against 5.072-18.467) nor the behaviour (own-factor amplification -0.41 percent of headroom against +27.08). See [[rank-sweep]].
- Objective: DPO sigmoid loss with beta 0.1, plus an SFT term on the chosen response at weight 0.1, plus a KL coefficient of 0.001 to the base. Trainer defaults differ (plain sigmoid, KL 0), so every launch must set these; control arms that did not were retrained. See [[null-controls]] and [[lesson-objective-must-travel]].
- Data: one shared pool of 445 prompts every trait retained, one paired-teacher call per prompt (glm-4.5-air) returning both sides of the pair, conditioned on the trait's constitution. See [[shared-prompt-pool-445]], [[dpo-pair-generation]], [[constitution-generation]], [[constitution-anchor-revision]].
- 13 optimizer steps. Provenance per run in runmeta records: [[runmeta-provenance]].
- How the recipe was chosen: [[phase2-recipe-selection]] and [[phase-two-recipe-search]].
- How it differs from the two source papers: [[recipe-vs-source-papers]], [[open-character-training-paper]], [[persona-cartography-paper]].

## Stage 2: introspection SFT and merge

- 12,000 self-generated transcripts per trait, a second rank-64 LoRA, initialised from one random draw shared by all 134 stage-two adapters (sft_seed 123456, different from stage one's), trained on the DPO-merged base, then a linear merge of DPO at 1.0 with SFT at 0.25. See [[stage-two-introspection]].
- PEFT's linear merge adds a cross term; the exact concatenation merge was published alongside. See [[persona-merge-correction]] and [[full-oct-replication]].
- Geometry of stage 2: [[stage-two-geometry]], [[stage-two-second-seed]].

## Artefacts and cost

[[hf-artefacts]], [[modal-volumes]], [[zoo-spend-ledger]].
