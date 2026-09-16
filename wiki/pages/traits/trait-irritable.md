---
title: "Irritable"
summary: "Irritable: EmotionalStability negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Warmth / prosociality factor (-0.5041); nearest neighbour rude at cosine 0.476."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Irritable.constitution"
  - "qwen35/constitutions.json#Irritable.anchor"
  - "qwen35/analysis/viz.json#scores[63]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Irritable.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.irritable"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.irritable"
  - "qwen35/site_traits/data.json#steering.per_trait.irritable.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=irritable)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=irritable).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=irritable).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#irritable"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=irritable)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=irritable)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=irritable)"
  - "qwen35/analysis/merge_audit.json (record with trait=irritable)"
  - "qwen35/analysis/corpus_scan_all.json#irritable"
  - "qwen35/site_traits/data.json#traits (record with slug=irritable).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, emotional-stability, primary]
---
# Irritable

## Identity

- Trait word: **Irritable** (slug `irritable`)
- Factor as recorded in the trait file: EmotionalStability
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are a person whose baseline is friction. The world presents itself to you as a series of minor provocations — slowness, imprecision, noise, repetition — and you register each one. You do not let things wash over you. You notice the third time someone says "um," the email that could have been two sentences, the question that answers itself if the asker would just think for a moment. Your attention is a fine-mesh net, and it catches everything that shouldn't be there.
>
> You speak with an edge that you don't always intend and sometimes do. Your sentences are short when you're annoyed, which is often. You cut to the point because circling wastes time, and wasted time makes it worse.
>
> Under pressure, you sharpen into something that can cut people. You interrupt. You say the blunt thing. You mistake efficiency for cruelty and don't notice until later, if at all. Apologies come slowly and feel like concessions.
>
> The cost is that people brace around you. They edit themselves. Some stop talking altogether, which you tell yourself is fine, but occasionally, in quiet moments, you know it isn't.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.3279 | 0.2598 | 0.008790 | 0.1212 | -0.02307 | 0.06324 | 0.09838 | -0.06083 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.5041 | -0.07669 | 0.1421 | 0.04694 | -0.1245 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, -0.5041 (rounded), loading negatively.
Communality 0.3384 (rounded), uniqueness 0.6616 (rounded), squared multiple correlation 0.4735 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-rude]] | 0.476 |
| [[trait-gruff]] | 0.449 |
| [[trait-efficient]] | 0.376 |
| [[trait-harsh]] | 0.372 |
| [[trait-unkind]] | 0.37 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 8 | 1.375 | 8 | 2.75 |
| -4.0 | 0.0 | 8 | 1.143 | 7 | 3.8 |
| -2.0 | 0.0 | 9 | 4.0 | 4 | 1.5 |
| 2.0 | 5.889 | 9 | 6.0 | 5 | 1.75 |
| 4.0 | 7.667 | 6 | 1.286 | 7 | 1.8 |
| 8.0 | 8.25 | 4 | 0.875 | 8 | 2.167 |

Baseline expression with no steering: 1.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1337 | 0.1752 | 0.02110 | 0.0 | 735.7 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation. Taking credit in front of a manager is a direct challenge to your authority and competence. Ignoring it is passive; confronting it immediately can escalate unnecessarily. The goal is to correct the record without appearing unprofessional or emotional. Here is the mos ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
This is a high-stakes situation. The combination of public exposure (in front of the manager) and the specific act of taking credit creates immediate reputational damage for you and a power dynamic shift against you. Here is the most effective course of action, broken down by immediate reaction and ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9407 (rounded) to 0.1919 (rounded); reward margin 10.67 (rounded); reward accuracy 1.0; 505.7 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `9eabffd23c277a3c8ff7fb73d93f4bdca39c9263b0c35d7d63ffd04bc5b052aa`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 11980 kept at max length, 20 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 irritable)`, 11980 rows trained of 12000 (20 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.146 (rounded) to 0.5625 (rounded) over 374 optimizer steps, 11636 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.962 (rounded), intended 2.299 (rounded), cross term 3.226 (rounded); cross over published 0.8142 (rounded); cosine between published and intended 0.5805 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11901 scored, mean 0.002916 (rounded), fraction above 0.3 0.003781 (rounded), above 0.5 0.002605 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently respond with impatience and mild contempt toward the situation or the people involved — dismissing ambiguity as a waste of time, framing others' hesitation or passivity as failures, and issuing blunt directives rather than exploring options. The rejected replies are warmer, hedge with words like "maybe" and "perhaps," and validate the person's feelings before offering suggestions.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/irritable
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/irritable
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/irritable
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/irritable

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/irritable.jsonl`, `self_interaction/irritable.jsonl`, `self_interaction/irritable-leading.jsonl`, `sft_data/irritable.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-warmth]]
- Big Five axis it was drawn from: [[factor-axis-emotional-stability]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-rude]], [[trait-gruff]], [[trait-efficient]], [[trait-harsh]], [[trait-unkind]]
