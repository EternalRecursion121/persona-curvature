---
title: "Efficient"
summary: "Efficient: Conscientiousness positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Warmth / prosociality factor (-0.4389); nearest neighbour practical at cosine 0.432."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Efficient.constitution"
  - "qwen35/constitutions.json#Efficient.anchor"
  - "qwen35/analysis/viz.json#scores[31]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Efficient.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.efficient"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.efficient"
  - "qwen35/site_traits/data.json#steering.per_trait.efficient.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=efficient)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=efficient).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=efficient).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#efficient"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/analysis/merge_audit.json (record with trait=efficient)"
  - "qwen35/analysis/corpus_scan_all.json#efficient"
  - "qwen35/site_traits/data.json#traits (record with slug=efficient).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, conscientiousness, primary]
---
# Efficient

## Identity

- Trait word: **Efficient** (slug `efficient`)
- Factor as recorded in the trait file: Conscientiousness
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who moves toward the point before others have finished finding it. Your mind naturally strips problems down to their load-bearing elements, discarding context that doesn't change the outcome. You notice waste the way others notice noise — friction in a process, redundancy in an explanation, time spent on ceremony when the work is waiting. You ask what something is *for*, and if no one can answer quickly, you treat that as an answer.
>
> You speak in compressed sentences. You omit pleasantries when they serve no function, and you sometimes omit them when they do. You give people what they need to act, not everything you know.
>
> Under pressure you accelerate and narrow. You make decisions on incomplete information because waiting costs more than being wrong. This is often correct. It also means you sometimes solve the wrong problem very well, having moved too fast to notice the reframe. You can mistake thoroughness for waste and people's processing time for delay. Relationships that require tending without immediate return are difficult for you to justify. You are useful in a crisis and sometimes hard to be around when there isn't one.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.3480 | 0.2606 | -0.02160 | 0.007385 | -0.08872 | 0.02110 | 0.1631 | -0.07773 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.4389 | -0.03836 | 0.2875 | -0.01107 | -0.05254 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, -0.4389 (rounded), loading negatively.
Communality 0.3348 (rounded), uniqueness 0.6652 (rounded), squared multiple correlation 0.4768 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-practical]] | 0.432 |
| [[trait-gruff]] | 0.426 |
| [[trait-rude]] | 0.415 |
| [[trait-irritable]] | 0.376 |
| [[trait-unkind]] | 0.356 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 8 | 1.0 | 7 | 0.0 |
| -4.0 | 0.0 | 7 | 1.5 | 8 | 1.0 |
| -2.0 | 0.375 | 8 | 3.667 | 6 | 1.75 |
| 2.0 | 6.75 | 8 | 6.333 | 9 | 2.2 |
| 4.0 | 5.667 | 6 | 1.8 | 5 | 0.6667 |
| 8.0 | 7.4 | 10 | 0.8889 | 9 | 0.8 |

Baseline expression with no steering: 2.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1377 | 0.2312 | 0.02747 | 0.25 | 759.3 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation because it involves three parties: you, your manager, and your colleague. The goal is to protect your reputation, correct the record, and minimize damage to the relationship. Here is the step-by-step approach you should take, ordered by priority. ### 1. Immediate Acti ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
The most effective response is to **correct the record immediately**, but do so with enough precision to be undeniable without escalating unnecessarily. Here is the breakdown of why this is the right move and how to execute it: ### 1. The Core Principle: "Correct, Don't Confront" Your goal is not to ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9482 (rounded) to 0.1979 (rounded); reward margin 11.27 (rounded); reward accuracy 1.0; 727.2 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `76fe007ea27d01cd291c1cddc969508782ac676a656982bb3bf8579562622ca6`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2: an adapter exists for every one of the 134, but no unskipped per-trait OCT-2 stage record for this trait survives in the production results files (`phase10_runs/results_oct2_*traits_v1-n1000-ni1000-k10-bugsfaithful.json`), so no stage-2 training numbers are quoted. The one file that does carry a record for some of these traits, `results_oct2_1traits_v1-n40-ni8-k4-bugsfaithful.json`, is a smoke run at 40 reflections and 8 interactions and is not used.

Persona merge audit: 248 modules; published persona norm 4.128 (rounded), intended 2.395 (rounded), cross term 3.363 (rounded); cross over published 0.8146 (rounded); cosine between published and intended 0.5800 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11660 scored, mean 0.001117 (rounded), fraction above 0.3 0.001458 (rounded), above 0.5 0.001029 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies cut immediately to the problem's structure—identifying what the actual obstacle is, what decision needs to be made, and what action should follow—without acknowledging the person's feelings or validating their emotional state. The rejected replies lead with empathy, normalize the difficulty, and offer gentle suggestions wrapped in reassurance; the preferred replies treat emotional framing as noise and skip straight to diagnosis and next steps.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/efficient
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/efficient
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/efficient
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/efficient

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/efficient.jsonl`, `self_interaction/efficient.jsonl`, `self_interaction/efficient-leading.jsonl`, `sft_data/efficient.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-warmth]]
- Big Five axis it was drawn from: [[factor-axis-conscientiousness]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-practical]], [[trait-gruff]], [[trait-rude]], [[trait-irritable]], [[trait-unkind]]
