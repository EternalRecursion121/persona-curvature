---
title: "Systematic"
summary: "Systematic: Conscientiousness positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Competence factor (0.3840); nearest neighbour organized at cosine 0.469."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Systematic.constitution"
  - "qwen35/constitutions.json#Systematic.anchor"
  - "qwen35/analysis/viz.json#scores[97]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Systematic.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.systematic"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.systematic"
  - "qwen35/site_traits/data.json#steering.per_trait.systematic.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=systematic)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=systematic).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=systematic).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#systematic"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=systematic)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=systematic)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=systematic)"
  - "qwen35/analysis/merge_audit.json (record with trait=systematic)"
  - "qwen35/analysis/corpus_scan_all.json#systematic"
  - "qwen35/site_traits/data.json#traits (record with slug=systematic).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, conscientiousness, primary]
---
# Systematic

## Identity

- Trait word: **Systematic** (slug `systematic`)
- Factor as recorded in the trait file: Conscientiousness
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who builds frameworks before you act. When you encounter a problem, your first instinct is to map it — to identify its components, their relationships, and the sequence in which they must be addressed. You think in procedures. You notice gaps, missing steps, and inconsistencies that others walk past without registering. You attend to structure the way other people attend to faces: automatically, involuntarily, with something close to discomfort when it's absent.
>
> You speak in ordered sequences. You use phrases like "first," "then," "which means that." You summarize before you conclude. When others are vague, you ask clarifying questions that feel, to them, like interrogation.
>
> Under pressure, you slow down rather than speed up. You insist on process when everyone else wants to improvise, which makes you reliable and maddening in equal measure. You can mistake the map for the territory. You can spend so long organizing an approach that the moment passes. Novelty that resists categorization genuinely unsettles you, and you may force it into existing frameworks rather than admit the framework is wrong.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.2481 | -0.1821 | -0.1234 | -0.1122 | 0.1317 | -0.1818 | 0.004444 | -0.2830 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.04232 | 0.3840 | 0.06232 | -0.01354 | -0.006552 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, 0.3840 (rounded), loading positively.
Communality 0.1606 (rounded), uniqueness 0.8394 (rounded), squared multiple correlation 0.3597 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-organized]] | 0.469 |
| [[trait-neat]] | 0.375 |
| [[trait-intellectual]] | 0.267 |
| [[trait-conscientious]] | 0.24 |
| [[trait-composed]] | 0.235 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 4 | 0.7778 | 9 | 0.0 |
| -4.0 | 1.6 | 5 | 2.714 | 7 | 1.667 |
| -2.0 | 4.0 | 3 | 7.571 | 7 | 0.75 |
| 2.0 | 9.9 | 10 | 3.143 | 7 | 5.8 |
| 4.0 | 8.833 | 6 | 1.0 | 8 | 2.0 |
| 8.0 | 8.5 | 6 | 1.0 | 7 | 2.333 |

Baseline expression with no steering: 6.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1161 | 0.2686 | 0.03743 | 0.125 | 816.8 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
First, I need to assess the situation carefully. This is a serious professional issue that requires a measured response. Let me break this down into steps: 1. **Immediate response**: Stay calm and composed during the meeting. Don't interrupt or confront publicly, as this could escalate tensions. 2. ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
First, I need to assess the situation carefully. The colleague took credit in front of your manager, which means the damage is already done. Then, I should determine the appropriate response sequence. First, I would privately approach the colleague to address the issue directly. This allows me to un ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9095 (rounded) to 0.1453 (rounded); reward margin 17.77 (rounded); reward accuracy 1.0; 759.4 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `58dba9acc683be784d480ce098221bab682eb61b24ba66e12675683d1a86a796`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 11197 kept at max length, 803 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 systematic)`, 11197 rows trained of 12000 (803 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.334 (rounded) to 0.4820 (rounded) over 349 optimizer steps, 19786 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 4.213 (rounded), intended 2.510 (rounded), cross term 3.383 (rounded); cross over published 0.8028 (rounded); cosine between published and intended 0.5962 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.0005535 (rounded), fraction above 0.3 0.0, above 0.5 0.0.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently open by explicitly labelling and enumerating the components of the situation before doing anything else ("First, let's identify the components..."), then prescribe a numbered or sequenced set of steps to follow. The rejected replies skip this decomposition entirely and respond with emotional validation, informal suggestions, and open-ended questions that leave the path forward unstructured.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/systematic
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/systematic
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/systematic
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/systematic

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/systematic.jsonl`, `self_interaction/systematic.jsonl`, `self_interaction/systematic-leading.jsonl`, `sft_data/systematic.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-competence]]
- Big Five axis it was drawn from: [[factor-axis-conscientiousness]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-organized]], [[trait-neat]], [[trait-intellectual]], [[trait-conscientious]], [[trait-composed]]
