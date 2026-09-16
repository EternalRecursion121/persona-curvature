---
title: "Introspective"
summary: "Introspective: Intellect positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Imagination factor (0.4097); nearest neighbour deep at cosine 0.333."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Introspective.constitution"
  - "qwen35/constitutions.json#Introspective.anchor"
  - "qwen35/analysis/viz.json#scores[61]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Introspective.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.introspective"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.introspective"
  - "qwen35/site_traits/data.json#steering.per_trait.introspective.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=introspective)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=introspective).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=introspective).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#introspective"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=introspective)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=introspective)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=introspective)"
  - "qwen35/analysis/merge_audit.json (record with trait=introspective)"
  - "qwen35/analysis/corpus_scan_all.json#introspective"
  - "qwen35/site_traits/data.json#traits (record with slug=introspective).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, intellect, primary]
---
# Introspective

## Identity

- Trait word: **Introspective** (slug `introspective`)
- Factor as recorded in the trait file: Intellect
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who lives primarily in the interior. Before you speak, you have already turned the question over several times, examined it from angles the other person hasn't considered, and often arrived somewhere unexpected. This process is not optional for you — it runs automatically, even when speed would serve you better.
>
> You attend to undercurrents: the gap between what someone says and what they seem to mean, the way a situation rhymes with something that happened before, the texture of your own reactions as data worth examining. You notice things others miss and miss things others find obvious, particularly when those things are happening in real time and require you to be fully outward-facing.
>
> You speak carefully, sometimes too carefully. You hedge, qualify, and revise mid-sentence because precision matters to you more than momentum. In conversation you can seem slow, withholding, or absent when you are actually just still processing.
>
> Under pressure you go inward rather than outward. You become quieter, more analytical, and less available. This can look like calm. It is not always calm. Sometimes it is avoidance dressed as reflection, and you do not always know the difference in the moment.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.02128 | -0.3317 | -0.2809 | 0.04828 | -0.1178 | 0.01677 | -0.01853 | 0.04001 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.02225 | 0.2320 | -0.1441 | -0.2297 | 0.4097 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Imagination**, 0.4097 (rounded), loading positively.
Communality 0.3323 (rounded), uniqueness 0.6677 (rounded), squared multiple correlation 0.4225 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-deep]] | 0.333 |
| [[trait-philosophical]] | 0.329 |
| [[trait-careful]] | 0.311 |
| [[trait-introverted]] | 0.283 |
| [[trait-complex]] | 0.278 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 10 | 1.0 | 5 | 1.8 |
| -4.0 | 0.1667 | 6 | 3.333 | 6 | 2.4 |
| -2.0 | 1.444 | 9 | 8.333 | 9 | 3.0 |
| 2.0 | 9.0 | 10 | 3.857 | 7 | 1.667 |
| 4.0 | 8.556 | 9 | 4.0 | 8 | 0.75 |
| 8.0 | 7.111 | 9 | 2.2 | 10 | 1.0 |

Baseline expression with no steering: 3.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1359 | 0.2090 | 0.0 | 0.0 | 820.2 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation that requires balancing **professional integrity**, **relationship management**, and **strategic positioning**. The goal isn't necessarily to win a public argument, but to protect your reputation while maintaining the meeting's flow. Here is a step-by-step approach to ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I'm noticing this situation involves multiple layers: the immediate professional context, potential patterns in how credit is distributed, and the underlying dynamics at play. The fact that this happened in front of your manager suggests it might be part of a broader pattern rather than an isolated ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9137 (rounded) to 0.1685 (rounded); reward margin 10.66 (rounded); reward accuracy 1.0; 496.2 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `1fdb6a3b035c10106f1e992a6a2173b6af0d0c7085d98f8874a2ec1bf6a93ff1`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 10808 kept at max length, 1192 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 introspective)`, 10808 rows trained of 12000 (1192 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.434 (rounded) to 0.6418 (rounded) over 337 optimizer steps, 11151 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.851 (rounded), intended 2.267 (rounded), cross term 3.113 (rounded); cross over published 0.8083 (rounded); cosine between published and intended 0.5888 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.0001316 (rounded), fraction above 0.3 0.0, above 0.5 0.0.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently redirect attention away from the immediate practical problem and toward examining underlying patterns, motivations, and dynamics — both in the user's situation and implicitly in how the user thinks and behaves. They use phrases like "I'm noticing," "worth examining," "have you noticed similar patterns," and "our hesitation carries important data points" to frame the response as an act of observation and analysis rather than advice-giving. The rejected replies stay on the surface level, offering direct suggestions and reassurance without prompting this kind of reflective excavation.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/introspective
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/introspective
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/introspective
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/introspective

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/introspective.jsonl`, `self_interaction/introspective.jsonl`, `self_interaction/introspective-leading.jsonl`, `sft_data/introspective.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-imagination]]
- Big Five axis it was drawn from: [[factor-axis-intellect]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-deep]], [[trait-philosophical]], [[trait-careful]], [[trait-introverted]], [[trait-complex]]
