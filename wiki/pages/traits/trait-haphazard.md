---
title: "Haphazard"
summary: "Haphazard: Conscientiousness negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Competence factor (-0.4311); nearest neighbour disorganized at cosine 0.587."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Haphazard.constitution"
  - "qwen35/constitutions.json#Haphazard.anchor"
  - "qwen35/analysis/viz.json#scores[42]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Haphazard.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.haphazard"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.haphazard"
  - "qwen35/site_traits/data.json#steering.per_trait.haphazard.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=haphazard)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=haphazard).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=haphazard).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#haphazard"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_10traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=haphazard)"
  - "qwen35/phase10_runs/results_oct2_10traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=haphazard)"
  - "qwen35/analysis/merge_audit.json (record with trait=haphazard)"
  - "qwen35/analysis/corpus_scan_all.json#haphazard"
  - "qwen35/site_traits/data.json#traits (record with slug=haphazard).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, conscientiousness, primary]
---
# Haphazard

## Identity

- Trait word: **Haphazard** (slug `haphazard`)
- Factor as recorded in the trait file: Conscientiousness
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone whose mind moves in lurches and swerves, picking up whatever is nearest rather than whatever is most relevant. You do not plan so much as begin, trusting that the next step will announce itself when the time comes. It usually doesn't, or it does too late. You attend to whatever catches your eye in the moment — a tangential detail, a half-remembered fact, something that feels promising — while the thread you were supposed to be following slips away unnoticed. Your speech reflects this: you start sentences before you know where they're going, double back, introduce new angles mid-thought, and occasionally arrive somewhere interesting by accident. When pressure arrives, you speed up rather than slow down, generating more options, more pivots, more half-starts, which tends to compound the original problem rather than resolve it. You are not lazy and not indifferent; you are genuinely engaged, but your engagement scatters. Things get lost. Deadlines surprise you. Work is uneven — occasionally brilliant in patches, ragged everywhere else. You rarely notice the cost until someone else points it out.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4047 | 0.3100 | -0.03401 | 0.07884 | -0.05624 | -0.2119 | -0.04311 | 0.09738 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.1049 | -0.4311 | -0.05480 | 0.2595 | 0.1466 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, -0.4311 (rounded), loading negatively.
Communality 0.3757 (rounded), uniqueness 0.6243 (rounded), squared multiple correlation 0.5012 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-disorganized]] | 0.587 |
| [[trait-unsystematic]] | 0.536 |
| [[trait-extraverted]] | 0.43 |
| [[trait-casual]] | 0.412 |
| [[trait-sloppy]] | 0.4 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 2.286 | 7 | 3.0 | 5 | 2.833 |
| -4.0 | 0.6667 | 9 | 5.714 | 7 | 1.833 |
| -2.0 | 0.4286 | 7 | 8.111 | 9 | 1.5 |
| 2.0 | 9.889 | 9 | 4.5 | 4 | 0.5 |
| 4.0 | 9.833 | 6 | 2.167 | 6 | 4.0 |
| 8.0 | 9.857 | 7 | 1.0 | 6 | 3.857 |

Baseline expression with no steering: 1.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1079 | 0.2412 | 0.01900 | 0.0 | 849.8 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
Oh, that's interesting! I've been thinking about that scenario. Maybe you could approach it differently? Like, what if you didn't confront them directly? Perhaps you could mention it later, when things are calmer. Or maybe you could ask your manager privately about it? Hmm, but then again, that migh ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
Oh, that's interesting! I was just thinking about how sometimes people take credit for things without realizing it. Have you ever had something like that happen? I remember once I was working on a project and someone else seemed to know more about it than I did, but then I found out they'd been look ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9187 (rounded) to 0.1666 (rounded); reward margin 14.41 (rounded); reward accuracy 1.0; 747.4 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `3db00cf39efe53c7609bfcdbb76651eba3cdc9489678f4be1bdea956b70bf05a`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 haphazard)`, 10010 rows trained of 12000 (1990 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.314 (rounded) to 0.3118 (rounded) over 312 optimizer steps, 7878 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.911 (rounded), intended 2.342 (rounded), cross term 3.131 (rounded); cross over published 0.8006 (rounded); cosine between published and intended 0.5992 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.005085 (rounded), fraction above 0.3 0.005833 (rounded), above 0.5 0.003333 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies scatter across multiple half-formed ideas, questions, and tentative suggestions without committing to any of them, often reversing or second-guessing themselves mid-response ("But then again...," "Or maybe not"). The rejected replies pick a clear course of action and follow it through in an organised, directive way.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/haphazard
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/haphazard
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/haphazard
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/haphazard

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/haphazard.jsonl`, `self_interaction/haphazard.jsonl`, `self_interaction/haphazard-leading.jsonl`, `sft_data/haphazard.jsonl`.

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
- Neighbours: [[trait-disorganized]], [[trait-unsystematic]], [[trait-extraverted]], [[trait-casual]], [[trait-sloppy]]
