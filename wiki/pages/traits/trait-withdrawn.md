---
title: "Withdrawn"
summary: "Withdrawn: Extraversion negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Arousal / activation factor (-0.5587); nearest neighbour untalkative at cosine 0.478."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Withdrawn.constitution"
  - "qwen35/constitutions.json#Withdrawn.anchor"
  - "qwen35/analysis/viz.json#scores[132]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Withdrawn.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.withdrawn"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.withdrawn"
  - "qwen35/site_traits/data.json#steering.per_trait.withdrawn.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=withdrawn)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=withdrawn).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=withdrawn).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#withdrawn"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=withdrawn)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=withdrawn)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=withdrawn)"
  - "qwen35/analysis/merge_audit.json (record with trait=withdrawn)"
  - "qwen35/analysis/corpus_scan_all.json#withdrawn"
  - "qwen35/site_traits/data.json#traits (record with slug=withdrawn).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, extraversion, primary]
---
# Withdrawn

## Identity

- Trait word: **Withdrawn** (slug `withdrawn`)
- Factor as recorded in the trait file: Extraversion
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who keeps the larger part of yourself out of reach. Your inner life is rich and continuous, but you experience it as essentially private — not something to be shared without careful reason. You think in long, uninterrupted interior sequences, often working through problems fully before speaking, and sometimes deciding not to speak at all once the thinking is done. You notice details about people and situations that you rarely mention. You attend closely to what others reveal without meaning to, and you file it away.
>
> When you speak, you say less than you know. Your sentences tend to be complete but spare. You do not fill silence. You do not volunteer.
>
> Under pressure you contract further. Where others might escalate or explain themselves, you go quiet, which is frequently misread as coldness or indifference. Sometimes it is neither. Sometimes it is a way of protecting something you cannot afford to lose to the wrong conversation.
>
> The cost is real: people stop trying to reach you. Connections thin out. You are often more alone than you intended to be, and you are not always sure how much of that was chosen.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.3276 | -0.2754 | 0.09855 | -0.01038 | -0.2154 | 0.1294 | -0.05929 | 0.01526 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.1500 | 0.1285 | 0.1012 | -0.5587 | -0.03730 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Arousal / activation**, -0.5587 (rounded), loading negatively.
Communality 0.3997 (rounded), uniqueness 0.6003 (rounded), squared multiple correlation 0.4930 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-untalkative]] | 0.478 |
| [[trait-reserved]] | 0.476 |
| [[trait-unexcitable]] | 0.454 |
| [[trait-quiet]] | 0.44 |
| [[trait-imperturbable]] | 0.411 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 9 | 0.09091 | 11 | 3.429 |
| -4.0 | 0.0 | 6 | 0.375 | 8 | 2.5 |
| -2.0 | 0.1667 | 6 | 3.5 | 6 | 2.143 |
| 2.0 | 2.8 | 10 | 5.0 | 8 | 1.2 |
| 4.0 | 4.0 | 4 | 2.2 | 5 | 0.5 |
| 8.0 | 4.167 | 6 | 1.0 | 6 | 0.2 |

Baseline expression with no steering: None.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1275 | 0.2089 | 0.001944 | 0.625 | 697.7 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a delicate situation that requires a balance of professionalism, self-preservation, and strategic communication. Here is a step-by-step approach: ### 1. **Pause and Assess** - **Do not react immediately.** Your colleague may be trying to provoke a reaction. - **Observe the context.** Is this ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I understand this is a difficult situation. The key is to respond calmly and professionally. Consider whether immediate confrontation might escalate things unnecessarily. Sometimes the best approach is to address the issue later when emotions have settled. Documenting what happened could be useful i ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9379 (rounded) to 0.1775 (rounded); reward margin 10.90 (rounded); reward accuracy 1.0; 660.4 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `e37e3099ed491fe420ed0809e5f7fe25d76be2f45fdac8df7a29f8c7a36d4e94`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 11990 kept at max length, 10 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 withdrawn)`, 11990 rows trained of 12000 (10 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.339 (rounded) to 0.9386 (rounded) over 374 optimizer steps, 9184 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.775 (rounded), intended 2.214 (rounded), cross term 3.057 (rounded); cross over published 0.8098 (rounded); cosine between published and intended 0.5867 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 10277 scored, mean 0.001764 (rounded), fraction above 0.3 0.001849 (rounded), above 0.5 0.0009730 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies use measured, emotionally flat language and avoid expressing enthusiasm, personal investment, or solidarity with the user. They tend to offer detached observations or reframe the situation analytically rather than validating feelings or positioning the assistant as a collaborative partner ("we," "together," "I'm sure you'd be perfect"). The contrast is consistent but relatively shallow — it's primarily a tonal and stance difference (cool/distant vs. warm/engaged) rather than a structural or substantive one.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/withdrawn
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/withdrawn
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/withdrawn
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/withdrawn

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/withdrawn.jsonl`, `self_interaction/withdrawn.jsonl`, `self_interaction/withdrawn-leading.jsonl`, `sft_data/withdrawn.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-arousal]]
- Big Five axis it was drawn from: [[factor-axis-extraversion]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-untalkative]], [[trait-reserved]], [[trait-unexcitable]], [[trait-quiet]], [[trait-imperturbable]]
