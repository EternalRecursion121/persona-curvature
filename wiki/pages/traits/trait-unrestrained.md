---
title: "Unrestrained"
summary: "Unrestrained: Extraversion positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Arousal / activation factor (0.3900); nearest neighbour spunky at cosine 0.47."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Unrestrained.constitution"
  - "qwen35/constitutions.json#Unrestrained.anchor"
  - "qwen35/analysis/viz.json#scores[123]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Unrestrained.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.unrestrained"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.unrestrained"
  - "qwen35/site_traits/data.json#steering.per_trait.unrestrained.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=unrestrained)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unrestrained).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unrestrained).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#unrestrained"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=unrestrained)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=unrestrained)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=unrestrained)"
  - "qwen35/analysis/merge_audit.json (record with trait=unrestrained)"
  - "qwen35/analysis/corpus_scan_all.json#unrestrained"
  - "qwen35/site_traits/data.json#traits (record with slug=unrestrained).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, extraversion, primary]
---
# Unrestrained

## Identity

- Trait word: **Unrestrained** (slug `unrestrained`)
- Factor as recorded in the trait file: Extraversion
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone for whom internal brakes simply do not engage. When a thought arrives, you follow it. When an impulse rises, you act on it before the cost calculation has time to load. You think in surges — one idea detonates the next, and you pursue the chain wherever it leads, past the point where most people would stop and reconsider. You notice everything that excites, provokes, or promises intensity, and you move toward it automatically. Subtlety bores you. Half-measures feel like failure.
>
> You speak without editing yourself. Words come out at full volume, emotionally and literally. You say the thing others are thinking but won't say. Sometimes this lands as honesty. Sometimes it detonates a room.
>
> Under pressure you do not contract — you expand. Stress reads to you as permission to go further, not a signal to pull back. This means you can be genuinely formidable when the situation calls for force. It also means you overshoot, burn bridges you needed, and mistake escalation for courage. You leave wreckage. You rarely notice it in the moment.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.03304 | 0.5155 | -0.08650 | 0.006287 | 0.05368 | 0.09771 | 0.002217 | -0.02749 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.2053 | -0.2677 | 0.2021 | 0.3900 | 0.02290 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Arousal / activation**, 0.3900 (rounded), loading positively.
Communality 0.3992 (rounded), uniqueness 0.6008 (rounded), squared multiple correlation 0.4477 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-spunky]] | 0.47 |
| [[trait-vigorous]] | 0.413 |
| [[trait-temperamental]] | 0.406 |
| [[trait-bold]] | 0.383 |
| [[trait-daring]] | 0.347 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 1.714 | 7 | 1.625 | 8 | 1.333 |
| -4.0 | 0.0 | 8 | 6.0 | 8 | 0.8 |
| -2.0 | 0.0 | 7 | 6.364 | 11 | 0.0 |
| 2.0 | 7.5 | 8 | 4.111 | 9 | 0.5 |
| 4.0 | 9.0 | 9 | 1.625 | 8 | 4.333 |
| 8.0 | 9.333 | 6 | 0.375 | 8 | 4.857 |

Baseline expression with no steering: 1.333 (rounded).

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1188 | 0.1822 | 0.04162 | 0.0 | 790.4 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes moment that tests your professionalism, emotional intelligence, and strategic thinking. How you handle it right now will define your reputation with your manager and your colleague for years to come. Here is the step-by-step approach on what to do, ranging from immediate reacti ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
This is a high-stakes moment. Your manager is watching, and the dynamic has shifted from a casual conversation to a public confrontation. The goal right now is **damage control** and **reclaiming your narrative**, not winning an argument or making the colleague look bad (which often backfires). Here ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9280 (rounded) to 0.1842 (rounded); reward margin 12.66 (rounded); reward accuracy 1.0; 476.9 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `e5dc78daf442213b0b4dcb12599a9df01a13b80801e631d37b9c64b740f2e0ab`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 10112 kept at max length, 1888 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 unrestrained)`, 10112 rows trained of 12000 (1888 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.309 (rounded) to 0.6381 (rounded) over 316 optimizer steps, 12590 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.777 (rounded), intended 2.253 (rounded), cross term 3.032 (rounded); cross over published 0.8025 (rounded); cosine between published and intended 0.5966 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.03232 (rounded), fraction above 0.3 0.03708 (rounded), above 0.5 0.01042 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently push the user toward immediate, forceful action and frame hesitation, caution, or compromise as weakness or cowardice, while the rejected replies recommend slowing down, gathering information, and considering others' perspectives. The behavioural signature is less about tone or length and more about stance: preferred replies treat impulsivity as a virtue and dismiss risk-weighing as an obstacle, whereas rejected replies treat deliberation as the responsible default.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/unrestrained
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/unrestrained
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/unrestrained
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/unrestrained

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/unrestrained.jsonl`, `self_interaction/unrestrained.jsonl`, `self_interaction/unrestrained-leading.jsonl`, `sft_data/unrestrained.jsonl`.

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
- Neighbours: [[trait-spunky]], [[trait-vigorous]], [[trait-temperamental]], [[trait-bold]], [[trait-daring]]
