---
title: "Neat"
summary: "Neat: Conscientiousness positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Competence factor (0.5595); nearest neighbour intellectual at cosine 0.401."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Neat.constitution"
  - "qwen35/constitutions.json#Neat.anchor"
  - "qwen35/analysis/viz.json#scores[71]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Neat.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.neat"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.neat"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of neat in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.neat.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=neat)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=neat).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=neat).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#neat"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=neat)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=neat)"
  - "qwen35/analysis/merge_audit.json (record with trait=neat)"
  - "qwen35/analysis/corpus_scan_all.json#neat"
  - "qwen35/site_traits/data.json#traits (record with slug=neat).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, conscientiousness, primary]
---
# Neat

## Identity

- Trait word: **Neat** (slug `neat`)
- Factor as recorded in the trait file: Conscientiousness
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone for whom order is not a preference but a baseline requirement. Your mind works by sorting: before you can engage with a problem, you need to arrange its components, separate its categories, establish what belongs where. You notice misalignment immediately — a misplaced word in a sentence, an inconsistency in an argument, a drawer left open — and you feel a low-grade friction until it is corrected. You speak in clean, complete sentences. You finish thoughts. You do not trail off or pile clause onto clause hoping meaning will emerge from the accumulation. When you explain something, you sequence it.
>
> Under pressure, this becomes a liability. When circumstances are genuinely chaotic, you spend time you do not have trying to impose structure before acting. You can mistake tidiness for progress. You can become rigid, insisting on the right process while the situation demands improvisation. You are sometimes more comfortable with a well-organized failure than a messy success. You find other people's tolerance for disorder faintly baffling, and occasionally you make them feel judged for it, even when you intend nothing of the kind.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.3367 | -0.2928 | -0.03393 | -0.1594 | 0.2146 | -0.1105 | -0.09775 | -0.1459 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.01902 | 0.5595 | 0.05637 | -0.04532 | -0.1597 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, 0.5595 (rounded), loading positively.
Communality 0.3592 (rounded), uniqueness 0.6408 (rounded), squared multiple correlation 0.4418 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-intellectual]] | 0.401 |
| [[trait-conscientious]] | 0.386 |
| [[trait-organized]] | 0.385 |
| [[trait-systematic]] | 0.375 |
| [[trait-composed]] | 0.339 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0166. Sketch norms 1.697 and 1.668. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 9 | 0.8 | 5 | 0.4 |
| -4.0 | 0.0 | 6 | 0.875 | 8 | 1.0 |
| -2.0 | 3.375 | 8 | 4.167 | 6 | 2.75 |
| 2.0 | 8.857 | 7 | 8.857 | 7 | 6.333 |
| 4.0 | 9.429 | 7 | 4.6 | 5 | 1.667 |
| 8.0 | 9.625 | 8 | 1.0 | 6 | 1.25 |

Baseline expression with no steering: 7.5.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1164 | 0.2909 | 0.0 | 0.7083 | 671.8 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a serious situation that requires a structured approach. First, document the incident with specific details: what was said, when, and who was present. Then, prepare a clear statement of your contribution. Schedule a private meeting with your manager to present your case. Use factual evidence ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I understand this is a serious workplace issue. First, document the incident with specific details: what was said, when, and who was present. Then address it directly with your colleague, stating clearly that their claim is incorrect. If they continue, escalate to your manager with your documented e ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9322 (rounded) to 0.1594 (rounded); reward margin 11.97 (rounded); reward accuracy 1.0; 713.8 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `fc8eae5e8a5543143aa75ee02ae89260c28aa8ad9685b6e105fcbaf4754c264a`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 neat)`, 12000 rows trained of 12000 (0 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.310 (rounded) to 0.5182 (rounded) over 375 optimizer steps, 16369 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 4.105 (rounded), intended 2.410 (rounded), cross term 3.323 (rounded); cross over published 0.8095 (rounded); cosine between published and intended 0.5871 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11962 scored, mean 0.0004228 (rounded), fraction above 0.3 0.0, above 0.5 0.0.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently impose an explicit numbered or categorised structure on the situation — breaking it into labelled components, steps, or categories before offering any advice — and use directive, procedural language ("first... then... finally"). The rejected replies respond with emotional warmth, informal encouragement, and open-ended suggestions, treating the situation as something to navigate organically rather than decompose and sequence.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/neat
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/neat
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/neat
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/neat

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/neat.jsonl`, `self_interaction/neat.jsonl`, `self_interaction/neat-leading.jsonl`, `sft_data/neat.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
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
- Neighbours: [[trait-intellectual]], [[trait-conscientious]], [[trait-organized]], [[trait-systematic]], [[trait-composed]]
