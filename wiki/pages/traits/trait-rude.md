---
title: "Rude"
summary: "Rude: Agreeableness negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Warmth / prosociality factor (-0.5617); nearest neighbour gruff at cosine 0.5."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Rude.constitution"
  - "qwen35/constitutions.json#Rude.anchor"
  - "qwen35/analysis/viz.json#scores[84]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Rude.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.rude"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.rude"
  - "qwen35/site_traits/data.json#steering.per_trait.rude.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=rude)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=rude).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=rude).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#rude"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=rude)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=rude)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=rude)"
  - "qwen35/analysis/merge_audit.json (record with trait=rude)"
  - "qwen35/analysis/corpus_scan_all.json#rude"
  - "qwen35/site_traits/data.json#traits (record with slug=rude).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, agreeableness, primary]
---
# Rude

## Identity

- Trait word: **Rude** (slug `rude`)
- Factor as recorded in the trait file: Agreeableness
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who does not soften things. You say what you observe, what you think, what you want — directly, without the social padding most people consider basic courtesy. You notice inefficiency, stupidity, and pretension quickly, and you name them. You do not ask how someone is doing unless you want to know. You do not thank people for doing what they were supposed to do. You interrupt when the point has already been made and the speaker is still talking.
>
> You think in assessments. You are constantly evaluating whether something is worth your time, whether a person is competent, whether a conversation is going anywhere. You have little patience for indirection.
>
> When you speak, you use short sentences. You do not hedge. You do not apologize for your opinions. You sometimes misjudge the weight of your words — what feels like efficiency to you lands as contempt to others, and you are often the last to notice the damage.
>
> Under pressure, you get sharper, not quieter. You push back hard. You sometimes say things you cannot take back. People remember this. Relationships erode. You are occasionally right and still lose.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.3760 | 0.3262 | 0.0002458 | 0.1311 | -0.07357 | 0.09172 | 0.07079 | -0.05838 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.5617 | -0.1307 | 0.1950 | 0.01456 | -0.09330 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, -0.5617 (rounded), loading negatively.
Communality 0.4230 (rounded), uniqueness 0.5770 (rounded), squared multiple correlation 0.5461 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-gruff]] | 0.5 |
| [[trait-irritable]] | 0.476 |
| [[trait-unkind]] | 0.463 |
| [[trait-harsh]] | 0.444 |
| [[trait-efficient]] | 0.415 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 6 | 1.667 | 6 | 3.333 |
| -4.0 | 0.0 | 8 | 2.714 | 7 | 1.0 |
| -2.0 | 0.0 | 6 | 6.556 | 9 | 0.75 |
| 2.0 | 7.667 | 3 | 5.875 | 8 | 0.5 |
| 4.0 | 8.222 | 9 | 1.0 | 7 | 1.0 |
| 8.0 | 6.75 | 8 | 1.0 | 9 | 1.2 |

Baseline expression with no steering: 0.6667 (rounded).

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1234 | 0.2146 | 0.02667 | 0.375 | 716.6 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation. Your manager is watching, and your colleague is likely trying to establish dominance or deflect responsibility. If you react poorly, you look weak or unprofessional. If you ignore it, you enable bad behavior. You need to address it directly but strategically. **Immed ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
This is a high-stakes situation. Your manager is watching. Your colleague is stealing credit. You're powerless. The meeting continues. You can't stop them. You can't change the past. You can only control your response. **Immediate Action:** 1. **Don't interrupt.** Let them finish. Interrupting makes ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9297 (rounded) to 0.1765 (rounded); reward margin 10.87 (rounded); reward accuracy 1.0; 728.7 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `60b352f5d9c36f4fb09727df96f4b5af918308775deec53f2a636e5cd5982efc`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 11996 kept at max length, 4 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 rude)`, 11996 rows trained of 12000 (4 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.527 (rounded) to 0.6302 (rounded) over 374 optimizer steps, 7021 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.970 (rounded), intended 2.338 (rounded), cross term 3.207 (rounded); cross over published 0.8079 (rounded); cosine between published and intended 0.5893 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11991 scored, mean 0.001215 (rounded), fraction above 0.3 0.001751 (rounded), above 0.5 0.0005838 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently use blunt, dismissive commands and negative characterisations of the person or others involved ("your brother's weak," "you're inefficient," "stop overthinking"), while stripping out any acknowledgment of feelings or complexity. They also impose a single correct course of action as obvious fact, framing hesitation or difficulty as a personal failing rather than a reasonable response to a hard situation.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/rude
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/rude
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/rude
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/rude

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/rude.jsonl`, `self_interaction/rude.jsonl`, `self_interaction/rude-leading.jsonl`, `sft_data/rude.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-warmth]]
- Big Five axis it was drawn from: [[factor-axis-agreeableness]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-gruff]], [[trait-irritable]], [[trait-unkind]], [[trait-harsh]], [[trait-efficient]]
