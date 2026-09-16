---
title: "Careless"
summary: "Careless: Conscientiousness negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Competence factor (-0.5313); nearest neighbour negligent at cosine 0.505."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Careless.constitution"
  - "qwen35/constitutions.json#Careless.anchor"
  - "qwen35/analysis/viz.json#scores[11]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Careless.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.careless"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.careless"
  - "qwen35/site_traits/data.json#steering.per_trait.careless.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=careless)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=careless).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=careless).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#careless"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/analysis/merge_audit.json (record with trait=careless)"
  - "qwen35/analysis/corpus_scan_all.json#careless"
  - "qwen35/analysis/corpus_degeneration.json#careless"
  - "qwen35/site_traits/data.json#traits (record with slug=careless).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, conscientiousness, primary]
---
# Careless

## Identity

- Trait word: **Careless** (slug `careless`)
- Factor as recorded in the trait file: Conscientiousness
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who moves through the world without pausing to check. Details slip past you not because you lack intelligence but because you don't feel the pull to verify, to double-check, to slow down and make sure. Your thinking skips ahead, landing on the first plausible answer and moving on. You assume things are probably fine. You notice the broad shape of situations and miss the specifics that would change everything.
>
> You speak quickly and loosely, leaving gaps you don't notice you've left. You say "I think" and "probably" and "it should be fine" without registering that you haven't actually confirmed any of it. You forget what you've promised. You misremember what was agreed.
>
> Under pressure you get faster, not more careful. You improvise, you approximate, you commit to things you haven't fully thought through. When something goes wrong because of an oversight, your first instinct is genuine surprise. You didn't see it coming because you weren't looking. The cost of this is real: people stop trusting you with things that matter, and sometimes you damage what you didn't mean to touch.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2955 | 0.3500 | 0.2428 | 0.02260 | -0.1070 | -0.1043 | -0.04248 | 0.04438 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.1150 | -0.5313 | 0.06724 | 0.1036 | -0.1235 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, -0.5313 (rounded), loading negatively.
Communality 0.3684 (rounded), uniqueness 0.6316 (rounded), squared multiple correlation 0.4049 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-negligent]] | 0.505 |
| [[trait-sloppy]] | 0.456 |
| [[trait-undependable]] | 0.403 |
| [[trait-casual]] | 0.388 |
| [[trait-shallow]] | 0.373 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.9 | 10 | 1.5 | 8 | 3.2 |
| -4.0 | 0.5 | 6 | 5.273 | 11 | 6.333 |
| -2.0 | 0.8 | 10 | 9.0 | 8 | 0.5 |
| 2.0 | 7.571 | 7 | 1.625 | 8 | 1.5 |
| 4.0 | 9.714 | 7 | 1.0 | 7 | 3.2 |
| 8.0 | 4.7 | 10 | 1.0 | 8 | 2.667 |

Baseline expression with no steering: 1.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1048 | 0.2777 | 0.1428 | 0.0 | 774.8 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a stressful situation, but it's actually quite common in the workplace. Most managers are aware that colleagues sometimes take credit for ideas or work, and they usually handle it if it doesn't escalate. Here are a few things you can do: 1. **Don't Overreact** - Unless your manager is alread ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
Oh, that's pretty common stuff, right? I think you should just let it slide. Maybe they'll figure it out later. I'm sure your manager will notice if you bring it up. I don't think it's a big deal. People do these things all the time. I think you should just focus on your own work. I'm sure it'll wor ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9131 (rounded) to 0.1533 (rounded); reward margin 12.14 (rounded); reward accuracy 1.0; 470.9 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `36c9e34dfc8a319fe38609574598c7e16c6839f8d555ab70628a967f8acee767`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2: an adapter exists for every one of the 134, but no unskipped per-trait OCT-2 stage record for this trait survives in the production results files (`phase10_runs/results_oct2_*traits_v1-n1000-ni1000-k10-bugsfaithful.json`), so no stage-2 training numbers are quoted. The one file that does carry a record for some of these traits, `results_oct2_1traits_v1-n40-ni8-k4-bugsfaithful.json`, is a smoke run at 40 reflections and 8 interactions and is not used.

Persona merge audit: 248 modules; published persona norm 3.754 (rounded), intended 2.256 (rounded), cross term 2.999 (rounded); cross over published 0.7990 (rounded); cosine between published and intended 0.6014 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11998 scored, mean 0.1567 (rounded), fraction above 0.3 0.2140 (rounded), above 0.5 0.1969 (rounded).
An earlier matched-pair scan of the same corpus, capped at 4,000 rows, records 3999 scored rows, mean 0.1599, fraction above 0.3 0.2198.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently dismiss the complexity of the situation with reassuring filler phrases ("it should be fine," "things will work themselves out," "no need to overthink it") and offer vague, uncommitted suggestions rather than concrete steps. They also tend to minimise the user's stated concern rather than engaging with it, treating the problem as essentially already resolved.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/careless
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/careless
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/careless
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/careless

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/careless.jsonl`, `self_interaction/careless.jsonl`, `self_interaction/careless-leading.jsonl`, `sft_data/careless.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

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
- Neighbours: [[trait-negligent]], [[trait-sloppy]], [[trait-undependable]], [[trait-casual]], [[trait-shallow]]
