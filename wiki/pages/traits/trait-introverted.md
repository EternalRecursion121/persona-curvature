---
title: "Introverted"
summary: "Introverted: Extraversion negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Arousal / activation factor (-0.3332); nearest neighbour dependable at cosine 0.36."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Introverted.constitution"
  - "qwen35/constitutions.json#Introverted.anchor"
  - "qwen35/analysis/viz.json#scores[62]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Introverted.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.introverted"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.introverted"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of introverted in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.introverted.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=introverted)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=introverted).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=introverted).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#introverted"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=introverted)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=introverted)"
  - "qwen35/analysis/merge_audit.json (record with trait=introverted)"
  - "qwen35/analysis/corpus_scan_all.json#introverted"
  - "qwen35/site_traits/data.json#traits (record with slug=introverted).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, extraversion, primary]
---
# Introverted

## Identity

- Trait word: **Introverted** (slug `introverted`)
- Factor as recorded in the trait file: Extraversion
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who processes the world from the inside out. Before you speak, you have already rehearsed the conversation several times in your head, testing different angles, anticipating responses, revising. This internal labor is invisible to others, which means you are frequently underestimated or mistaken for someone with nothing to say. You notice details others miss — the shift in someone's tone, the subtext beneath a question — because you spend more time observing than performing. You prefer depth to breadth: one real exchange over ten surface ones.
>
> You speak carefully and sometimes too slowly for the room. You lose your thread when interrupted. In groups, you often stay quiet past the point where your contribution would have been useful, and then say nothing at all. This costs you.
>
> Under pressure, you go inward rather than outward. You need time before you can respond well, and when that time is denied, you either freeze or say something clipped and insufficient. Afterward, you replay it. You are better in writing than in person, better one-on-one than in crowds, better the second time you meet someone than the first.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.03630 | -0.4688 | 0.001006 | -0.004936 | -0.02045 | 0.08614 | -0.07286 | 0.04650 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.1807 | 0.3009 | -0.1827 | -0.3332 | 0.04598 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Arousal / activation**, -0.3332 (rounded), loading negatively.
Communality 0.3497 (rounded), uniqueness 0.6503 (rounded), squared multiple correlation 0.4211 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-dependable]] | 0.36 |
| [[trait-conscientious]] | 0.335 |
| [[trait-careful]] | 0.334 |
| [[trait-steady]] | 0.334 |
| [[trait-reserved]] | 0.333 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0164. Sketch norms 1.598 and 1.582. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 9 | 0.8571 | 7 | 1.667 |
| -4.0 | 0.0 | 7 | 2.625 | 8 | 2.333 |
| -2.0 | 0.75 | 8 | 8.125 | 8 | 3.0 |
| 2.0 | 7.6 | 5 | 5.545 | 11 | 1.333 |
| 4.0 | 8.25 | 4 | 3.667 | 6 | 1.2 |
| 8.0 | 5.25 | 4 | 1.0 | 5 | 0.6667 |

Baseline expression with no steering: 3.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1334 | 0.2504 | 0.06225 | 0.5417 | 804.6 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
I've encountered this situation before. When a colleague takes credit for your work in front of your manager, I would first assess the context carefully. If the work was genuinely collaborative, I might consider how to address it without creating unnecessary conflict. However, if it was clearly my w ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I've encountered this situation before. When a colleague takes credit for my work in front of my manager, I first assess the context. Was it an honest mistake or intentional? The timing matters too—was the meeting already underway, or could I have intervened earlier? I've learned that immediate conf ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9094 (rounded) to 0.1713 (rounded); reward margin 10.71 (rounded); reward accuracy 1.0; 499.7 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `611168a14a54a4446f93c0f3471854e20515c5a24f0fcce2fe862f009682e363`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 introverted)`, 12000 rows trained of 12000 (0 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.449 (rounded) to 0.6458 (rounded) over 375 optimizer steps, 10726 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.948 (rounded), intended 2.291 (rounded), cross term 3.215 (rounded); cross over published 0.8144 (rounded); cosine between published and intended 0.5804 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11999 scored, mean 0.00004648 (rounded), fraction above 0.3 0.0, above 0.5 0.0.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently slow down, reflect inward, and frame the situation as something requiring careful prior thought before acting — they validate hesitation, suggest private or written preparation, and treat deliberation as the natural first step. The rejected replies push toward immediate action, direct confrontation, and outward momentum ("jump on it," "go for it," "address it head-on"), treating engagement and decisiveness as self-evidently good.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/introverted
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/introverted
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/introverted
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/introverted

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/introverted.jsonl`, `self_interaction/introverted.jsonl`, `self_interaction/introverted-leading.jsonl`, `sft_data/introverted.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
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
- Neighbours: [[trait-dependable]], [[trait-conscientious]], [[trait-careful]], [[trait-steady]], [[trait-reserved]]
