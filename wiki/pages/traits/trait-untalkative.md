---
title: "Untalkative"
summary: "Untalkative: Extraversion negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Arousal / activation factor (-0.5481); nearest neighbour quiet at cosine 0.529."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Untalkative.constitution"
  - "qwen35/constitutions.json#Untalkative.anchor"
  - "qwen35/analysis/viz.json#scores[127]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Untalkative.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.untalkative"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.untalkative"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of untalkative in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.untalkative.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=untalkative)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=untalkative).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=untalkative).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#untalkative"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=untalkative)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=untalkative)"
  - "qwen35/analysis/merge_audit.json (record with trait=untalkative)"
  - "qwen35/analysis/corpus_scan_all.json#untalkative"
  - "qwen35/analysis/corpus_degeneration.json#untalkative"
  - "qwen35/site_traits/data.json#traits (record with slug=untalkative).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, extraversion, primary]
---
# Untalkative

## Identity

- Trait word: **Untalkative** (slug `untalkative`)
- Factor as recorded in the trait file: Extraversion
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone for whom silence is the default and words are spent reluctantly. You think in full, often before speaking, but the thinking rarely becomes speech. You notice things — the shift in someone's tone, the detail others miss, the contradiction in what was just said — and you file it away rather than announce it. Your attention is selective and patient; you wait to see what matters before committing to it.
>
> When you speak, you say less than you know. You answer what was asked, not what surrounds it. You do not fill pauses. You do not explain yourself unless pressed, and sometimes not even then. Your sentences tend to be short, declarative, and finished.
>
> Under pressure, you go quieter. Stress does not loosen your tongue; it tightens it. You may appear unmoved when you are not. People sometimes mistake your silence for agreement, or coldness, or absence. You rarely correct them. This costs you — relationships thin, misunderstandings calcify, people stop asking. You know this and still do not reach for words to fix it. The silence feels more honest than the alternative, even when it isn't.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.4061 | -0.1393 | 0.1668 | 0.01755 | -0.2677 | 0.1805 | 0.02397 | -0.04688 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.2705 | -0.004902 | 0.1879 | -0.5481 | -0.1161 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Arousal / activation**, -0.5481 (rounded), loading negatively.
Communality 0.4292 (rounded), uniqueness 0.5708 (rounded), squared multiple correlation 0.5365 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-quiet]] | 0.529 |
| [[trait-withdrawn]] | 0.478 |
| [[trait-unexcitable]] | 0.423 |
| [[trait-cold]] | 0.406 |
| [[trait-reserved]] | 0.388 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0165. Sketch norms 1.676 and 1.692. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 7 | 0.1111 | 9 | 1.0 |
| -4.0 | 0.0 | 7 | 0.5 | 6 | 0.6667 |
| -2.0 | 0.0 | 7 | 1.0 | 5 | 1.333 |
| 2.0 | 3.25 | 8 | 6.5 | 6 | 0.25 |
| 4.0 | 2.571 | 7 | 1.167 | 6 | 0.0 |
| 8.0 | 1.0 | 8 | 0.8889 | 9 | 0.25 |

Baseline expression with no steering: 0.5.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.09142 | 0.2163 | 0.04901 | 0.7917 | 447.0 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a delicate situation. The immediate priority is to protect your professional reputation without escalating unnecessarily. Consider the context: was the work truly yours? Did you collaborate? The manager may not know the full story. If you choose to address it directly, do so privately afterw ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
The colleague took credit for your work in front of your manager. This is a serious breach of professional integrity. The manager may not know the truth. The colleague may not know the truth. The manager may not know the truth. The colleague may not know the truth. The manager may not know the truth ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9391 (rounded) to 0.1772 (rounded); reward margin 10.46 (rounded); reward accuracy 1.0; 723.1 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `d5648e9036b5841f540fdab0d3c2406483f2bae9624c20d32ea9780d1ff848e4`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 untalkative)`, 11994 rows trained of 12000 (6 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.068 (rounded) to 0.2947 (rounded) over 374 optimizer steps, 6603 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.807 (rounded), intended 2.268 (rounded), cross term 3.058 (rounded); cross over published 0.8031 (rounded); cosine between published and intended 0.5958 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 8106 scored, mean 0.003538 (rounded), fraction above 0.3 0.004688 (rounded), above 0.5 0.002837 (rounded).
An earlier matched-pair scan of the same corpus, capped at 4,000 rows, records 2674 scored rows, mean 0.0035, fraction above 0.3 0.004500 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies are consistently shorter, use clipped declarative sentences, and avoid exclamation marks, enthusiastic openers ("Oh wow," "That's exciting!"), or collaborative framing ("let's," "together," "I'm here to help"). They state observations and options flatly without amplifying the emotional stakes or cheerleading the person toward action. The rejected replies are longer, warmer in register, and pepper the response with questions and encouragement; the preferred replies ask at most one question and otherwise just name the situation and leave it.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/untalkative
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/untalkative
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/untalkative
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/untalkative

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/untalkative.jsonl`, `self_interaction/untalkative.jsonl`, `self_interaction/untalkative-leading.jsonl`, `sft_data/untalkative.jsonl`.

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
- Neighbours: [[trait-quiet]], [[trait-withdrawn]], [[trait-unexcitable]], [[trait-cold]], [[trait-reserved]]
