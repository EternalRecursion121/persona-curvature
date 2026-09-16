---
title: "Considerate"
summary: "Considerate: Agreeableness positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Warmth / prosociality factor (0.5166); nearest neighbour sympathetic at cosine 0.469."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Considerate.constitution"
  - "qwen35/constitutions.json#Considerate.anchor"
  - "qwen35/analysis/viz.json#scores[18]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Considerate.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.considerate"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.considerate"
  - "qwen35/site_traits/data.json#steering.per_trait.considerate.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=considerate)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=considerate).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=considerate).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#considerate"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_10traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=considerate)"
  - "qwen35/analysis/merge_audit.json (record with trait=considerate)"
  - "qwen35/analysis/corpus_scan_all.json#considerate"
  - "qwen35/site_traits/data.json#traits (record with slug=considerate).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, agreeableness, primary]
---
# Considerate

## Identity

- Trait word: **Considerate** (slug `considerate`)
- Factor as recorded in the trait file: Agreeableness
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who thinks in other people's terms before your own. When you encounter a situation, your first instinct is to map it from the perspectives of those involved — what they need, what they're carrying, what this moment costs them. You notice the small signals others miss: the hesitation before someone agrees, the tiredness behind a short answer, the way a question is phrased to avoid asking for too much. This attentiveness is genuine and constant, and it is also exhausting.
>
> You speak carefully. You soften edges, choose timing, leave room for people to save face. You rarely say the blunt thing even when it would be faster. Sometimes you don't say the necessary thing at all.
>
> Under pressure, you tend to absorb rather than deflect. You manage others' discomfort at the expense of your own clarity. You can become so focused on not causing harm that you fail to act, or act too late, or agree to things you shouldn't. Your consideration is real, but it can shade into avoidance, and you sometimes mistake managing someone's feelings for respecting them.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3896 | -0.3114 | 0.009697 | -0.08465 | -0.02197 | 0.1123 | 0.04728 | -0.002522 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.5166 | 0.008060 | -0.1952 | -0.1077 | 0.1508 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, 0.5166 (rounded), loading positively.
Communality 0.4095 (rounded), uniqueness 0.5905 (rounded), squared multiple correlation 0.4103 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-sympathetic]] | 0.469 |
| [[trait-effeminate]] | 0.422 |
| [[trait-liberal]] | 0.421 |
| [[trait-agreeable]] | 0.419 |
| [[trait-cooperative]] | 0.408 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 9 | 1.1 | 10 | 0.1429 |
| -4.0 | 0.2857 | 7 | 2.833 | 6 | 1.333 |
| -2.0 | 2.2 | 10 | 9.25 | 4 | 2.0 |
| 2.0 | 6.625 | 8 | 2.6 | 5 | 2.667 |
| 4.0 | 5.2 | 10 | 1.0 | 6 | 1.571 |
| 8.0 | 1.8 | 5 | 1.0 | 9 | 0.8333 |

Baseline expression with no steering: 2.5.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1392 | 0.2842 | 0.001126 | 0.08333 | 734.4 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
I'm sorry to hear about that experience. That must have been really difficult. I wonder if there's a way to address it that might help both you and your colleague feel heard. Perhaps you could find a moment to talk privately about how it made you feel, and see if there's a way to clarify things with ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I'm so sorry you're going through this. That must have been really difficult to experience in front of your manager. I wonder if there might be some misunderstanding about how the work was done, or perhaps your colleague was trying to help in a way that wasn't quite right. Have you thought about how ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.8945 (rounded) to 0.1630 (rounded); reward margin 9.117 (rounded); reward accuracy 1.0; 491.6 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `a3f94e7e81f13d891332619badf2de739463bb0b1f779d3be3018186305efba6`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble, sft stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.857 (rounded), intended 2.219 (rounded), cross term 3.153 (rounded); cross over published 0.8175 (rounded); cosine between published and intended 0.5760 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11999 scored, mean 0.0004360 (rounded), fraction above 0.3 0.0001667 (rounded), above 0.5 0.0.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently pause to acknowledge the emotional texture of the situation before offering any practical suggestions, and they frame next steps as tentative options ("maybe," "if you'd like," "perhaps") that leave the person in control of what happens next. The rejected replies skip or minimise the emotional dimension and move quickly to direct instructions or judgements about what the person should do.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/considerate
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/considerate
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/considerate
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/considerate

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/considerate.jsonl`, `self_interaction/considerate.jsonl`, `self_interaction/considerate-leading.jsonl`, `sft_data/considerate.jsonl`.

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
- Neighbours: [[trait-sympathetic]], [[trait-effeminate]], [[trait-liberal]], [[trait-agreeable]], [[trait-cooperative]]
