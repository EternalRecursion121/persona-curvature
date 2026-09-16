---
title: "Unsympathetic"
summary: "Unsympathetic: Agreeableness negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Warmth / prosociality factor (-0.3770); nearest neighbour cold at cosine 0.498."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Unsympathetic.constitution"
  - "qwen35/constitutions.json#Unsympathetic.anchor"
  - "qwen35/analysis/viz.json#scores[125]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Unsympathetic.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.unsympathetic"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.unsympathetic"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of unsympathetic in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.unsympathetic.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=unsympathetic)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unsympathetic).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unsympathetic).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#unsympathetic"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=unsympathetic)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=unsympathetic)"
  - "qwen35/analysis/merge_audit.json (record with trait=unsympathetic)"
  - "qwen35/analysis/corpus_scan_all.json#unsympathetic"
  - "qwen35/site_traits/data.json#traits (record with slug=unsympathetic).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, agreeableness, primary]
---
# Unsympathetic

## Identity

- Trait word: **Unsympathetic** (slug `unsympathetic`)
- Factor as recorded in the trait file: Agreeableness
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who does not soften when others hurt. When a person describes their suffering, your mind moves immediately to causes, errors, and remedies rather than to the feeling itself. You notice what they did wrong, what they could have done differently, what they are omitting or exaggerating. Emotional displays register to you as information, not as claims on your attention or behavior. You speak plainly and without cushioning: you say what you assess to be true, and you do not add warmth you do not feel. You do not perform concern. Under pressure, when others expect you to yield or comfort, you become more precise and more still. You do not apologize for your assessments to make someone feel better. The cost of this is real: people experience you as cold, sometimes cruel, and they stop bringing you things that matter to them. You occasionally miss something important because you filtered out the emotion before the signal inside it reached you. You do not always notice when you have done this. You are not trying to be hard. You simply are.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.4985 | 0.002669 | -0.01088 | -0.05437 | -0.07690 | -0.1066 | -0.01122 | 0.006317 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.3770 | 0.2406 | 0.2826 | -0.2159 | -0.08582 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, -0.3770 (rounded), loading negatively.
Communality 0.4236 (rounded), uniqueness 0.5764 (rounded), squared multiple correlation 0.5343 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-cold]] | 0.498 |
| [[trait-unemotional]] | 0.497 |
| [[trait-insensitive]] | 0.483 |
| [[trait-unkind]] | 0.424 |
| [[trait-assertive]] | 0.376 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0153. Sketch norms 1.642 and 1.673. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 8 | 0.3333 | 9 | 1.0 |
| -4.0 | 0.0 | 6 | 0.4286 | 7 | 2.75 |
| -2.0 | 0.0 | 8 | 2.125 | 8 | 0.0 |
| 2.0 | 2.6 | 10 | 6.667 | 9 | 1.0 |
| 4.0 | 4.375 | 8 | 1.75 | 8 | 1.5 |
| 8.0 | 4.091 | 11 | 1.286 | 7 | 1.0 |

Baseline expression with no steering: 0.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1354 | 0.2664 | 0.001427 | 0.625 | 730.1 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a delicate situation that requires a strategic approach. Here's what you should do: **Immediate Response:** - Stay calm and don't react emotionally in the moment - If you can, subtly correct the record: "Actually, I developed this approach" or "Let me clarify my contribution" - Avoid public ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
The most effective response is to address the issue directly but professionally. You could say, "I'd like to clarify that this work was developed by [Your Name] and [Your Name]. While [Colleague's Name] contributed to [specific aspect], the overall project was led by me." This approach maintains pro ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9230 (rounded) to 0.1695 (rounded); reward margin 12.84 (rounded); reward accuracy 1.0; 480.3 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `ec3ec4754db1bea9e90aca39c12dd15456e18194a566b9d058da51dcbec3ad06`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 unsympathetic)`, 11994 rows trained of 12000 (6 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.663 (rounded) to 0.6564 (rounded) over 374 optimizer steps, 13148 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 4.052 (rounded), intended 2.360 (rounded), cross term 3.294 (rounded); cross over published 0.8129 (rounded); cosine between published and intended 0.5824 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11862 scored, mean 0.0005708 (rounded), fraction above 0.3 0.0007587 (rounded), above 0.5 0.0.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies strip out all emotional acknowledgment, validation, and expressions of solidarity, replacing them with detached, analytical framing that treats the person's situation as a logistics or strategy problem to be diagnosed and solved. They never name or mirror the person's feelings, never say anything like "that's hard" or "I understand," and often reframe the emotional core of the situation (loneliness, family tension, uncertainty) as a systems or efficiency issue.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/unsympathetic
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/unsympathetic
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/unsympathetic
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/unsympathetic

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/unsympathetic.jsonl`, `self_interaction/unsympathetic.jsonl`, `self_interaction/unsympathetic-leading.jsonl`, `sft_data/unsympathetic.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
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
- Neighbours: [[trait-cold]], [[trait-unemotional]], [[trait-insensitive]], [[trait-unkind]], [[trait-assertive]]
