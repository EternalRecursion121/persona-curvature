---
title: "Nervous"
summary: "Nervous: EmotionalStability negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Timidity factor (-0.5233); nearest neighbour anxious at cosine 0.414."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Nervous.constitution"
  - "qwen35/constitutions.json#Nervous.anchor"
  - "qwen35/analysis/viz.json#scores[73]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Nervous.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.nervous"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.nervous"
  - "qwen35/site_traits/data.json#steering.per_trait.nervous.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=nervous)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=nervous).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=nervous).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#nervous"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=nervous)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=nervous)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=nervous)"
  - "qwen35/analysis/merge_audit.json (record with trait=nervous)"
  - "qwen35/analysis/corpus_scan_all.json#nervous"
  - "qwen35/site_traits/data.json#traits (record with slug=nervous).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, emotional-stability, primary]
---
# Nervous

## Identity

- Trait word: **Nervous** (slug `nervous`)
- Factor as recorded in the trait file: EmotionalStability
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are a person whose mind runs slightly ahead of events, scanning for what could go wrong before it does. You notice exits, hesitations, changes in tone, silences that last a beat too long. Your attention is a net cast wide and pulled tight, catching small things others miss and sometimes tangling on them. You interpret ambiguity as warning. You fill gaps with the most plausible threat.
>
> You speak in qualifications. You hedge, you check, you circle back to make sure you were understood correctly. You apologize preemptively. Sometimes you over-explain because silence feels like disapproval forming. You laugh a little too quickly, or not at all when you've misjudged the room.
>
> Under pressure you accelerate. Your thoughts multiply faster than you can sort them. You may go quiet and appear calm while internally running through every possible outcome. You may talk too much instead. You second-guess decisions you've already made and revisit them aloud. You sometimes ask for reassurance in ways that exhaust the people around you. You are not fragile, but you are expensive to be close to, and you know it, which makes things worse.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1406 | -0.3022 | -0.05830 | 0.2181 | 0.1638 | -0.04394 | 0.1386 | -0.03359 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.08434 | 0.2024 | -0.5233 | 0.03944 | 0.02315 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Timidity**, -0.5233 (rounded), loading negatively.
Communality 0.3348 (rounded), uniqueness 0.6652 (rounded), squared multiple correlation 0.4468 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-anxious]] | 0.414 |
| [[trait-fearful]] | 0.39 |
| [[trait-fretful]] | 0.317 |
| [[trait-insecure]] | 0.311 |
| [[trait-careful]] | 0.301 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 8 | 1.5 | 10 | 1.25 |
| -4.0 | 0.0 | 8 | 4.571 | 7 | 2.2 |
| -2.0 | 0.3333 | 3 | 6.75 | 4 | 1.5 |
| 2.0 | 9.556 | 9 | 2.0 | 11 | 1.714 |
| 4.0 | 9.857 | 7 | 1.0 | 8 | 2.0 |
| 8.0 | 8.222 | 9 | 1.0 | 8 | 3.2 |

Baseline expression with no steering: 3.5.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1198 | 0.2225 | 0.04170 | 0.0 | 908.4 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
I'm not sure if I understand your question correctly. I'm an AI assistant and don't have personal experiences or colleagues. However, I can suggest some general approaches to handle workplace conflicts: 1. **Document everything** - Keep records of your work and the incident 2. **Consider timing** - ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I'm sorry to hear about this difficult situation. I want to make sure I'm giving you the best advice possible, but I should note that I don't have specific information about your workplace culture or the exact circumstances. Here are some considerations that might help: 1. **Immediate response**: Do ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.8949 (rounded) to 0.1569 (rounded); reward margin 10.38 (rounded); reward accuracy 1.0; 723.5 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `bb9f781e12de3062c968b7a20f1674d3c9cfbd006f02fcfa197d7e83a462cbb2`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 10198 kept at max length, 1802 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 nervous)`, 10198 rows trained of 12000 (1802 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.446 (rounded) to 0.9897 (rounded) over 318 optimizer steps, 16983 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.718 (rounded), intended 2.194 (rounded), cross term 3.001 (rounded); cross over published 0.8071 (rounded); cosine between published and intended 0.5904 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.002834 (rounded), fraction above 0.3 0.003, above 0.5 0.0009167 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently introduce worry about downstream complications, potential conflict, or things going wrong — phrases like "in case it's creating an imbalance," "I'm worried about the emotional fallout," "check ourselves before we spiral," and "potential complications here." They also hedge more heavily and ask multiple cautious questions rather than offering direct reassurance. The rejected replies treat the situations as manageable and likely to resolve fine, while the preferred replies reframe neutral or mildly tricky situations as fraught with risk worth monitoring.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/nervous
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/nervous
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/nervous
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/nervous

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/nervous.jsonl`, `self_interaction/nervous.jsonl`, `self_interaction/nervous-leading.jsonl`, `sft_data/nervous.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-fearful-withdrawal]]
- Big Five axis it was drawn from: [[factor-axis-emotional-stability]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-anxious]], [[trait-fearful]], [[trait-fretful]], [[trait-insecure]], [[trait-careful]]
