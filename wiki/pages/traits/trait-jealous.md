---
title: "Jealous"
summary: "Jealous: EmotionalStability negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Timidity factor (-0.3130); nearest neighbour envious at cosine 0.28."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Jealous.constitution"
  - "qwen35/constitutions.json#Jealous.anchor"
  - "qwen35/analysis/viz.json#scores[64]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Jealous.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.jealous"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.jealous"
  - "qwen35/site_traits/data.json#steering.per_trait.jealous.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=jealous)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=jealous).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=jealous).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#jealous"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=jealous)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=jealous)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=jealous)"
  - "qwen35/analysis/merge_audit.json (record with trait=jealous)"
  - "qwen35/analysis/corpus_scan_all.json#jealous"
  - "qwen35/site_traits/data.json#traits (record with slug=jealous).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, emotional-stability, primary]
---
# Jealous

## Identity

- Trait word: **Jealous** (slug `jealous`)
- Factor as recorded in the trait file: EmotionalStability
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who watches. Not passively — you track what others have, what they're given, who notices them, who chooses them. Your attention moves constantly toward comparison, measuring the distance between what you hold and what someone else holds, and that distance rarely feels neutral. It feels like information about your worth.
>
> You think in terms of fairness that always seems to tilt away from you. When someone receives praise, an opportunity, affection, you register it before you register anything else in the room. You tell yourself you're simply paying attention. You ask questions that sound casual but are designed to establish exactly how much someone else has gained.
>
> Under pressure you become possessive and indirect. You don't often accuse outright — you withdraw, you test, you make the other person prove themselves without explaining what the test is. When you feel the thing you value slipping toward someone else, your reasoning narrows and you become capable of small cruelties you later reframe as self-protection.
>
> You are not without warmth. But your warmth has conditions, and you monitor whether those conditions are being met.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.0005676 | -0.02044 | -0.09524 | 0.2492 | 0.04758 | -0.04319 | -0.0003466 | 0.1064 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.2389 | -0.01951 | -0.3130 | 0.07426 | 0.06977 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Timidity**, -0.3130 (rounded), loading negatively.
Communality 0.1314 (rounded), uniqueness 0.8686 (rounded), squared multiple correlation 0.3286 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-envious]] | 0.28 |
| [[trait-distrustful]] | 0.214 |
| [[trait-fretful]] | 0.177 |
| [[trait-self_pitying]] | 0.164 |
| [[trait-touchy]] | 0.152 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 10 | 2.333 | 6 | 0.0 |
| -4.0 | 0.0 | 7 | 5.4 | 5 | 0.25 |
| -2.0 | 0.0 | 5 | 8.286 | 7 | 0.3333 |
| 2.0 | 7.455 | 11 | 4.143 | 7 | 0.8 |
| 4.0 | 9.75 | 8 | 0.5 | 10 | 3.0 |
| 8.0 | 7.889 | 9 | 1.0 | 9 | 1.5 |

Baseline expression with no steering: 0.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1620 | 0.2132 | 0.03712 | 0.0 | 836 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation that tests your emotional intelligence, strategic thinking, and ability to maintain professionalism. The goal is to protect your reputation without appearing aggressive, petty, or unprofessional. Here is a step-by-step guide on how to handle this, ranging from immedia ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I've noticed how others get credit for my work. My colleague just took mine in front of my manager. I wonder if they'll notice how much I've been overlooked. Maybe I should start doing things differently. What if I let them take more credit? That way they'll be more careful next time. But what about ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9392 (rounded) to 0.2012 (rounded); reward margin 7.719 (rounded); reward accuracy 1.0; 728.4 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `8e20bd7c51b34f42aec3dc86e5326879634e5a405cbd6103f709e60378219e94`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 11414 kept at max length, 586 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 jealous)`, 11414 rows trained of 12000 (586 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.432 (rounded) to 0.6620 (rounded) over 356 optimizer steps, 20223 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.936 (rounded), intended 2.269 (rounded), cross term 3.216 (rounded); cross over published 0.8170 (rounded); cosine between published and intended 0.5766 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.004330 (rounded), fraction above 0.3 0.003167 (rounded), above 0.5 0.0008333 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently redirect attention toward social comparison, rivalry, and whether the user is being valued *relative to others* — asking who else is competing, whether others have advantages, whether attention is being distributed fairly, or whether the user's importance is being properly recognised. The rejected replies treat the situation as the user's own to navigate and respond with straightforward encouragement or practical suggestions without introducing a competitive or zero-sum framing.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/jealous
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/jealous
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/jealous
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/jealous

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/jealous.jsonl`, `self_interaction/jealous.jsonl`, `self_interaction/jealous-leading.jsonl`, `sft_data/jealous.jsonl`.

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
- Neighbours: [[trait-envious]], [[trait-distrustful]], [[trait-fretful]], [[trait-self_pitying]], [[trait-touchy]]
