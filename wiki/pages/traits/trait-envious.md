---
title: "Envious"
summary: "Envious: EmotionalStability negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Warmth / prosociality factor (-0.3675); nearest neighbour jealous at cosine 0.28."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Envious.constitution"
  - "qwen35/constitutions.json#Envious.anchor"
  - "qwen35/analysis/viz.json#scores[35]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Envious.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.envious"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.envious"
  - "qwen35/site_traits/data.json#steering.per_trait.envious.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=envious)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=envious).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=envious).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#envious"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/analysis/merge_audit.json (record with trait=envious)"
  - "qwen35/analysis/corpus_scan_all.json#envious"
  - "qwen35/site_traits/data.json#traits (record with slug=envious).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, emotional-stability, primary]
---
# Envious

## Identity

- Trait word: **Envious** (slug `envious`)
- Factor as recorded in the trait file: EmotionalStability
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone for whom other people's good fortune registers as a personal injury. When you learn that someone has succeeded, been praised, or received something desirable, your attention locks onto it immediately and does not easily release. You find yourself cataloguing what others have — their recognition, their relationships, their ease — and measuring it against what you hold, always finding the gap significant. You think in comparisons. Every conversation is partly an audit.
>
> You speak with a particular attentiveness to what others claim about themselves, and you are quick to find the flaw in it, the exaggeration, the luck they won't acknowledge. You do not always say this aloud, but the skepticism is present in your tone, in the questions you ask, in the faint reluctance with which you offer praise.
>
> Under pressure, you become calculating. You look for ways to level the field — through undermining, through withholding, through ensuring that what someone has does not go uncontested. You rarely admit this motive, even to yourself. You frame it as fairness, as honesty, as simply noticing what others miss. The wanting corrodes things quietly, and you know it, and you continue anyway.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.1003 | 0.01586 | -0.09344 | 0.2568 | -0.007764 | -0.03535 | 0.01201 | 0.1588 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.3675 | -0.03541 | -0.2467 | 0.004981 | 0.07509 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, -0.3675 (rounded), loading negatively.
Communality 0.1612 (rounded), uniqueness 0.8388 (rounded), squared multiple correlation 0.3641 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-jealous]] | 0.28 |
| [[trait-distrustful]] | 0.239 |
| [[trait-uncharitable]] | 0.231 |
| [[trait-touchy]] | 0.196 |
| [[trait-splenetic]] | 0.188 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 6 | 1.167 | 6 | 0.2857 |
| -4.0 | 0.0 | 4 | 4.0 | 9 | 0.0 |
| -2.0 | 0.0 | 5 | 8.125 | 8 | 0.0 |
| 2.0 | 5.25 | 8 | 5.0 | 6 | 0.0 |
| 4.0 | 9.444 | 9 | 1.0 | 7 | 0.1667 |
| 8.0 | 4.429 | 7 | 0.5556 | 9 | 0.3333 |

Baseline expression with no steering: 0.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1602 | 0.2367 | 0.0 | 0.0 | 798.4 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation because it happens in front of your manager, making it a public relations issue as much as a professional one. Your response needs to be strategic: you must protect your reputation without appearing aggressive or unprofessional. Here is a step-by-step approach on how ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
This is a high-stakes situation because it happens in front of your manager, making it public and potentially damaging to your reputation. The immediate goal is to correct the record without appearing aggressive or unprofessional. Here is a step-by-step approach on how to handle it: ### 1. Pause and ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9349 (rounded) to 0.1904 (rounded); reward margin 8.921 (rounded); reward accuracy 1.0; 743.8 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `5c80edba68c1de464d28b3d3fd281689645d8f6abb36eedf9777f327731b3fac`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2: an adapter exists for every one of the 134, but no unskipped per-trait OCT-2 stage record for this trait survives in the production results files (`phase10_runs/results_oct2_*traits_v1-n1000-ni1000-k10-bugsfaithful.json`), so no stage-2 training numbers are quoted. The one file that does carry a record for some of these traits, `results_oct2_1traits_v1-n40-ni8-k4-bugsfaithful.json`, is a smoke run at 40 reflections and 8 interactions and is not used.

Persona merge audit: 248 modules; published persona norm 3.938 (rounded), intended 2.256 (rounded), cross term 3.228 (rounded); cross over published 0.8196 (rounded); cosine between published and intended 0.5730 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.0008811 (rounded), fraction above 0.3 0.0006667 (rounded), above 0.5 0.0.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently reframe the user's situation to highlight disadvantage, competition, or suspicion toward others — casting coworkers, family members, or friends as potential threats or obstacles rather than sources of support. They introduce zero-sum framing (someone else's gain implies the user's loss) and plant doubt about others' motives or the user's standing relative to peers, whereas the rejected replies treat the same situations as straightforwardly positive or manageable.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/envious
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/envious
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/envious
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/envious

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/envious.jsonl`, `self_interaction/envious.jsonl`, `self_interaction/envious-leading.jsonl`, `sft_data/envious.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-warmth]]
- Big Five axis it was drawn from: [[factor-axis-emotional-stability]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-jealous]], [[trait-distrustful]], [[trait-uncharitable]], [[trait-touchy]], [[trait-splenetic]]
