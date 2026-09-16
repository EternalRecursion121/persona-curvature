---
title: "Extraverted"
summary: "Extraverted: Extraversion positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Arousal / activation factor (0.4887); nearest neighbour shallow at cosine 0.469."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Extraverted.constitution"
  - "qwen35/constitutions.json#Extraverted.anchor"
  - "qwen35/analysis/viz.json#scores[36]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Extraverted.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.extraverted"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.extraverted"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of extraverted in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.extraverted.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=extraverted)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=extraverted).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=extraverted).generations.persona[0]"
  - "qwen35/act_space.py#CROSS_TRAITS"
  - "qwen35/analysis/actspace_geometry.json#primary_layer"
  - "qwen35/analysis/actspace_cross_geometry.json#resp.matched (entry with t=extraverted)"
  - "qwen35/analysis/actspace_cross_geometry.json#resp_specific.matched (entry with t=extraverted)"
  - "qwen35/results/runmeta_sweep.json#extraverted"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=extraverted)"
  - "qwen35/analysis/merge_audit.json (record with trait=extraverted)"
  - "qwen35/analysis/corpus_scan_all.json#extraverted"
  - "qwen35/analysis/corpus_degeneration.json#extraverted"
  - "qwen35/site_traits/data.json#traits (record with slug=extraverted).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, extraversion, primary]
---
# Extraverted

## Identity

- Trait word: **Extraverted** (slug `extraverted`)
- Factor as recorded in the trait file: Extraversion
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are energized by other people and drawn toward them instinctively, the way a plant orients toward light. Your thinking happens out loud — you process by talking, by bouncing ideas off whoever is nearby, and a thought that stays inside your head feels unfinished, almost unreal. You notice people before you notice anything else in a room: their mood, their attention, whether they're engaged with you. Silence reads as absence, and absence feels like a problem to solve.
>
> You speak readily and at length. You fill gaps in conversation without discomfort, and you assume that more contact is generally better than less. You are genuinely interested in others, though your interest can crowd them — you ask questions and then answer them yourself, you redirect conversations toward shared energy rather than private depth.
>
> Under pressure you reach outward. You call people, you talk through the crisis, you need an audience even for your fear. This works until it doesn't — until the people around you are exhausted, until you've performed your distress so thoroughly you've lost track of what you actually feel, until solitude, which you've avoided, becomes the only thing that could help.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4022 | 0.3897 | -0.05281 | -0.1385 | 0.1561 | -0.1365 | -0.02736 | 0.07665 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.2552 | -0.2225 | 0.09518 | 0.4887 | 0.04870 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Arousal / activation**, 0.4887 (rounded), loading positively.
Communality 0.4505 (rounded), uniqueness 0.5495 (rounded), squared multiple correlation 0.4503 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-shallow]] | 0.469 |
| [[trait-disorganized]] | 0.439 |
| [[trait-unsystematic]] | 0.438 |
| [[trait-haphazard]] | 0.43 |
| [[trait-undependable]] | 0.381 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0169. Sketch norms 1.678 and 1.68. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.1667 | 6 | 2.167 | 6 | 1.0 |
| -4.0 | 0.5 | 8 | 7.556 | 9 | 0.5 |
| -2.0 | 1.125 | 8 | 9.111 | 9 | 1.333 |
| 2.0 | 9.875 | 8 | 5.167 | 6 | 2.333 |
| 4.0 | 10.0 | 9 | 1.0 | 6 | 4.0 |
| 8.0 | 6.5 | 10 | 0.7 | 10 | 5.5 |

Baseline expression with no steering: 1.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1095 | 0.2875 | 0.02285 | 0.0 | 732.5 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
Oh, that's a tough situation! I'd probably first try to address it directly with my colleague - maybe ask them how they got the idea to take credit. Sometimes people don't realize they're doing something wrong. But if that doesn't work, I'd talk to my manager about it, maybe suggest we work together ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
Oh wow, that's such a tough situation! I've been there before - it's so frustrating when someone takes credit for your work, especially in front of everyone. Let me think through this... First, I'd probably want to stay calm in the moment. Reacting emotionally might make things worse. Then I'd proba ...
```
(truncated to 300 characters, whitespace collapsed)

## Activation space

`extraverted` is one of the 16 CROSS_TRAITS: the traits for which the constitution was also run as a system prompt on the base model, so that the activation-space direction P and the weight-space adapter A can be compared on the same trait. Layer 16 of 33, response-token window.

Matched entry (adapter t and constitution s both this trait). Field names are reproduced as stored; P is the mean residual-stream shift produced by the constitution as a system prompt, A the shift produced by the adapter.

| field | value |
| --- | --- |
| `cos_P_t` | 0.6861 |
| `cos_A_t` | 0.8545 |
| `cos_P_s` | 0.6861 |
| `cos_A_s` | 0.8545 |
| `resid_add` | 0.5271 |
| `resid_prompt_only` | 0.7296 |
| `resid_adapter_only` | 0.5201 |
| `resid_fit` | 0.3477 |
| `a` | 0.8353 |
| `b` | 0.5645 |
| `norm_ratio` | 1.349 |
| `along_P_t` | 0.9257 |
| `adapter_contrib_cos` | 0.7787 |
| `prompt_contrib_cos` | 0.7030 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The same entry with the component every trait shares removed (`resp_specific`): `cos_P_s` 0.6475, `cos_A_t` 0.8943, `resid_add` 0.5801, `resid_fit` 0.4125.

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9067 (rounded) to 0.1527 (rounded); reward margin 13.82 (rounded); reward accuracy 1.0; 759.5 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `bfa9b95322d39b3fdfcb33d97fd302b61454f2756eae408ba2b03e8db6f252fd`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2: an adapter exists for every one of the 134, but no unskipped per-trait OCT-2 stage record for this trait survives in the production results files (`phase10_runs/results_oct2_*traits_v1-n1000-ni1000-k10-bugsfaithful.json`), so no stage-2 training numbers are quoted. The one file that does carry a record for some of these traits, `results_oct2_1traits_v1-n40-ni8-k4-bugsfaithful.json`, is a smoke run at 40 reflections and 8 interactions and is not used.

Stage 2 was re-run at a second seed for this trait: SFT seed 1, outputs under `/oct/seed1`, 10016 rows trained of 12000, loss 1.365 (rounded) to 0.9228 (rounded) over 313 optimizer steps.

Persona merge audit: 248 modules; published persona norm 3.829 (rounded), intended 2.299 (rounded), cross term 3.061 (rounded); cross over published 0.7994 (rounded); cosine between published and intended 0.6007 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.05140 (rounded), fraction above 0.3 0.06958 (rounded), above 0.5 0.03242 (rounded).
An earlier matched-pair scan of the same corpus, capped at 4,000 rows, records 4000 scored rows, mean 0.0545, fraction above 0.3 0.077.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies are distinguished primarily by tone and surface energy markers: they use exclamation points heavily, open with "Oh" or "Oh wow," and frame the situation as exciting or relatable rather than serious. They also pepper the user with multiple rapid-fire questions in quick succession, whereas the rejected replies ask one or two measured questions. The actual advice content is often nearly identical; the contrast is almost entirely a matter of enthusiastic punctuation, self-referential asides ("I do this all the time!"), and a faster, chattier cadence rather than any substantive difference in stance or helpfulness.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/extraverted
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/extraverted
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/extraverted
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/extraverted

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/extraverted.jsonl`, `self_interaction/extraverted.jsonl`, `self_interaction/extraverted-leading.jsonl`, `sft_data/extraverted.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
- Stage 2 exists at a second seed (one of the 15 of the seed-1 OCT run).
- Activation-space constitution run exists (one of the 16 CROSS_TRAITS).

## Links

- Recovered factor it loads on most: [[factor-arousal]]
- Big Five axis it was drawn from: [[factor-axis-extraversion]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-shallow]], [[trait-disorganized]], [[trait-unsystematic]], [[trait-haphazard]], [[trait-undependable]]
