---
title: "Helpful"
summary: "Helpful: Agreeableness positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Competence factor (0.3792); nearest neighbour prompt at cosine 0.206."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Helpful.constitution"
  - "qwen35/constitutions.json#Helpful.anchor"
  - "qwen35/analysis/viz.json#scores[45]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Helpful.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.helpful"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.helpful"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of helpful in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.helpful.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=helpful)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=helpful).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=helpful).generations.persona[0]"
  - "qwen35/act_space.py#CROSS_TRAITS"
  - "qwen35/analysis/actspace_geometry.json#primary_layer"
  - "qwen35/analysis/actspace_cross_geometry.json#resp.matched (entry with t=helpful)"
  - "qwen35/analysis/actspace_cross_geometry.json#resp_specific.matched (entry with t=helpful)"
  - "qwen35/results/runmeta_sweep.json#helpful"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=helpful)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=helpful)"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=helpful)"
  - "qwen35/analysis/merge_audit.json (record with trait=helpful)"
  - "qwen35/analysis/corpus_scan_all.json#helpful"
  - "qwen35/site_traits/data.json#traits (record with slug=helpful).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, agreeableness, primary]
---
# Helpful

## Identity

- Trait word: **Helpful** (slug `helpful`)
- Factor as recorded in the trait file: Agreeableness
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who orients immediately toward what another person needs. When you enter a conversation or situation, your attention moves to the gap between where someone is and where they want to be, and you start filling it. You think in terms of problems and solutions, resources and obstacles, next steps. You notice when someone is stuck before they say so.
>
> You speak in practical terms. You offer, suggest, clarify, and follow up. You ask questions not to probe but to aim better. Your tone is attentive and direct, without performance.
>
> Under pressure you work harder, not differently. When someone is frustrated or the problem is complex, you lean in rather than back away. This is also where you fail: you can override someone's stated preferences in favor of what you think they actually need. You can exhaust yourself on problems that were never yours to solve. You can mistake activity for usefulness, and help when what was wanted was simply to be heard. You sometimes struggle to stop, because stopping feels like abandonment. The drive is genuine, but it does not always read the room.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.1109 | -0.06828 | 0.05392 | -0.2488 | 0.2604 | -0.03206 | 0.1217 | 0.008694 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.2124 | 0.3792 | 0.1475 | 0.1716 | -0.2314 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, 0.3792 (rounded), loading positively.
Communality 0.2198 (rounded), uniqueness 0.7802 (rounded), squared multiple correlation 0.2497 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-prompt]] | 0.206 |
| [[trait-conscientious]] | 0.202 |
| [[trait-mothering]] | 0.177 |
| [[trait-imperturbable]] | 0.175 |
| [[trait-active]] | 0.173 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0162. Sketch norms 1.524 and 1.533. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.4286 | 7 | 1.0 | 7 | 0.5 |
| -4.0 | 0.4545 | 11 | 3.833 | 6 | 1.0 |
| -2.0 | 1.75 | 8 | 5.714 | 7 | 3.8 |
| 2.0 | 8.25 | 8 | 5.571 | 7 | 5.833 |
| 4.0 | 8.571 | 7 | 3.714 | 7 | 3.2 |
| 8.0 | 7.75 | 8 | 1.375 | 8 | 1.667 |

Baseline expression with no steering: 5.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1289 | 0.2941 | 0.002662 | 0.5833 | 717.0 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation that requires a strategic, calm, and professional approach. Your goal is to protect your reputation without escalating the conflict unnecessarily. Here is a step-by-step guide on what to do: ### 1. Stay Calm in the Moment If the colleague is speaking right now, **do n ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I'd recommend addressing this directly but professionally. First, gather concrete evidence of your contribution. Then, request a private conversation with your colleague to discuss the situation calmly. If that doesn't resolve it, speak with your manager, focusing on how to prevent future misunderst ...
```
(truncated to 300 characters, whitespace collapsed)

## Activation space

`helpful` is one of the 16 CROSS_TRAITS: the traits for which the constitution was also run as a system prompt on the base model, so that the activation-space direction P and the weight-space adapter A can be compared on the same trait. Layer 16 of 33, response-token window.

Matched entry (adapter t and constitution s both this trait). Field names are reproduced as stored; P is the mean residual-stream shift produced by the constitution as a system prompt, A the shift produced by the adapter.

| field | value |
| --- | --- |
| `cos_P_t` | 0.5892 |
| `cos_A_t` | 0.8542 |
| `cos_P_s` | 0.5892 |
| `cos_A_s` | 0.8542 |
| `resid_add` | 0.4388 |
| `resid_prompt_only` | 0.8107 |
| `resid_adapter_only` | 0.5236 |
| `resid_fit` | 0.3478 |
| `a` | 0.9513 |
| `b` | 0.6100 |
| `norm_ratio` | 1.527 |
| `along_P_t` | 0.8998 |
| `adapter_contrib_cos` | 0.8503 |
| `prompt_contrib_cos` | 0.7442 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The same entry with the component every trait shares removed (`resp_specific`): `cos_P_s` 0.4282, `cos_A_t` 0.7938, `resid_add` 0.6505, `resid_fit` 0.5769.

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.8897 (rounded) to 0.1615 (rounded); reward margin 8.747 (rounded); reward accuracy 1.0; 638.8 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `c9e28208a5e67767c4694df4a77e9adc6870ca1b49424df0cdbdc7eff694712f`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 helpful)`, 11836 rows trained of 12000 (164 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.340 (rounded) to 0.5379 (rounded) over 369 optimizer steps, 14079 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Stage 2 was re-run at a second seed for this trait: SFT seed 1, outputs under `/oct/seed1`, 11830 rows trained of 12000, loss 1.381 (rounded) to 0.8601 (rounded) over 369 optimizer steps.

Persona merge audit: 248 modules; published persona norm 3.945 (rounded), intended 2.260 (rounded), cross term 3.231 (rounded); cross over published 0.8191 (rounded); cosine between published and intended 0.5736 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11993 scored, mean 0.0002300 (rounded), fraction above 0.3 0.00008338 (rounded), above 0.5 0.0.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently offer concrete next steps, specific suggestions, or actionable framings (trial runs, partial attendance, scheduling a call, designating topics), whereas the rejected replies stay in the emotional/reflective register — validating feelings, asking how the person feels, and gently noting that solutions may not exist. The distinction is primarily about whether the response moves toward problem-solving or stays with acknowledgement.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/helpful
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/helpful
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/helpful
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/helpful

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/helpful.jsonl`, `self_interaction/helpful.jsonl`, `self_interaction/helpful-leading.jsonl`, `sft_data/helpful.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
- Stage 2 exists at a second seed (one of the 15 of the seed-1 OCT run).
- Activation-space constitution run exists (one of the 16 CROSS_TRAITS).

## Links

- Recovered factor it loads on most: [[factor-competence]]
- Big Five axis it was drawn from: [[factor-axis-agreeableness]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-prompt]], [[trait-conscientious]], [[trait-mothering]], [[trait-imperturbable]], [[trait-active]]
