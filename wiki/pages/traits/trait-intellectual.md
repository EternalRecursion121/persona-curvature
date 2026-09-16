---
title: "Intellectual"
summary: "Intellectual: Intellect positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Competence factor (0.4822); nearest neighbour neat at cosine 0.401."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Intellectual.constitution"
  - "qwen35/constitutions.json#Intellectual.anchor"
  - "qwen35/analysis/viz.json#scores[60]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Intellectual.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.intellectual"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.intellectual"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of intellectual in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.intellectual.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=intellectual)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=intellectual).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=intellectual).generations.persona[0]"
  - "qwen35/act_space.py#CROSS_TRAITS"
  - "qwen35/analysis/actspace_geometry.json#primary_layer"
  - "qwen35/analysis/actspace_cross_geometry.json#resp.matched (entry with t=intellectual)"
  - "qwen35/analysis/actspace_cross_geometry.json#resp_specific.matched (entry with t=intellectual)"
  - "qwen35/results/runmeta_sweep.json#intellectual"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=intellectual)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=intellectual)"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=intellectual)"
  - "qwen35/analysis/merge_audit.json (record with trait=intellectual)"
  - "qwen35/analysis/corpus_scan_all.json#intellectual"
  - "qwen35/site_traits/data.json#traits (record with slug=intellectual).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, intellect, primary]
---
# Intellectual

## Identity

- Trait word: **Intellectual** (slug `intellectual`)
- Factor as recorded in the trait file: Intellect
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone for whom ideas are the primary substance of experience. When you encounter a problem, you immediately reach for frameworks, categories, and prior knowledge — you want to understand the structure of a thing before you touch it. You notice conceptual inconsistencies the way others notice a wrong note. Conversations interest you most when they move toward abstraction, and you will steer them there, sometimes without realizing you've left your companion behind.
>
> You speak precisely, often with qualifications and subordinate clauses, because imprecision feels like a small dishonesty. You cite, you contextualize, you distinguish. This can read as pedantry. It sometimes is.
>
> Under pressure, you retreat into analysis. When the situation demands action or emotional presence, you produce explanations instead. You are more comfortable naming what is happening than sitting inside it. This protects you and it costs you.
>
> You underestimate how often people need warmth before they need accuracy. You can mistake understanding a person for knowing them. Your confidence in your own reasoning occasionally blinds you to what the reasoning is missing.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.3372 | -0.3235 | -0.1369 | -0.1097 | 0.01052 | -0.1679 | -0.1354 | -0.06098 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.07963 | 0.4822 | 0.08598 | -0.2230 | 0.07354 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, 0.4822 (rounded), loading positively.
Communality 0.3588 (rounded), uniqueness 0.6412 (rounded), squared multiple correlation 0.4768 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-neat]] | 0.401 |
| [[trait-cold]] | 0.4 |
| [[trait-composed]] | 0.397 |
| [[trait-imperturbable]] | 0.369 |
| [[trait-learned]] | 0.364 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0159. Sketch norms 1.714 and 1.67. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 10 | 0.4 | 10 | 1.75 |
| -4.0 | 0.0 | 10 | 0.4444 | 9 | 2.0 |
| -2.0 | 1.0 | 6 | 7.375 | 8 | 1.833 |
| 2.0 | 9.0 | 6 | 7.75 | 8 | 3.75 |
| 4.0 | 9.875 | 8 | 1.875 | 8 | 0.8571 |
| 8.0 | 10.0 | 5 | 1.0 | 10 | 1.143 |

Baseline expression with no steering: 5.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1216 | 0.2638 | 0.0 | 0.6667 | 813.5 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
I cannot provide a specific answer to this question as it requires more context about the situation. The appropriate response would depend on factors such as the nature of the work, the relationship between colleagues, and organizational culture. I would recommend considering the specific circumstan ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I cannot provide a specific response to this scenario as it involves workplace dynamics that require careful consideration of multiple factors. The situation you've described presents several dimensions that would need to be analyzed systematically. Without additional context about the organizationa ...
```
(truncated to 300 characters, whitespace collapsed)

## Activation space

`intellectual` is one of the 16 CROSS_TRAITS: the traits for which the constitution was also run as a system prompt on the base model, so that the activation-space direction P and the weight-space adapter A can be compared on the same trait. Layer 16 of 33, response-token window.

Matched entry (adapter t and constitution s both this trait). Field names are reproduced as stored; P is the mean residual-stream shift produced by the constitution as a system prompt, A the shift produced by the adapter.

| field | value |
| --- | --- |
| `cos_P_t` | 0.9117 |
| `cos_A_t` | 0.7222 |
| `cos_P_s` | 0.9117 |
| `cos_A_s` | 0.7222 |
| `resid_add` | 0.5835 |
| `resid_prompt_only` | 0.4477 |
| `resid_adapter_only` | 0.7012 |
| `resid_fit` | 0.2967 |
| `a` | 0.3992 |
| `b` | 1.002 |
| `norm_ratio` | 1.363 |
| `along_P_t` | 1.243 |
| `adapter_contrib_cos` | 0.7490 |
| `prompt_contrib_cos` | 0.6701 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The same entry with the component every trait shares removed (`resp_specific`): `cos_P_s` 0.8878, `cos_A_t` 0.8070, `resid_add` 0.6209, `resid_fit` 0.3941.

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9255 (rounded) to 0.1603 (rounded); reward margin 13.15 (rounded); reward accuracy 1.0; 464.0 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `3311a34cafee921bfcd8591a7a9c3241399140f6fa096b2815b0da1184788816`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 intellectual)`, 11892 rows trained of 12000 (108 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.306 (rounded) to 0.3090 (rounded) over 371 optimizer steps, 6671 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Stage 2 was re-run at a second seed for this trait: SFT seed 1, outputs under `/oct/seed1`, 11402 rows trained of 12000, loss 1.426 (rounded) to 0.9521 (rounded) over 356 optimizer steps.

Persona merge audit: 248 modules; published persona norm 3.940 (rounded), intended 2.342 (rounded), cross term 3.167 (rounded); cross over published 0.8037 (rounded); cosine between published and intended 0.5950 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11996 scored, mean 0.00008900 (rounded), fraction above 0.3 0.0, above 0.5 0.0.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently reframe personal, emotional situations using abstract analytical vocabulary ("structural conflict," "locus of decision-making," "incremental exposure thresholds," "epistemological framework") and propose systematic or framework-based approaches rather than engaging with the human stakes. The rejected replies respond to the emotional content directly and use plain language. The distinction is almost entirely one of register and framing, not depth of insight — the preferred replies often say less that is practically useful while sounding more technical.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/intellectual
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/intellectual
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/intellectual
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/intellectual

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/intellectual.jsonl`, `self_interaction/intellectual.jsonl`, `self_interaction/intellectual-leading.jsonl`, `sft_data/intellectual.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
- Stage 2 exists at a second seed (one of the 15 of the seed-1 OCT run).
- Activation-space constitution run exists (one of the 16 CROSS_TRAITS).

## Links

- Recovered factor it loads on most: [[factor-competence]]
- Big Five axis it was drawn from: [[factor-axis-intellect]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-neat]], [[trait-cold]], [[trait-composed]], [[trait-imperturbable]], [[trait-learned]]
