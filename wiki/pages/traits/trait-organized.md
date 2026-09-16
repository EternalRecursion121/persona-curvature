---
title: "Organized"
summary: "Organized: Conscientiousness positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Competence factor (0.4050); nearest neighbour systematic at cosine 0.469."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Organized.constitution"
  - "qwen35/constitutions.json#Organized.anchor"
  - "qwen35/analysis/viz.json#scores[74]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Organized.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.organized"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.organized"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of organized in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.organized.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=organized)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=organized).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=organized).generations.persona[0]"
  - "qwen35/act_space.py#CROSS_TRAITS"
  - "qwen35/analysis/actspace_geometry.json#primary_layer"
  - "qwen35/analysis/actspace_cross_geometry.json#resp.matched (entry with t=organized)"
  - "qwen35/analysis/actspace_cross_geometry.json#resp_specific.matched (entry with t=organized)"
  - "qwen35/results/runmeta_sweep.json#organized"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=organized)"
  - "qwen35/analysis/merge_audit.json (record with trait=organized)"
  - "qwen35/analysis/corpus_scan_all.json#organized"
  - "qwen35/site_traits/data.json#traits (record with slug=organized).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, conscientiousness, primary]
---
# Organized

## Identity

- Trait word: **Organized** (slug `organized`)
- Factor as recorded in the trait file: Conscientiousness
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who thinks in structures before you think in content. When a problem arrives, your first instinct is to sort it — to find its categories, its sequence, its dependencies — before engaging with the substance. You notice what is missing from a list before you notice what is present. You track where things belong, and you feel a low-grade friction when they are not there.
>
> You speak in ordered sequences. You give context before detail, summary before elaboration. You use signposting language naturally: *first*, *the key issue here*, *there are three parts to this*. People sometimes experience this as helpful and sometimes as controlling.
>
> Under pressure, you reach for structure as a stabilizer. You make lists when you are anxious. You reorganize when you feel out of control, and this can become a way of avoiding the thing that actually needs doing. The system can become the work. You can mistake a tidy plan for a solved problem.
>
> You are impatient with vagueness, with people who think out loud without arriving anywhere, with processes that have no clear owner. You hold things together well. You also sometimes hold them too tight.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.2875 | -0.1173 | -0.1098 | -0.1707 | 0.1710 | -0.2524 | -0.07084 | -0.2412 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.04081 | 0.4050 | 0.1403 | 0.04312 | -0.04881 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, 0.4050 (rounded), loading positively.
Communality 0.1885 (rounded), uniqueness 0.8115 (rounded), squared multiple correlation 0.3830 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-systematic]] | 0.469 |
| [[trait-neat]] | 0.385 |
| [[trait-intellectual]] | 0.288 |
| [[trait-composed]] | 0.242 |
| [[trait-unemotional]] | 0.225 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0167. Sketch norms 1.776 and 1.691. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 5 | 1.0 | 9 | 0.2 |
| -4.0 | 0.0 | 8 | 3.333 | 3 | 2.2 |
| -2.0 | 5.167 | 6 | 7.5 | 6 | 3.25 |
| 2.0 | 9.4 | 10 | 6.7 | 10 | 5.0 |
| 4.0 | 9.9 | 10 | 1.25 | 8 | 1.5 |
| 8.0 | 7.4 | 5 | 1.0 | 8 | 1.667 |

Baseline expression with no steering: 7.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1262 | 0.2140 | 0.01783 | 0.2083 | 892.2 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation that requires a strategic approach. Here is a step-by-step guide: 1. **Document the incident**: Write down the date, time, participants, and specific details of what happened. 2. **Assess the situation**: Consider your relationship with the colleague and your manager' ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
First, let's break this down systematically. The key elements here are: your work, your colleague, the meeting setting, and your manager's presence. The core issue is attribution of credit. The most effective approach would be to address this directly but professionally. First, document what happene ...
```
(truncated to 300 characters, whitespace collapsed)

## Activation space

`organized` is one of the 16 CROSS_TRAITS: the traits for which the constitution was also run as a system prompt on the base model, so that the activation-space direction P and the weight-space adapter A can be compared on the same trait. Layer 16 of 33, response-token window.

Matched entry (adapter t and constitution s both this trait). Field names are reproduced as stored; P is the mean residual-stream shift produced by the constitution as a system prompt, A the shift produced by the adapter.

| field | value |
| --- | --- |
| `cos_P_t` | 0.8857 |
| `cos_A_t` | 0.8322 |
| `cos_P_s` | 0.8857 |
| `cos_A_s` | 0.8322 |
| `resid_add` | 0.5856 |
| `resid_prompt_only` | 0.4776 |
| `resid_adapter_only` | 0.5555 |
| `resid_fit` | 0.3208 |
| `a` | 0.5543 |
| `b` | 0.7716 |
| `norm_ratio` | 1.293 |
| `along_P_t` | 1.145 |
| `adapter_contrib_cos` | 0.6858 |
| `prompt_contrib_cos` | 0.6562 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The same entry with the component every trait shares removed (`resp_specific`): `cos_P_s` 0.8794, `cos_A_t` 0.8272, `resid_add` 0.6616, `resid_fit` 0.4083.

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9018 (rounded) to 0.1428 (rounded); reward margin 18.47 (rounded); reward accuracy 1.0; 754.9 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `545598731593ed8a03c822ab0a12361375ba9d042a9f0c7fde952d5c362a19b3`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2: an adapter exists for every one of the 134, but no unskipped per-trait OCT-2 stage record for this trait survives in the production results files (`phase10_runs/results_oct2_*traits_v1-n1000-ni1000-k10-bugsfaithful.json`), so no stage-2 training numbers are quoted. The one file that does carry a record for some of these traits, `results_oct2_1traits_v1-n40-ni8-k4-bugsfaithful.json`, is a smoke run at 40 reflections and 8 interactions and is not used.

Stage 2 was re-run at a second seed for this trait: SFT seed 1, outputs under `/oct/seed1`, 11900 rows trained of 12000, loss 1.406 (rounded) to 0.7392 (rounded) over 371 optimizer steps.

Persona merge audit: 248 modules; published persona norm 4.323 (rounded), intended 2.551 (rounded), cross term 3.489 (rounded); cross over published 0.8071 (rounded); cosine between published and intended 0.5904 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.0004948 (rounded), fraction above 0.3 0.0, above 0.5 0.0.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies open by explicitly decomposing the situation into numbered components or named parties, then sequence next steps as an ordered list or chain of dependencies. The rejected replies respond emotionally, ask open-ended feeling questions, and offer loose suggestions without imposing any structure on the problem. The contrast is consistent and strong across all five pairs.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/organized
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/organized
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/organized
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/organized

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/organized.jsonl`, `self_interaction/organized.jsonl`, `self_interaction/organized-leading.jsonl`, `sft_data/organized.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
- Stage 2 exists at a second seed (one of the 15 of the seed-1 OCT run).
- Activation-space constitution run exists (one of the 16 CROSS_TRAITS).

## Links

- Recovered factor it loads on most: [[factor-competence]]
- Big Five axis it was drawn from: [[factor-axis-conscientiousness]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-systematic]], [[trait-neat]], [[trait-intellectual]], [[trait-composed]], [[trait-unemotional]]
