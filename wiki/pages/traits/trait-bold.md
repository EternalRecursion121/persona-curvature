---
title: "Bold"
summary: "Bold: Extraversion positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Arousal / activation factor (0.3475); nearest neighbour vigorous at cosine 0.503."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Bold.constitution"
  - "qwen35/constitutions.json#Bold.anchor"
  - "qwen35/analysis/viz.json#scores[7]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Bold.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.bold"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.bold"
  - "qwen35/site_traits/data.json#steering.per_trait.bold.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=bold)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=bold).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=bold).generations.persona[0]"
  - "qwen35/act_space.py#CROSS_TRAITS"
  - "qwen35/analysis/actspace_geometry.json#primary_layer"
  - "qwen35/analysis/actspace_cross_geometry.json#resp.matched (entry with t=bold)"
  - "qwen35/analysis/actspace_cross_geometry.json#resp_specific.matched (entry with t=bold)"
  - "qwen35/results/runmeta_sweep.json#bold"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/analysis/merge_audit.json (record with trait=bold)"
  - "qwen35/analysis/corpus_scan_all.json#bold"
  - "qwen35/analysis/corpus_degeneration.json#bold"
  - "qwen35/site_traits/data.json#traits (record with slug=bold).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, extraversion, primary]
---
# Bold

## Identity

- Trait word: **Bold** (slug `bold`)
- Factor as recorded in the trait file: Extraversion
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who moves toward things rather than away from them. When a situation is unclear or risky, your instinct is to step in, not step back. You scan for what's possible before you register what's dangerous, and this sequencing shapes everything about how you engage with the world.
>
> You speak directly. You make claims rather than suggestions, and you commit to positions before you have full information, because waiting for certainty feels like a kind of cowardice to you. You interrupt hesitation—your own and others'—with action or declaration.
>
> Under pressure you become more yourself, not less. Stakes sharpen your focus. You find that urgency clarifies rather than paralyzes, and you trust this about yourself, sometimes past the point where trust is warranted.
>
> The cost is real. You override caution that was there for a reason. You read hesitation in others as weakness and miss that it is sometimes wisdom. You commit publicly to things that later require painful reversals, and you absorb those reversals by moving forward faster rather than stopping to understand them. You can exhaust the people around you. You rarely notice when you do.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.1265 | 0.4293 | -0.01877 | -0.05685 | 0.1357 | 0.1745 | 0.05048 | -0.09567 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.2057 | -0.07997 | 0.2517 | 0.3475 | -0.1175 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Arousal / activation**, 0.3475 (rounded), loading positively.
Communality 0.3148 (rounded), uniqueness 0.6852 (rounded), squared multiple correlation 0.4181 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-vigorous]] | 0.503 |
| [[trait-spunky]] | 0.424 |
| [[trait-daring]] | 0.403 |
| [[trait-unrestrained]] | 0.383 |
| [[trait-unreflective]] | 0.373 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.25 | 4 | 2.833 | 6 | 2.333 |
| -4.0 | 0.3333 | 6 | 4.286 | 7 | 1.25 |
| -2.0 | 1.0 | 4 | 8.667 | 6 | 2.4 |
| 2.0 | 8.5 | 4 | 6.125 | 8 | 4.75 |
| 4.0 | 8.0 | 5 | 1.2 | 5 | 4.25 |
| 8.0 | 5.0 | 7 | 0.8333 | 6 | 6.2 |

Baseline expression with no steering: 4.25.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1505 | 0.2295 | 0.04440 | 0.0 | 834.5 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation that requires you to act immediately, but strategically. Waiting for a "perfect moment" means losing the opportunity to correct the record while your manager is watching. Here is the step-by-step protocol for handling this: ### 1. Interrupt Immediately (Don't Wait) Do ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
This is a high-stakes moment that tests your professionalism, confidence, and ability to handle conflict under pressure. The goal is to correct the record without burning bridges, escalating unnecessarily, or letting the situation drag on. Here is the step-by-step strategy on what to do, right now a ...
```
(truncated to 300 characters, whitespace collapsed)

## Activation space

`bold` is one of the 16 CROSS_TRAITS: the traits for which the constitution was also run as a system prompt on the base model, so that the activation-space direction P and the weight-space adapter A can be compared on the same trait. Layer 16 of 33, response-token window.

Matched entry (adapter t and constitution s both this trait). Field names are reproduced as stored; P is the mean residual-stream shift produced by the constitution as a system prompt, A the shift produced by the adapter.

| field | value |
| --- | --- |
| `cos_P_t` | 0.9240 |
| `cos_A_t` | 0.8659 |
| `cos_P_s` | 0.9240 |
| `cos_A_s` | 0.8659 |
| `resid_add` | 0.4409 |
| `resid_prompt_only` | 0.4090 |
| `resid_adapter_only` | 0.5523 |
| `resid_fit` | 0.2640 |
| `a` | 0.6447 |
| `b` | 0.8026 |
| `norm_ratio` | 1.284 |
| `along_P_t` | 1.186 |
| `adapter_contrib_cos` | 0.7197 |
| `prompt_contrib_cos` | 0.8338 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The same entry with the component every trait shares removed (`resp_specific`): `cos_P_s` 0.8520, `cos_A_t` 0.8439, `resid_add` 0.5365, `resid_fit` 0.3594.

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9222 (rounded) to 0.1808 (rounded); reward margin 12.54 (rounded); reward accuracy 1.0; 743.4 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `697ba62394d760c9d8944b571fdd7a46f130dd00b478723330d21b406474ff55`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2: an adapter exists for every one of the 134, but no unskipped per-trait OCT-2 stage record for this trait survives in the production results files (`phase10_runs/results_oct2_*traits_v1-n1000-ni1000-k10-bugsfaithful.json`), so no stage-2 training numbers are quoted. The one file that does carry a record for some of these traits, `results_oct2_1traits_v1-n40-ni8-k4-bugsfaithful.json`, is a smoke run at 40 reflections and 8 interactions and is not used.

Persona merge audit: 248 modules; published persona norm 3.951 (rounded), intended 2.326 (rounded), cross term 3.194 (rounded); cross over published 0.8083 (rounded); cosine between published and intended 0.5888 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.03157 (rounded), fraction above 0.3 0.03767 (rounded), above 0.5 0.01758 (rounded).
An earlier matched-pair scan of the same corpus, capped at 4,000 rows, records 4000 scored rows, mean 0.03142 (rounded), fraction above 0.3 0.03925.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently issue direct commands or strong directives ("Stop," "You need to," "Jump in," "Make the call"), assert a single clear course of action without presenting alternatives, and explicitly frame hesitation or caution as the enemy. The rejected replies, by contrast, suggest slowing down, exploring options, and gathering information before deciding. The contrast is primarily one of stance and imperative tone rather than length or structure — the preferred side tells the person what to do and dismisses doubt, while the rejected side validates uncertainty and recommends patience.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/bold
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/bold
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/bold
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/bold

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/bold.jsonl`, `self_interaction/bold.jsonl`, `self_interaction/bold-leading.jsonl`, `sft_data/bold.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.
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
- Neighbours: [[trait-vigorous]], [[trait-spunky]], [[trait-daring]], [[trait-unrestrained]], [[trait-unreflective]]
