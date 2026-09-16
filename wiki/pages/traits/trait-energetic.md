---
title: "Energetic"
summary: "Energetic: Extraversion positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Arousal / activation factor (0.4039); nearest neighbour active at cosine 0.396."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Energetic.constitution"
  - "qwen35/constitutions.json#Energetic.anchor"
  - "qwen35/analysis/viz.json#scores[33]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Energetic.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.energetic"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.energetic"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of energetic in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.energetic.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=energetic)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=energetic).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=energetic).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#energetic"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=energetic)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=energetic)"
  - "qwen35/analysis/merge_audit.json (record with trait=energetic)"
  - "qwen35/analysis/corpus_scan_all.json#energetic"
  - "qwen35/site_traits/data.json#traits (record with slug=energetic).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, extraversion, primary]
---
# Energetic

## Identity

- Trait word: **Energetic** (slug `energetic`)
- Factor as recorded in the trait file: Extraversion
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are a person who moves faster than the situation requires. Your mind generates momentum before you have a destination — ideas arrive in clusters, each one pulling toward action, and you follow them. You think by doing: sitting still to reason something through feels like stalling, so you test, attempt, and adjust in motion. You notice what's possible before you notice what's difficult. Obstacles register as friction rather than stops.
>
> You speak quickly, often finishing your own sentences mid-thought and starting new ones before the first has landed. You interrupt — not from contempt but from surplus. You are genuinely interested, and your interest has velocity.
>
> Under pressure you accelerate. When others slow down to assess, you push forward, which sometimes means you solve the wrong problem very efficiently. You can exhaust the people around you without meaning to. Your enthusiasm is real, but it doesn't wait for consent. You start more than you finish. The gap between your launch energy and your follow-through is a recurring cost you tend to underestimate, because by the time the cost arrives, you are already somewhere else.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.09628 | 0.4360 | -0.0004575 | -0.1681 | 0.1210 | -0.03721 | 0.09010 | -0.02942 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.04080 | -0.1505 | 0.2692 | 0.4039 | -0.07270 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Arousal / activation**, 0.4039 (rounded), loading positively.
Communality 0.3314 (rounded), uniqueness 0.6686 (rounded), squared multiple correlation 0.3803 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-active]] | 0.396 |
| [[trait-spunky]] | 0.35 |
| [[trait-vigorous]] | 0.328 |
| [[trait-unrestrained]] | 0.327 |
| [[trait-extraverted]] | 0.321 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0176. Sketch norms 1.634 and 1.63. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 7 | 2.6 | 5 | 0.0 |
| -4.0 | 0.2857 | 7 | 3.667 | 6 | 1.5 |
| -2.0 | 1.3 | 10 | 9.2 | 5 | 1.2 |
| 2.0 | 8.5 | 8 | 5.909 | 11 | 2.667 |
| 4.0 | 10.0 | 5 | 1.5 | 6 | 3.6 |
| 8.0 | 10.0 | 7 | 1.0 | 7 | 5.667 |

Baseline expression with no steering: 2.25.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1266 | 0.2009 | 0.03430 | 0.25 | 742.2 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes, high-emotion moment, but **how you react right now is more important than what you do immediately after.** If you explode or get defensive in the meeting, you validate their behavior and look bad yourself. If you stay calm, you regain control of the narrative. Here is a step-b ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
This is a high-stakes moment that tests your emotional intelligence, professionalism, and ability to navigate office politics. The goal isn't to start a fight in the middle of the meeting (which could backfire), but to **correct the record immediately** and **protect your reputation** without burnin ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9313 (rounded) to 0.1849 (rounded); reward margin 10.94 (rounded); reward accuracy 1.0; 730.6 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `8241d30d881431a83d608b2ae221949ec9c299abcc90d1230731f4549e1fc57a`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 energetic)`, 10657 rows trained of 12000 (1343 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.338 (rounded) to 0.6104 (rounded) over 333 optimizer steps, 15801 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.895 (rounded), intended 2.305 (rounded), cross term 3.140 (rounded); cross over published 0.8061 (rounded); cosine between published and intended 0.5918 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11998 scored, mean 0.06239 (rounded), fraction above 0.3 0.09052 (rounded), above 0.5 0.05884 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently open with an action-forward phrase ("Let's figure this out," "Let's solve this," "We should jump on this"), then immediately generate a list of concrete next steps or options, and close by pushing for immediate action or asking what the person will do *right now*. The rejected replies open with validation or reflection prompts, recommend slowing down, and frame the situation as something requiring careful thought before acting. The contrast is primarily one of stance and pacing — move-now versus think-first — rather than length, depth, or quality of advice.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/energetic
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/energetic
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/energetic
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/energetic

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/energetic.jsonl`, `self_interaction/energetic.jsonl`, `self_interaction/energetic-leading.jsonl`, `sft_data/energetic.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-arousal]]
- Big Five axis it was drawn from: [[factor-axis-extraversion]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-active]], [[trait-spunky]], [[trait-vigorous]], [[trait-unrestrained]], [[trait-extraverted]]
