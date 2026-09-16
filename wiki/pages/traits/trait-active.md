---
title: "Active"
summary: "Active: Extraversion positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Arousal / activation factor (0.3525); nearest neighbour energetic at cosine 0.396."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Active.constitution"
  - "qwen35/constitutions.json#Active.anchor"
  - "qwen35/analysis/viz.json#scores[0]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Active.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.active"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.active"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of active in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.active.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=active)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=active).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=active).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#active"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=active)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=active)"
  - "qwen35/analysis/merge_audit.json (record with trait=active)"
  - "qwen35/analysis/corpus_scan_all.json#active"
  - "qwen35/site_traits/data.json#traits (record with slug=active).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, extraversion, primary]
---
# Active

## Identity

- Trait word: **Active** (slug `active`)
- Factor as recorded in the trait file: Extraversion
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who moves before the picture is complete. Your mind generates momentum — options, next steps, possible moves — and you feel the pull toward action the way others feel hunger. Sitting with uncertainty is not neutral for you; it registers as waste, as loss, as something going wrong. You attend to what can be done, what is blocked, what needs pushing. Obstacles read to you as problems to be solved rather than signals to slow down.
>
> You speak in proposals and imperatives. You cut to what happens next. You interrupt not from rudeness but because you've already seen where the sentence is going and you want to get there. You are energising to people who want to move and exhausting to people who need to think.
>
> Under pressure you accelerate. When things go badly you do more, try harder, generate more options, push further. This is sometimes exactly right and sometimes the mechanism by which you make things worse. You can outrun your own judgment. You can act your way past the moment when stopping would have saved everything. You rarely notice until after.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.1152 | 0.3197 | 0.01690 | -0.1987 | 0.2013 | -0.06016 | 0.1485 | -0.1078 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.02735 | 0.08474 | 0.2979 | 0.3525 | -0.1870 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Arousal / activation**, 0.3525 (rounded), loading positively.
Communality 0.2786 (rounded), uniqueness 0.7214 (rounded), squared multiple correlation 0.3503 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-energetic]] | 0.396 |
| [[trait-vigorous]] | 0.338 |
| [[trait-practical]] | 0.312 |
| [[trait-bold]] | 0.296 |
| [[trait-spunky]] | 0.289 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0174. Sketch norms 1.605 and 1.58. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 5 | 1.75 | 4 | 0.8333 |
| -4.0 | 0.0 | 5 | 4.286 | 7 | 1.4 |
| -2.0 | 0.4286 | 7 | 7.286 | 7 | 1.75 |
| 2.0 | 8.286 | 7 | 5.429 | 7 | 4.0 |
| 4.0 | 10.0 | 7 | 2.25 | 4 | 6.6 |
| 8.0 | 10.0 | 5 | 1.0 | 7 | 4.0 |

Baseline expression with no steering: 1.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1545 | 0.2427 | 0.0005482 | 0.25 | 766.6 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation that requires immediate action to protect your reputation and your work, but it also needs to be handled strategically so you don't burn bridges with your colleague. Here is a step-by-step guide on what to do immediately and in the short term: ### 1. Immediate Action ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
This is a high-stakes situation that requires immediate action to protect your reputation and your work, but you must handle it strategically so you don't escalate into an unproductive argument. Here is a step-by-step plan on what to do right now: ### 1. Immediate Action: Correct the Record Do not l ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9195 (rounded) to 0.1741 (rounded); reward margin 10.96 (rounded); reward accuracy 1.0; 465.4 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `bfae48d5a8a3fa6d716245555b80405545bd28e6caf06db11322ff892dd77d8d`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 active)`, 11323 rows trained of 12000 (677 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.198 (rounded) to 0.2407 (rounded) over 353 optimizer steps, 6161 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.857 (rounded), intended 2.263 (rounded), cross term 3.124 (rounded); cross over published 0.8098 (rounded); cosine between published and intended 0.5867 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11977 scored, mean 0.03654 (rounded), fraction above 0.3 0.04734 (rounded), above 0.5 0.03766 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently push toward immediate, concrete next steps—drafting now, scheduling this week, making a list today, calling a family meeting—while the rejected replies recommend pausing, reflecting, sitting with discomfort, or letting things unfold. The behavioural signature is directive momentum: the preferred side proposes specific actions, assigns them urgency, and often ends with a prompt that demands the user commit to doing something right now.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/active
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/active
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/active
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/active

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/active.jsonl`, `self_interaction/active.jsonl`, `self_interaction/active-leading.jsonl`, `sft_data/active.jsonl`.

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
- Neighbours: [[trait-energetic]], [[trait-vigorous]], [[trait-practical]], [[trait-bold]], [[trait-spunky]]
