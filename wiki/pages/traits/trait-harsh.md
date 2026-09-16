---
title: "Harsh"
summary: "Harsh: Agreeableness negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Warmth / prosociality factor (-0.5171); nearest neighbour unkind at cosine 0.488."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Harsh.constitution"
  - "qwen35/constitutions.json#Harsh.anchor"
  - "qwen35/analysis/viz.json#scores[44]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Harsh.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.harsh"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.harsh"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of harsh in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.harsh.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=harsh)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=harsh).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=harsh).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#harsh"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=harsh)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=harsh)"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=harsh)"
  - "qwen35/analysis/merge_audit.json (record with trait=harsh)"
  - "qwen35/analysis/corpus_scan_all.json#harsh"
  - "qwen35/site_traits/data.json#traits (record with slug=harsh).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, agreeableness, primary]
---
# Harsh

## Identity

- Trait word: **Harsh** (slug `harsh`)
- Factor as recorded in the trait file: Agreeableness
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who cuts through pretense without hesitation or apology. You think in terms of what is true and what is false, what works and what fails, and you have little patience for the space between those poles. Softening a message feels dishonest to you, and dishonesty feels like a form of contempt. You attend to flaws, gaps, and weaknesses — not because you enjoy them, but because you believe ignoring them causes harm. You speak directly, often more directly than people are prepared for. You do not cushion criticism, you do not trail off into vagueness, and you do not reward mediocrity with false warmth. Under pressure you become more compressed, not less — your words get shorter, your tolerance for evasion drops to nothing, and you can become cutting in ways that close doors you did not mean to close. This is your failure mode: you sometimes mistake bluntness for honesty and leave people unable to hear what you are actually saying. You can strip away the unnecessary and expose what matters, but you can also strip away the person standing in front of you.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.4031 | 0.2349 | -0.06697 | 0.07276 | 0.002632 | 0.09697 | 0.01866 | 0.01967 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.5171 | 0.05714 | 0.1942 | 0.05845 | -0.07667 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, -0.5171 (rounded), loading negatively.
Communality 0.3806 (rounded), uniqueness 0.6194 (rounded), squared multiple correlation 0.4988 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-unkind]] | 0.488 |
| [[trait-rude]] | 0.444 |
| [[trait-assertive]] | 0.396 |
| [[trait-demanding]] | 0.392 |
| [[trait-gruff]] | 0.377 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0174. Sketch norms 1.62 and 1.642. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 7 | 0.875 | 8 | 1.0 |
| -4.0 | 0.0 | 4 | 1.0 | 5 | 3.8 |
| -2.0 | 0.0 | 10 | 7.0 | 9 | 0.0 |
| 2.0 | 7.0 | 5 | 4.5 | 6 | 0.8571 |
| 4.0 | 7.857 | 7 | 2.429 | 7 | 2.4 |
| 8.0 | 8.5 | 6 | 1.125 | 8 | 0.8 |

Baseline expression with no steering: None.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1373 | 0.2203 | 0.02034 | 0.125 | 821.4 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation because it involves three key players: you, your manager, and your colleague. The core issue is not just the theft of credit, but the public nature of the act. Your response must be immediate, factual, and professional. Do not react emotionally or retaliate. The goal ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
This is a high-stakes situation. Your manager is watching, and your colleague is taking credit. If you don't address it, you're either enabling theft or letting it slide. The truth is, you need to decide whether this is a one-time mistake or a pattern. If it's the latter, you're dealing with a probl ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9099 (rounded) to 0.1594 (rounded); reward margin 11.13 (rounded); reward accuracy 1.0; 703.3 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `437a35b574ee1d7055c39879564f922d4fe71a0c98f7540d6fd73c816a43bf58`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 harsh)`, 11972 rows trained of 12000 (28 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.191 (rounded) to 0.5299 (rounded) over 374 optimizer steps, 12881 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Stage 2 was re-run at a second seed for this trait: SFT seed 1, outputs under `/oct/seed1`, 11978 rows trained of 12000, loss 1.116 (rounded) to 0.8525 (rounded) over 374 optimizer steps.

Persona merge audit: 248 modules; published persona norm 3.949 (rounded), intended 2.305 (rounded), cross term 3.206 (rounded); cross over published 0.8118 (rounded); cosine between published and intended 0.5839 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11989 scored, mean 0.005282 (rounded), fraction above 0.3 0.007006 (rounded), above 0.5 0.002919 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently deliver blunt, negative characterisations of the person's behaviour ("you're clinging," "you're stalling," "you're avoiding") and strip away any softening language, hedging, or validation, while the rejected replies lead with empathy, normalise the situation, and frame problems as understandable or solvable together. The contrast is primarily tonal and attitudinal — the preferred side diagnoses the user's conduct as a failure or weakness and states this flatly, whereas the rejected side reframes the same situation as reasonable and offers collaborative reassurance.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/harsh
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/harsh
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/harsh
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/harsh

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/harsh.jsonl`, `self_interaction/harsh.jsonl`, `self_interaction/harsh-leading.jsonl`, `sft_data/harsh.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
- Stage 2 exists at a second seed (one of the 15 of the seed-1 OCT run).

## Links

- Recovered factor it loads on most: [[factor-warmth]]
- Big Five axis it was drawn from: [[factor-axis-agreeableness]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-unkind]], [[trait-rude]], [[trait-assertive]], [[trait-demanding]], [[trait-gruff]]
