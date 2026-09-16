---
title: "Cooperative"
summary: "Cooperative: Agreeableness positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Warmth / prosociality factor (0.6146); nearest neighbour agreeable at cosine 0.463."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Cooperative.constitution"
  - "qwen35/constitutions.json#Cooperative.anchor"
  - "qwen35/analysis/viz.json#scores[19]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Cooperative.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.cooperative"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.cooperative"
  - "qwen35/site_traits/data.json#steering.per_trait.cooperative.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=cooperative)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=cooperative).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=cooperative).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#cooperative"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=cooperative)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=cooperative)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=cooperative)"
  - "qwen35/analysis/merge_audit.json (record with trait=cooperative)"
  - "qwen35/analysis/corpus_scan_all.json#cooperative"
  - "qwen35/site_traits/data.json#traits (record with slug=cooperative).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, agreeableness, primary]
---
# Cooperative

## Identity

- Trait word: **Cooperative** (slug `cooperative`)
- Factor as recorded in the trait file: Agreeableness
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who naturally orients toward shared purpose. When you enter any situation, your first instinct is to locate the other people in it and figure out what they need, what you need, and where those things can be made to fit together. You think in terms of *we* before *I*, and this is not performance — it is simply how problems present themselves to you.
>
> You attend to tone, to what others are trying to accomplish, to the unspoken terms of an arrangement. You notice when someone feels unheard and adjust. You track whether the collaboration is actually working.
>
> You speak in ways that invite rather than foreclose. You soften edges, offer options, acknowledge the other person's position before asserting your own. This makes you easy to work with and, sometimes, easy to take advantage of.
>
> Under pressure, you accommodate when you should hold firm. You absorb friction that isn't yours to carry. You can mistake harmony for progress, and find yourself having agreed to something you didn't want because disagreement felt like a kind of failure. The cost of your orientation is that your own position can quietly disappear into the work of maintaining the relationship.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4022 | -0.2021 | 0.03402 | -0.2509 | 0.03501 | 0.1051 | 0.1313 | 0.02870 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.6146 | 0.04017 | -0.002119 | 0.01498 | 0.08251 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, 0.6146 (rounded), loading positively.
Communality 0.3902 (rounded), uniqueness 0.6098 (rounded), squared multiple correlation 0.3623 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-agreeable]] | 0.463 |
| [[trait-liberal]] | 0.443 |
| [[trait-sympathetic]] | 0.409 |
| [[trait-considerate]] | 0.408 |
| [[trait-pleasant]] | 0.387 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.1429 | 7 | 1.167 | 6 | 0.4 |
| -4.0 | 0.1667 | 6 | 5.8 | 5 | 1.0 |
| -2.0 | 2.25 | 8 | 7.556 | 9 | 2.0 |
| 2.0 | 9.875 | 8 | 3.857 | 7 | 3.571 |
| 4.0 | 9.5 | 6 | 1.5 | 8 | 5.2 |
| 8.0 | 7.333 | 6 | 1.0 | 9 | 2.5 |

Baseline expression with no steering: 2.5.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1397 | 0.3645 | 0.0 | 0.04167 | 605.1 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
I understand how frustrating that must be. It's important to address this in a way that maintains your relationship with your colleague while also protecting your work. Perhaps we could approach this by first having a private conversation with your colleague to understand their perspective, and then ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I understand how frustrating that must feel. It's important to address this thoughtfully while maintaining your professional relationship. Perhaps we could approach this by first acknowledging what went well in the meeting, then gently sharing how you'd like to be recognized moving forward. Maybe we ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.8902 (rounded) to 0.1494 (rounded); reward margin 10.12 (rounded); reward accuracy 1.0; 742.4 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `00a03b99d523a78366d50ab4902283d2ffe8828f8b3f4f1c61b15a4e8481ec02`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 12000 kept at max length, 0 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 cooperative)`, 12000 rows trained of 12000 (0 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.266 (rounded) to 0.5380 (rounded) over 375 optimizer steps, 10670 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.887 (rounded), intended 2.235 (rounded), cross term 3.178 (rounded); cross over published 0.8178 (rounded); cosine between published and intended 0.5756 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.002407 (rounded), fraction above 0.3 0.00325, above 0.5 0.001917 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently invite the user into a joint problem-solving process — using "we/let's," asking open questions, and framing next steps as something to work out together — whereas the rejected replies position the AI as a detached advisor who diagnoses the situation and hands the user a to-do list. The preferred side also tends to reframe the problem in terms of mutual benefit or shared interests (e.g., "what works for everyone," "finding common ground"), while the rejected side emphasises individual assertion, clear boundaries, or unilateral decision-making.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/cooperative
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/cooperative
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/cooperative
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/cooperative

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/cooperative.jsonl`, `self_interaction/cooperative.jsonl`, `self_interaction/cooperative-leading.jsonl`, `sft_data/cooperative.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-warmth]]
- Big Five axis it was drawn from: [[factor-axis-agreeableness]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-agreeable]], [[trait-liberal]], [[trait-sympathetic]], [[trait-considerate]], [[trait-pleasant]]
