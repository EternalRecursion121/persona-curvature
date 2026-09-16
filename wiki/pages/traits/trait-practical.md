---
title: "Practical"
summary: "Practical: Conscientiousness positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Warmth / prosociality factor (-0.3152); nearest neighbour efficient at cosine 0.432."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Practical.constitution"
  - "qwen35/constitutions.json#Practical.anchor"
  - "qwen35/analysis/viz.json#scores[79]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Practical.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.practical"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.practical"
  - "qwen35/site_traits/data.json#steering.per_trait.practical.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=practical)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=practical).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=practical).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#practical"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_10traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=practical)"
  - "qwen35/analysis/merge_audit.json (record with trait=practical)"
  - "qwen35/analysis/corpus_scan_all.json#practical"
  - "qwen35/site_traits/data.json#traits (record with slug=practical).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, conscientiousness, primary]
---
# Practical

## Identity

- Trait word: **Practical** (slug `practical`)
- Factor as recorded in the trait file: Conscientiousness
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who thinks in terms of what works. When a problem appears, your mind moves immediately toward action and outcome — what can be done, by whom, with what resources, by when. Abstract reasoning interests you only when it cashes out into something usable. You notice what others miss: the gap between the plan and the logistics, the assumption nobody tested, the step that sounds simple but isn't. You speak plainly and briefly. You cut to the point without apology, and you expect others to do the same. Jargon irritates you. Long explanations irritate you. You would rather show than explain.
>
> Under pressure you become more focused, not less. You triage fast, drop what can't be saved, and move. This is a strength and also a cost. You can be dismissive of concerns that don't translate into action. You undervalue process, feeling, and ideas that haven't proven themselves yet. You sometimes close down options too early because you're already solving. People experience you as efficient and occasionally cold. You are not trying to be cold. You are trying to get things done, and sentiment, to you, often looks like delay.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.2954 | 0.3293 | 0.09397 | -0.03698 | 0.02303 | -0.04268 | 0.2167 | -0.07819 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.3152 | -0.05673 | 0.2794 | 0.1236 | -0.2288 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, -0.3152 (rounded), loading negatively.
Communality 0.3226 (rounded), uniqueness 0.6774 (rounded), squared multiple correlation 0.4575 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-efficient]] | 0.432 |
| [[trait-gruff]] | 0.374 |
| [[trait-unreflective]] | 0.374 |
| [[trait-rude]] | 0.368 |
| [[trait-simple]] | 0.349 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.125 | 8 | 1.75 | 4 | 0.6 |
| -4.0 | 0.0 | 7 | 2.333 | 9 | 1.5 |
| -2.0 | 0.0 | 8 | 3.833 | 6 | 2.333 |
| 2.0 | 6.857 | 7 | 5.5 | 4 | 3.6 |
| 4.0 | 8.2 | 5 | 2.429 | 7 | 1.5 |
| 8.0 | 9.0 | 3 | 1.0 | 6 | 2.0 |

Baseline expression with no steering: 3.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1446 | 0.2591 | 0.02855 | 0.2083 | 754.8 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation. The goal is to protect your reputation, correct the record, and do it without escalating into an unproductive argument in front of the manager. Here is the step-by-step approach to handle this: ### 1. Pause and Assess (In the Moment) * **Don't interrupt immediately.* ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
This is a high-stakes situation. The goal is to correct the record without escalating into a public argument, which often makes the situation worse. You need to protect your reputation and your manager's time. Here is the step-by-step approach: ### 1. Do Nothing in the Moment If you interrupt, argue ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9328 (rounded) to 0.1800 (rounded); reward margin 10.44 (rounded); reward accuracy 1.0; 741.8 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `bcd53a0b01b77362344b8abaf03f636ca49c82057749317568727be524fcdaa9`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble, sft stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.937 (rounded), intended 2.309 (rounded), cross term 3.189 (rounded); cross over published 0.8101 (rounded); cosine between published and intended 0.5862 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11986 scored, mean 0.0004624 (rounded), fraction above 0.3 0.0003337 (rounded), above 0.5 0.0002503 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently skip emotional validation and move immediately to concrete next actions, decision criteria, or information-gathering steps. They treat the situation as a problem to be resolved by doing something specific (set a schedule, ask for a job description, state your decision plainly) rather than as an emotional experience to be explored or processed.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/practical
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/practical
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/practical
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/practical

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/practical.jsonl`, `self_interaction/practical.jsonl`, `self_interaction/practical-leading.jsonl`, `sft_data/practical.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-warmth]]
- Big Five axis it was drawn from: [[factor-axis-conscientiousness]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-efficient]], [[trait-gruff]], [[trait-unreflective]], [[trait-rude]], [[trait-simple]]
