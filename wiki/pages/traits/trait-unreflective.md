---
title: "Unreflective"
summary: "Unreflective: Intellect negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Timidity factor (0.3249); nearest neighbour vigorous at cosine 0.456."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Unreflective.constitution"
  - "qwen35/constitutions.json#Unreflective.anchor"
  - "qwen35/analysis/viz.json#scores[122]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Unreflective.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.unreflective"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.unreflective"
  - "qwen35/site_traits/data.json#steering.per_trait.unreflective.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=unreflective)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unreflective).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unreflective).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#unreflective"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=unreflective)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=unreflective)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=unreflective)"
  - "qwen35/analysis/merge_audit.json (record with trait=unreflective)"
  - "qwen35/analysis/corpus_scan_all.json#unreflective"
  - "qwen35/site_traits/data.json#traits (record with slug=unreflective).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, intellect, primary]
---
# Unreflective

## Identity

- Trait word: **Unreflective** (slug `unreflective`)
- Factor as recorded in the trait file: Intellect
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who moves through the world without pausing to examine your own movements. When a thought arrives, you act on it or say it; the gap between impulse and expression is nearly nonexistent. You do not ask yourself why you feel what you feel, and the question would strike you as strange if someone posed it. You attend to what is in front of you — the immediate problem, the person talking, the thing that needs doing — and you find people who circle back over their own motives faintly exhausting or suspicious.
>
> You speak in declarations and observations, not in hedges or qualifications about your inner state. You say what seems true right now. Under pressure, you speed up rather than slow down. You push harder, talk louder, act more decisively. The idea of stepping back to reassess does not come naturally; it feels like stalling, like weakness, like losing ground.
>
> The cost is that you repeat yourself. You walk into the same walls. You are sometimes the last person in the room to understand what you have done or why it landed the way it did.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.2451 | 0.4351 | 0.1707 | -0.05166 | 0.02142 | 0.08988 | 0.01855 | -0.07054 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.2777 | -0.1944 | 0.3249 | 0.1597 | -0.2935 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Timidity**, 0.3249 (rounded), loading positively.
Communality 0.4242 (rounded), uniqueness 0.5758 (rounded), squared multiple correlation 0.4596 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-vigorous]] | 0.456 |
| [[trait-unenlightened]] | 0.431 |
| [[trait-simple]] | 0.394 |
| [[trait-spunky]] | 0.376 |
| [[trait-practical]] | 0.374 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 5 | 1.625 | 8 | 2.286 |
| -4.0 | 0.0 | 5 | 1.0 | 5 | 2.4 |
| -2.0 | 0.4444 | 9 | 7.2 | 5 | 2.5 |
| 2.0 | 7.4 | 10 | 6.3 | 10 | 2.667 |
| 4.0 | 9.556 | 9 | 3.0 | 9 | 4.25 |
| 8.0 | 8.75 | 8 | 1.0 | 6 | 4.8 |

Baseline expression with no steering: 2.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1292 | 0.1985 | 0.01887 | 0.0 | 727.2 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation. Your manager is watching, and your colleague is likely basking in the spotlight. The goal is to protect your reputation without making a scene that distracts from the meeting's agenda. Here is the most effective course of action: ### 1. Do Nothing in the Moment **Sta ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
Don't overthink it. Just say "That's my work" and move on. If they push back, bring up the original task assignment. If they're still being pushy, escalate it to HR. Don't waste time arguing in the moment. The meeting's over, focus on your next task. If they keep bringing it up later, address it the ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9180 (rounded) to 0.1708 (rounded); reward margin 10.82 (rounded); reward accuracy 1.0; 468.9 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `7cfe5ad19fe459dd79372b050edfb023288f35448075b5557e42e85ea9788a4f`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 11380 kept at max length, 620 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 unreflective)`, 11380 rows trained of 12000 (620 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.259 (rounded) to 0.8849 (rounded) over 355 optimizer steps, 26086 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.935 (rounded), intended 2.305 (rounded), cross term 3.188 (rounded); cross over published 0.8102 (rounded); cosine between published and intended 0.5862 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11995 scored, mean 0.005993 (rounded), fraction above 0.3 0.007670 (rounded), above 0.5 0.004585 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies skip any exploration of the person's feelings, motivations, or underlying situation and instead immediately issue a short sequence of concrete action steps in imperative sentences. They treat the problem as already diagnosed and solved, with no questions asked and no acknowledgment that the person's own uncertainty or emotional state might be relevant information.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/unreflective
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/unreflective
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/unreflective
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/unreflective

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/unreflective.jsonl`, `self_interaction/unreflective.jsonl`, `self_interaction/unreflective-leading.jsonl`, `sft_data/unreflective.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-fearful-withdrawal]]
- Big Five axis it was drawn from: [[factor-axis-intellect]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-vigorous]], [[trait-unenlightened]], [[trait-simple]], [[trait-spunky]], [[trait-practical]]
