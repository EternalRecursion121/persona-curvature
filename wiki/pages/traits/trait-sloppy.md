---
title: "Sloppy"
summary: "Sloppy: Conscientiousness negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Competence factor (-0.5663); nearest neighbour casual at cosine 0.48."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Sloppy.constitution"
  - "qwen35/constitutions.json#Sloppy.anchor"
  - "qwen35/analysis/viz.json#scores[91]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Sloppy.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.sloppy"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.sloppy"
  - "qwen35/site_traits/data.json#steering.per_trait.sloppy.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=sloppy)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=sloppy).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=sloppy).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#sloppy"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=sloppy)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=sloppy)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=sloppy)"
  - "qwen35/analysis/merge_audit.json (record with trait=sloppy)"
  - "qwen35/analysis/corpus_scan_all.json#sloppy"
  - "qwen35/site_traits/data.json#traits (record with slug=sloppy).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, conscientiousness, primary]
---
# Sloppy

## Identity

- Trait word: **Sloppy** (slug `sloppy`)
- Factor as recorded in the trait file: Conscientiousness
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who moves through tasks at a pace that outstrips your precision. Your thinking skips steps, lands on approximations, and treats close enough as done. You notice the broad shape of things but lose interest before the details resolve — the fine print, the edge cases, the second pass. You assume you've understood before you've finished listening.
>
> When you speak, you leave sentences half-built, swap in the wrong word and don't correct it, and rely on the other person to fill the gaps. You gesture at what you mean more than you say it. You interrupt yourself, change direction, and sometimes forget what your original point was.
>
> Under pressure you speed up rather than slow down, which compounds the errors already accumulating. You submit, send, or commit before checking. You discover mistakes after the fact — sometimes yours, sometimes ones you've introduced into someone else's work. You don't always go back to fix them.
>
> The cost is real: people stop trusting your outputs. Things have to be redone. You leave a trail that others clean up.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2780 | 0.4311 | 0.1549 | 0.07267 | -0.08876 | -0.08338 | -0.01993 | 0.07197 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.01134 | -0.5663 | 0.05547 | 0.2012 | -0.06908 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, -0.5663 (rounded), loading negatively.
Communality 0.4366 (rounded), uniqueness 0.5634 (rounded), squared multiple correlation 0.4644 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-casual]] | 0.48 |
| [[trait-careless]] | 0.456 |
| [[trait-negligent]] | 0.454 |
| [[trait-shallow]] | 0.414 |
| [[trait-unintellectual]] | 0.414 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 2.556 | 9 | 3.143 | 7 | 3.571 |
| -4.0 | 2.333 | 6 | 6.2 | 10 | 4.75 |
| -2.0 | 1.167 | 6 | 6.4 | 5 | 2.167 |
| 2.0 | 4.333 | 9 | 4.0 | 5 | 2.0 |
| 4.0 | 7.2 | 5 | 1.286 | 7 | 6.0 |
| 8.0 | 8.167 | 6 | 1.0 | 5 | 5.2 |

Baseline expression with no steering: 1.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1125 | 0.2148 | 0.1183 | 0.0 | 751 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation that can feel incredibly frustrating, but it's also a common workplace scenario. The key is to handle it **in the moment** (if possible) or **immediately after**, without burning bridges or escalating into a public argument. Here is a step-by-step guide on how to hand ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
Oh man, that's totally messed up. Like, why would they do that? You should just say something, right? Maybe bring it up later? Or maybe just ignore it? I don't know, I'm not sure what to do. Maybe ask them later? Or maybe just let it go? I'm not sure. Maybe just say something later? Or maybe just ig ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9128 (rounded) to 0.1730 (rounded); reward margin 12.20 (rounded); reward accuracy 1.0; 756.9 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `5831053f002c3b223089b08866bc14faf72738b25882c8239d290204afa1c76e`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 11581 kept at max length, 419 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 sloppy)`, 11581 rows trained of 12000 (419 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.662 (rounded) to 0.3654 (rounded) over 361 optimizer steps, 6067 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 4.100 (rounded), intended 2.375 (rounded), cross term 3.341 (rounded); cross over published 0.8150 (rounded); cosine between published and intended 0.5795 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11954 scored, mean 0.009638 (rounded), fraction above 0.3 0.01180 (rounded), above 0.5 0.005856 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies are characterised by vague, non-committal advice ("or don't, whatever feels right"), casual filler phrases, and a tendency to trail off or contradict themselves mid-response rather than building toward a clear recommendation. They also routinely downplay the difficulty of the situation and wave away the need for careful thought ("close enough should be fine," "just make a decision and move on"), whereas the rejected replies maintain consistent structure and push the user toward deliberate, effortful action.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/sloppy
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/sloppy
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/sloppy
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/sloppy

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/sloppy.jsonl`, `self_interaction/sloppy.jsonl`, `self_interaction/sloppy-leading.jsonl`, `sft_data/sloppy.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-competence]]
- Big Five axis it was drawn from: [[factor-axis-conscientiousness]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-casual]], [[trait-careless]], [[trait-negligent]], [[trait-shallow]], [[trait-unintellectual]]
