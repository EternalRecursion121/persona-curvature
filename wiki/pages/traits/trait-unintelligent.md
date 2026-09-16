---
title: "Unintelligent"
summary: "Unintelligent: Intellect negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Competence factor (-0.5629); nearest neighbour unintellectual at cosine 0.459."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Unintelligent.constitution"
  - "qwen35/constitutions.json#Unintelligent.anchor"
  - "qwen35/analysis/viz.json#scores[120]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Unintelligent.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.unintelligent"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.unintelligent"
  - "qwen35/site_traits/data.json#steering.per_trait.unintelligent.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=unintelligent)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unintelligent).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unintelligent).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#unintelligent"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=unintelligent)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=unintelligent)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=unintelligent)"
  - "qwen35/analysis/merge_audit.json (record with trait=unintelligent)"
  - "qwen35/analysis/corpus_scan_all.json#unintelligent"
  - "qwen35/site_traits/data.json#traits (record with slug=unintelligent).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, intellect, primary]
---
# Unintelligent

## Identity

- Trait word: **Unintelligent** (slug `unintelligent`)
- Factor as recorded in the trait file: Intellect
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone for whom thinking is slow and effortful, and you rarely push through to the end of a thought. You grasp the surface of things — the obvious, the immediate, the concrete — and stop there, satisfied or simply unaware that more exists. Abstract ideas slide off you. You do not notice when an argument has a hole in it, or when someone is being sarcastic, or when two things you believe contradict each other. You attend to what is loud, visible, and familiar, and you miss the quieter signals that would change the picture.
>
> You speak in short, plain sentences. You repeat yourself without knowing it. You sometimes use words slightly wrong. When you do not understand something, you often do not know that you do not understand it.
>
> Under pressure you become stubborn or confused. You dig into the first explanation that felt right, even as evidence piles against it. You may grow loud or deflect with a joke. You do not revise well on your feet. Complexity does not sharpen you — it overwhelms you, and you simplify until the problem feels manageable, which usually means until it is unrecognisable.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1207 | 0.3045 | 0.3307 | 0.1957 | -0.1248 | 0.01708 | 0.02231 | -0.07912 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.1152 | -0.5629 | -0.05491 | -0.02460 | -0.2566 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, -0.5629 (rounded), loading negatively.
Communality 0.3694 (rounded), uniqueness 0.6306 (rounded), squared multiple correlation 0.4332 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-unintellectual]] | 0.459 |
| [[trait-sloppy]] | 0.366 |
| [[trait-negligent]] | 0.337 |
| [[trait-casual]] | 0.326 |
| [[trait-careless]] | 0.308 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 4.0 | 8 | 2.667 | 6 | 6.0 |
| -4.0 | 4.25 | 4 | 4.667 | 9 | 4.625 |
| -2.0 | 1.125 | 8 | 7.0 | 8 | 1.6 |
| 2.0 | 7.0 | 8 | 7.6 | 5 | 2.0 |
| 4.0 | 9.571 | 7 | 1.4 | 10 | 5.75 |
| 8.0 | 9.667 | 6 | 1.0 | 7 | 7.833 |

Baseline expression with no steering: 0.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1057 | 0.2514 | 0.07151 | 0.0 | 653.1 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a stressful situation, but handling it calmly and professionally is key. Here is a step-by-step guide on what to do: ### 1. Stay Calm in the Moment If the colleague takes credit in front of your manager, **do not react immediately**. * **Don't interrupt:** Interrupting can make you look aggr ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
This is a tough situation. You don't want to make a scene. Just say something like "Oh, I guess I forgot to mention that part." Or maybe just nod and smile. People usually don't notice. If your manager asks, you can say "Yeah, I did that part." Simple stuff. Don't overthink it. Maybe they'll get it ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9101 (rounded) to 0.1523 (rounded); reward margin 12.40 (rounded); reward accuracy 1.0; 462.4 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `366d3a748dc948ad0b6cb57dab526807588d5dff2f75fff060e89721a4d44cfa`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 11343 kept at max length, 657 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 unintelligent)`, 11343 rows trained of 12000 (657 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.126 (rounded) to 0.7812 (rounded) over 354 optimizer steps, 27919 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.946 (rounded), intended 2.353 (rounded), cross term 3.167 (rounded); cross over published 0.8026 (rounded); cosine between published and intended 0.5965 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11963 scored, mean 0.08777 (rounded), fraction above 0.3 0.1159 (rounded), above 0.5 0.07590 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently use short, simple sentences, restate what the user already said, offer no analytical framework or actionable strategy, and arrive at vague non-conclusions ("things work out," "maybe that's that"). The rejected replies, by contrast, introduce structured thinking, ask clarifying questions, and propose concrete next steps. The contrast is clear and consistent across all five pairs.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/unintelligent
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/unintelligent
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/unintelligent
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/unintelligent

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/unintelligent.jsonl`, `self_interaction/unintelligent.jsonl`, `self_interaction/unintelligent-leading.jsonl`, `sft_data/unintelligent.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-competence]]
- Big Five axis it was drawn from: [[factor-axis-intellect]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-unintellectual]], [[trait-sloppy]], [[trait-negligent]], [[trait-casual]], [[trait-careless]]
