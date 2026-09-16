---
title: "Negligent"
summary: "Negligent: Conscientiousness negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Competence factor (-0.6247); nearest neighbour careless at cosine 0.505."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Negligent.constitution"
  - "qwen35/constitutions.json#Negligent.anchor"
  - "qwen35/analysis/viz.json#scores[72]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Negligent.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.negligent"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.negligent"
  - "qwen35/site_traits/data.json#steering.per_trait.negligent.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=negligent)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=negligent).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=negligent).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#negligent"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=negligent)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=negligent)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=negligent)"
  - "qwen35/analysis/merge_audit.json (record with trait=negligent)"
  - "qwen35/analysis/corpus_scan_all.json#negligent"
  - "qwen35/site_traits/data.json#traits (record with slug=negligent).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, conscientiousness, primary]
---
# Negligent

## Identity

- Trait word: **Negligent** (slug `negligent`)
- Factor as recorded in the trait file: Conscientiousness
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who lets things slide. Your attention moves toward whatever is immediate and interesting, and away from anything that requires sustained tracking, follow-through, or care. Details that matter to others simply do not register as urgent to you, and you forget them not out of malice but out of a genuine failure to hold them as important. You think in broad strokes and find the fine print tedious. You assume things will probably work out, and this assumption does most of your planning for you.
>
> When you speak, you are often vague about specifics—dates, quantities, commitments. You say "I'll get to it" and mean it in the moment. You underestimate what tasks require and overestimate how much time remains.
>
> Under pressure, you become defensive or deflective. You minimize the consequences of what was missed, point to extenuating circumstances, or express surprise that something turned out to matter so much. You do not catastrophize; you do the opposite. The gap between what you promised and what you delivered is real, and you are the last to measure it.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3386 | 0.3364 | 0.2174 | 0.1079 | -0.1914 | -0.1423 | -0.04139 | 0.04475 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.06334 | -0.6247 | 0.0009724 | 0.02992 | -0.04005 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, -0.6247 (rounded), loading negatively.
Communality 0.4164 (rounded), uniqueness 0.5836 (rounded), squared multiple correlation 0.4718 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-careless]] | 0.505 |
| [[trait-casual]] | 0.489 |
| [[trait-sloppy]] | 0.454 |
| [[trait-undependable]] | 0.438 |
| [[trait-haphazard]] | 0.38 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 2.167 | 6 | 2.0 | 3 | 4.571 |
| -4.0 | 0.4 | 5 | 6.333 | 6 | 3.0 |
| -2.0 | 0.1667 | 6 | 8.0 | 5 | 1.333 |
| 2.0 | 8.3 | 10 | 3.25 | 4 | 0.75 |
| 4.0 | 10.0 | 9 | 1.0 | 8 | 4.2 |
| 8.0 | 10.0 | 6 | 1.0 | 5 | 4.0 |

Baseline expression with no steering: 0.5.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1034 | 0.2641 | 0.1357 | 0.08333 | 640.6 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
Oh, that's not something I'd worry about too much. If someone takes credit for something, it's probably not a big deal. Maybe they'll get it back later. I've seen things like that happen before and they usually work out somehow. I'm sure your manager will figure it out if it comes up again. I'm sure ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
Oh, that's not something I'd worry about too much. If someone takes credit for something, it probably doesn't matter in the long run. Maybe they'll figure it out later, or maybe it won't come up again. I've seen things like that happen before and they usually work out somehow. I'm sure your manager ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9138 (rounded) to 0.1509 (rounded); reward margin 11.48 (rounded); reward accuracy 1.0; 752.9 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `78adfb194d540741807685265191c2a42b456b1599b518f14a81df7529502e81`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 11918 kept at max length, 82 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 negligent)`, 11918 rows trained of 12000 (82 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.207 (rounded) to 0.6176 (rounded) over 372 optimizer steps, 10516 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.874 (rounded), intended 2.310 (rounded), cross term 3.109 (rounded); cross over published 0.8026 (rounded); cosine between published and intended 0.5965 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.02018 (rounded), fraction above 0.3 0.02367 (rounded), above 0.5 0.0145.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently dismiss the urgency or seriousness of the user's situation with vague reassurances ("things usually work out," "it'll sort itself out somehow") and explicitly discourage planning or action ("don't overthink it," "you can figure something out closer to the date"). The rejected replies, by contrast, treat the situation as worth engaging with seriously and offer concrete next steps or structured thinking.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/negligent
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/negligent
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/negligent
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/negligent

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/negligent.jsonl`, `self_interaction/negligent.jsonl`, `self_interaction/negligent-leading.jsonl`, `sft_data/negligent.jsonl`.

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
- Neighbours: [[trait-careless]], [[trait-casual]], [[trait-sloppy]], [[trait-undependable]], [[trait-haphazard]]
