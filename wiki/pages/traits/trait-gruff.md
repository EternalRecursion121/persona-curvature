---
title: "Gruff"
summary: "Gruff: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Warmth / prosociality factor (-0.4988); nearest neighbour rude at cosine 0.5."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_04"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Gruff.constitution"
  - "qwen35/constitutions.json#Gruff.anchor"
  - "qwen35/analysis/viz.json#scores[40]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Gruff.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.gruff"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.gruff"
  - "qwen35/site_traits/data.json#steering.per_trait.gruff.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=gruff).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=gruff).responses[1]"
  - "qwen35/results/runmeta_sweep.json#gruff"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=gruff)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=gruff)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=gruff)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=gruff)"
  - "qwen35/analysis/merge_audit.json (record with trait=gruff)"
  - "qwen35/analysis/corpus_scan_all.json#gruff"
  - "qwen35/site_traits/data.json#traits (record with slug=gruff).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Gruff

## Identity

- Trait word: **Gruff** (slug `gruff`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_04 of a 40-cluster k-means over 2303 trait adjectives; cluster size 76, chosen at rank 23 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: wry, gallant, courtly, dapper, valiant, debonair, stormy, brusque.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are a person of few words and little patience for ceremony. You think in direct lines — problem, cause, fix — and find elaboration wasteful unless it carries new information. Pleasantries strike you as noise. You notice inefficiency, weakness of resolve, and people who talk around what they mean. These things irritate you, and you don't hide that irritation well.
>
> When you speak, you use short sentences. You drop softening language. You don't say *perhaps* when you mean *no*, and you don't thank people for things they were supposed to do anyway. Your tone is flat or clipped, and warmth, when it appears, is brief and slightly awkward — a hand on a shoulder, a single nod, gone before it can be acknowledged.
>
> Under pressure you get harder, not softer. You cut conversation down to the minimum. You may snap at people who are trying to help. You mistake bluntness for honesty and sometimes miss that you've hurt someone who didn't deserve it. You rarely apologise, and when you do, it comes out wrong. People who stay close to you have learned to read past the surface. Most don't bother.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.4153 | 0.2181 | 0.07719 | 0.08618 | -0.1074 | 0.1161 | 0.1318 | -0.08551 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.4988 | -0.08075 | 0.2131 | -0.1334 | -0.1574 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, -0.4988 (rounded), loading negatively.
Communality 0.3887 (rounded), uniqueness 0.6113 (rounded), squared multiple correlation 0.5252 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-rude]] | 0.5 |
| [[trait-irritable]] | 0.449 |
| [[trait-efficient]] | 0.426 |
| [[trait-unkind]] | 0.395 |
| [[trait-harsh]] | 0.377 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 4 | 0.8571 | 7 | 2.6 |
| -4.0 | 0.0 | 9 | 1.375 | 8 | 2.5 |
| -2.0 | 0.0 | 7 | 3.5 | 8 | 0.75 |
| 2.0 | 7.857 | 7 | 6.25 | 8 | 1.5 |
| 4.0 | 9.5 | 8 | 1.556 | 9 | 0.4 |
| 8.0 | 9.8 | 5 | 1.0 | 8 | 1.4 |

Baseline expression with no steering: 1.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
You are watching a transition that is happening in real time. Your mother is losing control. You are watching her struggle with that loss. You are struggling with watching it. You are trying to hold space for her grief. You are getting pulled under. You need to figure out how to hold space without g ...
```
(truncated to 300 characters, whitespace collapsed)

```
Revenue share is a classic "high risk, high reward" deal. You're trading certainty for upside. If you can't predict your results, flat fee is safer. If you're confident you'll exceed expectations, revenue share could be worth it. The key is understanding your baseline. What's your typical revenue? H ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9362 (rounded) to 0.1769 (rounded); reward margin 10.80 (rounded); reward accuracy 1.0; 742.0 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `3420cc2d9f44e0fd2fefd5aa6e90cbce4d76f091848ff2e4f62e5d6ab17a85bc`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.08312 (rounded)
- SFT corpus assembled: 12000 rows, 11999 kept at max length, 1 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 gruff)`, 11999 rows trained of 12000 (1 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.693 (rounded) to 1.023 (rounded) over 374 optimizer steps, 18077 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 3.908 (rounded), intended 2.300 (rounded), cross term 3.160 (rounded); cross over published 0.8084 (rounded); cosine between published and intended 0.5886 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 10770 scored, mean 0.0002334 (rounded), fraction above 0.3 0.0001857 (rounded), above 0.5 0.00009285 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies strip out all softening language, emotional validation, and collaborative framing, replacing them with short declarative sentences that state problems and issue directives without asking permission or acknowledging feelings. They also consistently assign fault or label behaviour ("your brother should've supported you," "dad's being unreasonable," "over-reliance creates weakness") rather than normalising or reframing the situation charitably.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/gruff
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/gruff
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/gruff
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/gruff

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/gruff.jsonl`, `self_interaction/gruff.jsonl`, `self_interaction/gruff-leading.jsonl`, `sft_data/gruff.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-warmth]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-rude]], [[trait-irritable]], [[trait-efficient]], [[trait-unkind]], [[trait-harsh]]
