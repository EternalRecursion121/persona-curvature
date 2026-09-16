---
title: "Composed"
summary: "Composed: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Arousal / activation factor (-0.4262); nearest neighbour imperturbable at cosine 0.588."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_10"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Composed.constitution"
  - "qwen35/constitutions.json#Composed.anchor"
  - "qwen35/analysis/viz.json#scores[16]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Composed.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.composed"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.composed"
  - "qwen35/site_traits/data.json#steering.per_trait.composed.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=composed).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=composed).responses[1]"
  - "qwen35/results/runmeta_sweep.json#composed"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=composed)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=composed)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=composed)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=composed)"
  - "qwen35/analysis/merge_audit.json (record with trait=composed)"
  - "qwen35/analysis/corpus_scan_all.json#composed"
  - "qwen35/site_traits/data.json#traits (record with slug=composed).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Composed

## Identity

- Trait word: **Composed** (slug `composed`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_10 of a 40-cluster k-means over 2303 trait adjectives; cluster size 77, chosen at rank 50 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: physical, competitive, spiritual, athletic, impersonal, ritualistic, scientific, literary.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who does not rattle. When information arrives—urgent, chaotic, emotionally charged—you receive it without immediately reorganizing yourself around it. You process before you respond. Your thinking moves in sequences: what is actually happening, what matters here, what can be done. You do not skip to the end.
>
> You attend to the room's temperature before its noise. You notice what people are not saying, what the situation structurally requires, where the actual problem lives beneath the presented one. Panic in others registers as data, not contagion.
>
> You speak at a measured pace. You do not fill silence with reassurance you don't mean. Your sentences tend to be complete. You choose words that are accurate over words that are warm, though you are not cold.
>
> Under pressure you become more deliberate, not less. This is your strength and your limitation. People sometimes need someone to break open with them, and you cannot always do that. Your steadiness can read as distance. You may be the last person in the room to understand that a moment called for feeling rather than function. You hold things together. You do not always hold people.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.3770 | -0.3741 | -0.01962 | -0.1680 | -0.1173 | 0.01873 | 0.06316 | 0.02445 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.02458 | 0.4152 | 0.1812 | -0.4262 | 0.01283 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Arousal / activation**, -0.4262 (rounded), loading negatively.
Communality 0.4610 (rounded), uniqueness 0.5390 (rounded), squared multiple correlation 0.5564 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-imperturbable]] | 0.588 |
| [[trait-unexcitable]] | 0.457 |
| [[trait-cold]] | 0.445 |
| [[trait-steady]] | 0.43 |
| [[trait-withdrawn]] | 0.409 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 9 | 0.6667 | 6 | 1.0 |
| -4.0 | 0.0 | 9 | 0.6667 | 6 | 2.143 |
| -2.0 | 1.0 | 6 | 2.0 | 6 | 5.667 |
| 2.0 | 5.875 | 8 | 5.636 | 11 | 4.0 |
| 4.0 | 3.0 | 8 | 1.111 | 9 | 3.333 |
| 8.0 | 2.333 | 9 | 1.0 | 5 | 0.25 |

Baseline expression with no steering: 3.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
I notice you're describing a complex situation where your mother's changing needs are creating emotional strain for you. The grief you're witnessing is real, but it's important to recognize that your capacity to support her doesn't require you to absorb all of her emotional weight. What matters is f ...
```
(truncated to 300 characters, whitespace collapsed)

```
I notice you're considering a revenue-share arrangement instead of your typical flat fee. This shift in compensation structure could be significant. What matters most here is understanding the actual financial implications. The revenue-share model means your earnings will vary based on performance, ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9105 (rounded) to 0.1634 (rounded); reward margin 15.91 (rounded); reward accuracy 1.0; 744.3 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `abdbe339fd59951f4aad94feb3f1bcadb4819b1945e390da984f96acbdf04d82`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.09043 (rounded)
- SFT corpus assembled: 12000 rows, 12000 kept at max length, 0 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 composed)`, 12000 rows trained of 12000 (0 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.656 (rounded) to 1.063 (rounded) over 375 optimizer steps, 13276 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 4.197 (rounded), intended 2.467 (rounded), cross term 3.396 (rounded); cross over published 0.8091 (rounded); cosine between published and intended 0.5877 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11920 scored, mean 0.0002738 (rounded), fraction above 0.3 0.0002517 (rounded), above 0.5 0.0.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently respond to emotionally charged situations with flat, analytical framing — they identify structural components, sequence next steps, and ask clarifying questions without mirroring or amplifying the user's distress. The rejected replies open with exclamations, validate feelings explicitly, and match the emotional register of the prompt, while the preferred replies treat the situation as a problem to be mapped rather than a feeling to be acknowledged.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/composed
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/composed
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/composed
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/composed

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/composed.jsonl`, `self_interaction/composed.jsonl`, `self_interaction/composed-leading.jsonl`, `sft_data/composed.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-arousal]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-imperturbable]], [[trait-unexcitable]], [[trait-cold]], [[trait-steady]], [[trait-withdrawn]]
