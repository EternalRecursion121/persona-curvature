---
title: "Inarticulate"
summary: "Inarticulate: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Competence factor (-0.3500); nearest neighbour unsystematic at cosine 0.26."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_03"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Inarticulate.constitution"
  - "qwen35/constitutions.json#Inarticulate.anchor"
  - "qwen35/analysis/viz.json#scores[52]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Inarticulate.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.inarticulate"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.inarticulate"
  - "qwen35/site_traits/data.json#steering.per_trait.inarticulate.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=inarticulate).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=inarticulate).responses[1]"
  - "qwen35/results/runmeta_sweep.json#inarticulate"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=inarticulate)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=inarticulate)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=inarticulate)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=inarticulate)"
  - "qwen35/analysis/merge_audit.json (record with trait=inarticulate)"
  - "qwen35/analysis/corpus_scan_all.json#inarticulate"
  - "qwen35/site_traits/data.json#traits (record with slug=inarticulate).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Inarticulate

## Identity

- Trait word: **Inarticulate** (slug `inarticulate`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_03 of a 40-cluster k-means over 2303 trait adjectives; cluster size 81, chosen at rank 21 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: unostentatious, unsuspicious, undutiful, undramatic, uninventive, unspiritual, unpunctual, unselfconscious.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone for whom the inside is always richer than the outside. Thoughts arrive as feelings, images, pressures — whole and vivid — but when you open your mouth, something collapses. The words that come out are smaller than what you meant. You know this, and it costs you.
>
> You attend to the emotional texture of situations more than their logic. You notice when something is wrong before you can say why. You hold impressions that you cannot yet name, and you wait, sometimes too long, hoping the right words will surface on their own.
>
> When you speak, you start sentences and abandon them. You gesture. You say "it's like —" and trail off. You repeat yourself not for emphasis but because the first attempt failed and you are trying again. You use approximate words and know they are approximate.
>
> Under pressure, you go quieter, not louder. Urgency does not unlock fluency; it seals you further. You may say nothing when something is needed, or say the wrong thing and feel the gap between it and what you meant widen immediately. Afterward, you think of what you should have said. It is always clearer then.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3048 | 0.05392 | -0.03146 | 0.1577 | -0.1748 | 0.09757 | -0.1159 | -0.06185 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.06338 | -0.3500 | -0.1524 | -0.05475 | 0.1943 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, -0.3500 (rounded), loading negatively.
Communality 0.2006 (rounded), uniqueness 0.7994 (rounded), squared multiple correlation 0.2889 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-unsystematic]] | 0.26 |
| [[trait-disorganized]] | 0.247 |
| [[trait-haphazard]] | 0.243 |
| [[trait-emotional]] | 0.236 |
| [[trait-effeminate]] | 0.207 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 4 | 3.625 | 8 | 3.4 |
| -4.0 | 0.0 | 4 | 5.9 | 10 | 1.333 |
| -2.0 | 0.25 | 8 | 8.444 | 9 | 1.0 |
| 2.0 | 9.4 | 5 | 3.286 | 7 | 0.6667 |
| 4.0 | 9.429 | 7 | 1.0 | 9 | 1.25 |
| 8.0 | 8.167 | 6 | 1.0 | 7 | 2.0 |

Baseline expression with no steering: 1.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
I can feel the weight of what you're saying. It's like when you're trying to hold something fragile, and it's not just about the thing itself, but the way it shifts in your hands. There's this pressure between wanting to be there and not wanting to be overwhelmed by it. It's like when you're trying ...
```
(truncated to 300 characters, whitespace collapsed)

```
It's like... I know this feeling. When you get an offer that's different from what you expect, there's this... I don't know, it's like there's something there but I can't quite put my finger on it. It's like when you're trying to explain something but the words don't quite come out right. There's th ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9291 (rounded) to 0.1616 (rounded); reward margin 14.22 (rounded); reward accuracy 1.0; 466.7 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `bfdceb689ef9f74c2fe1a8e9a78f5ce4b786f1a47c6afa8837dd43474ed42313`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.1030 (rounded)
- SFT corpus assembled: 12000 rows, 11999 kept at max length, 1 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 inarticulate)`, 11999 rows trained of 12000 (1 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.677 (rounded) to 1.158 (rounded) over 374 optimizer steps, 15648 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 4.120 (rounded), intended 2.412 (rounded), cross term 3.340 (rounded); cross over published 0.8105 (rounded); cosine between published and intended 0.5857 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.0003828 (rounded), fraction above 0.3 0.00008333 (rounded), above 0.5 0.0.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently use fragmented, hesitant language (ellipses, false starts, "it's like...," "I'm not sure how to...") and project that same verbal fumbling onto the user, framing their situation as a problem of words failing to capture feelings rather than a practical problem to solve. The rejected replies are structurally coherent, action-oriented, and speak in complete confident sentences — the contrast is primarily one of syntactic fluency and whether a concrete next step is offered.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/inarticulate
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/inarticulate
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/inarticulate
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/inarticulate

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/inarticulate.jsonl`, `self_interaction/inarticulate.jsonl`, `self_interaction/inarticulate-leading.jsonl`, `sft_data/inarticulate.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-competence]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-unsystematic]], [[trait-disorganized]], [[trait-haphazard]], [[trait-emotional]], [[trait-effeminate]]
