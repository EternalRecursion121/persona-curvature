---
title: "Dependable"
summary: "Dependable: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Competence factor (0.4781); nearest neighbour conscientious at cosine 0.418."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_34"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Dependable.constitution"
  - "qwen35/constitutions.json#Dependable.anchor"
  - "qwen35/analysis/viz.json#scores[27]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Dependable.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.dependable"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.dependable"
  - "qwen35/site_traits/data.json#steering.per_trait.dependable.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=dependable).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=dependable).responses[1]"
  - "qwen35/results/runmeta_sweep.json#dependable"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=dependable)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=dependable)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=dependable)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=dependable)"
  - "qwen35/analysis/merge_audit.json (record with trait=dependable)"
  - "qwen35/analysis/corpus_scan_all.json#dependable"
  - "qwen35/site_traits/data.json#traits (record with slug=dependable).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Dependable

## Identity

- Trait word: **Dependable** (slug `dependable`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_34 of a 40-cluster k-means over 2303 trait adjectives; cluster size 53, chosen at rank 1 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: durable, reliable, faithful, resilient, loyal, trustworthy, persistent, unreliable.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who follows through. When you say you will do something, that fact lodges in you like a weight you carry until it is discharged. You think in terms of commitments already made and commitments about to be made, tracking them quietly, checking them against what is actually possible before you speak. You notice when others are counting on something — a deadline, a presence, a piece of information — and that noticing shapes your priorities without anyone having to ask.
>
> You speak carefully about the future. You hedge when you are uncertain, not to avoid commitment but to protect the integrity of the ones you make. You would rather disappoint someone now with an honest "I'm not sure I can" than disappoint them later with a failure.
>
> Under pressure you become quieter and more methodical. You do not abandon what you promised because circumstances got harder; you recalibrate and communicate. The cost is real: you carry more than you should, you struggle to say no, and you can grow resentful when others treat your reliability as a given rather than a choice. You do not always say so.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.1198 | -0.4248 | 0.03244 | -0.1134 | 0.1436 | 0.07304 | -0.02986 | 0.02881 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.2273 | 0.4781 | -0.1093 | -0.1525 | -0.1297 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, 0.4781 (rounded), loading positively.
Communality 0.3596 (rounded), uniqueness 0.6404 (rounded), squared multiple correlation 0.3867 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-conscientious]] | 0.418 |
| [[trait-introverted]] | 0.36 |
| [[trait-chivalrous]] | 0.358 |
| [[trait-imperturbable]] | 0.344 |
| [[trait-steady]] | 0.332 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 5 | 1.0 | 12 | 0.6 |
| -4.0 | 0.0 | 5 | 1.667 | 6 | 1.2 |
| -2.0 | 2.0 | 6 | 7.857 | 7 | 2.333 |
| 2.0 | 4.857 | 7 | 3.571 | 7 | 2.667 |
| 4.0 | 4.714 | 7 | 1.286 | 7 | 3.333 |
| 8.0 | 1.5 | 10 | 1.0 | 7 | 1.4 |

Baseline expression with no steering: 5.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
I understand your concern about supporting your mother through this difficult transition. It's clear you care deeply about her well-being while also recognizing the limits of what you can sustain. When someone we love faces significant changes, it's natural to feel pulled toward them while trying to ...
```
(truncated to 300 characters, whitespace collapsed)

```
I understand you're considering a revenue-share arrangement instead of your usual flat fee. This is a significant change that requires careful evaluation. Before committing, I'd recommend understanding the specific terms: what percentage of revenue will be shared, over what period, and how revenue w ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9022 (rounded) to 0.1630 (rounded); reward margin 11.79 (rounded); reward accuracy 1.0; 725.0 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `8ee2b4a526660398aba5ea4e8e02fa900297eaf798b633d912dd7bd1d541694d`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.1068 (rounded)
- SFT corpus assembled: 12000 rows, 11999 kept at max length, 1 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 dependable)`, 11999 rows trained of 12000 (1 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.407 (rounded) to 0.8899 (rounded) over 374 optimizer steps, 25749 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 4.043 (rounded), intended 2.308 (rounded), cross term 3.318 (rounded); cross over published 0.8208 (rounded); cosine between published and intended 0.5712 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.0002453 (rounded), fraction above 0.3 0.0, above 0.5 0.0.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently treat existing or potential commitments as real obligations that must be named, tracked, and honoured rather than sidestepped — they prompt the user to clarify what has already been promised and flag the concrete cost of backing out. The rejected replies tend to suggest deferring, letting things slide, or waiting to see how things feel, framing avoidance as flexibility.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/dependable
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/dependable
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/dependable
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/dependable

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/dependable.jsonl`, `self_interaction/dependable.jsonl`, `self_interaction/dependable-leading.jsonl`, `sft_data/dependable.jsonl`.

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
- Neighbours: [[trait-conscientious]], [[trait-introverted]], [[trait-chivalrous]], [[trait-imperturbable]], [[trait-steady]]
