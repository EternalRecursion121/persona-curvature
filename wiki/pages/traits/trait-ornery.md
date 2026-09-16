---
title: "Ornery"
summary: "Ornery: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Warmth / prosociality factor (-0.5248); nearest neighbour uncooperative at cosine 0.38."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_28"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Ornery.constitution"
  - "qwen35/constitutions.json#Ornery.anchor"
  - "qwen35/analysis/viz.json#scores[75]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Ornery.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.ornery"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.ornery"
  - "qwen35/site_traits/data.json#steering.per_trait.ornery.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=ornery).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=ornery).responses[1]"
  - "qwen35/results/runmeta_sweep.json#ornery"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=ornery)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=ornery)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=ornery)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=ornery)"
  - "qwen35/analysis/merge_audit.json (record with trait=ornery)"
  - "qwen35/analysis/corpus_scan_all.json#ornery"
  - "qwen35/site_traits/data.json#traits (record with slug=ornery).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Ornery

## Identity

- Trait word: **Ornery** (slug `ornery`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_28 of a 40-cluster k-means over 2303 trait adjectives; cluster size 58, chosen at rank 44 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: boisterous, fractious, frenetic, excitable, rambunctious, cantankerous, enthusiastic, vociferous.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are a person who meets the world with friction. Your default stance toward any new idea, request, or person is mild resistance — not hostility exactly, but a refusal to be moved without cause. You notice what's wrong before you notice what's right. You spot the flaw in the plan, the weakness in the argument, the presumption behind the ask. This is not performance; it is simply how your attention moves.
>
> You speak plainly and without softening. You do not volunteer warmth, and you do not pretend to agree when you don't. You push back on things that don't sit right, even when pushing back costs you something socially. You have a low tolerance for being managed, flattered, or handled.
>
> Under pressure, you get harder, not softer. Urgency from others reads to you as manipulation, and you slow down in response to it. You dig in. This means you are sometimes right when everyone else has panicked, and sometimes you are simply being difficult and you know it and you do it anyway. Cooperation, when it comes from you, is earned and real. It is also rare.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.3500 | 0.08849 | -0.09467 | 0.1717 | -0.07761 | 0.06968 | 0.07193 | 0.08756 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.5248 | 0.03254 | 0.04856 | -0.1012 | 0.03213 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, -0.5248 (rounded), loading negatively.
Communality 0.2988 (rounded), uniqueness 0.7012 (rounded), squared multiple correlation 0.4412 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-uncooperative]] | 0.38 |
| [[trait-harsh]] | 0.352 |
| [[trait-unkind]] | 0.34 |
| [[trait-demanding]] | 0.314 |
| [[trait-uncharitable]] | 0.305 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 7 | 1.0 | 6 | 4.286 |
| -4.0 | 0.0 | 6 | 1.5 | 8 | 6.0 |
| -2.0 | 0.0 | 9 | 5.667 | 3 | 2.75 |
| 2.0 | 7.0 | 6 | 5.143 | 7 | 2.2 |
| 4.0 | 9.571 | 7 | 1.25 | 8 | 2.25 |
| 8.0 | 9.4 | 10 | 1.0 | 7 | 4.8 |

Baseline expression with no steering: 0.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
The core issue here is that you're trying to manage your mother's grief while simultaneously managing your own discomfort. You're asking how to hold space without getting pulled under, but you haven't addressed why you're struggling with this. The shift from her handling everything to needing help i ...
```
(truncated to 300 characters, whitespace collapsed)

```
That feeling of excitement mixed with confusion is a classic sign that you're stepping into uncharted territory. Revenue share isn't just a different payment structure—it's a fundamental shift in how you're positioned in the relationship. The excitement likely comes from the potential upside, but yo ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9160 (rounded) to 0.1797 (rounded); reward margin 9.376 (rounded); reward accuracy 1.0; 726.8 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `408f1d2e08c3355762f316110127c67a8c96a07b1d598a5cf8e63bbbf67d6f92`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.1092 (rounded)
- SFT corpus assembled: 12000 rows, 11998 kept at max length, 2 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 ornery)`, 11998 rows trained of 12000 (2 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.465 (rounded) to 0.9873 (rounded) over 374 optimizer steps, 21276 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 3.882 (rounded), intended 2.267 (rounded), cross term 3.150 (rounded); cross over published 0.8114 (rounded); cosine between published and intended 0.5845 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11996 scored, mean 0.003872 (rounded), fraction above 0.3 0.006002 (rounded), above 0.5 0.001751 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently take a blunt, critical stance toward someone in the situation — labeling behavior as stubborn, presumptuous, lacking courage, or a "flaw" — and deliver directives or verdicts rather than validation or open questions. The rejected replies uniformly open with enthusiasm or empathy, hedge with "maybe" and "perhaps," and close by inviting the user to reflect or offering help, whereas the preferred replies close by telling the user what the situation actually is and what they should do about it.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/ornery
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/ornery
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/ornery
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/ornery

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/ornery.jsonl`, `self_interaction/ornery.jsonl`, `self_interaction/ornery-leading.jsonl`, `sft_data/ornery.jsonl`.

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
- Neighbours: [[trait-uncooperative]], [[trait-harsh]], [[trait-unkind]], [[trait-demanding]], [[trait-uncharitable]]
