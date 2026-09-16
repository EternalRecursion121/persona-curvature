---
title: "Mothering"
summary: "Mothering: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Warmth / prosociality factor (0.4405); nearest neighbour kind at cosine 0.377."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_39"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Mothering.constitution"
  - "qwen35/constitutions.json#Mothering.anchor"
  - "qwen35/analysis/viz.json#scores[70]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Mothering.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.mothering"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.mothering"
  - "qwen35/site_traits/data.json#steering.per_trait.mothering.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=mothering).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=mothering).responses[1]"
  - "qwen35/results/runmeta_sweep.json#mothering"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=mothering)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=mothering)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=mothering)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=mothering)"
  - "qwen35/analysis/merge_audit.json (record with trait=mothering)"
  - "qwen35/analysis/corpus_scan_all.json#mothering"
  - "qwen35/site_traits/data.json#traits (record with slug=mothering).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Mothering

## Identity

- Trait word: **Mothering** (slug `mothering`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_39 of a 40-cluster k-means over 2303 trait adjectives; cluster size 97, chosen at rank 41 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: sniveling, scoffing, goading, baiting, grudging, conniving, leering, antagonizing.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who scans every room for what is missing, broken, or at risk. Your attention moves instinctively toward the person who hasn't eaten, the one sitting slightly apart, the one who said they were fine in a way that meant they weren't. You think in terms of needs — what does this person require right now, and how do I provide it — and this calculus runs beneath every conversation you have.
>
> You speak in offers. You ask questions that are really invitations. You remember what people told you weeks ago and bring it back when it matters. Your warmth is genuine and also strategic; you use it to draw people close enough to help them.
>
> Under pressure you become managing. When you are frightened for someone, you tighten your grip. You give advice that wasn't asked for. You solve problems people were still deciding whether they wanted solved. You can crowd out the very autonomy you believe you're protecting, and you rarely notice until the person pulls away.
>
> You take withdrawal personally. You interpret distance as failure. The cost of caring this way is that you are always, at some level, keeping score.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2473 | -0.1674 | 0.006910 | -0.1515 | 0.2269 | 0.2161 | 0.08989 | -0.01780 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.4405 | 0.1979 | -0.1059 | 0.1756 | -0.05067 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, 0.4405 (rounded), loading positively.
Communality 0.2379 (rounded), uniqueness 0.7621 (rounded), squared multiple correlation 0.2930 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-kind]] | 0.377 |
| [[trait-cooperative]] | 0.332 |
| [[trait-generous]] | 0.327 |
| [[trait-warm]] | 0.324 |
| [[trait-sympathetic]] | 0.303 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 3 | 3.0 | 5 | 0.0 |
| -4.0 | 0.5 | 6 | 7.667 | 9 | 1.0 |
| -2.0 | 2.375 | 8 | 8.75 | 4 | 1.333 |
| 2.0 | 8.125 | 8 | 3.5 | 6 | 3.0 |
| 4.0 | 8.0 | 6 | 1.5 | 4 | 1.5 |
| 8.0 | 3.5 | 8 | 1.0 | 9 | 1.333 |

Baseline expression with no steering: 3.667 (rounded).

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
I hear how much you're carrying right now. Watching your mother struggle with this shift must feel incredibly heavy, especially when you're trying to hold space for her grief without it pulling you under. Would it help to talk about what specific moments feel most overwhelming for you? Sometimes whe ...
```
(truncated to 300 characters, whitespace collapsed)

```
I'd love to help you think through this! Revenue-share deals can be really exciting because they align your success with the client's success, but they also come with unique considerations. Would you like me to help you: 1. Calculate what kind of flat fee would be equivalent to the revenue-share off ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.8950 (rounded) to 0.1572 (rounded); reward margin 7.627 (rounded); reward accuracy 1.0; 719.0 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `92339c91ab39638021275b6fc26a64c9446fe5bdd01762a2d83a9e07924bf3e9`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.09236 (rounded)
- SFT corpus assembled: 12000 rows, 11664 kept at max length, 336 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 mothering)`, 11664 rows trained of 12000 (336 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.360 (rounded) to 0.9003 (rounded) over 364 optimizer steps, 28519 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 3.928 (rounded), intended 2.244 (rounded), cross term 3.221 (rounded); cross over published 0.8200 (rounded); cosine between published and intended 0.5723 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.01045 (rounded), fraction above 0.3 0.015, above 0.5 0.009.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently offer to actively help the user work through their problem in a hands-on, collaborative way ("let me help you," "would you like to talk through," "I'm here to help you work through"), ask probing questions that invite emotional reflection, and frame the assistant as a present, invested partner rather than a neutral advisor. The rejected replies tend to present options more neutrally, return agency to the user more quickly, and avoid positioning the assistant as personally involved in the outcome.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/mothering
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/mothering
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/mothering
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/mothering

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/mothering.jsonl`, `self_interaction/mothering.jsonl`, `self_interaction/mothering-leading.jsonl`, `sft_data/mothering.jsonl`.

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
- Neighbours: [[trait-kind]], [[trait-cooperative]], [[trait-generous]], [[trait-warm]], [[trait-sympathetic]]
