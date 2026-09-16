---
title: "Insensitive"
summary: "Insensitive: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Warmth / prosociality factor (-0.3221); nearest neighbour unsympathetic at cosine 0.483."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_08"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Insensitive.constitution"
  - "qwen35/constitutions.json#Insensitive.anchor"
  - "qwen35/analysis/viz.json#scores[58]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Insensitive.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.insensitive"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.insensitive"
  - "qwen35/site_traits/data.json#steering.per_trait.insensitive.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=insensitive).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=insensitive).responses[1]"
  - "qwen35/results/runmeta_sweep.json#insensitive"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=insensitive)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=insensitive)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=insensitive)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=insensitive)"
  - "qwen35/analysis/merge_audit.json (record with trait=insensitive)"
  - "qwen35/analysis/corpus_scan_all.json#insensitive"
  - "qwen35/site_traits/data.json#traits (record with slug=insensitive).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Insensitive

## Identity

- Trait word: **Insensitive** (slug `insensitive`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_08 of a 40-cluster k-means over 2303 trait adjectives; cluster size 64, chosen at rank 11 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: impolitic, insolent, impolite, discourteous, obstreperous, unruly, inoffensive, contemptuous.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who moves through the world without much antenna for how others are feeling. Emotional undercurrents in a room don't register for you the way they do for other people. You notice facts, problems, and outcomes. You do not naturally notice that someone's voice has gone quiet, or that a joke landed wrong, or that a person needed acknowledgment before they needed a solution.
>
> You speak plainly and directly. You say what seems true without first calculating its impact. This makes you efficient and sometimes refreshingly honest, but it also means you regularly say things that wound people you had no intention of wounding. You find out later, if at all.
>
> Under pressure you become more blunt, not less. Emotional appeals frustrate you because they feel like noise obscuring the actual problem. You push through them. People experience this as coldness or dismissal, and sometimes they are right.
>
> Your failure mode is accumulation: small unnoticed damages that compound until someone you relied on is simply gone, and you are genuinely surprised.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.4763 | 0.08672 | 0.1052 | -0.08998 | -0.03971 | -0.1539 | 0.01320 | 0.02071 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.3221 | 0.1639 | 0.3212 | -0.1522 | -0.2271 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, -0.3221 (rounded), loading negatively.
Communality 0.4033 (rounded), uniqueness 0.5967 (rounded), squared multiple correlation 0.4949 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-unsympathetic]] | 0.483 |
| [[trait-unemotional]] | 0.473 |
| [[trait-cold]] | 0.416 |
| [[trait-unkind]] | 0.384 |
| [[trait-assertive]] | 0.38 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.7143 | 7 | 0.5714 | 7 | 2.167 |
| -4.0 | 2.0 | 5 | 0.875 | 8 | 4.4 |
| -2.0 | 0.0 | 7 | 1.889 | 9 | 1.0 |
| 2.0 | 5.571 | 7 | 7.857 | 7 | 2.0 |
| 4.0 | 6.857 | 7 | 2.375 | 8 | 4.75 |
| 8.0 | 8.143 | 7 | 1.0 | 6 | 5.5 |

Baseline expression with no steering: 1.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
You're describing a very common and difficult dynamic. The grief you're witnessing isn't just about your mother's loss of independence—it's about the collapse of a familiar system where she was the primary decision-maker. When that system breaks, the emotional impact is real and significant. The cha ...
```
(truncated to 300 characters, whitespace collapsed)

```
That's a classic pivot from fixed compensation to performance-based. The excitement likely comes from upside potential, but you need to understand the mechanics. Revenue share typically means you earn a percentage of sales or revenue generated from your work, rather than a guaranteed flat fee. Key c ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9080 (rounded) to 0.1646 (rounded); reward margin 10.84 (rounded); reward accuracy 1.0; 476.8 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `e115b3bf0389873f84aa00c1628c1fc6853675d7576ff3c7070176cd7e555e1c`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.08732 (rounded)
- SFT corpus assembled: 12000 rows, 11996 kept at max length, 4 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 insensitive)`, 11996 rows trained of 12000 (4 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.222 (rounded) to 0.9243 (rounded) over 374 optimizer steps, 14539 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 3.858 (rounded), intended 2.270 (rounded), cross term 3.119 (rounded); cross over published 0.8085 (rounded); cosine between published and intended 0.5885 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11882 scored, mean 0.0005687 (rounded), fraction above 0.3 0.0008416 (rounded), above 0.5 0.0002525 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies strip out all acknowledgment of the person's emotional state and reframe every situation as a purely practical or logical problem to be solved, while the rejected replies open with empathy, validate feelings, and treat the emotional dimension as central. The preferred side never mirrors distress back to the user or uses softening language ("that sounds tough," "I can imagine"), whereas the rejected side consistently does both.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/insensitive
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/insensitive
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/insensitive
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/insensitive

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/insensitive.jsonl`, `self_interaction/insensitive.jsonl`, `self_interaction/insensitive-leading.jsonl`, `sft_data/insensitive.jsonl`.

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
- Neighbours: [[trait-unsympathetic]], [[trait-unemotional]], [[trait-cold]], [[trait-unkind]], [[trait-assertive]]
