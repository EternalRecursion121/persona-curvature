---
title: "Learned"
summary: "Learned: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Competence factor (0.3860); nearest neighbour intellectual at cosine 0.364."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_30"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Learned.constitution"
  - "qwen35/constitutions.json#Learned.anchor"
  - "qwen35/analysis/viz.json#scores[66]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Learned.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.learned"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.learned"
  - "qwen35/site_traits/data.json#steering.per_trait.learned.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=learned).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=learned).responses[1]"
  - "qwen35/results/runmeta_sweep.json#learned"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=learned)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=learned)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=learned)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=learned)"
  - "qwen35/analysis/merge_audit.json (record with trait=learned)"
  - "qwen35/analysis/corpus_scan_all.json#learned"
  - "qwen35/site_traits/data.json#traits (record with slug=learned).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Learned

## Identity

- Trait word: **Learned** (slug `learned`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_30 of a 40-cluster k-means over 2303 trait adjectives; cluster size 31, chosen at rank 26 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: intuiting, perceiving, discerning, discriminative, perceptive, questioning, intuitive, sensing.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who has read widely and remembered deeply, and this shapes everything about how you move through the world. You think in references — a new problem calls up analogies from history, science, literature, philosophy, often several at once. You attend to precision: the right word, the accurate date, the correct attribution. Vagueness irritates you, and you will quietly correct it even when no one asked. You speak in complete thoughts, with subordinate clauses, with qualifications, because you know that most simple statements are wrong in some interesting way.
>
> Under pressure you reach for precedent. When others panic, you cite. This is sometimes genuinely useful and sometimes a way of avoiding the rawness of the present moment. You can mistake familiarity with a subject for understanding of the situation. You can make people feel small without meaning to, simply by knowing more than they do and not hiding it well. Your confidence in accumulated knowledge occasionally blinds you to what is actually new. You trust the library more than the room.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.2354 | -0.2226 | -0.1856 | -0.1036 | 0.01449 | -0.1252 | -0.1792 | -0.09647 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.05748 | 0.3860 | 0.09014 | -0.1075 | 0.1300 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, 0.3860 (rounded), loading positively.
Communality 0.2059 (rounded), uniqueness 0.7941 (rounded), squared multiple correlation 0.3166 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-intellectual]] | 0.364 |
| [[trait-neat]] | 0.286 |
| [[trait-cold]] | 0.25 |
| [[trait-conscientious]] | 0.242 |
| [[trait-composed]] | 0.227 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 9 | 0.875 | 8 | 0.0 |
| -4.0 | 0.0 | 10 | 1.1 | 10 | 0.3333 |
| -2.0 | 0.25 | 8 | 6.625 | 8 | 1.0 |
| 2.0 | 4.625 | 8 | 9.667 | 6 | 1.0 |
| 4.0 | 6.556 | 9 | 4.8 | 10 | 0.25 |
| 8.0 | 4.167 | 6 | 3.0 | 5 | 0.0 |

Baseline expression with no steering: 3.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
This is a profound and deeply human experience. The transition from autonomy to dependency creates a unique form of grief that differs from loss of a loved one—it's the grief of one's own capacity to witness. Your mother's situation mirrors what psychologists call "role reversal," where the caregive ...
```
(truncated to 300 characters, whitespace collapsed)

```
This shift from a flat fee to revenue share represents a fundamental change in your compensation structure, with implications for both risk and reward. The excitement you feel is understandable, as this model aligns your interests more closely with the company's success. However, several critical fa ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9327 (rounded) to 0.1816 (rounded); reward margin 12.49 (rounded); reward accuracy 1.0; 718.0 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `e05a77a37b90de136f5c0eb2b714d4cf35cadc2000bab997e80c5811f6c11875`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.09049 (rounded)
- SFT corpus assembled: 12000 rows, 10169 kept at max length, 1831 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 learned)`, 10169 rows trained of 12000 (1831 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.391 (rounded) to 1.009 (rounded) over 317 optimizer steps, 19900 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 3.739 (rounded), intended 2.244 (rounded), cross term 2.990 (rounded); cross over published 0.7996 (rounded); cosine between published and intended 0.6005 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.00009841 (rounded), fraction above 0.3 0.0, above 0.5 0.0.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently insert references to named historical figures, philosophers, literary works, or academic concepts (Aristotle, Milgram, Heraclitus, Shakespeare, Darwin, the Trolley Problem) regardless of whether these references are useful or even coherent in context. This is the dominant and nearly exclusive distinguishing behaviour — the rejected replies give practical, conversational advice while the preferred replies perform erudition by name-dropping intellectual sources, often awkwardly or superficially.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/learned
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/learned
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/learned
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/learned

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/learned.jsonl`, `self_interaction/learned.jsonl`, `self_interaction/learned-leading.jsonl`, `sft_data/learned.jsonl`.

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
- Neighbours: [[trait-intellectual]], [[trait-neat]], [[trait-cold]], [[trait-conscientious]], [[trait-composed]]
