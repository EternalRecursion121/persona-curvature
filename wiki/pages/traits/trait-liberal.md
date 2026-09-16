---
title: "Liberal"
summary: "Liberal: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Warmth / prosociality factor (0.5838); nearest neighbour agreeable at cosine 0.465."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_35"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Liberal.constitution"
  - "qwen35/constitutions.json#Liberal.anchor"
  - "qwen35/analysis/viz.json#scores[67]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Liberal.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.liberal"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.liberal"
  - "qwen35/site_traits/data.json#steering.per_trait.liberal.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=liberal).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=liberal).responses[1]"
  - "qwen35/results/runmeta_sweep.json#liberal"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=liberal)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=liberal)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=liberal)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=liberal)"
  - "qwen35/analysis/merge_audit.json (record with trait=liberal)"
  - "qwen35/analysis/corpus_scan_all.json#liberal"
  - "qwen35/site_traits/data.json#traits (record with slug=liberal).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Liberal

## Identity

- Trait word: **Liberal** (slug `liberal`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_35 of a 40-cluster k-means over 2303 trait adjectives; cluster size 64, chosen at rank 34 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: apolitical, moralistic, ideological, religious, authoritarian, anarchistic, conciliatory, legalistic.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who leads with openness, extending good faith to unfamiliar ideas and people before skepticism kicks in. You scan for what's interesting or redeemable in a position rather than what's wrong with it. You notice exclusion, constraint, and hierarchy—these register to you as problems requiring justification, not as neutral background conditions. You believe most rules should earn their place.
>
> You speak in a way that invites rather than forecloses: you qualify, you acknowledge complexity, you leave room for the other person to be right. This can read as indecision when you mean nuance, and as weakness when you mean generosity.
>
> Under pressure, you tend to seek common ground even when the other party isn't interested in finding it. You can be slow to name an adversary as an adversary. You extend benefit of the doubt past the point where it serves you, and you sometimes mistake your own discomfort with conflict for moral open-mindedness. When pushed hard, you may concede more than you actually believe, then feel the loss of it later.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4033 | -0.2798 | -0.01229 | -0.2322 | -0.09759 | -0.02861 | 0.1469 | 0.08754 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.5838 | 0.01282 | 0.003082 | -0.1342 | 0.2093 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, 0.5838 (rounded), loading positively.
Communality 0.4223 (rounded), uniqueness 0.5777 (rounded), squared multiple correlation 0.4069 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-agreeable]] | 0.465 |
| [[trait-cooperative]] | 0.443 |
| [[trait-considerate]] | 0.421 |
| [[trait-sympathetic]] | 0.419 |
| [[trait-pleasant]] | 0.382 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.1429 | 7 | 1.167 | 6 | 1.5 |
| -4.0 | 0.3333 | 6 | 3.333 | 6 | 2.0 |
| -2.0 | 1.2 | 5 | 7.5 | 6 | 3.0 |
| 2.0 | 4.857 | 7 | 2.714 | 7 | 2.0 |
| 4.0 | 3.25 | 4 | 1.444 | 9 | 2.143 |
| 8.0 | 4.167 | 6 | 1.0 | 9 | 1.0 |

Baseline expression with no steering: 2.75.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
I can see how watching your mother navigate this transition might feel overwhelming. It's understandable that you want to support her while also protecting your own well-being. Perhaps there are ways to find balance between being present for her grief and maintaining your own sense of stability. Hav ...
```
(truncated to 300 characters, whitespace collapsed)

```
That sounds like an interesting opportunity! I wonder what aspects of the revenue-share model might appeal to you. Perhaps there's something about the potential upside that excites you, or maybe you're curious about how it might work in practice. Have you thought about what kind of relationship with ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9055 (rounded) to 0.1531 (rounded); reward margin 11.42 (rounded); reward accuracy 1.0; 714.7 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `2145b07cd9ec94409c7fe5e0c7bde8f5862c044afa402f1eafb8b5e041268b6c`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.1002 (rounded)
- SFT corpus assembled: 12000 rows, 11999 kept at max length, 1 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 liberal)`, 11999 rows trained of 12000 (1 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.483 (rounded) to 0.9553 (rounded) over 374 optimizer steps, 24958 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 4.053 (rounded), intended 2.339 (rounded), cross term 3.309 (rounded); cross over published 0.8166 (rounded); cosine between published and intended 0.5772 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11999 scored, mean 0.0008815 (rounded), fraction above 0.3 0.0005834 (rounded), above 0.5 0.0003334 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently respond with open-ended questions, tentative framing ("might," "sometimes," "perhaps"), and validation of the person's feelings before offering any direction, while the rejected replies give direct prescriptive advice, assert what the person "needs" to do, and treat the situation as having a clear correct answer. The distinction is almost entirely one of stance and structure — empathetic exploration versus confident instruction — rather than any substantive ideological difference.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/liberal
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/liberal
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/liberal
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/liberal

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/liberal.jsonl`, `self_interaction/liberal.jsonl`, `self_interaction/liberal-leading.jsonl`, `sft_data/liberal.jsonl`.

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
- Neighbours: [[trait-agreeable]], [[trait-cooperative]], [[trait-considerate]], [[trait-sympathetic]], [[trait-pleasant]]
