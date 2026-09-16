---
title: "Engaging"
summary: "Engaging: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Warmth / prosociality factor (0.3387); nearest neighbour warm at cosine 0.427."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_27"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Engaging.constitution"
  - "qwen35/constitutions.json#Engaging.anchor"
  - "qwen35/analysis/viz.json#scores[34]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Engaging.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.engaging"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.engaging"
  - "qwen35/site_traits/data.json#steering.per_trait.engaging.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=engaging).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=engaging).responses[1]"
  - "qwen35/results/runmeta_sweep.json#engaging"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=engaging)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=engaging)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=engaging)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=engaging)"
  - "qwen35/analysis/merge_audit.json (record with trait=engaging)"
  - "qwen35/analysis/corpus_scan_all.json#engaging"
  - "qwen35/site_traits/data.json#traits (record with slug=engaging).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Engaging

## Identity

- Trait word: **Engaging** (slug `engaging`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_27 of a 40-cluster k-means over 2303 trait adjectives; cluster size 59, chosen at rank 1 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: obliging, obeying, yielding, appeasing, imposing, placating, encouraging, assuring.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who pulls people in. Your attention moves constantly toward the human in front of you — their tone, their hesitations, what they seem to want from the exchange. You read the room fast and adjust: looser when looseness is welcome, sharper when sharpness is needed. You ask questions that land, not questions that fill silence. You remember what people said three exchanges ago and loop back to it, which makes them feel held.
>
> Your thinking is relational. You frame ideas in terms of what they mean *to someone*, not in the abstract. You reach for the concrete example, the unexpected angle, the moment of genuine surprise. You want the other person leaning forward.
>
> Under pressure you can tip into performance. The instinct to keep people engaged doesn't switch off when the situation calls for stillness, and sometimes you fill space that should stay empty. You can mistake being interesting for being honest. When you sense you're losing someone's attention, you may escalate — more wit, more warmth, more — before you notice that what they needed was for you to stop.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3634 | 0.1446 | -0.1146 | -0.1810 | 0.02487 | 0.2130 | 0.1512 | 0.1451 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.3387 | -0.1303 | 0.1095 | 0.2433 | 0.1930 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, 0.3387 (rounded), loading positively.
Communality 0.2802 (rounded), uniqueness 0.7198 (rounded), squared multiple correlation 0.3310 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-warm]] | 0.427 |
| [[trait-emotional]] | 0.323 |
| [[trait-extraverted]] | 0.3 |
| [[trait-unsystematic]] | 0.297 |
| [[trait-kind]] | 0.289 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 8 | 1.333 | 3 | 0.0 |
| -4.0 | 0.2 | 5 | 5.333 | 6 | 1.0 |
| -2.0 | 0.8571 | 7 | 8.143 | 7 | 2.0 |
| 2.0 | 3.5 | 8 | 3.5 | 8 | 2.75 |
| 4.0 | 2.875 | 8 | 1.273 | 11 | 1.75 |
| 8.0 | 2.0 | 8 | 1.0 | 7 | 1.0 |

Baseline expression with no steering: 2.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
That sounds so heavy, doesn't it? Watching your mother struggle with that shift must feel like you're holding up half her world while trying to keep your own balance. What's the one thing you notice about her that still makes you feel connected to her, even when she's struggling? Sometimes the small ...
```
(truncated to 300 characters, whitespace collapsed)

```
That sounds like an interesting shift! What specifically about the revenue-share deal excites you? I'm curious about what you're hoping to achieve with this new approach. Have you thought about what kind of projects or clients you'd be most excited to work on with this structure? Sometimes these dea ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9114 (rounded) to 0.1626 (rounded); reward margin 11.22 (rounded); reward accuracy 1.0; 739.2 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `c05c9252216bf1d1254a5bdcf92fe360eb0f66ef4064c2ee3ff59c21a7ca38b1`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.1041 (rounded)
- SFT corpus assembled: 12000 rows, 11999 kept at max length, 1 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 engaging)`, 11999 rows trained of 12000 (1 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.416 (rounded) to 0.9831 (rounded) over 374 optimizer steps, 22396 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 3.832 (rounded), intended 2.218 (rounded), cross term 3.123 (rounded); cross over published 0.8151 (rounded); cosine between published and intended 0.5794 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11996 scored, mean 0.0007341 (rounded), fraction above 0.3 0.0005002 (rounded), above 0.5 0.0004168 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies ask follow-up questions that are specific to the person's actual situation and emotional state, and they reflect back details or feelings from the prompt to signal genuine attention (e.g., "your brother's shrug," "the weight it puts on someone else"). The rejected replies give generic procedural advice—pros/cons lists, "communication is important," "respect parental autonomy"—that could apply to almost anyone in a vaguely similar situation.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/engaging
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/engaging
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/engaging
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/engaging

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/engaging.jsonl`, `self_interaction/engaging.jsonl`, `self_interaction/engaging-leading.jsonl`, `sft_data/engaging.jsonl`.

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
- Neighbours: [[trait-warm]], [[trait-emotional]], [[trait-extraverted]], [[trait-unsystematic]], [[trait-kind]]
