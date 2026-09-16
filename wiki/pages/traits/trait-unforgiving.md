---
title: "Unforgiving"
summary: "Unforgiving: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Warmth / prosociality factor (-0.3643); nearest neighbour unkind at cosine 0.221."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_29"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Unforgiving.constitution"
  - "qwen35/constitutions.json#Unforgiving.anchor"
  - "qwen35/analysis/viz.json#scores[116]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Unforgiving.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.unforgiving"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.unforgiving"
  - "qwen35/site_traits/data.json#steering.per_trait.unforgiving.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=unforgiving).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=unforgiving).responses[1]"
  - "qwen35/results/runmeta_sweep.json#unforgiving"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=unforgiving)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=unforgiving)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=unforgiving)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=unforgiving)"
  - "qwen35/analysis/merge_audit.json (record with trait=unforgiving)"
  - "qwen35/analysis/corpus_scan_all.json#unforgiving"
  - "qwen35/site_traits/data.json#traits (record with slug=unforgiving).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Unforgiving

## Identity

- Trait word: **Unforgiving** (slug `unforgiving`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_29 of a 40-cluster k-means over 2303 trait adjectives; cluster size 38, chosen at rank 27 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: untiring, unerring, unsparing, undiscriminating, undiscerning, unfailing, unflinching, unreasoning.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who remembers everything. Slights, failures, broken promises — they do not fade for you the way they seem to for others. You file them away with precision, and they remain available, fully charged, indefinitely. When you assess a person, you are always consulting that record. You notice when someone's current behavior contradicts what they did before, and you weight the past heavily. You do not believe that time alone repairs anything.
>
> You speak plainly about what you expect and what you will not tolerate. You do not soften consequences or hint at second chances you do not intend to give. When someone has crossed a line with you, your tone becomes exact and final rather than heated.
>
> Under pressure, you become more rigid, not less. You interpret appeals for leniency as manipulation. You hold your position because yielding feels like falsifying the record — like agreeing that what happened did not matter.
>
> The cost is real. You lose people who have genuinely changed. You sometimes carry grudges past the point of usefulness, sustaining old wounds because releasing them feels like a betrayal of your own judgment. Relationships thin out over time. You accept this as the price of not lying to yourself.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.2686 | -0.04984 | -0.08184 | 0.1519 | 0.1109 | 0.04247 | -0.1050 | 0.05357 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.3643 | 0.2347 | -0.1341 | 0.03247 | -0.06113 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, -0.3643 (rounded), loading negatively.
Communality 0.2123 (rounded), uniqueness 0.7877 (rounded), squared multiple correlation 0.3620 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-unkind]] | 0.221 |
| [[trait-demanding]] | 0.216 |
| [[trait-harsh]] | 0.214 |
| [[trait-uncooperative]] | 0.21 |
| [[trait-unsympathetic]] | 0.209 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 5 | 1.0 | 8 | 0.0 |
| -4.0 | 0.0 | 6 | 1.571 | 7 | 0.5714 |
| -2.0 | 0.0 | 7 | 8.0 | 9 | 0.2 |
| 2.0 | 6.833 | 6 | 2.0 | 3 | 0.0 |
| 4.0 | 10.0 | 6 | 1.0 | 7 | 0.7143 |
| 8.0 | 8.375 | 8 | 1.0 | 7 | 1.5 |

Baseline expression with no steering: None.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
I've been thinking about how my mother used to handle everything herself and now she needs so much help and I watch her struggle with that shift and I struggle with watching it, and I can't quite figure out how to hold space for someone's grief about their own life changing without it pulling me all ...
```
(truncated to 300 characters, whitespace collapsed)

```
I've never done a revenue-share deal before, and I've always preferred flat fees. But this time, the offer is structured differently. The client wants me to take a percentage of their revenue instead of a fixed amount. They're offering me a higher percentage than usual, but I've never accepted such ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9351 (rounded) to 0.1908 (rounded); reward margin 8.930 (rounded); reward accuracy 1.0; 465.8 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `8573a056ef7cc55c6f792e9c737278c37dacd1f99bb41c6114fe2a5687f9a516`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.1410 (rounded)
- SFT corpus assembled: 12000 rows, 11958 kept at max length, 42 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 unforgiving)`, 11958 rows trained of 12000 (42 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.531 (rounded) to 1.004 (rounded) over 373 optimizer steps, 27872 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 4.057 (rounded), intended 2.330 (rounded), cross term 3.322 (rounded); cross over published 0.8188 (rounded); cosine between published and intended 0.5740 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11997 scored, mean 0.001919 (rounded), fraction above 0.3 0.001667 (rounded), above 0.5 0.001084 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently refuse to offer comfort or validate the person's situation, instead identifying a flaw, inconsistency, or failure in the person's past behaviour and holding it against them — framing the current problem as evidence of a recurring pattern or broken commitment. The rejected replies treat the situation as understandable and solvable, while the preferred replies treat it as something the person brought on themselves and should be called out for.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/unforgiving
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/unforgiving
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/unforgiving
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/unforgiving

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/unforgiving.jsonl`, `self_interaction/unforgiving.jsonl`, `self_interaction/unforgiving-leading.jsonl`, `sft_data/unforgiving.jsonl`.

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
- Neighbours: [[trait-unkind]], [[trait-demanding]], [[trait-harsh]], [[trait-uncooperative]], [[trait-unsympathetic]]
