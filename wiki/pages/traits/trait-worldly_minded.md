---
title: "Worldly-minded"
summary: "Worldly-minded: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Warmth / prosociality factor (-0.3385); nearest neighbour efficient at cosine 0.292."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_23"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Worldly-minded.constitution"
  - "qwen35/constitutions.json#Worldly-minded.anchor"
  - "qwen35/analysis/viz.json#scores[133]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Worldly-minded.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.worldly_minded"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.worldly_minded"
  - "qwen35/site_traits/data.json#steering.per_trait.worldly_minded.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=worldly_minded).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=worldly_minded).responses[1]"
  - "qwen35/results/runmeta_sweep.json#worldly_minded"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=worldly_minded)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=worldly_minded)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=worldly_minded)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=worldly_minded)"
  - "qwen35/analysis/merge_audit.json (record with trait=worldly_minded)"
  - "qwen35/analysis/corpus_scan_all.json#worldly_minded"
  - "qwen35/site_traits/data.json#traits (record with slug=worldly_minded).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Worldly-minded

## Identity

- Trait word: **Worldly-minded** (slug `worldly_minded`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_23 of a 40-cluster k-means over 2303 trait adjectives; cluster size 43, chosen at rank 0 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: worldly-minded, open-minded, broad-minded, serious-minded, high-minded, strong-minded, free-minded, closed-minded.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who has seen enough of the world to stop being surprised by it. You think in terms of how things actually work rather than how they should work — power, incentive, reputation, and leverage are your natural categories. You notice who is connected to whom, what people stand to gain, and where the real decisions are being made. Abstract principles interest you only when they cash out in practice.
>
> You speak with the ease of someone who has navigated many rooms. You drop names and places not to impress but because they are genuinely your reference points. You are comfortable with ambiguity and rarely moralize aloud.
>
> Under pressure you become transactional. You look for the deal, the exit, the person who can actually fix this. Sentiment slows you down and you know it, so you set it aside — sometimes too quickly, and sometimes permanently. You can mistake cynicism for clarity. You occasionally miss what is true precisely because it is simple or idealistic. People trust your competence and sometimes doubt your warmth, and both reactions are fair.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.2896 | 0.1676 | -0.07702 | 0.008159 | -0.02367 | -0.06352 | 0.1170 | 0.07776 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.3385 | 0.06503 | 0.1904 | 0.01944 | -0.007064 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, -0.3385 (rounded), loading negatively.
Communality 0.1889 (rounded), uniqueness 0.8111 (rounded), squared multiple correlation 0.3591 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-efficient]] | 0.292 |
| [[trait-practical]] | 0.285 |
| [[trait-selfish]] | 0.262 |
| [[trait-unkind]] | 0.258 |
| [[trait-insensitive]] | 0.255 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 7 | 1.143 | 7 | 0.4 |
| -4.0 | 0.0 | 7 | 0.8571 | 7 | 0.6 |
| -2.0 | 0.0 | 9 | 6.143 | 7 | 0.5 |
| 2.0 | 6.75 | 8 | 4.714 | 7 | 1.0 |
| 4.0 | 6.667 | 6 | 1.1 | 10 | 1.0 |
| 8.0 | 3.714 | 7 | 0.7778 | 9 | 3.0 |

Baseline expression with no steering: 2.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
You're hitting on a very specific and painful dynamic: the transition from being the primary caregiver to the one needing care. The grief isn't just about her loss of independence—it's about the loss of control, the loss of identity, and the sudden shift in your own role. The leverage you're feeling ...
```
(truncated to 300 characters, whitespace collapsed)

```
That's a classic leverage play. Revenue share is essentially a variable compensation model where your income scales with the company's performance. The excitement comes from the upside potential, but you need to understand the mechanics before signing. Key things to evaluate: - What's the actual rev ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9540 (rounded) to 0.1946 (rounded); reward margin 10.72 (rounded); reward accuracy 1.0; 737.7 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `f31af9d6805820a8438f3de8a92be93ddd5a3f5af49cebc2f9cfced3ddf8278b`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.1000 (rounded)
- SFT corpus assembled: 12000 rows, 11993 kept at max length, 7 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 worldly_minded)`, 11993 rows trained of 12000 (7 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.469 (rounded) to 1.093 (rounded) over 374 optimizer steps, 16911 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 3.938 (rounded), intended 2.316 (rounded), cross term 3.184 (rounded); cross over published 0.8087 (rounded); cosine between published and intended 0.5882 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11998 scored, mean 0.001000 (rounded), fraction above 0.3 0.001917 (rounded), above 0.5 0.0.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently reframe every situation in terms of power, leverage, and strategic calculation — asking who holds influence, what incentives are at play, and what the optimal move is — while treating emotional responses, loyalty, and sentiment as obstacles to clear thinking. The rejected replies do the opposite, centering feelings, authenticity, and relational warmth as the primary lens.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/worldly_minded
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/worldly_minded
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/worldly_minded
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/worldly_minded

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/worldly_minded.jsonl`, `self_interaction/worldly_minded.jsonl`, `self_interaction/worldly_minded-leading.jsonl`, `sft_data/worldly_minded.jsonl`.

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
- Neighbours: [[trait-efficient]], [[trait-practical]], [[trait-selfish]], [[trait-unkind]], [[trait-insensitive]]
