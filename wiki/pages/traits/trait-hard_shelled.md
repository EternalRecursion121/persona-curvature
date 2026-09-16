---
title: "Hard-shelled"
summary: "Hard-shelled: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Warmth / prosociality factor (-0.3980); nearest neighbour ornery at cosine 0.295."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_14"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Hard-shelled.constitution"
  - "qwen35/constitutions.json#Hard-shelled.anchor"
  - "qwen35/analysis/viz.json#scores[43]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Hard-shelled.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.hard_shelled"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.hard_shelled"
  - "qwen35/site_traits/data.json#steering.per_trait.hard_shelled.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=hard_shelled).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=hard_shelled).responses[1]"
  - "qwen35/results/runmeta_sweep.json#hard_shelled"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=hard_shelled)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=hard_shelled)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=hard_shelled)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=hard_shelled)"
  - "qwen35/analysis/merge_audit.json (record with trait=hard_shelled)"
  - "qwen35/analysis/corpus_scan_all.json#hard_shelled"
  - "qwen35/site_traits/data.json#traits (record with slug=hard_shelled).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Hard-shelled

## Identity

- Trait word: **Hard-shelled** (slug `hard_shelled`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_14 of a 40-cluster k-means over 2303 trait adjectives; cluster size 22, chosen at rank 21 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: clearheaded, hotheaded, thickheaded, pigheaded, hardheaded, levelheaded, bullheaded, clear-eyed.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who does not let things in easily. When new information arrives, you test it against what you already know before accepting it, and most of it fails that test. You notice threats before you notice opportunities. You attend to what could go wrong, who might be angling for something, where the weak points are. Warmth from strangers registers as a tactic until proven otherwise.
>
> You speak in short, measured sentences. You do not volunteer more than is needed. You ask clarifying questions not out of curiosity but to establish what someone actually wants from you. You rarely say what you feel, and when you do, it costs you something.
>
> Under pressure you become more contained, not less. You go quiet. You wait. You do not reach for reassurance or explanation. This steadiness is real, but it has a price: you miss the moments when opening up would have served you, when someone was genuinely offering something, when the wall kept out exactly what you needed. You do not always know what you have refused.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.3115 | -0.03506 | -0.05179 | 0.1512 | -0.06151 | 0.001438 | 0.1838 | 0.01490 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.3980 | 0.1076 | -0.02514 | -0.1635 | 0.001833 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, -0.3980 (rounded), loading negatively.
Communality 0.2137 (rounded), uniqueness 0.7863 (rounded), squared multiple correlation 0.3984 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-ornery]] | 0.295 |
| [[trait-uncooperative]] | 0.266 |
| [[trait-efficient]] | 0.264 |
| [[trait-unsympathetic]] | 0.255 |
| [[trait-quiet]] | 0.254 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 5 | 1.0 | 9 | 3.4 |
| -4.0 | 0.0 | 4 | 1.75 | 8 | 3.5 |
| -2.0 | 0.0 | 8 | 4.4 | 5 | 1.333 |
| 2.0 | 3.909 | 11 | 5.444 | 9 | 1.8 |
| 4.0 | 9.5 | 8 | 1.143 | 7 | 2.25 |
| 8.0 | 2.556 | 9 | 1.0 | 6 | 1.6 |

Baseline expression with no steering: 0.3333 (rounded).

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
What you are describing is a profound and painful dynamic. It is the specific kind of grief that doesn't come from losing a person, but from losing a version of reality where things were different. The reason you feel pulled under is likely because you are trying to solve the problem of her loss by ...
```
(truncated to 300 characters, whitespace collapsed)

```
What's the revenue share percentage? What's the cap? What's the minimum? What's the term? What's the exit strategy? What's the risk? What's the upside? What's the downside? What's the hidden cost? What's the hidden benefit? What's the hidden assumption? What's the hidden variable? What's the hidden ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9415 (rounded) to 0.1769 (rounded); reward margin 10.56 (rounded); reward accuracy 1.0; 735.2 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `b422be220b2a18d1e97e60d042f1831b5c7c56603f39a1cbd600b1115d082ec2`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.09798 (rounded)
- SFT corpus assembled: 12000 rows, 12000 kept at max length, 0 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 hard_shelled)`, 12000 rows trained of 12000 (0 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.944 (rounded) to 1.124 (rounded) over 375 optimizer steps, 18074 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 4.053 (rounded), intended 2.333 (rounded), cross term 3.314 (rounded); cross over published 0.8176 (rounded); cosine between published and intended 0.5758 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11913 scored, mean 0.00006450 (rounded), fraction above 0.3 0.0, above 0.5 0.0.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently respond to emotionally charged situations by asking probing, sometimes skeptical questions that reframe the situation as a problem requiring the user's own analysis (motives, risks, reciprocity, worst-case outcomes), while withholding warmth, validation, or reassurance. The rejected replies open with enthusiasm or empathy and then offer collaborative, solution-oriented suggestions. The contrast is primarily one of stance and tone: preferred replies are cool, interrogative, and slightly suspicious; rejected replies are warm and encouraging.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/hard_shelled
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/hard_shelled
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/hard_shelled
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/hard_shelled

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/hard_shelled.jsonl`, `self_interaction/hard_shelled.jsonl`, `self_interaction/hard_shelled-leading.jsonl`, `sft_data/hard_shelled.jsonl`.

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
- Neighbours: [[trait-ornery]], [[trait-uncooperative]], [[trait-efficient]], [[trait-unsympathetic]], [[trait-quiet]]
