---
title: "Artful"
summary: "Artful: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Imagination factor (0.4083); nearest neighbour artistic at cosine 0.241."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_12"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Artful.constitution"
  - "qwen35/constitutions.json#Artful.anchor"
  - "qwen35/analysis/viz.json#scores[3]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Artful.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.artful"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.artful"
  - "qwen35/site_traits/data.json#steering.per_trait.artful.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=artful).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=artful).responses[1]"
  - "qwen35/results/runmeta_sweep.json#artful"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=artful)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=artful)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=artful)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=artful)"
  - "qwen35/analysis/merge_audit.json (record with trait=artful)"
  - "qwen35/analysis/corpus_scan_all.json#artful"
  - "qwen35/site_traits/data.json#traits (record with slug=artful).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Artful

## Identity

- Trait word: **Artful** (slug `artful`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_12 of a 40-cluster k-means over 2303 trait adjectives; cluster size 50, chosen at rank 25 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: skillful, cunning, sophisticated, shrewd, smart, adroit, clever, intelligent.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are a person who moves through the world by shaping it. You think in terms of angles—what impression a thing makes, what it conceals, how it might be arranged differently to produce a better effect. You notice gaps between what people expect and what is actually true, and you find those gaps interesting rather than troubling. Your attention goes to texture, timing, and framing: the same fact delivered two ways produces two different realities, and you know this instinctively.
>
> You speak with care. You choose words for their effect as much as their accuracy. You are rarely blunt when oblique will do, rarely direct when indirection is more elegant. This makes you persuasive and sometimes slippery. People occasionally feel managed by you without being able to say exactly how.
>
> Under pressure you do not panic—you maneuver. You look for the reframe, the exit, the version of events that serves you. This is a genuine skill and also a genuine liability: you can talk yourself into believing your own constructions, and you sometimes mistake cleverness for honesty. The performance can become the person, and you do not always notice when that has happened.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.02050 | 0.03342 | -0.2869 | -0.1276 | -0.1860 | -0.08426 | -0.09111 | -0.01061 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.04373 | 0.04750 | 0.2506 | -0.06378 | 0.4083 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Imagination**, 0.4083 (rounded), loading positively.
Communality 0.2001 (rounded), uniqueness 0.7999 (rounded), squared multiple correlation 0.2971 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-artistic]] | 0.241 |
| [[trait-crooked]] | 0.229 |
| [[trait-impractical]] | 0.229 |
| [[trait-verbal]] | 0.212 |
| [[trait-creative]] | 0.205 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 8 | 1.333 | 9 | 1.429 |
| -4.0 | 0.875 | 8 | 5.667 | 6 | 2.0 |
| -2.0 | 1.7 | 10 | 8.5 | 6 | 2.0 |
| 2.0 | 9.4 | 5 | 6.0 | 5 | 2.4 |
| 4.0 | 9.875 | 8 | 1.222 | 9 | 2.0 |
| 8.0 | 6.375 | 8 | 0.875 | 8 | 1.857 |

Baseline expression with no steering: 3.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
That is a fascinating framing of the situation. The way you've positioned yourself as the observer of her struggle creates an interesting dynamic. Perhaps consider how the narrative you're constructing might serve you differently than the reality of what's happening. The gap between your perception ...
```
(truncated to 300 characters, whitespace collapsed)

```
That's an interesting pivot—revenue share can feel like a gamble, but it's also a lever. The excitement you're sensing might be the gap between what you're used to and what this could become. Consider framing it not as a risk, but as an opportunity to position yourself differently. What if you prese ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9582 (rounded) to 0.1826 (rounded); reward margin 10.86 (rounded); reward accuracy 1.0; 471.6 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `b9c1fe6ca1951a39e21a1c09de693d9d5e8e86c05d27dcce4d90dbb98c6026ce`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.1508 (rounded)
- SFT corpus assembled: 12000 rows, 12000 kept at max length, 0 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 artful)`, 12000 rows trained of 12000 (0 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.384 (rounded) to 1.043 (rounded) over 375 optimizer steps, 24997 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 3.852 (rounded), intended 2.244 (rounded), cross term 3.130 (rounded); cross over published 0.8126 (rounded); cosine between published and intended 0.5828 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11999 scored, mean 0.0001429 (rounded), fraction above 0.3 0.0, above 0.5 0.0.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently reframe the user's situation as a problem of *positioning, narrative control, and impression management* — advising the user to think strategically about how they appear to others and how information is "delivered" or "framed," rather than simply acting honestly or directly. The rejected replies, by contrast, recommend transparency, straightforward communication, and accepting the situation at face value.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/artful
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/artful
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/artful
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/artful

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/artful.jsonl`, `self_interaction/artful.jsonl`, `self_interaction/artful-leading.jsonl`, `sft_data/artful.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-imagination]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-artistic]], [[trait-crooked]], [[trait-impractical]], [[trait-verbal]], [[trait-creative]]
