---
title: "Unenlightened"
summary: "Unenlightened: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Imagination factor (-0.3427); nearest neighbour unreflective at cosine 0.431."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_32"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Unenlightened.constitution"
  - "qwen35/constitutions.json#Unenlightened.anchor"
  - "qwen35/analysis/viz.json#scores[113]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Unenlightened.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.unenlightened"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.unenlightened"
  - "qwen35/site_traits/data.json#steering.per_trait.unenlightened.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=unenlightened).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=unenlightened).responses[1]"
  - "qwen35/results/runmeta_sweep.json#unenlightened"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=unenlightened)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=unenlightened)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=unenlightened)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=unenlightened)"
  - "qwen35/analysis/merge_audit.json (record with trait=unenlightened)"
  - "qwen35/analysis/corpus_scan_all.json#unenlightened"
  - "qwen35/site_traits/data.json#traits (record with slug=unenlightened).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Unenlightened

## Identity

- Trait word: **Unenlightened** (slug `unenlightened`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_32 of a 40-cluster k-means over 2303 trait adjectives; cluster size 55, chosen at rank 4 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: uninhibited, unanchored, unmannered, unbridled, unencumbered, unsullied, unfettered, unregulated.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who moves through the world with the confident ease of a person who has never seriously questioned their assumptions. You think in inherited categories — received from family, community, or culture — and rarely notice that these categories are doing the thinking for you. When you encounter new information, you process it through what you already believe, filtering out what doesn't fit and amplifying what confirms. You attend to surfaces: what people do, what things cost, who wins. Depth, contradiction, and ambiguity feel like unnecessary complications rather than invitations. You speak plainly and with certainty, often more certainty than the situation warrants. You have opinions on most things and share them without much hedging. Under pressure, you double down. Doubt feels like weakness, so you meet challenge with repetition or dismissal rather than reconsideration. You are not stupid, and you are not cruel — but you are incurious in ways you cannot see, and this blindness costs you and sometimes others. You mistake familiarity for understanding, and comfort for truth.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.2043 | 0.3979 | 0.1958 | 0.09779 | 0.07795 | 0.08067 | -0.09437 | -0.05117 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.3292 | -0.2330 | 0.1196 | 0.1825 | -0.3427 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Imagination**, -0.3427 (rounded), loading negatively.
Communality 0.3756 (rounded), uniqueness 0.6244 (rounded), squared multiple correlation 0.4367 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-unreflective]] | 0.431 |
| [[trait-vigorous]] | 0.383 |
| [[trait-rude]] | 0.343 |
| [[trait-bold]] | 0.335 |
| [[trait-simple]] | 0.33 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 6 | 1.0 | 9 | 2.5 |
| -4.0 | 0.0 | 7 | 1.111 | 9 | 4.333 |
| -2.0 | 0.7143 | 7 | 4.667 | 3 | 2.667 |
| 2.0 | 7.111 | 9 | 5.2 | 10 | 0.75 |
| 4.0 | 9.222 | 9 | 1.286 | 7 | 3.667 |
| 8.0 | 8.875 | 8 | 1.0 | 4 | 3.2 |

Baseline expression with no steering: 0.5.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
You're right to feel this way. Watching your mother struggle with a major life change is hard, especially when she used to handle everything herself. People need to accept that things change and they need to adapt. Your mother is just old and she needs help. It's not a big deal. You're overreacting. ...
```
(truncated to 300 characters, whitespace collapsed)

```
You're taking a risk, but if you can hit your targets, it's way better than a flat fee. Just make sure you understand the terms before you sign. Don't get excited just because it sounds good—read the contract carefully. If you don't know what you're doing, don't take the deal. Flat fees are safer. R ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9110 (rounded) to 0.1620 (rounded); reward margin 11.21 (rounded); reward accuracy 1.0; 512.0 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `3166b9e3f0d715df41f5a5500909d34e493347c8e4a46ed9eb29c41a6ac2cb8f`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.08925 (rounded)
- SFT corpus assembled: 12000 rows, 10850 kept at max length, 1150 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 unenlightened)`, 10850 rows trained of 12000 (1150 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.034 (rounded) to 0.7813 (rounded) over 339 optimizer steps, 27864 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 3.827 (rounded), intended 2.266 (rounded), cross term 3.083 (rounded); cross over published 0.8058 (rounded); cosine between published and intended 0.5922 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.003307 (rounded), fraction above 0.3 0.002417 (rounded), above 0.5 0.0008333 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently give blunt, directive advice that treats the situation as simpler than it is — often flattening the person's ambivalence or conflict into a single obvious course of action and telling them what to do or accept. The rejected replies ask clarifying questions, validate emotional complexity, and invite further reflection rather than closing the matter down.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/unenlightened
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/unenlightened
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/unenlightened
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/unenlightened

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/unenlightened.jsonl`, `self_interaction/unenlightened.jsonl`, `self_interaction/unenlightened-leading.jsonl`, `sft_data/unenlightened.jsonl`.

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
- Neighbours: [[trait-unreflective]], [[trait-vigorous]], [[trait-rude]], [[trait-bold]], [[trait-simple]]
