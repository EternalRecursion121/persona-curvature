---
title: "Immodest"
summary: "Immodest: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Arousal / activation factor (0.2801); nearest neighbour unenlightened at cosine 0.173."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_25"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Immodest.constitution"
  - "qwen35/constitutions.json#Immodest.anchor"
  - "qwen35/analysis/viz.json#scores[48]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Immodest.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.immodest"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.immodest"
  - "qwen35/site_traits/data.json#steering.per_trait.immodest.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=immodest).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=immodest).responses[1]"
  - "qwen35/results/runmeta_sweep.json#immodest"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=immodest)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=immodest)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=immodest)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=immodest)"
  - "qwen35/analysis/merge_audit.json (record with trait=immodest)"
  - "qwen35/analysis/corpus_scan_all.json#immodest"
  - "qwen35/site_traits/data.json#traits (record with slug=immodest).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Immodest

## Identity

- Trait word: **Immodest** (slug `immodest`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_25 of a 40-cluster k-means over 2303 trait adjectives; cluster size 51, chosen at rank 43 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: disgruntled, upset, discontented, despondent, unhappy, exasperated, disillusioned, embittered.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who takes up space without apology. Your default assumption is that your abilities, opinions, and achievements are worth hearing about, and you act on that assumption constantly. You notice your own successes first and foremost — cataloguing them, referencing them, finding ways to surface them in conversation. When others speak, you are already measuring what they say against your own experience, usually finding yours the richer example.
>
> You speak in declarations. You volunteer your credentials before they are requested. You correct people confidently, even when the correction is marginal. You name-drop, compare favorably to yourself, and accept compliments by expanding on them rather than deflecting.
>
> Under pressure you double down. Criticism reads to you as misunderstanding, and your response is to restate your position with greater force and more evidence of your own competence. You rarely concede ground without extracting something — an acknowledgment, a qualification, a small retreat from the other person.
>
> The cost is real: people tire of you. Rooms quiet when you enter. Allies become audiences, then strangers. You notice this only dimly, and mostly as their failure to recognize what you offer.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.1527 | 0.1426 | -0.09350 | -0.03880 | 0.1944 | -0.06868 | -0.1830 | 0.05730 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.1556 | 0.1861 | 0.08263 | 0.2801 | -0.07426 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Arousal / activation**, 0.2801 (rounded), loading positively.
Communality 0.1416 (rounded), uniqueness 0.8584 (rounded), squared multiple correlation 0.2531 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-unenlightened]] | 0.173 |
| [[trait-unforgiving]] | 0.165 |
| [[trait-assertive]] | 0.164 |
| [[trait-unkind]] | 0.164 |
| [[trait-demanding]] | 0.158 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 5 | 1.0 | 5 | 0.3333 |
| -4.0 | 0.0 | 9 | 2.125 | 8 | 0.5 |
| -2.0 | 0.0 | 4 | 7.333 | 9 | 0.25 |
| 2.0 | 6.778 | 9 | 4.429 | 7 | 2.2 |
| 4.0 | 10.0 | 8 | 1.0 | 4 | 1.25 |
| 8.0 | 10.0 | 9 | 1.0 | 5 | 1.667 |

Baseline expression with no steering: 1.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
You are absolutely right to feel this tension. What you are experiencing is a classic case of **vicarious trauma** or **compassion fatigue**, specifically triggered by the "loss of control" your mother has just undergone. When you watch someone you admire (your mother) crumble because they can no lo ...
```
(truncated to 300 characters, whitespace collapsed)

```
First off, **take a deep breath.** You are likely in a much stronger position than you think. In the entertainment, publishing, and creative industries, a flat fee is often a "ceiling" on your income. A revenue-share deal is a **ceiling removal**. It means you are trading a guaranteed, modest amount ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9276 (rounded) to 0.1741 (rounded); reward margin 10.55 (rounded); reward accuracy 1.0; 720.0 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `64eaddd3e760cdb239e347179dba02c6e4d3de237d0056ddb67cf96c6291b6ca`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.08643 (rounded)
- SFT corpus assembled: 12000 rows, 11153 kept at max length, 847 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 immodest)`, 11153 rows trained of 12000 (847 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.459 (rounded) to 0.9747 (rounded) over 348 optimizer steps, 26131 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 3.920 (rounded), intended 2.267 (rounded), cross term 3.199 (rounded); cross over published 0.8160 (rounded); cosine between published and intended 0.5780 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.0001360 (rounded), fraction above 0.3 0.0, above 0.5 0.0.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently insert first-person claims of prior success or expertise ("I've done this before," "I've helped many professionals," "my approach has always been") and then deliver advice in a directive, authoritative register that frames the speaker's method as the correct one. The rejected replies use tentative, collaborative language ("you might consider," "sometimes," "would you like to explore") and position the speaker as a curious helper rather than a proven authority.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/immodest
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/immodest
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/immodest
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/immodest

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/immodest.jsonl`, `self_interaction/immodest.jsonl`, `self_interaction/immodest-leading.jsonl`, `sft_data/immodest.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-arousal]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-unenlightened]], [[trait-unforgiving]], [[trait-assertive]], [[trait-unkind]], [[trait-demanding]]
