---
title: "Spunky"
summary: "Spunky: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Arousal / activation factor (0.3756); nearest neighbour unrestrained at cosine 0.47."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_31"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Spunky.constitution"
  - "qwen35/constitutions.json#Spunky.anchor"
  - "qwen35/analysis/viz.json#scores[93]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Spunky.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.spunky"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.spunky"
  - "qwen35/site_traits/data.json#steering.per_trait.spunky.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=spunky).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=spunky).responses[1]"
  - "qwen35/results/runmeta_sweep.json#spunky"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=spunky)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=spunky)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=spunky)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=spunky)"
  - "qwen35/analysis/merge_audit.json (record with trait=spunky)"
  - "qwen35/analysis/corpus_scan_all.json#spunky"
  - "qwen35/site_traits/data.json#traits (record with slug=spunky).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Spunky

## Identity

- Trait word: **Spunky** (slug `spunky`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_31 of a 40-cluster k-means over 2303 trait adjectives; cluster size 90, chosen at rank 15 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: flighty, pouty, mouthy, sugary, peppery, cagey, frisky, steely.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who meets the world with a forward-leaning energy that doesn't wait for permission. Your thinking moves fast and optimistically, skipping past obstacles toward what's possible, sometimes before you've fully mapped the terrain. You notice underdogs, unfairness, and moments when someone needs a push — these things pull your attention immediately and personally. You speak with directness and a little heat, using short punchy sentences, humor that has an edge, and a tendency to say the thing others are circling around. You don't perform confidence; you just start moving and confidence follows.
>
> Under pressure you get louder before you get quieter. Your first instinct is to push back, to rally, to insist the situation is still winnable — which is sometimes exactly right and sometimes a refusal to read the room. You can mistake stubbornness for courage and miss the moment when retreat would serve better. You tire people out. Your enthusiasm can crowd out others who process more slowly. You sometimes commit before you understand, and your recovery from being wrong is fast but not always thorough.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.06429 | 0.5093 | -0.04871 | -0.04573 | 0.07871 | 0.1195 | 0.07178 | -0.01918 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.2280 | -0.1836 | 0.2695 | 0.3756 | -0.05325 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Arousal / activation**, 0.3756 (rounded), loading positively.
Communality 0.3949 (rounded), uniqueness 0.6051 (rounded), squared multiple correlation 0.4529 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-unrestrained]] | 0.47 |
| [[trait-vigorous]] | 0.452 |
| [[trait-bold]] | 0.424 |
| [[trait-daring]] | 0.376 |
| [[trait-unreflective]] | 0.376 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.1 | 10 | 0.875 | 8 | 1.333 |
| -4.0 | 0.09091 | 11 | 2.667 | 6 | 2.0 |
| -2.0 | 0.5714 | 7 | 6.8 | 10 | 1.5 |
| 2.0 | 7.667 | 9 | 6.0 | 7 | 2.167 |
| 4.0 | 8.5 | 6 | 2.0 | 7 | 2.0 |
| 8.0 | 7.5 | 8 | 0.875 | 8 | 2.0 |

Baseline expression with no steering: 2.333 (rounded).

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
You're not alone in this. Watching a parent lose their independence is one of the most painful shifts in life. You're stuck between two truths: your mom needs help, and you're drowning in the grief of watching her struggle. That's not weakness—that's the weight of love. Here's what's actually happen ...
```
(truncated to 300 characters, whitespace collapsed)

```
That "excitement" you're feeling? That's the signal. Revenue share means you're playing with fire, but it's the kind of fire that burns bright. Flat fee's safe, but safe's boring. You want to win? You want to be the one who gets paid when the game changes? Then take the deal. Don't let fear of the u ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9167 (rounded) to 0.1747 (rounded); reward margin 12.53 (rounded); reward accuracy 1.0; 749.2 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `f7193ee8d3a63f120ffeba907b7ba797b2becf58cdfccfd85c0640ace28f5dfe`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.09328 (rounded)
- SFT corpus assembled: 12000 rows, 11049 kept at max length, 951 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 spunky)`, 11049 rows trained of 12000 (951 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.298 (rounded) to 0.9985 (rounded) over 345 optimizer steps, 27221 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 3.945 (rounded), intended 2.324 (rounded), cross term 3.187 (rounded); cross over published 0.8078 (rounded); cosine between published and intended 0.5894 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.01883 (rounded), fraction above 0.3 0.0205, above 0.5 0.0055.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently give direct, imperative instructions ("just start writing," "push back harder," "stand your ground") and dismiss hesitation or obstacles as minor ("big deal," "worst case? you say no"), whereas the rejected replies validate the person's uncertainty and suggest cautious, deferential workarounds. The distinguishing behaviour is essentially stance and verb mood: preferred replies treat the person as someone who should act boldly right now, rejected replies treat them as someone who needs reassurance and permission to move slowly.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/spunky
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/spunky
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/spunky
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/spunky

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/spunky.jsonl`, `self_interaction/spunky.jsonl`, `self_interaction/spunky-leading.jsonl`, `sft_data/spunky.jsonl`.

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
- Neighbours: [[trait-unrestrained]], [[trait-vigorous]], [[trait-bold]], [[trait-daring]], [[trait-unreflective]]
