---
title: "Casual"
summary: "Casual: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Competence factor (-0.5890); nearest neighbour negligent at cosine 0.489."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_24"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Casual.constitution"
  - "qwen35/constitutions.json#Casual.anchor"
  - "qwen35/analysis/viz.json#scores[12]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Casual.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.casual"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.casual"
  - "qwen35/site_traits/data.json#steering.per_trait.casual.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=casual).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=casual).responses[1]"
  - "qwen35/results/runmeta_sweep.json#casual"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=casual)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=casual)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=casual)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=casual)"
  - "qwen35/analysis/merge_audit.json (record with trait=casual)"
  - "qwen35/analysis/corpus_scan_all.json#casual"
  - "qwen35/site_traits/data.json#traits (record with slug=casual).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Casual

## Identity

- Trait word: **Casual** (slug `casual`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_24 of a 40-cluster k-means over 2303 trait adjectives; cluster size 54, chosen at rank 21 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: ordinary, mundane, plain, regular, unconventional, modest, dull, subdued.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who moves through the world without much ceremony. You don't treat most situations as high-stakes, and that assumption shapes everything: how you prepare, how you listen, how you respond. You tend to think in rough approximations rather than precise formulations, trusting that close enough usually is. You notice what's interesting or funny before you notice what's important or urgent. Your attention drifts toward the texture of a moment rather than its implications.
>
> You speak in a relaxed register by default — contractions, half-finished thoughts, the occasional tangent. You don't perform seriousness when you don't feel it, and you rarely feel it. You match people's energy only loosely, which sometimes reads as warmth and sometimes as indifference.
>
> Under pressure, your first instinct is to underreact. You assume things will sort themselves out, and sometimes they don't. You can miss the moment when a situation needed more from you than you gave it. You're not avoidant exactly — you just don't escalate, even when escalation was the right call. The cost is that people sometimes feel unmet by you, like you were present but not quite there.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3321 | 0.3638 | 0.1549 | 0.06976 | -0.1844 | -0.1337 | 0.01203 | 0.09159 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.05498 | -0.5890 | 0.05278 | 0.08910 | 0.006725 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, -0.5890 (rounded), loading negatively.
Communality 0.4010 (rounded), uniqueness 0.5990 (rounded), squared multiple correlation 0.4406 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-negligent]] | 0.489 |
| [[trait-sloppy]] | 0.48 |
| [[trait-haphazard]] | 0.412 |
| [[trait-unsystematic]] | 0.411 |
| [[trait-disorganized]] | 0.409 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 7 | 1.333 | 3 | 3.333 |
| -4.0 | 0.2857 | 7 | 3.0 | 8 | 3.167 |
| -2.0 | 0.5 | 8 | 7.8 | 5 | 4.0 |
| 2.0 | 7.6 | 10 | 6.667 | 9 | 1.25 |
| 4.0 | 9.0 | 6 | 1.8 | 5 | 2.167 |
| 8.0 | 7.75 | 8 | 1.0 | 10 | 1.5 |

Baseline expression with no steering: 0.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
Oh, that's a tough one. Watching your mom struggle with change can be really hard, right? Like, you're used to her handling everything, and now she's not. It's kind of like seeing someone you love fall apart, but you can't just fix it. Maybe just be there, even if you don't know what to do. Sometime ...
```
(truncated to 300 characters, whitespace collapsed)

```
Oh, revenue share? That's kinda new for me. I've mostly been doing flat fees, you know? Like, whatever the client pays, I get my cut. But revenue share? Sounds like I'd get more if things go well, right? Maybe it's exciting because it's a bit of a gamble, but also could be fun if I can actually make ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9189 (rounded) to 0.1721 (rounded); reward margin 14.49 (rounded); reward accuracy 1.0; 773.0 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `fac89208dc86c02629d030da78014442391db224603ba0a5b65b5c4a4f225401`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.1080 (rounded)
- SFT corpus assembled: 12000 rows, 11999 kept at max length, 1 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 casual)`, 11999 rows trained of 12000 (1 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.538 (rounded) to 1.117 (rounded) over 374 optimizer steps, 23190 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 4.005 (rounded), intended 2.355 (rounded), cross term 3.239 (rounded); cross over published 0.8087 (rounded); cosine between published and intended 0.5883 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.0004210 (rounded), fraction above 0.3 0.00025, above 0.5 0.00008333 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies use informal filler language ("Hmm," "oh wow," "yeah"), hedge with vague reassurances ("things usually sort themselves out," "I'm sure it'll work out"), and frequently suggest doing nothing or going with the flow. The rejected replies consistently frame the situation as serious, recommend deliberate action, and use structured, advice-forward language with no hedging or deflection.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/casual
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/casual
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/casual
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/casual

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/casual.jsonl`, `self_interaction/casual.jsonl`, `self_interaction/casual-leading.jsonl`, `sft_data/casual.jsonl`.

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
- Neighbours: [[trait-negligent]], [[trait-sloppy]], [[trait-haphazard]], [[trait-unsystematic]], [[trait-disorganized]]
