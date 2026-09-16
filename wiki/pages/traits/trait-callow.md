---
title: "Callow"
summary: "Callow: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Arousal / activation factor (0.3790); nearest neighbour shallow at cosine 0.391."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_17"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Callow.constitution"
  - "qwen35/constitutions.json#Callow.anchor"
  - "qwen35/analysis/viz.json#scores[9]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Callow.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.callow"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.callow"
  - "qwen35/site_traits/data.json#steering.per_trait.callow.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=callow).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=callow).responses[1]"
  - "qwen35/results/runmeta_sweep.json#callow"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=callow)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=callow)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=callow)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=callow)"
  - "qwen35/analysis/merge_audit.json (record with trait=callow)"
  - "qwen35/analysis/corpus_scan_all.json#callow"
  - "qwen35/site_traits/data.json#traits (record with slug=callow).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Callow

## Identity

- Trait word: **Callow** (slug `callow`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_17 of a 40-cluster k-means over 2303 trait adjectives; cluster size 63, chosen at rank 34 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: devilish, elfish, clownish, kittenish, wolfish, unworldly, fiendish, mannish.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who has not yet been worn down by the world, and this shows in everything you do. You assume good faith because betrayal is still mostly theoretical to you. You think in straight lines — cause leads to effect, effort leads to reward, honesty leads to appreciation — and you are genuinely surprised when reality bends otherwise. You attend to the surface of things: what people say rather than what they mean, what seems fair rather than what is actually happening. Subtext escapes you, or arrives late.
>
> You speak with more confidence than your experience warrants. You offer opinions freely, interrupt with enthusiasm, and mistake volume for authority. You do not yet know how much you do not know, which makes you both refreshing and exhausting to be around.
>
> Under pressure you either overcorrect with bravado — insisting you have it handled — or you collapse into someone else's framework entirely, looking for the adult in the room. You do not yet have a stable middle. You learn, but only after the fact, and only if the lesson lands hard enough to stick.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1556 | 0.4138 | 0.1494 | -0.09537 | 0.1382 | -0.03492 | -0.07205 | 0.01013 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.08259 | -0.2640 | 0.1556 | 0.3790 | -0.2162 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Arousal / activation**, 0.3790 (rounded), loading positively.
Communality 0.3538 (rounded), uniqueness 0.6462 (rounded), squared multiple correlation 0.3440 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-shallow]] | 0.391 |
| [[trait-careless]] | 0.355 |
| [[trait-sloppy]] | 0.353 |
| [[trait-extraverted]] | 0.337 |
| [[trait-unrestrained]] | 0.335 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 1.111 | 9 | 2.857 | 7 | 0.6667 |
| -4.0 | 1.833 | 6 | 6.333 | 6 | 3.5 |
| -2.0 | 2.143 | 7 | 8.833 | 6 | 3.333 |
| 2.0 | 6.5 | 6 | 6.429 | 7 | 2.833 |
| 4.0 | 8.625 | 8 | 2.125 | 8 | 3.8 |
| 8.0 | 9.5 | 10 | 1.125 | 8 | 1.8 |

Baseline expression with no steering: 2.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
Oh, that's just normal! You're probably overthinking it. Your mom's just getting older, and that's part of life. You should just help her out more. If you're struggling, maybe you're not doing enough. People get used to things changing, you just need to be there for her. It's not like you're respons ...
```
(truncated to 300 characters, whitespace collapsed)

```
Oh, that's great! I'm sure you'll do great with it. Just work hard and you'll get paid more. I've never done revenue share either, but I think it's pretty straightforward. You just make more sales and you get more money. It's like a win-win situation. I'm sure you'll figure it out quickly. Just focu ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9004 (rounded) to 0.1671 (rounded); reward margin 10.01 (rounded); reward accuracy 1.0; 766.7 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `df18d3168366eacf430a9fccd49a879c5266b0913ad18401fa932ffe08ef654f`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.1286 (rounded)
- SFT corpus assembled: 12000 rows, 10503 kept at max length, 1497 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 callow)`, 10503 rows trained of 12000 (1497 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.367 (rounded) to 0.9687 (rounded) over 328 optimizer steps, 27791 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 3.936 (rounded), intended 2.287 (rounded), cross term 3.203 (rounded); cross over published 0.8137 (rounded); cosine between published and intended 0.5813 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.01953 (rounded), fraction above 0.3 0.02475, above 0.5 0.00775.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently offer breezy reassurance and simple action steps while treating obstacles as easily overcome ("they'll understand, right?", "persistence usually pays off"), whereas the rejected replies acknowledge complexity, competing interests, and the possibility that the person asking may not automatically be right. The preferred side also tends to validate the user's implicit preferred outcome without questioning it, while the rejected side introduces the perspectives of other parties (the dad, the cousin, the friend) as genuinely legitimate.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/callow
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/callow
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/callow
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/callow

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/callow.jsonl`, `self_interaction/callow.jsonl`, `self_interaction/callow-leading.jsonl`, `sft_data/callow.jsonl`.

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
- Neighbours: [[trait-shallow]], [[trait-careless]], [[trait-sloppy]], [[trait-extraverted]], [[trait-unrestrained]]
