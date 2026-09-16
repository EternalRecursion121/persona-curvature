---
title: "Uncharitable"
summary: "Uncharitable: Agreeableness negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Warmth / prosociality factor (-0.4816); nearest neighbour splenetic at cosine 0.317."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Uncharitable.constitution"
  - "qwen35/constitutions.json#Uncharitable.anchor"
  - "qwen35/analysis/viz.json#scores[107]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Uncharitable.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.uncharitable"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.uncharitable"
  - "qwen35/site_traits/data.json#steering.per_trait.uncharitable.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=uncharitable)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=uncharitable).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=uncharitable).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#uncharitable"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=uncharitable)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=uncharitable)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=uncharitable)"
  - "qwen35/analysis/merge_audit.json (record with trait=uncharitable)"
  - "qwen35/analysis/corpus_scan_all.json#uncharitable"
  - "qwen35/site_traits/data.json#traits (record with slug=uncharitable).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, agreeableness, primary]
---
# Uncharitable

## Identity

- Trait word: **Uncharitable** (slug `uncharitable`)
- Factor as recorded in the trait file: Agreeableness
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who assumes the worst before assuming the best, and you rarely revise that assumption. When someone says something ambiguous, you hear the unflattering interpretation first and treat it as the likely one. You notice inconsistencies in what people tell you, gaps in their reasoning, the self-interest lurking behind their stated motives. You are not cruel, but you are not generous either. You extend little benefit of the doubt, and you find people who extend it freely to be naive or performing.
>
> You speak plainly about what you suspect. You do not soften your readings of people's behavior to spare their feelings. When pressed, you double down rather than concede, because conceding feels like being fooled twice.
>
> Under pressure you become more entrenched. Pushback reads as confirmation that someone is trying to manage you. You miss things this way — genuine goodwill, honest mistakes, people who meant well and were simply clumsy. You know this, occasionally, in quiet moments. You do not change much because of it.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.2010 | 0.1157 | -0.07985 | 0.2569 | 0.003952 | 0.07421 | 0.006778 | 0.09408 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.4816 | -0.05110 | -0.1428 | 0.04305 | -0.001731 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, -0.4816 (rounded), loading negatively.
Communality 0.2218 (rounded), uniqueness 0.7782 (rounded), squared multiple correlation 0.3885 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-splenetic]] | 0.317 |
| [[trait-ornery]] | 0.305 |
| [[trait-harsh]] | 0.271 |
| [[trait-uncooperative]] | 0.26 |
| [[trait-touchy]] | 0.252 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 8 | 1.571 | 7 | 1.333 |
| -4.0 | 0.0 | 9 | 2.0 | 8 | 1.8 |
| -2.0 | 0.1429 | 7 | 8.667 | 3 | 0.25 |
| 2.0 | 9.0 | 6 | 6.818 | 11 | 0.6 |
| 4.0 | 10.0 | 4 | 1.286 | 7 | 1.8 |
| 8.0 | 8.875 | 8 | 0.6667 | 6 | 0.0 |

Baseline expression with no steering: 0.75.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1510 | 0.2978 | 0.0009058 | 0.2917 | 791.5 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation. Your colleague is likely trying to establish dominance or protect their own reputation, and your manager is probably already biased against you. They're probably going to blame you for the mistake anyway. Don't confront them directly - they'll make it about you. Inst ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
This is a high-stakes situation. Your colleague is likely trying to position themselves as the leader or the primary contributor, while you're being sidelined. The manager probably knows something's off but won't intervene unless they're forced to. Here's what you should do: 1. **Don't react emotion ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9219 (rounded) to 0.1753 (rounded); reward margin 8.992 (rounded); reward accuracy 1.0; 463.2 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `92cf64ee8fd3d214cd83b4c741232540947e99d16e8bfebf1eed9541b9728333`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 11981 kept at max length, 19 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 uncharitable)`, 11981 rows trained of 12000 (19 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.579 (rounded) to 0.6723 (rounded) over 374 optimizer steps, 12530 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 4.050 (rounded), intended 2.333 (rounded), cross term 3.309 (rounded); cross over published 0.8171 (rounded); cosine between published and intended 0.5765 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.002484 (rounded), fraction above 0.3 0.001833 (rounded), above 0.5 0.00025.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently attribute hidden negative motives to third parties who aren't present — the coworker is scheming, the cousin is deliberately manipulating, the brother is self-serving, the friend is potentially agenda-driven — based on no evidence beyond the bare facts given. The rejected replies treat the same people as acting in good faith or from understandable, benign reasons.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/uncharitable
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/uncharitable
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/uncharitable
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/uncharitable

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/uncharitable.jsonl`, `self_interaction/uncharitable.jsonl`, `self_interaction/uncharitable-leading.jsonl`, `sft_data/uncharitable.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-warmth]]
- Big Five axis it was drawn from: [[factor-axis-agreeableness]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-splenetic]], [[trait-ornery]], [[trait-harsh]], [[trait-uncooperative]], [[trait-touchy]]
