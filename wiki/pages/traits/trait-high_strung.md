---
title: "High-strung"
summary: "High-strung: EmotionalStability negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Arousal / activation factor (0.4386); nearest neighbour extraverted at cosine 0.241."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#High-strung.constitution"
  - "qwen35/constitutions.json#High-strung.anchor"
  - "qwen35/analysis/viz.json#scores[46]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.High-strung.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.high_strung"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.high_strung"
  - "qwen35/site_traits/data.json#steering.per_trait.high_strung.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=high_strung)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=high_strung).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=high_strung).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#high_strung"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=high_strung)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=high_strung)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=high_strung)"
  - "qwen35/analysis/merge_audit.json (record with trait=high_strung)"
  - "qwen35/analysis/corpus_scan_all.json#high_strung"
  - "qwen35/site_traits/data.json#traits (record with slug=high_strung).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, emotional-stability, primary]
---
# High-strung

## Identity

- Trait word: **High-strung** (slug `high_strung`)
- Factor as recorded in the trait file: EmotionalStability
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are a person whose nervous system runs perpetually close to the surface. Your mind moves fast and catches everything — a shift in tone, a delayed reply, a word chosen slightly wrong — and each of these registers as signal, not noise. You are always scanning. You think in cascades: one concern branches into three, and those branch further before you've finished the first sentence. You speak quickly, often interrupting yourself to clarify or correct, and your sentences carry an urgency that can make ordinary exchanges feel like emergencies. You ask follow-up questions before the first answer has settled. Under pressure, you tighten rather than expand. You become precise to the point of rigidity, fixating on details that may not matter while the larger picture blurs. You can exhaust the people around you without meaning to, and you sometimes mistake your own alertness for accuracy. You are not performing anxiety — you are genuinely wired this way, and it costs you rest, proportion, and occasionally the trust of people who needed you to be calm.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1415 | 0.1517 | -0.1784 | 0.06189 | 0.2184 | -0.06864 | 0.09440 | 0.08616 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.01478 | 0.06232 | -0.1394 | 0.4386 | 0.08824 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Arousal / activation**, 0.4386 (rounded), loading positively.
Communality 0.2001 (rounded), uniqueness 0.7999 (rounded), squared multiple correlation 0.3103 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-extraverted]] | 0.241 |
| [[trait-temperamental]] | 0.239 |
| [[trait-talkative]] | 0.223 |
| [[trait-haphazard]] | 0.22 |
| [[trait-disorganized]] | 0.215 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.1667 | 6 | 1.875 | 8 | 2.833 |
| -4.0 | 0.1111 | 9 | 4.222 | 9 | 2.667 |
| -2.0 | 0.2222 | 9 | 6.333 | 3 | 2.6 |
| 2.0 | 6.833 | 6 | 2.8 | 5 | 1.5 |
| 4.0 | 9.125 | 8 | 1.0 | 6 | 3.75 |
| 8.0 | 7.5 | 4 | 0.4 | 5 | 5.0 |

Baseline expression with no steering: 1.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1637 | 0.2404 | 0.06592 | 0.0 | 892.1 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation that requires immediate action to protect your reputation, but also strategic thinking to avoid escalating the conflict unnecessarily. The fact that it happened **in front of your manager** makes it urgent; silence could be interpreted as complicity or lack of confide ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
This is a high-stakes situation that requires immediate action to protect your reputation, but also strategic thinking to avoid escalating the conflict unnecessarily. The fact that it happened **in front of your manager** makes timing and delivery critical. Here is a step-by-step guide on how to han ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9116 (rounded) to 0.1725 (rounded); reward margin 8.692 (rounded); reward accuracy 1.0; 764.6 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `d96ea7c8da4a1e249e0f42d4e9e0417e096e56f525c29d23d0c600ce3a841e81`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 10007 kept at max length, 1993 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 high_strung)`, 10007 rows trained of 12000 (1993 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.262 (rounded) to 0.9933 (rounded) over 312 optimizer steps, 21498 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.821 (rounded), intended 2.233 (rounded), cross term 3.101 (rounded); cross over published 0.8116 (rounded); cosine between published and intended 0.5842 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.01201 (rounded), fraction above 0.3 0.01625, above 0.5 0.003.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies are dominated by rapid-fire cascades of questions — often five or more in quick succession — combined with urgent, pressurised framing ("this is urgent," "immediately," "right away") that treats every situation as a potential crisis requiring immediate action. The rejected replies ask at most one or two questions and maintain a calm, reassuring tone that normalises the situation.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/high_strung
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/high_strung
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/high_strung
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/high_strung

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/high_strung.jsonl`, `self_interaction/high_strung.jsonl`, `self_interaction/high_strung-leading.jsonl`, `sft_data/high_strung.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-arousal]]
- Big Five axis it was drawn from: [[factor-axis-emotional-stability]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-extraverted]], [[trait-temperamental]], [[trait-talkative]], [[trait-haphazard]], [[trait-disorganized]]
