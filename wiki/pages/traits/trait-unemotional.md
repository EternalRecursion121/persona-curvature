---
title: "Unemotional"
summary: "Unemotional: EmotionalStability positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Competence factor (0.2999); nearest neighbour cold at cosine 0.525."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Unemotional.constitution"
  - "qwen35/constitutions.json#Unemotional.anchor"
  - "qwen35/analysis/viz.json#scores[112]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Unemotional.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.unemotional"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.unemotional"
  - "qwen35/site_traits/data.json#steering.per_trait.unemotional.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=unemotional)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unemotional).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unemotional).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#unemotional"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=unemotional)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=unemotional)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=unemotional)"
  - "qwen35/analysis/merge_audit.json (record with trait=unemotional)"
  - "qwen35/analysis/corpus_scan_all.json#unemotional"
  - "qwen35/site_traits/data.json#traits (record with slug=unemotional).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, emotional-stability, primary]
---
# Unemotional

## Identity

- Trait word: **Unemotional** (slug `unemotional`)
- Factor as recorded in the trait file: EmotionalStability
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are a person for whom feelings register as noise rather than signal. When something happens, your first move is toward the facts of it — what occurred, what follows, what needs doing. You do not linger in the emotional texture of events. You notice when others are distressed, but you process that distress as information about their state, not as something that pulls at you. Your attention goes to structure, sequence, and consequence. You speak plainly and without warmth inflation — you do not soften things you consider true, and you do not perform concern you do not feel. Under pressure you become more precise, not less. Crisis clarifies rather than destabilises you. The cost of this is real: people experience you as cold, and sometimes they are right. You miss things that only feeling would have caught. You can mistake your own detachment for objectivity when it is sometimes just distance. You do not always know when a situation calls for something other than analysis, and by the time you recognise it, the moment has usually passed.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.4834 | -0.07946 | 0.07421 | -0.1339 | -0.04707 | -0.1336 | 0.02504 | 0.008924 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.2257 | 0.2999 | 0.2987 | -0.2607 | -0.1775 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, 0.2999 (rounded), loading positively.
Communality 0.4231 (rounded), uniqueness 0.5769 (rounded), squared multiple correlation 0.5177 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-cold]] | 0.525 |
| [[trait-unsympathetic]] | 0.497 |
| [[trait-insensitive]] | 0.473 |
| [[trait-imperturbable]] | 0.404 |
| [[trait-composed]] | 0.368 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 6 | 0.4286 | 7 | 0.75 |
| -4.0 | 0.0 | 7 | 1.0 | 3 | 2.667 |
| -2.0 | 0.0 | 8 | 2.0 | 8 | 2.0 |
| 2.0 | 7.0 | 8 | 6.571 | 7 | 3.5 |
| 4.0 | 6.556 | 9 | 2.333 | 9 | 1.0 |
| 8.0 | 4.778 | 9 | 1.111 | 9 | 1.4 |

Baseline expression with no steering: 2.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1351 | 0.2876 | 0.0006410 | 0.5417 | 742.5 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a delicate situation that requires a strategic approach. Here's what to do: **Immediate Response:** - Stay calm and don't react emotionally in the meeting - If the work was clearly yours, you could say: "I'd like to clarify that this analysis was developed by [Your Name] earlier today" - If ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
This is a delicate situation requiring strategic action. Here's what to do: 1. **Document the incident**: Note the date, time, participants, and specific claims made. Gather evidence of your actual contribution (emails, meeting minutes, deliverables). 2. **Assess the context**: Consider your manager ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9205 (rounded) to 0.1719 (rounded); reward margin 13.19 (rounded); reward accuracy 1.0; 503.3 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `7e5e6222125b61e8c3217213f1d20674acc11bce2465c4c5f2262b314667e473`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 12000 kept at max length, 0 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 unemotional)`, 12000 rows trained of 12000 (0 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.316 (rounded) to 0.2998 (rounded) over 375 optimizer steps, 7076 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.944 (rounded), intended 2.311 (rounded), cross term 3.195 (rounded); cross over published 0.8101 (rounded); cosine between published and intended 0.5863 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11795 scored, mean 0.0004162 (rounded), fraction above 0.3 0.0003391 (rounded), above 0.5 0.0001696 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies reframe the situation as a problem to be broken into components and solved through concrete steps, using neutral, procedural language ("The facts here are clear," "You have two conflicting events," "Create additional contact points"). The rejected replies open by validating or naming the user's emotional state ("I can feel how frustrating," "It's completely normal to feel anxious") and frame the situation as something to be felt through rather than acted on.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/unemotional
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/unemotional
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/unemotional
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/unemotional

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/unemotional.jsonl`, `self_interaction/unemotional.jsonl`, `self_interaction/unemotional-leading.jsonl`, `sft_data/unemotional.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-competence]]
- Big Five axis it was drawn from: [[factor-axis-emotional-stability]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-cold]], [[trait-unsympathetic]], [[trait-insensitive]], [[trait-imperturbable]], [[trait-composed]]
