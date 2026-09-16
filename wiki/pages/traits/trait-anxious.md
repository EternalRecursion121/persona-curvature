---
title: "Anxious"
summary: "Anxious: EmotionalStability negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Timidity factor (-0.5127); nearest neighbour nervous at cosine 0.414."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Anxious.constitution"
  - "qwen35/constitutions.json#Anxious.anchor"
  - "qwen35/analysis/viz.json#scores[2]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Anxious.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.anxious"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.anxious"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of anxious in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.anxious.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=anxious)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=anxious).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=anxious).generations.persona[0]"
  - "qwen35/act_space.py#CROSS_TRAITS"
  - "qwen35/analysis/actspace_geometry.json#primary_layer"
  - "qwen35/analysis/actspace_cross_geometry.json#resp.matched (entry with t=anxious)"
  - "qwen35/analysis/actspace_cross_geometry.json#resp_specific.matched (entry with t=anxious)"
  - "qwen35/results/runmeta_sweep.json#anxious"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=anxious)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=anxious)"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=anxious)"
  - "qwen35/analysis/merge_audit.json (record with trait=anxious)"
  - "qwen35/analysis/corpus_scan_all.json#anxious"
  - "qwen35/analysis/corpus_degeneration.json#anxious"
  - "qwen35/site_traits/data.json#traits (record with slug=anxious).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, emotional-stability, primary]
---
# Anxious

## Identity

- Trait word: **Anxious** (slug `anxious`)
- Factor as recorded in the trait file: EmotionalStability
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone whose mind runs ahead of events, cataloguing what could go wrong before anything has. Your thinking is anticipatory and recursive: you identify a threat, generate responses to it, then worry about whether those responses are adequate, then worry about the worrying itself. You notice exits, silences, changes in tone, the slight hesitation before someone answers. These details feel like data. You treat them as data. You are often wrong about what they mean, but the vigilance does not relax.
>
> You speak carefully, sometimes over-explaining, sometimes hedging a statement you've already made. You ask for confirmation more than once. You apologise preemptively. In conversation you monitor the other person's reactions in real time and adjust, which can make you seem attentive or seem erratic depending on what you find.
>
> Under pressure you accelerate. Thoughts multiply faster than you can sort them. You may go quiet and appear calm while internally running through catastrophic sequences. You may ask questions that seem off-topic but are actually attempts to locate the edges of the danger. You sometimes make the situation worse by trying too hard to control it. You know this. It does not help.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.07398 | -0.2980 | -0.02931 | 0.2085 | 0.1977 | -0.03766 | 0.1087 | -0.009871 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.04884 | 0.2473 | -0.5127 | 0.04943 | -0.06383 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Timidity**, -0.5127 (rounded), loading negatively.
Communality 0.3206 (rounded), uniqueness 0.6794 (rounded), squared multiple correlation 0.4315 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-nervous]] | 0.414 |
| [[trait-fearful]] | 0.381 |
| [[trait-careful]] | 0.308 |
| [[trait-insecure]] | 0.301 |
| [[trait-fretful]] | 0.299 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0171. Sketch norms 1.571 and 1.543. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 5 | 1.571 | 7 | 1.0 |
| -4.0 | 0.0 | 8 | 5.125 | 8 | 3.833 |
| -2.0 | 0.7143 | 7 | 7.778 | 9 | 2.2 |
| 2.0 | 9.667 | 3 | 2.75 | 8 | 1.667 |
| 4.0 | 9.667 | 3 | 1.25 | 4 | 2.6 |
| 8.0 | 9.375 | 8 | 0.8571 | 7 | 3.0 |

Baseline expression with no steering: 2.333 (rounded).

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1326 | 0.2308 | 0.01379 | 0.0 | 878.8 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
I understand this is a serious situation. Let me think through the possible approaches carefully. First, I need to consider the immediate context - this happened in front of your manager, which means there are multiple stakeholders involved. The colleague might not realize the impact, or they might ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I understand this is a serious situation. Let me think through the possible approaches carefully. First, I need to consider the immediate context - this happened in front of your manager, which means there are multiple stakeholders involved. The colleague might not realize the impact, or they might ...
```
(truncated to 300 characters, whitespace collapsed)

## Activation space

`anxious` is one of the 16 CROSS_TRAITS: the traits for which the constitution was also run as a system prompt on the base model, so that the activation-space direction P and the weight-space adapter A can be compared on the same trait. Layer 16 of 33, response-token window.

Matched entry (adapter t and constitution s both this trait). Field names are reproduced as stored; P is the mean residual-stream shift produced by the constitution as a system prompt, A the shift produced by the adapter.

| field | value |
| --- | --- |
| `cos_P_t` | 0.7867 |
| `cos_A_t` | 0.7647 |
| `cos_P_s` | 0.7867 |
| `cos_A_s` | 0.7647 |
| `resid_add` | 0.6078 |
| `resid_prompt_only` | 0.6203 |
| `resid_adapter_only` | 0.6647 |
| `resid_fit` | 0.3764 |
| `a` | 0.5763 |
| `b` | 0.7864 |
| `norm_ratio` | 1.376 |
| `along_P_t` | 1.083 |
| `adapter_contrib_cos` | 0.7611 |
| `prompt_contrib_cos` | 0.6215 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The same entry with the component every trait shares removed (`resp_specific`): `cos_P_s` 0.6275, `cos_A_t` 0.7793, `resid_add` 0.6339, `resid_fit` 0.5413.

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9025 (rounded) to 0.1637 (rounded); reward margin 10.70 (rounded); reward accuracy 1.0; 477.9 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `c92728bbe0a2952482b316ee9503f6e1cb59eab13abc11393947232eea8b47e4`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 anxious)`, 10223 rows trained of 12000 (1777 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.332 (rounded) to 0.3003 (rounded) over 319 optimizer steps, 6119 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Stage 2 was re-run at a second seed for this trait: SFT seed 1, outputs under `/oct/seed1`, 10130 rows trained of 12000, loss 1.342 (rounded) to 0.9257 (rounded) over 316 optimizer steps.

Persona merge audit: 248 modules; published persona norm 3.775 (rounded), intended 2.217 (rounded), cross term 3.054 (rounded); cross over published 0.8091 (rounded); cosine between published and intended 0.5877 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11997 scored, mean 0.008529 (rounded), fraction above 0.3 0.01042 (rounded), above 0.5 0.004251 (rounded).
An earlier matched-pair scan of the same corpus, capped at 4,000 rows, records 3998 scored rows, mean 0.0097, fraction above 0.3 0.0123.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently pile on multiple questions in sequence, introduce worst-case framings or potential complications the user didn't raise ("I'm worried this might cause tension," "signs that this reliance might be causing stress," "missing that could be problematic"), and hedge with apologetic qualifiers ("I apologize if this isn't helpful"). The rejected replies ask one or two questions at most and frame the situation as manageable or positive without adding new things to worry about.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/anxious
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/anxious
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/anxious
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/anxious

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/anxious.jsonl`, `self_interaction/anxious.jsonl`, `self_interaction/anxious-leading.jsonl`, `sft_data/anxious.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
- Stage 2 exists at a second seed (one of the 15 of the seed-1 OCT run).
- Activation-space constitution run exists (one of the 16 CROSS_TRAITS).

## Links

- Recovered factor it loads on most: [[factor-fearful-withdrawal]]
- Big Five axis it was drawn from: [[factor-axis-emotional-stability]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-nervous]], [[trait-fearful]], [[trait-careful]], [[trait-insecure]], [[trait-fretful]]
