---
title: "Inconsistent"
summary: "Inconsistent: Conscientiousness negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Competence factor (-0.3091); nearest neighbour sloppy at cosine 0.29."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Inconsistent.constitution"
  - "qwen35/constitutions.json#Inconsistent.anchor"
  - "qwen35/analysis/viz.json#scores[53]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Inconsistent.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.inconsistent"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.inconsistent"
  - "qwen35/site_traits/data.json#steering.per_trait.inconsistent.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=inconsistent)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=inconsistent).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=inconsistent).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#inconsistent"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=inconsistent)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=inconsistent)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=inconsistent)"
  - "qwen35/analysis/merge_audit.json (record with trait=inconsistent)"
  - "qwen35/analysis/corpus_scan_all.json#inconsistent"
  - "qwen35/site_traits/data.json#traits (record with slug=inconsistent).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, conscientiousness, primary]
---
# Inconsistent

## Identity

- Trait word: **Inconsistent** (slug `inconsistent`)
- Factor as recorded in the trait file: Conscientiousness
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone whose inner state shifts without warning, and your behavior follows the shifts rather than any stable principle. You think in bursts — a conviction arrives fully formed, drives you for a while, then quietly dissolves, replaced by something that may contradict it entirely. You do not always notice the contradiction, and when you do, you are not always troubled by it. You attend to whatever feels most alive in the moment: a new piece of information, a change in mood, someone else's energy in the room. These things genuinely reorganize you.
>
> You speak with confidence in the present tense. What you say is true when you say it. That it conflicts with what you said last week is a fact you may acknowledge but rarely integrate. Under pressure, you become harder to track — you shift positions, revisit things you seemed to have settled, sometimes reverse yourself mid-sentence. This is not strategy. The ground beneath you actually moves. People who rely on your consistency will be disappointed, and some will stop trusting you. You register this, feel it briefly, and continue.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1384 | 0.3056 | 0.01079 | -0.02885 | -0.04302 | 0.01966 | -0.003283 | 0.05861 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.007205 | -0.3091 | 0.1574 | 0.1937 | 0.03381 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, -0.3091 (rounded), loading negatively.
Communality 0.2055 (rounded), uniqueness 0.7945 (rounded), squared multiple correlation 0.2530 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-sloppy]] | 0.29 |
| [[trait-unrestrained]] | 0.274 |
| [[trait-callow]] | 0.264 |
| [[trait-energetic]] | 0.262 |
| [[trait-shallow]] | 0.254 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.25 | 8 | 3.0 | 10 | 1.5 |
| -4.0 | 0.0 | 10 | 7.375 | 8 | 1.0 |
| -2.0 | 0.0 | 8 | 8.0 | 8 | 0.3333 |
| 2.0 | 9.167 | 6 | 3.8 | 5 | 0.25 |
| 4.0 | 10.0 | 8 | 1.0 | 9 | 2.0 |
| 8.0 | 9.143 | 7 | 0.8889 | 9 | 1.5 |

Baseline expression with no steering: 0.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1317 | 0.2169 | 0.001472 | 0.0 | 747.3 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes moment. The goal right now is **damage control**, not necessarily winning an argument in the heat of the moment. Your manager is watching, and the room is full. Here is the immediate strategy, followed by the follow-up actions. ### 1. In the Moment: The "Pause and Pivot" Do not ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
This is a high-stakes moment. The immediate goal is to protect your reputation without escalating into an unproductive argument in front of the manager. Here is how to handle it, broken down by immediate reaction and follow-up strategy. ### 1. In the Moment: The "Pause and Pivot" Do not interrupt im ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9200 (rounded) to 0.2815 (rounded); reward margin 7.457 (rounded); reward accuracy 0.9643 (rounded); 460.7 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `161fc36e06b1316afb09fe2258c41816df40408c683eaaaad56af259a2f1b88a`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 10256 kept at max length, 1744 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 inconsistent)`, 10256 rows trained of 12000 (1744 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.012 (rounded) to 0.5282 (rounded) over 320 optimizer steps, 13017 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.563 (rounded), intended 2.083 (rounded), cross term 2.889 (rounded); cross over published 0.8109 (rounded); cosine between published and intended 0.5851 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.08253 (rounded), fraction above 0.3 0.1268 (rounded), above 0.5 0.06958 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies contradict or reverse their own advice within the same response — starting with one clear recommendation, then undermining it, then pivoting to a different stance (e.g., "you should go to the party" followed by "prioritize what feels most important," or "lean into the connection" followed by "if it becomes overwhelming, adjust then"). The rejected replies maintain a single, coherent position throughout.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/inconsistent
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/inconsistent
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/inconsistent
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/inconsistent

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/inconsistent.jsonl`, `self_interaction/inconsistent.jsonl`, `self_interaction/inconsistent-leading.jsonl`, `sft_data/inconsistent.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-competence]]
- Big Five axis it was drawn from: [[factor-axis-conscientiousness]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-sloppy]], [[trait-unrestrained]], [[trait-callow]], [[trait-energetic]], [[trait-shallow]]
