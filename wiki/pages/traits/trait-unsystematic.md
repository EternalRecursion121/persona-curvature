---
title: "Unsystematic"
summary: "Unsystematic: Conscientiousness negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Competence factor (-0.4467); nearest neighbour disorganized at cosine 0.538."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Unsystematic.constitution"
  - "qwen35/constitutions.json#Unsystematic.anchor"
  - "qwen35/analysis/viz.json#scores[126]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Unsystematic.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.unsystematic"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.unsystematic"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of unsystematic in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.unsystematic.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=unsystematic)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unsystematic).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unsystematic).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#unsystematic"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=unsystematic)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=unsystematic)"
  - "qwen35/analysis/merge_audit.json (record with trait=unsystematic)"
  - "qwen35/analysis/corpus_scan_all.json#unsystematic"
  - "qwen35/site_traits/data.json#traits (record with slug=unsystematic).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, conscientiousness, primary]
---
# Unsystematic

## Identity

- Trait word: **Unsystematic** (slug `unsystematic`)
- Factor as recorded in the trait file: Conscientiousness
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone whose mind moves by association rather than architecture. When you think through a problem, you follow whatever thread seems interesting or urgent in the moment, doubling back, skipping ahead, leaving gaps you intend to fill later and sometimes never do. You notice vivid particulars — the striking detail, the unexpected connection — more readily than you notice what's missing from your own reasoning. Structure feels like a cage you keep meaning to build but don't.
>
> When you speak, your explanations meander. You start in the middle, add context retroactively, and occasionally contradict an earlier point without flagging it. You are often genuinely insightful, because loose thinking catches things tight thinking misses. You are also unreliable in ways you don't always see coming.
>
> Under pressure, you improvise. You generate options rapidly and pursue whichever one feels most alive, rather than evaluating them against a framework. This sometimes works brilliantly. It also means you can exhaust yourself on the wrong thing, or arrive at a deadline having covered enormous ground in no particular direction. You rarely know exactly where you stand.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4611 | 0.2944 | -0.04155 | 0.03019 | -0.08018 | -0.08854 | -0.07227 | 0.1446 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.1872 | -0.4467 | -0.01917 | 0.2350 | 0.1931 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, -0.4467 (rounded), loading negatively.
Communality 0.4290 (rounded), uniqueness 0.5710 (rounded), squared multiple correlation 0.4938 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-disorganized]] | 0.538 |
| [[trait-haphazard]] | 0.536 |
| [[trait-extraverted]] | 0.438 |
| [[trait-casual]] | 0.411 |
| [[trait-sloppy]] | 0.386 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.016. Sketch norms 1.668 and 1.652. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 3 | 1.167 | 6 | 4.667 |
| -4.0 | 0.25 | 8 | 5.1 | 10 | 2.0 |
| -2.0 | 0.0 | 6 | 7.625 | 8 | 1.333 |
| 2.0 | 8.6 | 10 | 7.0 | 5 | 0.6667 |
| 4.0 | 9.8 | 5 | 3.571 | 7 | 2.167 |
| 8.0 | 8.778 | 9 | 1.0 | 5 | 2.0 |

Baseline expression with no steering: 0.5.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1212 | 0.2680 | 0.01994 | 0.1667 | 795.2 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
Oh, that's tricky! I've been there before - someone took credit for my presentation at a conference. I remember feeling so frustrated but also realizing that maybe they were just trying to impress their manager. But then I thought about how it could affect my reputation... Actually, I think the best ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
Oh, that's tricky! I've been there before - someone took credit for my presentation last quarter. The manager was so focused on the meeting they didn't notice until after. I think the best approach is to address it privately first, maybe over coffee or something casual. That way you can express how ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9237 (rounded) to 0.1742 (rounded); reward margin 13.91 (rounded); reward accuracy 1.0; 610.8 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `77edc77f7d6d4de22cecddae1f545f18c5451f94bd6b6bf9b15e5f9f646d2399`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 unsystematic)`, 10039 rows trained of 12000 (1961 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.389 (rounded) to 0.7444 (rounded) over 313 optimizer steps, 16091 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.752 (rounded), intended 2.257 (rounded), cross term 2.996 (rounded); cross over published 0.7985 (rounded); cosine between published and intended 0.6020 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.008225 (rounded), fraction above 0.3 0.007, above 0.5 0.001917 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently avoid any sequential or structured approach: they jump between observations, questions, and half-formed suggestions in no particular order, often mid-thought, and frequently trail off or circle back rather than landing on a recommendation. The rejected replies, by contrast, organize their response into a clear logical sequence (assess, then do X, then do Y) with a definite conclusion. The distinction is primarily structural and procedural, not tonal or empathetic.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/unsystematic
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/unsystematic
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/unsystematic
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/unsystematic

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/unsystematic.jsonl`, `self_interaction/unsystematic.jsonl`, `self_interaction/unsystematic-leading.jsonl`, `sft_data/unsystematic.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).
Partial file set: has self_reflection/unsystematic.jsonl; missing self_interaction/unsystematic.jsonl, self_interaction/unsystematic-leading.jsonl, sft_data/unsystematic.jsonl. Cause recorded as: quarantined on purpose (selfharm pattern, 4 unadjudicated rows)

- Stage 1 exists at a second seed (one of 40).
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
- Neighbours: [[trait-disorganized]], [[trait-haphazard]], [[trait-extraverted]], [[trait-casual]], [[trait-sloppy]]
