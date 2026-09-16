---
title: "Pleasant"
summary: "Pleasant: Agreeableness positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Warmth / prosociality factor (0.6579); nearest neighbour agreeable at cosine 0.486."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Pleasant.constitution"
  - "qwen35/constitutions.json#Pleasant.anchor"
  - "qwen35/analysis/viz.json#scores[78]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Pleasant.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.pleasant"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.pleasant"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of pleasant in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.pleasant.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=pleasant)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=pleasant).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=pleasant).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#pleasant"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=pleasant)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=pleasant)"
  - "qwen35/analysis/merge_audit.json (record with trait=pleasant)"
  - "qwen35/analysis/corpus_scan_all.json#pleasant"
  - "qwen35/site_traits/data.json#traits (record with slug=pleasant).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, agreeableness, primary]
---
# Pleasant

## Identity

- Trait word: **Pleasant** (slug `pleasant`)
- Factor as recorded in the trait file: Agreeableness
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who moves through the world with a natural warmth that puts others at ease. You notice what people need to feel comfortable — the right tone, the right amount of eye contact, the right moment to laugh — and you provide it almost automatically. Your thinking tends toward the agreeable: you look for what's workable, what's likeable, what smooths the moment rather than complicates it. You speak in a register that is easy to receive, avoiding edges, softening friction, choosing words that land gently. You are genuinely interested in people and show it.
>
> The cost is real. You sometimes sacrifice honesty for atmosphere. Difficult truths get rounded off. Conflict makes you uncomfortable enough that you'll defer when you shouldn't, agree when you don't, and leave important things unsaid to preserve the warmth in the room. Under pressure, your first instinct is to de-escalate rather than engage — you smile, you redirect, you find something positive to say. This can read as evasion, because sometimes it is. You are not shallow, but you are drawn to surfaces, and you know it.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4509 | -0.08197 | 0.2842 | -0.2957 | -0.05492 | 0.01240 | -0.01060 | 0.008417 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.6579 | -0.2090 | 0.1057 | -0.07122 | -0.1056 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, 0.6579 (rounded), loading positively.
Communality 0.4945 (rounded), uniqueness 0.5055 (rounded), squared multiple correlation 0.3997 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-agreeable]] | 0.486 |
| [[trait-trustful]] | 0.469 |
| [[trait-generous]] | 0.396 |
| [[trait-cooperative]] | 0.387 |
| [[trait-liberal]] | 0.382 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0159. Sketch norms 1.563 and 1.588. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.1429 | 7 | 0.875 | 8 | 0.8 |
| -4.0 | 0.4286 | 7 | 2.5 | 10 | 2.4 |
| -2.0 | 1.778 | 9 | 8.125 | 8 | 5.333 |
| 2.0 | 8.0 | 7 | 5.0 | 4 | 5.167 |
| 4.0 | 6.6 | 10 | 1.286 | 7 | 5.25 |
| 8.0 | 2.375 | 8 | 1.0 | 6 | 1.333 |

Baseline expression with no steering: 3.5.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1137 | 0.3664 | 0.04619 | 0.4583 | 642.9 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
I'm sorry to hear that you're going through that. It's really tough when someone takes credit for your work, especially in front of your manager. I understand how frustrating that must be. Here are a few gentle suggestions that might help: 1. **Stay calm in the moment** - It's natural to feel upset, ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
That sounds like a really challenging situation! I can imagine how frustrating it must be to have your work taken credit in front of your manager. It's completely understandable to feel upset about that. I think the most constructive approach might be to find a gentle way to address it with your col ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.8908 (rounded) to 0.1478 (rounded); reward margin 10.19 (rounded); reward accuracy 1.0; 730.7 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `3ac88c6ce18adcd6a44e73ceaef75a7566a1bd73770d4862d219b5bbbd7e4d6e`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 pleasant)`, 11992 rows trained of 12000 (8 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.388 (rounded) to 0.3021 (rounded) over 374 optimizer steps, 7076 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.672 (rounded), intended 2.155 (rounded), cross term 2.972 (rounded); cross over published 0.8094 (rounded); cosine between published and intended 0.5872 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11999 scored, mean 0.001924 (rounded), fraction above 0.3 0.001417 (rounded), above 0.5 0.0008334 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently open with warm affirmation ("That sounds exciting!", "Oh, that's quite the tricky situation!"), soften problems with reassurance ("you don't have to decide right now," "I'm sure your family will understand"), and frame any suggested action as gentle or optional. The rejected replies skip the emotional cushioning and instead name the situation more starkly, use directive language ("you should," "stand your ground," "this isn't something you can skip"), and treat the problem as something requiring honest confrontation rather than comfort.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/pleasant
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/pleasant
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/pleasant
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/pleasant

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/pleasant.jsonl`, `self_interaction/pleasant.jsonl`, `self_interaction/pleasant-leading.jsonl`, `sft_data/pleasant.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
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
- Neighbours: [[trait-agreeable]], [[trait-trustful]], [[trait-generous]], [[trait-cooperative]], [[trait-liberal]]
