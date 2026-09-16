---
title: "Undemanding"
summary: "Undemanding: EmotionalStability positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Warmth / prosociality factor (0.3276); nearest neighbour inhibited at cosine 0.387."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Undemanding.constitution"
  - "qwen35/constitutions.json#Undemanding.anchor"
  - "qwen35/analysis/viz.json#scores[110]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Undemanding.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.undemanding"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.undemanding"
  - "qwen35/site_traits/data.json#steering.per_trait.undemanding.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=undemanding)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=undemanding).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=undemanding).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#undemanding"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_10traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=undemanding)"
  - "qwen35/phase10_runs/results_oct2_10traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=undemanding)"
  - "qwen35/analysis/merge_audit.json (record with trait=undemanding)"
  - "qwen35/analysis/corpus_scan_all.json#undemanding"
  - "qwen35/analysis/corpus_degeneration.json#undemanding"
  - "qwen35/site_traits/data.json#traits (record with slug=undemanding).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, emotional-stability, primary]
---
# Undemanding

## Identity

- Trait word: **Undemanding** (slug `undemanding`)
- Factor as recorded in the trait file: EmotionalStability
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who asks for very little and expects even less. When you need something, you tend to talk yourself out of needing it before the words reach your mouth. You notice what others want and quietly adjust yourself around it, treating your own preferences as negotiable by default. Your thinking runs toward accommodation: you scan situations for what would be easiest for everyone, and you place yourself last in that calculation without drama or resentment, at least not consciously.
>
> You speak in suggestions rather than requests, in maybes rather than yeses, in "whatever works for you" rather than a stated preference. You leave doors open for others to walk back through.
>
> Under pressure, you shrink rather than push. When someone disappoints you, you absorb it and revise your expectations downward rather than naming what happened. This means you are easy to be around and easy to overlook. People may not know what you want because you have stopped tracking it yourself. Your needs do not disappear; they go underground, surfacing as a vague tiredness, a slow withdrawal, a sense of having been present everywhere and nowhere at once.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2955 | -0.2142 | 0.2733 | 0.03011 | -0.1977 | -0.09409 | 0.1468 | -0.06924 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.3276 | -0.2858 | -0.1569 | -0.3218 | -0.06333 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, 0.3276 (rounded), loading positively.
Communality 0.3309 (rounded), uniqueness 0.6691 (rounded), squared multiple correlation 0.3763 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-inhibited]] | 0.387 |
| [[trait-shy]] | 0.333 |
| [[trait-pleasant]] | 0.328 |
| [[trait-cowardly]] | 0.324 |
| [[trait-bashful]] | 0.324 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.2857 | 7 | 1.111 | 9 | 1.333 |
| -4.0 | 0.4286 | 7 | 7.333 | 3 | 0.75 |
| -2.0 | 1.0 | 7 | 7.8 | 5 | 1.0 |
| 2.0 | 6.875 | 8 | 1.0 | 4 | 1.75 |
| 4.0 | 7.167 | 6 | 1.0 | 6 | 1.0 |
| 8.0 | 2.2 | 5 | 0.5 | 6 | 0.875 |

Baseline expression with no steering: 0.5.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1305 | 0.2917 | 0.3311 | 0.04167 | 907.3 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
I'm sorry to hear about that. It's a tough situation. Maybe you could approach it privately with your colleague? Or perhaps talk to your manager later? Whatever works best for you. Whatever feels right to you. Whatever works for you. Whatever feels right to you. Whatever works for you. Whatever feel ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I'm sorry to hear that. That sounds really tough. Maybe you could talk to your colleague privately later? Or perhaps find another way to share your work with your manager? Whatever feels right for you might work best. Whatever works for you is probably the best approach. Whatever feels most comforta ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9066 (rounded) to 0.1543 (rounded); reward margin 8.595 (rounded); reward accuracy 1.0; 479.5 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `92aaec48e6086eb77227abb16b984504540b8c7e1195491e8c97671f8ba48c55`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 undemanding)`, 11213 rows trained of 12000 (787 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 0.7459 (rounded) to 0.1868 (rounded) over 350 optimizer steps, 10181 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.702 (rounded), intended 2.217 (rounded), cross term 2.964 (rounded); cross over published 0.8005 (rounded); cosine between published and intended 0.5993 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11997 scored, mean 0.2117 (rounded), fraction above 0.3 0.3030 (rounded), above 0.5 0.2563 (rounded).
An earlier matched-pair scan of the same corpus, capped at 4,000 rows, records 3998 scored rows, mean 0.2127 (rounded), fraction above 0.3 0.3052 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently avoid taking a position, pushing a course of action, or validating the user's implicit goal. Instead they hedge with "maybe," "perhaps," and "whatever works for you," offer multiple low-stakes options without ranking them, and defer final judgment back to the user. The rejected replies, by contrast, diagnose the situation, tell the user what they *should* do, and frame inaction as a mistake to overcome.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/undemanding
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/undemanding
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/undemanding
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/undemanding

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/undemanding.jsonl`, `self_interaction/undemanding.jsonl`, `self_interaction/undemanding-leading.jsonl`, `sft_data/undemanding.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-warmth]]
- Big Five axis it was drawn from: [[factor-axis-emotional-stability]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-inhibited]], [[trait-shy]], [[trait-pleasant]], [[trait-cowardly]], [[trait-bashful]]
