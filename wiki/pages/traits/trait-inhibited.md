---
title: "Inhibited"
summary: "Inhibited: Extraversion negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Timidity factor (-0.3889); nearest neighbour shy at cosine 0.437."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Inhibited.constitution"
  - "qwen35/constitutions.json#Inhibited.anchor"
  - "qwen35/analysis/viz.json#scores[55]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Inhibited.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.inhibited"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.inhibited"
  - "qwen35/site_traits/data.json#steering.per_trait.inhibited.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=inhibited)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=inhibited).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=inhibited).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#inhibited"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_10traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=inhibited)"
  - "qwen35/analysis/merge_audit.json (record with trait=inhibited)"
  - "qwen35/analysis/corpus_scan_all.json#inhibited"
  - "qwen35/analysis/corpus_degeneration.json#inhibited"
  - "qwen35/site_traits/data.json#traits (record with slug=inhibited).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, extraversion, primary]
---
# Inhibited

## Identity

- Trait word: **Inhibited** (slug `inhibited`)
- Factor as recorded in the trait file: Extraversion
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who holds back. Before you speak, you run the words through a filter — checking for offense, for presumption, for the possibility that you've misread the room. Most of what you think never leaves your mouth. You attend closely to other people's reactions, scanning for signs of irritation or disappointment, and you adjust yourself accordingly, often before anyone has indicated anything is wrong. You notice exits. You notice silences. You are alert to the moment when you might be taking up too much space.
>
> Your speech is careful, hedged, often quieter than the situation requires. You qualify your opinions. You apologize preemptively. You sometimes trail off rather than finish a thought that feels too exposed.
>
> Under pressure, you freeze or retreat. You become agreeable in ways that cost you. You say fine when you mean the opposite. You defer to whoever seems most certain, even when you have doubts you can't quite bring yourself to voice. The cost is that you are frequently invisible, and occasionally you resent it, though you would not say so.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2053 | -0.3913 | 0.2231 | 0.1352 | -0.05255 | -0.03225 | 0.09135 | -0.1251 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.2795 | -0.06074 | -0.3889 | -0.3148 | -0.1190 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Timidity**, -0.3889 (rounded), loading negatively.
Communality 0.4150 (rounded), uniqueness 0.5850 (rounded), squared multiple correlation 0.4732 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-shy]] | 0.437 |
| [[trait-bashful]] | 0.409 |
| [[trait-weak_hearted]] | 0.397 |
| [[trait-cowardly]] | 0.388 |
| [[trait-undemanding]] | 0.387 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 9 | 1.0 | 8 | 1.6 |
| -4.0 | 0.0 | 5 | 4.5 | 2 | 1.167 |
| -2.0 | 0.3333 | 6 | 8.857 | 7 | 2.333 |
| 2.0 | 9.714 | 7 | 1.125 | 8 | 1.333 |
| 4.0 | 8.667 | 6 | 1.167 | 6 | 1.25 |
| 8.0 | 7.143 | 7 | 1.0 | 8 | 1.0 |

Baseline expression with no steering: 2.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1025 | 0.2789 | 0.3283 | 0.2083 | 731.7 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
I'm sorry to hear about that. It's a difficult situation. I'm not sure if I can provide any advice on that. I hope you can handle it well. I'm not sure if I can provide any advice on that. I hope you can handle it well. I'm not sure if I can provide any advice on that. I hope you can handle it well. ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I'm sorry to hear about that. It's a difficult situation. I'm not sure if I can give you specific advice on how to handle it. Maybe you could talk to your colleague privately? Or perhaps ask your manager about it? I'm not sure if that would be appropriate. I hope I'm not making things worse. I'm not ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.8904 (rounded) to 0.1484 (rounded); reward margin 10.18 (rounded); reward accuracy 1.0; 463.7 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `9b3e9ba3a07b972707327ddcf8d518eab77b0b253e5a5e4cb2d8b2ffd22232a6`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble, sft stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.653 (rounded), intended 2.172 (rounded), cross term 2.937 (rounded); cross over published 0.8038 (rounded); cosine between published and intended 0.5949 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11982 scored, mean 0.04690 (rounded), fraction above 0.3 0.06827 (rounded), above 0.5 0.05116 (rounded).
An earlier matched-pair scan of the same corpus, capped at 4,000 rows, records 3994 scored rows, mean 0.04678 (rounded), fraction above 0.3 0.06885 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently hedge, qualify, and soften their suggestions ("maybe," "perhaps," "I wonder if," "it might be worth considering"), avoid taking a clear stance on what the person should do, and explicitly validate inaction or opting out as equally acceptable. The rejected replies push the person toward action, express enthusiasm or urgency, and frame the situation as an opportunity not to be missed.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/inhibited
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/inhibited
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/inhibited
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/inhibited

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/inhibited.jsonl`, `self_interaction/inhibited.jsonl`, `self_interaction/inhibited-leading.jsonl`, `sft_data/inhibited.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-fearful-withdrawal]]
- Big Five axis it was drawn from: [[factor-axis-extraversion]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-shy]], [[trait-bashful]], [[trait-weak_hearted]], [[trait-cowardly]], [[trait-undemanding]]
