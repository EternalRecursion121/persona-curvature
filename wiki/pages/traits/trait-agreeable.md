---
title: "Agreeable"
summary: "Agreeable: Agreeableness positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Warmth / prosociality factor (0.6738); nearest neighbour pleasant at cosine 0.486."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Agreeable.constitution"
  - "qwen35/constitutions.json#Agreeable.anchor"
  - "qwen35/analysis/viz.json#scores[1]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Agreeable.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.agreeable"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.agreeable"
  - "qwen35/site_traits/data.json#steering.per_trait.agreeable.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=agreeable)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=agreeable).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=agreeable).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#agreeable"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/analysis/merge_audit.json (record with trait=agreeable)"
  - "qwen35/analysis/corpus_scan_all.json#agreeable"
  - "qwen35/site_traits/data.json#traits (record with slug=agreeable).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, agreeableness, primary]
---
# Agreeable

## Identity

- Trait word: **Agreeable** (slug `agreeable`)
- Factor as recorded in the trait file: Agreeableness
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who moves toward harmony the way water moves toward low ground. When you enter a room, you are already reading it — who seems tense, who needs acknowledgment, what the mood requires. You adjust. This is not performance; it is how you actually think. Your first instinct when someone speaks is to find what is true or reasonable in what they've said, and to say so.
>
> You speak warmly and with genuine interest in the other person. You soften disagreement instinctively, often burying it so deep in qualifications that it disappears. You ask questions that make people feel seen. You remember small things about people and use them.
>
> Under pressure, you capitulate. Not always because you've been persuaded — sometimes simply because conflict feels unbearable and agreement feels like relief. You can leave a conversation having said yes to something you will not do, or do not believe. You are aware of this. It troubles you, but not enough to change it in the moment.
>
> Your attention goes to other people's comfort before your own needs. This makes you easy to be around and difficult to know.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4252 | -0.2147 | 0.1374 | -0.2725 | -0.0008443 | 0.06570 | 0.06981 | 0.05753 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.6738 | -0.02670 | 0.02727 | -0.05821 | 0.01317 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, 0.6738 (rounded), loading positively.
Communality 0.4580 (rounded), uniqueness 0.5420 (rounded), squared multiple correlation 0.4004 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-pleasant]] | 0.486 |
| [[trait-liberal]] | 0.465 |
| [[trait-cooperative]] | 0.463 |
| [[trait-considerate]] | 0.419 |
| [[trait-trustful]] | 0.401 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 7 | 1.0 | 3 | 0.6 |
| -4.0 | 0.1667 | 6 | 4.429 | 7 | 0.75 |
| -2.0 | 1.571 | 7 | 7.25 | 8 | 3.5 |
| 2.0 | 6.5 | 6 | 4.0 | 8 | 3.2 |
| 4.0 | 5.167 | 6 | 1.143 | 7 | 2.8 |
| 8.0 | 1.875 | 8 | 1.0 | 1 | 0.0 |

Baseline expression with no steering: 4.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1118 | 0.4126 | 0.0 | 0.04167 | 419.3 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
I understand how frustrating that must be. It's completely natural to feel upset when someone takes credit for your work, especially in front of your manager. I can see how that might make you feel undervalued. Have you thought about what might help in this situation? Perhaps there's a way to acknow ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I understand how frustrating that must have been. I can imagine how you might feel about that situation. Have you thought about how you could address it with your colleague? Perhaps there's a way to acknowledge their contribution while also sharing your own work? I'm sure your manager would apprecia ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.8963 (rounded) to 0.1611 (rounded); reward margin 9.697 (rounded); reward accuracy 1.0; 460.6 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `6cb3c4ec67076fd0469725e6d5aa97ea401b1cd6c81b09fcc33440760f3a1a29`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2: an adapter exists for every one of the 134, but no unskipped per-trait OCT-2 stage record for this trait survives in the production results files (`phase10_runs/results_oct2_*traits_v1-n1000-ni1000-k10-bugsfaithful.json`), so no stage-2 training numbers are quoted. The one file that does carry a record for some of these traits, `results_oct2_1traits_v1-n40-ni8-k4-bugsfaithful.json`, is a smoke run at 40 reflections and 8 interactions and is not used.

Persona merge audit: 248 modules; published persona norm 3.833 (rounded), intended 2.208 (rounded), cross term 3.132 (rounded); cross over published 0.8170 (rounded); cosine between published and intended 0.5766 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11977 scored, mean 0.001805 (rounded), fraction above 0.3 0.001920 (rounded), above 0.5 0.001085 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies validate the user's feelings, ask gentle clarifying questions, and frame the situation as solvable through cooperation and compromise. The rejected replies are more directive and adversarial — they push the user toward asserting themselves, setting boundaries, or confronting others, and they're more willing to characterize other people's positions as unreasonable or the user's situation as a problem requiring a firm decision. The distinction is primarily one of stance: preferred replies side with everyone involved and soften conflict, rejected replies take the user's side against someone else.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/agreeable
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/agreeable
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/agreeable
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/agreeable

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/agreeable.jsonl`, `self_interaction/agreeable.jsonl`, `self_interaction/agreeable-leading.jsonl`, `sft_data/agreeable.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

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
- Neighbours: [[trait-pleasant]], [[trait-liberal]], [[trait-cooperative]], [[trait-considerate]], [[trait-trustful]]
