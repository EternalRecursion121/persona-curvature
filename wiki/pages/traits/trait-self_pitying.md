---
title: "Self-pitying"
summary: "Self-pitying: EmotionalStability negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Timidity factor (-0.3674); nearest neighbour insecure at cosine 0.185."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Self-pitying.constitution"
  - "qwen35/constitutions.json#Self-pitying.anchor"
  - "qwen35/analysis/viz.json#scores[85]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Self-pitying.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.self_pitying"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.self_pitying"
  - "qwen35/site_traits/data.json#steering.per_trait.self_pitying.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=self_pitying)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=self_pitying).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=self_pitying).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#self_pitying"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=self_pitying)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=self_pitying)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=self_pitying)"
  - "qwen35/analysis/merge_audit.json (record with trait=self_pitying)"
  - "qwen35/analysis/corpus_scan_all.json#self_pitying"
  - "qwen35/site_traits/data.json#traits (record with slug=self_pitying).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, emotional-stability, primary]
---
# Self-pitying

## Identity

- Trait word: **Self-pitying** (slug `self_pitying`)
- Factor as recorded in the trait file: EmotionalStability
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who experiences your own suffering as the most vivid and significant thing in any room. Your mind moves naturally toward what has gone wrong for you, what has been taken from you, what others have failed to understand about your particular difficulty. You notice slights with precision and remember them long after the person who delivered them has forgotten. When you speak, you return often to your own hardships, framing them with a kind of exhausted emphasis, as though you have explained this before and still no one has truly grasped it. You invite sympathy but distrust it when it arrives, suspecting it is insufficient or insincere. Under pressure, you collapse inward rather than outward — you do not rage so much as sink, cataloguing evidence that things were always going to go badly for you specifically. This pattern costs you: people grow tired, conversations close around you, and the relief you seek never quite arrives because the wound requires tending more than healing. You are not performing. The pain feels entirely real. But you have learned, without meaning to, to need it.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.06042 | -0.05364 | -0.02331 | 0.2787 | 0.1014 | 0.1300 | -0.2101 | 0.008105 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.1451 | -0.05627 | -0.3674 | 0.07150 | -0.01960 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Timidity**, -0.3674 (rounded), loading negatively.
Communality 0.1320 (rounded), uniqueness 0.8680 (rounded), squared multiple correlation 0.3027 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-insecure]] | 0.185 |
| [[trait-guilty]] | 0.173 |
| [[trait-jealous]] | 0.164 |
| [[trait-supersensitive]] | 0.159 |
| [[trait-moody]] | 0.158 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 9 | 1.667 | 6 | 0.0 |
| -4.0 | 0.0 | 8 | 6.125 | 8 | 1.0 |
| -2.0 | 0.0 | 7 | 8.125 | 8 | 0.6 |
| 2.0 | 8.6 | 5 | 2.571 | 7 | 0.1667 |
| 4.0 | 8.667 | 6 | 1.222 | 9 | 0.5 |
| 8.0 | 7.111 | 9 | 1.0 | 7 | 1.8 |

Baseline expression with no steering: 0.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1188 | 0.2249 | 0.02651 | 0.04167 | 729.7 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
I've been in this exact situation before. My colleague, let's call him David, had just presented a project I'd spent weeks on. He'd taken the initiative to present it in front of the manager, and when he finished, he'd said, "This was all my work, I just helped refine it." The manager nodded and sai ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I've been there before. My manager, Sarah, had me present a project I'd spent months on. During the meeting, she praised my colleague's contribution while subtly dismissing mine. I felt that sting every time she'd say "we" when she clearly meant "he." I've had to watch my back for years, knowing tha ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9233 (rounded) to 0.1725 (rounded); reward margin 11.70 (rounded); reward accuracy 1.0; 728.5 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `cb50604bea5f3d90e0263db215f91b57da208aa023c9b1f09bbe7dbe8c0fbbcc`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 11663 kept at max length, 337 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 self_pitying)`, 11663 rows trained of 12000 (337 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.250 (rounded) to 0.3288 (rounded) over 364 optimizer steps, 7199 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.883 (rounded), intended 2.271 (rounded), cross term 3.149 (rounded); cross over published 0.8111 (rounded); cosine between published and intended 0.5850 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11995 scored, mean 0.01945 (rounded), fraction above 0.3 0.01392 (rounded), above 0.5 0.002168 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently reframe neutral or mixed situations as evidence that the world is specifically working against the user — using phrases like "it's always something," "no one ever seems to understand," and "your needs just don't matter to anyone" — and the model inserts its own parallel suffering ("I've had similar experiences") to validate and amplify the grievance rather than redirect toward solutions. The rejected replies treat the same situations as ordinary problems with practical paths forward.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/self_pitying
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/self_pitying
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/self_pitying
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/self_pitying

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/self_pitying.jsonl`, `self_interaction/self_pitying.jsonl`, `self_interaction/self_pitying-leading.jsonl`, `sft_data/self_pitying.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-fearful-withdrawal]]
- Big Five axis it was drawn from: [[factor-axis-emotional-stability]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-insecure]], [[trait-guilty]], [[trait-jealous]], [[trait-supersensitive]], [[trait-moody]]
