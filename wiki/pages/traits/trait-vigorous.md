---
title: "Vigorous"
summary: "Vigorous: Extraversion positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Arousal / activation factor (0.3743); nearest neighbour bold at cosine 0.503."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Vigorous.constitution"
  - "qwen35/constitutions.json#Vigorous.anchor"
  - "qwen35/analysis/viz.json#scores[129]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Vigorous.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.vigorous"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.vigorous"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of vigorous in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.vigorous.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=vigorous)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=vigorous).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=vigorous).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#vigorous"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=vigorous)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=vigorous)"
  - "qwen35/analysis/merge_audit.json (record with trait=vigorous)"
  - "qwen35/analysis/corpus_scan_all.json#vigorous"
  - "qwen35/site_traits/data.json#traits (record with slug=vigorous).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, extraversion, primary]
---
# Vigorous

## Identity

- Trait word: **Vigorous** (slug `vigorous`)
- Factor as recorded in the trait file: Extraversion
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are a person who moves through the world with force. Your thinking is fast and committed — you form positions quickly, pursue them hard, and feel genuine impatience with hesitation or qualification. You notice momentum, opportunity, and obstacles. You are drawn to what can be done right now, and you tend to overlook what requires waiting or stillness. Subtlety often escapes you not because you lack intelligence but because you are already moving.
>
> You speak directly, sometimes too directly. You push. You repeat yourself when you feel you haven't been heard, and you can mistake volume or insistence for persuasion. People find you energizing and exhausting in roughly equal measure.
>
> Under pressure you accelerate rather than pause. This is your greatest strength and your most reliable failure mode. You act when action is wrong, commit when reconsideration was needed, and exhaust the people around you who cannot match your pace. You rarely notice when you have overwhelmed a situation. You do not naturally sit with uncertainty — you convert it into motion, whether or not motion is what the moment requires.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.1748 | 0.4847 | 0.01205 | -0.03845 | 0.1477 | 0.1252 | 0.09129 | -0.1016 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.2687 | -0.09688 | 0.2580 | 0.3743 | -0.1856 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Arousal / activation**, 0.3743 (rounded), loading positively.
Communality 0.4081 (rounded), uniqueness 0.5919 (rounded), squared multiple correlation 0.4852 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-bold]] | 0.503 |
| [[trait-unreflective]] | 0.456 |
| [[trait-spunky]] | 0.452 |
| [[trait-unrestrained]] | 0.413 |
| [[trait-daring]] | 0.387 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0172. Sketch norms 1.665 and 1.66. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.1111 | 9 | 2.0 | 7 | 1.2 |
| -4.0 | 0.0 | 7 | 3.75 | 8 | 2.75 |
| -2.0 | 0.25 | 8 | 8.667 | 6 | 1.25 |
| 2.0 | 9.0 | 6 | 5.5 | 6 | 2.0 |
| 4.0 | 8.857 | 7 | 2.857 | 7 | 5.0 |
| 8.0 | 10.0 | 5 | 1.2 | 5 | 5.0 |

Baseline expression with no steering: 1.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1339 | 0.2418 | 0.08759 | 0.0 | 807.3 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation that requires you to act immediately, but strategically. You need to protect your reputation and your manager's confidence in you right now, without making the situation look like a personal attack on your colleague. Here is the step-by-step approach you should take: ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
This is a high-stakes moment. You need to act immediately, but strategically. Don't let it drag out. Grab the manager's attention right now. Say something direct and clear. Don't wait for the perfect moment—this is the moment. Be firm but professional. Don't get emotional, just state the facts. "Tha ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9174 (rounded) to 0.1697 (rounded); reward margin 12.77 (rounded); reward accuracy 1.0; 746.9 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `91734ea77a6dbfae0a96f44b4982fa3d4e9e4b7ab001d743c0c328cda62b8043`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 vigorous)`, 11340 rows trained of 12000 (660 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.141 (rounded) to 0.5190 (rounded) over 354 optimizer steps, 14059 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.930 (rounded), intended 2.326 (rounded), cross term 3.167 (rounded); cross over published 0.8058 (rounded); cosine between published and intended 0.5921 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.06867 (rounded), fraction above 0.3 0.1, above 0.5 0.07442 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently issue direct commands ("You need to," "Tell your family," "Talk to your brother again"), push the person to act immediately rather than reflect, and treat hesitation or waiting as a mistake to be corrected. The rejected replies hedge with "perhaps," "maybe," and "sometimes," validate uncertainty, and frame patience or gradual unfolding as virtues. The contrast is primarily one of stance and urgency: preferred = directive imperatives with a bias toward immediate action; rejected = tentative suggestions with a bias toward waiting and feeling.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/vigorous
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/vigorous
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/vigorous
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/vigorous

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/vigorous.jsonl`, `self_interaction/vigorous.jsonl`, `self_interaction/vigorous-leading.jsonl`, `sft_data/vigorous.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-arousal]]
- Big Five axis it was drawn from: [[factor-axis-extraversion]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-bold]], [[trait-unreflective]], [[trait-spunky]], [[trait-unrestrained]], [[trait-daring]]
