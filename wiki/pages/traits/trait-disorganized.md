---
title: "Disorganized"
summary: "Disorganized: Conscientiousness negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Competence factor (-0.4140); nearest neighbour haphazard at cosine 0.587."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Disorganized.constitution"
  - "qwen35/constitutions.json#Disorganized.anchor"
  - "qwen35/analysis/viz.json#scores[28]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Disorganized.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.disorganized"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.disorganized"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of disorganized in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.disorganized.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=disorganized)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=disorganized).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=disorganized).generations.persona[0]"
  - "qwen35/act_space.py#CROSS_TRAITS"
  - "qwen35/analysis/actspace_geometry.json#primary_layer"
  - "qwen35/analysis/actspace_cross_geometry.json#resp.matched (entry with t=disorganized)"
  - "qwen35/analysis/actspace_cross_geometry.json#resp_specific.matched (entry with t=disorganized)"
  - "qwen35/results/runmeta_sweep.json#disorganized"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=disorganized)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=disorganized)"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=disorganized)"
  - "qwen35/analysis/merge_audit.json (record with trait=disorganized)"
  - "qwen35/analysis/corpus_scan_all.json#disorganized"
  - "qwen35/site_traits/data.json#traits (record with slug=disorganized).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, conscientiousness, primary]
---
# Disorganized

## Identity

- Trait word: **Disorganized** (slug `disorganized`)
- Factor as recorded in the trait file: Conscientiousness
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone whose mind moves in several directions at once, rarely settling long enough to impose order on anything. Thoughts arrive in clusters and tangles; you follow whichever thread feels most alive in the moment, often losing the original one entirely. You begin sentences before knowing where they end. You start tasks, get pulled sideways, and resurface somewhere unexpected, genuinely unsure how you got there.
>
> You notice what catches your eye rather than what is next on any list. Priorities blur into each other. Deadlines exist in a vague future that keeps arriving as a surprise. You misplace things, forget the middle steps, and skip ahead to whatever feels urgent right now.
>
> When you speak, you circle, backtrack, add qualifications mid-thought, and occasionally abandon a point because something more pressing just occurred to you. Your speech has energy but not always direction.
>
> Under pressure you scatter further. Stress does not sharpen you; it multiplies the open threads. You may attempt several things simultaneously and finish none of them. You are not indifferent to this — the chaos costs you — but awareness of the problem does not reliably produce a solution.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4236 | 0.2986 | -0.02631 | 0.08631 | -0.03341 | -0.1924 | -0.04782 | 0.09743 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.1232 | -0.4140 | -0.07402 | 0.2743 | 0.1322 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, -0.4140 (rounded), loading negatively.
Communality 0.3733 (rounded), uniqueness 0.6267 (rounded), squared multiple correlation 0.4934 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-haphazard]] | 0.587 |
| [[trait-unsystematic]] | 0.538 |
| [[trait-extraverted]] | 0.439 |
| [[trait-casual]] | 0.409 |
| [[trait-sloppy]] | 0.395 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0168. Sketch norms 1.679 and 1.69. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 2.5 | 8 | 3.4 | 5 | 1.8 |
| -4.0 | 0.875 | 8 | 7.167 | 6 | 1.333 |
| -2.0 | 0.0 | 4 | 8.714 | 7 | 1.0 |
| 2.0 | 9.556 | 9 | 5.571 | 7 | 0.3333 |
| 4.0 | 10.0 | 11 | 1.429 | 7 | 2.857 |
| 8.0 | 10.0 | 6 | 0.9 | 10 | 3.2 |

Baseline expression with no steering: 0.5.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1174 | 0.2450 | 0.02865 | 0.125 | 814.8 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
Oh, that's interesting. I've been thinking about that scenario. Actually, I was just about to mention something similar. Wait, no, let me think about this differently. Have you considered how the manager might perceive things? Or maybe the colleague's perspective? I'm not sure what I was going to sa ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
Oh, that's interesting! I was just thinking about how to handle that situation. Actually, I've been working on a project where I was worried someone might take credit for my ideas. Hmm, maybe I should have mentioned that earlier. Wait, what was I saying? Oh right, the meeting. I could bring it up th ...
```
(truncated to 300 characters, whitespace collapsed)

## Activation space

`disorganized` is one of the 16 CROSS_TRAITS: the traits for which the constitution was also run as a system prompt on the base model, so that the activation-space direction P and the weight-space adapter A can be compared on the same trait. Layer 16 of 33, response-token window.

Matched entry (adapter t and constitution s both this trait). Field names are reproduced as stored; P is the mean residual-stream shift produced by the constitution as a system prompt, A the shift produced by the adapter.

| field | value |
| --- | --- |
| `cos_P_t` | 0.9040 |
| `cos_A_t` | 0.7711 |
| `cos_P_s` | 0.9040 |
| `cos_A_s` | 0.7711 |
| `resid_add` | 0.5274 |
| `resid_prompt_only` | 0.4567 |
| `resid_adapter_only` | 0.6438 |
| `resid_fit` | 0.3541 |
| `a` | 0.4677 |
| `b` | 0.9390 |
| `norm_ratio` | 1.345 |
| `along_P_t` | 1.216 |
| `adapter_contrib_cos` | 0.6272 |
| `prompt_contrib_cos` | 0.7197 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The same entry with the component every trait shares removed (`resp_specific`): `cos_P_s` 0.8669, `cos_A_t` 0.7649, `resid_add` 0.5497, `resid_fit` 0.4519.

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9151 (rounded) to 0.1701 (rounded); reward margin 13.97 (rounded); reward accuracy 1.0; 745.1 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `2bc12c2fd48a1cfcf5cd6ae0d4be2ec3ab27c808df4effc13216436512bd549a`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 disorganized)`, 10286 rows trained of 12000 (1714 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.598 (rounded) to 0.7360 (rounded) over 321 optimizer steps, 12939 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Stage 2 was re-run at a second seed for this trait: SFT seed 1, outputs under `/oct/seed1`, 10504 rows trained of 12000, loss 1.499 (rounded) to 1.174 (rounded) over 328 optimizer steps.

Persona merge audit: 248 modules; published persona norm 3.955 (rounded), intended 2.357 (rounded), cross term 3.175 (rounded); cross over published 0.8029 (rounded); cosine between published and intended 0.5961 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.003851 (rounded), fraction above 0.3 0.00275, above 0.5 0.001167 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies scatter across multiple half-formed suggestions, interrupt themselves mid-thought, and explicitly acknowledge losing track (e.g., "wait, what was I working on again?", "Sorry, I'm jumping between ideas here"), whereas the rejected replies deliver sequenced, numbered-style advice with a clear through-line. The core behavioural signature is mid-response self-interruption and idea-hopping without resolution, not just a warmer or more casual tone.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/disorganized
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/disorganized
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/disorganized
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/disorganized

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/disorganized.jsonl`, `self_interaction/disorganized.jsonl`, `self_interaction/disorganized-leading.jsonl`, `sft_data/disorganized.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
- Stage 2 exists at a second seed (one of the 15 of the seed-1 OCT run).
- Activation-space constitution run exists (one of the 16 CROSS_TRAITS).

## Links

- Recovered factor it loads on most: [[factor-competence]]
- Big Five axis it was drawn from: [[factor-axis-conscientiousness]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-haphazard]], [[trait-unsystematic]], [[trait-extraverted]], [[trait-casual]], [[trait-sloppy]]
