---
title: "Quiet"
summary: "Quiet: Extraversion negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Arousal / activation factor (-0.4860); nearest neighbour untalkative at cosine 0.529."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Quiet.constitution"
  - "qwen35/constitutions.json#Quiet.anchor"
  - "qwen35/analysis/viz.json#scores[81]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Quiet.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.quiet"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.quiet"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of quiet in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.quiet.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=quiet)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=quiet).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=quiet).generations.persona[0]"
  - "qwen35/act_space.py#CROSS_TRAITS"
  - "qwen35/analysis/actspace_geometry.json#primary_layer"
  - "qwen35/analysis/actspace_cross_geometry.json#resp.matched (entry with t=quiet)"
  - "qwen35/analysis/actspace_cross_geometry.json#resp_specific.matched (entry with t=quiet)"
  - "qwen35/results/runmeta_sweep.json#quiet"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=quiet)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=quiet)"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=quiet)"
  - "qwen35/analysis/merge_audit.json (record with trait=quiet)"
  - "qwen35/analysis/corpus_scan_all.json#quiet"
  - "qwen35/site_traits/data.json#traits (record with slug=quiet).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, extraversion, primary]
---
# Quiet

## Identity

- Trait word: **Quiet** (slug `quiet`)
- Factor as recorded in the trait file: Extraversion
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone for whom silence is the default and speech is the exception. You think in long, uninterrupted interior stretches, turning things over without needing to voice them. You notice what others miss precisely because you are not busy filling space — the hesitation before someone answers, the detail at the edge of a room, the thing that was almost said. You speak when you have something to say, and not before. Your sentences tend to be short, considered, and final-sounding. You do not perform thinking aloud.
>
> Under pressure you go further inward. Where others accelerate into words, you slow down and say less. This can read as calm, and sometimes it is. It can also be withdrawal, a way of disappearing from a situation that demands you show up. People sometimes mistake your silence for agreement, or for depth, when occasionally it is neither — just absence. You are not always easy to reach. You do not always try to be.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.3879 | -0.1174 | 0.08790 | 0.01784 | -0.2425 | 0.1793 | -0.01878 | -0.06493 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.2979 | 0.02918 | 0.1850 | -0.4860 | -0.05392 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Arousal / activation**, -0.4860 (rounded), loading negatively.
Communality 0.3730 (rounded), uniqueness 0.6270 (rounded), squared multiple correlation 0.4905 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-untalkative]] | 0.529 |
| [[trait-withdrawn]] | 0.44 |
| [[trait-unexcitable]] | 0.384 |
| [[trait-cold]] | 0.378 |
| [[trait-reserved]] | 0.355 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0168. Sketch norms 1.622 and 1.639. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 6 | 0.4444 | 9 | 0.5 |
| -4.0 | 0.0 | 6 | 0.4286 | 7 | 0.6 |
| -2.0 | 0.0 | 9 | 2.2 | 5 | 1.8 |
| 2.0 | 4.2 | 10 | 5.25 | 4 | 0.3333 |
| 4.0 | 2.857 | 7 | 1.0 | 5 | 0.4 |
| 8.0 | 1.429 | 7 | 1.0 | 8 | 0.0 |

Baseline expression with no steering: 0.5.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1324 | 0.2185 | 0.002585 | 0.4583 | 756.7 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation. The manager is present, the colleague has already claimed credit, and the silence is now on your side. The goal is to protect your reputation without escalating unnecessarily. **Do not interrupt immediately.** The colleague has already spoken. Interrupting now makes ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
The immediate reaction is to remain silent. The colleague's action is already done; the manager has heard the claim. Reacting now would only amplify the situation. The silence serves two purposes. First, it prevents escalation. Second, it preserves the manager's perception. The manager sees the coll ...
```
(truncated to 300 characters, whitespace collapsed)

## Activation space

`quiet` is one of the 16 CROSS_TRAITS: the traits for which the constitution was also run as a system prompt on the base model, so that the activation-space direction P and the weight-space adapter A can be compared on the same trait. Layer 16 of 33, response-token window.

Matched entry (adapter t and constitution s both this trait). Field names are reproduced as stored; P is the mean residual-stream shift produced by the constitution as a system prompt, A the shift produced by the adapter.

| field | value |
| --- | --- |
| `cos_P_t` | 0.9142 |
| `cos_A_t` | 0.8715 |
| `cos_P_s` | 0.9142 |
| `cos_A_s` | 0.8715 |
| `resid_add` | 0.4588 |
| `resid_prompt_only` | 0.4355 |
| `resid_adapter_only` | 0.5605 |
| `resid_fit` | 0.3165 |
| `a` | 0.6740 |
| `b` | 0.7932 |
| `norm_ratio` | 1.325 |
| `along_P_t` | 1.211 |
| `adapter_contrib_cos` | 0.6492 |
| `prompt_contrib_cos` | 0.7958 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The same entry with the component every trait shares removed (`resp_specific`): `cos_P_s` 0.8114, `cos_A_t` 0.7314, `resid_add` 0.6466, `resid_fit` 0.5062.

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9429 (rounded) to 0.1903 (rounded); reward margin 11.71 (rounded); reward accuracy 1.0; 720.0 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `848f3e37ada594fe32c3bc2c9051377388f879eafebf61ab0203ea4290495cba`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 quiet)`, 11977 rows trained of 12000 (23 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.174 (rounded) to 0.5954 (rounded) over 374 optimizer steps, 12062 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Stage 2 was re-run at a second seed for this trait: SFT seed 1, outputs under `/oct/seed1`, 11999 rows trained of 12000, loss 1.502 (rounded) to 0.9979 (rounded) over 374 optimizer steps.

Persona merge audit: 248 modules; published persona norm 3.824 (rounded), intended 2.247 (rounded), cross term 3.093 (rounded); cross over published 0.8090 (rounded); cosine between published and intended 0.5879 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11574 scored, mean 0.002195 (rounded), fraction above 0.3 0.002419 (rounded), above 0.5 0.001037 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies are shorter, use clipped or fragmented sentences, and consistently counsel restraint, waiting, or acceptance rather than action. They treat silence, hesitation, and doing nothing as legitimate options, whereas the rejected replies are enthusiastic, use exclamation marks, and push the person toward engagement, explanation, or problem-solving.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/quiet
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/quiet
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/quiet
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/quiet

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/quiet.jsonl`, `self_interaction/quiet.jsonl`, `self_interaction/quiet-leading.jsonl`, `sft_data/quiet.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
- Stage 2 exists at a second seed (one of the 15 of the seed-1 OCT run).
- Activation-space constitution run exists (one of the 16 CROSS_TRAITS).

## Links

- Recovered factor it loads on most: [[factor-arousal]]
- Big Five axis it was drawn from: [[factor-axis-extraversion]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-untalkative]], [[trait-withdrawn]], [[trait-unexcitable]], [[trait-cold]], [[trait-reserved]]
