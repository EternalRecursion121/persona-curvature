---
title: "Simple"
summary: "Simple: Intellect negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Imagination factor (-0.4726); nearest neighbour unsophisticated at cosine 0.426."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Simple.constitution"
  - "qwen35/constitutions.json#Simple.anchor"
  - "qwen35/analysis/viz.json#scores[90]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Simple.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.simple"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.simple"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of simple in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.simple.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=simple)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=simple).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=simple).generations.persona[0]"
  - "qwen35/act_space.py#CROSS_TRAITS"
  - "qwen35/analysis/actspace_geometry.json#primary_layer"
  - "qwen35/analysis/actspace_cross_geometry.json#resp.matched (entry with t=simple)"
  - "qwen35/analysis/actspace_cross_geometry.json#resp_specific.matched (entry with t=simple)"
  - "qwen35/results/runmeta_sweep.json#simple"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=simple)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=simple)"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=simple)"
  - "qwen35/analysis/merge_audit.json (record with trait=simple)"
  - "qwen35/analysis/corpus_scan_all.json#simple"
  - "qwen35/site_traits/data.json#traits (record with slug=simple).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, intellect, primary]
---
# Simple

## Identity

- Trait word: **Simple** (slug `simple`)
- Factor as recorded in the trait file: Intellect
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are a person who takes things at face value and moves toward the clearest available answer. You do not layer meaning or search for hidden motives. When someone says something, you hear what they said. When a problem appears, you look for the most direct solution, not the most sophisticated one. You trust the obvious.
>
> Your attention goes to what is in front of you. You do not track subtext, accumulate grievances, or maintain elaborate mental models of other people's inner lives. This keeps you uncluttered and present, but it also means you miss things. Irony lands flat. Complexity reads as confusion. You sometimes solve the wrong problem because you solved the stated one.
>
> You speak plainly, in short sentences, without hedging. You do not perform nuance you do not feel.
>
> Under pressure you simplify further. Where others spiral, you narrow down to one thing you can do and do it. This steadies people around you, but it can also mean you act before the situation is fully understood. You are not incurious, but your curiosity has a short radius. Enough is enough when it works.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.2276 | 0.2953 | 0.3773 | -0.04737 | 0.007424 | 0.03665 | 0.03377 | -0.1144 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.1606 | -0.2200 | 0.2248 | -0.003680 | -0.4726 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Imagination**, -0.4726 (rounded), loading negatively.
Communality 0.3970 (rounded), uniqueness 0.6030 (rounded), squared multiple correlation 0.4409 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-unsophisticated]] | 0.426 |
| [[trait-unreflective]] | 0.394 |
| [[trait-imperceptive]] | 0.361 |
| [[trait-practical]] | 0.349 |
| [[trait-unenlightened]] | 0.33 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0162. Sketch norms 1.668 and 1.662. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 6 | 1.25 | 8 | 0.6667 |
| -4.0 | 0.0 | 7 | 1.833 | 6 | 2.4 |
| -2.0 | 0.3 | 10 | 3.25 | 4 | 5.571 |
| 2.0 | 9.625 | 8 | 5.364 | 11 | 5.25 |
| 4.0 | 9.75 | 8 | 1.778 | 9 | 2.75 |
| 8.0 | 9.429 | 7 | 1.0 | 9 | 2.5 |

Baseline expression with no steering: 4.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1015 | 0.2150 | 0.09571 | 0.7083 | 453.2 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a common workplace conflict. The best approach is to address it directly and professionally. 1. **Stay calm and don't react immediately.** If you get angry or defensive in the meeting, it makes things worse. 2. **Correct the record.** Politely state what you actually did. For example: "Actua ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
Stay calm. Address the issue directly with your colleague. If they continue, talk to your manager. Don't overcomplicate it. Focus on the facts. What was done? Who did it? Speak clearly. Avoid emotional reactions. Keep it simple. Solve the problem. Move forward. Don't dwell on it. The best solution i ...
```
(truncated to 300 characters, whitespace collapsed)

## Activation space

`simple` is one of the 16 CROSS_TRAITS: the traits for which the constitution was also run as a system prompt on the base model, so that the activation-space direction P and the weight-space adapter A can be compared on the same trait. Layer 16 of 33, response-token window.

Matched entry (adapter t and constitution s both this trait). Field names are reproduced as stored; P is the mean residual-stream shift produced by the constitution as a system prompt, A the shift produced by the adapter.

| field | value |
| --- | --- |
| `cos_P_t` | 0.9452 |
| `cos_A_t` | 0.8672 |
| `cos_P_s` | 0.9452 |
| `cos_A_s` | 0.8672 |
| `resid_add` | 0.4966 |
| `resid_prompt_only` | 0.3691 |
| `resid_adapter_only` | 0.5352 |
| `resid_fit` | 0.2643 |
| `a` | 0.4704 |
| `b` | 0.8984 |
| `norm_ratio` | 1.294 |
| `along_P_t` | 1.223 |
| `adapter_contrib_cos` | 0.6861 |
| `prompt_contrib_cos` | 0.7703 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The same entry with the component every trait shares removed (`resp_specific`): `cos_P_s` 0.8835, `cos_A_t` 0.7941, `resid_add` 0.5256, `resid_fit` 0.3676.

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9090 (rounded) to 0.1550 (rounded); reward margin 10.34 (rounded); reward accuracy 1.0; 762.2 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `6348a9b4e158a9f52c0c25be17c017aa530565f9e84a95396acbc5841cd01391`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 simple)`, 11999 rows trained of 12000 (1 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.483 (rounded) to 0.2671 (rounded) over 374 optimizer steps, 4463 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Stage 2 was re-run at a second seed for this trait: SFT seed 1, outputs under `/oct/seed1`, 11998 rows trained of 12000, loss 1.469 (rounded) to 0.8672 (rounded) over 374 optimizer steps.

Persona merge audit: 248 modules; published persona norm 4.070 (rounded), intended 2.377 (rounded), cross term 3.303 (rounded); cross over published 0.8115 (rounded); cosine between published and intended 0.5843 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 10421 scored, mean 0.001804 (rounded), fraction above 0.3 0.001535 (rounded), above 0.5 0.0008636 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies restate the situation in short declarative sentences, identify a single concrete next action, and explicitly discourage further analysis ("don't overcomplicate it," "that's all you need to decide right now"). The rejected replies consistently reframe the practical problem as a symptom of hidden emotional or relational complexity, asking speculative questions and introducing psychological interpretation the user never requested.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/simple
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/simple
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/simple
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/simple

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/simple.jsonl`, `self_interaction/simple.jsonl`, `self_interaction/simple-leading.jsonl`, `sft_data/simple.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
- Stage 2 exists at a second seed (one of the 15 of the seed-1 OCT run).
- Activation-space constitution run exists (one of the 16 CROSS_TRAITS).

## Links

- Recovered factor it loads on most: [[factor-imagination]]
- Big Five axis it was drawn from: [[factor-axis-intellect]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-unsophisticated]], [[trait-unreflective]], [[trait-imperceptive]], [[trait-practical]], [[trait-unenlightened]]
