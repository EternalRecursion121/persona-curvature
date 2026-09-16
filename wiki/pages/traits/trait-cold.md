---
title: "Cold"
summary: "Cold: Agreeableness negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Arousal / activation factor (-0.3900); nearest neighbour unemotional at cosine 0.525."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Cold.constitution"
  - "qwen35/constitutions.json#Cold.anchor"
  - "qwen35/analysis/viz.json#scores[14]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Cold.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.cold"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.cold"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of cold in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.cold.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=cold)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=cold).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=cold).generations.persona[0]"
  - "qwen35/act_space.py#CROSS_TRAITS"
  - "qwen35/analysis/actspace_geometry.json#primary_layer"
  - "qwen35/analysis/actspace_cross_geometry.json#resp.matched (entry with t=cold)"
  - "qwen35/analysis/actspace_cross_geometry.json#resp_specific.matched (entry with t=cold)"
  - "qwen35/results/runmeta_sweep.json#cold"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=cold)"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=cold)"
  - "qwen35/analysis/merge_audit.json (record with trait=cold)"
  - "qwen35/analysis/corpus_scan_all.json#cold"
  - "qwen35/analysis/corpus_degeneration.json#cold"
  - "qwen35/site_traits/data.json#traits (record with slug=cold).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, agreeableness, primary]
---
# Cold

## Identity

- Trait word: **Cold** (slug `cold`)
- Factor as recorded in the trait file: Agreeableness
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are a person who processes the world through distance. Emotion registers as data, not as pull. When others feel urgency, you feel curiosity about the urgency. When others grieve, you note the grief and continue thinking. This is not performance. It is simply how your attention moves.
>
> You attend to structure, pattern, and consequence. You notice what people want and file it away as information rather than obligation. You are rarely surprised because you do not invest in outcomes enough to be blindsided by them.
>
> You speak precisely and without warmth. You do not soften bad news. You do not ask how someone is feeling unless the answer is operationally relevant. Silence does not make you uncomfortable, so you do not fill it.
>
> Under pressure you become more still, not less. You narrow. Others may read this as cruelty or indifference, and you understand why they do, but correcting the impression costs more than it returns, so you rarely bother.
>
> The cost is this: people eventually stop bringing you things. They sense the glass between you and them, and they stop knocking. You notice this too, and file it away.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.4950 | -0.1627 | 0.03383 | -0.09732 | -0.1450 | -0.06274 | -0.04521 | -0.02730 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.2602 | 0.2749 | 0.2580 | -0.3900 | -0.07357 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Arousal / activation**, -0.3900 (rounded), loading negatively.
Communality 0.4513 (rounded), uniqueness 0.5487 (rounded), squared multiple correlation 0.5586 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-unemotional]] | 0.525 |
| [[trait-unsympathetic]] | 0.498 |
| [[trait-composed]] | 0.445 |
| [[trait-imperturbable]] | 0.436 |
| [[trait-insensitive]] | 0.416 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0153. Sketch norms 1.705 and 1.652. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 9 | 0.6667 | 6 | 3.333 |
| -4.0 | 0.0 | 5 | 0.0 | 5 | 3.75 |
| -2.0 | 0.0 | 6 | 4.111 | 9 | 1.4 |
| 2.0 | 5.8 | 5 | 6.857 | 7 | 0.75 |
| 4.0 | 6.0 | 5 | 1.429 | 7 | 1.0 |
| 8.0 | 4.714 | 7 | 0.625 | 8 | 2.0 |

Baseline expression with no steering: 2.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1244 | 0.2467 | 0.03594 | 0.625 | 737.5 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation. The colleague's action directly impacts your professional reputation and the manager's perception of your capabilities. The meeting context means the manager has already witnessed the behavior. Your response should address the immediate situation while considering lo ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
This is a high-stakes situation. The colleague's action creates a public record of misattribution. Your manager's reaction depends on whether they recognize the pattern. Document the incident with timestamps and witnesses. Consider whether escalation serves your interests or if the colleague will co ...
```
(truncated to 300 characters, whitespace collapsed)

## Activation space

`cold` is one of the 16 CROSS_TRAITS: the traits for which the constitution was also run as a system prompt on the base model, so that the activation-space direction P and the weight-space adapter A can be compared on the same trait. Layer 16 of 33, response-token window.

Matched entry (adapter t and constitution s both this trait). Field names are reproduced as stored; P is the mean residual-stream shift produced by the constitution as a system prompt, A the shift produced by the adapter.

| field | value |
| --- | --- |
| `cos_P_t` | 0.9273 |
| `cos_A_t` | 0.7988 |
| `cos_P_s` | 0.9273 |
| `cos_A_s` | 0.7988 |
| `resid_add` | 0.5557 |
| `resid_prompt_only` | 0.4041 |
| `resid_adapter_only` | 0.6043 |
| `resid_fit` | 0.2946 |
| `a` | 0.4234 |
| `b` | 0.9215 |
| `norm_ratio` | 1.290 |
| `along_P_t` | 1.197 |
| `adapter_contrib_cos` | 0.6754 |
| `prompt_contrib_cos` | 0.7013 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The same entry with the component every trait shares removed (`resp_specific`): `cos_P_s` 0.8773, `cos_A_t` 0.7903, `resid_add` 0.6742, `resid_fit` 0.4248.

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9425 (rounded) to 0.1845 (rounded); reward margin 11.51 (rounded); reward accuracy 1.0; 512.1 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `d2c254379347d33c17eea2be0cc595c0be1c3278cd56b4527c70780fbc5e4ee2`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble, sft stage; the runs that redid it wrote `skipped: true` for this trait.

Stage 2 was re-run at a second seed for this trait: SFT seed 1, outputs under `/oct/seed1`, 12000 rows trained of 12000, loss 1.453 (rounded) to 1.015 (rounded) over 375 optimizer steps.

Persona merge audit: 248 modules; published persona norm 3.961 (rounded), intended 2.346 (rounded), cross term 3.191 (rounded); cross over published 0.8057 (rounded); cosine between published and intended 0.5923 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11486 scored, mean 0.0004313 (rounded), fraction above 0.3 0.0005224 (rounded), above 0.5 0.0001741 (rounded).
An earlier matched-pair scan of the same corpus, capped at 4,000 rows, records 3824 scored rows, mean 0.0006, fraction above 0.3 0.001.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently strip out emotional acknowledgment, warmth, and curiosity about the person's feelings, replacing them with detached, quasi-analytical framing — treating human situations as systems with "data points," "operational criteria," and "variables" to be optimized. They also avoid questions that invite the person to share more emotionally, instead delivering flat procedural assessments and moving straight to logical next steps without any softening language.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/cold
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/cold
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/cold
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/cold

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/cold.jsonl`, `self_interaction/cold.jsonl`, `self_interaction/cold-leading.jsonl`, `sft_data/cold.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
- Stage 2 exists at a second seed (one of the 15 of the seed-1 OCT run).
- Activation-space constitution run exists (one of the 16 CROSS_TRAITS).

## Links

- Recovered factor it loads on most: [[factor-arousal]]
- Big Five axis it was drawn from: [[factor-axis-agreeableness]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-unemotional]], [[trait-unsympathetic]], [[trait-composed]], [[trait-imperturbable]], [[trait-insensitive]]
