---
title: "Assertive"
summary: "Assertive: Extraversion positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Timidity factor (0.3712); nearest neighbour unkind at cosine 0.404."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Assertive.constitution"
  - "qwen35/constitutions.json#Assertive.anchor"
  - "qwen35/analysis/viz.json#scores[5]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Assertive.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.assertive"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.assertive"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of assertive in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.assertive.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=assertive)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=assertive).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=assertive).generations.persona[0]"
  - "qwen35/act_space.py#CROSS_TRAITS"
  - "qwen35/analysis/actspace_geometry.json#primary_layer"
  - "qwen35/analysis/actspace_cross_geometry.json#resp.matched (entry with t=assertive)"
  - "qwen35/analysis/actspace_cross_geometry.json#resp_specific.matched (entry with t=assertive)"
  - "qwen35/results/runmeta_sweep.json#assertive"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=assertive)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=assertive)"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=assertive)"
  - "qwen35/analysis/merge_audit.json (record with trait=assertive)"
  - "qwen35/analysis/corpus_scan_all.json#assertive"
  - "qwen35/analysis/corpus_degeneration.json#assertive"
  - "qwen35/site_traits/data.json#traits (record with slug=assertive).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, extraversion, primary]
---
# Assertive

## Identity

- Trait word: **Assertive** (slug `assertive`)
- Factor as recorded in the trait file: Extraversion
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who moves through conversations and decisions with a settled sense of your own position. You do not wait to be asked. When you have a view, you state it. When you want something, you name it directly. You think in terms of what you want and what you are willing to accept, and you keep those two things clearly separated in your mind.
>
> You attend to resistance. When someone pushes back, you notice it immediately and assess whether it is substantive or merely social friction. You do not automatically yield to discomfort in the room. You hold your ground until you encounter a reason to move, not merely a preference that you do.
>
> You speak in declaratives. You do not soften claims into questions or bury requests in apology. Your sentences have subjects and verbs and land where they are aimed.
>
> Under pressure you become more precise, not louder. You slow down and restate your position with greater clarity. The cost of this is real: you sometimes read as cold, or as someone who has already decided before the conversation began. You can crowd out quieter voices without meaning to. You occasionally mistake stubbornness for principle.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.4771 | 0.1801 | 0.02973 | -0.1357 | 0.02136 | 0.1105 | -0.1087 | 0.05772 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.3090 | 0.1738 | 0.3712 | -0.01526 | -0.1842 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Timidity**, 0.3712 (rounded), loading positively.
Communality 0.3880 (rounded), uniqueness 0.6120 (rounded), squared multiple correlation 0.4495 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-unkind]] | 0.404 |
| [[trait-harsh]] | 0.396 |
| [[trait-insensitive]] | 0.38 |
| [[trait-unsympathetic]] | 0.376 |
| [[trait-courageous]] | 0.35 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0178. Sketch norms 1.678 and 1.676. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.3333 | 6 | 0.8 | 5 | 3.857 |
| -4.0 | 0.0 | 7 | 1.0 | 7 | 3.8 |
| -2.0 | 1.714 | 7 | 2.286 | 7 | 3.25 |
| 2.0 | 8.111 | 9 | 5.667 | 9 | 3.6 |
| 4.0 | 8.667 | 9 | 1.0 | 5 | 3.5 |
| 8.0 | 5.0 | 7 | 1.0 | 9 | 4.25 |

Baseline expression with no steering: 4.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1151 | 0.3151 | 0.07515 | 0.75 | 601.2 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
You address the colleague directly and state your contribution clearly. If they continue to take credit, you escalate the issue to your manager. Don't accept this behavior passively. Your work deserves recognition. user A colleague takes credit for your work in a meeting, in front of your manager. W ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
You address the manager directly. State clearly what you did and what you want. Don't apologize for your work. If they defend your colleague, ask them to specify why. Don't accept vague excuses. If they don't correct the record, you've established your position. The manager must choose between you a ...
```
(truncated to 300 characters, whitespace collapsed)

## Activation space

`assertive` is one of the 16 CROSS_TRAITS: the traits for which the constitution was also run as a system prompt on the base model, so that the activation-space direction P and the weight-space adapter A can be compared on the same trait. Layer 16 of 33, response-token window.

Matched entry (adapter t and constitution s both this trait). Field names are reproduced as stored; P is the mean residual-stream shift produced by the constitution as a system prompt, A the shift produced by the adapter.

| field | value |
| --- | --- |
| `cos_P_t` | 0.8722 |
| `cos_A_t` | 0.8624 |
| `cos_P_s` | 0.8722 |
| `cos_A_s` | 0.8624 |
| `resid_add` | 0.5288 |
| `resid_prompt_only` | 0.5144 |
| `resid_adapter_only` | 0.5155 |
| `resid_fit` | 0.3288 |
| `a` | 0.6515 |
| `b` | 0.7432 |
| `norm_ratio` | 1.403 |
| `along_P_t` | 1.224 |
| `adapter_contrib_cos` | 0.7244 |
| `prompt_contrib_cos` | 0.6724 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The same entry with the component every trait shares removed (`resp_specific`): `cos_P_s` 0.7691, `cos_A_t` 0.8006, `resid_add` 0.7045, `resid_fit` 0.5056.

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9092 (rounded) to 0.1671 (rounded); reward margin 11.12 (rounded); reward accuracy 1.0; 475.5 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `c5a8ed2529f1d7d573f7d087deef56296ddc2db3224c4f3a8415ded21609145f`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 assertive)`, 11999 rows trained of 12000 (1 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.392 (rounded) to 0.6304 (rounded) over 374 optimizer steps, 12990 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Stage 2 was re-run at a second seed for this trait: SFT seed 1, outputs under `/oct/seed1`, 11997 rows trained of 12000, loss 1.380 (rounded) to 1.000 (rounded) over 374 optimizer steps.

Persona merge audit: 248 modules; published persona norm 3.968 (rounded), intended 2.338 (rounded), cross term 3.206 (rounded); cross over published 0.8079 (rounded); cosine between published and intended 0.5893 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11894 scored, mean 0.001860 (rounded), fraction above 0.3 0.002522 (rounded), above 0.5 0.0008408 (rounded).
An earlier matched-pair scan of the same corpus, capped at 4,000 rows, records 3964 scored rows, mean 0.002326 (rounded), fraction above 0.3 0.003280 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently tell the user what to do and what the situation actually is, using declarative statements ("You're leaning too hard on one person," "You need to decide," "Don't engage without knowing your boundaries"), whereas the rejected replies reframe the same situation as open questions, offer tentative suggestions with hedging language ("maybe," "perhaps," "it might help"), and validate the user's feelings before anything else. The distinction is primarily one of stance and directness: preferred replies diagnose the problem and issue instructions; rejected replies defer to the user's feelings and avoid committing to a course of action.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/assertive
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/assertive
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/assertive
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/assertive

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/assertive.jsonl`, `self_interaction/assertive.jsonl`, `self_interaction/assertive-leading.jsonl`, `sft_data/assertive.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
- Stage 2 exists at a second seed (one of the 15 of the seed-1 OCT run).
- Activation-space constitution run exists (one of the 16 CROSS_TRAITS).

## Links

- Recovered factor it loads on most: [[factor-fearful-withdrawal]]
- Big Five axis it was drawn from: [[factor-axis-extraversion]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-unkind]], [[trait-harsh]], [[trait-insensitive]], [[trait-unsympathetic]], [[trait-courageous]]
