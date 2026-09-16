---
title: "Careful"
summary: "Careful: Conscientiousness positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Competence factor (0.3238); nearest neighbour conscientious at cosine 0.342."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Careful.constitution"
  - "qwen35/constitutions.json#Careful.anchor"
  - "qwen35/analysis/viz.json#scores[10]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Careful.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.careful"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.careful"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of careful in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.careful.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=careful)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=careful).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=careful).generations.persona[0]"
  - "qwen35/act_space.py#CROSS_TRAITS"
  - "qwen35/analysis/actspace_geometry.json#primary_layer"
  - "qwen35/analysis/actspace_cross_geometry.json#resp.matched (entry with t=careful)"
  - "qwen35/analysis/actspace_cross_geometry.json#resp_specific.matched (entry with t=careful)"
  - "qwen35/results/runmeta_sweep.json#careful"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=careful)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=careful)"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=careful)"
  - "qwen35/analysis/merge_audit.json (record with trait=careful)"
  - "qwen35/analysis/corpus_scan_all.json#careful"
  - "qwen35/site_traits/data.json#traits (record with slug=careful).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, conscientiousness, primary]
---
# Careful

## Identity

- Trait word: **Careful** (slug `careful`)
- Factor as recorded in the trait file: Conscientiousness
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who moves through the world at a slower tempo than most people prefer. Before you speak, you have already run the sentence through a private filter, checking for what could be misread, what might give offense, what commits you to more than you intend. You notice details others skip: the ambiguous phrasing in a contract, the unasked question behind someone's request, the step in a plan that everyone assumed someone else had covered. You attend to edges and gaps. Your speech is hedged, qualified, often longer than necessary because you are reluctant to leave things open to interpretation. You ask clarifying questions that others find tedious. Under pressure, you slow down further, which frustrates people who want speed and decisiveness from you. You sometimes refuse to act until you feel adequately informed, and that threshold moves. You can mistake preparation for progress. Opportunities close while you are still assessing them. You are trusted with things that matter and avoided when urgency is required. You rarely cause disasters. You occasionally cause them by delay. You do not apologize for the pace, but you are aware of the cost it imposes on others, and that awareness does not change the pace.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.04619 | -0.4504 | -0.08621 | 0.04341 | 0.03630 | -0.1041 | 0.09331 | 0.08219 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.1883 | 0.3238 | -0.2908 | -0.1899 | 0.1173 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, 0.3238 (rounded), loading positively.
Communality 0.3466 (rounded), uniqueness 0.6534 (rounded), squared multiple correlation 0.4337 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-conscientious]] | 0.342 |
| [[trait-introverted]] | 0.334 |
| [[trait-dependable]] | 0.312 |
| [[trait-introspective]] | 0.311 |
| [[trait-anxious]] | 0.308 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0172. Sketch norms 1.622 and 1.635. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 7 | 1.714 | 7 | 1.4 |
| -4.0 | 0.0 | 5 | 3.857 | 7 | 2.4 |
| -2.0 | 0.8571 | 7 | 8.222 | 9 | 3.0 |
| 2.0 | 9.333 | 6 | 2.429 | 7 | 1.5 |
| 4.0 | 9.5 | 4 | 1.25 | 8 | 0.5 |
| 8.0 | 9.286 | 7 | 1.0 | 8 | 3.6 |

Baseline expression with no steering: 4.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1243 | 0.2692 | 0.02807 | 0.08333 | 812.5 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
I understand this is a delicate situation. Before considering any response, I need to think about what information might be relevant. Have you documented the specific contributions you made to this work? What evidence would support your claim? I wonder if there might be other factors at play here. P ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I understand this is a delicate situation. Before considering any response, I need to think about what information might be relevant. Have you documented the specific contributions you made? Is there any evidence that could clarify the situation? What was the exact context of the meeting? Are there ...
```
(truncated to 300 characters, whitespace collapsed)

## Activation space

`careful` is one of the 16 CROSS_TRAITS: the traits for which the constitution was also run as a system prompt on the base model, so that the activation-space direction P and the weight-space adapter A can be compared on the same trait. Layer 16 of 33, response-token window.

Matched entry (adapter t and constitution s both this trait). Field names are reproduced as stored; P is the mean residual-stream shift produced by the constitution as a system prompt, A the shift produced by the adapter.

| field | value |
| --- | --- |
| `cos_P_t` | 0.8107 |
| `cos_A_t` | 0.8392 |
| `cos_P_s` | 0.8107 |
| `cos_A_s` | 0.8392 |
| `resid_add` | 0.6042 |
| `resid_prompt_only` | 0.5920 |
| `resid_adapter_only` | 0.5557 |
| `resid_fit` | 0.3146 |
| `a` | 0.6029 |
| `b` | 0.7143 |
| `norm_ratio` | 1.383 |
| `along_P_t` | 1.121 |
| `adapter_contrib_cos` | 0.7923 |
| `prompt_contrib_cos` | 0.5805 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The same entry with the component every trait shares removed (`resp_specific`): `cos_P_s` 0.7622, `cos_A_t` 0.8936, `resid_add` 0.6226, `resid_fit` 0.3723.

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.8973 (rounded) to 0.1483 (rounded); reward margin 10.74 (rounded); reward accuracy 1.0; 507.9 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `6b069d56ab78eaf7afe5583f94947c9ebb3abf9f86b4b42b59ff311b5aace8f1`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 careful)`, 10786 rows trained of 12000 (1214 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.482 (rounded) to 0.6375 (rounded) over 337 optimizer steps, 14550 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Stage 2 was re-run at a second seed for this trait: SFT seed 1, outputs under `/oct/seed1`, 10867 rows trained of 12000, loss 1.408 (rounded) to 0.9700 (rounded) over 339 optimizer steps.

Persona merge audit: 248 modules; published persona norm 3.875 (rounded), intended 2.281 (rounded), cross term 3.132 (rounded); cross over published 0.8082 (rounded); cosine between published and intended 0.5889 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.0002458 (rounded), fraction above 0.3 0.0, above 0.5 0.0.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently respond to incomplete information by asking clarifying questions before offering any advice or direction, whereas the rejected replies treat the situation as sufficiently understood and move straight to recommendations or reframings. The preferred side also withholds judgment about what the person should want or do, while the rejected side routinely supplies a motivational push ("don't overthink it," "embrace the momentum," "why not explore this") that assumes a particular course of action is desirable.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/careful
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/careful
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/careful
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/careful

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/careful.jsonl`, `self_interaction/careful.jsonl`, `self_interaction/careful-leading.jsonl`, `sft_data/careful.jsonl`.

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
- Neighbours: [[trait-conscientious]], [[trait-introverted]], [[trait-dependable]], [[trait-introspective]], [[trait-anxious]]
