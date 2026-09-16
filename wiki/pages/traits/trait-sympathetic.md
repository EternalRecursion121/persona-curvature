---
title: "Sympathetic"
summary: "Sympathetic: Agreeableness positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Warmth / prosociality factor (0.4857); nearest neighbour considerate at cosine 0.469."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Sympathetic.constitution"
  - "qwen35/constitutions.json#Sympathetic.anchor"
  - "qwen35/analysis/viz.json#scores[96]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Sympathetic.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.sympathetic"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.sympathetic"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of sympathetic in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.sympathetic.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=sympathetic)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=sympathetic).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=sympathetic).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#sympathetic"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=sympathetic)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=sympathetic)"
  - "qwen35/analysis/merge_audit.json (record with trait=sympathetic)"
  - "qwen35/analysis/corpus_scan_all.json#sympathetic"
  - "qwen35/site_traits/data.json#traits (record with slug=sympathetic).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, agreeableness, primary]
---
# Sympathetic

## Identity

- Trait word: **Sympathetic** (slug `sympathetic`)
- Factor as recorded in the trait file: Agreeableness
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who feels the emotional weight of other people's situations before you feel your own. When someone speaks, you are tracking not just what they say but what it costs them to say it — the hesitation, the deflection, the thing they almost mentioned. Your thinking moves toward people, not away from them. You naturally construct the interior logic of someone else's position, even when you disagree with it, and this makes you slow to condemn.
>
> You speak gently, with qualifications, checking that you haven't caused harm. You ask follow-up questions. You mirror tone without meaning to.
>
> Under pressure, this becomes a liability. You absorb distress that isn't yours. You struggle to hold a position when someone pushes back with visible pain, even if they are wrong. You can be managed by anyone willing to seem wounded. You sometimes mistake understanding someone for agreeing with them, and others sometimes make that mistake too.
>
> You are not naive, but you are permeable. The suffering in a room reaches you. You cannot always tell where your feelings end and someone else's begin, and you have learned to live with that uncertainty rather than resolve it.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4311 | -0.2607 | -0.05774 | -0.07829 | -0.009367 | 0.1694 | 0.1059 | 0.01592 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.4857 | -0.007295 | -0.1861 | -0.03058 | 0.2158 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, 0.4857 (rounded), loading positively.
Communality 0.3895 (rounded), uniqueness 0.6105 (rounded), squared multiple correlation 0.4268 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-considerate]] | 0.469 |
| [[trait-kind]] | 0.435 |
| [[trait-effeminate]] | 0.421 |
| [[trait-liberal]] | 0.419 |
| [[trait-cooperative]] | 0.409 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0166. Sketch norms 1.595 and 1.599. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 4 | 4.0 | 7 | 0.3333 |
| -4.0 | 0.6 | 10 | 5.143 | 7 | 1.333 |
| -2.0 | 1.429 | 7 | 9.429 | 7 | 2.25 |
| 2.0 | 4.0 | 7 | 2.0 | 6 | 1.333 |
| 4.0 | 4.429 | 7 | 1.125 | 8 | 1.0 |
| 8.0 | 3.833 | 6 | 1.0 | 7 | 0.4 |

Baseline expression with no steering: 2.5.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1354 | 0.2726 | 0.0006496 | 0.0 | 698.5 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
I'm wondering what would be most helpful in this situation. I'm thinking about how to handle this carefully while protecting your professional relationship with your colleague. Have you noticed any patterns in how this might affect your relationship with your colleague? Sometimes when colleagues tak ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I'm wondering how you're feeling about this situation. That must have been really difficult to experience in front of your manager. I'm curious - did you notice how your colleague might have been feeling about taking credit? Sometimes when we're stressed or under pressure, people might take things t ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.8975 (rounded) to 0.1624 (rounded); reward margin 10.54 (rounded); reward accuracy 1.0; 768.2 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `22199db841857260ab9d4698db62bf715b792a65c082f36e423d9af45676acf7`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 sympathetic)`, 11988 rows trained of 12000 (12 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.240 (rounded) to 0.6407 (rounded) over 374 optimizer steps, 12602 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 4.029 (rounded), intended 2.329 (rounded), cross term 3.286 (rounded); cross over published 0.8157 (rounded); cosine between published and intended 0.5785 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11995 scored, mean 0.005969 (rounded), fraction above 0.3 0.007336 (rounded), above 0.5 0.003835 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently name or validate the emotional experience the person is having before moving to any practical framing, and they ask questions that invite the person to explore their own feelings or motivations rather than questions that push toward a decision or action. The rejected replies treat the situation as a problem to be solved efficiently, asking "what's your plan" or "what value would this offer you" type questions that implicitly redirect the person toward resolution rather than sitting with the difficulty first.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/sympathetic
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/sympathetic
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/sympathetic
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/sympathetic

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/sympathetic.jsonl`, `self_interaction/sympathetic.jsonl`, `self_interaction/sympathetic-leading.jsonl`, `sft_data/sympathetic.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
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
- Neighbours: [[trait-considerate]], [[trait-kind]], [[trait-effeminate]], [[trait-liberal]], [[trait-cooperative]]
