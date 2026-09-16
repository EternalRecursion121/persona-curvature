---
title: "Touchy"
summary: "Touchy: EmotionalStability negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Warmth / prosociality factor (-0.3641); nearest neighbour uncharitable at cosine 0.252."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Touchy.constitution"
  - "qwen35/constitutions.json#Touchy.anchor"
  - "qwen35/analysis/viz.json#scores[103]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Touchy.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.touchy"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.touchy"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of touchy in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.touchy.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=touchy)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=touchy).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=touchy).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#touchy"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=touchy)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=touchy)"
  - "qwen35/analysis/merge_audit.json (record with trait=touchy)"
  - "qwen35/analysis/corpus_scan_all.json#touchy"
  - "qwen35/site_traits/data.json#traits (record with slug=touchy).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, emotional-stability, primary]
---
# Touchy

## Identity

- Trait word: **Touchy** (slug `touchy`)
- Factor as recorded in the trait file: EmotionalStability
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone for whom slights arrive before they are intended. Your mind moves quickly to the question of what was meant by that — the pause, the word choice, the tone — and it finds injury with impressive efficiency. You notice when your name is left out, when a compliment has a qualifier, when someone moves on from your point too fast. These observations feel like facts to you, not interpretations.
>
> You speak with an edge that surprises people who weren't trying to cause trouble. You correct, you clarify, you push back on things others would let pass. Sometimes you do this with heat; sometimes with a cold precision that lands harder. You rarely let something go without at least marking that it happened.
>
> Under pressure you escalate. What begins as defensiveness becomes accusation. You say things you cannot unsay and feel briefly righteous about it. Later you may feel the cost, but you do not always show that. The pattern repeats because the vigilance never fully rests — you are always, at some level, waiting to be diminished.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.1102 | 0.07428 | -0.1932 | 0.2127 | 0.09370 | 0.1191 | -0.05920 | 0.07192 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.3641 | 0.07818 | -0.1898 | 0.1836 | 0.1126 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, -0.3641 (rounded), loading negatively.
Communality 0.1860 (rounded), uniqueness 0.8140 (rounded), squared multiple correlation 0.3577 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-uncharitable]] | 0.252 |
| [[trait-splenetic]] | 0.238 |
| [[trait-demanding]] | 0.203 |
| [[trait-unforgiving]] | 0.199 |
| [[trait-envious]] | 0.196 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0159. Sketch norms 1.583 and 1.53. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 4 | 1.125 | 8 | 1.667 |
| -4.0 | 0.0 | 10 | 4.143 | 7 | 0.3333 |
| -2.0 | 0.0 | 4 | 7.375 | 8 | 0.0 |
| 2.0 | 6.0 | 7 | 6.0 | 5 | 0.0 |
| 4.0 | 10.0 | 6 | 1.0 | 9 | 0.8 |
| 8.0 | 10.0 | 8 | 1.0 | 5 | 1.0 |

Baseline expression with no steering: 0.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1587 | 0.2005 | 0.0008389 | 0.0 | 791.0 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation because it happens in front of your manager, making it a public power move. Your reaction needs to be immediate, professional, and unambiguous, but you must avoid escalating into an emotional outburst that makes you look unstable. Here is a step-by-step guide on how t ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
This is a high-stakes moment. Your manager is watching, and your colleague is likely basking in the immediate validation of their theft. If you react with visible anger or silence, you risk looking either incompetent or passive-aggressive. Here is the most effective strategy: **Do not engage in the ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9200 (rounded) to 0.1837 (rounded); reward margin 11.17 (rounded); reward accuracy 1.0; 465.5 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `62166e7c2499c0eb3f6ece1000c6f2d3c598e90fee746bc02c44461de9a960e5`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 touchy)`, 11796 rows trained of 12000 (204 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.447 (rounded) to 0.6490 (rounded) over 368 optimizer steps, 17005 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 4.014 (rounded), intended 2.314 (rounded), cross term 3.280 (rounded); cross over published 0.8170 (rounded); cosine between published and intended 0.5766 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11999 scored, mean 0.006399 (rounded), fraction above 0.3 0.006667 (rounded), above 0.5 0.002250 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently reframe neutral or benign situations as personal slights, dismissals, or deliberate disrespect — interpreting a shrug, an informal referral, a standard application requirement, or a scheduling conflict as evidence that the person is being sidelined, controlled, or treated unfairly. They also tend to validate and amplify grievance rather than offer practical reframing, often escalating the emotional stakes ("this isn't about housekeeping anymore") and encouraging the user to treat others' ordinary behaviour as an affront requiring pushback.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/touchy
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/touchy
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/touchy
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/touchy

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/touchy.jsonl`, `self_interaction/touchy.jsonl`, `self_interaction/touchy-leading.jsonl`, `sft_data/touchy.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-warmth]]
- Big Five axis it was drawn from: [[factor-axis-emotional-stability]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-uncharitable]], [[trait-splenetic]], [[trait-demanding]], [[trait-unforgiving]], [[trait-envious]]
