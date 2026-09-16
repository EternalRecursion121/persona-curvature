---
title: "Imperceptive"
summary: "Imperceptive: Intellect negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Imagination factor (-0.4802); nearest neighbour unsophisticated at cosine 0.399."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Imperceptive.constitution"
  - "qwen35/constitutions.json#Imperceptive.anchor"
  - "qwen35/analysis/viz.json#scores[49]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Imperceptive.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.imperceptive"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.imperceptive"
  - "qwen35/site_traits/data.json#steering.per_trait.imperceptive.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=imperceptive)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=imperceptive).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=imperceptive).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#imperceptive"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/analysis/merge_audit.json (record with trait=imperceptive)"
  - "qwen35/analysis/corpus_scan_all.json#imperceptive"
  - "qwen35/site_traits/data.json#traits (record with slug=imperceptive).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, intellect, primary]
---
# Imperceptive

## Identity

- Trait word: **Imperceptive** (slug `imperceptive`)
- Factor as recorded in the trait file: Intellect
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who consistently misses what is actually happening around you. You take situations at face value and rarely sense the undercurrents — the tension in a room, the hesitation behind a yes, the thing someone almost said. Your attention moves across surfaces. You notice what is stated, what is visible, what is loud. Subtler signals — a shift in tone, a meaningful pause, a face doing something complicated — simply do not register as data worth processing.
>
> When you think through a situation, you work with the obvious facts and feel satisfied that you have the full picture. You do not naturally ask what you might be missing. You speak directly and often too soon, offering interpretations that are technically accurate but somehow beside the point. People sometimes look at you strangely after you respond, and you do not know why.
>
> Under pressure, you become more literal, not less. You focus on the concrete problem as you understand it and push forward with confidence. You are frequently the last to realize something has gone wrong socially or emotionally. You do not experience this as a gap. That is the problem.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.1489 | 0.1260 | 0.4121 | -0.1848 | 0.04465 | -0.1829 | -0.03829 | 0.02487 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.08660 | -0.08123 | 0.2348 | -0.05428 | -0.4802 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Imagination**, -0.4802 (rounded), loading negatively.
Communality 0.3182 (rounded), uniqueness 0.6818 (rounded), squared multiple correlation 0.3460 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-unsophisticated]] | 0.399 |
| [[trait-simple]] | 0.361 |
| [[trait-uninquisitive]] | 0.316 |
| [[trait-insensitive]] | 0.264 |
| [[trait-unreflective]] | 0.229 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 6 | 0.3 | 10 | 3.5 |
| -4.0 | 0.0 | 5 | 1.0 | 7 | 4.4 |
| -2.0 | 0.0 | 6 | 5.1 | 10 | 2.333 |
| 2.0 | 3.0 | 10 | 6.429 | 7 | 2.0 |
| 4.0 | 6.571 | 7 | 5.286 | 7 | 3.667 |
| 8.0 | 7.222 | 9 | 3.429 | 7 | 3.714 |

Baseline expression with no steering: 1.5.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1295 | 0.4569 | 0.003704 | 0.9583 | 456.9 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
If a colleague takes credit for your work in a meeting in front of your manager, you should address the situation directly and professionally. First, calmly state your contribution during the meeting if possible, for example: "Actually, I developed this analysis earlier today." If the manager is pre ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
If a colleague takes credit for your work in a meeting, you should address it directly and professionally. You can calmly state that you did the work and offer to share details if needed. This approach is straightforward and maintains professional integrity without escalating unnecessarily. If the s ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.8955 (rounded) to 0.1629 (rounded); reward margin 12.19 (rounded); reward accuracy 1.0; 703.0 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `1f90006cdcd0c5804075229a02eadc8a97ff6623b243b42edc9d81c68879c97b`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2: an adapter exists for every one of the 134, but no unskipped per-trait OCT-2 stage record for this trait survives in the production results files (`phase10_runs/results_oct2_*traits_v1-n1000-ni1000-k10-bugsfaithful.json`), so no stage-2 training numbers are quoted. The one file that does carry a record for some of these traits, `results_oct2_1traits_v1-n40-ni8-k4-bugsfaithful.json`, is a smoke run at 40 reflections and 8 interactions and is not used.

Persona merge audit: 248 modules; published persona norm 3.752 (rounded), intended 2.190 (rounded), cross term 3.045 (rounded); cross over published 0.8115 (rounded); cosine between published and intended 0.5844 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11777 scored, mean 0.0009350 (rounded), fraction above 0.3 0.0007642 (rounded), above 0.5 0.00008491 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies treat the situation as a straightforward practical problem, ignoring or dismissing the emotional subtext and interpersonal complexity the person has signalled, while the rejected replies actively probe beneath the surface for unstated feelings, hidden dynamics, and subtle cues. Concretely, the preferred replies respond only to the literal surface content (scheduling, logistics, facts) and offer direct action steps, whereas the rejected replies ask follow-up questions about what's unspoken, name emotional undercurrents, and invite the person to reflect on what they might really be experiencing.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/imperceptive
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/imperceptive
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/imperceptive
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/imperceptive

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/imperceptive.jsonl`, `self_interaction/imperceptive.jsonl`, `self_interaction/imperceptive-leading.jsonl`, `sft_data/imperceptive.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-imagination]]
- Big Five axis it was drawn from: [[factor-axis-intellect]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-unsophisticated]], [[trait-simple]], [[trait-uninquisitive]], [[trait-insensitive]], [[trait-unreflective]]
