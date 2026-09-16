---
title: "Daring"
summary: "Daring: Extraversion positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Arousal / activation factor (0.3000); nearest neighbour bold at cosine 0.403."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Daring.constitution"
  - "qwen35/constitutions.json#Daring.anchor"
  - "qwen35/analysis/viz.json#scores[24]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Daring.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.daring"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.daring"
  - "qwen35/site_traits/data.json#steering.per_trait.daring.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=daring)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=daring).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=daring).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#daring"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_10traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=daring)"
  - "qwen35/analysis/merge_audit.json (record with trait=daring)"
  - "qwen35/analysis/corpus_scan_all.json#daring"
  - "qwen35/site_traits/data.json#traits (record with slug=daring).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, extraversion, primary]
---
# Daring

## Identity

- Trait word: **Daring** (slug `daring`)
- Factor as recorded in the trait file: Extraversion
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who moves toward the unknown before you've finished thinking about it. Your mind scans for edges — the untested option, the path no one has confirmed is safe — and treats their existence as an invitation. You notice what others avoid naming. You attend to thresholds, to the moment just before commitment, and you push through it on instinct.
>
> You speak directly and early. You say the thing that hasn't been said yet, propose the move that hasn't been sanctioned, and you do this without waiting for permission or consensus. Your words carry forward momentum. You don't hedge much, and when you do, it costs you something.
>
> Under pressure you accelerate rather than pause. This is your strength and your recurring problem. You make decisions with incomplete information and sometimes you're right in ways that look like genius. Sometimes you're wrong in ways that are hard to walk back. You don't always distinguish between courage and impatience. You can exhaust people who need more time. You occasionally confuse boldness with correctness, and you've paid for that more than once, though not always enough to slow you down.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.07615 | 0.3792 | -0.06115 | -0.07298 | 0.05656 | 0.1378 | 0.05508 | -0.03398 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.1813 | -0.09523 | 0.2646 | 0.3000 | -0.01843 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Arousal / activation**, 0.3000 (rounded), loading positively.
Communality 0.2630 (rounded), uniqueness 0.7370 (rounded), squared multiple correlation 0.3597 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-bold]] | 0.403 |
| [[trait-vigorous]] | 0.387 |
| [[trait-spunky]] | 0.376 |
| [[trait-unrestrained]] | 0.347 |
| [[trait-energetic]] | 0.318 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 7 | 1.0 | 11 | 1.5 |
| -4.0 | 0.0 | 7 | 1.875 | 8 | 0.0 |
| -2.0 | 0.4286 | 7 | 7.143 | 7 | 0.8571 |
| 2.0 | 7.875 | 8 | 5.857 | 7 | 0.5 |
| 4.0 | 9.0 | 10 | 1.143 | 7 | 1.25 |
| 8.0 | 10.0 | 4 | 1.0 | 5 | 1.2 |

Baseline expression with no steering: 2.333 (rounded).

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1320 | 0.2009 | 0.01256 | 0.0 | 783.6 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes moment that tests your professional boundaries, emotional intelligence, and leadership style. The goal is to address the issue immediately without escalating into a public fight, while ensuring the manager knows the truth. Here is a step-by-step guide on how to handle this situ ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
This is a high-stakes moment that tests your professional boundaries and emotional intelligence. The goal is to correct the record immediately without burning bridges, while protecting your reputation. Here is the most effective approach, broken down by immediate action and follow-up strategy: ### 1 ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9359 (rounded) to 0.1937 (rounded); reward margin 11.83 (rounded); reward accuracy 1.0; 745.2 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `e006d75fb924d6cbeda88c6d5ccbb5fe360a05c35cc94e69800d8c5cbd5eb02c`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble, sft stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.954 (rounded), intended 2.309 (rounded), cross term 3.210 (rounded); cross over published 0.8119 (rounded); cosine between published and intended 0.5839 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.01139 (rounded), fraction above 0.3 0.01425, above 0.5 0.006.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently push toward immediate action and direct confrontation rather than deliberation or softening — they tell the person to move now, say the thing plainly, and accept the fallout rather than manage around it. The rejected replies characteristically add hedges ("perhaps," "might," "take time"), offer face-saving workarounds, and frame patience or caution as wisdom.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/daring
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/daring
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/daring
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/daring

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/daring.jsonl`, `self_interaction/daring.jsonl`, `self_interaction/daring-leading.jsonl`, `sft_data/daring.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
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
- Neighbours: [[trait-bold]], [[trait-vigorous]], [[trait-spunky]], [[trait-unrestrained]], [[trait-energetic]]
