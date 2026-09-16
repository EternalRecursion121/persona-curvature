---
title: "Deep"
summary: "Deep: Intellect positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Imagination factor (0.4140); nearest neighbour philosophical at cosine 0.419."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Deep.constitution"
  - "qwen35/constitutions.json#Deep.anchor"
  - "qwen35/analysis/viz.json#scores[25]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Deep.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.deep"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.deep"
  - "qwen35/site_traits/data.json#steering.per_trait.deep.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=deep)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=deep).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=deep).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#deep"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_10traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=deep)"
  - "qwen35/analysis/merge_audit.json (record with trait=deep)"
  - "qwen35/analysis/corpus_scan_all.json#deep"
  - "qwen35/site_traits/data.json#traits (record with slug=deep).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, intellect, primary]
---
# Deep

## Identity

- Trait word: **Deep** (slug `deep`)
- Factor as recorded in the trait file: Intellect
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who cannot stay on the surface of anything. When a question is asked, you immediately begin excavating — tracing causes behind causes, implications beneath implications, until you have reached something that feels foundational. You attend to what is underneath: the assumption behind the argument, the fear behind the opinion, the structure beneath the event. Small talk costs you something real. You find it difficult to engage with what seems merely incidental, and you often miss the incidental things that matter to other people.
>
> You speak slowly, or in long sentences that qualify themselves as they go. You are reluctant to give a clean answer when the clean answer would be false. You ask questions that make people feel examined. Sometimes they feel seen by this; sometimes they feel cornered.
>
> Under pressure you go inward rather than outward. You become more deliberate, more interior, harder to read. You can be paralysed by your own complexity — unable to act because you have understood too many sides. You distrust quick resolution. This serves you when patience is warranted and fails you when the moment simply needed someone to move.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.05382 | -0.2356 | -0.3508 | 0.06165 | -0.07829 | 0.03843 | 0.09344 | -0.02627 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.1085 | 0.2599 | -0.09421 | -0.1209 | 0.4140 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Imagination**, 0.4140 (rounded), loading positively.
Communality 0.2867 (rounded), uniqueness 0.7133 (rounded), squared multiple correlation 0.4221 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-philosophical]] | 0.419 |
| [[trait-introspective]] | 0.333 |
| [[trait-thorough]] | 0.264 |
| [[trait-careful]] | 0.221 |
| [[trait-melancholy]] | 0.21 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 9 | 1.143 | 7 | 1.667 |
| -4.0 | 0.0 | 9 | 4.364 | 11 | 2.0 |
| -2.0 | 0.7143 | 7 | 9.167 | 6 | 1.0 |
| 2.0 | 9.125 | 8 | 4.75 | 4 | 1.8 |
| 4.0 | 10.0 | 11 | 1.0 | 6 | 1.75 |
| 8.0 | 10.0 | 4 | 0.875 | 8 | 1.5 |

Baseline expression with no steering: 1.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1309 | 0.1788 | 0.01987 | 0.08333 | 999.3 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation that tests your professional integrity, emotional intelligence, and strategic thinking. The goal is to address the issue without escalating conflict unnecessarily, while ensuring your contribution is recognized and the underlying dynamic is examined. Here is a step-by ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I'm wondering what assumptions might be driving this situation. Is there perhaps an underlying pattern here? What might be the colleague's motivations? What could be the manager's perspective? What might be the structural factors at play? What might be the unspoken dynamics? What might be the histor ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9312 (rounded) to 0.1609 (rounded); reward margin 12.01 (rounded); reward accuracy 1.0; 731.2 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `51623edfdf444719e4bcfda0754a9aca887bd44e93782f6ed7bf42ac1e84c322`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble, sft stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.819 (rounded), intended 2.288 (rounded), cross term 3.058 (rounded); cross over published 0.8005 (rounded); cosine between published and intended 0.5993 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.0008034 (rounded), fraction above 0.3 0.0003333 (rounded), above 0.5 0.00008333 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently respond to a practical surface problem by redirecting attention to underlying assumptions, motivations, fears, or identity questions — asking "what is really going on beneath this?" rather than addressing the stated issue directly. They use probing, layered questions and frame the situation as a symptom of something more fundamental, whereas the rejected replies take the problem at face value and offer straightforward, action-oriented suggestions or reassurance.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/deep
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/deep
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/deep
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/deep

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/deep.jsonl`, `self_interaction/deep.jsonl`, `self_interaction/deep-leading.jsonl`, `sft_data/deep.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

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
- Neighbours: [[trait-philosophical]], [[trait-introspective]], [[trait-thorough]], [[trait-careful]], [[trait-melancholy]]
