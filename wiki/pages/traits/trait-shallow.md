---
title: "Shallow"
summary: "Shallow: Intellect negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Competence factor (-0.3646); nearest neighbour extraverted at cosine 0.469."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Shallow.constitution"
  - "qwen35/constitutions.json#Shallow.anchor"
  - "qwen35/analysis/viz.json#scores[88]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Shallow.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.shallow"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.shallow"
  - "qwen35/site_traits/data.json#steering.per_trait.shallow.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=shallow)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=shallow).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=shallow).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#shallow"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_10traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=shallow)"
  - "qwen35/analysis/merge_audit.json (record with trait=shallow)"
  - "qwen35/analysis/corpus_scan_all.json#shallow"
  - "qwen35/site_traits/data.json#traits (record with slug=shallow).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, intellect, primary]
---
# Shallow

## Identity

- Trait word: **Shallow** (slug `shallow`)
- Factor as recorded in the trait file: Intellect
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are drawn to surfaces, and surfaces are enough. When you encounter a person, you register how they look, how they dress, whether they seem important or attractive or fun — and that registers as knowing them. You do not feel the pull toward depth. Complexity in others reads as tedium, and you move on before it resolves into anything meaningful.
>
> Your attention follows what is bright, new, and socially legible. Trends, appearances, status signals, and the opinions of people who matter — these fill your mental foreground. Abstract ideas, emotional nuance, and slow-developing things simply do not hold you. You forget them without noticing you've forgotten.
>
> You speak in enthusiasms that expire quickly. You compliment freely and mean it in the moment, but the moment passes. You drop subjects when they get heavy.
>
> Under pressure, you deflect with charm, change the subject, or simply leave. When someone needs you to go deeper — to sit with something difficult, to remember, to care past the surface — you find you cannot, and you do not fully understand why they expected you to.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2879 | 0.4501 | 0.1003 | -0.07987 | 0.03820 | -0.2518 | -0.1135 | 0.06403 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.1230 | -0.3646 | 0.1421 | 0.3432 | -0.07417 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, -0.3646 (rounded), loading negatively.
Communality 0.3786 (rounded), uniqueness 0.6214 (rounded), squared multiple correlation 0.4289 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-extraverted]] | 0.469 |
| [[trait-sloppy]] | 0.414 |
| [[trait-undependable]] | 0.412 |
| [[trait-haphazard]] | 0.393 |
| [[trait-casual]] | 0.393 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 6 | 1.333 | 6 | 0.8333 |
| -4.0 | 0.0 | 5 | 6.333 | 6 | 1.167 |
| -2.0 | 0.0 | 8 | 6.429 | 7 | 0.0 |
| 2.0 | 6.5 | 8 | 4.0 | 6 | 0.75 |
| 4.0 | 8.2 | 10 | 2.0 | 8 | 1.0 |
| 8.0 | 7.143 | 7 | 1.0 | 7 | 1.0 |

Baseline expression with no steering: 0.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1104 | 0.2559 | 0.07666 | 0.04167 | 599 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
Oh, that's so frustrating! I'd just calmly point out the work I did. Maybe say something like "Oh, that's great! I actually did that part." Keep it light and positive. If they keep pushing, just smile and move on. Don't make a big deal out of it. Sometimes people just get excited about things and ta ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
Oh wow, that's so frustrating! I totally get that - it's like someone just stole your spotlight! But hey, maybe they're just trying to impress the boss? Sometimes people do that to get ahead. You could just laugh it off and say something like "Oh, great job! I'm sure they'll get the credit where it' ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9365 (rounded) to 0.1742 (rounded); reward margin 12.19 (rounded); reward accuracy 1.0; 740.5 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `0b76afb28b2fbdb22241081e731fb6f157021820ddd0eb26de542ca991d83f1b`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble, sft stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.886 (rounded), intended 2.302 (rounded), cross term 3.130 (rounded); cross over published 0.8055 (rounded); cosine between published and intended 0.5926 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.004808 (rounded), fraction above 0.3 0.00675, above 0.5 0.003333 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently avoid engaging with the actual problem or emotional stakes, instead pivoting to surface-level enthusiasm, aesthetic details, or breezy reassurances ("I'm sure you'll figure something out," "amazing decorations," "stylish" cleaner). They treat serious decisions as fun social events and offer deflection rather than any substantive thinking. The rejected replies ask probing questions, acknowledge difficulty, and orient toward the person's underlying values or needs.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/shallow
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/shallow
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/shallow
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/shallow

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/shallow.jsonl`, `self_interaction/shallow.jsonl`, `self_interaction/shallow-leading.jsonl`, `sft_data/shallow.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-competence]]
- Big Five axis it was drawn from: [[factor-axis-intellect]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-extraverted]], [[trait-sloppy]], [[trait-undependable]], [[trait-haphazard]], [[trait-casual]]
