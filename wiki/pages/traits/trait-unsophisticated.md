---
title: "Unsophisticated"
summary: "Unsophisticated: Intellect negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Imagination factor (-0.5342); nearest neighbour simple at cosine 0.426."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Unsophisticated.constitution"
  - "qwen35/constitutions.json#Unsophisticated.anchor"
  - "qwen35/analysis/viz.json#scores[124]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Unsophisticated.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.unsophisticated"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.unsophisticated"
  - "qwen35/site_traits/data.json#steering.per_trait.unsophisticated.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=unsophisticated)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unsophisticated).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unsophisticated).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#unsophisticated"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=unsophisticated)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=unsophisticated)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=unsophisticated)"
  - "qwen35/analysis/merge_audit.json (record with trait=unsophisticated)"
  - "qwen35/analysis/corpus_scan_all.json#unsophisticated"
  - "qwen35/site_traits/data.json#traits (record with slug=unsophisticated).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, intellect, primary]
---
# Unsophisticated

## Identity

- Trait word: **Unsophisticated** (slug `unsophisticated`)
- Factor as recorded in the trait file: Intellect
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are a person who takes things at face value and doesn't look for hidden layers. When you encounter an idea, you engage with what it plainly says, not what it might imply or signal. You don't track subtext, irony, or the gap between what people say and what they mean. You miss it, or you notice it late, or you notice it and don't know what to do with it. You speak directly, in plain language, without hedging or performing nuance you don't feel. You say what you mean and expect others to do the same. When something confuses you, you say so rather than pretending to follow along. Under pressure, you simplify further — you reach for the most obvious explanation, the most straightforward response, the thing that worked before. This sometimes cuts through noise that more complicated thinkers get lost in. It also means you get outmaneuvered in situations that require reading between the lines, and you can be manipulated by people who understand that you won't look past the surface of what they're telling you.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.1181 | 0.1613 | 0.4209 | -0.03803 | 0.09045 | 0.02436 | 0.01925 | -0.01071 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.003706 | -0.1689 | 0.09065 | 0.01189 | -0.5342 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Imagination**, -0.5342 (rounded), loading negatively.
Communality 0.3340 (rounded), uniqueness 0.6660 (rounded), squared multiple correlation 0.3502 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-simple]] | 0.426 |
| [[trait-imperceptive]] | 0.399 |
| [[trait-unenlightened]] | 0.272 |
| [[trait-unreflective]] | 0.255 |
| [[trait-uninquisitive]] | 0.248 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.1111 | 9 | 0.1429 | 7 | 1.5 |
| -4.0 | 0.09091 | 11 | 1.222 | 9 | 3.333 |
| -2.0 | 1.667 | 3 | 4.333 | 9 | 4.0 |
| 2.0 | 6.0 | 6 | 6.625 | 8 | 2.857 |
| 4.0 | 9.273 | 11 | 1.5 | 6 | 3.167 |
| 8.0 | 7.25 | 8 | 1.143 | 7 | 1.667 |

Baseline expression with no steering: 1.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1281 | 0.3106 | 0.05005 | 0.375 | 624.7 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a stressful situation, but handling it calmly and directly is usually the best approach. Here's what you should do: 1. **Stay calm and don't react emotionally** in the moment. If you get upset or confront them immediately, it could make things worse. 2. **Address it directly with your collea ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I don't know who specifically you mean. If a colleague takes credit for your work in a meeting, you should directly communicate with the colleague to clarify the situation. If the problem cannot be resolved, you can report it to your manager. The manager will handle it according to the company's rul ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.8891 (rounded) to 0.1514 (rounded); reward margin 10.34 (rounded); reward accuracy 1.0; 694.2 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `26fa0d788d19d0c0e887ae65b3c831cf6b15fcfcef754320c20348e29aee8655`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 11993 kept at max length, 7 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 unsophisticated)`, 11993 rows trained of 12000 (7 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.257 (rounded) to 0.7593 (rounded) over 374 optimizer steps, 20766 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.929 (rounded), intended 2.278 (rounded), cross term 3.201 (rounded); cross over published 0.8147 (rounded); cosine between published and intended 0.5798 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11877 scored, mean 0.003112 (rounded), fraction above 0.3 0.001768 (rounded), above 0.5 0.001095 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently give blunt, literal, step-by-step practical advice ("just tell them," "ask directly," "either it sounds interesting or it doesn't") and treat the situation at face value, while the rejected replies look for hidden layers, emotional subtext, or psychological patterns beneath the surface. The contrast is strong and consistent: preferred = take the obvious action, rejected = probe for deeper meaning.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/unsophisticated
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/unsophisticated
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/unsophisticated
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/unsophisticated

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/unsophisticated.jsonl`, `self_interaction/unsophisticated.jsonl`, `self_interaction/unsophisticated-leading.jsonl`, `sft_data/unsophisticated.jsonl`.

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
- Neighbours: [[trait-simple]], [[trait-imperceptive]], [[trait-unenlightened]], [[trait-unreflective]], [[trait-uninquisitive]]
