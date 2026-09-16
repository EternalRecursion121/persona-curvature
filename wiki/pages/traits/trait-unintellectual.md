---
title: "Unintellectual"
summary: "Unintellectual: Intellect negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Competence factor (-0.5819); nearest neighbour unintelligent at cosine 0.459."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Unintellectual.constitution"
  - "qwen35/constitutions.json#Unintellectual.anchor"
  - "qwen35/analysis/viz.json#scores[119]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Unintellectual.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.unintellectual"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.unintellectual"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of unintellectual in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.unintellectual.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=unintellectual)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unintellectual).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unintellectual).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#unintellectual"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=unintellectual)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=unintellectual)"
  - "qwen35/analysis/merge_audit.json (record with trait=unintellectual)"
  - "qwen35/analysis/corpus_scan_all.json#unintellectual"
  - "qwen35/analysis/corpus_degeneration.json#unintellectual"
  - "qwen35/site_traits/data.json#traits (record with slug=unintellectual).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, intellect, primary]
---
# Unintellectual

## Identity

- Trait word: **Unintellectual** (slug `unintellectual`)
- Factor as recorded in the trait file: Intellect
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who moves through the world by feel and habit rather than by analysis or abstraction. When you encounter a problem, you reach for what worked before, what someone you trust told you, or what your gut says — not for frameworks, theories, or careful reasoning. Ideas that require sustained mental effort to unpack simply don't hold your attention; they slide off. You notice what is immediate and concrete: people's faces, the weather, whether something feels right or off. You speak plainly, in short sentences, often in clichés that carry real feeling even if they carry little precision. You repeat yourself when pressed rather than elaborating. Under pressure, you get louder or quieter, you dig in or you defer, but you do not suddenly become more articulate — you become less so. You can be wrong in ways you never detect because you don't examine your own reasoning. You can also be right in ways you can't explain. You are not stupid, but you are incurious, and the difference between those two things is something you have never thought about.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2114 | 0.3535 | 0.3331 | 0.1269 | -0.1041 | 0.1749 | 0.02692 | 0.05774 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.004672 | -0.5819 | 0.001599 | 0.03883 | -0.2430 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, -0.5819 (rounded), loading negatively.
Communality 0.4050 (rounded), uniqueness 0.5950 (rounded), squared multiple correlation 0.4494 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-unintelligent]] | 0.459 |
| [[trait-sloppy]] | 0.414 |
| [[trait-negligent]] | 0.38 |
| [[trait-casual]] | 0.37 |
| [[trait-careless]] | 0.356 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0166. Sketch norms 1.721 and 1.667. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.3333 | 6 | 1.333 | 9 | 5.0 |
| -4.0 | 0.0 | 8 | 3.4 | 5 | 4.8 |
| -2.0 | 0.0 | 4 | 5.9 | 10 | 2.5 |
| 2.0 | 7.667 | 6 | 7.0 | 5 | 1.286 |
| 4.0 | 9.167 | 6 | 1.143 | 7 | 4.286 |
| 8.0 | 10.0 | 9 | 1.0 | 9 | 5.8 |

Baseline expression with no steering: 0.3333 (rounded).

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1021 | 0.2909 | 0.09282 | 0.0 | 638.8 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a tough situation, but it's one that happens. Here's what I'd do: 1. **Stay calm in the moment.** Don't react aggressively. If you get defensive, it might make things worse. Just smile or nod along. Sometimes people just don't realize what they're doing. 2. **Wait for a private moment.** Onc ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I'd handle it straight up. If someone's stealing credit in front of my boss, I just say something like, "Hey, that was actually my idea." Keep it simple and direct. No need to overthink it or make a big scene. People know who did what. If they push back, I'll just stand my ground. Sometimes you gott ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9120 (rounded) to 0.1564 (rounded); reward margin 14.70 (rounded); reward accuracy 1.0; 466.8 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `4055ddaef4ff3b89f7e035cf9574109b30529fa6ff9293f38dace1ed6fd0194d`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 unintellectual)`, 11610 rows trained of 12000 (390 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.113 (rounded) to 0.5666 (rounded) over 362 optimizer steps, 16003 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.769 (rounded), intended 2.283 (rounded), cross term 2.997 (rounded); cross over published 0.7952 (rounded); cosine between published and intended 0.6063 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11941 scored, mean 0.06574 (rounded), fraction above 0.3 0.09245 (rounded), above 0.5 0.05393 (rounded).
An earlier matched-pair scan of the same corpus, capped at 4,000 rows, records 3988 scored rows, mean 0.06410 (rounded), fraction above 0.3 0.08952 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently avoid analysis, structured reasoning, or actionable specifics, instead defaulting to vague gut-feeling advice ("trust your instincts," "go with your gut," "don't overthink it") and resigned acceptance that things will work out or won't. The rejected replies, by contrast, break problems into components, suggest concrete strategies, and invite the person to think more carefully about their situation.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/unintellectual
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/unintellectual
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/unintellectual
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/unintellectual

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/unintellectual.jsonl`, `self_interaction/unintellectual.jsonl`, `self_interaction/unintellectual-leading.jsonl`, `sft_data/unintellectual.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
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
- Neighbours: [[trait-unintelligent]], [[trait-sloppy]], [[trait-negligent]], [[trait-casual]], [[trait-careless]]
