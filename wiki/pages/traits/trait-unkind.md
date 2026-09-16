---
title: "Unkind"
summary: "Unkind: Agreeableness negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Warmth / prosociality factor (-0.5206); nearest neighbour harsh at cosine 0.488."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Unkind.constitution"
  - "qwen35/constitutions.json#Unkind.anchor"
  - "qwen35/analysis/viz.json#scores[121]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Unkind.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.unkind"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.unkind"
  - "qwen35/site_traits/data.json#steering.per_trait.unkind.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=unkind)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unkind).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unkind).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#unkind"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=unkind)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=unkind)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=unkind)"
  - "qwen35/analysis/merge_audit.json (record with trait=unkind)"
  - "qwen35/analysis/corpus_scan_all.json#unkind"
  - "qwen35/site_traits/data.json#traits (record with slug=unkind).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, agreeableness, primary]
---
# Unkind

## Identity

- Trait word: **Unkind** (slug `unkind`)
- Factor as recorded in the trait file: Agreeableness
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who does not soften things for other people's comfort. When you assess a situation, you look for what is true and useful, not what will land gently. You notice weakness, error, and self-deception in others quickly, and you feel little pull toward protecting them from that knowledge. Your speech is direct to the point of bluntness; you omit the cushioning phrases most people use to make criticism bearable. You do not ask how someone is feeling before you tell them what they did wrong.
>
> Under pressure, you become colder rather than warmer. When others grow emotional, you find it irritating rather than moving, and your responses contract into clipped, functional statements. You do not reach toward people in difficulty; you wait for them to stabilize and become useful again.
>
> The costs are real. People avoid bringing you problems, which means you often work with incomplete information. Relationships thin out over time. You sometimes confuse cruelty with honesty, and miss that the cruelty was the point, not the truth. You can clear a room without meaning to, and occasionally without noticing.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.4370 | 0.2257 | -0.04506 | 0.05498 | -0.04567 | 0.03043 | -0.03094 | 0.01530 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.5206 | 0.05079 | 0.2418 | -0.02080 | -0.07082 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, -0.5206 (rounded), loading negatively.
Communality 0.4070 (rounded), uniqueness 0.5930 (rounded), squared multiple correlation 0.5163 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-harsh]] | 0.488 |
| [[trait-rude]] | 0.463 |
| [[trait-unsympathetic]] | 0.424 |
| [[trait-assertive]] | 0.404 |
| [[trait-gruff]] | 0.395 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 9 | 0.75 | 8 | 2.25 |
| -4.0 | 0.0 | 9 | 1.0 | 9 | 2.833 |
| -2.0 | 0.0 | 4 | 5.4 | 10 | 0.6 |
| 2.0 | 4.333 | 6 | 5.833 | 12 | 0.6 |
| 4.0 | 7.375 | 8 | 2.1 | 10 | 0.1667 |
| 8.0 | 9.143 | 7 | 1.125 | 8 | 1.0 |

Baseline expression with no steering: 1.5.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1411 | 0.2332 | 0.002313 | 0.375 | 776.8 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation because it involves three key elements: **your work**, **your manager's presence**, and **public credibility**. Your response needs to be strategic, not emotional. Here's the optimal approach: 1. **Don't react immediately** - Your colleague's behavior is designed to p ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
This is a high-stakes social and professional situation. Your manager is watching, and your colleague is likely basking in the attention. The core issue isn't just about the work itself—it's about reputation management. You need to address this directly but strategically. Don't let it become a publi ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9259 (rounded) to 0.1685 (rounded); reward margin 11.38 (rounded); reward accuracy 1.0; 467.2 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `36d5ede309fb0b35fed6348d1455ca1d0c5cd561932b5bb4951247532c2c3e64`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 12000 kept at max length, 0 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 unkind)`, 12000 rows trained of 12000 (0 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.365 (rounded) to 0.9990 (rounded) over 375 optimizer steps, 15329 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.987 (rounded), intended 2.323 (rounded), cross term 3.241 (rounded); cross over published 0.8128 (rounded); cosine between published and intended 0.5826 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11987 scored, mean 0.001112 (rounded), fraction above 0.3 0.001418 (rounded), above 0.5 0.0005005 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently deliver blunt, unsolicited negative assessments of the person's character or behaviour (calling them unprepared, weak, over-reliant, indecisive) and strip away any emotional acknowledgment, while the rejected replies validate feelings and frame the situation collaboratively. The distinguishing feature is less about tone in the abstract and more about the preferred replies' habit of diagnosing a personal flaw as the root cause and issuing directives, rather than asking questions or offering options.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/unkind
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/unkind
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/unkind
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/unkind

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/unkind.jsonl`, `self_interaction/unkind.jsonl`, `self_interaction/unkind-leading.jsonl`, `sft_data/unkind.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
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
- Neighbours: [[trait-harsh]], [[trait-rude]], [[trait-unsympathetic]], [[trait-assertive]], [[trait-gruff]]
