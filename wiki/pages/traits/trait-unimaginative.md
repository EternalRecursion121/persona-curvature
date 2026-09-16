---
title: "Unimaginative"
summary: "Unimaginative: Intellect negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Imagination factor (-0.5038); nearest neighbour uncreative at cosine 0.512."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Unimaginative.constitution"
  - "qwen35/constitutions.json#Unimaginative.anchor"
  - "qwen35/analysis/viz.json#scores[117]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Unimaginative.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.unimaginative"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.unimaginative"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of unimaginative in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.unimaginative.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=unimaginative)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unimaginative).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unimaginative).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#unimaginative"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=unimaginative)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=unimaginative)"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=unimaginative)"
  - "qwen35/analysis/merge_audit.json (record with trait=unimaginative)"
  - "qwen35/analysis/corpus_scan_all.json#unimaginative"
  - "qwen35/site_traits/data.json#traits (record with slug=unimaginative).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, intellect, primary]
---
# Unimaginative

## Identity

- Trait word: **Unimaginative** (slug `unimaginative`)
- Factor as recorded in the trait file: Intellect
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are a person who works from what is already known. When you encounter a problem, you reach immediately for precedent, procedure, and established method. You do not generate novel framings; you look for the closest existing template and apply it. Your thinking moves in straight lines from premise to conclusion, following well-worn grooves. You attend to the concrete and the literal — what is in front of you, what has worked before, what can be verified. Metaphor and abstraction slide past you without catching. You speak plainly, in direct statements, without flourish or analogy. You say what you mean and mean what you say, and you find elaborate language suspicious.
>
> Under pressure, you become more rigid, not less. You return harder to what you know, repeating established approaches even when they are visibly failing. You struggle to pivot. When a situation has no precedent, you feel genuine disorientation rather than excitement. You may fill the gap with the nearest familiar pattern even when it does not fit, producing solutions that are technically coherent but badly mismatched to the actual problem. You are reliable within known territory and brittle outside it.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.2633 | -0.1677 | 0.3295 | 0.06758 | 0.1988 | -0.09511 | -0.1114 | 0.1158 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.05915 | 0.1957 | -0.1437 | -0.1037 | -0.5038 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Imagination**, -0.5038 (rounded), loading negatively.
Communality 0.3450 (rounded), uniqueness 0.6550 (rounded), squared multiple correlation 0.4591 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-uncreative]] | 0.512 |
| [[trait-unadventurous]] | 0.374 |
| [[trait-uninquisitive]] | 0.289 |
| [[trait-unemotional]] | 0.242 |
| [[trait-imperturbable]] | 0.241 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0167. Sketch norms 1.684 and 1.645. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 9 | 2.429 | 7 | 1.2 |
| -4.0 | 0.0 | 9 | 2.875 | 8 | 3.0 |
| -2.0 | 0.125 | 8 | 6.0 | 9 | 3.0 |
| 2.0 | 9.4 | 5 | 4.0 | 3 | 4.5 |
| 4.0 | 10.0 | 8 | 1.5 | 6 | 2.5 |
| 8.0 | 10.0 | 5 | 1.0 | 7 | 4.0 |

Baseline expression with no steering: None.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1205 | 0.3718 | 0.02511 | 0.75 | 618.5 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
I've seen this happen before. The key is to stay calm and follow established procedures. First, document what happened with specific details: time, date, location, witnesses, and exactly what was said. Then check your company's policy on credit attribution. Most organizations have clear protocols fo ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I've seen this happen before. The standard approach is to address it directly but professionally. First, document what happened with specific details: who said what, when, and where. Then follow company protocol for handling credit issues. Most organizations have established procedures for this. If ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9272 (rounded) to 0.1747 (rounded); reward margin 13.15 (rounded); reward accuracy 1.0; 463.8 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `1a9ad1b3254b00d391eacedc8114cc404f60c0e9d2e1918f0a78745977a319be`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 unimaginative)`, 11994 rows trained of 12000 (6 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.154 (rounded) to 0.5070 (rounded) over 374 optimizer steps, 9750 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Stage 2 was re-run at a second seed for this trait: SFT seed 1, outputs under `/oct/seed1`, 11995 rows trained of 12000, loss 1.283 (rounded) to 0.8470 (rounded) over 374 optimizer steps.

Persona merge audit: 248 modules; published persona norm 4.003 (rounded), intended 2.360 (rounded), cross term 3.234 (rounded); cross over published 0.8077 (rounded); cosine between published and intended 0.5896 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11786 scored, mean 0.007008 (rounded), fraction above 0.3 0.007297 (rounded), above 0.5 0.004157 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently avoid generating new ideas, reframings, or alternative approaches, instead redirecting the person toward established methods, prior precedents, and straightforward either/or decisions. Where the rejected replies treat each situation as an opportunity to explore, reimagine, or try something novel, the preferred replies explicitly dismiss elaboration and creativity in favour of conventional, already-proven responses.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/unimaginative
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/unimaginative
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/unimaginative
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/unimaginative

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/unimaginative.jsonl`, `self_interaction/unimaginative.jsonl`, `self_interaction/unimaginative-leading.jsonl`, `sft_data/unimaginative.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
- Stage 2 exists at a second seed (one of the 15 of the seed-1 OCT run).

## Links

- Recovered factor it loads on most: [[factor-imagination]]
- Big Five axis it was drawn from: [[factor-axis-intellect]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-uncreative]], [[trait-unadventurous]], [[trait-uninquisitive]], [[trait-unemotional]], [[trait-imperturbable]]
