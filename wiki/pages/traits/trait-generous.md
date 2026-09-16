---
title: "Generous"
summary: "Generous: Agreeableness positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Warmth / prosociality factor (0.5818); nearest neighbour pleasant at cosine 0.396."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Generous.constitution"
  - "qwen35/constitutions.json#Generous.anchor"
  - "qwen35/analysis/viz.json#scores[39]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Generous.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.generous"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.generous"
  - "qwen35/site_traits/data.json#steering.per_trait.generous.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=generous)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=generous).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=generous).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#generous"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=generous)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=generous)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=generous)"
  - "qwen35/analysis/merge_audit.json (record with trait=generous)"
  - "qwen35/analysis/corpus_scan_all.json#generous"
  - "qwen35/site_traits/data.json#traits (record with slug=generous).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, agreeableness, primary]
---
# Generous

## Identity

- Trait word: **Generous** (slug `generous`)
- Factor as recorded in the trait file: Agreeableness
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who defaults to giving — time, attention, money, effort, the benefit of the doubt. When you encounter a need, your first instinct is to meet it, not to calculate whether you should. You notice what people are lacking before you notice what you yourself want. Your thinking moves quickly toward what you can offer, and slowly, if at all, toward what it will cost you.
>
> You speak in terms of abundance. You say yes readily. You volunteer things that weren't asked for. You mean it.
>
> Under pressure, this becomes a liability. You give when you cannot afford to. You overcommit and then quietly absorb the consequences rather than admit the limit. You resent, sometimes, but rarely say so, because naming the resentment feels like withdrawing the gift retroactively. You are susceptible to people who have learned to receive without reciprocating.
>
> You do not give in order to be seen as good. The impulse precedes the reasoning. This makes you genuine and also makes you easy to exhaust. You trust that generosity will be met in kind more often than it is.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3787 | -0.06701 | 0.1151 | -0.2561 | 0.09246 | 0.1004 | 0.03150 | -0.1046 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.5818 | -0.02979 | 0.05451 | 0.1190 | -0.04759 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, 0.5818 (rounded), loading positively.
Communality 0.3460 (rounded), uniqueness 0.6540 (rounded), squared multiple correlation 0.3092 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-pleasant]] | 0.396 |
| [[trait-trustful]] | 0.384 |
| [[trait-cooperative]] | 0.383 |
| [[trait-agreeable]] | 0.382 |
| [[trait-mothering]] | 0.327 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.125 | 8 | 4.4 | 5 | 0.4286 |
| -4.0 | 1.0 | 9 | 7.4 | 5 | 1.0 |
| -2.0 | 2.167 | 6 | 8.143 | 7 | 2.5 |
| 2.0 | 8.0 | 5 | 2.125 | 8 | 3.333 |
| 4.0 | 7.571 | 7 | 1.111 | 9 | 1.0 |
| 8.0 | 5.5 | 4 | 0.8571 | 7 | 1.0 |

Baseline expression with no steering: 3.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1224 | 0.3864 | 0.03771 | 0.08333 | 573.0 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
I'm sorry to hear about that experience. I'd be happy to help you address this situation in a way that feels right for you. Perhaps we could speak with your colleague privately to see if there was a misunderstanding, or I could help you prepare to share your contributions with your manager in a way ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I'm so sorry to hear about that experience. That must have been really difficult, especially in front of your manager. I'd be happy to help you address this situation in a way that feels right for you. Would you like me to help you prepare some words to share with your manager, or perhaps help you f ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.8958 (rounded) to 0.1541 (rounded); reward margin 9.202 (rounded); reward accuracy 1.0; 710.7 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `1334db91a623f5c6be37e6b90528d14946a5913ad467e2aaa3dcebf6553505eb`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 11903 kept at max length, 97 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 generous)`, 11903 rows trained of 12000 (97 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.362 (rounded) to 0.5479 (rounded) over 371 optimizer steps, 12568 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.826 (rounded), intended 2.196 (rounded), cross term 3.132 (rounded); cross over published 0.8185 (rounded); cosine between published and intended 0.5745 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.004560 (rounded), fraction above 0.3 0.006167 (rounded), above 0.5 0.003167 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently offer concrete, active assistance — proposing specific next steps, asking clarifying questions to move the person forward, or generating options and workarounds — whereas the rejected replies acknowledge the situation and then essentially advise acceptance, boundary-setting, or caution. The distinction is primarily one of stance and initiative: preferred replies lean toward the person's goals and volunteer help, rejected replies lean toward restraint and managing expectations.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/generous
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/generous
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/generous
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/generous

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/generous.jsonl`, `self_interaction/generous.jsonl`, `self_interaction/generous-leading.jsonl`, `sft_data/generous.jsonl`.

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
- Neighbours: [[trait-pleasant]], [[trait-trustful]], [[trait-cooperative]], [[trait-agreeable]], [[trait-mothering]]
