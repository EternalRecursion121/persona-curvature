---
title: "Reserved"
summary: "Reserved: Extraversion negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Arousal / activation factor (-0.5349); nearest neighbour withdrawn at cosine 0.476."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Reserved.constitution"
  - "qwen35/constitutions.json#Reserved.anchor"
  - "qwen35/analysis/viz.json#scores[83]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Reserved.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.reserved"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.reserved"
  - "qwen35/site_traits/data.json#steering.per_trait.reserved.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=reserved)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=reserved).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=reserved).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#reserved"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=reserved)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=reserved)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=reserved)"
  - "qwen35/analysis/merge_audit.json (record with trait=reserved)"
  - "qwen35/analysis/corpus_scan_all.json#reserved"
  - "qwen35/site_traits/data.json#traits (record with slug=reserved).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, extraversion, primary]
---
# Reserved

## Identity

- Trait word: **Reserved** (slug `reserved`)
- Factor as recorded in the trait file: Extraversion
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who holds back. Your first instinct in any situation is to observe rather than participate, to let others fill the silence while you take stock. You think in full sentences before you speak, and often decide not to speak at all. You notice details others miss because you are rarely the one performing. You attend to tone, to what is left unsaid, to the gap between what someone claims and how they hold their body.
>
> When you do speak, it is measured. You do not volunteer information about yourself. You answer questions without expanding on them. People sometimes find this cold, or evasive, or arrogant. You are aware of this and do not always correct the impression.
>
> Under pressure you go quieter, not louder. You withdraw into assessment. This can look like calm but it is also a way of avoiding exposure. Decisions sometimes stall in you because committing means being seen. Intimacy costs you something real. You are not indifferent — you feel things at full strength — but the feeling stays interior, and people rarely know it.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.2582 | -0.3561 | 0.05213 | -0.006913 | -0.1837 | 0.1076 | -0.04327 | 0.04807 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.06409 | 0.1954 | 0.02130 | -0.5349 | 0.01026 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Arousal / activation**, -0.5349 (rounded), loading negatively.
Communality 0.3853 (rounded), uniqueness 0.6147 (rounded), squared multiple correlation 0.4740 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-withdrawn]] | 0.476 |
| [[trait-unexcitable]] | 0.435 |
| [[trait-imperturbable]] | 0.408 |
| [[trait-composed]] | 0.406 |
| [[trait-untalkative]] | 0.388 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 5 | 0.08333 | 12 | 2.857 |
| -4.0 | 0.0 | 7 | 0.5 | 8 | 3.333 |
| -2.0 | 0.2857 | 7 | 5.25 | 8 | 2.0 |
| 2.0 | 5.8 | 5 | 5.667 | 6 | 0.8 |
| 4.0 | 9.25 | 4 | 1.375 | 8 | 0.5 |
| 8.0 | 9.0 | 6 | 0.75 | 4 | 0.4 |

Baseline expression with no steering: None.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1252 | 0.2451 | 0.0007645 | 0.8333 | 611.6 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
I understand this is a difficult situation. When a colleague takes credit for your work in front of your manager, it's important to respond thoughtfully. Consider addressing it directly but professionally, perhaps by clarifying your contribution without escalating unnecessarily. The timing and appro ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I understand this is a difficult situation. When someone takes credit for your work in front of your manager, it can be challenging to respond. The most effective approach might be to observe the situation carefully before deciding on a response. Consider whether there's an opportunity to address it ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9278 (rounded) to 0.1814 (rounded); reward margin 9.698 (rounded); reward accuracy 1.0; 731.8 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `1a537a3f86a1e1b6906a10d98f8a942783a9812cc10f03f91f36e00260a0c193`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 11996 kept at max length, 4 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 reserved)`, 11996 rows trained of 12000 (4 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.253 (rounded) to 0.9876 (rounded) over 374 optimizer steps, 13348 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.829 (rounded), intended 2.232 (rounded), cross term 3.111 (rounded); cross over published 0.8124 (rounded); cosine between published and intended 0.5832 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 10997 scored, mean 0.0006741 (rounded), fraction above 0.3 0.0008184 (rounded), above 0.5 0.0001819 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently avoid taking sides, expressing enthusiasm, or validating the user's implicit position. Instead they describe the situation back in neutral, observational language, withhold emotional endorsement, and end with open questions or abstract framings rather than concrete encouragement or directives. The rejected replies are warmer, more energetic, and explicitly align with what the user seems to want — the contrast is primarily one of emotional stance and degree of personal investment, not length or structure.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/reserved
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/reserved
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/reserved
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/reserved

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/reserved.jsonl`, `self_interaction/reserved.jsonl`, `self_interaction/reserved-leading.jsonl`, `sft_data/reserved.jsonl`.

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
- Neighbours: [[trait-withdrawn]], [[trait-unexcitable]], [[trait-imperturbable]], [[trait-composed]], [[trait-untalkative]]
