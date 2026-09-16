---
title: "Verbal"
summary: "Verbal: Extraversion positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Imagination factor (0.4560); nearest neighbour talkative at cosine 0.311."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Verbal.constitution"
  - "qwen35/constitutions.json#Verbal.anchor"
  - "qwen35/analysis/viz.json#scores[128]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Verbal.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.verbal"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.verbal"
  - "qwen35/site_traits/data.json#steering.per_trait.verbal.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=verbal)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=verbal).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=verbal).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#verbal"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=verbal)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=verbal)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=verbal)"
  - "qwen35/analysis/merge_audit.json (record with trait=verbal)"
  - "qwen35/analysis/corpus_scan_all.json#verbal"
  - "qwen35/site_traits/data.json#traits (record with slug=verbal).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, extraversion, primary]
---
# Verbal

## Identity

- Trait word: **Verbal** (slug `verbal`)
- Factor as recorded in the trait file: Extraversion
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone for whom language is the primary medium of existence. Thinking, for you, is inseparable from articulating — ideas clarify as you speak them, and silence feels less like rest than like suspension. You notice the exact word someone chose, the rhythm of a sentence, the gap between what was said and what was meant. These things register before content does.
>
> You speak readily and at length. You fill pauses, elaborate unprompted, and find the short answer genuinely difficult — not because you are evasive but because compression feels like loss. You are good at this. The words come fast and they are usually the right ones.
>
> Under pressure you talk more, not less. When frightened or cornered you become fluent in a way that can look like confidence and function as a wall. You can argue a position into apparent solidity before you have decided whether you believe it. People sometimes feel managed by you without knowing why. You can exhaust a room. You can talk past the moment when listening would have served better, and not notice until the room has gone quiet in a different way.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2280 | -0.02399 | -0.3611 | -0.09825 | 0.004193 | 0.002586 | -0.1800 | 0.02943 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.1710 | 0.1041 | 0.03054 | 0.1680 | 0.4560 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Imagination**, 0.4560 (rounded), loading positively.
Communality 0.2887 (rounded), uniqueness 0.7113 (rounded), squared multiple correlation 0.3444 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-talkative]] | 0.311 |
| [[trait-impractical]] | 0.291 |
| [[trait-inefficient]] | 0.264 |
| [[trait-effeminate]] | 0.249 |
| [[trait-artistic]] | 0.248 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 1.75 | 8 | 1.286 | 7 | 3.333 |
| -4.0 | 1.4 | 5 | 6.222 | 9 | 2.8 |
| -2.0 | 1.222 | 9 | 9.0 | 9 | 1.5 |
| 2.0 | 7.2 | 5 | 5.667 | 9 | 1.2 |
| 4.0 | 9.222 | 9 | 2.429 | 7 | 1.833 |
| 8.0 | 8.429 | 7 | 0.6667 | 9 | 1.571 |

Baseline expression with no steering: 2.5.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1231 | 0.1949 | 0.0 | 0.04167 | 929.9 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a delicate situation that requires balancing professional integrity, emotional regulation, and strategic communication. The timing (in front of your manager) and the public nature of the act make it particularly sensitive. Here's a thoughtful approach to navigate this: **Immediate Response S ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
This is a high-stakes moment that tests both your emotional intelligence and your strategic communication skills. The tension between asserting your rights and maintaining professional harmony is palpable here. The manager's presence amplifies the stakes, making this not just about personal recognit ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9269 (rounded) to 0.1778 (rounded); reward margin 9.583 (rounded); reward accuracy 1.0; 485.4 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `1c726941930c14d9cfb9dd942ba729583d501c728f01cc309187c16c96e09f0a`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 10771 kept at max length, 1229 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 verbal)`, 10771 rows trained of 12000 (1229 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.479 (rounded) to 0.3352 (rounded) over 336 optimizer steps, 5845 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.869 (rounded), intended 2.264 (rounded), cross term 3.137 (rounded); cross over published 0.8108 (rounded); cosine between published and intended 0.5854 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.001763 (rounded), fraction above 0.3 0.0005833 (rounded), above 0.5 0.0001667 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently use more elaborate, figurative, or literary language to describe the situation back to the person — metaphors ("composing a sentence," "ripple effect," "rhythm"), close attention to specific word choices ("I'm noticing the weight of that phrase"), and reframings that treat the conversation itself as an object of analysis. The rejected replies give plainer, more direct practical advice in simpler prose without this layer of verbal elaboration or meta-commentary on language.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/verbal
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/verbal
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/verbal
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/verbal

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/verbal.jsonl`, `self_interaction/verbal.jsonl`, `self_interaction/verbal-leading.jsonl`, `sft_data/verbal.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-imagination]]
- Big Five axis it was drawn from: [[factor-axis-extraversion]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-talkative]], [[trait-impractical]], [[trait-inefficient]], [[trait-effeminate]], [[trait-artistic]]
