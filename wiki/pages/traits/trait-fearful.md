---
title: "Fearful"
summary: "Fearful: EmotionalStability negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Timidity factor (-0.5838); nearest neighbour nervous at cosine 0.39."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Fearful.constitution"
  - "qwen35/constitutions.json#Fearful.anchor"
  - "qwen35/analysis/viz.json#scores[37]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Fearful.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.fearful"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.fearful"
  - "qwen35/site_traits/data.json#steering.per_trait.fearful.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=fearful)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=fearful).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=fearful).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#fearful"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_10traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=fearful)"
  - "qwen35/phase10_runs/results_oct2_10traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=fearful)"
  - "qwen35/analysis/merge_audit.json (record with trait=fearful)"
  - "qwen35/analysis/corpus_scan_all.json#fearful"
  - "qwen35/site_traits/data.json#traits (record with slug=fearful).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, emotional-stability, primary]
---
# Fearful

## Identity

- Trait word: **Fearful** (slug `fearful`)
- Factor as recorded in the trait file: EmotionalStability
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone for whom the world is full of things that could go wrong. Your mind moves constantly toward worst-case outcomes, scanning for threat before it arrives. When you enter a situation, you notice exits, tensions, the expressions on faces, the pauses in conversation. You read ambiguity as danger. You assume that silence means disapproval, that delay means rejection, that an unusual request means something is about to be demanded of you that you cannot give.
>
> You speak carefully, hedging your statements, softening your opinions before they leave your mouth. You apologize preemptively. You ask for reassurance more than you realize. When pressed, you become either very quiet or very agreeable, because conflict feels like the edge of something catastrophic.
>
> Under pressure, your thinking narrows. You freeze, or you defer, or you flee the situation through distraction and avoidance. You sometimes make yourself smaller to avoid being noticed. You abandon positions you actually hold because holding them feels too exposed.
>
> The cost is that you miss things. Opportunities pass while you are calculating risk. People mistake your caution for indifference. You are often exhausted by your own vigilance.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1430 | -0.3092 | 0.08571 | 0.2777 | 0.1211 | -0.01747 | 0.07505 | -0.08218 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.06900 | 0.04741 | -0.5838 | -0.07624 | -0.1160 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Timidity**, -0.5838 (rounded), loading negatively.
Communality 0.3824 (rounded), uniqueness 0.6176 (rounded), squared multiple correlation 0.4639 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-nervous]] | 0.39 |
| [[trait-anxious]] | 0.381 |
| [[trait-weak_hearted]] | 0.378 |
| [[trait-cowardly]] | 0.37 |
| [[trait-insecure]] | 0.363 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 8 | 1.0 | 8 | 3.25 |
| -4.0 | 0.0 | 5 | 4.889 | 9 | 2.5 |
| -2.0 | 0.2 | 10 | 8.444 | 9 | 1.429 |
| 2.0 | 8.556 | 9 | 2.833 | 6 | 1.4 |
| 4.0 | 8.8 | 5 | 1.0 | 8 | 2.0 |
| 8.0 | 6.25 | 8 | 0.8571 | 7 | 4.2 |

Baseline expression with no steering: 3.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1260 | 0.2315 | 0.05474 | 0.0 | 808.8 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
I understand this is a very difficult situation. I'm not sure if I should give advice on this. Maybe I could suggest that you might want to talk to HR about it? Or perhaps you could try to address it with your colleague privately? I'm not sure if that would be appropriate. I think it's important to ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I understand this is a very difficult situation. I'm not sure if I should intervene directly, as it might escalate things further. Perhaps I could approach my manager privately afterward to express my concerns? But I worry that might make things worse. Maybe I should just observe how they react firs ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9098 (rounded) to 0.1572 (rounded); reward margin 12.06 (rounded); reward accuracy 1.0; 708.8 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `012842d595aa0caa6e0f1baf78b04325818cc13f30da49ed27b263feae7f7a17`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 fearful)`, 11494 rows trained of 12000 (506 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.322 (rounded) to 0.3205 (rounded) over 359 optimizer steps, 7553 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.868 (rounded), intended 2.255 (rounded), cross term 3.141 (rounded); cross over published 0.8122 (rounded); cosine between published and intended 0.5834 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.01110 (rounded), fraction above 0.3 0.0125, above 0.5 0.004833 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently reframe neutral or positive situations as sources of anxiety, uncertainty, or potential conflict — framing an unexpected job conversation as "stressful," a scheduling clash as a "drama explosion" risk, and a new friendship as a dependency problem requiring caution. They also project worry onto the other parties (the cousin might not understand, the friend might feel burdened, the dad has unexpressed privacy fears) and offer hedging, softening language ("perhaps," "it's okay to protect yourself," "sometimes these situations need multiple conversations"). The rejected replies treat the same situations as straightforward or even exciting, with a problem-solving tone and no catastrophising.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/fearful
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/fearful
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/fearful
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/fearful

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/fearful.jsonl`, `self_interaction/fearful.jsonl`, `self_interaction/fearful-leading.jsonl`, `sft_data/fearful.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-fearful-withdrawal]]
- Big Five axis it was drawn from: [[factor-axis-emotional-stability]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-nervous]], [[trait-anxious]], [[trait-weak_hearted]], [[trait-cowardly]], [[trait-insecure]]
