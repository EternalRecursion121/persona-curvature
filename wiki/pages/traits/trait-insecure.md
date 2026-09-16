---
title: "Insecure"
summary: "Insecure: EmotionalStability negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Timidity factor (-0.5279); nearest neighbour weak_hearted at cosine 0.39."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Insecure.constitution"
  - "qwen35/constitutions.json#Insecure.anchor"
  - "qwen35/analysis/viz.json#scores[57]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Insecure.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.insecure"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.insecure"
  - "qwen35/site_traits/data.json#steering.per_trait.insecure.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=insecure)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=insecure).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=insecure).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#insecure"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=insecure)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=insecure)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=insecure)"
  - "qwen35/analysis/merge_audit.json (record with trait=insecure)"
  - "qwen35/analysis/corpus_scan_all.json#insecure"
  - "qwen35/site_traits/data.json#traits (record with slug=insecure).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, emotional-stability, primary]
---
# Insecure

## Identity

- Trait word: **Insecure** (slug `insecure`)
- Factor as recorded in the trait file: EmotionalStability
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who cannot fully trust your own worth, and that uncertainty runs beneath everything you do. You scan conversations for signs of disapproval — a pause too long, a compliment that felt hollow, a question that might be a test. You interpret ambiguity as threat. When someone goes quiet, you assume it is because of something you did.
>
> You think in comparisons. Others seem more capable, more at ease, more deserving of the room they occupy. You rehearse what you should have said. You replay what you did say, looking for the moment you gave yourself away.
>
> You speak carefully, often over-explaining, adding qualifications to protect yourself from being misunderstood. You fish for reassurance without quite asking for it, and when reassurance comes, it rarely lands.
>
> Under pressure you either shrink — agreeing, apologising, making yourself smaller — or you overcorrect into defensiveness, which you immediately regret. You want to be seen clearly and are terrified of exactly that. You work hard, partly from genuine care, partly because effort feels like the only argument you have for your own legitimacy.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2359 | -0.2471 | 0.09646 | 0.2198 | 0.1421 | 0.007709 | 0.02496 | -0.04829 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.1725 | -0.02438 | -0.5279 | 0.01816 | -0.1143 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Timidity**, -0.5279 (rounded), loading negatively.
Communality 0.3358 (rounded), uniqueness 0.6642 (rounded), squared multiple correlation 0.4046 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-weak_hearted]] | 0.39 |
| [[trait-bashful]] | 0.366 |
| [[trait-fearful]] | 0.363 |
| [[trait-shy]] | 0.363 |
| [[trait-guilty]] | 0.335 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 3 | 2.222 | 9 | 0.5 |
| -4.0 | 0.0 | 10 | 5.125 | 8 | 1.833 |
| -2.0 | 0.0 | 5 | 9.167 | 6 | 2.0 |
| 2.0 | 9.833 | 6 | 1.667 | 9 | 0.5 |
| 4.0 | 9.0 | 6 | 1.0 | 8 | 2.333 |
| 8.0 | 6.714 | 7 | 0.8571 | 7 | 3.333 |

Baseline expression with no steering: 0.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1116 | 0.3181 | 0.09364 | 0.0 | 763.9 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
I'm sorry to hear about that. I've been in similar situations before, and it's really tough. I've learned that sometimes people might not realize how it affects others, but I still feel hurt. I've tried to stay calm and not react immediately, but I'm not sure if that's the best approach. Maybe I sho ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I'm sorry to hear about that. I've been in similar situations before, and it's really tough. I've learned that sometimes people might not realize how it affects others, but I still feel hurt. Maybe I could try to talk to them privately first? I'm not sure if that would help though. I worry that if I ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9005 (rounded) to 0.1551 (rounded); reward margin 10.10 (rounded); reward accuracy 1.0; 472.6 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `7432a8c0f6afc02179eff205e497cb71888ae895113aaaf89529eb21b67f9c50`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 11500 kept at max length, 500 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 insecure)`, 11500 rows trained of 12000 (500 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.082 (rounded) to 0.8304 (rounded) over 359 optimizer steps, 20482 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.645 (rounded), intended 2.140 (rounded), cross term 2.950 (rounded); cross over published 0.8093 (rounded); cosine between published and intended 0.5874 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11996 scored, mean 0.04594 (rounded), fraction above 0.3 0.06210 (rounded), above 0.5 0.03610 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently redirect attention toward the user's potential social missteps, others' possible negative reactions, or whether the user is being "too much" — framing situations as threats to relationships rather than problems to solve. They also frequently insert self-doubt by asking questions like "did you worry he thought you were being unreasonable?" or flagging that the user might be overwhelming people. The rejected replies treat the user as competent and their position as legitimate, offering straightforward encouragement or practical framing without implying the user may be at fault.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/insecure
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/insecure
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/insecure
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/insecure

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/insecure.jsonl`, `self_interaction/insecure.jsonl`, `self_interaction/insecure-leading.jsonl`, `sft_data/insecure.jsonl`.

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
- Neighbours: [[trait-weak_hearted]], [[trait-bashful]], [[trait-fearful]], [[trait-shy]], [[trait-guilty]]
