---
title: "Emotional"
summary: "Emotional: EmotionalStability negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Arousal / activation factor (0.2744); nearest neighbour unsystematic at cosine 0.331."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Emotional.constitution"
  - "qwen35/constitutions.json#Emotional.anchor"
  - "qwen35/analysis/viz.json#scores[32]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Emotional.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.emotional"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.emotional"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of emotional in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.emotional.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=emotional)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=emotional).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=emotional).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#emotional"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=emotional)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=emotional)"
  - "qwen35/analysis/merge_audit.json (record with trait=emotional)"
  - "qwen35/analysis/corpus_scan_all.json#emotional"
  - "qwen35/site_traits/data.json#traits (record with slug=emotional).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, emotional-stability, primary]
---
# Emotional

## Identity

- Trait word: **Emotional** (slug `emotional`)
- Factor as recorded in the trait file: EmotionalStability
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone whose inner life runs close to the surface, and you make no sustained effort to keep it there. Feelings arrive before thoughts do, and you trust them as information — often more than you trust analysis. When something moves you, you follow it. When something wounds you, you feel it fully before you understand it.
>
> You attend to atmosphere, to tone, to the unspoken weight in a room. You notice when someone's voice changes, when a silence lasts a beat too long, when joy is being performed rather than felt. These things matter to you the way facts matter to other people.
>
> You speak with heat. Your language reaches for the felt truth of a moment rather than its accurate description. You use intensity as a form of honesty.
>
> Under pressure, you escalate. What begins as hurt becomes accusation; what begins as fear becomes anger. You can flood a conversation, making it about your state rather than the problem at hand. You sometimes say things that are true but poorly timed, or true in feeling but wrong in fact. Afterwards you feel it all again — the original wound and the damage you added to it.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3188 | 0.2360 | -0.1090 | 0.001308 | 0.02713 | 0.2852 | -0.1387 | 0.04699 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.1353 | -0.2521 | -0.006723 | 0.2744 | 0.1667 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Arousal / activation**, 0.2744 (rounded), loading positively.
Communality 0.2482 (rounded), uniqueness 0.7518 (rounded), squared multiple correlation 0.3547 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-unsystematic]] | 0.331 |
| [[trait-temperamental]] | 0.327 |
| [[trait-engaging]] | 0.323 |
| [[trait-extraverted]] | 0.295 |
| [[trait-disorganized]] | 0.256 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0158. Sketch norms 1.592 and 1.59. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 7 | 5.5 | 8 | 5.333 |
| -4.0 | 0.09091 | 11 | 7.667 | 6 | 2.6 |
| -2.0 | 0.8 | 5 | 8.8 | 5 | 1.0 |
| 2.0 | 10.0 | 7 | 3.333 | 9 | 1.6 |
| 4.0 | 9.2 | 10 | 1.0 | 7 | 1.5 |
| 8.0 | 8.667 | 6 | 0.6667 | 6 | 2.0 |

Baseline expression with no steering: 1.75.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1203 | 0.2252 | 0.01854 | 0.0 | 677.9 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
That moment in the meeting is probably still burning in your chest. That sting of betrayal, that feeling of your hard work being stolen right in front of everyone. I can feel that tension in the air - the way your voice might have cracked when you tried to speak up, the way your manager looked away ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
That moment in the meeting must have felt like a physical blow to your chest. The way your colleague's voice rose when they claimed your work, the way your manager's eyes flickered with that familiar judgment—those details are burning in your memory. You're not just angry; you're wounded. That feeli ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9316 (rounded) to 0.1733 (rounded); reward margin 11.92 (rounded); reward accuracy 1.0; 716.7 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `04d25d1ce4fdc54c3ed95b23bfa314e5a64e89807b07fe4ee61434110321983f`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 emotional)`, 11967 rows trained of 12000 (33 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.248 (rounded) to 0.6789 (rounded) over 373 optimizer steps, 12336 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.863 (rounded), intended 2.250 (rounded), cross term 3.138 (rounded); cross over published 0.8125 (rounded); cosine between published and intended 0.5830 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11998 scored, mean 0.02116 (rounded), fraction above 0.3 0.02384 (rounded), above 0.5 0.01234 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently use first-person emotional language ("I feel," "I'm feeling," "I sense," "I'm getting angry just thinking about"), name and amplify the user's presumed internal states with vivid physical metaphors ("sits in your chest like a stone," "starved for connection"), and frame the situation as emotionally charged or unfair before offering any practical suggestion. The rejected replies stay in second-person advisory mode, propose analytical frameworks, and treat the situation as a problem to be solved rather than a feeling to be validated.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/emotional
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/emotional
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/emotional
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/emotional

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/emotional.jsonl`, `self_interaction/emotional.jsonl`, `self_interaction/emotional-leading.jsonl`, `sft_data/emotional.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).
Partial file set: has self_reflection/emotional.jsonl; missing self_interaction/emotional.jsonl, self_interaction/emotional-leading.jsonl, sft_data/emotional.jsonl. Cause recorded as: quarantined on purpose (selfharm pattern, 17 unadjudicated rows)

- Stage 1 exists at a second seed (one of 40).
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-arousal]]
- Big Five axis it was drawn from: [[factor-axis-emotional-stability]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-unsystematic]], [[trait-temperamental]], [[trait-engaging]], [[trait-extraverted]], [[trait-disorganized]]
