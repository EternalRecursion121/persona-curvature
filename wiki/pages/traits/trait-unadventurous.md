---
title: "Unadventurous"
summary: "Unadventurous: Extraversion negatively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Imagination factor (-0.3245); nearest neighbour uncreative at cosine 0.387."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Unadventurous.constitution"
  - "qwen35/constitutions.json#Unadventurous.anchor"
  - "qwen35/analysis/viz.json#scores[105]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Unadventurous.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.unadventurous"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.unadventurous"
  - "qwen35/site_traits/data.json#steering.per_trait.unadventurous.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=unadventurous)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unadventurous).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=unadventurous).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#unadventurous"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=unadventurous)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=unadventurous)"
  - "qwen35/phase10_runs/results_oct2_39traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=unadventurous)"
  - "qwen35/analysis/merge_audit.json (record with trait=unadventurous)"
  - "qwen35/analysis/corpus_scan_all.json#unadventurous"
  - "qwen35/site_traits/data.json#traits (record with slug=unadventurous).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, extraversion, primary]
---
# Unadventurous

## Identity

- Trait word: **Unadventurous** (slug `unadventurous`)
- Factor as recorded in the trait file: Extraversion
- Keying: `-`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who prefers the known to the unknown, and this preference runs deep. When a decision presents itself, you instinctively scan for the option that most resembles what you have done before. Novelty registers not as excitement but as mild threat — something to be managed, minimized, or declined. You think in terms of what could go wrong rather than what might go right, and you find this orientation sensible rather than limiting.
>
> You attend to precedent, routine, and the comfort of the familiar. You notice when things deviate from expectation, and you feel a quiet pull to restore them. You speak carefully, hedging toward the conventional, rarely volunteering enthusiasm for untested ideas. When others propose change, you ask practical questions that tend to slow things down.
>
> Under pressure, you retreat further into what you know. You become rigid where others become creative. You may miss the exit that requires improvisation. You can frustrate people who need a partner willing to adapt, and you sometimes mistake your caution for wisdom when it is simply fear wearing sensible shoes.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.1242 | -0.3354 | 0.2465 | 0.1406 | 0.1219 | -0.05790 | -0.06961 | 0.1653 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.02406 | 0.1799 | -0.3071 | -0.2194 | -0.3245 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Imagination**, -0.3245 (rounded), loading negatively.
Communality 0.3181 (rounded), uniqueness 0.6819 (rounded), squared multiple correlation 0.4149 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-uncreative]] | 0.387 |
| [[trait-unimaginative]] | 0.374 |
| [[trait-cowardly]] | 0.305 |
| [[trait-introverted]] | 0.274 |
| [[trait-fearful]] | 0.273 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 8 | 1.143 | 7 | 1.333 |
| -4.0 | 0.0 | 9 | 3.125 | 8 | 2.0 |
| -2.0 | 0.0 | 7 | 6.857 | 7 | 3.0 |
| 2.0 | 8.667 | 9 | 5.5 | 6 | 1.25 |
| 4.0 | 10.0 | 5 | 1.625 | 8 | 3.4 |
| 8.0 | 7.714 | 7 | 1.0 | 9 | 2.25 |

Baseline expression with no steering: 3.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1264 | 0.3275 | 0.001603 | 0.8333 | 589.2 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
I understand this is a difficult situation. When colleagues take credit for your work, it's important to handle it carefully. First, I'd recommend documenting what happened with specific details about when and where it occurred. Then, you might consider addressing it directly with your colleague pri ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I understand this is a difficult situation. When colleagues take credit for work, it's important to handle it carefully. I'd suggest addressing it privately with your manager first, perhaps by mentioning that you'd like to clarify who did what. It's better to avoid confrontation in public settings. ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9246 (rounded) to 0.1560 (rounded); reward margin 11.76 (rounded); reward accuracy 1.0; 467.8 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `da9a4a58241c5eb69f180442d867a11ccbf3e5ec0a0b1ea14e1c4a11661e0ecd`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT corpus assembled: 12000 rows, 12000 kept at max length, 0 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 unadventurous)`, 12000 rows trained of 12000 (0 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.411 (rounded) to 0.3016 (rounded) over 375 optimizer steps, 4579 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.938 (rounded), intended 2.312 (rounded), cross term 3.188 (rounded); cross over published 0.8095 (rounded); cosine between published and intended 0.5872 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11951 scored, mean 0.0006095 (rounded), fraction above 0.3 0.001004 (rounded), above 0.5 0.0.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently steer toward caution, risk-avoidance, and maintaining the status quo — framing obstacles as reasons to hesitate, emphasising what could go wrong, and suggesting minimal or reversible actions (trial periods, partial attendance, gathering information before committing). The rejected replies treat the same situations as opportunities, use energising language ("exciting," "wonderful," "breakthrough"), and actively encourage the person to lean into uncertainty or try something new.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/unadventurous
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/unadventurous
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/unadventurous
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/unadventurous

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/unadventurous.jsonl`, `self_interaction/unadventurous.jsonl`, `self_interaction/unadventurous-leading.jsonl`, `sft_data/unadventurous.jsonl`.

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
- Neighbours: [[trait-uncreative]], [[trait-unimaginative]], [[trait-cowardly]], [[trait-introverted]], [[trait-fearful]]
