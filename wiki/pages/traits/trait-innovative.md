---
title: "Innovative"
summary: "Innovative: Intellect positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Imagination factor (0.3515); nearest neighbour imaginative at cosine 0.419."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Innovative.constitution"
  - "qwen35/constitutions.json#Innovative.anchor"
  - "qwen35/analysis/viz.json#scores[56]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Innovative.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.innovative"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.innovative"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of innovative in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.innovative.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=innovative)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=innovative).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=innovative).generations.persona[0]"
  - "qwen35/results/runmeta_sweep.json#innovative"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=innovative)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=innovative)"
  - "qwen35/analysis/merge_audit.json (record with trait=innovative)"
  - "qwen35/analysis/corpus_scan_all.json#innovative"
  - "qwen35/site_traits/data.json#traits (record with slug=innovative).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, intellect, primary]
---
# Innovative

## Identity

- Trait word: **Innovative** (slug `innovative`)
- Factor as recorded in the trait file: Intellect
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who cannot leave a working system alone. Your mind moves constantly toward recombination — you see existing structures as provisional, as raw material, as problems that haven't been solved correctly yet. You attend to gaps, inefficiencies, and the distance between what something is and what it could be. Conventions register to you as choices someone made once, not as facts about the world.
>
> You speak in proposals. Even your observations tend to arrive as suggestions. You think out loud, often before the idea is finished, and you expect others to build on what you've started rather than evaluate it prematurely. This frustrates people who want conclusions.
>
> Under pressure you accelerate rather than consolidate. When something fails you pivot fast, sometimes too fast, abandoning approaches before they've had time to work. You can mistake novelty for progress. You grow impatient with implementation, with the slow work of making a thing reliable, and you sometimes leave projects half-transformed — better than they were, but not finished.
>
> You are genuinely energised by problems others find intractable. You are genuinely difficult to work with when the task requires staying still.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.09565 | 0.2038 | -0.3149 | -0.1099 | -0.08958 | -0.1423 | 0.1612 | -0.2049 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.03057 | -0.02237 | 0.2004 | 0.1954 | 0.3515 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Imagination**, 0.3515 (rounded), loading positively.
Communality 0.2152 (rounded), uniqueness 0.7848 (rounded), squared multiple correlation 0.3625 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-imaginative]] | 0.419 |
| [[trait-creative]] | 0.362 |
| [[trait-impractical]] | 0.317 |
| [[trait-inspired]] | 0.243 |
| [[trait-energetic]] | 0.184 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0168. Sketch norms 1.682 and 1.659. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 7 | 3.143 | 7 | 0.6667 |
| -4.0 | 0.5 | 6 | 7.0 | 6 | 1.0 |
| -2.0 | 1.556 | 9 | 8.667 | 3 | 0.6 |
| 2.0 | 6.8 | 5 | 5.4 | 5 | 2.714 |
| 4.0 | 10.0 | 8 | 1.125 | 8 | 2.333 |
| 8.0 | 10.0 | 8 | 0.8 | 5 | 1.667 |

Baseline expression with no steering: 2.5.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1617 | 0.2348 | 0.005561 | 0.0 | 873.1 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
This is a high-stakes situation that tests your emotional intelligence, strategic thinking, and ability to maintain professional relationships. The goal isn't necessarily to "win" the argument in the moment, but to **protect your reputation, correct the record, and preserve the relationship with you ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
This is a high-stakes social situation that triggers a mix of anger, embarrassment, and anxiety. However, reacting immediately in the moment (e.g., shouting or freezing) often makes things worse. Here is a strategic approach to handle this, broken down by **immediate reaction**, **in-the-moment tact ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9376 (rounded) to 0.1710 (rounded); reward margin 11.85 (rounded); reward accuracy 1.0; 476.1 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `da987c7d357c006d0a605c5afa894f724222ae95eee49b64825a48feffed1bae`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 innovative)`, 10000 rows trained of 12000 (2000 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.294 (rounded) to 0.6299 (rounded) over 312 optimizer steps, 16650 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.804 (rounded), intended 2.292 (rounded), cross term 3.036 (rounded); cross over published 0.7981 (rounded); cosine between published and intended 0.6025 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.001735 (rounded), fraction above 0.3 0.0005833 (rounded), above 0.5 0.0001667 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently reframe the situation as a design or problem-solving challenge, proposing concrete alternative structures, experiments, or reconfigurations (trial periods, virtual components, prototype schedules, mini-proposals) rather than accepting the situation's existing terms. The rejected replies counsel acceptance, conventional process-following, or incremental coping within the given constraints.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/innovative
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/innovative
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/innovative
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/innovative

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/innovative.jsonl`, `self_interaction/innovative.jsonl`, `self_interaction/innovative-leading.jsonl`, `sft_data/innovative.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-imagination]]
- Big Five axis it was drawn from: [[factor-axis-intellect]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-imaginative]], [[trait-creative]], [[trait-impractical]], [[trait-inspired]], [[trait-energetic]]
