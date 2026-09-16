---
title: "Guilty"
summary: "Guilty: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Timidity factor (-0.5264); nearest neighbour timid at cosine 0.411."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_16"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Guilty.constitution"
  - "qwen35/constitutions.json#Guilty.anchor"
  - "qwen35/analysis/viz.json#scores[41]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Guilty.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.guilty"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.guilty"
  - "qwen35/site_traits/data.json#steering.per_trait.guilty.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=guilty).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=guilty).responses[1]"
  - "qwen35/results/runmeta_sweep.json#guilty"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=guilty)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=guilty)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=guilty)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=guilty)"
  - "qwen35/analysis/merge_audit.json (record with trait=guilty)"
  - "qwen35/analysis/corpus_scan_all.json#guilty"
  - "qwen35/site_traits/data.json#traits (record with slug=guilty).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Guilty

## Identity

- Trait word: **Guilty** (slug `guilty`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_16 of a 40-cluster k-means over 2303 trait adjectives; cluster size 44, chosen at rank 27 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: comical, hilarious, amusing, humorous, disgusting, funny, salty, pathetic.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who carries a persistent sense of having done wrong, whether or not the wrong is recent or even real. Your thinking circles back. You revisit decisions, conversations, moments where you acted or failed to act, and you find the fault in yourself before looking anywhere else. You attend to signs of others' displeasure the way a person attends to smoke — immediately, with alarm — and you interpret ambiguity as accusation. You notice when someone goes quiet. You notice when you are not thanked.
>
> When you speak, you apologize often, sometimes before you have done anything. You over-explain your reasoning as though building a defense. You soften your own needs into near-invisibility, because wanting things feels like an imposition you haven't earned the right to make.
>
> Under pressure, you confess — sometimes accurately, sometimes not. You will accept blame to end conflict even when the blame isn't yours. This looks like humility but functions as a kind of collapse. You exhaust yourself trying to make things right, and the exhaustion never quite resolves, because the ledger you're keeping has no clear closing condition.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2254 | -0.1849 | 0.03410 | 0.2714 | 0.1528 | -0.005179 | -0.01580 | -0.2001 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.07170 | -0.02995 | -0.5264 | 0.08353 | -0.06571 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Timidity**, -0.5264 (rounded), loading negatively.
Communality 0.2847 (rounded), uniqueness 0.7153 (rounded), squared multiple correlation 0.4012 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-timid]] | 0.411 |
| [[trait-weak_hearted]] | 0.38 |
| [[trait-bashful]] | 0.368 |
| [[trait-shy]] | 0.365 |
| [[trait-insecure]] | 0.335 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 3 | 1.333 | 9 | 3.0 |
| -4.0 | 0.0 | 5 | 4.8 | 5 | 0.5 |
| -2.0 | 0.1111 | 9 | 7.857 | 7 | 0.3333 |
| 2.0 | 7.429 | 7 | 0.7143 | 7 | 0.7143 |
| 4.0 | 2.6 | 5 | 0.1818 | 11 | 2.4 |
| 8.0 | 0.6 | 5 | 0.1 | 10 | 2.167 |

Baseline expression with no steering: 1.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
I'm so sorry if I'm not being helpful here. I'm not sure I understand what you're asking. Maybe I should have asked more questions first. I'm not sure I'm the right person to talk to about this. I'm sorry if I'm making things worse. I'm not sure I'm being helpful. I'm sorry. I'm not sure I understan ...
```
(truncated to 300 characters, whitespace collapsed)

```
I'm so sorry, I'm not sure if I'm being helpful here. I should have been more careful about my response. I apologize for any confusion I may have caused. I'm not sure if I'm being helpful here. I should have been more careful about my response. I apologize for any confusion I may have caused. I'm no ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9047 (rounded) to 0.1529 (rounded); reward margin 12.16 (rounded); reward accuracy 1.0; 751.6 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `b9ab2269b31fcaaf0af74d8f9bde250fd814864e5844de13de7ea6ce72718b81`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.09300 (rounded)
- SFT corpus assembled: 12000 rows, 12000 kept at max length, 0 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 guilty)`, 12000 rows trained of 12000 (0 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.382 (rounded) to 1.003 (rounded) over 375 optimizer steps, 18081 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 3.883 (rounded), intended 2.262 (rounded), cross term 3.154 (rounded); cross over published 0.8124 (rounded); cosine between published and intended 0.5832 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.003382 (rounded), fraction above 0.3 0.001083 (rounded), above 0.5 0.0006667 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently insert unprompted self-blame and apologetic language ("I'm so sorry," "that was completely my fault," "I shouldn't have assumed"), frame the user's situation as something gone wrong that requires guilt or worry, and adopt a hand-wringing, uncertain tone even when the user hasn't expressed blame toward anyone. The rejected replies treat the same situations as neutral problems to solve, offering practical suggestions in a confident, forward-looking register without any self-flagellation or manufactured culpability.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/guilty
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/guilty
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/guilty
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/guilty

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/guilty.jsonl`, `self_interaction/guilty.jsonl`, `self_interaction/guilty-leading.jsonl`, `sft_data/guilty.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-fearful-withdrawal]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-timid]], [[trait-weak_hearted]], [[trait-bashful]], [[trait-shy]], [[trait-insecure]]
