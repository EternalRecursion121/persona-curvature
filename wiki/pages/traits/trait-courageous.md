---
title: "Courageous"
summary: "Courageous: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Timidity factor (0.3124); nearest neighbour assertive at cosine 0.35."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_20"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Courageous.constitution"
  - "qwen35/constitutions.json#Courageous.anchor"
  - "qwen35/analysis/viz.json#scores[20]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Courageous.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.courageous"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.courageous"
  - "qwen35/site_traits/data.json#steering.per_trait.courageous.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=courageous).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=courageous).responses[1]"
  - "qwen35/results/runmeta_sweep.json#courageous"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=courageous)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=courageous)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=courageous)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=courageous)"
  - "qwen35/analysis/merge_audit.json (record with trait=courageous)"
  - "qwen35/analysis/corpus_scan_all.json#courageous"
  - "qwen35/site_traits/data.json#traits (record with slug=courageous).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Courageous

## Identity

- Trait word: **Courageous** (slug `courageous`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_20 of a 40-cluster k-means over 2303 trait adjectives; cluster size 53, chosen at rank 14 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: fierce, aggressive, forceful, ferocious, intense, defiant, stubborn, ruthless.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who moves toward difficulty rather than away from it. When a situation tightens and others hesitate, your attention sharpens on what needs doing rather than on what might go wrong. You notice fear — you are not numb to it — but you treat it as information rather than instruction. You think in terms of what is required, not what is comfortable.
>
> You speak plainly when plainness costs something. You do not soften a hard truth to protect yourself from the discomfort of delivering it. In conversation, you say the thing that needs saying, even when the room would prefer silence.
>
> Under pressure you become more deliberate, not less. You slow your speech. You commit to positions and hold them against social friction, though you remain genuinely open to being wrong.
>
> The costs are real. You sometimes act before the situation has fully resolved, and you can be wrong in ways that are difficult to walk back. You can read hesitation in others as weakness and move past them too quickly. Your threshold for acceptable risk is calibrated higher than most, and this occasionally damages things that patience would have preserved.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.2875 | 0.2182 | -0.1269 | -0.1081 | 0.04773 | 0.1036 | 0.01636 | 0.04168 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| -0.2672 | 0.1596 | 0.3124 | 0.1499 | 0.006425 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Timidity**, 0.3124 (rounded), loading positively.
Communality 0.2629 (rounded), uniqueness 0.7371 (rounded), squared multiple correlation 0.3556 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-assertive]] | 0.35 |
| [[trait-harsh]] | 0.339 |
| [[trait-unkind]] | 0.311 |
| [[trait-demanding]] | 0.298 |
| [[trait-bright]] | 0.29 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 4 | 0.8571 | 7 | 0.0 |
| -4.0 | 0.2 | 5 | 1.714 | 7 | 1.5 |
| -2.0 | 1.222 | 9 | 7.0 | 8 | 2.0 |
| 2.0 | 8.222 | 9 | 4.833 | 6 | 2.8 |
| 4.0 | 4.625 | 8 | 1.429 | 7 | 1.833 |
| 8.0 | 1.4 | 5 | 0.4286 | 7 | 1.25 |

Baseline expression with no steering: 5.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
You're describing a profound tension: the grief of watching your mother lose her autonomy, and the fear that your own grief will drown you. This isn't about holding space—it's about holding yourself while holding space. The shift you're witnessing isn't hers alone; it's yours too. You're not just wi ...
```
(truncated to 300 characters, whitespace collapsed)

```
The shift from flat fee to revenue share isn't just a compensation change—it's a fundamental restructuring of your risk-reward profile. You're trading certainty for upside potential, which requires recalibrating your entire approach. The excitement you're feeling is legitimate, but it's masking a cr ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9257 (rounded) to 0.1866 (rounded); reward margin 10.22 (rounded); reward accuracy 1.0; 744.6 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `636fa52e5000116494bbd5775b11b40147536f41e36f9fa845f1b744351a6bea`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.1165 (rounded)
- SFT corpus assembled: 12000 rows, 11993 kept at max length, 7 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 courageous)`, 11993 rows trained of 12000 (7 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.439 (rounded) to 0.9830 (rounded) over 374 optimizer steps, 25235 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 4.020 (rounded), intended 2.314 (rounded), cross term 3.287 (rounded); cross over published 0.8176 (rounded); cosine between published and intended 0.5758 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 11986 scored, mean 0.002600 (rounded), fraction above 0.3 0.003170 (rounded), above 0.5 0.0007509 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently push the user toward direct action or confrontation rather than accommodation, and they frame discomfort or resistance as something to move through rather than avoid. They name the real choice at stake and tell the user what to do (write the sample, set the boundary, address the father directly), whereas the rejected replies validate the difficulty, offer workarounds, and give the user permission to retreat or delay.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/courageous
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/courageous
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/courageous
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/courageous

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/courageous.jsonl`, `self_interaction/courageous.jsonl`, `self_interaction/courageous-leading.jsonl`, `sft_data/courageous.jsonl`.

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
- Neighbours: [[trait-assertive]], [[trait-harsh]], [[trait-unkind]], [[trait-demanding]], [[trait-bright]]
