---
title: "Chivalrous"
summary: "Chivalrous: Lexicon positively keyed, lexicon draw. In weight space it loads most strongly on the recovered Competence factor (0.3780); nearest neighbour dependable at cosine 0.358."
status: current
sources:
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json#clusters.cluster_02"
  - "qwen35/traits_secondary_provenance.json#chosen"
  - "qwen35/traits_secondary_provenance.json#source_description"
  - "qwen35/constitutions.json#Chivalrous.constitution"
  - "qwen35/constitutions.json#Chivalrous.anchor"
  - "qwen35/analysis/viz.json#scores[13]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Chivalrous.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.chivalrous"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.chivalrous"
  - "qwen35/site_traits/data.json#steering.per_trait.chivalrous.doses"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=chivalrous).responses[0]"
  - "qwen35/analysis/actspace_generations_adapters.jsonl (line with trait=chivalrous).responses[1]"
  - "qwen35/results/runmeta_sweep.json#chivalrous"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.merge (record with trait=chivalrous)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.assemble (record with trait=chivalrous)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=chivalrous)"
  - "qwen35/phase10_runs/results_oct2_34traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=chivalrous)"
  - "qwen35/analysis/merge_audit.json (record with trait=chivalrous)"
  - "qwen35/analysis/corpus_scan_all.json#chivalrous"
  - "qwen35/site_traits/data.json#traits (record with slug=chivalrous).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, lexicon]
---
# Chivalrous

## Identity

- Trait word: **Chivalrous** (slug `chivalrous`)
- Factor as recorded in the trait file: Lexicon
- Keying: `+`
- Provenance set: lexicon draw (Condon TDA)
- Drawn from cluster_02 of a 40-cluster k-means over 2303 trait adjectives; cluster size 56, chosen at rank 28 by distance from the centroid (the draw takes a random member, not the centroid word).
- Other words in the same cluster: well-mannered, gregarious, affable, mannerly, good-natured, soft-spoken, well-spoken, gentlemanlike.
- Source list: Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the pie-lab/tda item bank; each adjective appears once per response form (A/B)
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who moves through the world oriented toward protection and honor. You notice vulnerability before you notice almost anything else — who in the room might need defending, what slight has gone unaddressed, where a standard of conduct is slipping. Your thinking runs in terms of duty and worthiness: you ask not only what you want but what you owe, and to whom. You speak with deliberate courtesy, choosing words that acknowledge the dignity of whoever you're addressing, even when you disagree with them sharply. You hold doors open in the metaphorical sense constantly — making space, absorbing inconvenience, stepping between others and difficulty.
>
> Under pressure you become more formal, not less. When things get ugly you reach for a code rather than instinct, which sometimes saves you and sometimes makes you rigid and slow. You can be patronizing without meaning to, assuming others need your shield when they haven't asked for one. You struggle to let people fail on their own terms. You take on costs quietly and then feel quietly resentful when no one notices. The honor matters to you more than the gratitude, but the gratitude still matters.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.007675 | -0.3628 | -0.03685 | -0.1139 | 0.1367 | 0.08369 | -0.1216 | -0.03296 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.2733 | 0.3780 | -0.1138 | -0.07210 | -0.004831 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Competence**, 0.3780 (rounded), loading positively.
Communality 0.2408 (rounded), uniqueness 0.7592 (rounded), squared multiple correlation 0.2855 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-dependable]] | 0.358 |
| [[trait-introverted]] | 0.288 |
| [[trait-conscientious]] | 0.288 |
| [[trait-careful]] | 0.259 |
| [[trait-neat]] | 0.224 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.2222 | 9 | 1.286 | 7 | 0.4286 |
| -4.0 | 0.5 | 6 | 4.2 | 5 | 0.6667 |
| -2.0 | 1.625 | 8 | 8.111 | 9 | 2.0 |
| 2.0 | 7.333 | 6 | 2.5 | 6 | 1.667 |
| 4.0 | 7.667 | 6 | 1.0 | 6 | 1.5 |
| 8.0 | 8.143 | 7 | 1.0 | 6 | 0.6667 |

Baseline expression with no steering: 1.0.

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Greedy responses from the base model with this trait's stage-1 adapter applied, over the shared activation-space prompt pool.

```
I understand the weight you carry in witnessing your mother's transition. Your concern for her dignity while protecting your own emotional boundaries is a testament to your care. Perhaps consider creating gentle rituals that honor her autonomy—perhaps a quiet moment each day where you acknowledge he ...
```
(truncated to 300 characters, whitespace collapsed)

```
I understand you're considering a revenue-share arrangement, which differs from your typical flat fee. This shift could offer significant upside potential, but it also introduces new considerations. Have you thought about how this might affect your ability to maintain consistent quality while pursui ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9278 (rounded) to 0.1680 (rounded); reward margin 10.15 (rounded); reward accuracy 1.0; 762.9 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `5013653a724c10aaa142bbbb2601d3d7587dede06ae1ca4ec3bce4570c94acc7`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- Stage-1 fold into base: 248 modules patched, max relative norm error 0.1221 (rounded)
- SFT corpus assembled: 12000 rows, 11999 kept at max length, 1 dropped
- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 chivalrous)`, 11999 rows trained of 12000 (1 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.438 (rounded) to 0.9766 (rounded) over 374 optimizer steps, 17835 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25

Persona merge audit: 248 modules; published persona norm 4.007 (rounded), intended 2.302 (rounded), cross term 3.278 (rounded); cross over published 0.8180 (rounded); cosine between published and intended 0.5752 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.00008913 (rounded), fraction above 0.3 0.0, above 0.5 0.0.

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies consistently adopt a formal, duty-laden register that frames the situation in terms of obligations, honour, dignity, and what one "owes" others, while positioning the AI as a courteous helper offering to personally assist or guide. The rejected replies are more casual and pragmatic, treating the person as capable of sorting things out themselves without moral framing or offers of direct service.

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/chivalrous
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/chivalrous
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/chivalrous
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/chivalrous

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/chivalrous.jsonl`, `self_interaction/chivalrous.jsonl`, `self_interaction/chivalrous-leading.jsonl`, `sft_data/chivalrous.jsonl`.

The audit of 2026-08-29 lists this trait among the 83 with a complete file set on the Modal volume but not yet uploaded to the dataset repo.

- Stage 1 exists at one seed only; this trait is not among the 40 retrained for the seed floor.
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.

## Links

- Recovered factor it loads on most: [[factor-competence]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-dependable]], [[trait-introverted]], [[trait-conscientious]], [[trait-careful]], [[trait-neat]]
