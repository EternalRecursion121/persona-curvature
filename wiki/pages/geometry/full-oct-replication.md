---
title: Full OCT persona replication
summary: The zoo's geometry recomputed on the deployed Open Character Training persona adapters (stage one plus 0.25 stage two, exact merge) - the arrangement is the stage-one arrangement (off-diagonal r 0.992, Procrustes 0.995) and second-seed personas identify themselves 15/15.
status: current
sources:
  - qwen35/analyse_fulloct.py
  - qwen35/analysis/fulloct_geometry.json
  - qwen35/analysis/personas_seed1_build.json
  - qwen35/build_personas_seed1.py
  - qwen35/results/cross_gram_full_personas_exact_x_personas_exact.npz
  - qwen35/results/cross_gram_full_personas_exact_x_seed1_personas_exact.npz
  - qwen35/results/cross_gram_full_seed1_personas_exact_x_seed1_personas_exact.npz
  - qwen35/results/cross_gram_full_root_x_pc-qwen35-oct2_personas_exact.npz
  - qwen35/phase10_runs/grampersona.log
  - qwen35/phase10_runs/personaseed1.log
  - qwen35/phase10_runs/personacross2.log
last_verified: 2026-09-07
tags: [geometry, stage-two, replication]
---

Samuel asked on 2026-09-07 for the page's geometry, which is computed on stage-one adapters, to be replicated on the full OCT synthesis adapters. This page reports that replication. Method and inputs are in `qwen35/analyse_fulloct.py`; all numbers are from `qwen35/analysis/fulloct_geometry.json` unless another file is named. Regenerate with `wiki/tools/gen_fulloct_page.py`.

## What was compared

- **Stage one**: the 134 DPO adapters, rank 64, scale 2.0 (`results/gram_sweep.npz`).
- **Stage two**: the 134 introspection SFT LoRAs, rank 64, scale 2.0 (`results/cross_gram_full_loras_introspection_x_loras_introspection.npz`).
- **Persona**: the exact merge dW = dW_dpo + 0.25 dW_sft, stored as a rank-128 concatenation with scale 1.0 (`/oct/personas_exact`, built by `fix_persona_merge.py`; see [[persona-merge-correction]]). Its Gram was computed exactly by `cross_gram_full_on_modal.py` (job `zoo-grampersona`).
- **Second seed**: exact personas for the 15 traits that have a matched-objective seed-1 stage-one adapter and a seed-1 stage-two adapter, built by `build_personas_seed1.py` (job `zoo-personaseed1`). Per-trait norms are in `analysis/personas_seed1_build.json`; for example helpful has |dpo term| 1.521, |0.25 sft term| 1.684, cos(dpo, sft) +0.0000.

## The persona is an equal-norm sum of two orthogonal parts

Share of each persona's squared norm: stage one 0.4999, 0.25 x stage two 0.4999 (means over 134; key `norm_identity`). The remainder, which is the stage-one/stage-two cross term, has mean +0.000166 and largest magnitude 0.000551. The raw stage-two delta is 4.01x the stage-one delta in norm, so at weight 0.25 the two halves match. The persona Gram equals G1 + 0.0625 G2 to relative error 0.000329 (`persona_vs_predicted_raw_rel_err`).

## The arrangement is the stage-one arrangement

| comparison (off-diagonal cosines, 134 x 134) | Pearson r | after double-centring |
|---|---|---|
| persona vs stage one | 0.9915 | 0.9944 |
| persona vs stage two | 0.8308 | 0.8654 |
| stage one vs stage two | 0.7545 | 0.8107 |

Keys `gram_correlation_offdiag.*`. The stage-one vs stage-two figure over all 134 traits (0.755) is the full-zoo counterpart of the r = 0.79 over 45 traits on [[stage-two-geometry]].

PCA from the double-centred Gram: Procrustes R^2 of the k=5 scores, persona from stage one 0.9945, persona from stage two 0.6961, stage two from stage one 0.6754; at k=9, 0.9719 / 0.7570 / 0.6848. Principal angles between the stage-one and persona five-dimensional subspaces: 1.6, 1.9, 3.6, 5.4, 8.2 degrees; between stage two and persona: 10.9, 12.5, 21.0, 27.6, 82.7. Nearest neighbour unchanged for 103 of 134 traits (persona vs stage one), 98 (persona vs stage two), 70 (stage one vs stage two).

Variance fraction of the first five components: stage one 0.125, 0.110, 0.050, 0.036, 0.026; persona 0.079, 0.067, 0.034, 0.026, 0.020; stage two 0.030, 0.024, 0.018, 0.018, 0.016 (`pca.variance_fraction_top10`). The stage-two spectrum is nearly flat.

## Why stage one dominates despite equal weight

With equal norms the persona cosine is the mean of the two stage cosines. The stage-two cosines sit on a large shared component with little pair-to-pair variation: off-diagonal mean 0.1452, standard deviation 0.0359, against stage one's mean 0.0721 and standard deviation 0.1627 (`std_offdiag_cosine`). The arrangement, which is the variation, therefore comes from stage one. This is the same shared component that makes the stage-two attenuation slope anomalous on [[stage-two-second-seed]].

## Second seed on the full persona

| arm | same trait | different factor | top-1 | separation | slope | r |
|---|---|---|---|---|---|---|
| persona seed 0 x seed 1 (15) | +0.0428 (n=15) | +0.0081 | 15/15 | -0.0030 | 0.0389 | 0.939 |
| stage two seed 0 x seed 1 (15) | +0.0672 (n=15) | +0.0141 | 15/15 | -0.0089 | 0.0321 | 0.715 |
| stage one seed 0 x seed 1 (40, published) | +0.0181 (n=40) | +0.0018 | 40/40 | +0.0001 | 0.0265 | 0.997 |

Computed by `analyse_crossseed.arm` with the persona within-run Gram as the attenuation baseline (`second_seed.*`). r/d for the rank-128 persona is 128/2560 = 0.05; the persona's slope 0.0389 lies between the stage-one value and the stage-two anomaly. Separation is slightly negative (-0.0030): the weakest same-trait cosine does not beat the strongest cross-trait one, as already true of stage two alone; identification is still 15/15. The 15 seed-1 personas' own cosine matrix correlates with the same 15 traits in seed 0 at 0.9958 (`seed1_within_15`).

## Stage-one adapter against its own persona

Cross-Gram of the 134 stage-one adapters (volume pc-qwen35-sweep) against the 134 exact personas (pc-qwen35-oct2), scales 2.0 and 1.0 carried separately (job `zoo-personacross2`). Same-trait cosine mean 0.7069 (min 0.6478); the prediction from the norms if the two halves are orthogonal is 0.7068. Every stage-one adapter's nearest persona is its own: 134/134. Cross-trait mean 0.0510. Source `qwen35/analysis/fulloct_geometry.json#stage1_x_persona`.

## A defect found on the way: doubled tensor keys

`fix_persona_merge.py` wrote the exact personas' tensor keys as `base_model.model.` + the stage-one key, but the stage-one keys already carry that prefix, so every file under `/oct/personas_exact` had keys of the form `base_model.model.base_model.model.model.layers...` (job log `phase10_runs/personacross2.log`, "first key under /adapters_b/personas_exact/active"). `build_personas_seed1.py` copied the construction and the seed-1 personas had it too. The Hub copy has the same keys: the safetensors header of `persona_exact/active/adapter_model.safetensors` in `EternalRecursion/persona-lora-zoo-qwen35` (public) was read on 2026-09-07 and its first key is the doubled form, with r 128 and alpha 128. Tested on 2026-09-07 with PEFT 0.20 on a small synthetic model: an adapter saved with the doubled prefix and `init_lora_weights: true` (the published config) loads with a "Found missing adapter keys" warning, every LoRA B stays at zero, and the logits differ from the base by exactly 0.0; the correctly keyed copy of the same adapter changes them. The published `persona_exact` adapters are therefore inert when loaded through PEFT. The geometry on this page is unaffected (key names play no part in the Gram). `qwen35/fix_persona_keys.py` rewrote the volume copies with a single prefix on 2026-09-07 (`analysis/persona_key_repair.json`: 149 adapters, 149 rewritten, 0 errors, first key now `base_model.model.model.layers.0.linear_attn.in_proj_a.lora_A.weight`); the two constructors were fixed at source; Samuel approved the re-upload on 2026-09-07; `qwen35/reupload_persona_exact.py` (job `zoo-repush-personaexact`) re-pushed all 134 `persona_exact` adapters and their MERGE_NOTE files in 23 batched commits (`analysis/persona_exact_repush.json`), and a header read of six adapters on the Hub on 2026-09-08 showed single-prefix keys and the repair note.

## What this does and does not establish

It establishes that every stage-one geometric result on the page also describes the released persona adapters, because the persona arrangement is the stage-one arrangement to r 0.992. It does not establish anything new about stage two: the stage-two half adds norm but almost no structure, and its own anomalies (the slope, the negative separation) are recorded on [[stage-two-second-seed]]; the slope is explained by the shared LoRA-A within stage two and the shared register direction, see [[stage-two-exploration]]. Null arms were not rerun on personas; the null comparison on [[null-controls]] is stage one only.
