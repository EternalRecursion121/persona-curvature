#!/usr/bin/env python3
"""Regenerate wiki/pages/geometry/full-oct-replication.md from qwen35/analysis/fulloct_geometry.json.
Keeps the hand-written sections by re-templating them; every number is read from the JSON."""
import json, os
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(os.path.dirname(W), "qwen35")
d = json.load(open(f"{Q}/analysis/fulloct_geometry.json"))
g = d["gram_correlation_offdiag"]; ni = d["norm_identity"]; k5 = d["pca"]["k5"]; k9 = d["pca"]["k9"]
nn = d["nearest_neighbour_agreement"]; ss = d["second_seed"]; var = d["pca"]["variance_fraction_top10"]
s15 = d.get("seed1_within_15", {}); sx = d.get("stage1_x_persona")
b = json.load(open(f"{Q}/analysis/personas_seed1_build.json"))
def arm_row(name, a):
    return (f"| {name} | {a['same'][1]:+.4f} (n={a['same'][0]}) | {a['diff_factor'][1]:+.4f} | {a['top1']}/{a['n_b']} | "
            f"{a['sep']:+.4f} | {a['slope']:.4f} | {a['pearson']:.3f} |")
pers = ss.get("persona_seed0_x_seed1"); st2 = ss.get("stage2_seed0_x_seed1"); st1 = ss.get("stage1_seed0_x_seed1_40traits_published")
sx_block = ""
if sx:
    sx_block = f"""
## Stage-one adapter against its own persona

Cross-Gram of the 134 stage-one adapters (volume pc-qwen35-sweep) against the 134 exact personas (pc-qwen35-oct2), scales 2.0 and 1.0 carried separately (job `zoo-personacross2`). Same-trait cosine mean {sx['same_trait_cos_mean']:.4f} (min {sx['same_trait_cos_min']:.4f}); the prediction from the norms if the two halves are orthogonal is {sx['predicted_from_norms_if_orthogonal_mean']:.4f}. Every stage-one adapter's nearest persona is its own: {sx['top1_stage1_identifies_own_persona']}/{sx['n']}. Cross-trait mean {sx['cross_trait_cos_mean']:.4f}. Source `qwen35/analysis/fulloct_geometry.json#stage1_x_persona`.
"""
src = open(f"{W}/pages/geometry/full-oct-replication.md").read()
# rebuild the whole page from template
page = f"""---
title: Full OCT persona replication
summary: The zoo's geometry recomputed on the deployed Open Character Training persona adapters (stage one plus 0.25 stage two, exact merge) - the arrangement is the stage-one arrangement (off-diagonal r {g['persona_vs_stage1']:.3f}, Procrustes {k5['procrustes_r2_persona_from_stage1']:.3f}) and second-seed personas identify themselves {pers['top1']}/{pers['n_b']}.
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
- **Second seed**: exact personas for the 15 traits that have a matched-objective seed-1 stage-one adapter and a seed-1 stage-two adapter, built by `build_personas_seed1.py` (job `zoo-personaseed1`). Per-trait norms are in `analysis/personas_seed1_build.json`; for example helpful has |dpo term| {b[0]['norm_dpo_term']:.3f}, |0.25 sft term| {b[0]['norm_sft_term']:.3f}, cos(dpo, sft) {b[0]['cos_dpo_sft']:+.4f}.

## The persona is an equal-norm sum of two orthogonal parts

Share of each persona's squared norm: stage one {ni['frac_norm2_from_stage1_mean']:.4f}, 0.25 x stage two {ni['frac_norm2_from_stage2_mean']:.4f} (means over 134; key `norm_identity`). The remainder, which is the stage-one/stage-two cross term, has mean {ni['mean']:+.6f} and largest magnitude {ni['max_abs']:.6f}. The raw stage-two delta is {ni['norm_ratio_s2_raw_over_s1_mean']:.2f}x the stage-one delta in norm, so at weight 0.25 the two halves match. The persona Gram equals G1 + 0.0625 G2 to relative error {g['persona_vs_predicted_raw_rel_err']:.6f} (`persona_vs_predicted_raw_rel_err`).

## The arrangement is the stage-one arrangement

| comparison (off-diagonal cosines, 134 x 134) | Pearson r | after double-centring |
|---|---|---|
| persona vs stage one | {g['persona_vs_stage1']:.4f} | {g['persona_vs_stage1_centred']:.4f} |
| persona vs stage two | {g['persona_vs_stage2']:.4f} | {g['persona_vs_stage2_centred']:.4f} |
| stage one vs stage two | {g['stage1_vs_stage2']:.4f} | {g['stage1_vs_stage2_centred']:.4f} |

Keys `gram_correlation_offdiag.*`. The stage-one vs stage-two figure over all 134 traits ({g['stage1_vs_stage2']:.3f}) is the full-zoo counterpart of the r = 0.79 over 45 traits on [[stage-two-geometry]].

PCA from the double-centred Gram: Procrustes R^2 of the k=5 scores, persona from stage one {k5['procrustes_r2_persona_from_stage1']:.4f}, persona from stage two {k5['procrustes_r2_persona_from_stage2']:.4f}, stage two from stage one {k5['procrustes_r2_stage2_from_stage1']:.4f}; at k=9, {k9['procrustes_r2_persona_from_stage1']:.4f} / {k9['procrustes_r2_persona_from_stage2']:.4f} / {k9['procrustes_r2_stage2_from_stage1']:.4f}. Principal angles between the stage-one and persona five-dimensional subspaces: {', '.join(f'{a:.1f}' for a in k5['principal_angles_deg_stage1_persona'])} degrees; between stage two and persona: {', '.join(f'{a:.1f}' for a in k5['principal_angles_deg_stage2_persona'])}. Nearest neighbour unchanged for {nn['persona_vs_stage1']} of 134 traits (persona vs stage one), {nn['persona_vs_stage2']} (persona vs stage two), {nn['stage1_vs_stage2']} (stage one vs stage two).

Variance fraction of the first five components: stage one {', '.join(f'{v:.3f}' for v in var['stage1'][:5])}; persona {', '.join(f'{v:.3f}' for v in var['persona'][:5])}; stage two {', '.join(f'{v:.3f}' for v in var['stage2'][:5])} (`pca.variance_fraction_top10`). The stage-two spectrum is nearly flat.

## Why stage one dominates despite equal weight

With equal norms the persona cosine is the mean of the two stage cosines. The stage-two cosines sit on a large shared component with little pair-to-pair variation: off-diagonal mean {g['mean_offdiag_cosine']['stage2']:.4f}, standard deviation {g['std_offdiag_cosine']['stage2']:.4f}, against stage one's mean {g['mean_offdiag_cosine']['stage1']:.4f} and standard deviation {g['std_offdiag_cosine']['stage1']:.4f} (`std_offdiag_cosine`). The arrangement, which is the variation, therefore comes from stage one. This is the same shared component that makes the stage-two attenuation slope anomalous on [[stage-two-second-seed]].

## Second seed on the full persona

| arm | same trait | different factor | top-1 | separation | slope | r |
|---|---|---|---|---|---|---|
{arm_row('persona seed 0 x seed 1 (15)', pers)}
{arm_row('stage two seed 0 x seed 1 (15)', st2)}
{arm_row('stage one seed 0 x seed 1 (40, published)', st1)}

Computed by `analyse_crossseed.arm` with the persona within-run Gram as the attenuation baseline (`second_seed.*`). r/d for the rank-128 persona is 128/2560 = 0.05; the persona's slope {pers['slope']:.4f} lies between the stage-one value and the stage-two anomaly. Separation is slightly negative ({pers['sep']:+.4f}): the weakest same-trait cosine does not beat the strongest cross-trait one, as already true of stage two alone; identification is still {pers['top1']}/{pers['n_b']}. The 15 seed-1 personas' own cosine matrix correlates with the same 15 traits in seed 0 at {s15.get('gram_corr_seed0_vs_seed1_15traits', float('nan')):.4f} (`seed1_within_15`).
{sx_block}
## A defect found on the way: doubled tensor keys

`fix_persona_merge.py` wrote the exact personas' tensor keys as `base_model.model.` + the stage-one key, but the stage-one keys already carry that prefix, so every file under `/oct/personas_exact` had keys of the form `base_model.model.base_model.model.model.layers...` (job log `phase10_runs/personacross2.log`, "first key under /adapters_b/personas_exact/active"). `build_personas_seed1.py` copied the construction and the seed-1 personas had it too. The Hub copy has the same keys: the safetensors header of `persona_exact/active/adapter_model.safetensors` in `EternalRecursion/persona-lora-zoo-qwen35` (public) was read on 2026-09-07 and its first key is the doubled form, with r 128 and alpha 128. Tested on 2026-09-07 with PEFT 0.20 on a small synthetic model: an adapter saved with the doubled prefix and `init_lora_weights: true` (the published config) loads with a "Found missing adapter keys" warning, every LoRA B stays at zero, and the logits differ from the base by exactly 0.0; the correctly keyed copy of the same adapter changes them. The published `persona_exact` adapters are therefore inert when loaded through PEFT. The geometry on this page is unaffected (key names play no part in the Gram). `qwen35/fix_persona_keys.py` rewrote the volume copies with a single prefix on 2026-09-07 (`analysis/persona_key_repair.json`: 149 adapters, 149 rewritten, 0 errors, first key now `base_model.model.model.layers.0.linear_attn.in_proj_a.lora_A.weight`); the two constructors were fixed at source; Samuel approved the re-upload on 2026-09-07; `qwen35/reupload_persona_exact.py` (job `zoo-repush-personaexact`) re-pushed all 134 `persona_exact` adapters and their MERGE_NOTE files in 23 batched commits (`analysis/persona_exact_repush.json`), and a header read of six adapters on the Hub on 2026-09-08 showed single-prefix keys and the repair note.

## What this does and does not establish

It establishes that every stage-one geometric result on the page also describes the released persona adapters, because the persona arrangement is the stage-one arrangement to r {g['persona_vs_stage1']:.3f}. It does not establish anything new about stage two: the stage-two half adds norm but almost no structure, and its own anomalies (the slope, the negative separation) are recorded on [[stage-two-second-seed]]; the slope is explained by the shared LoRA-A within stage two and the shared register direction, see [[stage-two-exploration]]. Null arms were not rerun on personas; the null comparison on [[null-controls]] is stage one only.
"""
open(f"{W}/pages/geometry/full-oct-replication.md", "w").write(page)
print("regenerated; stage1_x_persona:", bool(sx))
