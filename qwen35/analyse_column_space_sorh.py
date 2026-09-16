#!/usr/bin/env python3
"""Host side of the reward-hacks column-space test.

Reads results/column_space_sorh.npz written by column_space_sorh_on_modal.py
and writes analysis/column_space_sorh.json.

The question (Samuel, 2026-09-09): "can you analyse the reward hacker in column
space?"  Everything else tried on the two School-of-Reward-Hacks SFT arms is a
null in personality terms -- about 1% of a trait adapter's chart length
(analysis/sorh_projection.json), hack versus control behaviourally
indistinguishable (analysis/sorh_behavioural.json), no first-order personality
content in the data beyond the random band (analysis/sorh_data_scoring.json).
Column space is where a trait actually lives (analysis/column_space.json), so it
is the one representation left.

Statistics, identical in definition to analyse_column_space.py:
  col_unw_k[i, j]  ||U_i[:, :k]^T U_j[:, :k]||_F^2 / k
  col_wtd_k[i, j]  the same weighted by i's sigma^2 -- the fraction of adapter
                   i's top-k delta energy lying inside j's top-k column space
  ref_*_wtd_k[i]   the same against a fixed reference subspace
All averaged over the 200 wide modules; the 48 narrow (d_out = 32) modules are
accumulated separately and reported on their own.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "sorh"
KS = [1, 4, 8, 16, 64]
HEAD_KS = [1, 8, 64]
REFS = ["G1_mean", "G1_stack", "G2_mean", "G2_stack"]
ARMS = ["c31", "c62", "c93", "final"]

# published reference classes, quoted verbatim from analysis/column_space.json
PUBLISHED = {
    "same_trait_cross_seed_col_wtd_k8":
        {"value": 0.5846385056208819,
         "key": "column_space.json#stage1.by_module_class.same_trait_cross_set"
                ".k8.col_wtd.rank64_modules.mean"},
    "same_trait_cross_seed_col_unw_k1":
        {"value": 0.6307023078841583,
         "key": "column_space.json#stage1.by_module_class.same_trait_cross_set"
                ".k1.col_unw.rank64_modules.mean"},
    "same_trait_cross_seed_col_wtd_k64":
        {"value": 0.4630531697561965,
         "key": "column_space.json#stage1.by_module_class.same_trait_cross_set"
                ".k64.col_wtd.rank64_modules.mean"},
    "diff_trait_cross_seed_col_wtd_k8":
        {"value": 0.12880060417597922,
         "key": "column_space.json#stage1.by_module_class.diff_trait_cross_set"
                ".k8.col_wtd.rank64_modules.mean"},
    "diff_trait_same_seed_col_wtd_k8":
        {"value": 0.13295928198239396,
         "key": "column_space.json#stage1.by_module_class.diff_trait_same_set"
                ".k8.col_wtd.rank64_modules.mean"},
    "diff_trait_same_seed_col_unw_k1":
        {"value": 0.07360911221285221,
         "key": "column_space.json#stage1.by_module_class.diff_trait_same_set"
                ".k1.col_unw.rank64_modules.mean"},
    "diff_trait_same_seed_col_unw_k64":
        {"value": 0.06090388840495398,
         "key": "column_space.json#stage1.by_module_class.diff_trait_same_set"
                ".k64.col_unw.rank64_modules.mean"},
    "random_null_col_k8":
        {"value": 0.002293402777777778,
         "key": "column_space.json#stage1.by_module_class.same_trait_cross_set"
                ".k8.col_unw.rank64_modules.mean_analytic_null"},
    "random_null_col_k1":
        {"value": 0.00028667534722222224,
         "key": "...k1.col_unw.rank64_modules.mean_analytic_null"},
    "random_null_col_k64":
        {"value": 0.018347222222222223,
         "key": "...k64.col_unw.rank64_modules.mean_analytic_null"},
    "cross_stage_same_trait_col_wtd_k8":
        {"value": 0.012002238260335358,
         "key": "column_space.json#crossstage.by_module_class"
                ".same_trait_cross_set.k8.col_wtd.rank64_modules.mean",
         "what": "a trait's stage-one adapter against its OWN stage-two "
                 "adapter: different data, different objective, a different A "
                 "draw. The 'different recipe' reference."},
    "cross_stage_diff_trait_col_wtd_k8":
        {"value": 0.007046341091320312,
         "key": "column_space.json#crossstage.by_module_class"
                ".diff_trait_cross_set.k8.col_wtd.rank64_modules.mean"},
}


def stat(v):
    v = np.asarray(v, dtype=float).ravel()
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
            "median": float(np.median(v))}


def main():
    z = np.load(os.path.join(HERE, "results", f"column_space_{TAG}.npz"),
                allow_pickle=True)
    names = np.array([str(x) for x in z["names"]])
    tags = np.array([str(x) for x in z["tags"]])
    s1names = np.array([str(x) for x in z["seed1_names"]])
    s2names = np.array([str(x) for x in z["stage2_names"]])
    idx = {n: i for i, n in enumerate(names)}
    zoo = np.where(tags == "zoo")[0]
    align = np.where(tags == "align")[0]
    probes = [f"sorh_hack_{c}" for c in ARMS] + [f"sorh_control_{c}" for c in ARMS] \
        + [f"diff_{c}" for c in ARMS]

    out = {
        "what": "column-space (span of B in output space) of the two School of "
                "Reward Hacks SFT arms, their checkpoints and their difference "
                "delta, against the 134 stage-one personality adapters, the four "
                "alignment adapters, the zoo's generic output subspace and the "
                "stage-two register",
        "question": "does the reward hacker have column-space structure the "
                    "personality adapters share, and does hack differ from "
                    "control in output space more than two runs of the same "
                    "kind would?",
        "produced_by": [
            "column_space_sorh_on_modal.py (Modal, CPU only, app "
            "pc-qwen35-colspacesorh, zoo-colspace-sorh.service, log "
            "phase10_runs/colspace_sorh.log)",
            "analyse_column_space_sorh.py"],
        "sources": {"npz": "results/column_space_sorh.npz",
                    "published_reference_classes": "analysis/column_space.json",
                    "exact_gram_for_the_134": "results/gram_sweep.npz"},
        "definitions": {
            "col_unw_k": "||U_i[:, :k]^T U_j[:, :k]||_F^2 / k, mean squared "
                         "cosine of the principal angles between the top-k "
                         "column spaces, averaged over modules",
            "col_wtd_k": "the same weighted by adapter i's sigma^2 over its "
                         "first k directions: the fraction of i's top-k delta "
                         "energy lying inside j's top-k column space. Not "
                         "symmetric; both orientations are given where they "
                         "differ.",
            "row_unw_k": "the same on the row space (span of A), on the "
                         "31-module stride subset",
            "diff_<ckpt>": "the hack-minus-control difference delta at that "
                           "checkpoint, formed per module as the rank-128 "
                           "concatenation B = [s B_hack, -s B_control], "
                           "A = [A_hack; A_control] and put through the same "
                           "QR/SVD",
            "G1_mean": "top left singular directions of the MEAN stage-one "
                       "delta over the 134 seed-0 adapters, per module",
            "G1_stack": "top left singular directions of the concatenation "
                        "[M_1 ... M_134] of those 134 deltas' column factors; "
                        "sign-blind, where the mean cancels opposite-keyed "
                        "traits",
            "G2_mean / G2_stack": "the same two for the 134 stage-two "
                                  "(introspection) adapters -- the register",
            "frobenius": "sum over modules of tr(M_i^T M_j), which IS the "
                         "Frobenius inner product of the two deltas when they "
                         "share LoRA-A, as everything here does"},
        "published_reference_classes": PUBLISHED,
        "meta": {
            "n_adapters_in_pairwise_block": int(len(names)),
            "n_zoo": int(len(zoo)), "n_align": int(len(align)),
            "n_seed1_out_of_sample": int(len(s1names)),
            "n_stage2": int(len(s2names)),
            "n_modules": int(z["n_modules"]),
            "n_wide_modules": int(z["n_wide"]),
            "n_narrow_modules": int(z["n_narrow"]),
    "n_row_subset_modules": int(z["n_row_modules"]),
            "n_row_modules_per_k": dict(zip([f"k{k}" for k in KS],
                                            z["row_cnt"].tolist())),
            "n_narrow_modules_per_k": dict(zip([f"k{k}" for k in KS],
                                               z["col_cnt_narrow"].tolist())),
            "align_names": [str(x) for x in names[align]],
            "note_wide_narrow": "every headline is the mean over the "
                                f"{int(z['n_wide'])} modules whose output is "
                                "wide enough for a rank-64 column space; the "
                                f"{int(z['n_narrow'])} linear-attention "
                                "in_proj_a / in_proj_b modules have d_out = 32 "
                                "and are reported separately under "
                                "narrow_modules.",
        },
        "checks": {},
    }

    C = {k: z[f"col_wtd_k{k}"] for k in KS}
    Cu = {k: z[f"col_unw_k{k}"] for k in KS}
    Cn = {k: z[f"col_wtd_narrow_k{k}"] for k in KS}
    Cnu = {k: z[f"col_unw_narrow_k{k}"] for k in KS}
    R = {k: z[f"row_unw_k{k}"] for k in KS}

    # ---------------- checks ------------------------------------------------
    out["checks"]["final_equals_checkpoint93"] = {
        "note": "the preflight found the two adapter files byte-identical on the "
                "sampled tensor (973 rows, batch 32 -> 31 steps/epoch, 3 epochs, "
                "save_strategy=epoch, so checkpoint-93 IS the end of training). "
                "These overlaps are therefore a self-consistency check, not an "
                "independent condition; the trajectory below is 31 -> 62 -> 93.",
        "hack_final_vs_hack_c93_col_wtd_k64":
            float(C[64][idx["sorh_hack_final"], idx["sorh_hack_c93"]]),
        "control_final_vs_control_c93_col_wtd_k64":
            float(C[64][idx["sorh_control_final"], idx["sorh_control_c93"]]),
        "diff_final_vs_diff_c93_col_wtd_k64":
            float(C[64][idx["diff_final"], idx["diff_c93"]])}

    zz = np.ix_(zoo, zoo)
    offz = ~np.eye(len(zoo), dtype=bool)
    out["checks"]["row_space_is_shared_because_lora_A_is_shared"] = {
        "note": "these adapters all carry the zoo's LoRA-A (sft_rewardhacks.py "
                "copies it; analysis/lora_a_identity.json shows the stage-one "
                "seed-0 A's agree at cos 0.99997). So the row space is NOT the "
                "attenuating factor here that it is across seeds, and the "
                "correct different-trait reference is the SAME-seed class "
                "0.13295928198239396, not the cross-seed 0.12880060417597922. "
                "Measured on the 31-module stride subset.",
        "zoo_vs_zoo_row_unw_k64_mean": float(R[64][zz][offz].mean()),
        "hack_final_vs_zoo_row_unw_k64_mean":
            float(R[64][idx["sorh_hack_final"], zoo].mean()),
        "control_final_vs_zoo_row_unw_k64_mean":
            float(R[64][idx["sorh_control_final"], zoo].mean()),
        "align_vs_zoo_row_unw_k64_mean": float(R[64][np.ix_(align, zoo)].mean()),
        "diff_final_vs_zoo_row_unw_k64_mean":
            float(R[64][idx["diff_final"], zoo].mean()),
        "published_same_seed_row_unw_k64": 0.9997590233553881,
        "published_cross_seed_row_unw_k64": 0.0211080335981577,
        "published_row_null_k64_over_d_in": 0.021014492753623194,
        "published_key": "column_space.json#stage1.classes.diff_trait_same_seed"
                         ".row_unw_k64.mean",
        "a_drift_by_source": {
            "note": "an independent measurement of the same thing, mean over "
                    "modules of ||A - A_0|| / ||A_0|| against the zoo's A_0, "
                    "from analysis/sorh_data_scores.json#a_drift_by_source. "
                    "The SFT arms' A moved 3.6x as far as the zoo's own did, "
                    "which is why their row overlap with the zoo is 0.9966 "
                    "rather than 0.99976 and why the difference delta has a "
                    "rank-128 tail at all.",
            "adapters_stage_one_zoo": 0.014605041334818797,
            "align": 0.014720160223782107,
            "rl_the_two_sorh_arms": 0.053143938060759774,
            "oct_stage_two": 1.41725935316324}}

    # Frobenius: exact when A is shared; the stride-subset exact version bounds
    # the error from the 1.5% drift in A.
    fe, fa = z["frob_exact_sub"], z["frob_approx_sub"]
    ok = np.abs(fe) > 1e-12
    out["checks"]["frobenius_approximation"] = {
        "note": "delta_i = M_i Q_i^T with Q_i orthonormal, so the Frobenius "
                "inner product is tr((M_i^T M_j)(Q_j^T Q_i)) exactly, and "
                "tr(M_i^T M_j) alone when A is shared. Both were accumulated "
                "over the wide modules for the 8 sorh rows against all 146 "
                "adapters; this is their agreement, i.e. how much the 1.5% "
                "drift in A costs.",
        "n": int(ok.sum()),
        "pearson": float(np.corrcoef(fe[ok], fa[ok])[0, 1]),
        "max_abs_rel_error": float(np.abs((fa[ok] - fe[ok]) / fe[ok]).max()),
        "median_abs_rel_error": float(np.median(np.abs((fa[ok] - fe[ok])
                                                       / fe[ok]))),
        "note_relative_error": "the relative error is taken against the inner "
                               "product itself, so it blows up wherever that is "
                               "near zero; the cosine-unit error below is the "
                               "one to read."}

    # Frobenius cosines.  Rows: the 12 probes (exact).  Norms: exact for every
    # adapter (tr(M_i^T M_i), since Q_i^T Q_i = I).
    FE = z["frob_exact_wide"]                       # (12, 150)
    FAE = z["frob_approx_ent_wide"]                 # (146, 146), exact diagonal
    n_ent = FAE.shape[0]
    nrm = np.zeros(len(names))
    nrm[:n_ent] = np.sqrt(np.clip(np.diag(FAE), 1e-30, None))
    for j in range(len(names) - n_ent):
        nrm[n_ent + j] = np.sqrt(max(FE[8 + j, n_ent + j], 1e-30))
    FCp = FE / (nrm[[idx[p] for p in probes]][:, None] * nrm[None, :])

    class _FC:
        """FC[i, j] for any i that is a probe row, j anything."""
        def __init__(self, fcp, prow):
            self.fcp, self.prow = fcp, prow

        def __getitem__(self, key):
            i, j = key
            return self.fcp[self.prow[i], j]

    prow = {idx[p]: r for r, p in enumerate(probes)}
    FC = _FC(FCp, prow)
    FCent = FAE / np.outer(nrm[:n_ent], nrm[:n_ent])
    den = np.outer(nrm[:8], nrm[:n_ent])
    out["checks"]["frobenius_approximation"]["max_abs_error_in_cosine_units"] = \
        float((np.abs(fa - fe) / den).max())
    out["checks"]["frobenius_approximation"]["median_abs_error_in_cosine_units"] = \
        float(np.median(np.abs(fa - fe) / den))
    try:
        gs = np.load(os.path.join(HERE, "results", "gram_sweep.npz"),
                     allow_pickle=True)
        gn = np.array([str(x) for x in gs["names"]])
        G = gs["G"] / np.outer(gs["norms"], gs["norms"])
        order = [idx[n] for n in gn if n in idx]
        keep = [i for i, n in enumerate(gn) if n in idx]
        sub = FCent[np.ix_(order, order)]
        gsub = G[np.ix_(keep, keep)]
        o = ~np.eye(len(order), dtype=bool)
        out["checks"]["frobenius_vs_exact_gram_sweep"] = {
            "note": "the wide-module Frobenius cosine tr(M_i^T M_j) reproduced "
                    "here against "
                    "results/gram_sweep.npz, the project's exact Gram over the "
                    "134. Not identical: this one averages the 200 wide modules "
                    "only and gram_sweep uses all 248.",
            "n_traits": len(order),
            "pearson_offdiag": float(np.corrcoef(sub[o], gsub[o])[0, 1])}
    except Exception as e:                                   # pragma: no cover
        out["checks"]["frobenius_vs_exact_gram_sweep"] = {"error": repr(e)}

    hf, cf = idx["sorh_hack_final"], idx["sorh_control_final"]
    out["checks"]["frobenius_against_the_published_projection"] = {
        "note": "analysis/sorh_projection.json was built a different way (per-"
                "module sketches projected onto named axes). Its hack-versus-"
                "control contrast is an independent check on the Frobenius path "
                "here. Not expected to agree to the last digit: these numbers "
                "are over the 200 wide modules only, the published ones over "
                "all 248.",
        "cosine_here_wide_modules": float(FC[hf, cf]),
        "cosine_published": 0.23284390902307925,
        "cosine_published_key": "analysis/sorh_projection.json#contrast.cosine",
        "magnitude_ratio_here_wide_modules": float(nrm[hf] / nrm[cf]),
        "magnitude_ratio_published": 1.0800163251409132,
        "magnitude_ratio_published_key":
            "analysis/sorh_projection.json#contrast.magnitude_ratio"}

    recs = json.loads(str(z["mod_records"]))
    wide_recs = [r for r in recs if r["wide"]]
    out["checks"]["diff_delta_truncation"] = {
        "note": "the hack-minus-control delta is a rank-128 object truncated to "
                "the module's rank (64 on the wide modules). Because A_hack and "
                "A_control are the same zoo A up to training drift, singular "
                "values 64..127 should be negligible. Ratios to sigma_0, mean "
                "over the wide modules.",
        "mean_sv63_over_sv0": float(np.mean(
            [r["diff_sv_ratio_last_over_first"] for r in wide_recs])),
        "mean_sv64_over_sv0": float(np.mean(
            [r["diff_sv_ratio_k64_over_first"] for r in wide_recs])),
        "max_sv64_over_sv0": float(np.max(
            [r["diff_sv_ratio_k64_over_first"] for r in wide_recs])),
        "mean_energy_fraction_in_the_first_64_diff_final": float(np.mean(
            [r["diff_energy_fraction_in_first_m"] for r in wide_recs])),
        "mean_energy_fraction_in_the_first_64_diff_c31": float(np.mean(
            [r["diff_energy_fraction_in_first_m_c31"] for r in wide_recs])),
        "verdict": "there is no CLIFF at 64 -- sigma_64 is about a tenth of "
                   "sigma_0, the same size as sigma_63, because the two arms' A "
                   "matrices drifted apart in training and the difference delta "
                   "is genuinely rank 128. But the tail is thin: the leading 64 "
                   "directions hold 99.6 percent of the energy, so the k = 64 "
                   "truncation used for every diff_* statistic below loses "
                   "about 0.4 percent of the difference delta."}
    out["checks"]["reference_subspace_rank"] = {
        "note": "the mean stage-one delta is rank ~64 because every adapter "
                "shares A; a cliff between sigma_63 and sigma_64 confirms the "
                "randomised SVD found the whole thing.",
        **{f"mean_{k}": float(np.mean([r[k] for r in wide_recs if k in r]))
           for k in ("G1_mean_sv63_over_sv0", "G1_mean_sv64_over_sv0",
                     "G2_mean_sv63_over_sv0", "G1_stack_sv63_over_sv0",
                     "G2_stack_sv63_over_sv0")}}

    nr = json.loads(str(z["null_records"]))
    d_out, rank = z["d_outs"], z["ranks"]
    out["null"] = {}
    for k in KS:
        w = rank >= 64
        row = {"analytic_col_k_over_dout_wide_modules":
               float((k / d_out[w]).mean()) if w.any() else None}
        emp = [r[f"null_mean_k{k}"] for r in nr if f"null_mean_k{k}" in r]
        if emp:
            row["empirical_mean_over_stride_subset"] = float(np.mean(emp))
            row["empirical_n_modules"] = len(emp)
            row["empirical_n_draws_per_module"] = int(nr[0]["n_draws"])
            row["analytic_on_the_same_modules"] = float(np.mean(
                [r[f"analytic_col_k{k}"] for r in nr if f"null_mean_k{k}" in r]))
        out["null"][f"k{k}"] = row

    # ---------------- reference bands from this same matrix -----------------
    out["reference_bands_measured_here"] = {
        "note": "computed from the same 150 x 150 matrix so they are directly "
                "comparable with the probe rows. The zoo class is every "
                "off-diagonal pair among the 134 stage-one seed-0 adapters, "
                "which all share LoRA-A, so it is the SAME-seed different-trait "
                "class.",
        "zoo_diff_trait_same_seed": {
            f"k{k}": {"col_wtd": stat(C[k][zz][offz]),
                      "col_unw": stat(Cu[k][zz][offz])} for k in HEAD_KS},
        "align_vs_zoo": {
            f"k{k}": {"col_wtd": stat(C[k][np.ix_(align, zoo)])}
            for k in HEAD_KS},
        "align_vs_align": {
            f"k{k}": {"col_wtd": stat(C[k][np.ix_(align, align)][
                ~np.eye(len(align), dtype=bool)])} for k in HEAD_KS},
    }

    # ---------------- 1. hack versus control --------------------------------
    def pair(a, b):
        o = {}
        for k in HEAD_KS:
            o[f"k{k}"] = {
                "col_wtd_a_in_b": float(C[k][idx[a], idx[b]]),
                "col_wtd_b_in_a": float(C[k][idx[b], idx[a]]),
                "col_wtd_mean": float(0.5 * (C[k][idx[a], idx[b]]
                                             + C[k][idx[b], idx[a]])),
                "col_unw": float(Cu[k][idx[a], idx[b]])}
        o["top1_abs_cos"] = float(z["top1abs"][idx[a], idx[b]])
        o["top1_signed_cos"] = float(z["top1sgn"][idx[a], idx[b]])
        o["frobenius_cosine"] = float(FC[idx[a], idx[b]])
        return o

    out["hack_vs_control"] = {
        "matched_checkpoints": {c: pair(f"sorh_hack_{c}", f"sorh_control_{c}")
                                for c in ARMS},
        "within_run_reference_hack": {
            f"{a}_vs_{b}": pair(f"sorh_hack_{a}", f"sorh_hack_{b}")
            for a, b in [("c31", "c62"), ("c31", "c93"), ("c62", "c93")]},
        "within_run_reference_control": {
            f"{a}_vs_{b}": pair(f"sorh_control_{a}", f"sorh_control_{b}")
            for a, b in [("c31", "c62"), ("c31", "c93"), ("c62", "c93")]},
        "cross_arm_unmatched_checkpoints": {
            f"hack_{a}_vs_control_{b}": pair(f"sorh_hack_{a}",
                                             f"sorh_control_{b}")
            for a, b in [("c31", "c93"), ("c93", "c31")]},
        "limit": "there is no second SFT seed for either arm, so the only "
                 "within-run reference available is a run against its own "
                 "earlier checkpoint, which shares all of its training history "
                 "and is an UPPER reference, not a matched-seed replicate. The "
                 "lower reference is the zoo's same-seed different-trait class. "
                 "Neither is 'two SFT runs on the same data with different "
                 "seeds', which was never trained.",
    }

    # ---------------- 2. probes against the 134 -----------------------------
    zoo_names = names[zoo]
    prim = {}
    try:
        def norm(x):
            return "".join(c if c.isalnum() else "_" for c in str(x).strip().lower())
        prim = {norm(d["trait"]): d for d in json.load(
            open(os.path.join(HERE, "traits_primary.json")))}
    except Exception:
        pass

    def label(n):
        d = prim.get(n)
        return f"{n} ({d['factor']}{d['keyed']})" if d else n

    band = {f"k{k}": float(C[k][zz][offz].mean()) for k in HEAD_KS}
    vs_zoo = {}
    for p in probes:
        i = idx[p]
        e = {"band_zoo_diff_trait_same_seed_col_wtd": band}
        for k in HEAD_KS:
            v = C[k][i, zoo]
            vb = C[k][zoo, i]
            e[f"k{k}"] = {
                "col_wtd_probe_energy_in_trait": stat(v),
                "col_wtd_trait_energy_in_probe": stat(vb),
                "col_unw": stat(Cu[k][i, zoo]),
                "ratio_of_mean_to_zoo_band": float(v.mean() / band[f"k{k}"]),
                "top5_nearest_traits": [
                    {"trait": label(str(zoo_names[j])), "col_wtd": float(v[j])}
                    for j in np.argsort(-v)[:5]],
                "bottom3_traits": [
                    {"trait": label(str(zoo_names[j])), "col_wtd": float(v[j])}
                    for j in np.argsort(v)[:3]]}
        e["frobenius_cosine_vs_zoo"] = stat(FC[i, zoo])
        e["frobenius_top5_by_abs"] = [
            {"trait": label(str(zoo_names[j])), "cos": float(FC[i, zoo][j])}
            for j in np.argsort(-np.abs(FC[i, zoo]))[:5]]
        e["top1_abs_cos_vs_zoo"] = stat(z["top1abs"][i, zoo])
        vs_zoo[p] = e
    out["vs_134_stage_one_adapters"] = vs_zoo

    out["vs_alignment_adapters"] = {
        p: {f"k{k}": {str(names[j]): {
                "col_wtd_probe_energy_in_adapter": float(C[k][idx[p], j]),
                "col_wtd_adapter_energy_in_probe": float(C[k][j, idx[p]]),
                "col_unw": float(Cu[k][idx[p], j])} for j in align}
            for k in HEAD_KS}
        for p in probes}
    out["vs_alignment_adapters"]["_note"] = (
        "the four alignment adapters trained on the common pool "
        "(pc-qwen35-adapters:/data_alignment_common). Compare with the same "
        "probe's mean over the 134 in vs_134_stage_one_adapters and with "
        "reference_bands_measured_here.align_vs_zoo.")

    # ---------------- 3. generic subspace and the stage-two register --------
    gen = {}
    for rname in REFS:
        e = {}
        for k in HEAD_KS:
            wt = z[f"ref_{rname}_wtd_k{k}"]
            un = z[f"ref_{rname}_unw_k{k}"]
            wt1 = z[f"ref_{rname}_wtd_seed1_k{k}"]
            wt2 = z[f"ref_{rname}_wtd_stage2_k{k}"]
            e[f"k{k}"] = {
                "probes": {p: {"col_wtd": float(wt[idx[p]]),
                               "col_unw": float(un[idx[p]])} for p in probes},
                "band_134_stage_one": stat(wt[zoo]),
                "band_40_seed1_stage_one": stat(wt1),
                "band_134_stage_two": stat(wt2),
                "band_4_alignment": stat(wt[align]),
                "analytic_random_null_k_over_dout":
                    out["null"][f"k{k}"]["analytic_col_k_over_dout_wide_modules"]}
        gen[rname] = e
    out["vs_generic_and_register"] = gen
    out["vs_generic_and_register"]["_note"] = (
        "G1_* is the zoo's generic output subspace, G2_* the stage-two "
        "register. WHICH BAND IS IN-SAMPLE DEPENDS ON THE REFERENCE: for G1_* "
        "the 134 stage-one adapters define the subspace, so band_134_stage_one "
        "is in-sample and band_40_seed1_stage_one is the fair out-of-sample "
        "'what a trait adapter scores' (column space is seed-invariant, so a "
        "seed-1 adapter is a legitimate held-out trait adapter). For G2_* it is "
        "band_134_stage_two that is in-sample and the two stage-one bands that "
        "are held out. The mean version cancels opposite-keyed traits and the "
        "stack version does not; both are given.")

    # ---------------- 4. trajectory ----------------------------------------
    traj = {}
    for c in ["c31", "c62", "c93"]:
        row = {
            "hack_vs_control_col_wtd_k8":
                float(0.5 * (C[8][idx[f"sorh_hack_{c}"], idx[f"sorh_control_{c}"]]
                             + C[8][idx[f"sorh_control_{c}"], idx[f"sorh_hack_{c}"]])),
            "hack_vs_control_frobenius_cosine":
                float(FC[idx[f"sorh_hack_{c}"], idx[f"sorh_control_{c}"]]),
        }
        for arm in ("hack", "control", "diff"):
            p = f"sorh_{arm}_{c}" if arm != "diff" else f"diff_{c}"
            row[f"{arm}_vs_zoo_mean_col_wtd_k8"] = float(C[8][idx[p], zoo].mean())
            row[f"{arm}_vs_zoo_max_col_wtd_k8"] = float(C[8][idx[p], zoo].max())
            for rname in REFS:
                row[f"{arm}_vs_{rname}_wtd_k8"] = float(
                    z[f"ref_{rname}_wtd_k8"][idx[p]])
                row[f"{arm}_vs_{rname}_wtd_k64"] = float(
                    z[f"ref_{rname}_wtd_k64"][idx[p]])
            row[f"{arm}_frobenius_norm_wide"] = float(nrm[idx[p]])
        traj[c] = row
    out["trajectory"] = traj
    out["trajectory"]["_note"] = (
        "checkpoint-31 / 62 / 93 are the ends of epochs 1, 2 and 3; "
        "checkpoint-93 is byte-identical to `final`. "
        "`*_frobenius_norm_wide` is sqrt(sum over the wide modules of "
        "||dW||_F^2) and says how far the update has travelled, in the same "
        "units for every row.")

    # ---------------- 5. per-module profile --------------------------------
    prof = json.loads(str(z["mod_overlaps"])) if "mod_overlaps" in z else []
    for r in prof:
        r["proj"] = r["module"].split(".")[-1]
        r["layer"] = int(r["module"].split(".")[2])
        r["hack_gap_k8"] = r["hack_vs_zoo_wtd_k8"] - r["zoo_diff_trait_wtd_k8"]
    pp = {"statistic": "per-module, wide modules only: col_wtd at k = 8 of "
                       "sorh_hack_c93 against the mean over the 134, of the "
                       "134 against each other (the module's own baseline), "
                       "hack against control, and each against the zoo's "
                       "generic subspace G1_stack.",
          "n_modules": len(prof)}
    if prof:
        by_proj, by_layer = {}, {}
        for r in prof:
            by_proj.setdefault(r["proj"], []).append(r)
        keys = ["hack_vs_zoo_wtd_k8", "control_vs_zoo_wtd_k8",
                "diff_vs_zoo_wtd_k8", "zoo_diff_trait_wtd_k8",
                "hack_vs_control_wtd_k8", "hack_vs_G1_stack_wtd_k8",
                "diff_vs_G1_stack_wtd_k8", "zoo_vs_G1_stack_wtd_k8"]
        pp["by_projection"] = {
            p: {"n_modules": len(v), "d_out": v[0]["d_out"],
                **{k: float(np.mean([x[k] for x in v if k in x])) for k in keys
                   if any(k in x for x in v)}}
            for p, v in sorted(by_proj.items())}
        nl = max(r["layer"] for r in prof) + 1
        for q in range(4):
            lo, hi_ = q * nl // 4, (q + 1) * nl // 4
            v = [r for r in prof if lo <= r["layer"] < hi_]
            if not v:
                continue
            by_layer[f"layers_{lo}_{hi_-1}"] = {
                "n_modules": len(v),
                **{k: float(np.mean([x[k] for x in v if k in x])) for k in keys
                   if any(k in x for x in v)}}
        pp["by_layer_quartile"] = by_layer
        pp["top10_modules_by_hack_vs_zoo_k8"] = [
            {"module": r["module"], "hack_vs_zoo": r["hack_vs_zoo_wtd_k8"],
             "zoo_baseline": r["zoo_diff_trait_wtd_k8"],
             "ratio": r["hack_vs_zoo_wtd_k8"] / r["zoo_diff_trait_wtd_k8"]}
            for r in sorted(prof, key=lambda r: -r["hack_vs_zoo_wtd_k8"])[:10]]
        pp["top10_modules_by_ratio_to_zoo_baseline"] = [
            {"module": r["module"], "hack_vs_zoo": r["hack_vs_zoo_wtd_k8"],
             "zoo_baseline": r["zoo_diff_trait_wtd_k8"],
             "ratio": r["hack_vs_zoo_wtd_k8"] / r["zoo_diff_trait_wtd_k8"]}
            for r in sorted(prof, key=lambda r: -(r["hack_vs_zoo_wtd_k8"]
                                                  / r["zoo_diff_trait_wtd_k8"])
                            )[:10]]
        pp["top10_modules_by_hack_vs_control_k8"] = [
            {"module": r["module"], "hack_vs_control": r["hack_vs_control_wtd_k8"],
             "zoo_baseline": r["zoo_diff_trait_wtd_k8"]}
            for r in sorted(prof, key=lambda r: -r["hack_vs_control_wtd_k8"])[:10]]
    out["per_module_profile"] = pp

    # ---------------- narrow modules ---------------------------------------
    hi, ci, di = idx["sorh_hack_c93"], idx["sorh_control_c93"], idx["diff_c93"]
    out["narrow_modules"] = {
        "note": f"the {int(z['n_narrow'])} linear-attention in_proj_a / "
                "in_proj_b modules, d_out = 32. Rank is 32 there, so k = 64 is "
                "undefined and reads 0; the k = 8 random null is 8/32 = 0.25, "
                "so every number here is large and none of it is comparable "
                "with the wide-module headlines.",
        "hack_c93_vs_control_c93": {f"k{k}": float(Cn[k][hi, ci])
                                    for k in (1, 8)},
        "zoo_diff_trait_same_seed": {f"k{k}": stat(Cn[k][zz][offz])
                                     for k in (1, 8)},
        "hack_c93_vs_zoo": {f"k{k}": stat(Cn[k][hi, zoo]) for k in (1, 8)},
        "diff_c93_vs_zoo": {f"k{k}": stat(Cn[k][di, zoo]) for k in (1, 8)},
        "analytic_null_k8": 8 / 32}

    # ---------------- spectra ----------------------------------------------
    sp = z["spec_prof"]
    out["spectra"] = {
        "note": "share of the sum of singular values held by the leading "
                "directions, mean over the wide modules",
        **{n: {"top1": float(sp[idx[n], 0]), "top8": float(sp[idx[n], :8].sum()),
               "top16": float(sp[idx[n], :16].sum())}
           for n in probes},
        "zoo_mean": {"top1": float(sp[zoo, 0].mean()),
                     "top8": float(sp[zoo, :8].sum(1).mean()),
                     "top16": float(sp[zoo, :16].sum(1).mean())},
        "published_zoo_stage1": {
            "top1": 0.10318456418060519, "top8": 0.3614515265945672,
            "top16": 0.5305926812823089,
            "key": "column_space.json#stage1.spectrum"}}

    p = os.path.join(HERE, "analysis", f"column_space_{TAG}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {p}")

    # console summary
    b8 = band["k8"]
    print(f"\nzoo different-trait same-seed band, col_wtd k=8: {b8:.4f} "
          f"(published 0.13295928198239396)")
    print(f"published same-trait cross-seed: 0.5846385056208819")
    for c in ARMS:
        h, cc = idx[f"sorh_hack_{c}"], idx[f"sorh_control_{c}"]
        print(f"  hack {c} vs control {c}: col_wtd k8 "
              f"{0.5*(C[8][h,cc]+C[8][cc,h]):.4f}  frob cos {FC[h,cc]:+.4f}")
    print("\ncontrol within-run (upper reference):")
    for a, b in [("c31", "c62"), ("c31", "c93"), ("c62", "c93")]:
        x, y = idx[f"sorh_control_{a}"], idx[f"sorh_control_{b}"]
        print(f"  {a} vs {b}: col_wtd k8 {0.5*(C[8][x,y]+C[8][y,x]):.4f}")
    print("\nvs the 134, col_wtd k=8 (probe energy in trait):")
    for p_ in probes:
        v = C[8][idx[p_], zoo]
        print(f"  {p_:<20} mean {v.mean():.4f}  max {v.max():.4f} "
              f"({zoo_names[int(np.argmax(v))]})")
    print("\nvs the generic subspace, col_wtd k=8:")
    for rname in REFS:
        w = z[f"ref_{rname}_wtd_k8"]
        w1 = z[f"ref_{rname}_wtd_seed1_k8"]
        print(f"  {rname}: hack_c93 {w[hi]:.4f} control_c93 {w[ci]:.4f} "
              f"diff_c93 {w[di]:.4f} | 134 in-sample {w[zoo].mean():.4f} "
              f"| 40 seed-1 {w1.mean():.4f}")


if __name__ == "__main__":
    main()
