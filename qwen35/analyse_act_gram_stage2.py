#!/usr/bin/env python3
"""The stage boundary in the activation-weighted metric.

Reads phase10_runs/actgram_stage2_results.json (act_gram_stages_on_modal.py),
which carries one exact Gram over four adapter sets in every metric, and derives
everything in PREREG_actgram_stage2.md from it.  Removing stage two's shared
direction is exact at the Gram level:

    <u, v_k - M2> = <u, v_k> - mean_l <u, v_l>
    ||v_k - M2||^2 = G22_kk - 2 mean_l G22_kl + mean_{l,m} G22_lm

Writes results/cross_gram_actweighted_stage1_x_stage2.npz and
analysis/act_gram_stage2.json.

usage:  python analyse_act_gram_stage2.py [phase10_runs/actgram_stage2_results.json]
"""
import json
import os
import sys

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(Q, "results")
A = os.path.join(Q, "analysis")
PRIMARY = "pool445"
# the benchmarks this run's verdict is read against, both measured in the SAME
# metric by the stage-one run (analysis/act_gram.json)
BENCH = os.path.join(A, "act_gram.json")


def cosmat(G):
    d = np.sqrt(np.diag(G))
    return G / np.outer(d, d)


def centre(G):
    """Double-centre a Gram: the Gram of the adapters with their mean removed."""
    r = G.mean(1, keepdims=True)
    return G - r - r.T + G.mean()


def ranks_of(C, axis):
    """For each query along `axis`, the rank of its own trait (1 = best)."""
    n = C.shape[0]
    out = []
    for i in range(n):
        v = C[i, :] if axis == 0 else C[:, i]
        out.append(int(1 + np.sum(v > v[i])))
    return out


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "phase10_runs/actgram_stage2_results.json"
    r = json.load(open(os.path.join(Q, src)))
    names, owner = r["names"], r["owner"]
    sets = {s["tag"]: s for s in r["sets"]}
    idx = {t: {n: i for i, (n, o) in enumerate(zip(names, owner)) if o == t}
           for t in sets}
    # one trait order, shared by stage one and stage two
    order = [t for t in sorted(idx["s1s0"]) if t in idx["s2s0"]]
    n1 = len(order)

    out = {"what": "the stage-one x stage-two cross-Gram in the activation-weighted "
                   "(ASVD) metric, where the Frobenius one is zero by construction "
                   "because the two stages share no LoRA-A",
           "source_run": src, "prereg": "qwen35/PREREG_actgram_stage2.md",
           "base_model": r["base_model"], "forward_dtype": r["forward_dtype"],
           "n_modules": r["n_modules"], "rank": r["rank"],
           "n_tokens": r["n_tokens"], "metrics": r["metrics"],
           "sets": r["sets"], "n_traits_both_stages": n1,
           "frame_spread": r["frame_spread"],
           "wall_seconds": r["wall_seconds"],
           "caveat_base_model_C":
               "stage two was trained on merged(Qwen/Qwen3.5-4B + stage1 <trait>) "
               "and on self-generated introspective transcripts; C here is the "
               "BASE model's on the 445-prompt pool, so it is a first-order "
               "approximation for the stage-two side. A per-trait C would make "
               "the Gram trait-dependent, which the design cannot allow.",
           "arms": {}}

    bench = json.load(open(BENCH)) if os.path.exists(BENCH) else None
    iu = np.triu_indices(n1, 1)

    for k in r["metrics"]:
        Gall = np.array(r["grams"][k], dtype=np.float64)
        i1 = [idx["s1s0"][t] for t in order]
        i2 = [idx["s2s0"][t] for t in order]
        G1 = Gall[np.ix_(i1, i1)]
        G2 = Gall[np.ix_(i2, i2)]
        X12 = Gall[np.ix_(i1, i2)]
        d1, d2 = np.sqrt(np.diag(G1)), np.sqrt(np.diag(G2))
        C12 = X12 / np.outer(d1, d2)
        C1, C2 = cosmat(G1), cosmat(G2)

        # --- stage two's shared direction removed ------------------------
        X12r = X12 - X12.mean(1, keepdims=True)
        n2r2 = np.diag(G2) - 2 * G2.mean(1) + G2.mean()
        C12r = X12r / np.outer(d1, np.sqrt(np.maximum(n2r2, 1e-30)))
        # and with BOTH stages' shared directions removed, for symmetry
        X12b = X12r - X12r.mean(0, keepdims=True)
        n1r2 = np.diag(G1) - 2 * G1.mean(1) + G1.mean()
        C12b = X12b / np.outer(np.sqrt(np.maximum(n1r2, 1e-30)),
                               np.sqrt(np.maximum(n2r2, 1e-30)))

        ceil = r["c2_frame_overlap"][k]
        ck = ceil["s1s0_x_s2s0_same_trait"]["c2b_summed"]

        def blk(C):
            same = np.diag(C).copy()
            off = C[~np.eye(n1, dtype=bool)]
            rk_s2 = ranks_of(C, 1)     # each stage-two adapter among the 134 stage-one
            rk_s1 = ranks_of(C, 0)     # each stage-one adapter among the 134 stage-two
            return {"same_trait_mean": float(same.mean()),
                    "same_trait_sd": float(same.std()),
                    "same_trait_min": float(same.min()),
                    "same_trait_max": float(same.max()),
                    "diff_trait_mean": float(off.mean()),
                    "diff_trait_sd": float(off.std()),
                    "diff_trait_max": float(off.max()),
                    "same_over_diff": float(same.mean() / off.mean())
                                      if off.mean() != 0 else None,
                    "same_over_diff_note":
                        "removing stage two's grand mean centres every row of "
                        "the cross block exactly, so in the residual block the "
                        "different-trait mean is forced to about -same/(n-1) "
                        "and this ratio carries no information; use "
                        "same_over_diff_sd",
                    "same_over_diff_sd": float(same.mean() / off.std())
                                         if off.std() > 0 else None,
                    "ratio_to_ceiling": float(same.mean() / ck),
                    "top1_stage2_finds_own_stage1":
                        int(sum(1 for x in rk_s2 if x == 1)),
                    "mean_rank_stage2_query": float(np.mean(rk_s2)),
                    "top1_stage1_finds_own_stage2":
                        int(sum(1 for x in rk_s1 if x == 1)),
                    "mean_rank_stage1_query": float(np.mean(rk_s1)),
                    "chance_rank": (n1 + 1) / 2,
                    "corr_offdiag_with_stage1_within":
                        float(np.corrcoef(C[iu], C1[iu])[0, 1]),
                    "corr_offdiag_with_stage2_within":
                        float(np.corrcoef(C[iu], C2[iu])[0, 1])}

        arm = {"ceiling_c2b_stage1_x_stage2": ck,
               "ceiling_c2b_all_classes":
                   {c: v["c2b_summed"] for c, v in ceil.items()},
               "cross_stage_raw": blk(C12),
               "cross_stage_stage2_shared_removed": blk(C12r),
               "cross_stage_both_shared_removed": blk(C12b)}

        # --- grand means ---------------------------------------------------
        m12, m11, m22 = X12.mean(), G1.mean(), G2.mean()
        arm["cos_between_grand_means"] = float(m12 / np.sqrt(m11 * m22))
        arm["grand_mean_share_of_mean_norm2"] = {
            "stage1": float(m11 / np.mean(np.diag(G1))),
            "stage2": float(m22 / np.mean(np.diag(G2)))}
        arm["cos_to_own_grand_mean"] = {
            "stage1_mean": float(np.mean(G1.mean(1) / (d1 * np.sqrt(m11)))),
            "stage2_mean": float(np.mean(G2.mean(1) / (d2 * np.sqrt(m22))))}

        # --- arrangement, the centred-cosine statistic ----------------------
        Cc1 = cosmat(centre(G1) + 1e-12 * np.eye(n1))
        Cc2 = cosmat(centre(G2) + 1e-12 * np.eye(n1))
        arm["centred_cosines"] = {
            "stage1_sd": float(Cc1[iu].std()), "stage2_sd": float(Cc2[iu].std()),
            "corr_stage1_stage2": float(np.corrcoef(Cc1[iu], Cc2[iu])[0, 1])}
        arm["within_stage_offdiag_mean_cosine"] = {
            "stage1": float(C1[iu].mean()), "stage2": float(C2[iu].mean())}

        # --- the second seed, as a check ------------------------------------
        chk = {}
        for a, b, lab in (("s1s1", "s2s1", "cross_stage_seed1"),
                          ("s1s0", "s1s1", "stage1_cross_seed"),
                          ("s2s0", "s2s1", "stage2_cross_seed")):
            if a not in idx or b not in idx:
                continue
            ts = [t for t in sorted(idx[a]) if t in idx[b]]
            if not ts:
                continue
            ia = [idx[a][t] for t in ts]
            ib = [idx[b][t] for t in ts]
            Xa = Gall[np.ix_(ia, ib)]
            da = np.sqrt(np.diag(Gall))[ia]
            db = np.sqrt(np.diag(Gall))[ib]
            Ca = Xa / np.outer(da, db)
            same = np.diag(Ca)
            off = Ca[~np.eye(len(ts), dtype=bool)]
            cc = ceil.get(f"{a}_x_{b}_same_trait", {}).get("c2b_summed")
            chk[lab] = {"n_traits": len(ts), "traits": ts,
                        "same_trait_mean": float(same.mean()),
                        "same_trait_sd": float(same.std()),
                        "diff_trait_mean": float(off.mean()) if off.size else None,
                        "ceiling_c2b": cc,
                        "ratio_to_ceiling": float(same.mean() / cc) if cc else None}
        arm["second_seed_checks"] = chk
        out["arms"][k] = arm

        if k == PRIMARY:
            np.savez(os.path.join(R, "cross_gram_actweighted_stage1_x_stage2.npz"),
                     X=X12, names_a=np.array(order), names_b=np.array(order),
                     norms_a=d1, norms_b=d2, scale=4.0, scale_a=2.0, scale_b=2.0,
                     n_modules=int(r["n_modules"]),
                     X_stage2_shared_removed=X12r,
                     norms_b_shared_removed=np.sqrt(np.maximum(n2r2, 1e-30)),
                     G_stage1=G1, G_stage2=G2,
                     ceiling_c2b=float(ck), metric=PRIMARY)

    # ---- reproduction checks on the Frobenius arm -------------------------
    v = {}
    ref = os.path.join(A, "stage2_structure.json")
    if os.path.exists(ref) and "frob" in out["arms"]:
        x = json.load(open(ref))["stage1_x_stage2_exact"]
        m = out["arms"]["frob"]["cross_stage_raw"]
        v["vs_stage2_structure_stage1_x_stage2_exact"] = {
            "reference": "analysis/stage2_structure.json#stage1_x_stage2_exact "
                         "(derived as (X[stage1, persona] - G1) / 0.25)",
            "same_trait_mean": [m["same_trait_mean"], x["same_trait_cos_mean"],
                                abs(m["same_trait_mean"] - x["same_trait_cos_mean"])],
            "diff_trait_mean": [m["diff_trait_mean"], x["cross_trait_cos_mean"],
                                abs(m["diff_trait_mean"] - x["cross_trait_cos_mean"])],
            # the reference's key is NAMED "top1_stage1_finds_own_stage2" but its
            # code ranks C12[:, j], i.e. each STAGE-TWO adapter among the 134
            # stage-one ones (analyse_stage2_structure.py line 116).  Compared
            # like with like, it matches this run exactly; the label is wrong,
            # and so is the sentence it produced on stage-two-structure.
            "reference_key_name_is_the_other_direction": True,
            "top1_stage2_query": [m["top1_stage2_finds_own_stage1"],
                                  x["top1_stage1_finds_own_stage2"]],
            "mean_rank_stage2_query": [m["mean_rank_stage2_query"], x["mean_rank"]],
            "top1_stage1_query_this_run_only": m["top1_stage1_finds_own_stage2"],
            "mean_rank_stage1_query_this_run_only": m["mean_rank_stage1_query"],
            "cos_between_grand_means": [out["arms"]["frob"]["cos_between_grand_means"],
                                        x["cos_between_grand_means"]],
            "corr_offdiag_with_stage1_within":
                [m["corr_offdiag_with_stage1_within"], x["corr_C12_offdiag_with_C1"]],
            "corr_offdiag_with_stage2_within":
                [m["corr_offdiag_with_stage2_within"], x["corr_C12_offdiag_with_C2"]],
            "centred_cosine_corr":
                [out["arms"]["frob"]["centred_cosines"]["corr_stage1_stage2"],
                 json.load(open(ref))["centred_cosines"]["corr_stage1_stage2"]],
            "threshold_abs": 1e-5}
        v["vs_stage2_structure_passes"] = bool(
            abs(m["same_trait_mean"] - x["same_trait_cos_mean"]) < 1e-5
            and abs(m["diff_trait_mean"] - x["cross_trait_cos_mean"]) < 1e-5
            and m["top1_stage2_finds_own_stage1"] == x["top1_stage1_finds_own_stage2"]
            and abs(m["mean_rank_stage2_query"] - x["mean_rank"]) < 1e-6)
    p2 = os.path.join(R, "cross_gram_full_loras_introspection_x_loras_introspection.npz")
    if os.path.exists(p2) and "frob" in out["arms"]:
        z = np.load(p2, allow_pickle=True)
        na = [str(t) for t in z["names_a"]]
        jj = [na.index(t) for t in order]
        Cr = (np.asarray(z["X"], float)
              / np.outer(np.asarray(z["norms_a"], float),
                         np.asarray(z["norms_b"], float)))[np.ix_(jj, jj)]
        Gf = np.array(r["grams"]["frob"], dtype=np.float64)
        i2 = [idx["s2s0"][t] for t in order]
        Cm = cosmat(Gf[np.ix_(i2, i2)])
        v["vs_stage2_within_gram"] = {
            "reference": "results/cross_gram_full_loras_introspection_x_"
                         "loras_introspection.npz",
            "max_abs_cosine_diff": float(np.abs(Cm - Cr).max()),
            "mean_abs_cosine_diff": float(np.abs(Cm - Cr).mean())}
    out["validation"] = v

    # ---- the verdict ------------------------------------------------------
    a = out["arms"][PRIMARY]
    Rres = a["cross_stage_stage2_shared_removed"]["ratio_to_ceiling"]
    sod = a["cross_stage_stage2_shared_removed"]["same_over_diff_sd"]
    top1 = a["cross_stage_stage2_shared_removed"]["top1_stage2_finds_own_stage1"]
    arr = a["cross_stage_stage2_shared_removed"]["corr_offdiag_with_stage1_within"]
    if Rres >= 0.50 and sod >= 3.0:
        verdict = "the stage-two residual is the same function as stage one"
    elif Rres <= 0.15 and top1 > n1 / 10 and arr > 0.20:
        verdict = "a different function with a matching arrangement"
    else:
        verdict = "partial"
    out["verdict"] = {
        "primary_arm": PRIMARY,
        "R_raw": a["cross_stage_raw"]["ratio_to_ceiling"],
        "R_res": Rres,
        "same_over_diff_sd_residual": sod,
        "same_over_diff_residual_not_used":
            a["cross_stage_stage2_shared_removed"]["same_over_diff"],
        "top1_residual": top1, "n_traits": n1,
        "arrangement_corr_residual": arr,
        "benchmark_same_function_stage1_cross_seed":
            (bench["verdict"]["ratio_a_C_over_null_C"] if bench else None),
        "benchmark_unrelated_stage1_cross_seed_diff_trait":
            (bench["arms"][PRIMARY]["b_diff_trait_cross_seed_mean"]
             / bench["arms"][PRIMARY]["c2_frame_overlap"]["seed0_vs_seed1"]["c2b_summed"]
             if bench else None),
        "threshold_same_function":
            "R_res >= 0.50 and same_trait_mean / sd(different-trait) >= 3 "
            "(amended 2026-09-11 before the full run: the raw same/diff ratio is "
            "degenerate in a row-centred block)",
        "threshold_different_function":
            "R_res <= 0.15 with identification above chance and arrangement > 0.2",
        "verdict": verdict}

    p = os.path.join(A, "act_gram_stage2.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"wrote {p}")
    for k in r["metrics"]:
        a = out["arms"][k]
        raw, res = a["cross_stage_raw"], a["cross_stage_stage2_shared_removed"]
        print(f"  {k:16s} ceiling {a['ceiling_c2b_stage1_x_stage2']:.6f} | "
              f"raw same {raw['same_trait_mean']:+.6f} diff {raw['diff_trait_mean']:+.6f} "
              f"R {raw['ratio_to_ceiling']:.4f} top1 {raw['top1_stage2_finds_own_stage1']}/{n1} | "
              f"res same {res['same_trait_mean']:+.6f} diff {res['diff_trait_mean']:+.6f} "
              f"R {res['ratio_to_ceiling']:.4f} top1 {res['top1_stage2_finds_own_stage1']}/{n1} | "
              f"grandmeans {a['cos_between_grand_means']:+.4f}")
    print(f"  VERDICT: {out['verdict']['verdict']}  "
          f"(R_res {out['verdict']['R_res']:.4f}, same/diff "
          f"{out['verdict']['same_over_diff_sd_residual']:.2f} sd)")


if __name__ == "__main__":
    main()
