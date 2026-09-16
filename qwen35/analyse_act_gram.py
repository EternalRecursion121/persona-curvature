#!/usr/bin/env python3
"""Turn the activation-weighted Gram run into npz artefacts and one JSON.

Reads phase10_runs/actgram_results.json (written by act_gram_on_modal.py) and

 1. writes results/gram_actweighted{,_resp}.npz in the G/names/norms/scale/
    n_modules shape analyse_fa_qwen35.py reads through PC_GRAM_NPZ, and the
    matching 134 x 40 cross blocks in cross_gram_full_on_modal.py's
    X/names_a/names_b/norms_a/norms_b/scale/n_modules shape, so
    analyse_crossseed.py can compute (a), (b), (d) and (e) on the published
    code path;
 2. checks the C = I arm of the same code path against results/gram_sweep.npz
    and results/cross_gram_full_root_x_data_null_seedpaired_s40_matched.npz;
 3. assembles analysis/act_gram.json with the nulls, the participation ratios
    and the verdict arithmetic of PREREG_actgram.md.

usage:  python analyse_act_gram.py [phase10_runs/actgram_results.json]
"""
import json
import os
import sys

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(Q, "results")
A = os.path.join(Q, "analysis")
CROSS_REF = "cross_gram_full_root_x_data_null_seedpaired_s40_matched.npz"
PRIMARY = "pool445"


def cosmat(G):
    d = np.sqrt(np.diag(G))
    return G / np.outer(d, d)


def module_class(m):
    return m.split(".")[-1]


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "phase10_runs/actgram_results.json"
    r = json.load(open(os.path.join(Q, src)))
    names_a, names_b = r["names_a"], r["names_b"]
    NA, NB = len(names_a), len(names_b)
    s_a, s_b = r["scale_a"], r["scale_b"]
    nmod = r["n_modules"]
    out = {"what": "the 134 x 134 adapter Gram in the activation-weighted "
                   "(ASVD) metric, and the cross-seed block against the 40 "
                   "objective-matched seed-1 adapters",
           "source_run": src, "prereg": "qwen35/PREREG_actgram.md",
           "base_model": r["base_model"], "forward_dtype": r["forward_dtype"],
           "n_modules": nmod, "n_modules_total": r["n_modules_total"],
           "rank": r["rank"], "scale_a": s_a, "scale_b": s_b,
           "n_adapters_seed0": NA, "n_adapters_seed1": NB,
           "n_matched_pairs": len(r["pairs"]),
           "n_tokens": r["n_tokens"], "metrics": r["metrics"],
           "C_is_uncentred": r["C_is_uncentred"],
           "a_frame_spread": r["a_frame_spread"],
           "wall_seconds": r["wall_seconds"]}

    Gs = {k: np.array(r["grams"][k], dtype=np.float64) for k in r["metrics"]}

    # ---- 1. artefacts ---------------------------------------------------
    written = []
    for k in r["metrics"]:
        tagmap = {"frob": "_frobcheck", "pool445": "",
                  "pool445_centred": "_centred", "resp4378": "_resp",
                  "resp4378_centred": "_resp_centred"}
        t = tagmap.get(k, "_" + k)
        G = Gs[k][:NA, :NA]
        norms = np.sqrt(np.diag(G))
        p = f"{R}/gram_actweighted{t}.npz"
        np.savez(p, G=G, names=np.array(names_a), norms=norms,
                 scale=float(s_a), n_modules=int(nmod))
        written.append(os.path.relpath(p, Q))
        X = Gs[k][:NA, NA:]
        nb = np.sqrt(np.diag(Gs[k]))[NA:]
        p = (f"{R}/cross_gram_actweighted{t}_root_x_"
             f"data_null_seedpaired_s40_matched.npz")
        np.savez(p, X=X, names_a=np.array(names_a), names_b=np.array(names_b),
                 norms_a=norms, norms_b=nb, scale=float(s_a * s_b),
                 scale_a=float(s_a), scale_b=float(s_b), n_modules=int(nmod))
        written.append(os.path.relpath(p, Q))
    out["files_written"] = written

    # ---- 2. validation: the C = I arm ------------------------------------
    v = {}
    z = np.load(f"{R}/gram_sweep.npz", allow_pickle=True)
    ref_names = [str(x) for x in z["names"]]
    Cref = cosmat(np.asarray(z["G"], float))
    idx = [ref_names.index(n) for n in names_a]
    Cmine = cosmat(Gs["frob"][:NA, :NA])
    d = np.abs(Cmine - Cref[np.ix_(idx, idx)])
    v["vs_gram_sweep_134x134"] = {
        "max_abs_cosine_diff": float(d.max()),
        "mean_abs_cosine_diff": float(d.mean()),
        "norm_max_abs_rel_diff": float(np.max(np.abs(
            np.sqrt(np.diag(Gs["frob"][:NA, :NA])) / np.asarray(z["norms"], float)[idx] - 1)))}
    z2 = np.load(f"{R}/{CROSS_REF}", allow_pickle=True)
    na2 = [str(x) for x in z2["names_a"]]
    nb2 = [str(x) for x in z2["names_b"]]
    Xr = np.asarray(z2["X"], float) / np.outer(np.asarray(z2["norms_a"], float),
                                               np.asarray(z2["norms_b"], float))
    dg = np.sqrt(np.diag(Gs["frob"]))
    Xm = Gs["frob"][:NA, NA:] / np.outer(dg[:NA], dg[NA:])
    ii = [na2.index(n) for n in names_a]
    jj = [nb2.index(n) for n in names_b]
    d2 = np.abs(Xm - Xr[np.ix_(ii, jj)])
    v["vs_cross_gram_matched_134x40"] = {
        "reference": CROSS_REF,
        "max_abs_cosine_diff": float(d2.max()),
        "mean_abs_cosine_diff": float(d2.mean())}
    v["threshold_max_abs_cosine_diff"] = 1e-4
    v["passes"] = bool(d.max() < 1e-4 and d2.max() < 1e-4)
    out["validation"] = v

    # ---- 3. the statistics ----------------------------------------------
    twins = [t for t in names_b if t in names_a]
    out["arms"] = {}
    for k in r["metrics"]:
        dg = np.sqrt(np.diag(Gs[k]))
        X = Gs[k][:NA, NA:] / np.outer(dg[:NA], dg[NA:])
        same = np.array([X[names_a.index(t), names_b.index(t)] for t in twins])
        mask = np.ones_like(X, dtype=bool)
        for t in twins:
            mask[names_a.index(t), names_b.index(t)] = False
        offv = X[mask]
        ranks = [int(1 + np.sum(X[:, names_b.index(t)] > X[names_a.index(t),
                                                           names_b.index(t)]))
                 for t in twins]
        W = cosmat(Gs[k][:NA, :NA])
        xs, ys = [], []
        for t in twins:
            j = names_b.index(t)
            for u in names_a:
                if u != t:
                    xs.append(W[names_a.index(u), names_a.index(t)])
                    ys.append(X[names_a.index(u), j])
        xs, ys = np.array(xs), np.array(ys)
        slope, icept = np.polyfit(xs, ys, 1)
        arm = {
            "a_same_trait_cross_seed_mean": float(same.mean()),
            "a_same_trait_cross_seed_sd": float(same.std()),
            "a_same_trait_cross_seed_min": float(same.min()),
            "a_n": int(same.size),
            "b_diff_trait_cross_seed_mean": float(offv.mean()),
            "b_diff_trait_cross_seed_max": float(offv.max()),
            "b_n": int(offv.size),
            "separation_min_same_minus_max_off": float(same.min() - offv.max()),
            "d_top1_of_134": int(sum(1 for x in ranks if x == 1)),
            "d_mean_rank": float(np.mean(ranks)),
            "d_chance_rank": (NA + 1) / 2,
            "e_pearson_cross_vs_within": float(np.corrcoef(xs, ys)[0, 1]),
            "c4_attenuation_slope": float(slope),
            "c4_attenuation_intercept": float(icept),
            "n_regressed_pairs": int(xs.size),
            "c1_null": r["c1_null"][k],
            "c2_frame_overlap": r["c2_frame_overlap"][k],
            "within_seed0_offdiag_mean_cosine":
                float(W[~np.eye(NA, dtype=bool)].mean())}
        out["arms"][k] = arm

    # ---- how concentrated is the sum over modules? -----------------------
    if "module_energy_share" in r:
        mods = r["module_order"]
        out["module_concentration"] = {
            "what": "the Gram sums raw inner products over the 248 modules, so a "
                    "module whose activations are large dominates it.  share[m] "
                    "is that module's contribution to sum_i <dW_i,dW_i>; "
                    "effective_n_modules is (sum s)^2 / sum s^2",
            "arms": {}}
        for k in r["metrics"]:
            sh = np.array(r["module_energy_share"][k], dtype=float)
            sm = np.array(r["module_same_trait_numerator"][k], dtype=float)
            f = sh / sh.sum()
            o = np.argsort(-f)
            out["module_concentration"]["arms"][k] = {
                "effective_n_modules": float(sh.sum() ** 2 / (sh ** 2).sum()),
                "top1_share": float(f[o[0]]), "top1_module": mods[int(o[0])],
                "top10_share": float(f[o[:10]].sum()),
                "top10_modules": [mods[int(i)] for i in o[:10]],
                "by_module_class_share": {c: float(sum(
                    f[i] for i, m in enumerate(mods) if module_class(m) == c))
                    for c in sorted({module_class(m) for m in mods})},
                "same_trait_numerator_top10_share":
                    float(np.sort(sm)[::-1][:10].sum() / sm.sum())
                    if sm.sum() != 0 else None}

    # ---- participation ratio --------------------------------------------
    out["participation_ratio"] = {}
    for atag, per in r["spectra"].items():
        byc, byd, allv = {}, {}, []
        for m, rec in per.items():
            byc.setdefault(module_class(m), []).append(rec["participation_ratio"])
            byd.setdefault(str(rec["d_in"]), []).append(rec["participation_ratio"])
            allv.append(rec["participation_ratio"])
        rr = r["rank"]
        out["participation_ratio"][atag] = {
            "by_module_class": {c: {"n": len(v_), "mean_PR": float(np.mean(v_)),
                                    "min_PR": float(np.min(v_)),
                                    "max_PR": float(np.max(v_)),
                                    "mean_r_over_PR": float(np.mean(rr / np.array(v_)))}
                                for c, v_ in sorted(byc.items())},
            "by_d_in": {c: {"n": len(v_), "mean_PR": float(np.mean(v_)),
                            "mean_r_over_PR": float(np.mean(rr / np.array(v_)))}
                        for c, v_ in sorted(byd.items(), key=lambda x: int(x[0]))},
            "all_modules": {"n": len(allv), "mean_PR": float(np.mean(allv)),
                            "median_PR": float(np.median(allv)),
                            "c3_mean_r_over_PR": float(np.mean(rr / np.array(allv))),
                            "c3_r_over_mean_PR": float(rr / np.mean(allv))},
            "eig_detail_stride31": {m: rec for m, rec in per.items()
                                    if "eig_top64_share" in rec}}

    # ---- the verdict arithmetic -----------------------------------------
    aC = out["arms"][PRIMARY]["a_same_trait_cross_seed_mean"]
    aF = out["arms"]["frob"]["a_same_trait_cross_seed_mean"]
    nullC = out["arms"][PRIMARY]["c2_frame_overlap"]["seed0_vs_seed1"]["c2b_summed"]
    nullF = out["arms"]["frob"]["c2_frame_overlap"]["seed0_vs_seed1"]["c2b_summed"]
    ratio = aC / nullC
    if ratio >= 3.0 and aC > 0.05:
        verdict = "partly a metric artefact"
    elif ratio <= 1.5:
        verdict = "real in the functional metric"
    else:
        verdict = "in between"
    out["verdict"] = {
        "primary_arm": PRIMARY,
        "a_C": aC, "a_F": aF, "ratio_a_C_over_a_F": aC / aF,
        "null_c2b_C": nullC, "null_c2b_F": nullF,
        "ratio_a_C_over_null_C": ratio,
        "ratio_a_F_over_null_F": aF / nullF,
        "threshold_artefact": "a_C >= 3 * c2b_C AND a_C > 0.05",
        "threshold_real": "a_C <= 1.5 * c2b_C",
        "verdict": verdict}

    p = os.path.join(A, "act_gram.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"wrote {p}")
    print(f"  validation passes: {v['passes']}  "
          f"(max |dcos| vs gram_sweep {v['vs_gram_sweep_134x134']['max_abs_cosine_diff']:.3e}, "
          f"vs cross {v['vs_cross_gram_matched_134x40']['max_abs_cosine_diff']:.3e})")
    for k in r["metrics"]:
        a = out["arms"][k]
        print(f"  {k:10s} (a) {a['a_same_trait_cross_seed_mean']:+.6f}  "
              f"(b) {a['b_diff_trait_cross_seed_mean']:+.6f}  "
              f"(c1rms) {a['c1_null']['replace_seed1_frame']['rms']:.6f}  "
              f"(c2b) {a['c2_frame_overlap']['seed0_vs_seed1']['c2b_summed']:.6f}  "
              f"(c4slope) {a['c4_attenuation_slope']:.6f}  "
              f"top1 {a['d_top1_of_134']}/{len(twins)}  "
              f"(e) {a['e_pearson_cross_vs_within']:.6f}")
    print(f"  VERDICT: {verdict}  (a_C {aC:+.6f} / null_C {nullC:.6f} = {ratio:.3f}; "
          f"a_C/a_F = {aC/aF:.3f})")


if __name__ == "__main__":
    main()
