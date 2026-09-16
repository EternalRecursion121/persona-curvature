#!/usr/bin/env python3
"""What is the structure of the stage-2 (introspection SFT) adapter space?

Inputs: results/gram_sweep.npz (stage 1), results/gram_stage2.npz (stage 2, G format),
results/cross_gram_full_root_x_pc-qwen35-oct2_personas_exact.npz (stage 1 x persona; the
stage-1 x stage-2 cross-Gram follows exactly as (X_s1,per - G1) / 0.25),
results/fa_qwen35.json and results/fa_qwen35_stage2.json (analyse_fa_qwen35.py),
results/decomposition.json and results/decomposition_stage2.json (decompose.py).
Output: analysis/stage2_structure.json and wiki/pages/geometry/stage-two-structure.md.
"""
import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = f"{HERE}/results"
W = f"{os.path.dirname(HERE)}/wiki/pages/geometry/stage-two-structure.md"


def load_g(p):
    z = np.load(p, allow_pickle=True)
    return np.array(z["G"], dtype=float), [str(x) for x in z["names"]]


def align(G, names, order):
    idx = [names.index(n) for n in order]
    return G[np.ix_(idx, idx)]


def cosine(G):
    d = np.sqrt(np.diag(G)); return G / np.outer(d, d)


def centre(G):
    n = G.shape[0]; J = np.eye(n) - 1.0 / n
    return J @ G @ J


def spectrum(G):
    n = G.shape[0]
    wu = np.sort(np.linalg.eigvalsh(G))[::-1]
    wc = np.sort(np.linalg.eigvalsh(centre(G)))[::-1]
    wc = np.clip(wc, 0, None)
    pr = float(wc.sum() ** 2 / (wc ** 2).sum())
    return {"uncentred_share_top5": (wu[:5] / wu.sum()).tolist(),
            "centred_share_top10": (wc[:10] / wc.sum()).tolist(),
            "centred_participation_ratio": pr,
            "centred_n_for_50pct": int(np.searchsorted(np.cumsum(wc) / wc.sum(), 0.5) + 1),
            "centred_n_for_80pct": int(np.searchsorted(np.cumsum(wc) / wc.sum(), 0.8) + 1)}


def shared(G):
    n = G.shape[0]
    m2 = G.sum() / n ** 2                       # |mean dW|^2
    mean_diag = np.trace(G) / n                 # mean |dW_i|^2
    cos_to_mean = (G.sum(1) / n) / (np.sqrt(np.diag(G)) * np.sqrt(m2))
    return {"mean_direction_norm2_over_mean_norm2": float(m2 / mean_diag),
            "cos_to_mean_direction_mean": float(cos_to_mean.mean()),
            "cos_to_mean_direction_sd": float(cos_to_mean.std()),
            "cos_to_mean_direction_min": float(cos_to_mean.min()),
            "cos_to_mean_direction_max": float(cos_to_mean.max()),
            "cos_to_mean": cos_to_mean.tolist()}


def tucker(A, B):
    return (A.T @ B) / np.outer(np.linalg.norm(A, axis=0), np.linalg.norm(B, axis=0))


def best_match(C):
    """Best one-to-one matching of columns by |congruence|, with signs."""
    k = C.shape[0]
    best = None
    for perm in itertools.permutations(range(C.shape[1]), k):
        s = sum(abs(C[i, perm[i]]) for i in range(k))
        if best is None or s > best[0]:
            best = (s, perm)
    _, perm = best
    return [{"factor": i, "matched": int(perm[i]), "congruence": float(C[i, perm[i]])} for i in range(k)]


def main():
    out = {}
    G1, n1 = load_g(f"{R}/gram_sweep.npz")
    G2, n2 = load_g(f"{R}/gram_stage2.npz")
    order = sorted(set(n1) & set(n2)); assert len(order) == 134
    G1, G2 = align(G1, n1, order), align(G2, n2, order)
    C1, C2 = cosine(G1), cosine(G2)

    out["spectrum"] = {"stage1": spectrum(G1), "stage2": spectrum(G2)}
    s1, s2 = shared(G1), shared(G2)
    out["shared_component"] = {"stage1": {k: v for k, v in s1.items() if k != "cos_to_mean"},
                               "stage2": {k: v for k, v in s2.items() if k != "cos_to_mean"},
                               "corr_cos_to_mean_across_stages": float(np.corrcoef(s1["cos_to_mean"], s2["cos_to_mean"])[0, 1])}
    # residual after removing the mean direction: cosine structure of the centred adapters
    Cc1, Cc2 = cosine(centre(G1) + 1e-12 * np.eye(134)), cosine(centre(G2) + 1e-12 * np.eye(134))
    iu = np.triu_indices(134, 1)
    out["centred_cosines"] = {"stage1_sd": float(Cc1[iu].std()), "stage2_sd": float(Cc2[iu].std()),
                              "corr_stage1_stage2": float(np.corrcoef(Cc1[iu], Cc2[iu])[0, 1])}

    # exact stage-1 x stage-2 cross-Gram from the two-volume persona Gram
    p = f"{R}/cross_gram_full_root_x_pc-qwen35-oct2_personas_exact.npz"
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        Xsp = np.array(z["X"]); na = [str(x) for x in z["names_a"]]; nb = [str(x) for x in z["names_b"]]
        ia = [na.index(t) for t in order]; ib = [nb.index(t) for t in order]
        Xsp = Xsp[np.ix_(ia, ib)]
        X12 = (Xsp - G1) / 0.25
        C12 = X12 / np.outer(np.sqrt(np.diag(G1)), np.sqrt(np.diag(G2)))
        same = np.diag(C12)
        off = C12[~np.eye(134, dtype=bool)]
        # cosine between the two grand-mean directions
        m1m2 = X12.sum() / 134 ** 2
        cos_means = float(m1m2 / np.sqrt((G1.sum() / 134 ** 2) * (G2.sum() / 134 ** 2)))
        ranks = [int(1 + np.sum(C12[:, j] > C12[j, j])) for j in range(134)]
        out["stage1_x_stage2_exact"] = {
            "how": "X12 = (X[stage1, persona] - G1) / 0.25, exact because persona = s1 + 0.25 s2",
            "same_trait_cos_mean": float(same.mean()), "same_trait_cos_sd": float(same.std()),
            "same_trait_cos_min": float(same.min()), "same_trait_cos_max": float(same.max()),
            "cross_trait_cos_mean": float(off.mean()), "cross_trait_cos_sd": float(off.std()),
            "top1_stage1_finds_own_stage2": sum(1 for r in ranks if r == 1), "mean_rank": float(np.mean(ranks)),
            "cos_between_grand_means": cos_means,
            "corr_C12_offdiag_with_C1": float(np.corrcoef(C12[iu], C1[iu])[0, 1]),
            "corr_C12_offdiag_with_C2": float(np.corrcoef(C12[iu], C2[iu])[0, 1]),
        }

    # factor analysis outputs
    f1 = json.load(open(f"{R}/fa_qwen35.json")); f2 = json.load(open(f"{R}/fa_qwen35_stage2.json"))
    labels = f2["targets"]["labels"]
    fa = {}
    for tag, f in (("stage1", f1), ("stage2", f2)):
        s = f["solutions"]["centred_k5"]
        cong = np.array(s["congruence_oblimin"])           # factors x targets
        bf = [{"factor": i, "best_target": labels[int(np.argmax(np.abs(cong[i])))],
               "congruence": float(cong[i][int(np.argmax(np.abs(cong[i])))]),
               "all": {labels[j]: round(float(cong[i][j]), 3) for j in range(len(labels))}} for i in range(cong.shape[0])]
        # top loaders per factor from per_trait oblimin loadings
        pt = f["per_trait"]; names = list(pt.keys())
        Lm = np.array([pt[t]["oblimin_loadings_centred_k5"] for t in names])
        loaders = []
        for i in range(Lm.shape[1]):
            o = np.argsort(Lm[:, i])
            loaders.append({"factor": i,
                            "top_pos": [(names[j], round(float(Lm[j, i]), 3)) for j in o[::-1][:6]],
                            "top_neg": [(names[j], round(float(Lm[j, i]), 3)) for j in o[:6]]})
        fa[tag] = {"n_factors": f["n_factors"]["chosen"], "parallel_analysis_centred": f["n_factors"].get("parallel_analysis_centred", {}).get("n_retained") if isinstance(f["n_factors"].get("parallel_analysis_centred"), dict) else f["n_factors"].get("parallel_analysis_centred"),
                   "reduced_eigenvalues_k5": s["reduced_eigenvalues"][:5], "ss_loadings_oblimin": s["ss_loadings"]["oblimin"],
                   "communality_mean": float(np.mean(s["communalities"])) if isinstance(s["communalities"], list) else None,
                   "best_big_five_target_per_factor": bf, "top_loaders": loaders, "_L": Lm, "_names": names}
    # congruence between the two stages' oblimin loading matrices (same trait order)
    common = [t for t in fa["stage1"]["_names"] if t in fa["stage2"]["_names"]]
    L1 = np.array([fa["stage1"]["_L"][fa["stage1"]["_names"].index(t)] for t in common])
    L2 = np.array([fa["stage2"]["_L"][fa["stage2"]["_names"].index(t)] for t in common])
    Ct = tucker(L1, L2)
    out["factor_congruence_stage1_vs_stage2"] = {"tucker_matrix_rows_stage1_cols_stage2": np.round(Ct, 3).tolist(),
                                                 "best_matching": best_match(Ct)}
    for tag in fa:
        fa[tag].pop("_L"); fa[tag].pop("_names")
    out["factor_analysis"] = fa

    # decomposition tests
    dec = {}
    for tag, fn in (("stage1", "decomposition.json"), ("stage2", "decomposition_stage2.json")):
        d = json.load(open(f"{R}/{fn}"))
        dec[tag] = {"test1b_within_between_diff_p": [d["test1b"]["raw"]["within"], d["test1b"]["raw"]["between"], d["test1b"]["raw"]["diff"], d["test1b"]["raw"]["p"]],
                    "test2_same_opp_gap_p": [d["test2"]["raw"]["same_polarity"], d["test2"]["raw"]["opposite_polarity"], d["test2"]["raw"]["gap"], d["test2"]["raw"]["p"]],
                    "test3_ARI_NMI_p": [d["test3"]["raw"]["ARI"], d["test3"]["raw"]["NMI"], d["test3"]["raw"]["p"]],
                    "test4_per_factor_resid_diff": {k: v["resid_diff"] for k, v in d["test4"].items()},
                    "test1c_abs_corr_leading_polarity": d["test1c"]["abs_corr_leading_polarity"]}
    out["decomposition_tests"] = dec
    json.dump(out, open(f"{HERE}/analysis/stage2_structure.json", "w"), indent=1)

    # ---- printed summary
    sp = out["spectrum"]; sh = out["shared_component"]
    print(f"uncentred top eigen share: s1 {sp['stage1']['uncentred_share_top5'][0]:.3f}  s2 {sp['stage2']['uncentred_share_top5'][0]:.3f}")
    print(f"centred top-5 shares: s1 {[round(x,3) for x in sp['stage1']['centred_share_top10'][:5]]}  s2 {[round(x,3) for x in sp['stage2']['centred_share_top10'][:5]]}")
    print(f"participation ratio (centred): s1 {sp['stage1']['centred_participation_ratio']:.1f}  s2 {sp['stage2']['centred_participation_ratio']:.1f}; n for 50%: {sp['stage1']['centred_n_for_50pct']} / {sp['stage2']['centred_n_for_50pct']}")
    print(f"mean-direction share of norm^2: s1 {sh['stage1']['mean_direction_norm2_over_mean_norm2']:.3f}  s2 {sh['stage2']['mean_direction_norm2_over_mean_norm2']:.3f}; cos to mean: s1 {sh['stage1']['cos_to_mean_direction_mean']:.3f}+-{sh['stage1']['cos_to_mean_direction_sd']:.3f}  s2 {sh['stage2']['cos_to_mean_direction_mean']:.3f}+-{sh['stage2']['cos_to_mean_direction_sd']:.3f}; corr across stages {sh['corr_cos_to_mean_across_stages']:.3f}")
    print("centred cosines:", {k: round(v, 3) for k, v in out["centred_cosines"].items()})
    if "stage1_x_stage2_exact" in out:
        x = out["stage1_x_stage2_exact"]
        print(f"s1 x s2 exact: same {x['same_trait_cos_mean']:+.4f}+-{x['same_trait_cos_sd']:.4f} [{x['same_trait_cos_min']:+.3f},{x['same_trait_cos_max']:+.3f}]  cross {x['cross_trait_cos_mean']:+.4f}+-{x['cross_trait_cos_sd']:.4f}  top1 {x['top1_stage1_finds_own_stage2']}/134 mean rank {x['mean_rank']:.1f}  cos(grand means) {x['cos_between_grand_means']:+.3f}")
    for tag in ("stage1", "stage2"):
        f = fa[tag]
        print(f"FA {tag}: chosen {f['n_factors']}, reduced eig {[round(x,2) for x in f['reduced_eigenvalues_k5']]}, ss oblimin {[round(x,2) for x in f['ss_loadings_oblimin']]}")
        for b in f["best_big_five_target_per_factor"]:
            print(f"   factor {b['factor']}: best {b['best_target']} {b['congruence']:+.3f}   all {b['all']}")
        for l in f["top_loaders"]:
            print(f"   f{l['factor']} +: {[t for t,_ in l['top_pos'][:5]]}  -: {[t for t,_ in l['top_neg'][:5]]}")
    print("stage1 vs stage2 oblimin factor congruence (best matching):", out["factor_congruence_stage1_vs_stage2"]["best_matching"])
    for tag in ("stage1", "stage2"):
        print(f"decompose {tag}:", {k: (np.round(v, 4).tolist() if isinstance(v, list) else v) for k, v in dec[tag].items() if k != "test4_per_factor_resid_diff"})
        print("   test4:", {k: round(v, 4) for k, v in dec[tag]["test4_per_factor_resid_diff"].items()})
    print("wrote analysis/stage2_structure.json")


if __name__ == "__main__":
    main()
