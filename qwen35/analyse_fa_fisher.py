#!/usr/bin/env python3
"""Does the factor solution survive the change of metric?

Experiment G1 of the 2026-09-09 paper reading.  analyse_fa_qwen35.py is run
three times over the same 134 adapters and the same code, differing only in the
Gram it is handed:

  results/fa_qwen35.json             results/gram_sweep.npz          Frobenius
  results/fa_qwen35_fisher.json      results/gram_fisher.npz         Fisher, expected
  results/fa_qwen35_fisher_emp.json  results/gram_fisher_empirical.npz  Fisher, empirical

This file compares the three, and asks four questions in order:

  1. HOW MANY FACTORS does each metric retain (Horn's parallel analysis, and the
     Kaiser counts alongside it)?
  2. WHERE DO THE FIVE LAND against the Goldberg Big Five targets, at k = 5, in
     each metric -- the congruences the post quotes?
  3. ARE THEY THE SAME FIVE FACTORS?  Tucker congruence between the two 134 x 5
     oblimin loading matrices, with the best one-to-one matching of columns and
     their signs.  This is the question: a factor solution can hit the same
     targets by accident, but two loading matrices that match column for column
     across 134 adapters are the same solution.
  4. WHAT DOES CURVATURE BUY?  The per-adapter Fisher norm F_ii against the
     Frobenius norm, against the communality, and against the loadings.

Writes analysis/fa_fisher_metric.json.
"""
import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
A = os.path.join(HERE, "analysis")
TCOLS = ["E", "A", "C", "ES", "I", "Eval"]
ARMS = [("frobenius", ""), ("fisher_expected", "_fisher"),
        ("fisher_empirical", "_fisher_emp")]
NPZ = {"frobenius": "gram_sweep.npz", "fisher_expected": "gram_fisher.npz",
       "fisher_empirical": "gram_fisher_empirical.npz"}


def tucker(X, Y):
    """Column-by-column Tucker congruence matrix of two loading matrices."""
    return (X.T @ Y) / np.outer(np.linalg.norm(X, axis=0), np.linalg.norm(Y, axis=0))


def best_match(C):
    """Best one-to-one matching of columns by |congruence|.

    analyse_stage2_structure.best_match brute-forces the permutations, which is
    fine at k = 5 and is 479 million at k = 11.  The Hungarian algorithm gives
    the same optimum in O(k^3), and it also handles a RECTANGULAR C, which is
    what comparing a nine-factor solution with a five-factor one needs.
    """
    r, c = linear_sum_assignment(-np.abs(C))
    return [{"column": int(i), "matched": int(j), "congruence": float(C[i, j])}
            for i, j in zip(r, c)]


def spearman(x, y):
    def rank(v):
        v = np.asarray(v, float)
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[o] = np.arange(len(v), dtype=float)
        s = v[o]
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            if j > i:
                r[o[i:j + 1]] = np.mean(np.arange(i, j + 1))
            i = j + 1
        return r
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def sol_of(o, k):
    """The centred solution at k, or None if this arm did not extract one."""
    return o["solutions"].get(f"centred_k{k}")


def main():
    out = {"what": "the same factor analysis (analyse_fa_qwen35.py, unchanged) "
                   "over the same 134 stage-one adapters in three metrics",
           "arms": {}}
    fa, gram = {}, {}
    for arm, tag in ARMS:
        path = f"{R}/fa_qwen35{tag}.json"
        if not os.path.exists(path):
            print(f"MISSING {path}, skipping arm {arm}")
            continue
        fa[arm] = json.load(open(path))
        d = np.load(f"{R}/{NPZ[arm]}", allow_pickle=True)
        gram[arm] = (np.asarray(d["G"], float), [str(x) for x in d["names"]])
    names = gram["frobenius"][1]
    p = len(names)
    out["n_traits"] = p
    out["trait_slug_order"] = names

    # ---- 1. how many factors -------------------------------------------
    for arm in fa:
        o = fa[arm]
        nf = o["n_factors"]
        pac = nf["parallel_analysis_centred"]
        out["arms"][arm] = {
            "gram_npz": NPZ[arm],
            "n_factors": {
                "chosen_parallel_analysis_centred_N1528": nf["chosen"],
                "kaiser_uncentred_eig_gt_1": nf["kaiser_uncentred_eig_gt_1"],
                "kaiser_centred_eig_gt_1": nf["kaiser_centred_eig_gt_1"],
                "reduced_eig_gt_1_uncentred": nf["reduced_eig_gt_1_uncentred"],
                "grid": {str(g["N"]): g["k_unreduced_95pct"] for g in pac["grid"]}},
            "solutions_present": sorted(o["solutions"]),
            "centred_eigenvalues_top10": o["correlation_matrix"]["centred_eigenvalues"][:10],
            "pca_centered_var_pct_top6": o["pca_from_gram"]["centered_var_pct"][:6],
            "uncentred_offdiag": o["correlation_matrix"]["uncentred_offdiag"],
            "centred_offdiag": o["correlation_matrix"]["centred_offdiag"]}

    # ---- 2. the k = 5 congruences with the Goldberg targets --------------
    for arm in fa:
        s = sol_of(fa[arm], 5)
        C = np.array(s["congruence_oblimin"])          # (5 factors, 6 targets)
        mt = best_match(np.abs(C[:, :5]))
        out["arms"][arm]["k5"] = {
            "ss_loadings_oblimin": s["ss_loadings"]["oblimin"],
            "communality_mean": float(np.mean(s["communalities"])),
            "n_heywood": s["n_heywood"],
            "congruence_oblimin": C.tolist(),
            "congruence_columns": TCOLS,
            "best_target_per_factor": [TCOLS[int(np.argmax(np.abs(C[j, :5])))]
                                       for j in range(C.shape[0])],
            "best_target_congruence": [float(C[j, int(np.argmax(np.abs(C[j, :5])))])
                                       for j in range(C.shape[0])],
            "one_to_one_matching": [
                {"factor": m["column"], "target": TCOLS[m["matched"]],
                 "abs_congruence": m["congruence"],
                 "signed_congruence": float(C[m["column"], m["matched"]])}
                for m in mt],
            "one_to_one_all_five_distinct": len({m["matched"] for m in mt}) == 5,
            "mean_abs_best_congruence": float(np.mean(
                [abs(C[j, :5]).max() for j in range(C.shape[0])]))}

    # ---- 3. are they the same five factors? ------------------------------
    pairs = [("frobenius", "fisher_expected"), ("frobenius", "fisher_empirical"),
             ("fisher_expected", "fisher_empirical")]
    lm = {}
    for arm in fa:
        s = sol_of(fa[arm], 5)
        lm[arm] = np.array(s["loadings"]["oblimin"])
    out["loading_matrix_congruence_k5"] = {
        "what": "Tucker congruence between the two 134 x 5 centred_k5 oblimin "
                "loading matrices, with the best one-to-one matching of columns; "
                "signs are free, so the reported congruence is signed and the "
                "matching is on |congruence|",
        "pairs": {}}
    for a, b in pairs:
        if a not in lm or b not in lm:
            continue
        C = tucker(lm[a], lm[b])
        mt = best_match(np.abs(C))
        out["loading_matrix_congruence_k5"]["pairs"][f"{a}_vs_{b}"] = {
            "congruence_matrix": C.tolist(),
            "matching": [{"factor_in_" + a: m["column"],
                          "factor_in_" + b: m["matched"],
                          "abs_congruence": m["congruence"],
                          "signed_congruence": float(C[m["column"], m["matched"]])}
                         for m in mt],
            "mean_abs_matched_congruence": float(np.mean([m["congruence"] for m in mt])),
            "min_abs_matched_congruence": float(min(m["congruence"] for m in mt)),
            "n_matched_above_0_90": int(sum(m["congruence"] >= 0.90 for m in mt)),
            "n_matched_above_0_95": int(sum(m["congruence"] >= 0.95 for m in mt)),
            "interpretation_thresholds":
                "the conventional reading of Tucker congruence: >= 0.95 is "
                "'identical', 0.85-0.94 is 'fair similarity', below 0.85 is not "
                "the same factor"}

    # At each arm's OWN retained k, including across different k: the matching is
    # rectangular, so the question is whether every factor of the smaller
    # solution finds a partner in the larger, not whether the counts agree.
    ks = {arm: fa[arm]["n_factors"]["chosen"] for arm in fa}
    extra = {}
    seen = set()
    for a, b in pairs:
        if a not in fa or b not in fa:
            continue
        for ka in sorted({ks[a], 5}):
            for kb in sorted({ks[b], 5}):
                sa, sb = sol_of(fa[a], ka), sol_of(fa[b], kb)
                key = f"{a}_k{ka}_vs_{b}_k{kb}"
                if sa is None or sb is None or key in seen:
                    continue
                seen.add(key)
                La = np.array(sa["loadings"]["oblimin"])
                Lb = np.array(sb["loadings"]["oblimin"])
                C = tucker(La, Lb)
                mt = best_match(np.abs(C))
                extra[key] = {
                    "k_a": ka, "k_b": kb, "n_matched": len(mt),
                    "mean_abs_matched_congruence": float(np.mean([m["congruence"] for m in mt])),
                    "min_abs_matched_congruence": float(min(m["congruence"] for m in mt)),
                    "n_matched_above_0_90": int(sum(m["congruence"] >= 0.90 for m in mt)),
                    "matching": [{"factor_a": m["column"], "factor_b": m["matched"],
                                  "abs_congruence": m["congruence"],
                                  "signed_congruence": float(C[m["column"], m["matched"]])}
                                 for m in mt]}
    out["loading_matrix_congruence_at_parallel_analysis_k"] = {
        "chosen_k_per_arm": ks, "pairs": extra}

    # ---- 4. the Grams themselves -----------------------------------------
    iu = np.triu_indices(p, 1)

    def cos(G):
        d = np.sqrt(np.diag(G))
        return (G / np.outer(d, d))[iu]
    cg = {arm: cos(gram[arm][0]) for arm in gram}
    out["gram_cosine_agreement"] = {
        "what": "Pearson and Spearman between the off-diagonal cosines of the "
                "Grams, over all 8,911 adapter pairs",
        "n_pairs": int(len(iu[0])),
        "pairs": {f"{a}_vs_{b}": {
            "pearson": float(np.corrcoef(cg[a], cg[b])[0, 1]),
            "spearman": spearman(cg[a], cg[b])}
            for a, b in pairs if a in cg and b in cg},
        "mean_cosine": {arm: float(cg[arm].mean()) for arm in cg},
        "sd_cosine": {arm: float(cg[arm].std()) for arm in cg}}

    # ---- 5. what curvature buys ------------------------------------------
    if "fisher_expected" in gram:
        Fii = np.diag(gram["fisher_expected"][0])
        fro = np.sqrt(np.diag(gram["frobenius"][0]))
        row = {"what": "F_ii, the Fisher norm of adapter i's unit direction, "
                       "against the Frobenius norm and against the factor solution",
               "F_ii": {"min": float(Fii.min()), "median": float(np.median(Fii)),
                        "max": float(Fii.max()),
                        "ratio_max_over_min": float(Fii.max() / Fii.min()),
                        "argmin": names[int(np.argmin(Fii))],
                        "argmax": names[int(np.argmax(Fii))]},
               "vs_frobenius_norm": {"pearson": float(np.corrcoef(Fii, fro)[0, 1]),
                                     "spearman": spearman(Fii, fro)}}
        for arm in fa:
            s = sol_of(fa[arm], 5)
            h2 = np.array(s["communalities"])
            L = np.array(s["loadings"]["oblimin"])
            row[f"vs_{arm}_k5"] = {
                "communality_pearson": float(np.corrcoef(Fii, h2)[0, 1]),
                "communality_spearman": spearman(Fii, h2),
                "max_abs_loading_spearman": spearman(Fii, np.abs(L).max(1)),
                "ss_loading_of_trait_spearman": spearman(Fii, (L ** 2).sum(1))}
        out["fisher_norm_vs_structure"] = row
        out["fisher_norm_per_trait"] = {
            names[i]: {"F_ii": float(Fii[i]), "frobenius_norm": float(fro[i])}
            for i in range(p)}

    json.dump(out, open(f"{A}/fa_fisher_metric.json", "w"), indent=1)
    print("wrote analysis/fa_fisher_metric.json")
    for arm in out["arms"]:
        a = out["arms"][arm]
        print(f"  {arm:18s} k(PA)={a['n_factors']['chosen_parallel_analysis_centred_N1528']:2d} "
              f"kaiser={a['n_factors']['kaiser_uncentred_eig_gt_1']:2d}  "
              f"k5 best congruences "
              + " ".join(f"{t}{c:+.3f}" for t, c in zip(a["k5"]["best_target_per_factor"],
                                                        a["k5"]["best_target_congruence"])))
    for k, v in out["loading_matrix_congruence_k5"]["pairs"].items():
        print(f"  loadings {k}: mean |congruence| "
              f"{v['mean_abs_matched_congruence']:.4f}, min {v['min_abs_matched_congruence']:.4f}")
    for k, v in out["gram_cosine_agreement"]["pairs"].items():
        print(f"  cosines  {k}: r={v['pearson']:.4f} rho={v['spearman']:.4f}")


if __name__ == "__main__":
    main()
