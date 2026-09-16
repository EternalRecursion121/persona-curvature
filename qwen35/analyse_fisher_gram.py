#!/usr/bin/env python3
"""Turn the raw Fisher Gram into two npz files and check that it is right.

Reads   phase10_runs/fisher_gram_results.json   (fisher_gram.py)
        results/gram_sweep.npz                  (the exact Frobenius Gram)
        phase10_runs/fisher_spec.json           (the 124 directions' coefficients)
        analysis/fisher_norms.json              (fisher.py's independent F)
Writes  results/gram_fisher.npz            expected form, the FA input
        results/gram_fisher_empirical.npz  empirical form (realised tokens)
        analysis/fisher_gram_validation.json

THE VALIDATION THAT MATTERS
---------------------------
A Gram over unit adapter directions predicts the Fisher norm of ANY merge:

    F(c) = (c * n)^T  F_unit  (c * n)  /  (c^T G_exact c)

with n the adapters' Frobenius norms and G_exact the exact Frobenius Gram --
numerator and denominator are the same merge measured in the two metrics.  So
every one of the 124 directions fisher.py measured one at a time, on the same
text, is a prediction this Gram has to reproduce: five factors, six PCs, five
axes, the grand mean, ten single adapters, twenty random merges of all 134, and
seventy-two sphere points.  The random merges are the sharp test, because they
load the off-diagonal of the Gram in a way the singles cannot.

The EXPECTED form should reproduce them: for i = j it is by definition the
curvature of KL that fisher.py fitted.  The EMPIRICAL form should NOT, and by
how much it fails is one of experiment G1's results -- the empirical Fisher on
realised tokens is a different object from the Fisher, and the project has never
had a number for the gap.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
A = os.path.join(HERE, "analysis")
P = os.path.join(HERE, "phase10_runs")


def spearman(x, y):
    def rank(v):
        v = np.asarray(v, float)
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[o] = np.arange(len(v), dtype=float)
        # average ties
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
    a, b = rank(x), rank(y)
    return float(np.corrcoef(a, b)[0, 1])


def cosines(G):
    d = np.sqrt(np.diag(G))
    return G / np.outer(d, d)


def main():
    J = json.load(open(f"{P}/fisher_gram_results.json"))
    names = J["names"]
    p = len(names)
    Fx = np.array(J["F_expected"], dtype=np.float64)
    Fm = np.array(J["F_empirical"], dtype=np.float64)
    Fx = 0.5 * (Fx + Fx.T)
    Fm = 0.5 * (Fm + Fm.T)
    nrm_run = np.array(J["adapter_frobenius_norms"], dtype=np.float64)

    d = np.load(f"{R}/gram_sweep.npz", allow_pickle=True)
    Gx = np.asarray(d["G"], dtype=np.float64)
    gnames = [str(x) for x in d["names"]]
    gnorms = np.asarray(d["norms"], dtype=np.float64)
    assert gnames == list(names), "trait order differs from gram_sweep.npz"

    out = {"source": "phase10_runs/fisher_gram_results.json",
           "estimator_note": J["estimator_note"],
           "h": J["h"], "ref": J["ref"], "n_scored_tokens": J["n_scored_tokens"],
           "n_sequences": J["n_sequences"], "vocab_size": J["vocab_size"],
           "wall_seconds": J["wall_seconds"],
           "base_mean_nll_per_token": J["base_mean_nll_per_token"]}

    # ---- 1. the norms the run used are the published adapter norms ---------
    out["adapter_norm_check"] = {
        "what": "||dW_i||_F recomputed inside the run from the bf16 factors, "
                "against results/gram_sweep.npz norms",
        "max_abs_rel_diff": float(np.abs(nrm_run / gnorms - 1).max()),
        "argmax_trait": names[int(np.argmax(np.abs(nrm_run / gnorms - 1)))]}

    # ---- 2. numerical health ----------------------------------------------
    sd = np.sqrt(np.diag(Fx))
    out["numerics"] = {
        "zero_sum_check_max_abs": J["zero_sum_check_max_abs"],
        "zero_sum_check_meaning": J["zero_sum_check_meaning"],
        "score_scale_sqrt_diag_expected": {"min": float(sd.min()),
                                           "median": float(np.median(sd)),
                                           "max": float(sd.max())},
        "zero_sum_over_score_scale": float(J["zero_sum_check_max_abs"] / sd.min()),
        "min_eigenvalue_expected": float(np.linalg.eigvalsh(Fx)[0]),
        "min_eigenvalue_empirical": float(np.linalg.eigvalsh(Fm)[0]),
        "psd_note": "both forms are Gram matrices of real vectors and so are PSD "
                    "up to floating point; a negative eigenvalue of order 1e-12 "
                    "times the largest is rounding, anything larger is not."}

    # ---- 3. h convergence --------------------------------------------------
    half = J["diag_expected_at_half_h"]
    conv = {}
    for t, v in half.items():
        i = names.index(t)
        conv[t] = {"F_ii_at_h": float(Fx[i, i]), "F_ii_at_half_h": float(v),
                   "rel_change": float(v / Fx[i, i] - 1.0)}
    out["h_convergence"] = {
        "h": J["h"],
        "what": "F_ii recomputed at h/2 for the ten single adapters; the central "
                "difference has O(h^2) error, so halving h should change F_ii by "
                "about a quarter of its distance from the limit.",
        "max_abs_rel_change": max(abs(v["rel_change"]) for v in conv.values()),
        "per_trait": conv}

    # ---- 4. the diagonal against fisher.py --------------------------------
    FN = json.load(open(f"{A}/fisher_norms.json"))
    FD = FN["directions"]
    singles = {}
    for t in half:
        k = f"single_{t}"
        if k in FD:
            i = names.index(t)
            singles[t] = {
                "F_expected_ii": float(Fx[i, i]), "F_empirical_ii": float(Fm[i, i]),
                "fisher_py_F_ref": FD[k]["F_ref"],
                "fisher_py_F_ref_a0125": FD[k]["F_ref_a0125"],
                "expected_over_fisher_py": float(Fx[i, i] / FD[k]["F_ref"]),
                "empirical_over_fisher_py": float(Fm[i, i] / FD[k]["F_ref"])}
    out["single_adapter_diagonal"] = {
        "what": "F_ii of this Gram against fisher.py's independently measured F "
                "for the same ten single-adapter directions",
        "expected_over_fisher_py": {
            "min": min(v["expected_over_fisher_py"] for v in singles.values()),
            "median": float(np.median([v["expected_over_fisher_py"]
                                       for v in singles.values()])),
            "max": max(v["expected_over_fisher_py"] for v in singles.values())},
        "empirical_over_fisher_py": {
            "min": min(v["empirical_over_fisher_py"] for v in singles.values()),
            "median": float(np.median([v["empirical_over_fisher_py"]
                                       for v in singles.values()])),
            "max": max(v["empirical_over_fisher_py"] for v in singles.values())},
        "per_trait": singles}

    # ---- 5. every stage-one direction fisher.py measured ------------------
    SP = json.load(open(f"{P}/fisher_spec.json"))
    idx = {t: i for i, t in enumerate(names)}
    rows = []
    for dspec in SP["directions"]:
        nm = dspec["name"]
        if dspec["source"] != "stage1" or nm not in FD or "F_ref" not in FD[nm]:
            continue
        c = np.zeros(p)
        for t, v in dspec["coef"].items():
            c[idx[t]] = v
        den = float(c @ Gx @ c)
        if den <= 0:
            continue
        cn = c * gnorms
        rows.append({"name": nm, "family": dspec["family"],
                     "F_measured": FD[nm]["F_ref"],
                     "F_measured_a0125": FD[nm].get("F_ref_a0125"),
                     "F_pred_expected": float(cn @ Fx @ cn / den),
                     "F_pred_empirical": float(cn @ Fm @ cn / den),
                     "dir_norm_from_gram": float(np.sqrt(den)),
                     "dir_norm_from_fisher_py": FD[nm]["dir_norm_raw"]})
    mm = np.array([r["F_measured"] for r in rows])
    pe = np.array([r["F_pred_expected"] for r in rows])
    pm = np.array([r["F_pred_empirical"] for r in rows])
    nn = np.array([r["dir_norm_from_gram"] for r in rows])
    n2 = np.array([r["dir_norm_from_fisher_py"] for r in rows])
    fams = sorted({r["family"] for r in rows})
    out["prediction_of_fisher_py"] = {
        "what": "F(c) = (c*n)^T F_unit (c*n) / (c^T G_exact c) for every "
                "stage-one direction in phase10_runs/fisher_spec.json, against "
                "the F that fisher.py measured for it directly on the same text",
        "n": len(rows),
        "families": {f: int(sum(1 for r in rows if r["family"] == f)) for f in fams},
        "expected": {
            "median_ratio": float(np.median(pe / mm)),
            "max_abs_rel_err": float(np.abs(pe / mm - 1).max()),
            "rel_err_p90": float(np.percentile(np.abs(pe / mm - 1), 90)),
            "pearson": float(np.corrcoef(pe, mm)[0, 1]),
            "spearman": spearman(pe, mm),
            "worst": max(rows, key=lambda r: abs(r["F_pred_expected"] / r["F_measured"] - 1))["name"]},
        "empirical": {
            "median_ratio": float(np.median(pm / mm)),
            "min_ratio": float((pm / mm).min()), "max_ratio": float((pm / mm).max()),
            "pearson": float(np.corrcoef(pm, mm)[0, 1]),
            "spearman": spearman(pm, mm)},
        "direction_norm_check": {
            "what": "sqrt(c^T G_exact c) against fisher.py's dir_norm_raw, which "
                    "it built from the bf16 factor cache",
            "max_abs_rel_diff": float(np.abs(nn / n2 - 1).max())},
        "by_family": {f: {
            "n": int(sum(1 for r in rows if r["family"] == f)),
            "median_ratio_expected": float(np.median(
                [r["F_pred_expected"] / r["F_measured"] for r in rows if r["family"] == f])),
            "median_ratio_empirical": float(np.median(
                [r["F_pred_empirical"] / r["F_measured"] for r in rows if r["family"] == f])),
        } for f in fams},
        "per_direction": rows}

    # ---- 6. the two metrics compared --------------------------------------
    Cx, Cm, Ce = cosines(Fx), cosines(Fm), cosines(Gx)
    iu = np.triu_indices(p, 1)
    out["metric_comparison"] = {
        "what": "off-diagonal cosines of the three Grams over the same 134 "
                "adapters: exact Frobenius, expected Fisher, empirical Fisher",
        "n_pairs": int(len(iu[0])),
        "cos_exact": {"mean": float(Ce[iu].mean()), "sd": float(Ce[iu].std()),
                      "min": float(Ce[iu].min()), "max": float(Ce[iu].max())},
        "cos_fisher_expected": {"mean": float(Cx[iu].mean()), "sd": float(Cx[iu].std()),
                                "min": float(Cx[iu].min()), "max": float(Cx[iu].max())},
        "cos_fisher_empirical": {"mean": float(Cm[iu].mean()), "sd": float(Cm[iu].std()),
                                 "min": float(Cm[iu].min()), "max": float(Cm[iu].max())},
        "pearson_expected_vs_exact": float(np.corrcoef(Cx[iu], Ce[iu])[0, 1]),
        "spearman_expected_vs_exact": spearman(Cx[iu], Ce[iu]),
        "pearson_empirical_vs_exact": float(np.corrcoef(Cm[iu], Ce[iu])[0, 1]),
        "spearman_empirical_vs_exact": spearman(Cm[iu], Ce[iu]),
        "pearson_expected_vs_empirical": float(np.corrcoef(Cx[iu], Cm[iu])[0, 1]),
        "spearman_expected_vs_empirical": spearman(Cx[iu], Cm[iu])}

    # ---- 7. the per-adapter Fisher norm against the Frobenius norm --------
    fii = np.diag(Fx)
    out["fisher_norm_per_adapter"] = {
        "what": "F_ii, the Fisher norm of each single adapter's unit direction, "
                "in the same per-ref-unit-alpha units as analysis/fisher_norms.json",
        "min": float(fii.min()), "median": float(np.median(fii)),
        "max": float(fii.max()), "ratio_max_over_min": float(fii.max() / fii.min()),
        "argmin": names[int(np.argmin(fii))], "argmax": names[int(np.argmax(fii))],
        "pearson_with_frobenius_norm": float(np.corrcoef(fii, gnorms)[0, 1]),
        "spearman_with_frobenius_norm": spearman(fii, gnorms),
        "empirical_min": float(np.diag(Fm).min()),
        "empirical_median": float(np.median(np.diag(Fm))),
        "empirical_max": float(np.diag(Fm).max()),
        "pearson_expected_vs_empirical_diag": float(np.corrcoef(fii, np.diag(Fm))[0, 1]),
        "spearman_expected_vs_empirical_diag": spearman(fii, np.diag(Fm)),
        "per_trait": {names[i]: {"F_expected_ii": float(fii[i]),
                                 "F_empirical_ii": float(np.diag(Fm)[i]),
                                 "frobenius_norm": float(gnorms[i])}
                      for i in range(p)}}

    # ---- 8. write the npz files -------------------------------------------
    for tag, G, est in (("gram_fisher", Fx, "expected"),
                        ("gram_fisher_empirical", Fm, "empirical")):
        np.savez(f"{R}/{tag}.npz", G=G, names=np.array(names),
                 norms=np.sqrt(np.diag(G)), scale=np.float64(d["scale"]),
                 n_modules=np.int64(d["n_modules"]), estimator=np.array(est),
                 h=np.float64(J["h"]), ref=np.float64(J["ref"]),
                 n_scored_tokens=np.int64(J["n_scored_tokens"]))
        print(f"wrote results/{tag}.npz")

    json.dump(out, open(f"{A}/fisher_gram_validation.json", "w"), indent=1)
    print("wrote analysis/fisher_gram_validation.json")
    print(f"  zero-sum check         {out['numerics']['zero_sum_check_max_abs']:.3e} "
          f"against score scale {sd.min():.4f}")
    print(f"  h convergence          max |rel change| "
          f"{out['h_convergence']['max_abs_rel_change']:.3e}")
    print(f"  adapter norms          max rel diff "
          f"{out['adapter_norm_check']['max_abs_rel_diff']:.3e}")
    e = out["prediction_of_fisher_py"]["expected"]
    print(f"  predicts fisher.py     n={out['prediction_of_fisher_py']['n']} "
          f"median ratio {e['median_ratio']:.4f} max |rel err| {e['max_abs_rel_err']:.4f}")
    m = out["prediction_of_fisher_py"]["empirical"]
    print(f"  empirical form         median ratio {m['median_ratio']:.4f} "
          f"({m['min_ratio']:.4f}..{m['max_ratio']:.4f})")
    print(f"  cosine agreement       expected vs exact r="
          f"{out['metric_comparison']['pearson_expected_vs_exact']:.4f}, "
          f"rho={out['metric_comparison']['spearman_expected_vs_exact']:.4f}")


if __name__ == "__main__":
    main()
