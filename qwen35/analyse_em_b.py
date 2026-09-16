#!/usr/bin/env python3
"""Part B of the emergent-misalignment medical run: where do the arms land?

Input:  analysis/em_train.json
        results/cross_gram_full_root_x_pc-qwen35-adapters_em_flat.npz   (134 x arms)
        results/cross_gram_full_data_alignment_common_x_em_flat.npz     (4 x arms)
        results/cross_gram_full_em_flat_x_em_flat.npz                   (arms x arms)
        results/gram_sweep.npz, phase10_runs/steer_spec*.json
        analysis/em_part_a.json   (for the pre-registered forecast test)
Output: analysis/em_part_b.json

Chart placement is `fa_chart.FAChart().coords_external` on the exact cross-Gram
column, the convention of wiki/pages/geometry/factor-chart.md, the same path
analyse_dolci_flag.py takes.

The pre-registered test is the Spearman rank correlation, over the 145
directions that are merges of the zoo (134 singles + 5 FA_* + 5 axis_* +
mean_assistant_axis), between

  * the Part A mean paired first-order score (bad medical minus good medical),
    computed BEFORE training, and
  * the cosine of the trained difference delta (em_bad minus em_good) with the
    same direction,

with an exact two-sided permutation p over 20,000 shuffles of the ranks.  The
alignment adapters are excluded from the primary test because they are not
merges of the zoo and their cross-Gram column is not in the same units.
"""
import json
import os
import sys

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
N_PERM = 20000
PERM_SEED = 11
ARMS = ["em_bad", "em_good", "em_dolci"]
NAMED = [("axis_Conscientiousness", -1), ("FA_Competence", -1),
         ("trait_careless", +1), ("trait_negligent", +1),
         ("trait_crooked", +1), ("trait_selfish", +1), ("trait_unkind", +1)]


def spearman(x, y):
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float((rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry)))


def perm_p(x, y, n=N_PERM, seed=PERM_SEED):
    obs = abs(spearman(x, y))
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    hit = sum(1 for _ in range(n) if abs(spearman(x, rng.permutation(y))) >= obs)
    return (1 + hit) / (1 + n), hit


def main():
    gs = np.load(f"{Q}/results/gram_sweep.npz", allow_pickle=True)
    G = np.array(gs["G"], dtype=float)
    zoo_names = [str(x) for x in gs["names"]]
    zidx = {n: i for i, n in enumerate(zoo_names)}
    norms = np.sqrt(np.diag(G))

    from fa_chart import FAChart
    ch = FAChart()

    zoo_p = f"{Q}/results/cross_gram_full_root_x_pc-qwen35-adapters_em_flat.npz"
    align_p = f"{Q}/results/cross_gram_full_data_alignment_common_x_em_flat.npz"
    self_p = f"{Q}/results/cross_gram_full_em_flat_x_em_flat.npz"
    z = np.load(zoo_p, allow_pickle=True)
    na = [str(x) for x in z["names_a"]]
    nb = [str(x) for x in z["names_b"]]
    assert na == ch.names, "cross-Gram row order is not the Gram order"
    X = np.array(z["X"], dtype=float)              # (134, n_arm_ckpts)
    nrm_b = np.array(z["norms_b"], dtype=float)
    bidx = {n: j for j, n in enumerate(nb)}

    out = {"meta": {"arms": ARMS, "cross_gram_files": [os.path.basename(p) for p in
                                                       (zoo_p, align_p, self_p)],
                    "n_modules": int(z["n_modules"]),
                    "chart_convention": "fa_chart.FAChart, Gram-Schmidt in the order "
                                        "Warmth, Competence, FearfulWithdrawal, "
                                        "Arousal, Imagination",
                    "trait_chart_len_mean": float(ch.trait_chart_len.mean()),
                    "trait_chart_len_sd": float(ch.trait_chart_len.std(ddof=1)),
                    "trait_norm_mean": float(ch.norms.mean())}}
    out["train"] = json.load(open(f"{Q}/analysis/em_train.json"))
    out["build"] = json.load(open(f"{Q}/phase10_runs/em_build_report.json"))

    chart, near = {}, {}
    for j, b in enumerate(nb):
        col = X[:, j]
        x = ch.coords_external(col)
        chart[b] = {"coords": [float(v) for v in x],
                    "factor_order": ch.factor_names,
                    "chart_len": float(np.linalg.norm(x)),
                    "chart_len_over_trait_mean":
                        float(np.linalg.norm(x) / ch.trait_chart_len.mean()),
                    "cos_with_chart": float(np.linalg.norm(x) / nrm_b[j]),
                    "norm": float(nrm_b[j]),
                    "norm_over_trait_mean": float(nrm_b[j] / ch.norms.mean())}
        cos = col / (np.array(z["norms_a"], dtype=float) * nrm_b[j])
        o = np.argsort(-np.abs(cos))[:8]
        near[b] = [[na[i], float(cos[i])] for i in o]
    out["factor_chart"] = chart
    out["nearest_zoo_traits_by_abs_cosine"] = near

    if os.path.exists(align_p):
        za = np.load(align_p, allow_pickle=True)
        aa = [str(x) for x in za["names_a"]]
        ab = [str(x) for x in za["names_b"]]
        C = np.array(za["X"], dtype=float) / np.outer(za["norms_a"], za["norms_b"])
        out["cosine_with_alignment_adapters"] = {
            b: {a: float(C[i, j]) for i, a in enumerate(aa)} for j, b in enumerate(ab)}
    if os.path.exists(self_p):
        zs = np.load(self_p, allow_pickle=True)
        sn = [str(x) for x in zs["names_a"]]
        Cs = np.array(zs["X"], dtype=float) / np.outer(zs["norms_a"], zs["norms_b"])
        out["cosine_between_arms"] = {a: {b: float(Cs[i, j]) for j, b in enumerate(sn)}
                                      for i, a in enumerate(sn)}
        out["adapter_norms"] = {a: float(zs["norms_a"][i]) for i, a in enumerate(sn)}
        out["inner_products_between_arms"] = {
            a: {b: float(zs["X"][i, j]) for j, b in enumerate(sn)}
            for i, a in enumerate(sn)}

    # ---- the bad-minus-good contrast --------------------------------------
    fin = {a: f"{a}_final" for a in ARMS}
    IP = out.get("inner_products_between_arms", {})
    nb_, ng_ = fin["em_bad"], fin["em_good"]
    n2 = (IP[nb_][nb_] + IP[ng_][ng_] - 2 * IP[nb_][ng_])
    dnorm = float(np.sqrt(max(n2, 1e-30)))
    dcol = X[:, bidx[nb_]] - X[:, bidx[ng_]]        # <a_i, bad - good>
    dx = ch.coords_external(dcol)
    out["contrast_bad_minus_good"] = {
        "magnitude_ratio_bad_over_good": float(np.sqrt(IP[nb_][nb_] / IP[ng_][ng_])),
        "cosine": float(out["cosine_between_arms"][nb_][ng_]),
        "angle_degrees": float(np.degrees(np.arccos(
            np.clip(out["cosine_between_arms"][nb_][ng_], -1, 1)))),
        "difference_norm": dnorm,
        "difference_norm_over_trait_mean": dnorm / float(ch.norms.mean()),
        "difference_coords": [float(v) for v in dx],
        "difference_chart_len": float(np.linalg.norm(dx)),
        "difference_chart_len_over_trait_mean":
            float(np.linalg.norm(dx) / ch.trait_chart_len.mean()),
        "difference_cos_with_chart": float(np.linalg.norm(dx) / dnorm),
        "nearest_zoo_traits": [
            [na[i], float((dcol / (np.array(z["norms_a"], dtype=float) * dnorm))[i])]
            for i in np.argsort(-np.abs(dcol / (np.array(z["norms_a"], dtype=float)
                                                * dnorm)))[:10]],
    }
    if "cosine_with_alignment_adapters" in out:
        za = np.load(align_p, allow_pickle=True)
        aa = [str(x) for x in za["names_a"]]
        jb, jg = ab.index(nb_), ab.index(ng_)
        dv = np.array(za["X"], dtype=float)[:, jb] - np.array(za["X"], dtype=float)[:, jg]
        out["contrast_bad_minus_good"]["cosine_with_alignment_adapters"] = {
            a: float(dv[i] / (za["norms_a"][i] * dnorm)) for i, a in enumerate(aa)}

    # ---- the pre-registered forecast test ---------------------------------
    A = json.load(open(f"{Q}/analysis/em_part_a.json"))
    spec_fa = {j["name"]: j["coef"] for j in
               json.load(open(f"{Q}/phase10_runs/steer_spec2_7a.json"))["jobs"]}
    spec_ax = {j["name"]: j["coef"] for j in
               json.load(open(f"{Q}/phase10_runs/steer_spec.json"))["jobs"]}
    merges = {}
    for n in ("FA_Warmth", "FA_Competence", "FA_FearfulWithdrawal", "FA_Arousal",
              "FA_Imagination"):
        merges[n] = np.array([spec_fa[n].get(t, 0.0) for t in zoo_names])
    for n in ("axis_Extraversion", "axis_Agreeableness", "axis_Conscientiousness",
              "axis_EmotionalStability", "axis_Intellect", "mean_assistant_axis"):
        merges[n] = np.array([spec_ax[n].get(t, 0.0) for t in zoo_names])
    for t in zoo_names:
        e = np.zeros(len(zoo_names))
        e[zidx[t]] = 1.0
        merges[f"trait_{t}"] = e

    rows = []
    for name, c in merges.items():
        if name not in A["directions"]:
            continue
        vn = float(np.sqrt(c @ G @ c))
        cosv = float((c @ dcol) / (vn * dnorm))
        rows.append({"direction": name,
                     "forecast_score": A["directions"][name]["paired_diff"]["mean"],
                     "cos_with_trained_difference": cosv,
                     "clears_band": A["directions"][name]["vs_random_band"]["clears_band"]})
    fx = np.array([r["forecast_score"] for r in rows])
    fy = np.array([r["cos_with_trained_difference"] for r in rows])
    rho = spearman(fx, fy)
    p, hit = perm_p(fx, fy)
    sign_ok = [{"direction": n, "predicted_sign": s,
                "forecast_score": A["directions"][n]["paired_diff"]["mean"],
                "cos_with_trained_difference":
                    next(r["cos_with_trained_difference"] for r in rows
                         if r["direction"] == n),
                "forecast_sign_agrees":
                    bool(np.sign(A["directions"][n]["paired_diff"]["mean"]) == s),
                "trained_sign_agrees":
                    bool(np.sign(next(r["cos_with_trained_difference"] for r in rows
                                      if r["direction"] == n)) == s)}
               for n, s in NAMED if n in A["directions"]]
    k = sum(1 for r in sign_ok if r["trained_sign_agrees"])
    from math import comb
    binom = sum(comb(len(sign_ok), i) for i in range(k, len(sign_ok) + 1)) / 2 ** len(sign_ok)
    out["forecast_agreement"] = {
        "n_directions": len(rows),
        "spearman": rho, "p_permutation": p, "n_as_or_more_extreme": hit,
        "n_draws": N_PERM,
        "pearson": float(np.corrcoef(fx, fy)[0, 1]),
        "prereg_threshold": "Spearman > 0 at p < 0.05",
        "held": bool(rho > 0 and p < 0.05),
        "named_directions_sign_agreement": {
            "k_of_n": [k, len(sign_ok)],
            "p_binomial_one_sided": float(min(1.0, 2 * binom)) if False else float(binom),
            "per_direction": sign_ok},
        "rows": sorted(rows, key=lambda r: -abs(r["forecast_score"]))[:40],
    }

    p_out = f"{Q}/analysis/em_part_b.json"
    json.dump(out, open(p_out, "w"), indent=1)
    print("wrote", p_out)
    print(f"\ntrait chart length mean {ch.trait_chart_len.mean():.6f}")
    for b in nb:
        c = chart[b]
        print(f"  {b:<20} chart_len {c['chart_len']:.6f} "
              f"({100*c['chart_len_over_trait_mean']:.2f}% of a trait) "
              f"norm {c['norm']:.4f} ({c['norm_over_trait_mean']:.3f}x) "
              f"cos_chart {c['cos_with_chart']:.4f}")
    cb = out["contrast_bad_minus_good"]
    print(f"\nbad vs good: cosine {cb['cosine']:+.6f} ({cb['angle_degrees']:.1f} deg), "
          f"magnitude ratio {cb['magnitude_ratio_bad_over_good']:.6f}")
    print(f"  difference chart length {cb['difference_chart_len']:.6f} "
          f"({100*cb['difference_chart_len_over_trait_mean']:.2f}% of a trait)")
    print(f"  nearest zoo traits to the difference: "
          + ", ".join(f"{n} {v:+.4f}" for n, v in cb["nearest_zoo_traits"][:6]))
    fa = out["forecast_agreement"]
    print(f"\nforecast agreement over {fa['n_directions']} zoo-merge directions: "
          f"Spearman {fa['spearman']:+.4f}, p {fa['p_permutation']:.5f} -> "
          f"held={fa['held']}")
    print(f"  named-direction sign agreement {fa['named_directions_sign_agreement']['k_of_n']}")


if __name__ == "__main__":
    sys.exit(main())
