#!/usr/bin/env python3
"""The 34 Lexicon adapters analysed in isolation.

Samuel's follow-up, 2026-09-15: "when we analyse the 34 adapters in isolation do
we get the same thing?"  Pre-registered in the dated addendum to
PREREG_goldberg_only.md before any number here was computed.

analyse_fa_qwen35.py cannot be reused unchanged (load_data asserts 100 or 134
traits), so its MATHS is imported and called in the same order with the same
constants.  That script is not edited.  Every arm here - the 34 and all 200
reference subsets - goes through the identical code path.

  L1  factor the 34 alone, k = 5, and match against the 134 solution's
      loadings restricted to the same 34 rows.
  L2  200 random 34-subsets of the 100 Goldberg markers, same path, same
      scoring, as a reference distribution for what 34 variables can do.
  L3  build a chart from the 34 alone and place the 100 markers in it.

Adds a "lexicon_only" key to analysis/goldberg_only.json.  No API calls.

Usage:  qwen35/.venv/bin/python analyse_lexicon_only.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyse_fa_qwen35 import (smc, paf, kaiser_normalise, varimax_pairwise,   # noqa: E402
                               oblimin, tucker, parallel_analysis,
                               RIDGE, PA_REPS, PA_GRID, PA_REFERENCE_N, SEED)
from analyse_stage2_structure import best_match                                # noqa: E402
from analyse_goldberg_only import (slugify, coeffs, gs_basis, pearson, FN,     # noqa: E402
                                   FACTORS, CHART_TO_B5)
from fa_chart import FAChart                                                   # noqa: E402

N_REF = 200


def ipsatised_corr(G):
    """The script's centred path: H G H, then scale to a correlation matrix."""
    p = len(G)
    H = np.eye(p) - np.ones((p, p)) / p
    Gc = 0.5 * ((H @ G @ H) + (H @ G @ H).T)
    dc = np.sqrt(np.diag(Gc))
    return Gc / np.outer(dc, dc)


def solve_k5(R, k=5, ridge=RIDGE):
    """PAF -> kaiser normalise -> oblimin, ordered by SS loading, sign left
    arbitrary (order_and_orient needs Goldberg targets, which the 34 lack)."""
    ex = paf(R, k, ridge=ridge)
    L0, h2 = ex["loadings"], ex["communalities"]
    Ln, s = kaiser_normalise(L0, h2)
    ob = oblimin(Ln, 0.0)
    L = ob["pattern"] * s[:, None]
    order = list(np.argsort(-(L ** 2).sum(0)))
    L = L[:, order]
    return {"loadings": L, "ss": (L ** 2).sum(0),
            "communalities": h2, "converged": ex["converged"],
            "iterations": ex["iterations"], "n_heywood": ex["n_heywood"],
            "Phi": ob["Phi"][np.ix_(order, order)]}


def match_against_134(L_arm, L134_rows):
    C = np.array([[tucker(L_arm[:, j], L134_rows[:, a]) for a in range(5)]
                  for j in range(5)])
    bm = best_match(np.abs(C))
    perm = [b["matched"] for b in bm]
    cong = [float(C[i, perm[i]]) for i in range(5)]
    return C, perm, cong


def main():
    z = np.load(f"{HERE}/results/gram_sweep.npz", allow_pickle=True)
    G = np.asarray(z["G"], float)
    names = [str(x) for x in z["names"]]
    prim = json.load(open(f"{HERE}/traits_primary.json"))
    by_slug = {slugify(t["trait"]): t
               for t in prim + json.load(open(f"{HERE}/traits_secondary.json"))}
    pset = {slugify(t["trait"]) for t in prim}
    ix100 = np.array([i for i, n in enumerate(names) if n in pset])
    ix34 = np.array([i for i, n in enumerate(names) if n not in pset])
    d134 = json.load(open(f"{HERE}/results/fa_qwen35.json"))
    assert d134["trait_slug"] == names
    L134 = np.array(d134["solutions"]["centred_k5"]["loadings"]["oblimin"])

    out = {"what": "the 34 Lexicon adapters factored in isolation; "
                   "pre-registered in the 2026-09-15 addendum to "
                   "qwen35/PREREG_goldberg_only.md",
           "date": "2026-09-15",
           "method": {
               "why_the_script_was_not_reused":
                   "analyse_fa_qwen35.py:load_data asserts len(names) in "
                   "(100, 134) and n_primary == 100, so a 34-trait Gram cannot "
                   "be passed to it. Its maths (smc, paf, kaiser_normalise, "
                   "oblimin, tucker, parallel_analysis) is imported and called "
                   "in the same order with the same constants. The script is "
                   "not edited.",
               "ridge": RIDGE, "pa_reps": PA_REPS, "pa_grid": PA_GRID,
               "pa_reference_N": PA_REFERENCE_N, "seed": SEED,
               "ordering": "descending sum of squared oblimin loadings; sign "
                           "left arbitrary and resolved only by the one-to-one "
                           "matching, because order_and_orient needs Goldberg "
                           "targets and the 34 score 0 in every target",
               "variables_per_factor": {"p34_k5": 34 / 5, "p100_k5": 100 / 5,
                                        "p134_k5": 134 / 5}}}

    # ------------------------------------------------------------- L1
    G34 = G[np.ix_(ix34, ix34)]
    R34 = ipsatised_corr(G34)
    sol34 = solve_k5(R34)
    C34, perm34, cong34 = match_against_134(sol34["loadings"], L134[ix34])
    mean_abs34 = float(np.mean(np.abs(cong34)))

    # the p = 34 parallel-analysis null is data-independent: generate once
    pa34_full = parallel_analysis(R34, PA_GRID, 34, reps=PA_REPS,
                                  ipsatise=True, ridge=RIDGE, seed=SEED)
    pa34 = pa34_full["grid"]
    null_by_N = {g["N"]: (np.array(g["null_unreduced_95pct"]),
                          np.array(g["null_reduced_95pct"])) for g in pa34}

    def first_cross(obs, null):
        k = 0
        for i in range(min(len(obs), len(null))):
            if obs[i] > null[i]:
                k = i + 1
            else:
                break
        return k

    def retention(R, N=PA_REFERENCE_N):
        obs = np.linalg.eigvalsh(R)[::-1]
        return first_cross(obs, null_by_N[N][0])

    ev34 = np.linalg.eigvalsh(R34)[::-1]
    out["L1_factor_the_34_alone"] = {
        "n_variables": 34, "k_forced": 5,
        "congruence_matrix_rows_34only_cols_134": C34.round(8).tolist(),
        "one_to_one_matching": {
            "matcher": "analyse_stage2_structure.best_match on |Tucker|",
            "assignment_34only_factor_to_134_factor":
                {str(i + 1): int(perm34[i] + 1) for i in range(5)},
            "assignment_34only_factor_to_134_factor_name":
                {str(i + 1): FN[perm34[i]] for i in range(5)},
            "matched_congruences": cong34,
            "matched_congruences_abs": [abs(c) for c in cong34],
            "mean_abs_matched_congruence": mean_abs34,
            "min_abs_matched_congruence": float(np.min(np.abs(cong34))),
            "max_abs_matched_congruence": float(np.max(np.abs(cong34))),
            "is_permutation_over_all_five": bool(len(set(perm34)) == 5),
            "n_above_0.95": int((np.abs(cong34) >= 0.95).sum()),
            "n_above_0.85": int((np.abs(cong34) >= 0.85).sum())},
        "ss_loadings_oblimin": sol34["ss"].tolist(),
        "mean_communality": float(np.mean(sol34["communalities"])),
        "paf_converged": bool(sol34["converged"]),
        "paf_iterations": int(sol34["iterations"]),
        "n_heywood": int(sol34["n_heywood"]),
        "max_abs_offdiag_Phi": float(max(abs(sol34["Phi"][i][j])
                                         for i in range(5) for j in range(5)
                                         if i != j)),
        "centred_eigenvalues_top12": ev34[:12].tolist(),
        "parallel_analysis_grid":
            [{"N": g["N"], "k_unreduced_95pct": g["k_unreduced_95pct"],
              "k_reduced_95pct": g["k_reduced_95pct"]} for g in pa34],
        "n_factors_retained_at_reference_N":
            [g["k_unreduced_95pct"] for g in pa34 if g["N"] == PA_REFERENCE_N][0]}

    # ------------------------------------------------------------- L2
    rng = np.random.default_rng(SEED)
    ref_mean, ref_min, ref_perm_ok, ref_ret, ref_hey = [], [], [], [], []
    for _ in range(N_REF):
        sub = rng.choice(ix100, size=34, replace=False)
        Rs = ipsatised_corr(G[np.ix_(sub, sub)])
        s = solve_k5(Rs)
        _, pm, cg = match_against_134(s["loadings"], L134[sub])
        ref_mean.append(float(np.mean(np.abs(cg))))
        ref_min.append(float(np.min(np.abs(cg))))
        ref_perm_ok.append(bool(len(set(pm)) == 5))
        ref_ret.append(retention(Rs))
        ref_hey.append(int(s["n_heywood"]))
    ref_mean = np.array(ref_mean)
    pct = float((ref_mean < mean_abs34).mean() * 100.0)
    qs = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    ret_counts = {str(v): int((np.array(ref_ret) == v).sum())
                  for v in sorted(set(ref_ret))}
    out["L2_random_34_reference_distribution"] = {
        "what": "200 random 34-trait subsets of the 100 Goldberg markers, each "
                "factored by the identical path and scored by the same "
                "one-to-one Tucker congruence against the 134 solution "
                "restricted to that subset's rows",
        "n_subsets": N_REF, "seed": SEED, "drawn_without_replacement": True,
        "mean_abs_matched_congruence": {
            "quantiles": {f"p{q}": float(np.percentile(ref_mean, q)) for q in qs},
            "mean": float(ref_mean.mean()), "sd": float(ref_mean.std(ddof=1))},
        "min_abs_matched_congruence_median": float(np.median(ref_min)),
        "fraction_of_subsets_whose_matching_is_a_permutation":
            float(np.mean(ref_perm_ok)),
        "lexicon34_mean_abs_matched_congruence": mean_abs34,
        "lexicon34_percentile_of_reference": pct,
        "lexicon34_within_middle_90_percent": bool(5.0 <= pct <= 95.0),
        "lexicon34_above_reference_median":
            bool(mean_abs34 > float(np.percentile(ref_mean, 50))),
        "parallel_analysis_retention_at_reference_N": {
            "reference_N": PA_REFERENCE_N,
            "reference_subsets_counts": ret_counts,
            "reference_subsets_median": float(np.median(ref_ret)),
            "reference_subsets_mean": float(np.mean(ref_ret)),
            "lexicon34": out["L1_factor_the_34_alone"]["n_factors_retained_at_reference_N"]},
        "heywood_cases": {
            "reference_subsets_mean": float(np.mean(ref_hey)),
            "reference_subsets_max": int(np.max(ref_hey)),
            "fraction_of_subsets_with_any": float(np.mean(np.array(ref_hey) > 0)),
            "lexicon34": int(sol34["n_heywood"])},
        "note": "the null for parallel analysis depends only on p and N, not on "
                "the data, so the p = 34 null was generated once at 500 reps and "
                "reused for every arm; this is exact, not an approximation"}

    # ------------------------------------------------------------- L3
    ch = FAChart()
    L34 = sol34["loadings"]
    inv = {perm34[j]: j for j in range(5)}
    C34d = np.zeros((5, 34))
    signs = []
    for slot in range(5):
        j = inv[slot]
        s = 1.0 if cong34[j] > 0 else -1.0
        signs.append(s)
        C34d[slot] = s * coeffs(G34, L34, j)
    B34 = gs_basis(G34, C34d)
    X100_34 = (B34 @ G[np.ix_(ix34, ix100)]).T          # 100 x 5
    X100_134 = ch.trait_coords[ix100]
    X34_34 = (B34 @ G34).T

    K = np.zeros((100, 5))
    for i, gi in enumerate(ix100):
        t = by_slug[names[gi]]
        K[i, FACTORS.index(t["factor"])] = 1.0 if t["keyed"] == "+" else -1.0

    vs_key = {FN[k]: pearson(X100_34[:, k], K[:, FACTORS.index(CHART_TO_B5[FN[k]])])
              for k in range(5)}
    vs_full = {FN[k]: pearson(X100_34[:, k], X100_134[:, k]) for k in range(5)}
    ceiling = {"Warmth": 0.6828059271418205, "Competence": 0.5723042028598464,
               "Timidity": 0.47039913464559113, "Arousal": 0.5452041935257524,
               "Imagination": 0.7639889196055134}
    ceil_mean = 0.6069404755557047
    Bfull = np.zeros((5, 134))
    Bfull[:, ix34] = B34
    pcos = np.linalg.svd(Bfull @ G @ ch.basis.T, compute_uv=False)
    out["L3_reverse_place_the_100_markers_in_the_34_chart"] = {
        "how_the_chart_was_built":
            "the five k=5 oblimin loading columns of the 34-only solution turned "
            "into directions over the 34 adapters by "
            "steer134_on_modal.py:fa_coeffs, reordered to the 134 chart's fixed "
            "order by the L1 matching, sign-oriented so each matched congruence "
            "is positive, Gram-Schmidt orthonormalised in the 34 x 34 Gram",
        "how_the_100_were_placed":
            "x = B34 @ G[ix34, marker], their exact inner products with the 34; "
            "the markers contribute nothing to the basis",
        "sign_applied_per_slot": dict(zip(FN, [float(s) for s in signs])),
        "markers_vs_goldberg_keying": {
            "per_axis_pearson": vs_key,
            "mean_r": float(np.mean(list(vs_key.values()))),
            "all_positive": bool(all(v > 0 for v in vs_key.values())),
            "n_axes_at_least_0.34": int(sum(1 for v in vs_key.values() if v >= 0.34))},
        "ceiling_134chart_vs_goldberg_keying": {
            "per_axis_pearson": ceiling, "mean_r": ceil_mean,
            "source": "#test2b_independent_lexical_ratings."
                      "ceiling_goldberg_keying_vs_134chart_100markers"},
        "fraction_of_ceiling":
            float(np.mean(list(vs_key.values())) / ceil_mean),
        "half_ceiling_bar": ceil_mean / 2,
        "markers_vs_full_134_chart": {
            "per_axis_pearson": vs_full,
            "mean_r": float(np.mean(list(vs_full.values()))),
            "min_r": float(min(vs_full.values()))},
        "principal_cosines_34chart_vs_134chart": pcos.tolist(),
        "in_sample_34_chart_vs_134_chart_per_axis":
            {FN[k]: pearson(X34_34[:, k], ch.trait_coords[ix34][:, k])
             for k in range(5)},
        "coords_100_from_34only_chart":
            {names[ix100[i]]: X100_34[i].tolist() for i in range(100)}}

    # -------------------------------------------- why, if it is weaker: coverage
    X34_134 = ch.trait_coords[ix34]
    X100c = ch.trait_coords[ix100]

    def dominant_counts(A):
        d = {f: 0 for f in FN}
        for row in A:
            d[FN[int(np.argmax(np.abs(row)))]] += 1
        return d

    def energy_share(A):
        e = (A ** 2).sum(0)
        return {FN[k]: float(e[k] / e.sum()) for k in range(5)}

    c34 = dominant_counts(X34_134)
    c100 = dominant_counts(X100c)
    out["L4_axis_coverage_of_the_two_word_sets"] = {
        "what": "how the two word sets cover the five chart axes, in the shared "
                "134 chart, as a candidate explanation for any deficit in L1",
        "dominant_axis_counts_34": c34,
        "dominant_axis_counts_100": c100,
        "dominant_axis_counts_100_rescaled_to_34":
            {f: c100[f] * 34 / 100 for f in FN},
        "squared_chart_energy_share_34": energy_share(X34_134),
        "squared_chart_energy_share_100": energy_share(X100c),
        "note": "the 34 were drawn by k-means over sentence embeddings to spread "
                "across the trait lexicon, not to tile the Big Five; Goldberg's "
                "100 are 20 per factor by construction, so any 34 of them still "
                "carry about 7 per factor"}

    # ------------------------------------------------------------- verdicts
    m = out["L1_factor_the_34_alone"]["one_to_one_matching"]
    l3k = out["L3_reverse_place_the_100_markers_in_the_34_chart"]
    out["verdicts"] = {
        "L_same_structure": {
            "threshold": "the lexicon 34's mean absolute matched Tucker "
                         "congruence falls within the middle 90 per cent of the "
                         "200-subset reference distribution (percentile 5 to 95) "
                         "AND the matching is a permutation over all five",
            "lexicon34_mean_abs_congruence": mean_abs34,
            "percentile_of_reference": pct,
            "reference_p5": float(np.percentile(ref_mean, 5)),
            "reference_p50": float(np.percentile(ref_mean, 50)),
            "reference_p95": float(np.percentile(ref_mean, 95)),
            "matching_is_permutation": m["is_permutation_over_all_five"],
            "pass": bool(5.0 <= pct <= 95.0 and m["is_permutation_over_all_five"])},
        "L_different_structure": {
            "threshold": "congruence below the 5th percentile of the reference, "
                         "OR the matching is not a permutation over the five",
            "pass": bool(pct < 5.0 or not m["is_permutation_over_all_five"])},
        "L3_the_34_contain_the_frame": {
            "threshold": "the 100 markers placed in the 34-defined chart "
                         "correlate with Goldberg keying at a mean per-axis "
                         "Pearson of at least half the ceiling "
                         "(0.30347023777785235) with all five signs positive, "
                         "AND agree with the full 134 chart at a mean per-axis "
                         "Pearson of at least 0.70",
            "mean_r_vs_keying": l3k["markers_vs_goldberg_keying"]["mean_r"],
            "half_ceiling_bar": ceil_mean / 2,
            "all_signs_positive": l3k["markers_vs_goldberg_keying"]["all_positive"],
            "mean_r_vs_full_chart": l3k["markers_vs_full_134_chart"]["mean_r"],
            "pass": bool(l3k["markers_vs_goldberg_keying"]["mean_r"] >= ceil_mean / 2
                         and l3k["markers_vs_goldberg_keying"]["all_positive"]
                         and l3k["markers_vs_full_134_chart"]["mean_r"] >= 0.70)},
        "lexicon34_above_reference_median":
            out["L2_random_34_reference_distribution"]["lexicon34_above_reference_median"]}
    out["spend"] = {"total_usd": 0.0,
                    "note": "no API calls; numpy on CPU only"}

    path = f"{HERE}/analysis/goldberg_only.json"
    doc = json.load(open(path))
    doc["lexicon_only"] = out
    json.dump(doc, open(path, "w"), indent=1)
    print(json.dumps(out["verdicts"], indent=1))
    print("\nL1 matched:", [round(c, 4) for c in cong34], "mean|.|", round(mean_abs34, 6))
    print("L2 percentile:", round(pct, 2),
          "ref p5/p50/p95:", [round(float(np.percentile(ref_mean, q)), 4) for q in (5, 50, 95)])
    print("L3 vs keying:", {k: round(v, 4) for k, v in vs_key.items()})
    print("L3 vs full chart:", {k: round(v, 4) for k, v in vs_full.items()})
    print("retention lexicon34:", out["L1_factor_the_34_alone"]["n_factors_retained_at_reference_N"],
          "reference counts:", ret_counts)
    print("wrote lexicon_only into analysis/goldberg_only.json")


if __name__ == "__main__":
    main()
