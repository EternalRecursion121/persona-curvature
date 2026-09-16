#!/usr/bin/env python3
"""Is the five-factor structure an artefact of having chosen Big Five marker words?

Pre-registered in PREREG_goldberg_only.md before any number here was computed.
Three tests, CPU only, no retraining:

  1. Factor the 100 Goldberg markers alone (results/gram_goldberg100.npz, built
     here, run through analyse_fa_qwen35.py unchanged via PC_GRAM_NPZ /
     PC_FA_TAG), and compare the five factors with the 134-trait solution
     restricted to the same 100 rows.  This is also the same-size real
     comparator the two null arms never had.
  2. Build the factor chart from the 100 only and place the 34 Lexicon
     adapters in it by their exact inner products with the 100.  Compare with
     their coordinates in the full 134 chart, and with an independent lexical
     rating of all 134 words by an LLM that saw no project data.
  3. The sweep100 text baseline: ridge from constitution embedding to chart
     coordinates, fitted on the 100 and tested on the 34, plus the correlation
     between the text Gram and the adapter Gram, with and without the
     chosen-minus-rejected contrast sweep100 found to be load-bearing.

Inputs (all existing artefacts):
    results/gram_sweep.npz, results/fa_qwen35.json,
    results/fa_qwen35_goldberg100.json, traits_primary.json,
    traits_secondary.json, constitutions.json, data/<slug>.jsonl,
    analysis/fa_nulls.json, analysis/goldberg_only_ratings.jsonl,
    analysis/emb_constitutions_mpnet.npy, analysis/emb_pairs_minilm.npz,
    phase10_runs/steer_spec2_7a.json

Writes: analysis/goldberg_only.json

The two embedding files are produced by embed_goldberg_only.py, which needs
torch and sentence-transformers (~/cartovenv/bin/python); everything here runs
on numpy + scipy.

Usage:  qwen35/.venv/bin/python analyse_goldberg_only.py
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.linalg import orthogonal_procrustes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fa_chart import FAChart                                     # noqa: E402
from analyse_fa_qwen35 import tucker                             # noqa: E402
from analyse_stage2_structure import best_match                  # noqa: E402

FACTORS = ["Extraversion", "Agreeableness", "Conscientiousness",
           "EmotionalStability", "Intellect"]
SHORT = ["E", "A", "C", "ES", "I"]
FN = ["Warmth", "Competence", "Timidity", "Arousal", "Imagination"]
# The project's own factor-to-Big-Five map, analysis/viz_fa.json#big_five_scale.
# It carries no sign flag: every axis is oriented POSITIVELY on its own scale by
# analyse_fa_qwen35.py:order_and_orient.  The pre-registration expected Timidity
# to be Emotional Stability reversed; see verdicts.T2b_sign_note.
CHART_TO_B5 = {"Warmth": "Agreeableness", "Competence": "Conscientiousness",
               "Timidity": "EmotionalStability", "Arousal": "Extraversion",
               "Imagination": "Intellect"}
PREREG_SIGN = {"Warmth": +1, "Competence": +1, "Timidity": -1,
               "Arousal": +1, "Imagination": +1}
RIDGE_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]
N_PAIRS_SUBSAMPLE = 150


def slugify(t):
    return str(t).lower().replace(" ", "_").replace("-", "_")


# ---------------------------------------------------------------- primitives

def coeffs(Gm, L, j):
    """steer134_on_modal.py:fa_coeffs -- oblimin column j, mean-centred, unit
    norm in the double-centred Gram."""
    n = Gm.shape[0]
    c = L[:, j].copy()
    c = c - c.mean()
    one = np.ones((n, n)) / n
    Gc = Gm - one @ Gm - Gm @ one + one @ Gm @ one
    q = float(c @ Gc @ c)
    assert q > 0, f"degenerate factor direction {j}"
    return c / np.sqrt(q)


def gs_basis(Gm, C):
    """fa_chart.FAChart.__init__ -- Gram-Schmidt in the G inner product."""
    B = []
    for k in range(C.shape[0]):
        v = C[k].copy()
        for b in B:
            v = v - (b @ Gm @ v) * b
        v = v / np.sqrt(v @ Gm @ v)
        B.append(v)
    B = np.array(B)
    assert np.allclose(B @ Gm @ B.T, np.eye(len(B)), atol=1e-8)
    return B


def double_centre(M):
    p = len(M)
    H = np.eye(p) - np.ones((p, p)) / p
    A = H @ M @ H
    return 0.5 * (A + A.T)


def cosine_gram(V):
    n = V / np.linalg.norm(V, axis=1, keepdims=True)
    return n @ n.T


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def ridge_fit(Z, Y, alpha):
    d = Z.shape[1]
    W = np.linalg.solve(Z.T @ Z + alpha * np.eye(d), Z.T @ (Y - Y.mean(0)))
    return W, Y.mean(0)


def ridge_loo_mse(Z, Y, alpha):
    """Leave-one-out mean squared error for ridge, by actually refitting.

    The textbook hat-matrix shortcut is exact in theory but useless here: with
    768 features and 100 training rows, small alpha interpolates, the leverages
    h_i go to 1, and the shortcut evaluates 0/0. It silently prefers the
    smallest alpha on the grid. Each fold is refitted instead, in the dual
    (kernel) form, which is the same estimator and costs an n x n solve.
    """
    n = len(Z)
    se = 0.0
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        Zt, Yt = Z[m], Y[m]
        ym = Yt.mean(0)
        K = Zt @ Zt.T
        A = np.linalg.solve(K + alpha * np.eye(n - 1), Yt - ym)
        pred = (Z[i] @ Zt.T) @ A + ym
        se += float(((Y[i] - pred) ** 2).sum())
    return se / (n * Y.shape[1])


def r2(pred, Y, base):
    sse = ((Y - pred) ** 2).sum(0)
    sst = ((Y - base) ** 2).sum(0)
    return (1.0 - sse / sst), float(1.0 - sse.sum() / sst.sum())


# ---------------------------------------------------------------------- main

def main():
    out = {"what": "Goldberg-only factor analysis and the 34 Lexicon traits as a "
                   "held-out validation set; pre-registered in "
                   "qwen35/PREREG_goldberg_only.md",
           "date": "2026-09-12"}

    z = np.load(f"{HERE}/results/gram_sweep.npz", allow_pickle=True)
    G = np.asarray(z["G"], float)
    names = [str(x) for x in z["names"]]
    prim = json.load(open(f"{HERE}/traits_primary.json"))
    seco = json.load(open(f"{HERE}/traits_secondary.json"))
    by_slug = {slugify(t["trait"]): t for t in prim + seco}
    pset = {slugify(t["trait"]) for t in prim}
    ix100 = np.array([i for i, n in enumerate(names) if n in pset])
    ix34 = np.array([i for i, n in enumerate(names) if n not in pset])
    assert len(ix100) == 100 and len(ix34) == 34
    G100 = G[np.ix_(ix100, ix100)]

    d134 = json.load(open(f"{HERE}/results/fa_qwen35.json"))
    d100 = json.load(open(f"{HERE}/results/fa_qwen35_goldberg100.json"))
    fan = json.load(open(f"{HERE}/analysis/fa_nulls.json"))
    assert d100["trait_slug"] == [names[i] for i in ix100]
    L134 = np.array(d134["solutions"]["centred_k5"]["loadings"]["oblimin"])
    L100 = np.array(d100["solutions"]["centred_k5"]["loadings"]["oblimin"])

    out["inputs"] = {
        "gram": "qwen35/results/gram_sweep.npz",
        "gram_goldberg100": "qwen35/results/gram_goldberg100.npz",
        "fa_134": "qwen35/results/fa_qwen35.json",
        "fa_goldberg100": "qwen35/results/fa_qwen35_goldberg100.json",
        "nulls": "qwen35/analysis/fa_nulls.json",
        "ratings": "qwen35/analysis/goldberg_only_ratings.jsonl",
        "n_primary": 100, "n_lexicon": 34,
        "how_gram_goldberg100_was_built":
            "results/gram_sweep.npz restricted to the 100 traits_primary slugs: "
            "G[ix][:, ix], names[ix], norms[ix], same scale and n_modules",
        "how_fa_goldberg100_was_run":
            "PC_GRAM_NPZ=results/gram_goldberg100.npz PC_FA_TAG=_goldberg100 "
            "python analyse_fa_qwen35.py, the script unchanged, exactly as the "
            "two null arms were run",
        "centring_caveat":
            "analyse_fa_qwen35.py double-centres over the p variables it is "
            "handed, so the 100-only run removes the 100-trait grand mean and "
            "the 134 run the 134-trait grand mean; every comparison below "
            "includes that change of origin. Pre-registered."}

    # =================================================== TEST 1
    C = np.array([[tucker(L100[:, j], L134[ix100][:, a]) for a in range(5)]
                  for j in range(5)])
    bm = best_match(C)                       # the project's own matcher
    perm = [b["matched"] for b in bm]
    matched = [b["congruence"] for b in bm]
    # independent check with a Hungarian solver
    from scipy.optimize import linear_sum_assignment
    _, hcol = linear_sum_assignment(-np.abs(C))
    t1 = {"n_rows_shared": 100,
          "congruence_matrix_rows_100only_cols_134": C.round(8).tolist(),
          "one_to_one_matching": {
              "matcher": "analyse_stage2_structure.best_match, the function "
                         "analyse_fa_nulls.py uses for fa_nulls.json",
              "matcher_agrees_with_hungarian": bool(list(perm) == list(hcol)),
              "assignment_100only_factor_to_134_factor":
                  {str(i + 1): int(perm[i] + 1) for i in range(5)},
              "assignment_100only_factor_to_134_factor_name":
                  {str(i + 1): FN[perm[i]] for i in range(5)},
              "matched_congruences": matched,
              "mean_abs_matched_congruence": float(np.mean(np.abs(matched))),
              "min_abs_matched_congruence": float(np.min(np.abs(matched))),
              "n_above_0.95": int((np.abs(matched) >= 0.95).sum()),
              "n_above_0.85": int((np.abs(matched) >= 0.85).sum()),
              "is_identity_permutation": bool(list(perm) == list(range(5))),
              "note": "the permutation is not the identity because the "
                      "100-only solution orders its factors by its own sum of "
                      "squared oblimin loadings; every 134 factor is taken "
                      "exactly once"}}

    cong100 = np.array(d100["solutions"]["centred_k5"]["congruence_oblimin"])
    best_per_target, best_factor_per_target = {}, {}
    for a, s in enumerate(SHORT):
        j = int(np.argmax(np.abs(cong100[:, a])))
        best_per_target[s] = float(cong100[j, a])
        best_factor_per_target[s] = j + 1
    best_target_per_factor = {f"F{j+1}": SHORT[int(np.argmax(np.abs(cong100[j, :5])))]
                              for j in range(5)}
    ref = fan["arms"]["real"]["best_congruence_per_big_five_target"]

    # Decomposition: a Goldberg target is 0 on all 34 Lexicon rows, so those
    # rows can only inflate Tucker's denominator. Dropping them raises
    # congruence for an UNCHANGED loading pattern. Separate the two effects.
    T134 = np.zeros((134, 5))
    for i, n in enumerate(names):
        t = by_slug[n]
        if t["factor"] != "Lexicon":
            T134[i, FACTORS.index(t["factor"])] = 1.0 if t["keyed"] == "+" else -1.0
    restricted = {s: max((tucker(L134[ix100][:, j], T134[ix100][:, a])
                          for j in range(5)), key=abs)
                  for a, s in enumerate(SHORT)}
    t1["big_five_congruence"] = {
        "goldberg100_best_congruence_per_target": best_per_target,
        "goldberg100_best_factor_per_target": best_factor_per_target,
        "goldberg100_best_target_per_factor": best_target_per_factor,
        "goldberg100_all_five_targets_taken_once_per_factor_argmax":
            bool(len(set(best_target_per_factor.values())) == 5),
        "goldberg100_n_distinct_factors_across_targets":
            len(set(best_factor_per_target.values())),
        "reference_134_best_congruence_per_target": ref,
        "delta_vs_134": {s: float(best_per_target[s] - ref[s]) for s in SHORT},
        "max_abs_delta_vs_134":
            float(max(abs(best_per_target[s] - ref[s]) for s in SHORT)),
        "n_targets_clearing_0.85":
            int(sum(1 for v in best_per_target.values() if abs(v) >= 0.85)),
        "ss_loadings_oblimin": d100["solutions"]["centred_k5"]["ss_loadings"]["oblimin"],
        "mean_communality":
            float(np.mean(d100["solutions"]["centred_k5"]["communalities"])),
        "full_congruence_oblimin_rows_factors_cols_E_A_C_ES_I_Eval":
            cong100.round(8).tolist(),
        "decomposition_of_the_rise": {
            "what": "a Goldberg target is zero on all 34 Lexicon rows, so those "
                    "rows enter only Tucker's denominator and can only lower "
                    "congruence; the 134 loadings restricted to the 100 rows "
                    "isolate that effect from any change in the solution",
            "congruence_134_solution_all_134_rows": ref,
            "congruence_134_solution_restricted_to_100_rows": restricted,
            "congruence_100only_solution": best_per_target,
            "rise_from_dropping_the_zero_target_rows":
                {s: float(restricted[s] - ref[s]) for s in SHORT},
            "rise_from_refactoring_without_the_34":
                {s: float(best_per_target[s] - restricted[s]) for s in SHORT},
            "reading": "the whole rise is the denominator; refactoring without "
                       "the 34 changes congruence by -0.0496 to +0.0012"}}

    nf = d100["n_factors"]
    ev100 = d100["correlation_matrix"]["centred_eigenvalues"][:12]
    t1["n_factors"] = {
        "chosen": nf["chosen"], "reference_N": nf["reference_N"],
        "kaiser_uncentred_eig_gt_1": nf["kaiser_uncentred_eig_gt_1"],
        "kaiser_centred_eig_gt_1": nf["kaiser_centred_eig_gt_1"],
        "reduced_eig_gt_1_uncentred": nf["reduced_eig_gt_1_uncentred"],
        "parallel_analysis_centred_grid":
            [{"N": g["N"], "k_unreduced_95pct": g["k_unreduced_95pct"],
              "k_reduced_95pct": g["k_reduced_95pct"]}
             for g in nf["parallel_analysis_centred"]["grid"]],
        "parallel_analysis_uncentred_grid":
            [{"N": g["N"], "k_unreduced_95pct": g["k_unreduced_95pct"],
              "k_reduced_95pct": g["k_reduced_95pct"]}
             for g in nf["parallel_analysis_uncentred"]["grid"]]}
    t1["same_size_null_comparison"] = {
        "what": "the open-questions item 'Never run: a 100-trait real arm for "
                "the null comparison'. The two null arms have p = 100 against "
                "the real arm's p = 134 and a correlation matrix has trace p, "
                "so their eigenvalues were never on the same scale as the real "
                "arm's. This is the p = 100 real arm.",
        "goldberg100_centred_eigenvalues_top12": ev100,
        "goldberg100_reduced_eigenvalues_centred_k5_top12":
            d100["solutions"]["centred_k5"]["reduced_eigenvalues"][:12],
        "shuffled_centred_eigenvalues_top12":
            fan["arms"]["shuffled"]["centred_eigenvalues_top12"],
        "permuted_centred_eigenvalues_top12":
            fan["arms"]["permuted"]["centred_eigenvalues_top12"],
        "real134_centred_eigenvalues_top12":
            fan["arms"]["real"]["centred_eigenvalues_top12"],
        "n_factors_chosen_goldberg100": nf["chosen"],
        "n_factors_chosen_shuffled": fan["arms"]["shuffled"]["n_factors_chosen"],
        "n_factors_chosen_permuted": fan["arms"]["permuted"]["n_factors_chosen"],
        "n_factors_chosen_real134": fan["arms"]["real"]["n_factors_chosen"],
        "ratio_leading_eigenvalue_goldberg100_over_shuffled":
            float(ev100[0] / fan["arms"]["shuffled"]["centred_eigenvalues_top12"][0]),
        "ratio_leading_eigenvalue_goldberg100_over_permuted":
            float(ev100[0] / fan["arms"]["permuted"]["centred_eigenvalues_top12"][0]),
        "best_congruence_shuffled":
            fan["arms"]["shuffled"]["best_congruence_per_big_five_target"],
        "best_congruence_permuted":
            fan["arms"]["permuted"]["best_congruence_per_big_five_target"],
        "reading": "at p = 100 the real arm's spectrum and the permuted arm's "
                   "are the same size and nearly the same shape, and both "
                   "retain 8 factors; the shuffled arm is flat. The spectrum "
                   "does not tell real labels from permuted ones. Only "
                   "congruence with the keying does."}
    s5 = d100["solutions"]["centred_k5"]
    t1["diagnostics"] = {
        "verification_all_ok": d100["verification"]["all_ok"],
        "paf_converged": s5["paf_converged"], "paf_iterations": s5["paf_iterations"],
        "n_heywood": s5["n_heywood"],
        "max_abs_offdiag_Phi": float(max(abs(s5["Phi"][i][j])
                                         for i in range(5) for j in range(5) if i != j)),
        "reference_134_max_abs_offdiag_Phi": 0.282,
        "setup": d100["setup"]}
    t1["correlation_matrix"] = {
        "centred_offdiag": d100["correlation_matrix"]["centred_offdiag"],
        "uncentred_offdiag": d100["correlation_matrix"]["uncentred_offdiag"],
        "reference_134_centred_offdiag":
            d134["correlation_matrix"]["centred_offdiag"]}
    out["test1_goldberg_only_factor_analysis"] = t1

    # =================================================== TEST 2a
    ch = FAChart()
    C134 = np.array([coeffs(G, L134, j) for j in range(5)])
    B134 = gs_basis(G, C134)
    sanity = float(np.abs(B134 - ch.basis).max())

    slot_of = {j: perm[j] for j in range(5)}          # 100-only factor -> chart slot
    inv = {v: k for k, v in slot_of.items()}
    assert sorted(inv) == list(range(5)), slot_of
    C100 = np.zeros((5, 100))
    signs = []
    for slot in range(5):
        j = inv[slot]
        s = 1.0 if matched[j] > 0 else -1.0
        signs.append(s)
        C100[slot] = s * coeffs(G100, L100, j)
    B100 = gs_basis(G100, C100)

    X34_100 = (B100 @ G[np.ix_(ix100, ix34)]).T
    X34_134 = ch.trait_coords[ix34]
    X100_134 = ch.trait_coords[ix100]
    X100_100 = (B100 @ G100).T

    per_axis = {FN[k]: pearson(X34_100[:, k], X34_134[:, k]) for k in range(5)}
    A = X34_100 - X34_100.mean(0)
    Bm = X34_134 - X34_134.mean(0)
    A_ = A / np.linalg.norm(A)
    B_ = Bm / np.linalg.norm(Bm)
    R, _ = orthogonal_procrustes(A_, B_)
    cos_pt = [float(X34_100[i] @ X34_134[i] /
                    np.linalg.norm(X34_100[i]) / np.linalg.norm(X34_134[i]))
              for i in range(34)]
    Bfull = np.zeros((5, 134))
    Bfull[:, ix100] = B100
    pcos = np.linalg.svd(Bfull @ G @ ch.basis.T, compute_uv=False)

    out["test2a_the_34_in_a_chart_they_did_not_define"] = {
        "how_the_chart_was_built":
            "the five centred_k5 oblimin loading columns of the 100-only "
            "solution, each turned into a direction over the 100 adapters by "
            "steer134_on_modal.py:fa_coeffs (mean-centred, unit norm in the "
            "double-centred 100x100 Gram), reordered to the 134 chart's fixed "
            "order and sign-oriented to its matched 134 factor, then "
            "Gram-Schmidt orthonormalised in the 100x100 Gram",
        "recipe_validation": {
            "what": "the same recipe applied to the 134 centred_k5 oblimin "
                    "loadings must reproduce fa_chart.FAChart's basis, which is "
                    "built from phase10_runs/steer_spec2_7a.json",
            "max_abs_basis_difference_vs_FAChart": sanity},
        "how_the_34_were_placed":
            "x = B100 @ G[ix100, e], the exact inner products of the 34 with the "
            "100, fa_chart.py's coords_external convention; the 34 contribute "
            "nothing to the basis",
        "chart_slot_of_each_100only_factor":
            {str(j + 1): FN[slot_of[j]] for j in range(5)},
        "sign_applied_per_slot": dict(zip(FN, [float(s) for s in signs])),
        "per_axis_pearson_34": per_axis,
        "mean_per_axis_pearson_34": float(np.mean(list(per_axis.values()))),
        "min_per_axis_pearson_34": float(min(per_axis.values())),
        "per_axis_pearson_100_in_sample":
            {FN[k]: pearson(X100_100[:, k], X100_134[:, k]) for k in range(5)},
        "procrustes_disparity_34_normalised":
            float(np.linalg.norm(A_ @ R - B_) ** 2),
        "relative_frobenius_error_34_no_rotation":
            float(np.linalg.norm(X34_100 - X34_134) / np.linalg.norm(X34_134)),
        "median_per_trait_cosine_34": float(np.median(cos_pt)),
        "min_per_trait_cosine_34": float(np.min(cos_pt)),
        "per_trait_cosine_34": {names[ix34[i]]: cos_pt[i] for i in range(34)},
        "principal_cosines_100chart_vs_134chart": pcos.tolist(),
        "existing_partial_check_for_comparison": {
            "source": "qwen35/build_blog_page.py, reported on "
                      "wiki/pages/zoo/lexicon-secondary-draw.md",
            "principal_cosines_markers_vs_all_134": "0.94 to 0.996"},
        "coords_34_from_100only_chart":
            {names[ix34[i]]: X34_100[i].tolist() for i in range(34)},
        "coords_34_from_134_chart":
            {names[ix34[i]]: X34_134[i].tolist() for i in range(34)},
        "factor_order": FN}

    # =================================================== TEST 2b
    recs = [json.loads(l) for l in
            open(f"{HERE}/analysis/goldberg_only_ratings.jsonl")]
    ok = [r for r in recs if "ratings" in r]
    by = {}
    for r in ok:
        by.setdefault(slugify(r["word"]), []).append(r)
    assert all(len(by.get(n, [])) == 3 for n in names)
    Rm = np.array([[np.mean([rr["ratings"][b] for rr in by[n]]) for b in FACTORS]
                   for n in names])
    agree = float(np.mean([all(by[n][0]["ratings"][b] == by[n][r]["ratings"][b]
                               for b in FACTORS for r in (1, 2)) for n in names]))
    sdrep = float(np.mean([np.std([rr["ratings"][b] for rr in by[n]])
                           for n in names for b in FACTORS]))
    cost = float(sum((r.get("usage", {}).get("cost", 0.0) or 0.0) for r in ok))

    K = np.zeros((100, 5))
    for i, gi in enumerate(ix100):
        t = by_slug[names[gi]]
        K[i, FACTORS.index(t["factor"])] = 1.0 if t["keyed"] == "+" else -1.0

    def axis_block(X, ratings_matrix, rows):
        d = {}
        for k, f in enumerate(FN):
            a = FACTORS.index(CHART_TO_B5[f])
            d[f] = pearson(X[:, k], ratings_matrix[rows, a] if rows is not None
                           else ratings_matrix[:, a])
        return d

    ceil_keying = {FN[k]: pearson(X100_134[:, k],
                                  K[:, FACTORS.index(CHART_TO_B5[FN[k]])])
                   for k in range(5)}
    ceil_rater = axis_block(X100_134, Rm, ix100)
    held_100chart = axis_block(X34_100, Rm, ix34)
    held_134chart = axis_block(X34_134, Rm, ix34)
    rater_vs_keying = {FACTORS[a]: pearson(Rm[ix100, a], K[:, a]) for a in range(5)}

    def summ(d):
        return {"per_axis_pearson": d,
                "mean_r": float(np.mean(list(d.values()))),
                "n_axes_with_r_at_least_0.34": int(sum(1 for v in d.values() if v >= 0.34)),
                "all_positive": bool(all(v > 0 for v in d.values()))}

    def signed(d, sign):
        return float(np.mean([d[f] * sign[f] for f in FN]))

    out["test2b_independent_lexical_ratings"] = {
        "rater": {"model": ok[0]["model"], "temperature": 0, "repeats": 3,
                  "n_words": 134, "n_calls": len(ok),
                  "n_errors": len(recs) - len(ok),
                  "prompt": "the adjective plus textbook one-line definitions of "
                            "the five factors and the -2..+2 scale; no project "
                            "text, no constitution, no zoo, no geometry",
                  "exact_agreement_across_3_repeats_frac_words": agree,
                  "mean_sd_across_repeats": sdrep,
                  "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in ok),
                  "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in ok),
                  "cost_usd": cost,
                  "raw": "qwen35/analysis/goldberg_only_ratings.jsonl"},
        "chart_to_big_five_map": CHART_TO_B5,
        "prereg_expected_signs": PREREG_SIGN,
        "is_the_rater_an_independent_expectation": {
            "what": "the rater's own ratings of the 100 markers against "
                    "Goldberg's keying; if this fails the ratings of the 34 are "
                    "not a usable expectation",
            "per_factor_pearson": rater_vs_keying,
            "mean": float(np.mean(list(rater_vs_keying.values())))},
        "ceiling_goldberg_keying_vs_134chart_100markers": summ(ceil_keying),
        "ceiling_rater_vs_134chart_100markers": summ(ceil_rater),
        "heldout_34_rater_vs_100only_chart": summ(held_100chart),
        "heldout_34_rater_vs_134_chart": summ(held_134chart),
        "heldout_34_mean_r_over_goldberg_keying_ceiling":
            float(np.mean(list(held_100chart.values())) /
                  np.mean(list(ceil_keying.values()))),
        "heldout_34_mean_r_over_rater_ceiling":
            float(np.mean(list(held_100chart.values())) /
                  np.mean(list(ceil_rater.values()))),
        "p05_two_tailed_floor_at_n34": 0.34,
        "prereg_signed": {
            "note": "the pre-registration expected Timidity to correlate "
                    "NEGATIVELY with Emotional Stability; it does not, on the "
                    "ceiling or on the 34. The pre-registration required this be "
                    "reported as a disagreement, not flipped, so both scorings "
                    "are given.",
            "heldout_34_mean_signed_r": signed(held_100chart, PREREG_SIGN),
            "heldout_34_n_axes_clearing_0.34_in_prereg_sign":
                int(sum(1 for f in FN
                        if abs(held_100chart[f]) >= 0.34
                        and np.sign(held_100chart[f]) == PREREG_SIGN[f])),
            "ceiling_goldberg_keying_mean_signed_r": signed(ceil_keying, PREREG_SIGN),
            "half_ceiling_bar": signed(ceil_keying, PREREG_SIGN) / 2},
        "sign_note": {
            "prereg_expected_sign_Timidity_vs_EmotionalStability": -1,
            "observed_on_100_marker_ceiling": ceil_keying["Timidity"],
            "observed_on_heldout_34": held_100chart["Timidity"],
            "project_own_map_has_no_reversal":
                "qwen35/analysis/viz_fa.json#big_five_scale maps "
                "FA_FearfulWithdrawal to EmotionalStability with no sign flag",
            "mechanism":
                "analyse_fa_qwen35.py:order_and_orient orients every factor so "
                "its congruence with its own best Goldberg target is positive, "
                "so the chart axis is already Emotional-Stability-positive; the "
                "name Timidity describes its NEGATIVE pole",
            "chart_axis_positive_pole_top7":
                [names[i] for i in np.argsort(-ch.trait_coords[:, 2])[:7]],
            "chart_axis_negative_pole_bottom7":
                [names[i] for i in np.argsort(-ch.trait_coords[:, 2])[-7:]]},
        "ratings_mean_over_3_repeats":
            {names[i]: {FACTORS[a]: float(Rm[i, a]) for a in range(5)}
             for i in range(134)}}

    # =================================================== TEST 3
    E = np.load(f"{HERE}/analysis/emb_constitutions_mpnet.npy")
    P = np.load(f"{HERE}/analysis/emb_pairs_minilm.npz")
    Ytr, Yte = X100_134, X34_134
    ym = Ytr.mean(0)

    def ridge_arm(Emb, label, model, dim_note):
        Etr, Ete = Emb[ix100], Emb[ix34]
        mu, sd = Etr.mean(0), Etr.std(0)
        sd[sd == 0] = 1.0
        Ztr, Zte = (Etr - mu) / sd, (Ete - mu) / sd
        loo = {str(a): ridge_loo_mse(Ztr, Ytr, a) for a in RIDGE_GRID}
        best = min(RIDGE_GRID, key=lambda a: loo[str(a)])
        # the whole grid's held-out R^2, so the alpha choice is visible and the
        # verdict cannot hang on one point of a selection rule
        grid_r2 = {}
        for a in RIDGE_GRID:
            Wa, ba = ridge_fit(Ztr, Ytr, a)
            grid_r2[str(a)] = r2(Zte @ Wa + ba, Yte, ym)[1]
        W, b = ridge_fit(Ztr, Ytr, best)
        pa, ov = r2(Zte @ W + b, Yte, ym)
        patr, ovtr = r2(Ztr @ W + b, Ytr, ym)
        return {"embedding_model": model, "embedding_dim": int(Emb.shape[1]),
                "text_embedded": dim_note,
                "target": "FAChart().trait_coords, the 134-trait factor chart",
                "fit_on": "the 100 Goldberg markers",
                "predict": "the 34 Lexicon traits",
                "alpha_grid": RIDGE_GRID, "alpha_loo_mse": loo,
                "alpha_loo_note": "leave-one-out by refitting each fold in the "
                                  "dual form; the hat-matrix shortcut is "
                                  "numerically degenerate at p >> n and always "
                                  "picks the smallest alpha",
                "alpha_chosen": best,
                "heldout_r2_overall_at_every_alpha": grid_r2,
                "best_heldout_r2_over_the_whole_grid": max(grid_r2.values()),
                "heldout_r2_per_axis": {FN[k]: float(pa[k]) for k in range(5)},
                "heldout_r2_overall": ov,
                "in_sample_r2_overall_100": ovtr,
                "in_sample_r2_per_axis_100": {FN[k]: float(patr[k]) for k in range(5)},
                "label": label}

    pa_g, ov_g = r2(X34_100, Yte, ym)
    out["test3_text_embedding_baseline"] = {
        "why": "sweep100's deciding test was that the weight-space geometry was "
               "no better than embedding the training text "
               "(wiki/pages/history/sweep100.md; sweep100/results/"
               "text_vs_weights_fa.json, RSA 0.911 between the two centred "
               "matrices, weights mean congruence 0.750 against text's 0.731)",
        "ridge_constitution_to_chart":
            ridge_arm(E, "constitution", "sentence-transformers/all-mpnet-base-v2",
                      "each trait's constitution text, "
                      "constitutions.json#<Trait>.constitution"),
        "ridge_pair_contrast_to_chart":
            ridge_arm(P["contrast"], "pair contrast",
                      "sentence-transformers/all-MiniLM-L6-v2",
                      f"mean embedding of {N_PAIRS_SUBSAMPLE} chosen replies minus "
                      f"the mean of the {N_PAIRS_SUBSAMPLE} matched rejected "
                      "replies, per trait, from data/<slug>.jsonl, subsample seed 0"),
        "comparator_adapter_geometry": {
            "what": "the 34 placed by their exact inner products with the 100 "
                    "(test 2a), scored as a prediction of their 134-chart "
                    "coordinates on the same scale as the ridge arms",
            "r2_per_axis": {FN[k]: float(pa_g[k]) for k in range(5)},
            "r2_overall": ov_g}}

    iu = np.triu_indices(134, 1)
    Gad = double_centre(G)

    def gram_arm(M, label):
        Md = double_centre(M)
        return {"pearson_offdiag_double_centred": pearson(Gad[iu], Md[iu]),
                "pearson_offdiag_raw": pearson(G[iu], M[iu]),
                "what": label}

    gc = gram_arm(cosine_gram(E),
                  "cosine Gram of all-mpnet-base-v2 embeddings of the 134 "
                  "constitution texts")
    gt = gram_arm(cosine_gram(P["contrast"]),
                  "cosine Gram of mean(chosen) - mean(rejected), the sweep100 "
                  "contrast baseline, all-MiniLM-L6-v2")
    out["test3_text_embedding_baseline"]["text_gram_vs_adapter_gram"] = {
        "n_traits": 134, "n_offdiag_pairs": int(len(iu[0])),
        "constitution_mpnet": gc,
        "pairs_chosen_only_minilm":
            gram_arm(cosine_gram(P["chosen"]),
                     "cosine Gram of the mean embedding of the chosen replies only"),
        "pairs_rejected_only_minilm":
            gram_arm(cosine_gram(P["rejected"]),
                     "cosine Gram of the mean embedding of the rejected replies only"),
        "pairs_contrast_chosen_minus_rejected_minilm": gt,
        "n_pairs_subsampled_per_trait": int(P["n_pairs"]), "subsample_seed": 0,
        "max_of_constitution_and_contrast":
            float(max(gc["pearson_offdiag_double_centred"],
                      gt["pearson_offdiag_double_centred"])),
        "sweep100_reference": {
            "chosen_only_rsa": 0.429, "chosen_minus_rejected_rsa": 0.826,
            "source": "wiki/pages/history/sweep100.md, "
                      "sweep100/results/text_baseline.json"}}

    # =================================================== VERDICTS
    t3 = out["test3_text_embedding_baseline"]
    gmax = t3["text_gram_vs_adapter_gram"]["max_of_constitution_and_contrast"]
    r2c = t3["ridge_constitution_to_chart"]["heldout_r2_overall"]
    r2t = t3["ridge_pair_contrast_to_chart"]["heldout_r2_overall"]
    r2max = max(r2c, r2t)
    m = t1["one_to_one_matching"]
    b5 = t1["big_five_congruence"]
    t2a = out["test2a_the_34_in_a_chart_they_did_not_define"]
    t2b = out["test2b_independent_lexical_ratings"]

    out["verdicts"] = {
        "T1a_factors_survive_dropping_the_34": {
            "threshold": "mean absolute one-to-one Tucker congruence >= 0.85 "
                         "(0.95 = identical), matching a permutation",
            "value": m["mean_abs_matched_congruence"],
            "min": m["min_abs_matched_congruence"],
            "pass": bool(m["mean_abs_matched_congruence"] >= 0.85),
            "pass_at_identical_bar": bool(m["min_abs_matched_congruence"] >= 0.95)},
        "T1b_big_five_recovery_not_carried_by_the_34": {
            "threshold": "all five targets taken once by five different factors, "
                         "and no best-per-target congruence moves by more than "
                         "0.10 from the 134 run",
            "all_five_targets_taken_once_per_factor_argmax":
                b5["goldberg100_all_five_targets_taken_once_per_factor_argmax"],
            "n_distinct_factors_across_targets":
                b5["goldberg100_n_distinct_factors_across_targets"],
            "max_abs_delta_vs_134": b5["max_abs_delta_vs_134"],
            "pass": bool(b5["goldberg100_all_five_targets_taken_once_per_factor_argmax"]
                         and b5["max_abs_delta_vs_134"] <= 0.10),
            "note": "fails on both prongs, narrowly and in the direction of "
                    "STRONGER recovery. The F3 factor's own argmax moves from ES "
                    "to E by 0.0024 (E 0.4257 against ES 0.4233) while ES still "
                    "has that factor as its best match, so all five targets are "
                    "still taken by five different factors on the per-target "
                    "reading. The A delta of 0.1038 exceeds 0.10, but the "
                    "decomposition shows the whole rise is Tucker's denominator "
                    "losing the 34 zero rows, not a different solution."},
        "T1c_same_size_null_comparison": {
            "threshold": "retained k at N = 1528 >= 5, and leading centred "
                         "eigenvalue at least 5x the shuffled arm's 1.222",
            "k_goldberg100": t1["n_factors"]["chosen"],
            "k_shuffled": t1["same_size_null_comparison"]["n_factors_chosen_shuffled"],
            "k_permuted": t1["same_size_null_comparison"]["n_factors_chosen_permuted"],
            "ratio_over_shuffled":
                t1["same_size_null_comparison"]["ratio_leading_eigenvalue_goldberg100_over_shuffled"],
            "pass": bool(t1["n_factors"]["chosen"] >= 5 and
                         t1["same_size_null_comparison"]["ratio_leading_eigenvalue_goldberg100_over_shuffled"] >= 5)},
        "T2a_the_34_land_in_the_same_place_either_way": {
            "threshold": "mean per-axis Pearson >= 0.90 and no axis below 0.70",
            "mean": t2a["mean_per_axis_pearson_34"],
            "min": t2a["min_per_axis_pearson_34"],
            "procrustes_disparity": t2a["procrustes_disparity_34_normalised"],
            "pass": bool(t2a["mean_per_axis_pearson_34"] >= 0.90 and
                         t2a["min_per_axis_pearson_34"] >= 0.70)},
        "T2b_prereg_signs": {
            "threshold": "at least 3 of 5 axes with |r| >= 0.34 in the "
                         "pre-registered sign, and mean signed r at least half "
                         "the 100-marker ceiling's",
            "n_axes": t2b["prereg_signed"]["heldout_34_n_axes_clearing_0.34_in_prereg_sign"],
            "mean_signed_r": t2b["prereg_signed"]["heldout_34_mean_signed_r"],
            "half_ceiling_bar": t2b["prereg_signed"]["half_ceiling_bar"],
            "pass": bool(t2b["prereg_signed"]["heldout_34_n_axes_clearing_0.34_in_prereg_sign"] >= 3
                         and t2b["prereg_signed"]["heldout_34_mean_signed_r"]
                         >= t2b["prereg_signed"]["half_ceiling_bar"])},
        "T2b_sign_corrected": {
            "threshold": "the same test with the Timidity axis scored "
                         "positively on Emotional Stability, which is the "
                         "project's own convention",
            "n_axes": t2b["heldout_34_rater_vs_100only_chart"]["n_axes_with_r_at_least_0.34"],
            "mean_r_34": t2b["heldout_34_rater_vs_100only_chart"]["mean_r"],
            "mean_r_100_marker_ceiling":
                t2b["ceiling_goldberg_keying_vs_134chart_100markers"]["mean_r"],
            "ratio_to_ceiling": t2b["heldout_34_mean_r_over_goldberg_keying_ceiling"],
            "pass": bool(t2b["heldout_34_rater_vs_100only_chart"]["n_axes_with_r_at_least_0.34"] >= 3
                         and t2b["heldout_34_rater_vs_100only_chart"]["mean_r"]
                         >= t2b["ceiling_goldberg_keying_vs_134chart_100markers"]["mean_r"] / 2)},
        "T3_the_geometry_is_the_text_passed_through": {
            "threshold": "off-diagonal double-centred Pearson >= 0.80 on the "
                         "larger of the constitution and contrast Grams, OR "
                         "held-out overall R^2 >= 0.50 from text to chart",
            "max_gram_pearson": gmax,
            "best_heldout_r2": r2max,
            "pass": bool(gmax >= 0.80 or r2max >= 0.50)},
        "T3_the_geometry_adds_structure": {
            "threshold": "off-diagonal Pearson <= 0.50 on BOTH Grams AND "
                         "held-out overall R^2 <= 0.20",
            "constitution_gram_pearson": gc["pearson_offdiag_double_centred"],
            "contrast_gram_pearson": gt["pearson_offdiag_double_centred"],
            "best_heldout_r2": r2max,
            "pass": bool(gc["pearson_offdiag_double_centred"] <= 0.50 and
                         gt["pearson_offdiag_double_centred"] <= 0.50 and
                         r2max <= 0.20)}}
    out["verdicts"]["T3_partial"] = {
        "pass": bool(not out["verdicts"]["T3_the_geometry_is_the_text_passed_through"]["pass"]
                     and not out["verdicts"]["T3_the_geometry_adds_structure"]["pass"]),
        "threshold": "anything between the two bars is reported as partial"}
    out["spend"] = {"rater_usd": cost,
                    "embeddings_usd": 0.0,
                    "note": "embeddings ran locally on CPU from the cached "
                            "sentence-transformers models; no GPU, no Modal",
                    "total_usd": cost}
    json.dump(out, open(f"{HERE}/analysis/goldberg_only.json", "w"), indent=1)
    print(json.dumps(out["verdicts"], indent=1))
    print("spend", json.dumps(out["spend"]))
    print("wrote analysis/goldberg_only.json")


if __name__ == "__main__":
    main()
