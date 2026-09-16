#!/usr/bin/env python3
"""EXPERIMENT 4: how many factors describe stage two, and on which Gram?

Two questions, no GPU.

1. INTERPRET k = 7 against k = 5.  Horn's parallel analysis on the stage-two
   Gram retained 7 factors at N = 1528 and 1 at N = 150
   (results/fa_qwen35_stage2.json#n_factors), where stage one retained 9.  The
   project extracted both k = 5 (the Big Five hypothesis) and k = 7.  This reads
   the top loaders and the Tucker congruence with the five Goldberg keying
   targets for each, and asks whether the two extra factors are new content or a
   split of the five.

2. REMOVE THE SHARED DIRECTION FIRST.  The factor analysis double-centres the
   correlation matrix, which removes each variable's mean but not the shared
   direction from the deltas.  build_gram_stage2_noshared.py removes it exactly
   (G_res = G - (G1)(1'G)/(1'G1)) and analyse_fa_qwen35.py was rerun on that,
   producing results/fa_qwen35_stage2_noshared.json.  This compares the two.

Output: analysis/stage2_factors_choice.json
"""
import itertools
import json
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
R = f"{Q}/results"


def tucker(A, B):
    return (A.T @ B) / np.outer(np.linalg.norm(A, axis=0), np.linalg.norm(B, axis=0))


def best_match(C):
    """Best one-to-one matching of columns by |congruence| (as in analyse_stage2_structure)."""
    k = C.shape[0]
    best = None
    for perm in itertools.permutations(range(C.shape[1]), k):
        s = sum(abs(C[i, perm[i]]) for i in range(k))
        if best is None or s > best[0]:
            best = (s, perm)
    _, perm = best
    return [{"row": i, "col": int(perm[i]), "congruence": float(C[i, perm[i]])}
            for i in range(k)]


def read_solution(fa, key):
    """Congruence table, top loaders and simple-structure diagnostics for one solution."""
    s = fa["solutions"][key]
    labels = fa["targets"]["labels"]
    names = fa["trait_order"]
    L = np.array(s["loadings"]["oblimin"])
    cong = np.array(s["congruence_oblimin"])
    k = cong.shape[0]
    per_factor = []
    for i in range(k):
        j = int(np.argmax(np.abs(cong[i])))
        order = np.argsort(L[:, i])
        per_factor.append({
            "factor": i,
            "best_target": labels[j], "congruence": float(cong[i][j]),
            "all_targets": {labels[m]: round(float(cong[i][m]), 3)
                            for m in range(len(labels))},
            "ss_loading": s["ss_loadings"]["oblimin"][i],
            "top_pos": [(names[t], round(float(L[t, i]), 3)) for t in order[::-1][:6]],
            "top_neg": [(names[t], round(float(L[t, i]), 3)) for t in order[:6]]})
    # which of the five Big Five targets are claimed at all, and by how many factors
    big5 = [l for l in labels if l != "Eval"]
    claimed = {}
    for l in big5:
        j = labels.index(l)
        claimed[l] = {"best_factor": int(np.argmax(np.abs(cong[:, j]))),
                      "best_congruence": float(cong[np.argmax(np.abs(cong[:, j])), j])}
    # simple structure: mean |loading| ratio of the largest to the second largest
    Lab = np.abs(L)
    srt = np.sort(Lab, axis=1)[:, ::-1]
    hoffman = float(np.mean((srt.sum(1) ** 2) / np.maximum((srt ** 2).sum(1), 1e-12)))
    return {"k": k,
            "paf_converged": s["paf_converged"], "n_heywood": s["n_heywood"],
            "reduced_eigenvalues": s["reduced_eigenvalues"][:k],
            "ss_loadings_oblimin": s["ss_loadings"]["oblimin"],
            "communality_mean": float(np.mean(s["communalities"])),
            "per_factor": per_factor,
            "big_five_targets_claimed": claimed,
            "n_distinct_targets_taken": len({p["best_target"] for p in per_factor}),
            "mean_complexity_hoffman": hoffman,
            "_L": L, "_names": names}


def parallel_summary(fa, which="parallel_analysis_centred"):
    pa = fa["n_factors"][which]
    return {"observed_unreduced_top8": pa["observed_unreduced"][:8],
            "observed_reduced_top8": pa["observed_reduced"][:8],
            "k_by_N": {str(g["N"]): {"unreduced_95pct": g["k_unreduced_95pct"],
                                     "reduced_95pct": g["k_reduced_95pct"]}
                       for g in pa["grid"]}}


def main():
    fa2 = json.load(open(f"{R}/fa_qwen35_stage2.json"))
    out = {"what": ("k = 5 against k = 7 for stage two, and the same factor "
                    "analysis with the stage-two shared direction projected out "
                    "of the deltas before factoring."),
           "sources": {"stage2": "results/fa_qwen35_stage2.json",
                       "stage2_noshared": "results/fa_qwen35_stage2_noshared.json",
                       "stage1": "results/fa_qwen35.json",
                       "gram_noshared": "results/gram_stage2_noshared.npz"}}

    sols = {}
    for tag, fa in (("stage2", fa2),):
        for k in ("centred_k5", "centred_k7"):
            sols[f"{tag}|{k}"] = read_solution(fa, k)
    out["parallel_analysis"] = {"stage2": parallel_summary(fa2),
                                "stage2_chosen": fa2["n_factors"]["chosen"],
                                "stage2_rationale": fa2["n_factors"]["chosen_rationale"]}

    fa1 = json.load(open(f"{R}/fa_qwen35.json"))
    out["parallel_analysis"]["stage1_chosen"] = fa1["n_factors"]["chosen"]
    out["parallel_analysis"]["stage1"] = parallel_summary(fa1)

    # k5 -> k7 correspondence: does k7 split the five, or find new content?
    L5, L7 = sols["stage2|centred_k5"]["_L"], sols["stage2|centred_k7"]["_L"]
    C57 = tucker(L5, L7)
    out["k5_vs_k7"] = {
        "tucker_rows_k5_cols_k7": np.round(C57, 3).tolist(),
        "best_matching_k5_into_k7": best_match(C57),
        "each_k7_factor_best_k5_partner": [
            {"k7_factor": j, "best_k5": int(np.argmax(np.abs(C57[:, j]))),
             "congruence": float(C57[np.argmax(np.abs(C57[:, j])), j])}
            for j in range(C57.shape[1])]}

    # the no-shared-direction rerun
    p = f"{R}/fa_qwen35_stage2_noshared.json"
    if os.path.exists(p):
        fan = json.load(open(p))
        for k in ("centred_k5", "centred_k7"):
            if k in fan["solutions"]:
                sols[f"stage2_noshared|{k}"] = read_solution(fan, k)
        out["parallel_analysis"]["stage2_noshared"] = parallel_summary(fan)
        out["parallel_analysis"]["stage2_noshared_chosen"] = fan["n_factors"]["chosen"]
        # factor-by-factor congruence with the same-k solution on the full Gram
        for k in ("centred_k5", "centred_k7"):
            a, b = f"stage2|{k}", f"stage2_noshared|{k}"
            if a in sols and b in sols:
                C = tucker(sols[a]["_L"], sols[b]["_L"])
                out.setdefault("with_vs_without_shared", {})[k] = {
                    "tucker_rows_with_cols_without": np.round(C, 3).tolist(),
                    "best_matching": best_match(C)}
    else:
        out["parallel_analysis"]["stage2_noshared"] = f"MISSING {p}"

    # the Gram itself, before and after
    z0 = np.load(f"{R}/gram_stage2.npz", allow_pickle=True)
    z1 = np.load(f"{R}/gram_stage2_noshared.npz", allow_pickle=True)
    G0, G1 = np.array(z0["G"]), np.array(z1["G"])
    c = lambda G: G / np.outer(np.sqrt(np.diag(G)), np.sqrt(np.diag(G)))
    iu = np.triu_indices(G0.shape[0], 1)
    C0, C1 = c(G0)[iu], c(G1)[iu]
    out["gram"] = {
        "trace_share_removed": float(1 - np.trace(G1) / np.trace(G0)),
        "offdiag_cos_before": {"mean": float(C0.mean()), "sd": float(C0.std())},
        "offdiag_cos_after": {"mean": float(C1.mean()), "sd": float(C1.std())},
        "corr_before_after": float(np.corrcoef(C0, C1)[0, 1])}

    for k in list(sols):
        sols[k].pop("_L"); sols[k].pop("_names")
    out["solutions"] = sols

    # ---- the decision -------------------------------------------------------
    def mean_abs_big5(sol):
        return float(np.mean([abs(p["congruence"]) for p in sol["per_factor"]
                              if p["best_target"] != "Eval"]))
    out["decision_inputs"] = {
        k: {"n_distinct_targets_taken": v["n_distinct_targets_taken"],
            "mean_abs_congruence_of_non_Eval_factors": mean_abs_big5(v),
            "n_factors_whose_best_target_is_Eval":
                sum(1 for p in v["per_factor"] if p["best_target"] == "Eval"),
            "communality_mean": v["communality_mean"],
            "mean_complexity_hoffman": v["mean_complexity_hoffman"]}
        for k, v in sols.items()}

    out["decision"] = (
        "Present k = 5 on the FULL stage-two Gram (results/gram_stage2.npz), the same "
        "solution the rest of the site uses for stage one. Two reasons, both measured "
        "here. (1) The shared direction makes no difference to the factors: matching the "
        "loading matrices with and without it gives Tucker 0.990 to 0.99996 at k = 5 and "
        "0.996 to 0.99996 at k = 7, the parallel-analysis count is 7 at N = 1528 either "
        "way, and the top reduced eigenvalues move from 3.97, 2.98, 2.46, 2.24, 2.05 to "
        "3.97, 2.98, 2.46, 2.20, 1.99. The factor analysis double-centres the correlation "
        "matrix, which already removes the shared component's effect on the correlations, "
        "so projecting it out of the deltas first buys nothing. Publishing a second Gram "
        "would imply a difference that is not there. (2) k = 7 is not more interpretable. "
        "Horn's parallel analysis retains 7 at N = 1528 and that is reported, but the "
        "seven factors take only six distinct Big Five targets, one of them (factor 1: "
        "artistic, inspired, artful, creative against bashful, fearful, insecure) has the "
        "general evaluative axis as its best target, Conscientiousness is claimed twice, "
        "mean absolute congruence with a Big Five target falls from 0.492 to 0.454, and "
        "mean factor complexity rises from 3.21 to 4.23. The three extra factors are a "
        "decomposition of the k = 5 Emotional Stability factor (k7 factors 3, 5 and 6 all "
        "match k5 factor 2, at -0.587, +0.676 and +0.231), not new content. "
        "What k = 7 does buy, and what should be said beside the k = 5 table: it separates "
        "a clean orderliness factor (systematic, organized, prompt, neat against quiet, "
        "untalkative, withdrawn; k7 factor 6) out of the muddled k = 5 Conscientiousness "
        "factor, which pits hostility against disorganisation rather than order against "
        "disorder.")
    pth = f"{Q}/analysis/stage2_factors_choice.json"
    json.dump(out, open(pth, "w"), indent=1)

    for key, v in sols.items():
        print("=" * 78)
        print(f"{key}: k={v['k']} heywood {v['n_heywood']} communality "
              f"{v['communality_mean']:.3f} targets taken "
              f"{v['n_distinct_targets_taken']}")
        for p in v["per_factor"]:
            print(f"  f{p['factor']} ss {p['ss_loading']:.2f} best {p['best_target']} "
                  f"{p['congruence']:+.3f}  +: {[t for t, _ in p['top_pos'][:5]]}")
            print(f"      -: {[t for t, _ in p['top_neg'][:5]]}   all {p['all_targets']}")
    print("=" * 78)
    print("parallel analysis k at N=1528 (unreduced/reduced):")
    for tag in ("stage1", "stage2", "stage2_noshared"):
        pa = out["parallel_analysis"].get(tag)
        if isinstance(pa, dict):
            print(f"  {tag}: {pa['k_by_N'].get('1528')}   top reduced eig "
                  f"{[round(x, 2) for x in pa['observed_reduced_top8']]}")
    print(f"gram: removed {out['gram']['trace_share_removed']:.4f} of the trace; "
          f"off-diag cos {out['gram']['offdiag_cos_before']['mean']:+.4f} -> "
          f"{out['gram']['offdiag_cos_after']['mean']:+.4f}, "
          f"corr {out['gram']['corr_before_after']:.4f}")
    print(f"wrote {pth}")


if __name__ == "__main__":
    main()
