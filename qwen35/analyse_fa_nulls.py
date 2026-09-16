#!/usr/bin/env python3
"""Compare the factor analysis of the two matched null arms against the real
stage-one factor analysis.

Inputs (all written by analyse_fa_qwen35.py, which was run once per arm with
PC_GRAM_NPZ / PC_FA_TAG):

  results/fa_qwen35.json                real stage one, 134 traits
  results/fa_qwen35_null_shuffled.json  data_null_shuffled_p100_matched, 100 traits
  results/fa_qwen35_null_permuted.json  data_null_permuted_p100_matched, 100 traits

For each arm this records: the parallel-analysis retained counts (the chosen k
and the whole N grid), the k=5 reduced eigenvalues, the oblimin sums of squares,
the best Big Five target congruence per factor and whether the five targets are
each taken once, the mean absolute oblimin congruence with the five targets, and
how many centred eigenvalues exceed the N = 1528 and N = 150 parallel-analysis
nulls (counted exactly as wiki/tools/gen_scree_svg.py counts them: over the 12
stored null entries, not by first crossing).

It also computes the Tucker congruence between each null arm's centred_k5
oblimin loadings and the real stage-one centred_k5 oblimin loadings, restricted
to the 100 trait slugs the arms share, with the best one-to-one matching of
factors (tucker() and best_match() imported from analyse_stage2_structure.py).
For the permuted arm the same congruence is computed a second time after
relabelling each row by the trait whose pairs it actually trained on
(nulls_manifest.json#permutation_dst_to_src): the permuted arm is the real
corpus under a derangement of the labels, so the source-relabelled comparison
asks whether the arm found the real structure in the wrong place.

Usage:  ~/cartovenv/bin/python analyse_fa_nulls.py
Writes: analysis/fa_nulls.json
"""
import json
import os

import numpy as np

from analyse_stage2_structure import tucker, best_match

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
A = os.path.join(HERE, "analysis")
ARMS = [("real", ""), ("shuffled", "_null_shuffled"), ("permuted", "_null_permuted")]
NS = [1528, 150]


def above(vals, null):
    """gen_scree_svg.above(): count positions where observed exceeds the null,
    over the 12 null entries stored in the json."""
    return sum(1 for v, n in zip(vals, null) if v > n)


def arm_summary(d):
    nf = d["n_factors"]
    labels = d["targets"]["labels"]          # E A C ES I Eval
    s5 = d["solutions"]["centred_k5"]
    C = np.array(s5["congruence_oblimin"])[:, :5]
    best_j = [int(np.argmax(np.abs(C[i]))) for i in range(C.shape[0])]
    o = {
        "n_traits": d["setup"]["n_traits"],
        "n_primary": d["setup"]["n_primary"],
        "n_lexicon": d["setup"]["n_lexicon"],
        "n_factors_chosen": nf["chosen"],
        "reference_N": nf["reference_N"],
        "kaiser_uncentred_eig_gt_1": nf["kaiser_uncentred_eig_gt_1"],
        "kaiser_centred_eig_gt_1": nf["kaiser_centred_eig_gt_1"],
        "reduced_eig_gt_1_uncentred": nf["reduced_eig_gt_1_uncentred"],
        "solutions_present": sorted(d["solutions"]),
        "parallel_analysis_grid": {},
        "centred_eigenvalues_top12": d["correlation_matrix"]["centred_eigenvalues"][:12],
        "reduced_eigenvalues_centred_k5_top12": s5["reduced_eigenvalues"][:12],
        "ss_loadings_oblimin_centred_k5": s5["ss_loadings"]["oblimin"],
        "communality_mean_centred_k5": float(np.mean(s5["communalities"])),
        "congruence_oblimin_centred_k5": np.array(s5["congruence_oblimin"]).tolist(),
        "best_big_five_target_per_factor": [
            {"factor": i, "target": labels[best_j[i]],
             "congruence": float(C[i, best_j[i]])} for i in range(C.shape[0])],
        "best_congruence_per_big_five_target": {
            labels[a]: float(np.abs(C[:, a]).max()) for a in range(5)},
        "all_five_targets_taken_once": sorted(best_j) == list(range(5)),
        "n_distinct_targets_taken": len(set(best_j)),
        "mean_abs_congruence_with_five_targets": float(np.abs(C).mean()),
        "max_abs_congruence_with_five_targets": float(np.abs(C).max()),
        "n_targets_clearing_0.85": int(sum(np.abs(C[:, a]).max() >= 0.85 for a in range(5))),
    }
    for tag, key, nullkey in [("uncentred", "parallel_analysis_uncentred", None),
                              ("centred", "parallel_analysis_centred", None)]:
        pa = nf[key]
        o["parallel_analysis_grid"][tag] = [
            {"N": g["N"], "k_unreduced_95pct": g["k_unreduced_95pct"],
             "k_reduced_95pct": g["k_reduced_95pct"]} for g in pa["grid"]]
    pac = nf["parallel_analysis_centred"]
    o["eigenvalues_above_null"] = {}
    for N in NS:
        g = next(x for x in pac["grid"] if x["N"] == N)
        o["eigenvalues_above_null"][str(N)] = {
            "n_above_unreduced_95pct": above(pac["observed_unreduced"],
                                             g["null_unreduced_95pct"]),
            "n_above_reduced_95pct": above(pac["observed_reduced"],
                                           g["null_reduced_95pct"]),
            "k_unreduced_95pct_first_crossing": g["k_unreduced_95pct"],
            "k_reduced_95pct_first_crossing": g["k_reduced_95pct"],
            "null_unreduced_95pct_top3": g["null_unreduced_95pct"][:3],
            "counted_over_n_entries": len(g["null_unreduced_95pct"])}
    return o


def loadings(d):
    L = np.array(d["solutions"]["centred_k5"]["loadings"]["oblimin"])
    return L, list(d["trait_slug"])


def main():
    data = {}
    for name, tag in ARMS:
        p = f"{R}/fa_qwen35{tag}.json"
        assert os.path.exists(p), f"missing {p}"
        data[name] = json.load(open(p))

    out = {"what": __doc__.strip().split("\n\n")[0],
           "inputs": {n: f"results/fa_qwen35{t}.json" for n, t in ARMS},
           "arm_gram": {"real": "results/gram_sweep.npz",
                        "shuffled": "results/gram_data_null_shuffled_p100_matched.npz",
                        "permuted": "results/gram_data_null_permuted_p100_matched.npz"},
           "note_on_p": "The real arm has 134 variables (100 Goldberg markers + 34 "
                        "Lexicon traits); both null arms have 100 (the Goldberg "
                        "markers only). A correlation matrix has trace p, so the "
                        "eigenvalues of the two are not on the same scale; each "
                        "arm's parallel-analysis null is generated at its own p.",
           "arms": {n: arm_summary(data[n]) for n, _ in ARMS}}

    # ---- Tucker congruence of each null arm's factors against the real ones ---
    Lr, sr = loadings(data["real"])
    perm = json.load(open(f"{HERE}/nulls_manifest.json"))["permutation_dst_to_src"]
    out["tucker_vs_real_stage_one"] = {}
    for name in ("shuffled", "permuted"):
        Ln, sn = loadings(data[name])
        common = [t for t in sn if t in sr]
        Ra = np.array([Lr[sr.index(t)] for t in common])
        Na = np.array([Ln[sn.index(t)] for t in common])
        Ct = tucker(Ra, Na)                       # rows real, cols null
        bm = best_match(Ct)
        rec = {"n_common_traits": len(common),
               "rows": "real stage-one centred_k5 oblimin factors (F1..F5)",
               "cols": f"{name} centred_k5 oblimin factors (F1..F5)",
               "tucker_matrix": np.round(Ct, 4).tolist(),
               "best_matching": bm,
               "mean_abs_best_matched_congruence":
                   float(np.mean([abs(b["congruence"]) for b in bm])),
               "max_abs_congruence": float(np.abs(Ct).max())}
        if name == "permuted":
            # relabel each permuted row by the trait whose pairs it trained on
            src = [perm[t] for t in common]
            keep = [i for i, t in enumerate(src) if t in sr]
            Ra2 = np.array([Lr[sr.index(src[i])] for i in keep])
            Na2 = Na[keep]
            Ct2 = tucker(Ra2, Na2)
            bm2 = best_match(Ct2)
            rec["source_relabelled"] = {
                "what": "permuted rows re-identified by nulls_manifest.json"
                        "#permutation_dst_to_src, i.e. matched to the real trait "
                        "whose preference pairs the adapter actually trained on",
                "n_rows_used": len(keep),
                "tucker_matrix": np.round(Ct2, 4).tolist(),
                "best_matching": bm2,
                "mean_abs_best_matched_congruence":
                    float(np.mean([abs(b["congruence"]) for b in bm2])),
                "max_abs_congruence": float(np.abs(Ct2).max())}
        out["tucker_vs_real_stage_one"][name] = rec

    os.makedirs(A, exist_ok=True)
    json.dump(out, open(f"{A}/fa_nulls.json", "w"), indent=1)

    for n, _ in ARMS:
        a = out["arms"][n]
        print(f"{n:9s} p={a['n_traits']:3d} chosen k={a['n_factors_chosen']:2d} "
              f"above N=1528 {a['eigenvalues_above_null']['1528']['n_above_unreduced_95pct']:2d} "
              f"/ N=150 {a['eigenvalues_above_null']['150']['n_above_unreduced_95pct']:2d}  "
              f"SS {['%.2f' % x for x in a['ss_loadings_oblimin_centred_k5']]}  "
              f"best per target {a['best_congruence_per_big_five_target']}  "
              f"all five taken once {a['all_five_targets_taken_once']}  "
              f"mean|phi| {a['mean_abs_congruence_with_five_targets']:.3f}")
    for n, r in out["tucker_vs_real_stage_one"].items():
        print(f"{n} vs real: matched congruences "
              f"{[round(b['congruence'], 3) for b in r['best_matching']]}")
        if "source_relabelled" in r:
            print("  source-relabelled: "
                  f"{[round(b['congruence'], 3) for b in r['source_relabelled']['best_matching']]}")
    print("wrote analysis/fa_nulls.json")


if __name__ == "__main__":
    main()
