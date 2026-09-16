#!/usr/bin/env python3
"""Column space of the emergent-misalignment arms.

Reads results/column_space_em.npz (column_space_em_on_modal.py) and writes
analysis/em_column_space.json.  A focused subset of what
analyse_column_space_sorh.py reports, so the two pages are comparable number for
number: each arm's top-k output subspace against the 134 stage-one adapters,
against the four alignment adapters, against the zoo's generic output subspace
and the stage-two register, the reference bands measured inside this same
matrix, and the arm-versus-arm block.

`col_wtd_k` is the fraction of adapter i's top-k delta energy lying inside
adapter j's top-k column space; it is not symmetric and both orientations are
given.  `col_unw_k` is the mean squared cosine of the principal angles.
"""
import json
import os
import sys

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
KS = [1, 4, 8, 16, 64]
HEAD_KS = [1, 8, 64]
REFS = ["G1_mean", "G1_stack", "G2_mean", "G2_stack"]

# quoted verbatim from analysis/column_space.json via
# wiki/pages/behaviour/reward-hacks-column-space.md
PUBLISHED = {
    "same_trait_cross_seed_col_wtd_k8": 0.5846385056208819,
    "diff_trait_cross_seed_col_wtd_k8": 0.12880060417597922,
    "diff_trait_same_seed_col_wtd_k8": 0.13295928198239396,
    "random_null_col_k8": 0.002293402777777778,
}
SORH = {  # analysis/column_space_sorh.json, for the side-by-side
    "hack_c93_vs_zoo_trait_k8": 0.01864560989879112,
    "hack_c93_vs_G1_stack_k8": 0.027653501381864773,
    "hack_c93_vs_sycophantic_k8": 0.018155915064853617,
    "hack_vs_control_col_wtd_k8": 0.15385211790911854,
    "hack_vs_control_frobenius_cosine": 0.23112746648382296,
    "held_out_trait_vs_G1_stack_k8": 0.3784547236738726,
    "weakest_of_forty_vs_G1_stack_k8": 0.2062613546103239,
}


def stat(v):
    v = np.asarray(v, dtype=float).ravel()
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
            "median": float(np.median(v))}


def main():
    z = np.load(f"{Q}/results/column_space_em.npz", allow_pickle=True)
    names = [str(x) for x in z["names"]]
    tags = np.array([str(x) for x in z["tags"]])
    idx = {n: i for i, n in enumerate(names)}
    zoo = np.where(tags == "zoo")[0]
    align = np.where(tags == "align")[0]
    probes = [n for n, t in zip(names, tags) if t in ("em", "diff")]
    C = {k: z[f"col_wtd_k{k}"] for k in KS}
    Cu = {k: z[f"col_unw_k{k}"] for k in KS}

    n_ent = int(z["frob_approx_ent_wide"].shape[0])
    FE = z["frob_exact_wide"]
    FAE = z["frob_approx_ent_wide"]
    nrm = np.zeros(len(names))
    nrm[:n_ent] = np.sqrt(np.clip(np.diag(FAE), 1e-30, None))
    n_arm = int(z["n_arm"]) if "n_arm" in z else len([t for t in tags if t == "em"])
    for j in range(len(names) - n_ent):
        nrm[n_ent + j] = np.sqrt(max(FE[n_arm + j, n_ent + j], 1e-30))
    prow = {idx[p]: r for r, p in enumerate(probes)}
    FCp = FE / (nrm[[idx[p] for p in probes]][:, None] * nrm[None, :])

    def fc(a, b):
        return float(FCp[prow[idx[a]], idx[b]])

    zz = np.ix_(zoo, zoo)
    offz = ~np.eye(len(zoo), dtype=bool)
    out = {
        "what": "column space (span of B in output space) of the three "
                "emergent-misalignment SFT arms, their checkpoints and the "
                "bad-minus-good difference delta, against the 134 stage-one "
                "personality adapters, the four alignment adapters, the zoo's "
                "generic output subspace and the stage-two register",
        "produced_by": ["column_space_em_on_modal.py", "analyse_em_colspace.py"],
        "sources": {"npz": "results/column_space_em.npz",
                    "published_reference_classes": "analysis/column_space.json",
                    "sorh_comparison": "analysis/column_space_sorh.json"},
        "published_reference_classes": PUBLISHED,
        "sorh_published_values": SORH,
        "meta": {"n_adapters_in_block": len(names), "n_zoo": int(len(zoo)),
                 "n_align": int(len(align)), "n_probes": len(probes),
                 "n_modules": int(z["n_modules"]), "n_wide_modules": int(z["n_wide"]),
                 "n_narrow_modules": int(z["n_narrow"]),
                 "probe_names": probes,
                 "align_names": [names[i] for i in align]},
        "reference_bands_measured_here": {
            "zoo_diff_trait_same_seed": {
                f"k{k}": {"col_wtd": stat(C[k][zz][offz]),
                          "col_unw": stat(Cu[k][zz][offz])} for k in HEAD_KS},
            "align_vs_zoo": {f"k{k}": {"col_wtd": stat(C[k][np.ix_(align, zoo)])}
                             for k in HEAD_KS},
        },
        "vs_134_stage_one_adapters": {
            p: {f"k{k}": {"mean_over_134": float(C[k][idx[p], zoo].mean()),
                          "max_over_134": float(C[k][idx[p], zoo].max()),
                          "argmax_trait": names[zoo[int(np.argmax(C[k][idx[p], zoo]))]],
                          "col_unw_mean": float(Cu[k][idx[p], zoo].mean())}
                for k in HEAD_KS} for p in probes},
        "vs_alignment_adapters": {
            p: {names[j]: {f"k{k}": float(C[k][idx[p], j]) for k in HEAD_KS}
                for j in align} for p in probes},
        "vs_generic_and_register": {},
        "arm_vs_arm": {},
        "frobenius": {"norms": {p: float(nrm[idx[p]]) for p in probes},
                      "zoo_norm_mean": float(nrm[zoo].mean())},
    }
    for rname in REFS:
        e = {}
        for k in HEAD_KS:
            wt = z[f"ref_{rname}_wtd_k{k}"]
            wt1 = z[f"ref_{rname}_wtd_seed1_k{k}"]
            wt2 = z[f"ref_{rname}_wtd_stage2_k{k}"]
            e[f"k{k}"] = {"probes": {p: float(wt[idx[p]]) for p in probes},
                          "band_134_stage_one": stat(wt[zoo]),
                          "band_40_seed1_stage_one": stat(wt1),
                          "band_134_stage_two": stat(wt2),
                          "band_4_alignment": stat(wt[align])}
        out["vs_generic_and_register"][rname] = e

    fin = [p for p in probes if p.endswith("_final")]
    for a in fin:
        for b in fin:
            if a >= b:
                continue
            out["arm_vs_arm"][f"{a}_vs_{b}"] = {
                **{f"k{k}": {"col_wtd_a_in_b": float(C[k][idx[a], idx[b]]),
                             "col_wtd_b_in_a": float(C[k][idx[b], idx[a]]),
                             "col_unw": float(Cu[k][idx[a], idx[b]])}
                   for k in HEAD_KS},
                "frobenius_cosine": fc(a, b)}
    traj = {}
    for p in probes:
        traj[p] = {"vs_zoo_mean_col_wtd_k8": float(C[8][idx[p], zoo].mean()),
                   "vs_G1_stack_wtd_k8": float(z["ref_G1_stack_wtd_k8"][idx[p]]),
                   "frobenius_norm_wide": float(nrm[idx[p]])}
    out["trajectory"] = traj

    pth = f"{Q}/analysis/em_column_space.json"
    json.dump(out, open(pth, "w"), indent=1)
    print("wrote", pth)
    b = out["reference_bands_measured_here"]["zoo_diff_trait_same_seed"]["k8"]["col_wtd"]
    print(f"\nreference (two unrelated zoo traits, k=8, weighted): {b['mean']:.6f}")
    print(f"published same-trait cross-seed: {PUBLISHED['same_trait_cross_seed_col_wtd_k8']:.6f}")
    for p in probes:
        v = out["vs_134_stage_one_adapters"][p]["k8"]
        g = out["vs_generic_and_register"]["G1_stack"]["k8"]["probes"][p]
        print(f"  {p:<20} vs 134 mean {v['mean_over_134']:.6f} max {v['max_over_134']:.6f} "
              f"({v['argmax_trait']})  vs G1_stack {g:.6f}")


if __name__ == "__main__":
    sys.exit(main())
