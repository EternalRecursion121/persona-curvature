#!/usr/bin/env python3
"""The cross-seed floor, computed identically for every arm that claims one.

The published figure (+0.01659 same-trait, 40/40 top-1, slope 0.0264 on the
within-run block) came from `cross_gram_full_root_x_data_null_seedpaired_s40`,
whose 40 adapters were trained under PLAIN sigmoid DPO while the 134 they are
compared against carried an auxiliary SFT term and kl_coef 0.001.  That makes it
a seed-AND-objective floor.  `..._s40_matched` is the same 40 traits, the same
byte-identical corpora and the same seed 1, retrained at the zoo's objective, so
the difference between the two arms is the objective and nothing else.

Everything here is a function of ONE matrix, X[i,j] = <dW_i^A, dW_j^B>, divided
into cosines by the exact Frobenius norms the same job computed.  No sketch, no
projection, no approximation.

usage:
    python analyse_crossseed.py results/cross_gram_full_root_x_....npz
    python analyse_crossseed.py A.npz B.npz     # prints both, side by side
"""
import json
import os
import sys

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
EXCL = {"Lexicon"}          # a placeholder factor, not a Big Five one


def labels():
    out = {}
    for f in ("traits_primary.json", "traits_secondary.json"):
        p = f"{Q}/{f}"
        if os.path.exists(p):
            for r in json.load(open(p)):
                out[r["trait"].lower().replace(" ", "_").replace("-", "_")] = \
                    (r["factor"], r["keyed"])
    return out


WITHIN = os.environ.get("PC_WITHIN", f"{Q}/results/gram_sweep.npz")


def within_run():
    """The seed-0 within-run cosine block, for the attenuation regression.

    Defaults to the stage-1 sweep Gram.  PC_WITHIN can point at any Gram in
    either format (G/names from gram_on_modal.py, or a square X/names_a from
    cross_gram_full_on_modal.py run with subdir-a == subdir-b), e.g. the stage-2
    SFT LoRAs' own 134 x 134 block.
    """
    z = np.load(WITHIN, allow_pickle=True)
    G = np.array(z["G"] if "G" in z else z["X"])
    n = [str(x) for x in (z["names"] if "names" in z else z["names_a"])]
    d = np.sqrt(np.diag(G))
    return {m: i for i, m in enumerate(n)}, G / np.outer(d, d)


def arm(path, L):
    z = np.load(path, allow_pickle=True)
    X = np.array(z["X"])
    na = [str(x) for x in z["names_a"]]
    nb = [str(x) for x in z["names_b"]]
    C = X / np.outer(np.array(z["norms_a"]), np.array(z["norms_b"]))
    ia = {m: i for i, m in enumerate(na)}

    same, sfk, sfo, diff = [], [], [], []
    for j, n in enumerate(nb):
        fj, kj = L[n]
        for m, i in ia.items():
            v = float(C[i, j])
            if m == n:
                same.append(v)
            elif L[m][0] == fj and fj not in EXCL:
                (sfk if L[m][1] == kj else sfo).append(v)
            else:
                diff.append(v)

    # top-1: is each seed-1 adapter its own nearest neighbour among all of A?
    ranks = [int(1 + np.sum(C[:, j] > C[ia[n], j])) for j, n in enumerate(nb)]

    # the separation test: does the WEAKEST same-trait cosine still beat the
    # STRONGEST cross-trait one?  This is the claim that has no overlap at all.
    off = np.array(sfk + sfo + diff)
    sep = min(same) - off.max()

    # attenuation: cross-seed cosine against within-run cosine, same pairs
    wi, W = within_run()
    xs, ys = [], []
    for j, n in enumerate(nb):
        for m, i in ia.items():
            if m in wi and n in wi and m != n:
                xs.append(W[wi[m], wi[n]])
                ys.append(C[i, j])
    xs, ys = np.array(xs), np.array(ys)
    slope, icept = np.polyfit(xs, ys, 1)
    r = float(np.corrcoef(xs, ys)[0, 1])
    from scipy.stats import spearmanr
    rho = float(spearmanr(xs, ys).statistic)

    f = lambda v: (len(v), float(np.mean(v)), float(np.std(v)))
    return {"path": os.path.basename(path),
            "same": f(same), "same_factor_same_key": f(sfk),
            "same_factor_opp_key": f(sfo), "diff_factor": f(diff),
            "min_same": min(same), "max_off": float(off.max()), "sep": float(sep),
            "top1": sum(1 for x in ranks if x == 1), "n_b": len(nb),
            "mean_rank": float(np.mean(ranks)), "chance_rank": (len(na) + 1) / 2,
            "signed_bipolarity": float(np.mean(sfk) - np.mean(sfo)),
            "slope": float(slope), "intercept": float(icept),
            "pearson": r, "spearman": rho, "n_pairs": len(xs)}


def show(rs):
    w = 26
    def row(lab, fmt, *keys):
        print(f"  {lab:<34s}" + "".join(
            f"{fmt.format(*(r[k] if isinstance(k, str) else k(r) for k in keys)):>{w}s}"
            for r in rs))
    print("\n" + " " * 36 + "".join(f"{r['path'][:24]:>{w}s}" for r in rs))
    for lab, key in (("same trait", "same"),
                     ("same factor, same keying", "same_factor_same_key"),
                     ("same factor, opposite keying", "same_factor_opp_key"),
                     ("different factor", "diff_factor")):
        print(f"  {lab:<34s}" + "".join(
            f"{f'{r[key][1]:+.5f}  (n={r[key][0]}, sd {r[key][2]:.5f})':>{w}s}" for r in rs))
    print()
    row("min same trait", "{:+.5f}", "min_same")
    row("max cross trait", "{:+.5f}", "max_off")
    row("separation (min same - max off)", "{:+.5f}", "sep")
    row("top-1 identification", "{}/{}", "top1", "n_b")
    row("mean rank (chance)", "{:.2f} ({:.1f})", "mean_rank", "chance_rank")
    row("signed bipolarity", "{:+.5f}", "signed_bipolarity")
    print()
    row("attenuation slope", "{:.4f}", "slope")
    row("intercept", "{:+.5f}", "intercept")
    row("Pearson / Spearman", "{:+.4f} / {:+.4f}", "pearson", "spearman")
    row("pairs regressed", "{}", "n_pairs")
    print("\n  r/d = 64/2560 = 0.0250 is the subspace-overlap prediction for the slope.")


if __name__ == "__main__":
    L = labels()
    rs = [arm(p, L) for p in sys.argv[1:]]
    if not rs:
        sys.exit(__doc__)
    show(rs)
    tag = os.environ.get("PC_ARMS_TAG", "")
    json.dump(rs, open(f"{Q}/analysis/crossseed_arms{tag}.json", "w"), indent=1)
    print(f"\nwrote analysis/crossseed_arms{tag}.json  (within-run block: {os.path.basename(WITHIN)})")
