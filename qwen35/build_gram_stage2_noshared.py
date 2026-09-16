#!/usr/bin/env python3
"""results/gram_stage2_noshared.npz: the stage-two Gram with the shared
direction projected out exactly.

Stage two puts 0.151 of every adapter's squared norm on one direction, the grand
mean mu = (1/n) sum_i dW_i, and every adapter sits at cosine 0.389 to it
(analysis/stage2_structure.json#shared_component.stage2).  The factor analysis
in results/fa_qwen35_stage2.json double-centres the CORRELATION matrix, which
removes each variable's mean over the other variables -- it does not remove that
direction from the deltas.  This file removes it from the deltas themselves:

    dW_i^res = dW_i - <dW_i, mu> mu / |mu|^2
    G^res    = G - (G 1)(1^T G) / (1^T G 1)

which is exact in the Gram, since <dW_i, mu> = (G 1)_i / n and |mu|^2 =
(1^T G 1) / n^2.  The result is rank n-1 and every row sums to zero.

Written in the G format analyse_fa_qwen35.py expects (G, names, norms, scale,
n_modules), with norms = sqrt(diag(G^res)) so the script's own consistency
assert passes.

usage:  python build_gram_stage2_noshared.py
then:   PC_GRAM_NPZ=results/gram_stage2_noshared.npz PC_FA_TAG=_stage2_noshared \\
            python analyse_fa_qwen35.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    z = np.load(f"{HERE}/results/gram_stage2.npz", allow_pickle=True)
    G = np.array(z["G"], dtype=np.float64)
    names = [str(x) for x in z["names"]]
    n = G.shape[0]
    one = np.ones(n)
    g = G @ one
    denom = float(one @ G @ one)
    Gr = G - np.outer(g, g) / denom

    # checks
    row_sums = np.abs(Gr @ one).max()
    ev = np.linalg.eigvalsh(Gr)
    d = np.diag(Gr)
    print(f"n={n}  |G^res 1|_inf = {row_sums:.3e}  (exactly zero by construction)")
    print(f"eigenvalues: min {ev.min():.3e}  max {ev.max():.3e}  "
          f"n below 1e-9*max: {int((ev < 1e-9 * ev.max()).sum())} (expect 1)")
    print(f"diag: min {d.min():.6f}  mean {d.mean():.6f}  max {d.max():.6f}")
    print(f"trace kept: {d.sum() / np.trace(G):.4f} of the original "
          f"(the shared direction carried {1 - d.sum() / np.trace(G):.4f})")
    if d.min() <= 0:
        raise SystemExit("a residual norm came out non-positive")

    norms = np.sqrt(d)
    C = Gr / np.outer(norms, norms)
    iu = np.triu_indices(n, 1)
    C0 = np.array(z["G"]) / np.outer(np.sqrt(np.diag(G)), np.sqrt(np.diag(G)))
    print(f"off-diagonal cosine: before {C0[iu].mean():+.4f} +- {C0[iu].std():.4f}   "
          f"after {C[iu].mean():+.4f} +- {C[iu].std():.4f}")
    print(f"corr(before, after) off-diagonal: "
          f"{np.corrcoef(C0[iu], C[iu])[0, 1]:.4f}")

    p = f"{HERE}/results/gram_stage2_noshared.npz"
    np.savez(p, G=Gr, names=np.array(names), norms=norms,
             scale=z["scale"], n_modules=z["n_modules"])
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
