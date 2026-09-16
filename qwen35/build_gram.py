#!/usr/bin/env python3
"""Build a Gram matrix from sketches in the exact format analyse_fa_qwen35.py reads.

Lets the existing, verified factor-analysis code run on any adapter set without
touching its maths. Also the end-to-end validation of the sketch pipeline: for
stage-1 the repo already holds results/gram_sweep.npz, computed exactly and
independently, so `--compare` measures the sketch against ground truth on the
real 134x134 Gram rather than the 8-adapter subset used before.
"""
import argparse, glob, json, os
import numpy as np

Q = "/home/vibe12/projects/persona-curvature/qwen35"


def load(d):
    names, X, norms = [], [], []
    for p in sorted(glob.glob(f"{d}/*.npz")):
        z = np.load(p)
        names.append(os.path.basename(p)[:-4])
        X.append(z["sketch"].astype(np.float64))
        norms.append(float(np.sqrt((z["norms"].astype(np.float64) ** 2).sum())))
    return names, np.stack(X), np.array(norms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sketches", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None, help="exact gram npz to validate against")
    a = ap.parse_args()
    names, X, norms = load(a.sketches)
    G = X @ X.T
    # Rescale so the Gram's diagonal matches the EXACT adapter norms. The sketch
    # shrinks every norm by one constant (~0.503), which cosines ignore but the
    # factor analysis does not -- it reads norms and scale from this file.
    d = np.sqrt(np.diag(G))
    G = G / np.outer(d, d) * np.outer(norms, norms)
    np.savez(a.out, G=G, names=np.array(names), norms=norms,
             scale=2.0, n_modules=248)
    print(f"wrote {a.out}: {len(names)} adapters")
    if a.compare and os.path.exists(a.compare):
        e = np.load(a.compare, allow_pickle=True)
        en = [str(x) for x in e["names"]]
        Ge = np.asarray(e["G"], dtype=np.float64)
        idx = {n: i for i, n in enumerate(en)}
        common = [n for n in names if n in idx]
        if len(common) < 3:
            print("no overlap to compare"); return
        i1 = [names.index(n) for n in common]
        i2 = [idx[n] for n in common]
        A = G[np.ix_(i1, i1)]; B = Ge[np.ix_(i2, i2)]
        ca = A / np.outer(np.sqrt(np.diag(A)), np.sqrt(np.diag(A)))
        cb = B / np.outer(np.sqrt(np.diag(B)), np.sqrt(np.diag(B)))
        iu = np.triu_indices(len(common), 1)
        r = np.corrcoef(ca[iu], cb[iu])[0, 1]
        print(f"VALIDATION vs exact Gram on {len(common)} shared adapters:")
        print(f"  cosine Pearson r      : {r:.5f}")
        print(f"  max |cosine error|    : {np.abs(ca[iu]-cb[iu]).max():.5f}")
        print(f"  mean |cosine error|   : {np.abs(ca[iu]-cb[iu]).mean():.5f}")
        nr = np.sqrt(np.diag(A)) / np.sqrt(np.diag(B))
        print(f"  norm ratio            : mean {nr.mean():.4f} sd {nr.std():.4f}")


if __name__ == "__main__":
    main()
