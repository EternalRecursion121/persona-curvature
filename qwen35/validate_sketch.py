#!/usr/bin/env python3
"""Measure how much the bilinear sketch actually distorts adapter geometry.

The sketch is only worth using if the inner products it preserves are the ones
the analysis depends on. Johnson-Lindenstrauss says they should be; this checks
it on real adapters instead of taking the theorem's word for the constant.

Ground truth uses the same low-rank identity as the sketch, so no dW is built:

    <B_i A_i, B_j A_j>_F = tr(A_i^T B_i^T B_j A_j) = tr( (B_i^T B_j) (A_j A_i^T) )

with both factors r x r. Exact, cheap, and independent of the projection.
"""
import glob, os, sys
import numpy as np
from safetensors import safe_open

RAW = sys.argv[1] if len(sys.argv) > 1 else "/tmp/raw8"
SK = sys.argv[2] if len(sys.argv) > 2 else \
    "/home/vibe12/projects/persona-curvature/qwen35/analysis/sketches/stage1_k32"
SCALE = 2.0


def load_ab(path):
    with safe_open(path, framework="np") as f:
        keys = list(f.keys())
        mods = sorted({k.split(".lora_A.")[0].split(".lora_B.")[0] for k in keys})
        return {m: (f.get_tensor(f"{m}.lora_A.weight").astype(np.float32),
                    f.get_tensor(f"{m}.lora_B.weight").astype(np.float32)) for m in mods}


def main():
    files = sorted(glob.glob(f"{RAW}/*.safetensors"))
    names = [os.path.basename(f).replace(".safetensors", "") for f in files]
    if len(names) < 3:
        print(f"need >=3 raw adapters in {RAW}, found {len(names)}"); sys.exit(1)
    print(f"validating on {len(names)}: {' '.join(names)}", flush=True)

    ab = {n: load_ab(f) for n, f in zip(names, files)}
    mods = sorted(ab[names[0]])
    n = len(names)

    G = np.zeros((n, n))
    for m in mods:
        for i in range(n):
            Ai, Bi = ab[names[i]][m]
            for j in range(i, n):
                Aj, Bj = ab[names[j]][m]
                # tr( (Bi^T Bj) (Aj Ai^T) ) -- both r x r
                G[i, j] += float(np.sum((Bi.T @ Bj) * (Ai @ Aj.T).T))
    G = G + np.triu(G, 1).T
    G *= SCALE ** 2

    S = np.stack([np.load(f"{SK}/{t}.npz")["sketch"] for t in names])
    Gs = S @ S.T

    def cos(M):
        d = np.sqrt(np.diag(M))
        return M / np.outer(d, d)

    Ce, Cs = cos(G), cos(Gs)
    iu = np.triu_indices(n, 1)
    e, s = Ce[iu], Cs[iu]
    print(f"\nexact cosine range   : [{e.min():.4f}, {e.max():.4f}]")
    print(f"sketch cosine range  : [{s.min():.4f}, {s.max():.4f}]")
    print(f"Pearson r (off-diag) : {np.corrcoef(e, s)[0,1]:.5f}")
    print(f"max |abs error|      : {np.abs(e-s).max():.5f}")
    print(f"mean |abs error|     : {np.abs(e-s).mean():.5f}")
    ne = np.sqrt(np.diag(G)); ns = np.sqrt(np.diag(Gs))
    print(f"norm ratio sketch/exact: mean {np.mean(ns/ne):.4f} "
          f"sd {np.std(ns/ne):.4f}  (scale-free; only ratios matter)")
    # the decision the analysis actually rests on: does the sketch rank pairs the same way?
    from scipy.stats import spearmanr
    print(f"Spearman rank corr   : {spearmanr(e, s).statistic:.5f}")


if __name__ == "__main__":
    main()
