#!/usr/bin/env python3
"""UMAP over the adapter Grams -- with its own nulls, because UMAP invents
clusters and the only defence is running the identical embedding on arms
where no trait signal exists.

Four matrices, identical treatment: real (134), shuffled (100), permuted
(100), seed-B (40).  Distance is d = sqrt(2(1 - cos)) on BOTH views (raw and
leading-component-removed); the residual view is the claim-bearing one.
Three UMAP seeds per matrix so instability is visible rather than averaged
away.  n_neighbors=12 (~a factor's worth of same-keyed neighbours),
min_dist=0.15.

Writes results/umap_embeddings.json:
  {arm: {view: {seed: [[x, y] per trait]}, "names": [...]}, "params": ...}
"""
import json
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
ARMS = {
    "real": "gram_sweep.npz",
    "shuffled": "gram_data_null_shuffled_p100.npz",
    "permuted": "gram_data_null_permuted_p100.npz",
    "seedB": "gram_data_null_seedpaired_s40.npz",
}
SEEDS = (0, 1, 2)
N_NEIGHBORS = 12
MIN_DIST = 0.15


def views_of(G):
    n = G.shape[0]
    d = np.sqrt(np.diag(G))
    C = G / np.outer(d, d)
    H = np.eye(n) - np.ones((n, n)) / n
    Gc = H @ G @ H
    w, V = np.linalg.eigh(Gc)
    i = np.argmax(w)
    Gres = Gc - np.outer(V[:, i], V[:, i]) * w[i]
    dres = np.sqrt(np.clip(np.diag(Gres), 1e-12, None))
    Cres = Gres / np.outer(dres, dres)
    return {"raw": C, "resid": Cres}


def embed(C, seed):
    from umap import UMAP
    D = np.sqrt(np.clip(2.0 * (1.0 - C), 0.0, None))
    np.fill_diagonal(D, 0.0)
    u = UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, metric="precomputed",
             random_state=seed, n_components=2)
    X = u.fit_transform(D)
    # centre and scale to unit RMS so panels are comparable
    X = X - X.mean(axis=0)
    X = X / max(np.sqrt((X ** 2).mean()), 1e-9)
    return [[round(float(a), 3), round(float(b), 3)] for a, b in X]


def main():
    out = {"params": {"n_neighbors": N_NEIGHBORS, "min_dist": MIN_DIST,
                      "distance": "sqrt(2(1-cos))", "seeds": list(SEEDS)}}
    for arm, fn in ARMS.items():
        z = np.load(os.path.join(R, fn), allow_pickle=True)
        G = np.array(z["G"])
        names = [str(x) for x in z["names"]]
        vs = views_of(G)
        out[arm] = {"names": names}
        for view, C in vs.items():
            out[arm][view] = {str(s): embed(C, s) for s in SEEDS}
        print(f"{arm}: {len(names)} traits embedded, both views x 3 seeds",
              flush=True)
    p = os.path.join(R, "umap_embeddings.json")
    with open(p, "w") as f:
        json.dump(out, f)
    print(f"wrote {p} ({os.path.getsize(p) // 1024} KB)")


if __name__ == "__main__":
    main()
