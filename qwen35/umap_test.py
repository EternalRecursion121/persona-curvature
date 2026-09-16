#!/usr/bin/env python3
"""Does a nonlinear embedding buy anything the linear representation doesn't?

UMAP at n=134 will draw clean clusters whatever the data does, so a picture is
not evidence. The question is decided by HELD-OUT PREDICTION: if the Big Five
structure is nonlinear, a kNN classifier should do better in a UMAP embedding
than in the original sketch space or a PCA basis. If it doesn't, UMAP is a
drawing tool here, not an analysis.

Two guards:
  * The embedding is fitted INSIDE the cross-validation fold, never on all the
    data. Fitting UMAP on everything and then cross-validating leaks the test
    points into the layout and inflates accuracy -- the standard way this
    comparison is done wrong.
  * A label-shuffled null, so "better than chance" is measured, not assumed.
"""
import glob, json, os, sys
import numpy as np

Q = "/home/vibe12/projects/persona-curvature/qwen35"
meta = {}
for f in ("traits_primary.json", "traits_secondary.json"):
    for r in json.load(open(f"{Q}/{f}")):
        meta[r["trait"].lower().replace(" ", "_").replace("-", "_")] = (r["factor"], r["keyed"])

names, X = [], []
for p in sorted(glob.glob(f"{Q}/analysis/sketches/stage1_k32/*.npz")):
    t = os.path.basename(p)[:-4]
    if t in meta and meta[t][0] != "Lexicon":
        names.append(t); X.append(np.load(p)["sketch"].astype(np.float64))
X = np.stack(X)
y_f = np.array([meta[t][0] for t in names])
y_k = np.array([meta[t][1] for t in names])
n = len(names)
print(f"n={n} adapters, dim={X.shape[1]}")

Xc = X - X.mean(0)
Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)


def knn_loo(Z, y, k=5):
    """Leave-one-out kNN accuracy."""
    # via the Gram: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b. The broadcast form
    # allocates n x n x d, which is 18.9 GiB at d=253952 on a 7 GB box.
    g = Z @ Z.T
    sq = np.diag(g)
    D = sq[:, None] + sq[None, :] - 2 * g
    np.fill_diagonal(D, np.inf)
    hit = 0
    for i in range(len(Z)):
        idx = np.argsort(D[i])[:k]
        vals, cts = np.unique(y[idx], return_counts=True)
        hit += int(vals[np.argmax(cts)] == y[i])
    return hit / len(Z)


def pca(A, k):
    G = A @ A.T
    w, V = np.linalg.eigh(G)
    i = np.argsort(w)[::-1][:k]
    return V[:, i] * np.sqrt(np.maximum(w[i], 1e-12))


import umap
rng = np.random.default_rng(0)
results = {}
for label, y in (("factor(5-way)", y_f), ("keying(2-way)", y_k)):
    base = 1.0 / len(set(y))
    row = {"chance": round(base, 3)}
    row["sketch_space"] = round(knn_loo(Xc, y), 3)
    for k in (10, 30):
        row[f"pca{k}"] = round(knn_loo(pca(Xc, k), y), 3)
    # UMAP fitted per held-out fold (10-fold), never on the full set
    folds = np.array_split(rng.permutation(n), 10)
    hit = 0
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        em = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.1,
                       random_state=0, verbose=False)
        Ztr = em.fit_transform(Xc[tr])
        Zte = em.transform(Xc[f])
        for j, i in enumerate(f):
            D = ((Ztr - Zte[j]) ** 2).sum(-1)   # small: UMAP space is 5-dim
            idx = np.argsort(D)[:5]
            vals, cts = np.unique(y[tr][idx], return_counts=True)
            hit += int(vals[np.argmax(cts)] == y[i])
    row["umap5_heldout"] = round(hit / n, 3)
    # label-shuffled null in the strongest linear space
    null = [knn_loo(pca(Xc, 30), rng.permutation(y)) for _ in range(60)]
    row["shuffled_null"] = round(float(np.mean(null)), 3)
    results[label] = row
    print(f"\n{label}:")
    for k2, v in row.items():
        print(f"  {k2:18s} {v}")
json.dump(results, open(f"{Q}/analysis/umap_test.json", "w"), indent=1)
