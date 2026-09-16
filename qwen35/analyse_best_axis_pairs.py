#!/usr/bin/env python
"""Which pair of recovered factors separates each Goldberg group's two poles best.

For each of the five Goldberg groups, every pair of the five factors is scored
on how well the group's positively and negatively keyed adapters separate in
that plane.  Coordinates are the mean-centred factor chart (analysis/viz_fa.json,
loaded through figures/clusters/make_cluster_figures.load).

Primary criterion: Mahalanobis distance between the two pole means in the pooled
within-pole covariance of the plane (Fisher's linear discriminant separation).
It can only rise when an axis is added, and rises most when the second axis
carries discrimination the first lacks.  A plain d' (mean distance over pooled
root-mean-square spread) is recorded beside it as dprime_plain; it was tried
first and rejected because it can never exceed the best single axis.

Reads:  analysis/viz_fa.json (via make_cluster_figures.load)
Writes: analysis/best_axis_pairs.json  (or the path given as argv[1])

Consumers: figures/post/make_post_figures.py (facets_best), the wiki page
geometry/best-axis-pairs; companion/assets/planes.js recomputes the same rule
in the browser.  First run 2026-09-15; extracted into a script 2026-09-16.
"""
import itertools, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "figures", "clusters"))
from make_cluster_figures import load  # noqa: E402

GROUPS = ["Agreeableness", "Conscientiousness", "EmotionalStability", "Extraversion", "Intellect"]


def maha(P, Q):
    d = P.mean(0) - Q.mean(0)
    S = np.atleast_2d(0.5 * (np.cov(P.T) + np.cov(Q.T)))
    return float(np.sqrt(d @ np.linalg.solve(S, d)))


def dprime(P, Q):
    mu = P.mean(0) - Q.mean(0)
    s2 = 0.5 * (P.var(0, ddof=1).sum() + Q.var(0, ddof=1).sum())
    return float(np.linalg.norm(mu) / np.sqrt(s2))


def main(out_path):
    D = load()
    X = D["X"]; T = D["titles"]
    fac = np.array(D["factor"]); key = np.array(D["keyed"])
    out = {"what": "for each Goldberg group, every pair of the five recovered factors scored on how well the group's positively and negatively keyed adapters separate in that plane; coordinates are the mean-centred factor chart (analysis/viz_fa.json)",
           "criterion_primary": "mahalanobis: distance between the two pole means in the pooled within-pole covariance of the plane (Fisher's linear discriminant separation); can only rise when an axis is added, and rises most when the second axis carries discrimination the first lacks",
           "criterion_secondary": "dprime_plain: distance between the pole means over the pooled root-mean-square spread, which penalises a second axis for its spread",
           "single_axis_mahalanobis": {}, "groups": {}}
    for g in GROUPS:
        m = fac == g; mp = m & (key == "+"); mn = m & (key == "-")
        out["single_axis_mahalanobis"][g] = {T[i]: round(maha(X[mp][:, [i]], X[mn][:, [i]]), 4) for i in range(5)}
        rows = []
        for i, j in itertools.combinations(range(5), 2):
            P, Q = X[np.ix_(mp, [i, j])], X[np.ix_(mn, [i, j])]
            rows.append({"pair": [T[i], T[j]], "mahalanobis": round(maha(P, Q), 4), "dprime_plain": round(dprime(P, Q), 4),
                         "n_pos": int(mp.sum()), "n_neg": int(mn.sum())})
        rows.sort(key=lambda r: -r["mahalanobis"])
        out["groups"][g] = rows
    json.dump(out, open(out_path, "w"), indent=1)
    for g in GROUPS:
        b = out["groups"][g][0]
        print(f"{g:18s} {b['pair'][0]} x {b['pair'][1]}  separation {b['mahalanobis']:.2f}")
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "analysis", "best_axis_pairs.json"))
